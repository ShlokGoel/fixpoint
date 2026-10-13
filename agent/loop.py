"""The agent loop.

One iteration = one model call plus execution of every tool call in that
response. The loop keeps going while the model keeps calling tools, and stops
on exactly one of these causes (recorded in the trajectory):

  submit              the model called `submit` (and the scaffold accepted it)
  end_turn_no_submit  the model stopped calling tools twice in a row
  max_steps           step budget exhausted
  cost_cap            dollar budget exhausted
  wallclock_cap       time budget exhausted
  no_progress         repeated identical calls, or too many tool errors in a row
  context_overflow    still over the context budget after the allowed number of resets
  max_tokens_loop     responses kept getting cut off at max_tokens
  refusal             the model declined (stop_reason "refusal")
  api_error           the API call failed after the client's retries

The conversation is append-only (see agent/context.py): thinking blocks and
tool_use/tool_result pairs are passed back exactly as received.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import prompts
from .config import AgentConfig
from .context import ContextManager, WorkState
from .llm import LLMClient, Usage
from .sandbox import Sandbox
from .task import Task
from .tools import ToolExecutor, ToolResult, tool_specs
from .trajectory import TrajectoryLogger

EDIT_TOOLS = {"edit_file", "create_file"}
LOG_OUTPUT_CHARS = 20_000
RECENT_ACTIONS = 12


@dataclass
class EpisodeResult:
    instance_id: str
    stop_cause: str
    steps: int
    usage: Usage
    cost_usd: float
    patch: str
    wallclock_s: float
    submitted_summary: str | None = None
    files_read: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    localized_files: list[str] | None = None
    test_runs: int = 0
    context_resets: int = 0
    models_served: list[str] = field(default_factory=list)


def _describe(name: str, args: dict[str, Any], res: ToolResult) -> str:
    """One-line record of a tool call, used in context-reset summaries."""
    if name in ("bash", "run_tests"):
        target = str(args.get("command", ""))[:120]
    elif name == "search":
        target = f"/{args.get('pattern', '')}/ in {args.get('path', '.')}"
    elif name == "report_localization":
        target = ", ".join(args.get("files") or [])
    else:
        target = str(args.get("path") or args.get("name_pattern") or "")
    if res.is_error:
        outcome = "error: " + res.content.splitlines()[0][:100] if res.content else "error"
    elif "exit_code" in res.meta:
        outcome = f"exit {res.meta['exit_code']}"
    else:
        outcome = "ok"
    return f"{name}({target}) -> {outcome}"


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        sandbox: Sandbox,
        cfg: AgentConfig,
        logger: TrajectoryLogger | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.llm = llm
        self.sb = sandbox
        self.cfg = cfg
        self.log = logger or TrajectoryLogger(None)
        self.clock = clock
        self.tools = ToolExecutor(sandbox, cfg)
        self.ctx = ContextManager(cfg.context_budget_tokens, cfg.max_context_resets)
        self._submit_rejections: set[str] = set()
        self._recent: deque[str] = deque(maxlen=RECENT_ACTIONS)
        self._last_test: tuple[str, str] | None = None

    # ------------------------------------------------------------- scaffold
    def _gate(self, name: str) -> ToolResult | None:
        """Return a rejection if the scaffold forbids this call right now."""
        scaffold = self.cfg.scaffold
        if name in EDIT_TOOLS and scaffold.localization and self.tools.localized_files is None:
            return ToolResult(
                "Edit rejected: call report_localization with the files you plan to change first.",
                is_error=True,
                meta={"gated": "localization"},
            )
        if name == "submit":
            # Each check rejects at most once, so the agent can never deadlock.
            if not self.tools.files_edited and "no_edits" not in self._submit_rejections:
                self._submit_rejections.add("no_edits")
                return ToolResult(
                    "Submit rejected: you have not changed any files, so the patch would be empty. "
                    "If you really believe no change is needed, call submit again.",
                    is_error=True,
                    meta={"gated": "no_edits"},
                )
            if (
                scaffold.self_verification
                and self.tools.edited_since_last_test
                and "verify" not in self._submit_rejections
            ):
                self._submit_rejections.add("verify")
                return ToolResult(
                    "Submit rejected: you have not run any tests since your last edit. Run the relevant "
                    "tests with run_tests, check the result, then submit.",
                    is_error=True,
                    meta={"gated": "self_verification"},
                )
        return None

    def _dispatch(self, block: dict[str, Any]) -> ToolResult:
        name, args = block["name"], block.get("input") or {}
        res = self._gate(name) or self.tools.execute(name, args)
        self._recent.append(_describe(name, args, res))
        if name == "run_tests" and not res.is_error:
            self._last_test = (str(args.get("command", "")), res.content)
        return res

    def _work_state(self) -> WorkState:
        try:
            diff = self.sb.git_diff()
        except Exception:
            diff = ""
        return WorkState(
            localized_files=self.tools.localized_files,
            files_read=sorted(self.tools.files_read),
            files_edited=sorted(self.tools.files_edited),
            diff=diff,
            last_test_command=self._last_test[0] if self._last_test else None,
            last_test_output=self._last_test[1] if self._last_test else "",
            recent_actions=list(self._recent),
        )

    # ------------------------------------------------------------------ run
    def run(self, task: Task) -> EpisodeResult:
        cfg = self.cfg
        start = self.clock()
        specs = tool_specs(cfg.scaffold)
        system = prompts.render_system(self.sb.workdir)
        task_prompt = prompts.render_task(task, self.sb.workdir, cfg.scaffold)
        messages: list[dict[str, Any]] = [{"role": "user", "content": task_prompt}]
        self.log.log(
            "episode_start",
            instance_id=task.instance_id,
            repo=task.repo,
            base_commit=task.base_commit,
            image=task.image,
            model=cfg.model,
            config=cfg.to_dict(),
            config_fingerprint=cfg.fingerprint(),
            prompt_hashes=prompts.prompt_hashes(),
            tools=[s["name"] for s in specs],
            system_prompt=system,
            task_prompt=task_prompt,
        )

        usage = Usage()
        steps = 0
        prompt_tokens = 0
        window_exceeded = False
        nudged = False
        max_token_hits = 0
        consecutive_errors = 0
        recent_calls: deque[str] = deque(maxlen=cfg.max_repeated_calls)
        models: list[str] = []
        stop: str | None = None

        while stop is None:
            if steps >= cfg.max_steps:
                stop = "max_steps"
                break
            if cfg.max_cost_usd and usage.cost(cfg.pricing) >= cfg.max_cost_usd:
                stop = "cost_cap"
                break
            if self.clock() - start >= cfg.max_wallclock_s:
                stop = "wallclock_cap"
                break
            if window_exceeded or self.ctx.over_budget(prompt_tokens):
                if not self.ctx.can_reset():
                    stop = "context_overflow"
                    break
                state = self._work_state()
                messages = self.ctx.reset(task_prompt, state)
                self.log.log(
                    "context_reset",
                    step=steps,
                    reset_number=self.ctx.resets,
                    prompt_tokens=prompt_tokens,
                    window_exceeded=window_exceeded,
                    resume_prompt=messages[0]["content"],
                )
                prompt_tokens, window_exceeded, nudged = 0, False, False

            try:
                resp = self.llm.call(system, messages, specs)
            except Exception as e:
                self.log.log("api_error", step=steps, error=f"{type(e).__name__}: {e}")
                stop = "api_error"
                break

            steps += 1
            usage = usage + resp.usage
            prompt_tokens = resp.usage.prompt_tokens
            if resp.model not in models:
                models.append(resp.model)
            self.log.log(
                "model_response",
                step=steps,
                stop_reason=resp.stop_reason,
                content=resp.content,
                usage=resp.usage.to_dict(),
                cost_usd=round(resp.usage.cost(cfg.pricing), 6),
                cumulative_cost_usd=round(usage.cost(cfg.pricing), 6),
                model=resp.model,
                request_id=resp.request_id,
                latency_s=resp.latency_s,
            )

            if resp.stop_reason == "refusal":
                stop = "refusal"
                break
            if resp.stop_reason == "model_context_window_exceeded":
                window_exceeded = True  # reset at the top of the next iteration
                continue

            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if b.get("type") == "tool_use"]
            if not tool_uses:
                if resp.stop_reason == "max_tokens":
                    max_token_hits += 1
                    if max_token_hits >= 3:
                        stop = "max_tokens_loop"
                        break
                    messages.append(
                        {"role": "user", "content": "Your reply was cut off. Continue, and keep replies shorter."}
                    )
                    continue
                if nudged:
                    stop = "end_turn_no_submit"
                    break
                nudged = True
                self.log.log("nudge", step=steps)
                messages.append({"role": "user", "content": prompts.load("nudge").strip()})
                continue
            nudged = False

            results = []
            for block in tool_uses:
                res = self._dispatch(block)
                self.log.log(
                    "tool_call",
                    step=steps,
                    tool_use_id=block["id"],
                    name=block["name"],
                    input=block.get("input"),
                    output=res.content[:LOG_OUTPUT_CHARS],
                    output_chars=len(res.content),
                    is_error=res.is_error,
                    meta=res.meta,
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": res.content,
                        "is_error": res.is_error,
                    }
                )
                if res.meta.get("submitted"):
                    stop = "submit"
                consecutive_errors = consecutive_errors + 1 if res.is_error else 0
                recent_calls.append(block["name"] + json.dumps(block.get("input"), sort_keys=True))

            messages.append({"role": "user", "content": results})
            if stop:
                break
            if consecutive_errors >= cfg.max_consecutive_errors:
                stop = "no_progress"
                self.log.log("stuck", step=steps, reason="consecutive_errors", count=consecutive_errors)
            elif len(recent_calls) == cfg.max_repeated_calls and len(set(recent_calls)) == 1:
                stop = "no_progress"
                self.log.log("stuck", step=steps, reason="repeated_call", call=recent_calls[0][:300])

        try:
            patch = self.sb.git_diff()
        except Exception as e:
            self.log.log("patch_error", error=str(e))
            patch = ""

        result = EpisodeResult(
            instance_id=task.instance_id,
            stop_cause=stop or "unknown",
            steps=steps,
            usage=usage,
            cost_usd=round(usage.cost(cfg.pricing), 6),
            patch=patch,
            wallclock_s=round(self.clock() - start, 2),
            submitted_summary=self.tools.submitted_summary,
            files_read=sorted(self.tools.files_read),
            files_edited=sorted(self.tools.files_edited),
            localized_files=self.tools.localized_files,
            test_runs=self.tools.test_runs,
            context_resets=self.ctx.resets,
            models_served=models,
        )
        self.log.log(
            "episode_end",
            stop_cause=result.stop_cause,
            steps=steps,
            usage=usage.to_dict(),
            cost_usd=result.cost_usd,
            wallclock_s=result.wallclock_s,
            patch=patch,
            patch_lines=patch.count("\n"),
            files_read=result.files_read,
            files_edited=result.files_edited,
            localized_files=result.localized_files,
            test_runs=result.test_runs,
            context_resets=result.context_resets,
            submitted_summary=result.submitted_summary,
            models_served=models,
        )
        return result
