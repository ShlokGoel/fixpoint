"""Context management.

Current Claude models sign their thinking blocks over the conversation that
precedes them, and the API rejects a request that replays those blocks after
earlier history was edited. So the history is append-only: no trimming or
masking of old tool outputs in place.

Instead, when the prompt passes the token budget (or the model hits its context
window), the conversation is *reset*: a new one starts with the original task
plus a state summary built from facts the harness tracks itself (localized
files, files read and edited, the current diff, the last test run and the last
few actions). The summary is deterministic and costs no extra model call; the
model loses its earlier reasoning but keeps everything needed to continue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DIFF_MAX_CHARS = 6_000
TEST_TAIL_CHARS = 2_000


@dataclass
class WorkState:
    localized_files: list[str] | None = None
    files_read: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    diff: str = ""
    last_test_command: str | None = None
    last_test_output: str = ""
    recent_actions: list[str] = field(default_factory=list)


def _tail(text: str, n: int) -> str:
    return text if len(text) <= n else "[...]\n" + text[-n:]


def render_state(state: WorkState, reset_number: int) -> str:
    parts = [
        f"[Context reset #{reset_number}] The conversation grew past the context budget and was cleared. "
        "Your earlier messages are gone, but your changes in the repository are intact. "
        "Here is the state of your work so far:",
        f"- Files you localized: {', '.join(state.localized_files) if state.localized_files else '(not reported yet)'}",
        f"- Files you have read: {', '.join(state.files_read) if state.files_read else '(none)'}",
        f"- Files you have edited: {', '.join(state.files_edited) if state.files_edited else '(none)'}",
    ]
    if state.diff.strip():
        diff = state.diff if len(state.diff) <= DIFF_MAX_CHARS else state.diff[:DIFF_MAX_CHARS] + "\n[diff truncated]"
        parts.append(f"- Current diff against the base commit:\n```diff\n{diff.rstrip()}\n```")
    else:
        parts.append("- Current diff: (no changes yet)")
    if state.last_test_command:
        parts.append(
            f"- Last test run: `{state.last_test_command}`\n```\n{_tail(state.last_test_output, TEST_TAIL_CHARS)}\n```"
        )
    if state.recent_actions:
        parts.append("- Your most recent actions:\n" + "\n".join(f"  {a}" for a in state.recent_actions))
    parts.append("Continue from here. Re-read any file you need; do not redo work that is already done.")
    return "\n".join(parts)


class ContextManager:
    def __init__(self, budget_tokens: int, max_resets: int):
        self.budget_tokens = budget_tokens
        self.max_resets = max_resets
        self.resets = 0

    def over_budget(self, prompt_tokens: int) -> bool:
        return prompt_tokens > self.budget_tokens

    def can_reset(self) -> bool:
        return self.resets < self.max_resets

    def reset(self, task_prompt: str, state: WorkState) -> list[dict]:
        """Start a fresh conversation carrying the task and a summary of progress."""
        self.resets += 1
        return [{"role": "user", "content": f"{task_prompt}\n\n{render_state(state, self.resets)}"}]
