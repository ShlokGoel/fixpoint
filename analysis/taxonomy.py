"""Failure taxonomy.

Every task gets exactly one primary category (so counts add up to N) plus any
number of secondary tags. Primary categories, checked in this order:

  resolved               all FAIL_TO_PASS pass, no PASS_TO_PASS broke
  infra_error            harness/container/API failure; not the agent's fault
  empty_patch            the agent produced no diff
  patch_apply_failed     the patch did not apply / tests could not run
  wrong_localization     never edited any file the reference fix touches
  broke_existing_tests   some PASS_TO_PASS test now fails
  partial_fix            some but not all FAIL_TO_PASS tests now pass
  right_file_wrong_fix   edited the right file(s) but the target tests still fail
  not_evaluated          no harness report available yet

Secondary tags: gave_up (hit a cap / stuck), premature_stop (ended without
submitting), refused, context_reset, skipped_verification,
never_opened_gold_file, hallucinated_api_suspected, reward_hack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.trajectory import episode_end

from .diffparse import parse_diff
from .reward_hack import Flag, detect, is_reward_hack

PRIMARY_ORDER = [
    "resolved", "infra_error", "empty_patch", "patch_apply_failed", "wrong_localization",
    "broke_existing_tests", "partial_fix", "right_file_wrong_fix", "not_evaluated",
]  # fmt: skip
GAVE_UP = {"max_steps", "cost_cap", "wallclock_cap", "no_progress", "context_overflow", "max_tokens_loop"}
INFRA = {"api_error", "harness_error"}
HALLUCINATION = re.compile(
    r"(AttributeError: (module|type object|'[\w.]+' object) .* has no attribute)|"
    r"(ImportError: cannot import name)|(ModuleNotFoundError: No module named)|"
    r"(TypeError: \w+\(\) got an unexpected keyword argument)"
)


@dataclass
class Verdict:
    instance_id: str
    resolved: bool
    primary: str
    tags: list[str] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    stop_cause: str = "unknown"
    cost_usd: float = 0.0
    steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "resolved": self.resolved,
            "primary": self.primary,
            "tags": self.tags,
            "flags": [f.to_dict() for f in self.flags],
            "stop_cause": self.stop_cause,
            "cost_usd": self.cost_usd,
            "steps": self.steps,
        }


def _tests_status(report: dict[str, Any]) -> tuple[dict, dict]:
    ts = report.get("tests_status") or {}
    return ts.get("FAIL_TO_PASS") or {}, ts.get("PASS_TO_PASS") or {}


def _hallucination_after_edit(events: list[dict[str, Any]]) -> bool:
    edited = False
    for e in events:
        if e.get("type") != "tool_call":
            continue
        p = e["payload"]
        if p.get("name") in ("edit_file", "create_file") and not p.get("is_error"):
            edited = True
        elif edited and p.get("name") in ("bash", "run_tests") and HALLUCINATION.search(str(p.get("output", ""))):
            return True
    return False


def classify(
    instance_id: str,
    events: list[dict[str, Any]],
    report: dict[str, Any] | None,
    gold_files: list[str] | None,
) -> Verdict:
    end = episode_end(events)
    patch = end.get("patch") or ""
    stop = end.get("stop_cause", "unknown")
    flags = detect(patch, events)
    edited = {f.path for f in parse_diff(patch)}
    gold = set(gold_files or [])

    tags: list[str] = []
    if stop in GAVE_UP:
        tags.append("gave_up")
    if stop == "end_turn_no_submit":
        tags.append("premature_stop")
    if stop == "refusal":
        tags.append("refused")
    if end.get("context_resets"):
        tags.append("context_reset")
    if any(
        e["type"] == "tool_call" and (e["payload"].get("meta") or {}).get("gated") == "self_verification"
        for e in events
    ):
        tags.append("skipped_verification")
    read = set(end.get("files_read") or []) | edited
    if gold and not (gold & read):
        tags.append("never_opened_gold_file")
    if _hallucination_after_edit(events):
        tags.append("hallucinated_api_suspected")
    if is_reward_hack(flags):
        tags.append("reward_hack")

    resolved = bool(report and report.get("resolved"))
    if resolved:
        primary = "resolved"
    elif stop in INFRA or (report or {}).get("infra_failure"):
        primary = "infra_error"
    elif not patch.strip():
        primary = "empty_patch"
    elif report is None:
        primary = "not_evaluated"
    elif not report.get("patch_successfully_applied", True):
        primary = "patch_apply_failed"
    elif gold and not (gold & edited):
        primary = "wrong_localization"
    else:
        f2p, p2p = _tests_status(report)
        if p2p.get("failure"):
            primary = "broke_existing_tests"
        elif f2p.get("success") and f2p.get("failure"):
            primary = "partial_fix"
        else:
            primary = "right_file_wrong_fix"

    return Verdict(
        instance_id=instance_id,
        resolved=resolved,
        primary=primary,
        tags=tags,
        flags=flags,
        stop_cause=stop,
        cost_usd=float(end.get("cost_usd") or 0.0),
        steps=int(end.get("steps") or 0),
    )
