from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    """What the agent is allowed to see about a task.

    Deliberately excludes FAIL_TO_PASS / PASS_TO_PASS, the test patch and the
    gold patch: the agent must find and verify the fix on its own.
    """

    instance_id: str
    repo: str
    problem_statement: str
    base_commit: str = ""
    image: str | None = None
