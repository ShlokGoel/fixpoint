"""Prompt templates live as files in agent/prompts/ so they are versioned with the code.

Each episode logs the SHA-256 of every template it used, so a result can always
be matched to the exact prompt text that produced it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import ScaffoldConfig
from .task import Task

PROMPT_DIR = Path(__file__).parent / "prompts"


def load(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def prompt_hashes() -> dict[str, str]:
    return {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in sorted(PROMPT_DIR.glob("*.md"))}


def workflow_steps(scaffold: ScaffoldConfig) -> str:
    steps = []
    if scaffold.planning:
        steps.append("Plan: restate the issue in your own words and decide how you will find its cause.")
    if scaffold.localization:
        steps.append(
            "Localize: explore with search, find_files and read_file until you know which files must "
            "change, then call `report_localization`. Edits are rejected until you do."
        )
    else:
        steps.append("Explore the code with search, find_files and read_file to find the cause.")
    steps.append("Reproduce the problem if that is practical, for example with a short script run through bash.")
    steps.append("Fix the source code with edit_file.")
    if scaffold.self_verification:
        steps.append(
            "Verify: after your last edit, run the relevant existing tests with run_tests (for example "
            "`python -m pytest path/to/test_file.py -x -q`). `submit` is rejected if you have not run "
            "tests since your last edit."
        )
    steps.append("Submit with `submit`.")
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def render_system(workdir: str) -> str:
    return load("system").format(workdir=workdir)


def render_task(task: Task, workdir: str, scaffold: ScaffoldConfig) -> str:
    return load("task").format(
        problem_statement=task.problem_statement.strip(),
        repo=task.repo,
        workdir=workdir,
        workflow=workflow_steps(scaffold),
    )
