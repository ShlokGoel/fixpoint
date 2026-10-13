"""Loading SWE-bench Lite instances and choosing deterministic subsets."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from agent.task import Task

DEFAULT_DATASET = "SWE-bench/SWE-bench_Lite"
DIFF_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def load_instances(dataset: str = DEFAULT_DATASET, split: str = "test") -> list[dict[str, Any]]:
    """Load raw instances from Hugging Face or a local .json/.jsonl file."""
    if dataset.endswith(".jsonl"):
        return [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    if dataset.endswith(".json"):
        return json.loads(Path(dataset).read_text())
    try:
        from swebench.harness.utils import load_swebench_dataset

        return [dict(x) for x in load_swebench_dataset(dataset, split)]
    except ImportError:
        from datasets import load_dataset

        return [dict(x) for x in load_dataset(dataset, split=split)]


def image_for(instance: dict[str, Any], namespace: str = "swebench", arch: str = "x86_64") -> str:
    """Prebuilt per-instance image. Newer datasets name it; otherwise use the harness convention."""
    if instance.get("image"):
        return instance["image"]
    key = f"{namespace}/sweb.eval.{arch}.{instance['instance_id']}:latest"
    return key.replace("__", "_1776_")  # Docker Hub does not allow '__'


def to_task(instance: dict[str, Any]) -> Task:
    """Strip an instance down to what the agent may see (no tests, no gold patch)."""
    return Task(
        instance_id=instance["instance_id"],
        repo=instance["repo"],
        problem_statement=instance["problem_statement"],
        base_commit=instance.get("base_commit", ""),
        image=image_for(instance),
    )


def gold_files(instance: dict[str, Any]) -> list[str]:
    """Files touched by the reference fix. Used only by the analysis step, never by the agent."""
    return sorted({m.group(2) for m in DIFF_FILE_RE.finditer(instance.get("patch", ""))})


def select(
    instances: list[dict[str, Any]],
    instance_ids: list[str] | None = None,
    subset: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    by_id = {i["instance_id"]: i for i in instances}
    if instance_ids:
        missing = [i for i in instance_ids if i not in by_id]
        if missing:
            raise KeyError(f"Not in dataset: {missing}")
        return [by_id[i] for i in instance_ids]
    ids = sorted(by_id)
    if subset is not None and subset < len(ids):
        ids = sorted(random.Random(seed).sample(ids, subset))
    return [by_id[i] for i in ids]


def analysis_metadata(instance: dict[str, Any]) -> dict[str, Any]:
    def count(field: str) -> int:
        value = instance.get(field) or []
        return len(json.loads(value) if isinstance(value, str) else value)

    return {
        "instance_id": instance["instance_id"],
        "repo": instance["repo"],
        "gold_files": gold_files(instance),
        "n_fail_to_pass": count("FAIL_TO_PASS"),
        "n_pass_to_pass": count("PASS_TO_PASS"),
    }
