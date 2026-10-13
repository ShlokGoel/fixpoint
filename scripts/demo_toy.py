"""End-to-end demo on a tiny repo, then grade the patch with a held-out test.

    python scripts/demo_toy.py --scripted                  # no API key: replays a fixed script
    python scripts/demo_toy.py                             # real model, local temp dir (NOT isolated)
    python scripts/demo_toy.py --docker fixpoint-toy      # real model, Docker sandbox (recommended)

Build the Docker image first with:
    docker build -f examples/Dockerfile.toy -t fixpoint-toy examples/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.config import load_config, parse_set_args  # noqa: E402
from agent.fake_llm import ScriptedLLM, text, tool  # noqa: E402
from agent.loop import Agent  # noqa: E402
from agent.sandbox import DockerSandbox, LocalSandbox  # noqa: E402
from agent.task import Task  # noqa: E402
from agent.trajectory import TrajectoryLogger  # noqa: E402

BUGGY = "    mid = len(ordered) // 2\n    return ordered[mid]\n"
FIXED = (
    "    mid = len(ordered) // 2\n"
    "    if len(ordered) % 2 == 0:\n"
    "        return (ordered[mid - 1] + ordered[mid]) / 2\n"
    "    return ordered[mid]\n"
)


def scripted_turns() -> list:
    return [
        [
            text("The issue is about median() on even-length lists. Let me find it."),
            tool("search", pattern="def median"),
        ],
        [tool("read_file", path="stats_utils/core.py")],
        [
            tool(
                "report_localization",
                files=["stats_utils/core.py"],
                rationale="median() only returns the upper middle element and ignores even lengths.",
            )
        ],
        [tool("edit_file", path="stats_utils/core.py", old_str=BUGGY, new_str=FIXED)],
        [tool("run_tests", command="python -m pytest -q tests")],
        [
            tool(
                "submit",
                summary="median() returned ordered[mid] for every length. For even lengths it now "
                "averages the two middle values.",
            )
        ],
    ]


def local_repo() -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="toy-repo-"))
    shutil.copytree(ROOT / "examples" / "toy_repo", tmp, dirs_exist_ok=True)
    git = ["git", "-c", "user.email=demo@example.com", "-c", "user.name=demo"]
    subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
    subprocess.run([*git, "add", "-A"], cwd=tmp, check=True)
    subprocess.run([*git, "commit", "-qm", "toy repo base"], cwd=tmp, check=True)
    return tmp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scripted", action="store_true", help="replay a fixed script instead of calling the API")
    p.add_argument("--docker", metavar="IMAGE", help="run inside this Docker image (repo at /testbed)")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", metavar="KEY=VALUE")
    p.add_argument("--out", default=str(ROOT / "runs" / "toy-demo"))
    args = p.parse_args()

    cfg = load_config(args.config, parse_set_args(args.set))
    if args.scripted:
        llm = ScriptedLLM(scripted_turns())
    else:
        from agent.llm import AnthropicClient

        llm = AnthropicClient(cfg)

    if args.docker:
        sandbox = DockerSandbox(args.docker, cfg.sandbox).start()
    else:
        if not args.scripted:
            print(
                "WARNING: local mode runs model-chosen commands on this machine (in a temp dir). "
                "Use --docker for isolation."
            )
        sandbox = LocalSandbox(local_repo())

    task = Task(
        instance_id="toy__stats-median",
        repo="example/stats_utils",
        problem_statement=(ROOT / "examples" / "toy_issue.md").read_text(),
    )
    mode = "scripted" if args.scripted else "model"
    traj = Path(args.out) / f"{task.instance_id}.{mode}.jsonl"
    try:
        with TrajectoryLogger(traj) as log:
            result = Agent(llm, sandbox, cfg, log).run(task)
        # Grade with the held-out test, as the SWE-bench harness would.
        sandbox.write_file(
            f"{sandbox.workdir}/tests/test_hidden_median.py", (ROOT / "examples" / "toy_hidden_test.py").read_text()
        )
        graded = sandbox.exec("python -m pytest -q tests")
    finally:
        sandbox.close()

    print(
        f"stop_cause={result.stop_cause} steps={result.steps} cost=${result.cost_usd:.4f} models={result.models_served}"
    )
    print("\n--- patch ---\n" + (result.patch or "(empty)"))
    print("--- held-out grading ---\n" + graded.output.strip().splitlines()[-1])
    print(f"RESOLVED: {graded.exit_code == 0}")
    print(f"Trajectory: {traj}  (open viewer/trajectory_viewer.html and load it)")
    return 0 if graded.exit_code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
