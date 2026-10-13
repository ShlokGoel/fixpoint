"""Run the agent over SWE-bench Lite instances, one fresh container per task.

    python -m harness.run_agent --run-name sub30 --subset 30 --seed 0
    python -m harness.run_agent --run-name smoke --instance-ids django__django-11099 ...
    python -m harness.run_agent --run-name sub30-noloc --subset 30 --set scaffold.localization=false

Outputs (under runs/<run-name>/):
    run_meta.json            config, fingerprint, selected ids, versions
    instances_meta.jsonl     gold files + test counts, for analysis only
    trajectories/<id>.jsonl  full step-by-step log
    patches/<id>.diff        the agent's patch
    predictions.jsonl        SWE-bench prediction format, input to harness.evaluate
Re-running the same command resumes: finished instances are skipped.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from agent.config import AgentConfig, load_config, parse_set_args
from agent.llm import AnthropicClient
from agent.loop import Agent
from agent.sandbox import DockerSandbox
from agent.trajectory import TrajectoryLogger
from harness.tasks import DEFAULT_DATASET, analysis_metadata, load_instances, select, to_task

_lock = threading.Lock()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent)
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=Path(__file__).parent
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "") if out.returncode == 0 else None
    except OSError:
        return None


def _done_ids(run_dir: Path, retry_errors: bool) -> set[str]:
    predictions = run_dir / "predictions.jsonl"
    if not predictions.exists():
        return set()
    rows = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
    if retry_errors:
        from agent.trajectory import episode_end, read_trajectory

        def errored(iid: str) -> bool:
            path = run_dir / "trajectories" / f"{iid}.jsonl"
            stop = episode_end(read_trajectory(path)).get("stop_cause") if path.exists() else None
            return stop in ("harness_error", "api_error")

        kept = [r for r in rows if not errored(r["instance_id"])]
        if len(kept) != len(rows):
            print(f"Retrying {len(rows) - len(kept)} instance(s) that hit harness or API errors.")
            predictions.write_text("".join(json.dumps(r) + "\n" for r in kept))
        rows = kept
    return {r["instance_id"] for r in rows}


def run_one(instance: dict, cfg: AgentConfig, run_dir: Path, run_name: str, docker_bin: str) -> dict:
    task = to_task(instance)
    traj_path = run_dir / "trajectories" / f"{task.instance_id}.jsonl"
    sandbox = DockerSandbox(task.image, cfg.sandbox, docker_bin=docker_bin)
    with TrajectoryLogger(traj_path) as log:
        try:
            sandbox.start()
            head = sandbox.head_commit()
            dirty = sandbox.exec("git status --porcelain").stdout.strip()
            log.log(
                "env_check",
                head=head,
                base_commit=task.base_commit,
                head_matches_base=(not task.base_commit or head == task.base_commit),
                dirty=bool(dirty),
            )
            result = Agent(AnthropicClient(cfg), sandbox, cfg, log).run(task)
            patch, stop, cost, steps = result.patch, result.stop_cause, result.cost_usd, result.steps
        except Exception as e:
            log.log("harness_error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            log.log("episode_end", stop_cause="harness_error", steps=0, cost_usd=0.0, patch="")
            patch, stop, cost, steps = "", "harness_error", 0.0, 0
        finally:
            sandbox.close()

    (run_dir / "patches").mkdir(parents=True, exist_ok=True)
    (run_dir / "patches" / f"{task.instance_id}.diff").write_text(patch)
    prediction = {"instance_id": task.instance_id, "model_name_or_path": run_name, "model_patch": patch}
    with _lock, (run_dir / "predictions.jsonl").open("a") as fh:
        fh.write(json.dumps(prediction) + "\n")
    return {
        "instance_id": task.instance_id,
        "stop_cause": stop,
        "cost_usd": cost,
        "steps": steps,
        "patch_lines": patch.count("\n"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--split", default="test")
    p.add_argument("--instance-ids", nargs="+")
    p.add_argument("--subset", type=int, help="random subset size (seeded by --seed)")
    p.add_argument("--seed", type=int, help="overrides config seed for subset selection")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--docker-bin", default="docker")
    p.add_argument("--dry-run", action="store_true", help="print the selected instance ids and exit")
    p.add_argument("--retry-errors", action="store_true", help="re-run instances that hit harness/infra errors")
    args = p.parse_args(argv)

    overrides = parse_set_args(args.set)
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_config(args.config, overrides)

    instances = select(load_instances(args.dataset, args.split), args.instance_ids, args.subset, cfg.seed)
    if args.dry_run:
        print("\n".join(i["instance_id"] for i in instances))
        return 0

    run_dir = Path(args.runs_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("config_fingerprint") != cfg.fingerprint():
            print(f"Refusing to resume {run_dir}: config changed since it started. Use a new --run-name.")
            return 2
    else:
        import anthropic

        meta_path.write_text(
            json.dumps(
                {
                    "run_name": args.run_name,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "config": cfg.to_dict(),
                    "config_fingerprint": cfg.fingerprint(),
                    "dataset": args.dataset,
                    "split": args.split,
                    "instance_ids": [i["instance_id"] for i in instances],
                    "agent_git_commit": _git_commit(),
                    "python": sys.version.split()[0],
                    "anthropic_sdk": anthropic.__version__,
                    "platform": platform.platform(),
                },
                indent=2,
            )
        )
        with (run_dir / "instances_meta.jsonl").open("w") as fh:
            for inst in instances:
                fh.write(json.dumps(analysis_metadata(inst)) + "\n")

    done = _done_ids(run_dir, args.retry_errors)
    todo = [i for i in instances if i["instance_id"] not in done]
    print(
        f"{run_dir}: {len(instances)} selected, {len(done)} already done, {len(todo)} to run "
        f"(model={cfg.model}, fingerprint={cfg.fingerprint()})"
    )

    total = 0.0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, inst, cfg, run_dir, args.run_name, args.docker_bin): inst for inst in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            total += row["cost_usd"]
            print(
                f"[{n}/{len(todo)}] {row['instance_id']}: stop={row['stop_cause']} steps={row['steps']} "
                f"patch_lines={row['patch_lines']} cost=${row['cost_usd']:.3f} (run total ${total:.2f})"
            )
    print(f"Done. Next: python -m harness.evaluate --run-dir {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
