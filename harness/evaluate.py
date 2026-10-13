"""Score a run with the official SWE-bench harness, then collect its reports.

    python -m harness.evaluate --run-dir runs/sub30

The harness applies each patch in a fresh container from the instance image and
runs the hidden FAIL_TO_PASS / PASS_TO_PASS tests. A task counts as resolved
only if every FAIL_TO_PASS test passes and no PASS_TO_PASS test breaks.
Reports are merged into runs/<run>/eval_reports.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def collect_reports(run_dir: Path, run_id: str) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for path in (run_dir / "logs" / "run_evaluation" / run_id).glob("*/*/report.json"):
        data = json.loads(path.read_text())
        reports.update(data)
    return reports


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dataset", help="defaults to the dataset recorded in run_meta.json")
    p.add_argument("--split", default="test")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--collect-only", action="store_true", help="skip running the harness; just merge reports")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    run_id = meta["run_name"]
    dataset = args.dataset or meta.get("dataset")
    predictions = run_dir / "predictions.jsonl"
    if not predictions.exists():
        print(f"No predictions at {predictions}")
        return 1

    if not args.collect_only:
        cmd = [
            sys.executable, "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset,
            "--split", args.split,
            "--predictions_path", str(predictions),
            "--max_workers", str(args.max_workers),
            "--timeout", str(args.timeout),
            "--run_id", run_id,
            "--report_dir", str(run_dir),
        ]  # fmt: skip
        print("Running:", " ".join(cmd))
        # cwd=run_dir keeps the harness's logs/ directory inside this run.
        code = subprocess.call(cmd, cwd=run_dir)
        if code != 0:
            print(f"swebench harness exited with {code}; collecting whatever reports exist.")

    reports = collect_reports(run_dir, run_id)
    (run_dir / "eval_reports.json").write_text(json.dumps(reports, indent=2))
    n_pred = sum(1 for line in predictions.read_text().splitlines() if line.strip())
    resolved = sum(1 for r in reports.values() if r.get("resolved"))
    print(
        f"Collected {len(reports)}/{n_pred} reports; resolved {resolved}. "
        f"Next: python -m analysis.report summarize {args.run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
