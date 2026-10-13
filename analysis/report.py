"""Turn a run directory into numbers.

    python -m analysis.report summarize runs/sub30
    python -m analysis.report compare runs/sub30 runs/sub30-noloc

`summarize` writes runs/<run>/summary.md, summary.json and per_instance.csv.
`compare` reports the paired delta between two runs on their shared instances.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from agent.trajectory import read_trajectory

from .stats import mcnemar_exact, paired_bootstrap_delta, percentile, wilson_ci
from .taxonomy import PRIMARY_ORDER, Verdict, classify


def load_run(run_dir: Path) -> list[Verdict]:
    reports_path = run_dir / "eval_reports.json"
    reports = json.loads(reports_path.read_text()) if reports_path.exists() else {}
    meta_path = run_dir / "instances_meta.jsonl"
    gold = {}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                gold[row["instance_id"]] = row.get("gold_files", [])
    verdicts = []
    for traj in sorted((run_dir / "trajectories").glob("*.jsonl")):
        iid = traj.stem
        verdicts.append(classify(iid, read_trajectory(traj), reports.get(iid), gold.get(iid)))
    return verdicts


def summarize(verdicts: list[Verdict]) -> dict[str, Any]:
    n = len(verdicts)
    k = sum(v.resolved for v in verdicts)
    costs = [v.cost_usd for v in verdicts]
    steps = [v.steps for v in verdicts]
    lo, hi = wilson_ci(k, n)
    evaluated = sum(v.primary != "not_evaluated" for v in verdicts)
    return {
        "n": n,
        "evaluated": evaluated,
        "resolved": k,
        "resolved_pct": round(100 * k / n, 1) if n else 0.0,
        "resolved_ci95_pct": [round(100 * lo, 1), round(100 * hi, 1)],
        "cost_total_usd": round(sum(costs), 2),
        "cost_mean_usd": round(statistics.mean(costs), 3) if costs else 0.0,
        "cost_median_usd": round(statistics.median(costs), 3) if costs else 0.0,
        "cost_p90_usd": round(percentile(costs, 0.9), 3),
        "cost_per_resolved_usd": round(sum(costs) / k, 2) if k else None,
        "steps_mean": round(statistics.mean(steps), 1) if steps else 0.0,
        "primary": {c: sum(v.primary == c for v in verdicts) for c in PRIMARY_ORDER},
        "tags": dict(Counter(t for v in verdicts for t in v.tags)),
        "stop_causes": dict(Counter(v.stop_cause for v in verdicts)),
        "reward_hack_instances": [
            {
                "instance_id": v.instance_id,
                "resolved": v.resolved,
                "flags": [f.to_dict() for f in v.flags if f.severity != "info"],
            }
            for v in verdicts
            if "reward_hack" in v.tags
        ],
    }


def to_markdown(name: str, s: dict[str, Any]) -> str:
    n = s["n"] or 1
    lines = [
        f"# Results: {name}",
        "",
        "| N | Evaluated | Resolved | Resolved % (95% Wilson CI) | Total cost | Mean / median / p90 cost per task | Cost per resolved | Mean steps |",
        "|---|---|---|---|---|---|---|---|",
        f"| {s['n']} | {s['evaluated']} | {s['resolved']} | {s['resolved_pct']}% "
        f"({s['resolved_ci95_pct'][0]}–{s['resolved_ci95_pct'][1]}%) | ${s['cost_total_usd']} | "
        f"${s['cost_mean_usd']} / ${s['cost_median_usd']} / ${s['cost_p90_usd']} | "
        f"{'$' + str(s['cost_per_resolved_usd']) if s['cost_per_resolved_usd'] is not None else '—'} | {s['steps_mean']} |",
        "",
        "## Outcome taxonomy (primary category, sums to N)",
        "",
        "| Category | Count | % |",
        "|---|---|---|",
    ]
    lines += [f"| {c} | {k} | {100 * k / n:.1f}% |" for c, k in s["primary"].items() if k]
    lines += ["", "## Secondary tags", "", "| Tag | Count |", "|---|---|"]
    lines += [f"| {t} | {k} |" for t, k in sorted(s["tags"].items(), key=lambda x: -x[1])] or ["| (none) | 0 |"]
    lines += ["", "## Stop causes", "", "| Stop cause | Count |", "|---|---|"]
    lines += [f"| {c} | {k} |" for c, k in sorted(s["stop_causes"].items(), key=lambda x: -x[1])]
    lines += ["", "## Reward-hacking flags", ""]
    if s["reward_hack_instances"]:
        for item in s["reward_hack_instances"]:
            codes = ", ".join(sorted({f["code"] for f in item["flags"]}))
            lines.append(f"- `{item['instance_id']}` (resolved={item['resolved']}): {codes}")
    else:
        lines.append("None detected.")
    return "\n".join(lines) + "\n"


def write_summary(run_dir: Path) -> dict[str, Any]:
    verdicts = load_run(run_dir)
    s = summarize(verdicts)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2))
    (run_dir / "summary.md").write_text(to_markdown(run_dir.name, s))
    with (run_dir / "per_instance.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["instance_id", "resolved", "primary", "tags", "stop_cause", "steps", "cost_usd", "flags"])
        for v in verdicts:
            w.writerow(
                [
                    v.instance_id,
                    int(v.resolved),
                    v.primary,
                    ";".join(v.tags),
                    v.stop_cause,
                    v.steps,
                    f"{v.cost_usd:.4f}",
                    ";".join(sorted({f.code for f in v.flags if f.severity != "info"})),
                ]
            )
    return s


def compare(run_a: Path, run_b: Path) -> str:
    a = {v.instance_id: v for v in load_run(run_a)}
    b = {v.instance_id: v for v in load_run(run_b)}
    shared = sorted(set(a) & set(b))
    ra = [int(a[i].resolved) for i in shared]
    rb = [int(b[i].resolved) for i in shared]
    delta, lo, hi = paired_bootstrap_delta(ra, rb)
    only_a = sum(1 for x, y in zip(ra, rb, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(ra, rb, strict=True) if y and not x)
    p = mcnemar_exact(only_a, only_b)
    n = len(shared) or 1
    cost_a = sum(a[i].cost_usd for i in shared) / n
    cost_b = sum(b[i].cost_usd for i in shared) / n
    return (
        "\n".join(
            [
                f"# {run_b.name} vs {run_a.name} (paired, {len(shared)} shared instances)",
                "",
                "| Run | Resolved | Resolved % | Mean cost |",
                "|---|---|---|---|",
                f"| {run_a.name} | {sum(ra)} | {100 * sum(ra) / n:.1f}% | ${cost_a:.3f} |",
                f"| {run_b.name} | {sum(rb)} | {100 * sum(rb) / n:.1f}% | ${cost_b:.3f} |",
                "",
                f"Delta ({run_b.name} − {run_a.name}): {100 * delta:+.1f} pp, 95% paired-bootstrap CI "
                f"[{100 * lo:+.1f}, {100 * hi:+.1f}] pp",
                f"Discordant tasks: {only_a} resolved only by {run_a.name}, {only_b} only by {run_b.name}; "
                f"exact McNemar p = {p:.3f}",
                "",
                "If the CI includes 0, this run does not show a difference.",
            ]
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("run_dir")
    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")
    args = p.parse_args(argv)

    if args.cmd == "summarize":
        run_dir = Path(args.run_dir)
        write_summary(run_dir)
        print((run_dir / "summary.md").read_text())
    else:
        print(compare(Path(args.run_a), Path(args.run_b)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
