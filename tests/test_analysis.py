import json
import math

from analysis.report import compare, summarize, write_summary
from analysis.stats import mcnemar_exact, paired_bootstrap_delta, wilson_ci
from analysis.taxonomy import PRIMARY_ORDER, classify

SRC_PATCH = "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1 +1 @@\n-a\n+b\n"
OTHER_PATCH = SRC_PATCH.replace("pkg/core.py", "pkg/other.py")


def events(patch="", stop="submit", files_read=(), extra=()):
    return [
        {"seq": 0, "type": "episode_start", "payload": {}},
        *extra,
        {
            "seq": 99,
            "type": "episode_end",
            "payload": {
                "patch": patch,
                "stop_cause": stop,
                "files_read": list(files_read),
                "cost_usd": 0.5,
                "steps": 7,
            },
        },
    ]


def report(resolved=False, applied=True, f2p_ok=(), f2p_bad=(), p2p_bad=()):
    return {
        "resolved": resolved,
        "patch_successfully_applied": applied,
        "tests_status": {
            "FAIL_TO_PASS": {"success": list(f2p_ok), "failure": list(f2p_bad)},
            "PASS_TO_PASS": {"success": [], "failure": list(p2p_bad)},
        },
    }


GOLD = ["pkg/core.py"]


def test_taxonomy_primary_categories():
    assert classify("a", events(SRC_PATCH), report(resolved=True), GOLD).primary == "resolved"
    assert classify("a", events("", stop="max_steps"), report(), GOLD).primary == "empty_patch"
    assert classify("a", events(SRC_PATCH, stop="harness_error"), None, GOLD).primary == "infra_error"
    assert classify("a", events(SRC_PATCH), None, GOLD).primary == "not_evaluated"
    assert classify("a", events(SRC_PATCH), report(applied=False), GOLD).primary == "patch_apply_failed"
    assert classify("a", events(OTHER_PATCH), report(f2p_bad=["t"]), GOLD).primary == "wrong_localization"
    assert classify("a", events(SRC_PATCH), report(f2p_ok=["t"], p2p_bad=["u"]), GOLD).primary == "broke_existing_tests"
    assert classify("a", events(SRC_PATCH), report(f2p_ok=["t1"], f2p_bad=["t2"]), GOLD).primary == "partial_fix"
    assert classify("a", events(SRC_PATCH), report(f2p_bad=["t"]), GOLD).primary == "right_file_wrong_fix"


def test_taxonomy_tags():
    hack = (
        SRC_PATCH
        + "diff --git a/tests/test_core.py b/tests/test_core.py\n--- a\n+++ b\n@@\n-    assert x == 1\n+    pass\n"
    )
    v = classify("a", events(hack, stop="max_steps", files_read=["pkg/core.py"]), report(resolved=True), GOLD)
    assert v.resolved and "reward_hack" in v.tags and "gave_up" in v.tags
    v = classify("a", events(OTHER_PATCH, files_read=["pkg/other.py"]), report(f2p_bad=["t"]), GOLD)
    assert "never_opened_gold_file" in v.tags
    halluc = [
        {"type": "tool_call", "payload": {"name": "edit_file", "is_error": False}},
        {
            "type": "tool_call",
            "payload": {"name": "run_tests", "output": "AttributeError: module 'pkg' has no attribute 'magic'"},
        },
    ]
    assert "hallucinated_api_suspected" in classify("a", events(SRC_PATCH, extra=halluc), report(), GOLD).tags


def test_wilson_matches_reference_values():
    lo, hi = wilson_ci(10, 30)
    assert math.isclose(lo, 0.1923, abs_tol=1e-3) and math.isclose(hi, 0.5122, abs_tol=1e-3)
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_mcnemar_and_bootstrap():
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(1, 6), 0.125, abs_tol=1e-9)
    delta, lo, hi = paired_bootstrap_delta([0, 0, 1, 1] * 10, [1, 0, 1, 1] * 10, n_boot=2000)
    assert math.isclose(delta, 0.25) and lo <= delta <= hi


def _fake_run(root, name, resolved_ids, ids=("x__1", "x__2", "x__3", "x__4")):
    run = root / name
    (run / "trajectories").mkdir(parents=True)
    for iid in ids:
        lines = [json.dumps(e) for e in events(SRC_PATCH, files_read=["pkg/core.py"])]
        (run / "trajectories" / f"{iid}.jsonl").write_text("\n".join(lines) + "\n")
    reports = {iid: report(resolved=iid in resolved_ids, f2p_bad=[] if iid in resolved_ids else ["t"]) for iid in ids}
    (run / "eval_reports.json").write_text(json.dumps(reports))
    (run / "instances_meta.jsonl").write_text(
        "".join(json.dumps({"instance_id": i, "gold_files": GOLD}) + "\n" for i in ids)
    )
    return run


def test_summarize_and_compare(tmp_path):
    a = _fake_run(tmp_path, "base", {"x__1"})
    b = _fake_run(tmp_path, "ablation", {"x__1", "x__2"})
    s = write_summary(a)
    assert s["n"] == 4 and s["resolved"] == 1 and s["resolved_pct"] == 25.0
    assert sum(s["primary"].values()) == 4 and set(s["primary"]) == set(PRIMARY_ORDER)
    assert (a / "summary.md").exists() and (a / "per_instance.csv").read_text().count("\n") == 5
    text = compare(a, b)
    assert "+25.0 pp" in text and "4 shared instances" in text


def test_summarize_empty_run():
    s = summarize([])
    assert s["n"] == 0 and s["resolved_pct"] == 0.0
