import json
from dataclasses import fields

from agent.task import Task
from harness import evaluate, run_agent
from harness.tasks import gold_files, image_for, select, to_task

INSTANCE = {
    "instance_id": "django__django-11099",
    "repo": "django/django",
    "base_commit": "abc123",
    "problem_statement": "UsernameValidator allows trailing newline",
    "patch": "diff --git a/django/contrib/auth/validators.py b/django/contrib/auth/validators.py\n--- a\n+++ b\n",
    "test_patch": "diff --git a/tests/auth_tests/test_validators.py b/tests/auth_tests/test_validators.py\n",
    "FAIL_TO_PASS": '["test_unicode_validator"]',
    "PASS_TO_PASS": "[]",
}


def test_task_never_carries_tests_or_gold_patch():
    task = to_task(INSTANCE)
    names = {f.name for f in fields(Task)}
    assert not names & {"FAIL_TO_PASS", "PASS_TO_PASS", "test_patch", "patch", "hints_text"}
    assert "test_validators" not in json.dumps(task.__dict__)


def test_image_name_convention():
    assert image_for(INSTANCE) == "swebench/sweb.eval.x86_64.django_1776_django-11099:latest"
    assert image_for({**INSTANCE, "image": "custom/img:1"}) == "custom/img:1"


def test_gold_files():
    assert gold_files(INSTANCE) == ["django/contrib/auth/validators.py"]


def test_subset_is_deterministic_and_seeded():
    insts = [{**INSTANCE, "instance_id": f"r__r-{i}"} for i in range(50)]
    a = [i["instance_id"] for i in select(insts, subset=10, seed=0)]
    b = [i["instance_id"] for i in select(insts, subset=10, seed=0)]
    c = [i["instance_id"] for i in select(insts, subset=10, seed=1)]
    assert a == b and a != c and len(a) == 10 and a == sorted(a)


def test_run_agent_dry_run_with_local_dataset(tmp_path, capsys):
    ds = tmp_path / "ds.jsonl"
    ds.write_text("".join(json.dumps({**INSTANCE, "instance_id": f"r__r-{i}"}) + "\n" for i in range(5)))
    assert run_agent.main(["--run-name", "t", "--dataset", str(ds), "--subset", "2", "--dry-run"]) == 0
    assert len(capsys.readouterr().out.split()) == 2


def test_collect_reports(tmp_path):
    rep = tmp_path / "logs" / "run_evaluation" / "myrun" / "myrun" / "a__b-1"
    rep.mkdir(parents=True)
    (rep / "report.json").write_text(json.dumps({"a__b-1": {"resolved": True}}))
    assert evaluate.collect_reports(tmp_path, "myrun") == {"a__b-1": {"resolved": True}}
