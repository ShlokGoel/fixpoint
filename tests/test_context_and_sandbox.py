from agent.context import ContextManager, WorkState, render_state
from agent.sandbox import is_excluded_untracked


def test_reset_starts_fresh_conversation_with_state():
    cm = ContextManager(budget_tokens=100, max_resets=1)
    assert cm.over_budget(101) and not cm.over_budget(100)
    state = WorkState(
        localized_files=["pkg/core.py"],
        files_read=["pkg/core.py", "tests/test_core.py"],
        files_edited=["pkg/core.py"],
        diff="diff --git a/pkg/core.py b/pkg/core.py\n+fix\n",
        last_test_command="python -m pytest -q",
        last_test_output="1 failed",
        recent_actions=["edit_file(pkg/core.py) -> ok"],
    )
    msgs = cm.reset("TASK PROMPT", state)
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    body = msgs[0]["content"]
    assert body.startswith("TASK PROMPT") and "[Context reset #1]" in body
    for needle in ("pkg/core.py", "+fix", "python -m pytest -q", "1 failed", "edit_file(pkg/core.py)"):
        assert needle in body
    assert not cm.can_reset()


def test_render_state_handles_empty_and_huge_diffs():
    assert "(no changes yet)" in render_state(WorkState(), 1)
    big = render_state(WorkState(diff="+x\n" * 10_000), 2)
    assert "[diff truncated]" in big and len(big) < 8_000


def test_excluded_untracked_files():
    assert is_excluded_untracked("reproduce_issue.py")
    assert is_excluded_untracked("pkg/__pycache__/mod.cpython-311.pyc")
    assert is_excluded_untracked(".pytest_cache/v/cache/nodeids")
    assert not is_excluded_untracked("pkg/reproduce.py")  # only root-level scratch files
    assert not is_excluded_untracked("tests/test_new_behaviour.py")


def test_git_diff_skips_scratch_and_bytecode(toy_sandbox):
    toy_sandbox.write_file("reproduce.py", "print(1)\n")
    toy_sandbox.write_file("stats_utils/extra.py", "X = 1\n")
    toy_sandbox.exec("python -m pytest -q tests")  # creates __pycache__
    diff = toy_sandbox.git_diff()
    assert "stats_utils/extra.py" in diff
    assert "reproduce.py" not in diff and ".pyc" not in diff


def test_git_diff_does_not_touch_the_real_index(toy_sandbox):
    toy_sandbox.write_file("stats_utils/extra.py", "X = 1\n")
    toy_sandbox.git_diff()
    status = toy_sandbox.exec("git status --porcelain").stdout
    assert "?? stats_utils/extra.py" in status  # still untracked, nothing staged
    assert toy_sandbox.exec("git diff --cached --name-only").stdout.strip() == ""
