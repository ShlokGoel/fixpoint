from agent.config import AgentConfig
from agent.tools import ToolExecutor, tool_specs, truncate_middle

MEDIAN_LINE = "    return ordered[mid]\n"


def make(sandbox, **cfg_kw):
    return ToolExecutor(sandbox, AgentConfig(**cfg_kw))


def test_edit_file_unique_match(toy_sandbox):
    ex = make(toy_sandbox)
    res = ex.execute(
        "edit_file",
        {"path": "stats_utils/core.py", "old_str": MEDIAN_LINE, "new_str": "    return ordered[mid]  # edited\n"},
    )
    assert not res.is_error, res.content
    assert "# edited" in toy_sandbox.read_file("stats_utils/core.py")
    assert ex.files_edited == {"stats_utils/core.py"} and ex.edited_since_last_test


def test_edit_file_not_found_gives_whitespace_hint(toy_sandbox):
    ex = make(toy_sandbox)
    res = ex.execute("edit_file", {"path": "stats_utils/core.py", "old_str": "return ordered[mid]\n", "new_str": "x"})
    assert not res.is_error  # substring without indentation still matches once
    res = ex.execute("edit_file", {"path": "stats_utils/core.py", "old_str": "  mid = len(ordered)//2", "new_str": "x"})
    assert res.is_error and "not found" in res.content


def test_edit_file_multiple_matches_rejected(toy_sandbox):
    ex = make(toy_sandbox)
    res = ex.execute(
        "edit_file", {"path": "stats_utils/core.py", "old_str": "if not values:", "new_str": "if not values :"}
    )
    assert res.is_error and "matches 2 times" in res.content


def test_edit_file_guards(toy_sandbox):
    ex = make(toy_sandbox)
    assert ex.execute("edit_file", {"path": "stats_utils/core.py", "old_str": "", "new_str": "x"}).is_error
    assert ex.execute("edit_file", {"path": "stats_utils/core.py", "old_str": "mid", "new_str": "mid"}).is_error
    assert ex.execute("edit_file", {"path": "nope.py", "old_str": "a", "new_str": "b"}).is_error


def test_edit_file_warns_on_syntax_error(toy_sandbox):
    ex = make(toy_sandbox)
    res = ex.execute("edit_file", {"path": "stats_utils/core.py", "old_str": MEDIAN_LINE, "new_str": "    return (\n"})
    assert "no longer compiles" in res.content


def test_path_escape_rejected(toy_sandbox):
    ex = make(toy_sandbox)
    res = ex.execute("read_file", {"path": "../../etc/passwd"})
    assert res.is_error and "outside" in res.content


def test_read_file_ranges(toy_sandbox):
    ex = make(toy_sandbox)
    res = ex.execute("read_file", {"path": "stats_utils/core.py", "start_line": 10, "end_line": 11})
    assert res.content.splitlines()[0].startswith("10\t") and "def median" in res.content
    assert "stats_utils/core.py" in ex.files_read
    assert ex.execute("read_file", {"path": "stats_utils/core.py", "start_line": 999}).is_error


def test_create_file_refuses_overwrite(toy_sandbox):
    ex = make(toy_sandbox)
    assert ex.execute("create_file", {"path": "stats_utils/core.py", "content": "x"}).is_error
    assert not ex.execute("create_file", {"path": "stats_utils/new.py", "content": "x = 1\n"}).is_error


def test_search_and_find(toy_sandbox):
    ex = make(toy_sandbox)
    res = ex.execute("search", {"pattern": "def (mean|median)", "include": "*.py"})
    assert "stats_utils/core.py:" in res.content and "def median" in res.content
    assert "No matches" in ex.execute("search", {"pattern": "zzz_not_here"}).content
    res = ex.execute("find_files", {"name_pattern": "test_*.py"})
    assert res.content.strip() == "tests/test_core.py"


def test_bash_and_run_tests(toy_sandbox):
    ex = make(toy_sandbox)
    ex.edited_since_last_test = True
    res = ex.execute("run_tests", {"command": "python -m pytest -q tests"})
    assert "[exit code 0]" in res.content and ex.test_runs == 1 and not ex.edited_since_last_test


def test_bash_timeout(toy_sandbox):
    toy_sandbox.command_timeout_s = 1
    res = make(toy_sandbox).execute("bash", {"command": "sleep 5"})
    assert "TIMED OUT" in res.content


def test_bad_arguments_and_unknown_tool(toy_sandbox):
    ex = make(toy_sandbox)
    assert ex.execute("read_file", {"pth": "x"}).is_error
    assert ex.execute("rm_rf", {}).is_error


def test_tool_specs_follow_scaffold():
    cfg = AgentConfig()
    assert "report_localization" in {s["name"] for s in tool_specs(cfg.scaffold)}
    cfg.scaffold.localization = False
    assert "report_localization" not in {s["name"] for s in tool_specs(cfg.scaffold)}


def test_truncate_middle():
    s = "a" * 100 + "b" * 100
    out = truncate_middle(s, 50)
    assert out.startswith("a" * 30) and out.endswith("b" * 20) and "truncated" in out
    assert truncate_middle("short", 50) == "short"
