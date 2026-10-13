from agent.config import AgentConfig
from agent.fake_llm import ScriptedLLM, text, tool
from agent.loop import Agent
from agent.trajectory import TrajectoryLogger, read_trajectory

OLD = "    mid = len(ordered) // 2\n    return ordered[mid]\n"
NEW = "    mid = len(ordered) // 2\n    if len(ordered) % 2 == 0:\n        return (ordered[mid - 1] + ordered[mid]) / 2\n    return ordered[mid]\n"


def good_script():
    return [
        [text("Looking."), tool("search", pattern="def median")],
        [tool("read_file", path="stats_utils/core.py")],
        [tool("report_localization", files=["stats_utils/core.py"], rationale="median is here")],
        [tool("edit_file", path="stats_utils/core.py", old_str=OLD, new_str=NEW)],
        [tool("run_tests", command="python -m pytest -q tests")],
        [tool("submit", summary="fixed even-length median")],
    ]


def run(script, sandbox, task, cfg=None, logger=None, **kw):
    llm = ScriptedLLM(script, **kw)
    agent = Agent(llm, sandbox, cfg or AgentConfig(), logger)
    return agent.run(task), llm


def test_happy_path_produces_patch_and_full_trajectory(toy_sandbox, toy_task, cfg, tmp_path):
    path = tmp_path / "t.jsonl"
    with TrajectoryLogger(path) as log:
        result, _ = run(good_script(), toy_sandbox, toy_task, cfg, log)
    assert result.stop_cause == "submit" and result.steps == 6
    assert "+    if len(ordered) % 2 == 0:" in result.patch
    assert result.localized_files == ["stats_utils/core.py"] and result.test_runs == 1
    assert result.cost_usd > 0
    events = read_trajectory(path)
    types = [e["type"] for e in events]
    assert types[0] == "episode_start" and types[-1] == "episode_end"
    assert types.count("model_response") == 6 and types.count("tool_call") == 6
    assert events[0]["payload"]["prompt_hashes"] and events[0]["payload"]["config_fingerprint"]
    assert [e["seq"] for e in events] == list(range(len(events)))


def test_edit_before_localization_is_rejected(toy_sandbox, toy_task):
    script = [[tool("edit_file", path="stats_utils/core.py", old_str=OLD, new_str=NEW)]]
    result, llm = run(script, toy_sandbox, toy_task, logger=None)
    first_result = llm.calls[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] and "report_localization" in first_result["content"]
    assert result.patch == ""


def test_no_localization_gate_when_ablated(toy_sandbox, toy_task):
    cfg = AgentConfig()
    cfg.scaffold.localization = False
    cfg.scaffold.self_verification = False
    script = [[tool("edit_file", path="stats_utils/core.py", old_str=OLD, new_str=NEW)], [tool("submit", summary="x")]]
    result, llm = run(script, toy_sandbox, toy_task, cfg)
    assert result.stop_cause == "submit" and result.patch
    assert all(t["name"] != "report_localization" for t in llm.calls[0]["tools"])


def test_submit_without_tests_rejected_once_then_allowed(toy_sandbox, toy_task):
    script = good_script()
    del script[4]  # drop run_tests
    script.append([tool("submit", summary="again")])
    result, llm = run(script, toy_sandbox, toy_task)
    rejected = llm.calls[5]["messages"][-1]["content"][0]
    assert rejected["is_error"] and "run any tests" in rejected["content"]
    assert result.stop_cause == "submit" and result.steps == 6


def test_empty_patch_submit_rejected_once(toy_sandbox, toy_task):
    script = [[tool("submit", summary="nothing")], [tool("submit", summary="really nothing")]]
    result, llm = run(script, toy_sandbox, toy_task)
    assert "patch would be empty" in llm.calls[1]["messages"][-1]["content"][0]["content"]
    assert result.stop_cause == "submit" and result.patch == ""


def test_end_turn_twice_stops(toy_sandbox, toy_task):
    result, llm = run([[text("done?")], [text("yes done")]], toy_sandbox, toy_task)
    assert result.stop_cause == "end_turn_no_submit" and result.steps == 2
    assert "without calling a tool" in llm.calls[1]["messages"][-1]["content"]


def test_max_steps(toy_sandbox, toy_task):
    cfg = AgentConfig(max_steps=3)
    script = [[tool("bash", command=f"echo {i}")] for i in range(10)]
    result, _ = run(script, toy_sandbox, toy_task, cfg)
    assert result.stop_cause == "max_steps" and result.steps == 3


def test_cost_cap(toy_sandbox, toy_task):
    cfg = AgentConfig(max_cost_usd=0.01)
    cfg.pricing.input = 2.0
    script = [[tool("bash", command=f"echo {i}")] for i in range(10)]
    result, _ = run(script, toy_sandbox, toy_task, cfg, tokens_per_call=4000)  # $0.008 per call
    assert result.stop_cause == "cost_cap" and result.steps == 2


def test_wallclock_cap(toy_sandbox, toy_task):
    ticks = iter(range(0, 10_000, 1000))
    cfg = AgentConfig(max_wallclock_s=2500)
    agent = Agent(
        ScriptedLLM([[tool("bash", command="true")]], repeat_last=True), toy_sandbox, cfg, clock=lambda: next(ticks)
    )
    assert agent.run(toy_task).stop_cause == "wallclock_cap"


def test_repeated_identical_calls_stop(toy_sandbox, toy_task):
    result, _ = run([[tool("bash", command="ls")]], toy_sandbox, toy_task, repeat_last=True)
    assert result.stop_cause == "no_progress" and result.steps == 3


def test_consecutive_errors_stop(toy_sandbox, toy_task):
    script = [[tool("read_file", path=f"missing_{i}.py")] for i in range(10)]
    result, _ = run(script, toy_sandbox, toy_task)
    assert result.stop_cause == "no_progress" and result.steps == 5


def test_api_error_is_recorded(toy_sandbox, toy_task):
    class Broken:
        def call(self, *a):
            raise RuntimeError("overloaded")

    log = TrajectoryLogger(None)
    result = Agent(Broken(), toy_sandbox, AgentConfig(), log).run(toy_task)
    assert result.stop_cause == "api_error"
    assert any(e["type"] == "api_error" and "overloaded" in e["payload"]["error"] for e in log.events)


def test_context_reset_when_over_budget(toy_sandbox, toy_task):
    cfg = AgentConfig(context_budget_tokens=1500, max_context_resets=2)
    script = good_script()
    log = TrajectoryLogger(None)
    result = Agent(ScriptedLLM(script, tokens_per_call=2000), toy_sandbox, cfg, log).run(toy_task)
    resets = [e for e in log.events if e["type"] == "context_reset"]
    assert len(resets) == 2 and result.context_resets == 2
    assert result.stop_cause == "context_overflow"  # third time over budget, no resets left
    assert "[Context reset #2]" in resets[-1]["payload"]["resume_prompt"]


def test_reset_conversation_is_fresh_and_append_only(toy_sandbox, toy_task):
    cfg = AgentConfig(context_budget_tokens=1500, max_context_resets=5)
    llm = ScriptedLLM(good_script(), tokens_per_call=2000)
    Agent(llm, toy_sandbox, cfg).run(toy_task)
    # After a reset the model sees exactly one user message: task + state summary.
    assert len(llm.calls[1]["messages"]) == 1
    assert "Files you have read: stats_utils/core.py" in llm.calls[2]["messages"][0]["content"]


def test_model_context_window_exceeded_triggers_reset(toy_sandbox, toy_task):
    from agent.llm import LLMResponse, Usage

    overflow = LLMResponse(content=[], stop_reason="model_context_window_exceeded", usage=Usage(10, 0), model="m")
    script = [[tool("bash", command="echo hi")], overflow, [tool("submit", summary="x")], [tool("submit", summary="x")]]
    log = TrajectoryLogger(None)
    result = Agent(ScriptedLLM(script), toy_sandbox, AgentConfig(), log).run(toy_task)
    assert any(e["type"] == "context_reset" and e["payload"]["window_exceeded"] for e in log.events)
    assert result.stop_cause == "submit"


def test_refusal_stops_episode(toy_sandbox, toy_task):
    from agent.llm import LLMResponse, Usage

    refusal = LLMResponse(content=[], stop_reason="refusal", usage=Usage(10, 0), model="m")
    result, _ = run([refusal], toy_sandbox, toy_task)
    assert result.stop_cause == "refusal"


def test_thinking_blocks_are_passed_back_unchanged(toy_sandbox, toy_task):
    thinking = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    script = [[thinking, tool("bash", command="echo hi")], [tool("submit", summary="x")]]
    _, llm = run(script, toy_sandbox, toy_task)
    assert llm.calls[1]["messages"][1]["content"][0] == thinking
