from types import SimpleNamespace

from agent.config import AgentConfig, Pricing
from agent.llm import AnthropicClient, Usage, with_cache_breakpoint


def test_usage_cost_and_addition():
    pricing = Pricing(input=2.0, output=10.0, cache_write=2.5, cache_read=0.1)
    u = Usage(1_000_000, 100_000, 200_000, 2_000_000)
    assert abs(u.cost(pricing) - (2.0 + 1.0 + 0.5 + 0.2)) < 1e-9
    total = u + Usage(1, 2, 3, 4)
    assert total.input_tokens == 1_000_001 and total.cache_read_input_tokens == 2_000_004
    assert u.prompt_tokens == 3_200_000


def test_cache_breakpoint_only_on_newest_block_and_no_mutation():
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "out"}]},
    ]
    out = with_cache_breakpoint(msgs)
    assert "cache_control" in out[-1]["content"][-1]
    assert "cache_control" not in str(out[:-1])
    assert "cache_control" not in str(msgs)  # caller's history is untouched


def test_cache_breakpoint_converts_string_content():
    out = with_cache_breakpoint([{"role": "user", "content": "task"}])
    assert out[0]["content"] == [{"type": "text", "text": "task", "cache_control": {"type": "ephemeral"}}]


class _Block:
    def __init__(self, **kw):
        self.kw = kw

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in self.kw.items() if not (exclude_none and v is None)}


def test_anthropic_client_request_shape_and_normalisation():
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[
                _Block(type="text", text="ok", citations=None),
                _Block(type="tool_use", id="t1", name="bash", input={"command": "ls"}),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=5, cache_creation_input_tokens=None, cache_read_input_tokens=7
            ),
            model="claude-test-served",
            id="msg_1",
        )

    fake = SimpleNamespace(messages=SimpleNamespace(create=create))
    cfg = AgentConfig(model="claude-test", max_tokens_per_turn=123)
    resp = AnthropicClient(cfg, client=fake).call("sys", [{"role": "user", "content": "go"}], [{"name": "bash"}])

    assert captured["model"] == "claude-test" and captured["max_tokens"] == 123
    assert not {"temperature", "top_p", "top_k"} & set(captured)  # rejected by current models
    assert captured["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert captured["output_config"] == {"effort": "high"}
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert resp.content[0] == {"type": "text", "text": "ok"}  # None fields dropped
    assert resp.content[1]["name"] == "bash"
    assert resp.usage == Usage(10, 5, 0, 7)
    assert resp.model == "claude-test-served"


def test_thinking_and_effort_can_be_omitted_for_older_models():
    cfg = AgentConfig(thinking="", effort="")
    kwargs = AnthropicClient(cfg, client=object()).request_kwargs("s", [{"role": "user", "content": "x"}], [])
    assert "thinking" not in kwargs and "output_config" not in kwargs
    cfg = AgentConfig(thinking="between_tools")
    kwargs = AnthropicClient(cfg, client=object()).request_kwargs("s", [{"role": "user", "content": "x"}], [])
    assert kwargs["thinking"] == {"type": "between_tools"}  # takes no display field
