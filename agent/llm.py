"""Thin wrapper around the raw Anthropic Messages API.

No agent framework: one `messages.create` call per step. This module only
  - pins the model and sampling settings from the config,
  - adds prompt-cache breakpoints (system prompt + newest message),
  - normalises response blocks to plain dicts so they can be logged and replayed,
  - accounts tokens and dollars per call.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .config import AgentConfig, Pricing


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    @property
    def prompt_tokens(self) -> int:
        """Everything the model read on this call, cached or not."""
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens

    def cost(self, pricing: Pricing) -> float:
        return (
            self.input_tokens * pricing.input
            + self.output_tokens * pricing.output
            + self.cache_creation_input_tokens * pricing.cache_write
            + self.cache_read_input_tokens * pricing.cache_read
        ) / 1_000_000

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_api(cls, usage: Any) -> Usage:
        def get(name: str) -> int:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            return int(value or 0)

        return cls(
            get("input_tokens"),
            get("output_tokens"),
            get("cache_creation_input_tokens"),
            get("cache_read_input_tokens"),
        )


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: Usage
    model: str  # the model string the API reports it served
    request_id: str | None = None
    latency_s: float = 0.0


class LLMClient(Protocol):
    def call(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


def normalize_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return {k: v for k, v in block.items() if v is not None}
    return block.model_dump(exclude_none=True)


def with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Return a copy of `messages` with one cache breakpoint on the newest block.

    Only the newest message is marked, so the request never exceeds the API's
    breakpoint limit no matter how long the conversation gets.
    """
    if not messages:
        return messages
    out = copy.deepcopy(messages)
    last = out[-1]
    content = last["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
        last["content"] = content
    if content:
        content[-1]["cache_control"] = {"type": "ephemeral"}
    return out


class AnthropicClient:
    def __init__(self, cfg: AgentConfig, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=5, timeout=600.0)
        self.client = client
        self.cfg = cfg

    def request_kwargs(self, system: str, messages: list[dict], tools: list[dict]) -> dict[str, Any]:
        cfg = self.cfg
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens_per_turn,
            # The breakpoint on the system block caches tools + system together.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": tools,
            "messages": with_cache_breakpoint(messages),
        }
        # Sent explicitly (even when equal to the API default) so runs stay pinned
        # if a default changes later.
        if cfg.thinking:
            thinking: dict[str, Any] = {"type": cfg.thinking}
            if cfg.thinking == "adaptive" and cfg.thinking_display:
                thinking["display"] = cfg.thinking_display
            kwargs["thinking"] = thinking
        if cfg.effort:
            kwargs["output_config"] = {"effort": cfg.effort}
        return kwargs

    def call(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        start = time.monotonic()
        response = self.client.messages.create(**self.request_kwargs(system, messages, tools))
        return LLMResponse(
            content=[normalize_block(b) for b in response.content],
            stop_reason=str(response.stop_reason),
            usage=Usage.from_api(response.usage),
            model=str(getattr(response, "model", self.cfg.model)),
            request_id=getattr(response, "_request_id", None) or getattr(response, "id", None),
            latency_s=round(time.monotonic() - start, 3),
        )
