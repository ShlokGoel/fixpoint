"""A scripted stand-in for the model, used by tests and the offline demo.

It replays a fixed list of turns, so the loop, tools, gating and logging can be
exercised end to end without an API key. It is never used for reported results.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any

from .llm import LLMResponse, Usage

_ids = itertools.count(1)


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def tool(name: str, **inputs: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"toolu_fake_{next(_ids):04d}", "name": name, "input": inputs}


class ScriptedLLM:
    def __init__(self, turns: list[list[dict[str, Any]]], tokens_per_call: int = 1000, repeat_last: bool = False):
        self.turns = list(turns)
        self.tokens_per_call = tokens_per_call
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    def call(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        # Snapshot: the loop keeps appending to the same list after this call returns.
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        index = len(self.calls) - 1
        if index < len(self.turns) and isinstance(self.turns[index], LLMResponse):
            return self.turns[index]  # full control over stop_reason / usage
        if index < len(self.turns):
            blocks = self.turns[index]
        elif self.repeat_last and self.turns:
            blocks = [
                dict(b, id=f"toolu_fake_{next(_ids):04d}") if b["type"] == "tool_use" else b for b in self.turns[-1]
            ]
        else:
            blocks = [text("I have nothing further to do.")]
        stop = "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn"
        return LLMResponse(
            content=blocks,
            stop_reason=stop,
            usage=Usage(input_tokens=self.tokens_per_call, output_tokens=self.tokens_per_call // 10),
            model="scripted-fake",
        )
