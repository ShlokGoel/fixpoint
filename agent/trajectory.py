"""Structured trajectory logging: one JSON object per line, one file per task.

Every model response and every tool call is recorded in order, so any episode
can be replayed step by step (see viewer/trajectory_viewer.html) and audited
after the fact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.events: list[dict[str, Any]] = []
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("w", encoding="utf-8")

    def log(self, type_: str, **payload: Any) -> dict[str, Any]:
        event = {
            "seq": len(self.events),
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "type": type_,
            "payload": payload,
        }
        self.events.append(event)
        if self._fh:
            self._fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
            self._fh.flush()
        return event

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_trajectory(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def episode_end(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event["type"] == "episode_end":
            return event["payload"]
    return {}
