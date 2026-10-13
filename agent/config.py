"""Run configuration.

Everything that can change an agent's behaviour lives here and is written into
every trajectory, so any result can be traced back to the exact settings that
produced it. Unknown keys raise immediately: a typo in a YAML file should never
silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Pricing:
    """USD per million tokens. Verify against the current Anthropic pricing page."""

    input: float = 0.0
    output: float = 0.0
    cache_write: float = 0.0
    cache_read: float = 0.0


@dataclass
class ScaffoldConfig:
    """Optional scaffold stages. Each one can be ablated independently."""

    planning: bool = True
    localization: bool = True  # must call report_localization before any edit
    self_verification: bool = True  # must run tests after the last edit before submit


@dataclass
class SandboxConfig:
    workdir: str = "/testbed"
    network: str = "none"
    command_timeout_s: int = 180
    memory: str = "4g"
    cpus: str = "2"
    # Runs before every command. Activates the SWE-bench conda env when present,
    # and is a no-op in images without conda.
    env_prefix: str = (
        "if [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then "
        ". /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed; fi"
    )


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-5-5"
    # Covers thinking + visible output; thinking tokens are billed as output.
    max_tokens_per_turn: int = 16_000
    # Current models reject non-default temperature/top_p/top_k, so sampling is
    # pinned through these instead. Set thinking to "" to omit the field
    # (needed for older models that do not support adaptive thinking).
    thinking: str = "adaptive"  # "adaptive" | "between_tools" | ""
    thinking_display: str = "summarized"  # readable thinking in trajectories; "" for API default
    effort: str = "high"  # low | medium | high | xhigh | max; "" to omit
    seed: int = 0  # controls task sampling; API sampling itself is not seedable

    # Per-task caps. Whichever is hit first ends the episode.
    max_steps: int = 60
    max_cost_usd: float = 3.0
    max_wallclock_s: int = 1800

    # Context management (see agent/context.py)
    context_budget_tokens: int = 150_000
    max_context_resets: int = 3
    tool_output_max_chars: int = 12_000

    # Stuck detection
    max_consecutive_errors: int = 5
    max_repeated_calls: int = 3

    pricing: Pricing = field(default_factory=Pricing)
    scaffold: ScaffoldConfig = field(default_factory=ScaffoldConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Short stable hash of every setting, for tagging runs."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


_NESTED = {"pricing": Pricing, "scaffold": ScaffoldConfig, "sandbox": SandboxConfig}


def _build(cls, data: dict[str, Any], where: str):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config key(s) in {where}: {sorted(unknown)}")
    kwargs = {}
    for key, value in data.items():
        if cls is AgentConfig and key in _NESTED:
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{key}' must be a mapping")
            kwargs[key] = _build(_NESTED[key], value, f"{where}.{key}")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _coerce(raw: str) -> Any:
    """Parse a CLI override value with YAML rules (true/false, numbers, strings)."""
    return yaml.safe_load(raw)


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply dotted-key overrides, e.g. {"scaffold.localization": False}."""
    data = json.loads(json.dumps(data))  # deep copy
    for dotted, value in overrides.items():
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data


def parse_set_args(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        out[key.strip()] = _coerce(raw)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AgentConfig:
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    if overrides:
        data = apply_overrides(data, overrides)
    return _build(AgentConfig, data, str(path or "<defaults>"))
