from pathlib import Path

import pytest

from agent.config import load_config, parse_set_args

ROOT = Path(__file__).resolve().parent.parent


def test_default_yaml_loads():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert cfg.model
    assert cfg.pricing.input > 0
    assert cfg.scaffold.localization is True


def test_overrides_and_fingerprint():
    base = load_config(ROOT / "configs" / "default.yaml")
    ablated = load_config(ROOT / "configs" / "default.yaml", parse_set_args(["scaffold.localization=false"]))
    assert ablated.scaffold.localization is False
    assert base.fingerprint() != ablated.fingerprint()
    assert base.fingerprint() == load_config(ROOT / "configs" / "default.yaml").fingerprint()


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("max_step: 10\n")  # typo of max_steps
    with pytest.raises(ValueError, match="max_step"):
        load_config(bad)


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ValueError, match="localisation"):
        load_config(None, {"scaffold.localisation": False})


def test_set_arg_parsing():
    assert parse_set_args(["max_steps=5", "model=claude-x", "max_cost_usd=0.5"]) == {
        "max_steps": 5,
        "model": "claude-x",
        "max_cost_usd": 0.5,
    }
    with pytest.raises(ValueError):
        parse_set_args(["oops"])
