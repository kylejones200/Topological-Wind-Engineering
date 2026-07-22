"""Tests for config loading."""
from pathlib import Path

from config.load import load_config, get_repo_root


def test_load_default_config_has_expected_sections():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    assert "global" in cfg
    assert "nrel" in cfg
    assert "regime_transition" in cfg
    assert cfg["regime_transition"]["window_samples"] == 30


def test_get_repo_root_finds_pyproject():
    root = get_repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "config" / "default.yaml").is_file()
