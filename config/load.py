"""
Load configuration from YAML. Used by all runnable scripts.
No magic numbers: scripts get defaults from this config; override via CONFIG_PATH or --config.
"""
from pathlib import Path
import os
from typing import Any, Dict, Optional

def _find_config_path() -> Path:
    """Find config/default.yaml by walking up from cwd until we find repo root."""
    candidate = Path.cwd().resolve()
    for _ in range(20):
        path = candidate / "config" / "default.yaml"
        if path.is_file():
            return path
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    # Fallback: next to this file
    return Path(__file__).resolve().parent / "default.yaml"

def load_config(config_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Load config from YAML. Merges with env overrides (e.g. NREL_API_KEY).
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for config. Install with: pip install pyyaml")

    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH")
    if config_path is not None:
        path = Path(config_path).resolve()
    else:
        path = _find_config_path()

    if not path.is_file():
        return _default_config()

    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    # Env overrides
    if "nrel" in cfg and os.environ.get("NREL_API_KEY"):
        cfg["nrel"] = dict(cfg.get("nrel", {}))
        cfg["nrel"]["api_key"] = os.environ["NREL_API_KEY"]

    return _deep_merge(_default_config(), cfg)

def _default_config() -> Dict[str, Any]:
    """Minimal in-code defaults if YAML missing."""
    return {
        "global": {"random_seed": 42, "output_dir": "figures", "log_level": "INFO"},
        "nrel": {"lat": 41.5, "lon": -100.5, "years": [2010, 2011, 2012], "request_timeout_seconds": 120},
        "simulation": {"window_size": 288, "n_windows": 120, "n_turbines": 6, "rated_power_kw": 2000},
        "farm_coordination": {"figures_subdir": "figures_coordination", "test_size": 0.3},
        "wake_detection": {"figures_subdir": "figures_wake"},
        "yaw_mapper": {"figures_subdir": "figures_yaw", "window_size": 10},
        "regime_tda": {"figures_subdir": "figures"},
    }

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def get_repo_root() -> Path:
    """Return repo root (directory containing config/default.yaml or pyproject.toml)."""
    candidate = Path.cwd().resolve()
    for _ in range(20):
        if (candidate / "config" / "default.yaml").is_file() or (candidate / "pyproject.toml").is_file():
            return candidate
        candidate = candidate.parent
    return Path.cwd().resolve()
