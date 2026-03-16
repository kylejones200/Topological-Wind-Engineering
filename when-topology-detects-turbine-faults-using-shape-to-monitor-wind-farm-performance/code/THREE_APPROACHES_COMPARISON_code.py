"""Snippet: three approaches comparison. Run from repo root with optional --config."""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
for _ in range(15):
    if (_REPO_ROOT / "config" / "default.yaml").is_file() or (_REPO_ROOT / "pyproject.toml").is_file():
        break
    _REPO_ROOT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Snippet (requires power_curve, wind_speed, actual_power, mean in scope)
# expected_power = power_curve(wind_speed)
# deviation = actual_power - expected_power
# power_ratio = actual_power / expected_power
# label = 1 if mean(power_ratio) < 0.8 else 0

if __name__ == "__main__":
    from config.load import load_config
    import argparse
    parser = argparse.ArgumentParser(description="THREE_APPROACHES_COMPARISON snippet")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    load_config(args.config)