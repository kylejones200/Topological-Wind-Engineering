"""
Create turbulence classification animation. Run from repo root: python path/to/37_create_animation.py [--config path/to/config.yaml]
"""
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR
for _ in range(15):
    if (_REPO_ROOT / "config" / "default.yaml").is_file() or (_REPO_ROOT / "pyproject.toml").is_file():
        break
    _REPO_ROOT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import logging

from config.load import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
FPS, N_FRAMES = (10, 100)
n_samples = 400
turbulence_intensity = 0.1 + 0.15 * np.sin(2 * np.pi * np.arange(n_samples) / 200)
wind_speed = 8 + 2 * np.sin(2 * np.pi * np.arange(n_samples) / 100)
for i in range(n_samples):
    wind_speed[i] += np.random.normal(0, max(0.1, turbulence_intensity[i] * abs(wind_speed[i])))
fig = plt.figure(figsize=(14, 8), facecolor='white')
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1, ax2, ax3 = (fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def update(frame):
    """
    Perform update operation.

    Args:
        frame: Frame parameter.

    Returns:
        Result of the operation..
    """
    ax1.clear()
    ax2.clear()
    ax3.clear()
    end_idx = int(frame / N_FRAMES * n_samples)
    ax1.plot(wind_speed[:end_idx], 'black', linewidth=1.5)
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Wind Speed (m/s)', fontsize=10)
    ax1.set_title(f'Turbulent Wind Speed - Frame {frame + 1}/{N_FRAMES}', fontsize=11, fontweight='normal')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    if end_idx > 30:
        window_ti = turbulence_intensity[max(0, end_idx - 30):end_idx]
        ax2.plot(window_ti, 'gray', linewidth=2)
        ax2.set_xlabel('Recent Time', fontsize=10)
        ax2.set_ylabel('Turbulence Intensity', fontsize=10)
        ax2.set_title('Turbulence Intensity', fontsize=11, fontweight='normal')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    current_ti = turbulence_intensity[min(end_idx - 1, n_samples - 1)]
    classification = 'High' if current_ti > 0.2 else 'Medium' if current_ti > 0.15 else 'Low'
    color = 'red' if current_ti > 0.2 else 'orange' if current_ti > 0.15 else 'green'
    ax3.text(0.5, 0.7, f'{classification} Turbulence', ha='center', va='center', fontsize=16, fontweight='bold', color=color, transform=ax3.transAxes)
    ax3.text(0.5, 0.4, f'TI: {current_ti:.2f}', ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Classification', fontsize=11, fontweight='normal')
    return []


def _get_animation_output_path(cfg):
    """Resolve output directory from config and return path to GIF file."""
    turb = cfg.get("turbulence", {})
    figures_subdir = turb.get("figures_subdir", "figures_turbulence")
    out_dir = _SCRIPT_DIR / figures_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "37_turbulence_animation.gif"


def _build_and_save_animation(output_path):
    """Build FuncAnimation, save to output_path, and close figure."""
    logger.info("Creating turbulence animation...")
    anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True)
    anim.save(str(output_path), writer="pillow", fps=FPS, dpi=100)
    logger.info(f"✓ Animation saved: {output_path}")
    plt.close()


def main(config_path=None):
    cfg = load_config(config_path)
    seed = cfg.get("global", {}).get("random_seed", 42)
    np.random.seed(seed)
    output_path = _get_animation_output_path(cfg)
    _build_and_save_animation(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create turbulence animation")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)