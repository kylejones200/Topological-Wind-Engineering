"""
Create wake detection animation. Run from repo root: python path/to/35_create_animation.py [--config path/to/config.yaml]
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
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import logging

from config.load import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
FPS = 10
DURATION_SECONDS = 10
N_FRAMES = FPS * DURATION_SECONDS
n_samples = 400
time = np.arange(n_samples)
power1 = 2000 + 200 * np.sin(2 * np.pi * time / 100) + np.random.normal(0, 50, n_samples)
wind_direction = np.sin(2 * np.pi * time / 200)
in_wake = wind_direction > 0
power2 = np.where(in_wake, 0.65, 1.0) * (2000 + 200 * np.sin(2 * np.pi * time / 100))
power2 += np.random.normal(0, 50, n_samples)
fig = plt.figure(figsize=(14, 8), facecolor='white')
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
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
    window_size = 50
    current_idx = int(frame / N_FRAMES * (n_samples - window_size))
    end_idx = current_idx + window_size
    ax1.plot(time[:end_idx], power1[:end_idx], 'black', linewidth=2, label='Turbine 1 (Upstream)', alpha=0.7)
    ax1.plot(time[:end_idx], power2[:end_idx], 'gray', linewidth=2, label='Turbine 2 (Downstream)', alpha=0.7)
    wake_in_window = in_wake[current_idx:end_idx]
    if np.any(wake_in_window):
        wake_times = time[current_idx:end_idx][wake_in_window]
        for t in wake_times:
            ax1.axvspan(t - 0.5, t + 0.5, alpha=0.1, color='red')
    ax1.set_xlabel('Time (10-min intervals)', fontsize=10)
    ax1.set_ylabel('Power Output (kW)', fontsize=10)
    ax1.set_title(f'Wake Effect Detection - Frame {frame + 1}/{N_FRAMES}', fontsize=11, fontweight='normal')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    deficit = (power1[current_idx:end_idx] - power2[current_idx:end_idx]) / power1[current_idx:end_idx] * 100
    ax2.hist(deficit, bins=20, color='gray', alpha=0.7, edgecolor='black')
    ax2.axvline(deficit.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {deficit.mean():.1f}%')
    ax2.set_xlabel('Power Deficit (%)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Power Deficit Distribution', fontsize=11, fontweight='normal')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    wake_rate = np.sum(in_wake[current_idx:end_idx]) / window_size * 100
    status = 'WAKE DETECTED' if wake_rate > 30 else 'NO WAKE'
    color = 'red' if wake_rate > 30 else 'green'
    ax3.text(0.5, 0.7, status, ha='center', va='center', fontsize=16, fontweight='bold', color=color, transform=ax3.transAxes)
    ax3.text(0.5, 0.4, f'Wake Probability: {wake_rate:.1f}%', ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.5, 0.2, f'Avg Deficit: {deficit.mean():.1f}%', ha='center', va='center', fontsize=10, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Detection Status', fontsize=11, fontweight='normal')
    return []


def _get_animation_output_path(cfg):
    """Resolve output directory from config and return path to GIF file."""
    wd = cfg.get("wake_detection", {})
    figures_subdir = wd.get("figures_subdir", "figures")
    out_dir = _SCRIPT_DIR / figures_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "35_wake_detection_animation.gif"


def _build_and_save_animation(output_path):
    """Build FuncAnimation, save to output_path, and close figure."""
    logger.info("Creating wake detection animation...")
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
    parser = argparse.ArgumentParser(description="Create wake detection animation")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)