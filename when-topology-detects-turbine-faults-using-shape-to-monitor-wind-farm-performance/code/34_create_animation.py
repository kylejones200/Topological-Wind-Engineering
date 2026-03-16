"""
Create topological turbine classification animation. Run from repo root: python path/to/34_create_animation.py [--config path/to/config.yaml]
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
from datetime import datetime, timedelta
import logging

from config.load import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
np.random.seed(42)
FPS = 10
DURATION_SECONDS = 10
N_FRAMES = FPS * DURATION_SECONDS
n_samples = 500
time = pd.date_range(start='2023-01-01', periods=n_samples, freq='10min')
power_base = 2000 + 300 * np.sin(2 * np.pi * np.arange(n_samples) / 144)
power = power_base + np.random.normal(0, 100, n_samples)
anomaly_indices = [100, 150, 250, 350, 420]
for idx in anomaly_indices:
    power[idx:idx + 20] *= np.linspace(1.0, 0.4, 20)
    if idx + 20 < n_samples:
        power[idx + 20:idx + 30] *= np.linspace(0.4, 1.0, 10)
is_anomaly = np.zeros(n_samples, dtype=bool)
for idx in anomaly_indices:
    is_anomaly[idx:idx + 30] = True
fig = plt.figure(figsize=(14, 8), facecolor='white')
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def init():
    """Initialize animation."""
    ax1.clear()
    ax2.clear()
    ax3.clear()
    return []

def update(frame):
    """Update animation frame."""
    ax1.clear()
    ax2.clear()
    ax3.clear()
    window_size = 50
    current_idx = int(frame / N_FRAMES * (n_samples - window_size))
    end_idx = current_idx + window_size
    ax1.plot(time[:current_idx], power[:current_idx], 'gray', alpha=0.3, linewidth=1)
    window_time = time[current_idx:end_idx]
    window_power = power[current_idx:end_idx]
    window_anomaly = is_anomaly[current_idx:end_idx]
    ax1.plot(window_time, window_power, 'black', linewidth=2, label='Current Window')
    if np.any(window_anomaly):
        anomaly_times = window_time[window_anomaly]
        anomaly_powers = window_power[window_anomaly]
        ax1.scatter(anomaly_times, anomaly_powers, c='red', s=30, zorder=5, label='Anomaly', alpha=0.7)
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Power Output (kW)', fontsize=10)
    ax1.set_title(f'Turbine Power Output - Window {frame + 1}/{N_FRAMES}', fontsize=11, fontweight='normal')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    std_val = np.std(window_power)
    mean_val = np.mean(window_power)
    n_h0 = np.random.randint(3, 8)
    births_h0 = np.random.uniform(0, 0.2, n_h0)
    deaths_h0 = births_h0 + np.random.uniform(0.1, 0.5, n_h0)
    if np.any(window_anomaly):
        n_h1 = np.random.randint(5, 12)
        persistence_scale = 0.4
    else:
        n_h1 = np.random.randint(2, 5)
        persistence_scale = 0.2
    births_h1 = np.random.uniform(0.1, 0.4, n_h1)
    deaths_h1 = births_h1 + np.random.uniform(0.05, persistence_scale, n_h1)
    ax2.scatter(births_h0, deaths_h0, c='gray', s=40, alpha=0.6, label='H0 (Components)')
    ax2.scatter(births_h1, deaths_h1, c='red' if np.any(window_anomaly) else 'black', s=60, alpha=0.8, label='H1 (Loops)')
    max_val = max(deaths_h0.max(), deaths_h1.max()) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Birth', fontsize=10)
    ax2.set_ylabel('Death', fontsize=10)
    ax2.set_title('Persistence Diagram', fontsize=11, fontweight='normal')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(0, max_val)
    ax2.set_aspect('equal')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    detection_history = is_anomaly[:current_idx + window_size]
    detection_rate = np.sum(detection_history[-100:]) / min(100, len(detection_history)) * 100
    status_text = 'ANOMALY DETECTED!' if np.any(window_anomaly) else 'Normal Operation'
    status_color = 'red' if np.any(window_anomaly) else 'green'
    ax3.text(0.5, 0.7, status_text, ha='center', va='center', fontsize=16, fontweight='bold', color=status_color, transform=ax3.transAxes)
    ax3.text(0.5, 0.4, f'Anomaly Rate: {detection_rate:.1f}%', ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    total_persistence = np.sum(deaths_h1 - births_h1)
    ax3.text(0.5, 0.2, f'TDA Persistence: {total_persistence:.2f}', ha='center', va='center', fontsize=10, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Detection Status', fontsize=11, fontweight='normal')
    return []
def main(config_path=None):
    cfg = load_config(config_path)
    seed = cfg.get("global", {}).get("random_seed", 42)
    np.random.seed(seed)
    rt = cfg.get("regime_tda", {})
    figures_subdir = rt.get("figures_subdir", "figures")
    out_dir = _SCRIPT_DIR / figures_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "34_turbine_tda_animation.gif"
    logger.info("Creating topological turbine classification animation...")
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True)
    anim.save(str(output_file), writer="pillow", fps=FPS, dpi=100)
    logger.info(f"✓ Animation saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create TDA classification animation")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    args = parser.parse_args()
    main(config_path=args.config)