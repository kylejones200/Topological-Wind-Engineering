import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
np.random.seed(42)
FPS, N_FRAMES = (10, 100)
n_samples, time = (400, np.arange(400))
regime = np.zeros(n_samples)
for i in range(5):
    start = i * 80
    regime[start:start + 40] = 1
    regime[start + 40:start + 80] = 0
power = np.where(regime, 2000, 800) + np.random.normal(0, 100, n_samples)
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
    ax1.plot(time[:end_idx], power[:end_idx], 'black', linewidth=2)
    ax1.fill_between(time[:end_idx], 0, 2500, where=regime[:end_idx] == 1, alpha=0.1, color='red', label='High Power Regime')
    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Power (kW)', fontsize=10)
    ax1.set_title(f'Regime Transitions - Frame {frame + 1}/{N_FRAMES}', fontsize=11, fontweight='normal')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    if end_idx > 50:
        transitions = np.diff(regime[:end_idx].astype(int))
        n_transitions = np.sum(np.abs(transitions))
        ax2.bar(['Transitions'], [n_transitions], color='gray', edgecolor='black')
        ax2.set_ylabel('Count', fontsize=10)
        ax2.set_title('Detected Transitions', fontsize=11, fontweight='normal')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
    current_regime = 'High Power' if regime[min(end_idx - 1, n_samples - 1)] else 'Low Power'
    color = 'red' if current_regime == 'High Power' else 'gray'
    ax3.text(0.5, 0.5, current_regime, ha='center', va='center', fontsize=16, fontweight='bold', color=color, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Current Regime', fontsize=11, fontweight='normal')
    return []
logger.info('Creating animation for Article 36...')
anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True)
anim.save('36_regime_transitions_animation.gif', writer='pillow', fps=FPS, dpi=100)
logger.info('✓ Animation saved: 36_regime_transitions_animation.gif')
plt.close()