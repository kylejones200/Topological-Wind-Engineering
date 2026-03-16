import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
np.random.seed(42)
FPS, N_FRAMES = (10, 100)
n_turbines = 25
x_pos = np.repeat(np.arange(5), 5)
y_pos = np.tile(np.arange(5), 5)
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
    wind_dir_angle = frame / N_FRAMES * 2 * np.pi
    wind_x, wind_y = (np.cos(wind_dir_angle), np.sin(wind_dir_angle))
    power = np.ones(n_turbines) * 2000
    for i in range(n_turbines):
        for j in range(n_turbines):
            if i != j:
                dx, dy = (x_pos[j] - x_pos[i], y_pos[j] - y_pos[i])
                dist = np.sqrt(dx ** 2 + dy ** 2)
                alignment = (dx * wind_x + dy * wind_y) / (dist + 0.1)
                if alignment > 0.7 and dist < 3:
                    power[j] *= 0.7
    scatter = ax1.scatter(x_pos, y_pos, c=power, s=300, cmap='RdYlGn', vmin=1000, vmax=2000, edgecolors='black', linewidths=2)
    ax1.arrow(2, -0.5, wind_x, wind_y, head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.7)
    ax1.text(2, -1, 'Wind', ha='center', fontsize=10, color='blue')
    ax1.set_xlabel('X Position', fontsize=10)
    ax1.set_ylabel('Y Position', fontsize=10)
    ax1.set_title(f'Wind Farm Power Output - Frame {frame + 1}/{N_FRAMES}', fontsize=11, fontweight='normal')
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-2, 5)
    ax1.set_aspect('equal')
    plt.colorbar(scatter, ax=ax1, label='Power (kW)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.hist(power, bins=15, color='gray', alpha=0.7, edgecolor='black')
    ax2.axvline(power.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {power.mean():.0f} kW')
    ax2.set_xlabel('Power Output (kW)', fontsize=10)
    ax2.set_ylabel('Turbine Count', fontsize=10)
    ax2.set_title('Power Distribution', fontsize=11, fontweight='normal')
    ax2.legend(fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    total_power = power.sum()
    efficiency = total_power / (n_turbines * 2000) * 100
    ax3.text(0.5, 0.7, f'{efficiency:.1f}% Efficient', ha='center', va='center', fontsize=16, fontweight='bold', transform=ax3.transAxes)
    ax3.text(0.5, 0.4, f'Total: {total_power / 1000:.1f} MW', ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.5, 0.2, f'{np.sum(power < 1500)} turbines in wake', ha='center', va='center', fontsize=10, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Farm Status', fontsize=11, fontweight='normal')
    return []
logger.info('Creating animation for Article 40...')
anim = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000 / FPS, blit=True, repeat=True)
anim.save('40_farm_coordination_animation.gif', writer='pillow', fps=FPS, dpi=100)
logger.info('✓ Animation saved: 40_farm_coordination_animation.gif')
plt.close()