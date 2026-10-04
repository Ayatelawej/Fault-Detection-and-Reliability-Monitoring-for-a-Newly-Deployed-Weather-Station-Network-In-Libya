"""Render the verified one-hour binary RGFN architecture for the report."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    output = Path(__file__).resolve().parents[2] / 'outputs/figures/report/rgfn_architecture.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 11))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 12), ylim=(0, 11))
    ax.axis('off')
    ink = '#233343'

    def box(x, y, title, detail='', width=3.5, height=.95, fill='#EDF3F7'):
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
            boxstyle='round,pad=0.03,rounding_size=0.12', linewidth=1.25,
            edgecolor=ink, facecolor=fill, zorder=3))
        ax.text(x, y+(.17 if detail else 0), title, ha='center', va='center',
            fontsize=12, weight='bold', color=ink, zorder=4)
        if detail:
            ax.text(x, y-.19, detail, ha='center', va='center', fontsize=10.5,
                color=ink, linespacing=1.35, zorder=4)

    def arrow(start, end, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
            linewidth=1.4, color=ink, linestyle='--' if dashed else '-', zorder=2))

    ax.text(6, 10.65, 'RGFN: one-hour fault detection', ha='center', fontsize=18,
        weight='bold', color=ink)
    box(2, 9.5, 'Observation-based inputs',
        'Continuous features, missing-data\nindicator and time since last observation', height=1.15)
    box(10, 9.5, 'Rule-based inputs', 'Fault-warning indicators', height=1.15)
    box(2, 7.8, 'Sensor branch', 'Feed-forward neural network')
    box(10, 7.8, 'Rule branch', 'Feed-forward neural network')
    box(6, 9.5, 'Station characteristics', 'Static context', width=3.3)
    box(2, 5.6, 'Sensor fault score')
    box(10, 5.6, 'Rule fault score')
    box(6, 5.6, 'Learned gate', 'Chooses the branch weights', width=3.3, fill='#E3EFEA')
    box(6, 3.8, 'Combine the scores',
        'α × sensor score + (1 − α) × rule score', width=5.2)
    box(6, 2.5, 'Sigmoid', 'Converts the combined score to a value from 0 to 1', width=5.2)
    box(6, 1.2, 'Apply the selected threshold', 'Output: likely fault or non-fault', width=5.2)
    for x in (2, 10):
        arrow((x, 8.9), (x, 8.31))
        arrow((x, 7.29), (x, 6.12))
    arrow((3.3, 7.29), (4.7, 6.12), True)
    arrow((8.7, 7.29), (7.3, 6.12), True)
    ax.text(6, 7.1, 'Learned features from both branches', ha='center', fontsize=10,
        color=ink, bbox=dict(facecolor='white', edgecolor='none', pad=2), zorder=5)
    arrow((6, 8.99), (6, 6.12), True)
    arrow((2, 5.08), (3.36, 3.8))
    arrow((10, 5.08), (8.64, 3.8))
    arrow((6, 5.08), (6, 4.31))
    ax.text(6.16, 4.7, 'α and 1 − α', fontsize=10, va='center', color=ink)
    arrow((6, 3.28), (6, 3.01))
    arrow((6, 1.98), (6, 1.71))
    ax.text(6, .32, 'Branch weights are learned during training; the decision threshold is selected using validation data.',
        ha='center', fontsize=10, color=ink)
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    fig.savefig(output, dpi=250, facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
