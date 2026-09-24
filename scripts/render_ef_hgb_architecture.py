"""Render the implemented three-view binary EF-HGB detector for the report."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def main():
    output = Path(__file__).resolve().parents[1] / 'docs/figures/ef_hgb_detector_architecture.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10.5))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, 12), ylim=(0, 10.5))
    ax.axis('off')
    ink = '#233343'

    def box(x, y, title, detail='', width=3.55, height=.95, fill='#EDF3F7'):
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
            boxstyle='round,pad=0.03,rounding_size=0.12', linewidth=1.25,
            edgecolor=ink, facecolor=fill, zorder=3))
        ax.text(x, y+(.22 if detail else 0), title, ha='center', va='center',
            fontsize=12, weight='bold', color=ink, zorder=4)
        if detail:
            ax.text(x, y-.22, detail, ha='center', va='center', fontsize=10.5,
                color=ink, linespacing=1.4, zorder=4)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
            linewidth=1.4, color=ink, zorder=2))

    ax.text(6, 10.04, 'Structure of the final EF-HGB detector', ha='center',
        fontsize=18, weight='bold', color=ink)
    box(6, 8.95, 'One station-hour', '67 input columns, arranged into three overlapping views',
        width=6.4, height=1.05)
    box(2, 7.0, 'Full-feature view', 'All observation, rule,\navailability and station features', height=1.4)
    box(6, 7.0, 'Context view', 'Continuous features, data availability,\nelapsed time and station characteristics', height=1.4)
    box(10, 7.0, 'Rule view', 'Rule-based warning indicators\nand station characteristics', height=1.4)
    for x in (2, 6, 10):
        arrow((6, 8.39), (x, 7.75))
        box(x, 5.45, 'HGB classifier', 'Trained on this feature view')
        arrow((x, 6.25), (x, 5.97))
        symbol = {2: 'p_full', 6: 'p_context', 10: 'p_rule'}[x]
        box(x, 4.05, 'Fault prediction', symbol)
        arrow((x, 4.93), (x, 4.57))
    box(6, 2.55, 'Weighted combination',
        'p = w_full × p_full + w_context × p_context + w_rule × p_rule',
        width=9.1, height=1.0, fill='#E3EFEA')
    arrow((2, 3.53), (3.0, 3.09))
    arrow((6, 3.53), (6, 3.09))
    arrow((10, 3.53), (9.0, 3.09))
    box(6, 1.1, 'Apply the selected decision threshold',
        'Output: likely fault or non-fault', width=6.4)
    arrow((6, 2.01), (6, 1.62))
    ax.text(6, .23, 'Weights and threshold are selected using validation data and then held fixed for testing.',
        ha='center', fontsize=10, color=ink)
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    fig.savefig(output, dpi=250, facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
