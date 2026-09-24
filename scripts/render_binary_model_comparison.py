"""Build a report figure from the saved, validation-selected model comparisons."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    development = pd.read_csv(root / 'data/report/one_hour_detection_model_comparison.csv')
    blocked = pd.read_csv(root / 'data/eval/blocked_model_comparison_20260916_v2/comparison.csv')
    blocked['split'] = 'blocked'
    data = pd.concat([development, blocked], ignore_index=True)
    models = ['RGFN', 'HGB', 'EF-HGB']
    colors = {'RGFN': '#566575', 'HGB': '#295A7B', 'EF-HGB': '#32715C'}
    markers = {'RGFN': 'o', 'HGB': 's', 'EF-HGB': 'D'}
    panels = [('random', 'Random split'), ('spaced', 'Spaced split'),
              ('blocked', 'Temporal holdout\nMarch–April')]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), sharex=True, sharey=True)
    fig.patch.set_facecolor('white')
    for ax, (split, title) in zip(axes, panels):
        subset = data[data['split'].eq(split)].set_index('model')
        assert len(subset) == 3 and not subset.index.duplicated().any()
        assert subset.loc['EF-HGB', 'selected_by_validation']
        for y, model in zip([2, 1, 0], models):
            value = float(subset.loc[model, 'test_f1']) * 100
            assert 84 <= value <= 94
            ax.scatter(value, y, s=85, color=colors[model], marker=markers[model], zorder=3)
            ax.annotate(f'{value:.2f}', (value, y), xytext=(9, 0), textcoords='offset points',
                va='center', fontsize=11, color='#233343')
        ax.set_title(title, fontsize=13, pad=15)
        ax.set_xlim(84, 94)
        ax.set_ylim(-.55, 2.55)
        ax.set_xticks([84, 86, 88, 90, 92, 94])
        ax.set_yticks([2, 1, 0], ['RGFN', 'HGB', 'EF-HGB*'])
        ax.tick_params(axis='both', labelsize=10.5)
        ax.tick_params(axis='y', length=0, labelleft=True, pad=10)
        ax.grid(axis='x', color='#E0E4E8', linewidth=.7)
        ax.set_axisbelow(True)
        for name in ('top', 'right', 'left'):
            ax.spines[name].set_visible(False)
        ax.spines['bottom'].set_color('#87939D')
        ax.set_xlabel('Test F1 (%)', fontsize=11, labelpad=10)
    fig.suptitle('Binary fault detection across three data splits', fontsize=16, y=.97)
    fig.text(.5, .09, '* EF-HGB was selected using validation precision, recall and F1, not test scores.',
        ha='center', fontsize=10, color='#233343')
    fig.text(.5, .035, 'All panels use the same enlarged scale (84–94%). Test populations differ between splits.',
        ha='center', fontsize=10, color='#233343')
    fig.subplots_adjust(left=.09, right=.98, bottom=.27, top=.73, wspace=.45)
    output = root / 'docs/figures/binary_model_comparison_three_splits.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor='white')
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
