"""Generate current report figures from frozen predictions; no model fitting."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score, f1_score
from threadpoolctl import threadpool_limits
from src.workflows.build_july_evaluation_figures import load_selected_detector_evaluations

OUT = ROOT / "outputs/figures/report"
RELEASE = ROOT / "data/eval/final_system_release_20260924"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture_and_comparisons():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set(xlim=(0, 11), ylim=(0, 8))
    ax.axis('off')
    def box(x, y, w, h, text, color='#eaf2f8', edge='#285c85'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12', facecolor=color, edgecolor=edge, linewidth=1.3))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=10)
    def arrow(a, b, dashed=False):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#566776', lw=1.5, linestyle='--' if dashed else '-'))
    box(2.6, 6.6, 5.8, .8, 'Current station-hour: 67 input columns\nEngineered measurements, rule evidence and station context')
    branches = [(0.3, 'Full-view HGB\nAll 67 columns\nWeight = 0.90', False), (4.0, 'Context-view HGB\nMeasurements, availability\nand static context\nWeight = 0.00', True), (7.7, 'Rule-view HGB\nRule evidence and static context\nWeight = 0.10', False)]
    for x, label, inactive in branches:
        box(x, 4.2, 3.0, 1.3, label, '#f2f2f2' if inactive else '#eaf2f8')
        arrow((5.5, 6.48), (x+1.5, 5.62), inactive)
        arrow((x+1.5, 4.08), (5.5, 3.47), inactive)
    box(2.5, 2.4, 6, .95, 'Weighted fault score\np = 0.90 × p(full) + 0.00 × p(context) + 0.10 × p(rules)', '#e8f2ed', '#317963')
    arrow((5.5, 2.28), (5.5, 1.92))
    box(3, .9, 5, .9, 'Decision threshold = 0.50\np ≥ 0.50: fault alert; otherwise: no fault alert')
    ax.text(5.5, .32, 'Weights and threshold selected using temporal validation.\nFeature views overlap; context remains included in the full view.', ha='center', va='center', fontsize=9, color='#465562')
    ax.set_title('Structure of the final EF-HGB binary detector', fontsize=16, pad=12)
    save(fig, 'ef_hgb_architecture.png')

    comparison = pd.read_csv(RELEASE / 'binary_development_tables.csv')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True)
    for ax, split, title in zip(axes, ['random', 'temporal'], ['Standard holdout', 'Temporal holdout']):
        subset = comparison.loc[comparison.split.eq(split)].set_index('model')
        for y, name, color, marker in [(2, 'RGFN', '#566776', 'o'), (1, 'HGB', '#285c85', 's'), (0, 'EF-HGB', '#317963', 'D')]:
            value = 100*subset.loc[name, 'test_f1']
            ax.scatter(value, y, color=color, marker=marker, s=65, zorder=3)
            ax.text(value+.18, y, f'{value:.2f}', va='center', fontsize=10)
        ax.set(title=title, xlabel='Test F1 (%)', yticks=[0, 1, 2], yticklabels=['EF-HGB', 'HGB', 'RGFN'], xlim=(84, 94), ylim=(-.5, 2.5), xticks=[84, 86, 88, 90, 92, 94])
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    save(fig, 'binary_model_comparison.png')

    windows = pd.read_csv(ROOT / 'data/report/hgb_history_window_comparison.csv')
    windows = windows.loc[windows.split_scheme.eq('random')].sort_values('window_hours')
    assert windows.window_hours.tolist() == [1, 3, 5, 7, 12, 24]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for column, label, color in [('validation_f1', 'Validation', '#285c85'), ('test_f1', 'Development test', '#317963')]:
        ax.plot(windows.window_hours, 100*windows[column], label=label, color=color, marker='o')
    ax.set(title='HGB input-window comparison: standard holdout', xlabel='Input window (hours)', ylabel='Fault-class F1 (%)', xticks=windows.window_hours, ylim=(84, 95))
    ax.grid(alpha=.15)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'hgb_input_window_comparison.png')


def forecast_deciles():
    folder = ROOT / 'data/eval/final_forecast_comparison_20260924'
    predictions = pd.read_parquet(folder / 'predictions.parquet')
    chosen = pd.read_csv(folder / 'chosen.csv').set_index('horizon_h')
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for h in (1, 3, 6, 12, 24):
        part = predictions.loc[predictions.horizon_h.eq(h) & predictions.partition.eq('test') & predictions.model.eq(chosen.loc[h, 'model'])].copy()
        part['decile'] = pd.qcut(part.prediction, 10, labels=False, duplicates='drop')
        summary = part.groupby('decile').agg(mean_prediction=('prediction', 'mean'), mean_observed=('actual', 'mean'), rows=('actual', 'size')).reset_index()
        summary['horizon_h'] = h
        assert summary.rows.sum() == len(part)
        rows.append(summary)
        ax.plot(summary.mean_prediction, summary.mean_observed, marker='o', markersize=4, linewidth=1.2, label=f'{h} h')
    ax.plot([0, 100], [0, 100], linestyle='--', color='gray', linewidth=1, label='Equal means')
    ax.set(title='Mean health within prediction deciles', xlabel='Mean predicted health', ylabel='Mean observed health', xlim=(0, 100), ylim=(0, 100))
    ax.set_aspect('equal')
    ax.grid(alpha=.15)
    ax.legend(fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.text(.5, .015, 'Grouped averages; agreement does not measure individual forecast errors.', ha='center', fontsize=8)
    fig.savefig(OUT / 'forecast_prediction_deciles.png', dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(OUT / 'forecast_prediction_deciles.csv', index=False)


def forecasts():
    folder = ROOT / "data/eval/final_forecast_comparison_20260924"
    predictions = pd.read_parquet(folder / "predictions.parquet")
    chosen = pd.read_csv(folder / "chosen.csv").set_index("horizon_h")
    expected = pd.read_csv(folder / "metrics.csv")
    baseline = pd.read_parquet(ROOT / "data/eval/health_forecast/health_forecast_predictions.parquet")
    keys = ["station_id", "hour_utc"]
    rows = []
    band_rows = []
    for h in (1, 3, 6, 12, 24):
        family = chosen.loc[h, "model"]
        current = predictions.loc[(predictions.horizon_h == h) & (predictions.model == family) & (predictions.partition == "test")].copy()
        assert not current.duplicated(keys).any()
        mae = float(np.mean(np.abs(current.actual - current.prediction)))
        reference = expected.loc[(expected.horizon_h == h) & (expected.model == family) & (expected.partition == "test")].iloc[0]
        np.testing.assert_allclose(mae, reference.mae, atol=1e-10)
        row = dict(horizon_h=h, rows=len(current), model=family, forecast_mae=mae)
        actual_band = np.searchsorted([40, 60, 80], current.actual, side="right")
        predicted_band = np.searchsorted([40, 60, 80], current.prediction, side="right")
        band_row = dict(horizon_h=h, accuracy=100*np.mean(actual_band == predicted_band), balanced_accuracy=100*balanced_accuracy_score(actual_band, predicted_band), macro_f1=100*f1_score(actual_band, predicted_band, labels=[0, 1, 2, 3], average="macro", zero_division=0))
        np.testing.assert_allclose(band_row['accuracy'], reference.band_accuracy_pct, atol=1e-8)
        for name, method in [("persistence", "persistence"), ("trend", "recent_trend_24h"), ("roll_forward", "no_new_incident_roll_forward")]:
            part = baseline.loc[(baseline.horizon_h == h) & (baseline.target == "level") & (baseline.method == method)]
            matched = current.merge(part[keys + ["actual", "predicted"]], on=keys, how="left", validate="one_to_one", suffixes=("", "_baseline"), indicator=True)
            assert matched._merge.eq("both").all(), (h, method, "missing baseline rows")
            assert np.isfinite(matched[["actual", "actual_baseline", "predicted"]].to_numpy()).all()
            np.testing.assert_allclose(matched.actual, matched.actual_baseline, atol=1e-9, rtol=0)
            error = float(np.mean(np.abs(matched.actual - matched.predicted)))
            row[f"{name}_mae"] = error
            row[f"gain_vs_{name}_pct"] = 100 * (error - mae) / error
            if name == "persistence":
                band_row['persistence_accuracy'] = 100*np.mean(np.searchsorted([40, 60, 80], matched.actual, side="right") == np.searchsorted([40, 60, 80], matched.predicted, side="right"))
        rows.append(row)
        band_rows.append(band_row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "forecast_gains.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for column, label, color, marker in [
        ("persistence_mae", "Persistence", "#6b7c85", "o"),
        ("trend_mae", "24-hour trend", "#bb8300", "s"),
        ("roll_forward_mae", "No-new-incident roll-forward", "#8763b8", "^"),
        ("forecast_mae", "Selected CatBoost forecast", "#2176a5", "D"),
    ]:
        ax.plot(result.horizon_h, result[column], label=label, color=color, marker=marker, linewidth=1.8, markersize=5)
    ax.set(title="Development forecast error against fixed baselines", xlabel="Forecast horizon (hours)", ylabel="Mean absolute error (health points)", xticks=result.horizon_h, ylim=(0, 12))
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "forecast_baseline_comparison.png")
    bands = pd.DataFrame(band_rows)
    bands.to_csv(OUT / "development_band_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for column, label, color, marker in [("persistence_mae", "Persistence", "#6b7c85", "o"), ("trend_mae", "24-hour trend", "#bb8300", "s"), ("roll_forward_mae", "No-new-incident roll-forward", "#8763b8", "^"), ("forecast_mae", "Selected CatBoost forecast", "#2176a5", "D")]:
        axes[0].plot(result.horizon_h, result[column], label=label, color=color, marker=marker, markersize=4)
    axes[0].set(title="Health-score regression", ylabel="Mean absolute error (health points)", ylim=(0, 12))
    for column, label, style, color in [("accuracy", "Selected: band accuracy", "-", "#2176a5"), ("balanced_accuracy", "Selected: balanced accuracy", "--", "#2176a5"), ("macro_f1", "Selected: macro-F1", "-.", "#317963"), ("persistence_accuracy", "Persistence: band accuracy", "--", "#6b7c85")]:
        axes[1].plot(bands.horizon_h, bands[column], label=label, linestyle=style, color=color, marker="o", markersize=4)
    axes[1].set(title="Bands derived from regression forecasts", ylabel="Score (%)", ylim=(40, 100))
    for ax in axes:
        ax.set(xlabel="Forecast horizon (hours)", xticks=result.horizon_h)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), frameon=False, fontsize=8)
    fig.suptitle("Development performance across forecast horizons")
    save(fig, "development_regression_and_bands.png")
    print(result.to_string(index=False))


def july_forecasts():
    current = pd.read_parquet(RELEASE / "july_forecast_evaluation.parquet")
    baselines = pd.read_parquet(ROOT / "data/eval/july_2026_health_forecast/july_health_forecast_predictions.parquet")
    expected = pd.read_csv(RELEASE / "july_forecast_metrics.csv").set_index("horizon_h")
    rows = []
    for h in (1, 3, 6, 12, 24):
        part = current.loc[current.horizon_h.eq(h)]
        matched = part.merge(baselines.loc[baselines.horizon_h.eq(h)], on=["station_id", "hour_utc", "horizon_h"], how="left", validate="one_to_one", indicator=True)
        assert matched._merge.eq("both").all()
        np.testing.assert_allclose(matched.actual, matched.target_health_total, atol=1e-9, rtol=0)
        row = dict(horizon_h=h, rows=len(part), forecast_mae=float(np.mean(np.abs(matched.actual-matched.prediction))))
        for name, column in [("persistence", "predicted_persistence"), ("roll_forward", "predicted_no_new_incident_roll_forward")]:
            assert np.isfinite(matched[column]).all()
            row[f'{name}_mae'] = float(np.mean(np.abs(matched.actual-matched[column])))
        row['band_accuracy'] = 100*np.mean(np.searchsorted([40, 60, 80], matched.actual, side="right") == np.searchsorted([40, 60, 80], matched.prediction, side="right"))
        np.testing.assert_allclose([row['forecast_mae'], row['band_accuracy']], [expected.loc[h, 'mae'], expected.loc[h, 'band_accuracy_pct']], atol=1e-8)
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'july_forecast_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for column, label, color, marker in [('forecast_mae', 'Selected CatBoost forecast', '#285c85', 'o'), ('persistence_mae', 'Persistence', '#ab2932', 's'), ('roll_forward_mae', 'No-new-incident roll-forward', '#409771', '^')]:
        ax.plot(results.horizon_h, results[column], label=label, color=color, marker=marker)
    ax.set(title='July forecast error against fixed baselines', xlabel='Forecast horizon (hours)', ylabel='Mean absolute error (health points)', xticks=results.horizon_h, ylim=(0, 10))
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'july_forecast_baselines.png')
    dev = pd.read_csv(OUT / 'development_band_metrics.csv')
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(dev.horizon_h, dev.accuracy, label='Development test', color='#285c85', marker='o')
    ax.plot(results.horizon_h, results.band_accuracy, label='July test', color='#317963', marker='s')
    ax.set(title='Health-band accuracy derived from regression forecasts', xlabel='Forecast horizon (hours)', ylabel='Health-band accuracy (%)', xticks=results.horizon_h, ylim=(40, 100))
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    ax.spines[['top', 'right']].set_visible(False)
    save(fig, 'development_july_band_accuracy.png')


def binary():
    development, july = load_selected_detector_evaluations(july_ledger_path=RELEASE / "binary_predictions.parquet")
    evaluations = [(development, "Standard-holdout development test", 0.30), (july, "July: deployed full-development refit", 0.50)]
    expected = [[[18024, 120], [103, 1286]], [[11497, 367], [410, 1291]]]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (evaluation, label, threshold), matrix_expected in zip(axes, evaluations, expected):
        np.testing.assert_array_equal(evaluation.prediction, (evaluation.probability >= threshold).astype(int))
        matrix = confusion_matrix(evaluation.truth, evaluation.prediction, labels=[0, 1])
        np.testing.assert_array_equal(matrix, matrix_expected)
        rows.append(dict(population=label, rows=len(evaluation.truth), threshold=threshold, auroc=roc_auc_score(evaluation.truth, evaluation.probability), average_precision=average_precision_score(evaluation.truth, evaluation.probability), tn=int(matrix[0, 0]), fp=int(matrix[0, 1]), fn=int(matrix[1, 0]), tp=int(matrix[1, 1])))
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
        ax.set(title=f"{label}\nThreshold {threshold:.2f}", xticks=[0, 1], yticks=[0, 1], xticklabels=["Not fault", "Fault"], yticklabels=["Not fault", "Fault"], xlabel="Predicted class", ylabel="Reference class")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=17, color="white" if value > matrix.max()/2 else "black")
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, "binary_confusion_matrices.png")
    pd.DataFrame(rows).to_csv(OUT / "binary_discrimination_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for (evaluation, _, _), row, label, color in zip(evaluations, rows, ["Standard-holdout test", "July deployed refit"], ["#285c85", "#317963"]):
        fpr, tpr, _ = roc_curve(evaluation.truth, evaluation.probability)
        precision, recall, _ = precision_recall_curve(evaluation.truth, evaluation.probability)
        axes[0].plot(fpr, tpr, color=color, label=f"{label}: {row['auroc']:.4f}")
        axes[1].plot(recall, precision, color=color, label=f"{label}: AP {row['average_precision']:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=.8)
    axes[0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[1].set(title="Precision–recall curve", xlabel="Recall", ylabel="Precision")
    for ax in axes:
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.15)
        ax.legend(loc="lower left", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "binary_roc_pr_curves.png")
    print(json.dumps(rows, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        from src.workflows.draw_rgfn import main as draw_rgfn
        draw_rgfn()
        architecture_and_comparisons()
        forecast_deciles()
        forecasts()
        july_forecasts()
        binary()


if __name__ == "__main__":
    main()
