from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config.paths import FIGURES_DIR, PROJECT_ROOT
from src.model.hourly_baseline import (
    filter_eligible_examples,
    flatten_hourly_features,
    load_hourly_tensor,
    random_split_indices,
)
from src.model.hourly_detection import MASK_MODE_PER_HOUR


DEVELOPMENT_TENSOR_PATH = (
    PROJECT_ROOT / "data" / "hourly_detection" / "one_hour_final" / "hourly_detection_01h.npz"
)
DEVELOPMENT_MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "hourly_detection"
    / "one_hour_final"
    / "models"
    / "evidence_fusion"
    / "selected_ef_hgb_random_01h.joblib"
)
JULY_LEDGER_PATH = (
    PROJECT_ROOT
    / "data"
    / "eval"
    / "one_hour_candidate"
    / "july_ef_hgb_binary_predictions.parquet"
)
CLASS_DISTRIBUTION_PATH = FIGURES_DIR / "selected_ef_hgb_class_distribution.png"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "selected_ef_hgb_confusion_matrices.png"
ROC_PR_PATH = FIGURES_DIR / "selected_ef_hgb_roc_pr_curves.png"


@dataclass(frozen=True)
class BinaryEvaluation:
    name: str
    truth: np.ndarray
    probability: np.ndarray
    prediction: np.ndarray


def _require_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation inputs:\n" + "\n".join(missing))


def require_july_evaluation_inputs() -> None:
    _require_paths(
        {
            "development tensor": DEVELOPMENT_TENSOR_PATH,
            "selected EF-HGB model": DEVELOPMENT_MODEL_PATH,
            "July prediction ledger": JULY_LEDGER_PATH,
        }
    )


def load_selected_detector_evaluations(
    tensor_path: Path = DEVELOPMENT_TENSOR_PATH,
    model_path: Path = DEVELOPMENT_MODEL_PATH,
    july_ledger_path: Path = JULY_LEDGER_PATH,
) -> tuple[BinaryEvaluation, BinaryEvaluation]:
    _require_paths(
        {
            "development tensor": tensor_path,
            "selected EF-HGB model": model_path,
            "July prediction ledger": july_ledger_path,
        }
    )
    examples, _ = filter_eligible_examples(load_hourly_tensor(tensor_path))
    values, _, _ = flatten_hourly_features(examples, mask_mode=MASK_MODE_PER_HOUR)
    truth = np.asarray(examples["y_binary"], dtype=int)
    test_index = np.asarray(random_split_indices(truth, 2026)["test"], dtype=int)
    bundle = joblib.load(model_path)
    estimator = bundle["estimator"] if isinstance(bundle, dict) else bundle
    threshold = float(bundle["config"]["threshold"]) if isinstance(bundle, dict) else 0.30
    development_probability = estimator.predict_proba(values[test_index])[:, 1]
    development = BinaryEvaluation(
        name="Development held-out test",
        truth=truth[test_index],
        probability=development_probability,
        prediction=(development_probability >= threshold).astype(int),
    )
    july_ledger = pd.read_parquet(july_ledger_path)
    july = BinaryEvaluation(
        name="Independent July test",
        truth=july_ledger["truth_fault"].to_numpy(dtype=int),
        probability=july_ledger["random_probability"].to_numpy(dtype=float),
        prediction=july_ledger["random_prediction"].to_numpy(dtype=int),
    )
    return development, july


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_class_distribution_figure(
    evaluations: tuple[BinaryEvaluation, ...],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(evaluations), figsize=(10, 4.6), sharey=False)
    axes_array = np.atleast_1d(axes)
    for axis, evaluation in zip(axes_array, evaluations, strict=True):
        counts = np.bincount(evaluation.truth, minlength=2)
        bars = axis.bar(["Not fault", "Fault"], counts, color=["#2563eb", "#ef4444"])
        axis.set_title(evaluation.name)
        axis.set_ylabel("Station-hours")
        axis.set_ylim(0, float(counts.max()) * 1.18)
        axis.grid(axis="y", alpha=0.2)
        for bar, count in zip(bars, counts, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(count):,}\n({count / counts.sum():.2%})",
                ha="center",
                va="bottom",
            )
    figure.suptitle("Selected EF-HGB class distribution", fontweight="bold")
    figure.tight_layout()
    _save(figure, output_path)


def build_confusion_matrix_figure(
    evaluations: tuple[BinaryEvaluation, ...],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, len(evaluations), figsize=(10, 4.6))
    axes_array = np.atleast_1d(axes)
    for axis, evaluation in zip(axes_array, evaluations, strict=True):
        matrix = confusion_matrix(evaluation.truth, evaluation.prediction)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(evaluation.name)
        axis.set_xticks([0, 1], ["Not fault", "Fault"])
        axis.set_yticks([0, 1], ["Not fault", "Fault"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("Observed class")
        threshold = matrix.max() / 2.0
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{int(matrix[row, column]):,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=12,
                    fontweight="bold",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Selected EF-HGB confusion matrices", fontweight="bold")
    figure.tight_layout()
    _save(figure, output_path)


def build_roc_pr_figure(
    evaluations: tuple[BinaryEvaluation, ...],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ("#2563eb", "#ef4444")
    for color, evaluation in zip(colors, evaluations, strict=True):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            evaluation.truth,
            evaluation.probability,
        )
        precision, recall, _ = precision_recall_curve(
            evaluation.truth,
            evaluation.probability,
        )
        axes[0].plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            linewidth=2,
            label=f"{evaluation.name} · AUROC {roc_auc_score(evaluation.truth, evaluation.probability):.3f}",
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{evaluation.name} · AUPRC {average_precision_score(evaluation.truth, evaluation.probability):.3f}",
        )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="Random ranking")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, loc="lower left")
    figure.suptitle("Selected EF-HGB threshold-independent discrimination", fontweight="bold")
    figure.tight_layout()
    _save(figure, output_path)


def main() -> None:
    evaluations = load_selected_detector_evaluations()
    build_class_distribution_figure(evaluations, CLASS_DISTRIBUTION_PATH)
    build_confusion_matrix_figure(evaluations, CONFUSION_MATRIX_PATH)
    build_roc_pr_figure(evaluations, ROC_PR_PATH)
    for evaluation in evaluations:
        matrix = confusion_matrix(evaluation.truth, evaluation.prediction).ravel().tolist()
        print(
            f"{evaluation.name}: rows={len(evaluation.truth)} fault={int(evaluation.truth.sum())} "
            f"not_fault={int((1 - evaluation.truth).sum())} tn_fp_fn_tp={matrix} "
            f"auroc={roc_auc_score(evaluation.truth, evaluation.probability):.6f} "
            f"auprc={average_precision_score(evaluation.truth, evaluation.probability):.6f}"
        )
    print(f"class_distribution={CLASS_DISTRIBUTION_PATH}")
    print(f"confusion_matrices={CONFUSION_MATRIX_PATH}")
    print(f"roc_pr_curves={ROC_PR_PATH}")


if __name__ == "__main__":
    main()
