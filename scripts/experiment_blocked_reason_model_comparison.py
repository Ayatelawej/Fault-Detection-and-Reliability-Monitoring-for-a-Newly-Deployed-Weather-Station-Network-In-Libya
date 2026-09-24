"""Compare redesigned current-hour HGB, RGFN, and EF-HGB reason classifiers."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import json
import os
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from scripts.experiment_blocked_fault_detection import blocked_split, validate_split
from scripts.experiment_blocked_reason_codes import training_rows
from src.model.final_reason_codes import FREEZE, apply_output_policy, build_features, load_observations
from src.model.hourly_baseline import _fault_groups, load_hourly_tensor
from src.model.hourly_rgfn_training import (
    HourlyReasonCodeRgfnConfig,
    _fit_reason_code_rgfn_split,
    _reason_code_rgfn_test_partition,
    predict_reason_code_rgfn,
)
from src.model.reason_code_rebuild import (
    COMP,
    MECH,
    SEED,
    event_weights,
    feature_views,
    fit_estimator,
    multilabel_rows,
    probability,
)

THRESHOLDS = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)


def select_single_threshold(y, probability_values, weights, folds):
    candidates = []
    for threshold in THRESHOLDS:
        pred = probability_values >= threshold
        scores = []
        for fold in np.unique(folds):
            mask = folds == fold
            yy, pp, ww = y[mask], pred[mask], weights[mask]
            tp = ww[(yy == 1) & pp].sum()
            fp = ww[(yy == 0) & pp].sum()
            fn = ww[(yy == 1) & ~pp].sum()
            scores.append(2 * tp / max(2 * tp + fp + fn, 1e-12))
        candidates.append((float(np.mean(scores)), -threshold, threshold))
    chosen = max(candidates)
    return float(chosen[2]), float(chosen[0])


def policy_summaries(method, partition, scores_by_axis, thresholds_by_axis, targets_by_axis, groups, policies):
    summaries = []
    for axis, labels in (("mechanism", MECH), ("component", COMP)):
        scores = scores_by_axis[axis]
        thresholds = thresholds_by_axis[axis]
        y = targets_by_axis[axis]
        resolved = y.any(axis=1)
        for policy in policies:
            mode = ("minimum_one" if axis == "mechanism" else "threshold") if policy == "mixed" else policy
            pred = apply_output_policy(scores, thresholds, np.ones(len(y), dtype=bool), mode)
            _, summary = multilabel_rows(
                y[resolved], pred[resolved], groups[resolved], labels,
                {"method": method, "partition": partition, "axis": axis, "policy": policy},
            )
            summaries.append(summary)
    return summaries


def score_hgb_heads(heads, x, indices):
    result = {}
    thresholds = {}
    for axis, labels in (("mechanism", MECH), ("component", COMP)):
        scores = np.full((len(indices), len(labels)), np.nan, dtype=float)
        limits = np.full(len(labels), np.inf, dtype=float)
        for j, label in enumerate(labels):
            head = heads[(axis, label)]
            scores[:, j] = probability(head["model"], x[indices][:, head["view"]])
            limits[j] = head["threshold"]
        result[axis], thresholds[axis] = scores, limits
    return result, thresholds


def score_ef_heads(directory, x, indices):
    result, thresholds = {}, {}
    for axis, labels in (("mechanism", MECH), ("component", COMP)):
        scores = np.full((len(indices), len(labels)), np.nan, dtype=float)
        limits = np.full(len(labels), np.inf, dtype=float)
        for j, label in enumerate(labels):
            head = joblib.load(directory / "models" / f"{axis}_{label}.joblib")
            branch = np.column_stack(
                [probability(model, x[indices][:, columns]) for model, columns in zip(head["models"], head["views"])]
            )
            scores[:, j] = branch @ head["weights"]
            limits[j] = head["threshold"]
        result[axis], thresholds[axis] = scores, limits
    return result, thresholds


def run(out: Path) -> None:
    if out.exists():
        raise FileExistsError(out)
    tensor_path = ROOT / "data/hourly_detection/one_hour_final/hourly_detection_01h.npz"
    labels_path = ROOT / "data/eval/reason_code_rebuild_20260912_v2/aligned_reference.parquet"
    ef_directory = ROOT / "data/eval/blocked_reason_codes_20260914"
    z = load_hourly_tensor(tensor_path)
    hours = pd.to_datetime(z["hour"], utc=True)
    index = pd.MultiIndex.from_arrays([z["station_id"], hours], names=["station_id", "hour"])
    fault = np.asarray(z["y_binary"], dtype=int)
    groups = np.asarray([f"normal:{s}:{str(t)[:10]}" for s, t in index], dtype=object)
    for key, indices in _fault_groups(fault, z["source_episode_ids"]).items():
        groups[indices] = "event:" + key
    splits = blocked_split(hours, groups)
    validate_split(hours, groups, splits)

    reference = pd.read_parquet(labels_path)
    reference["hour"] = pd.to_datetime(reference["hour"], utc=True)
    reference = reference.set_index(["station_id", "hour"]).reindex(index)
    y_all = reference[MECH + COMP].to_numpy(dtype=int)
    y_axes = {"mechanism": y_all[:, : len(MECH)], "component": y_all[:, len(MECH):]}

    features, _ = build_features(
        load_observations(
            ROOT / "data/merged/station_hourly_merged.csv",
            ROOT / "data/features/external_residuals.parquet",
            FREEZE,
        )
    )
    x = features.reindex(index).to_numpy(dtype="float32")
    names = list(features.columns)
    del features
    out.mkdir(parents=True)
    (out / "models/hgb").mkdir(parents=True)

    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    # Plain HGB uses the full reason-specific view only. Thresholds are selected
    # from grouped OOF training predictions, matching the EF-HGB protocol.
    hgb_heads = {}
    hgb_selection = []
    for axis, labels in (("mechanism", MECH), ("component", COMP)):
        yy = y_axes[axis]
        fitrows = training_rows(train, fault, yy.any(axis=1))
        for j, label in enumerate(labels):
            target = yy[:, j]
            views = feature_views(names, axis, label)
            full = views[0]
            folds = np.zeros(len(fitrows), dtype=int)
            oof = np.zeros(len(fitrows), dtype=float)
            for fold, (fit_pos, oof_pos) in enumerate(GroupKFold(3).split(fitrows, groups=groups[fitrows])):
                folds[oof_pos] = fold
                model = fit_estimator(x[fitrows[fit_pos]][:, full], target[fitrows[fit_pos]], groups[fitrows[fit_pos]])
                oof[oof_pos] = probability(model, x[fitrows[oof_pos]][:, full])
            threshold, oof_f1 = select_single_threshold(target[fitrows], oof, event_weights(groups[fitrows]), folds)
            model = fit_estimator(x[fitrows][:, full], target[fitrows], groups[fitrows])
            head = {"model": model, "view": full, "threshold": threshold}
            hgb_heads[(axis, label)] = head
            joblib.dump(head, out / "models/hgb" / f"{axis}_{label}.joblib")
            hgb_selection.append({"axis": axis, "label": label, "threshold": threshold, "oof_event_f1": oof_f1})
            print(f"HGB {axis}/{label} threshold={threshold:.2f}", flush=True)
    pd.DataFrame(hgb_selection).to_csv(out / "hgb_selection.csv", index=False)

    policies = ("threshold", "minimum_one", "mixed")
    summaries = []
    score_cache = {}
    for partition, indices in (("validation", validation), ("test", test)):
        axis_targets = {axis: values[indices] for axis, values in y_axes.items()}
        axis_groups = groups[indices]
        hgb_scores, hgb_thresholds = score_hgb_heads(hgb_heads, x, indices)
        ef_scores, ef_thresholds = score_ef_heads(ef_directory, x, indices)
        summaries.extend(policy_summaries("HGB", partition, hgb_scores, hgb_thresholds, axis_targets, axis_groups, policies))
        summaries.extend(policy_summaries("EF-HGB", partition, ef_scores, ef_thresholds, axis_targets, axis_groups, policies))
        score_cache[("HGB", partition)] = (hgb_scores, hgb_thresholds)
        score_cache[("EF-HGB", partition)] = (ef_scores, ef_thresholds)

    # RGFN receives the redesigned current-hour targets, not the retired
    # episode-broadcast targets. Keep only normal or fully resolved rows so an
    # unknown reason is never silently treated as a negative label.
    fully_known = (fault == 0) | (y_axes["mechanism"].any(axis=1) & y_axes["component"].any(axis=1))
    train_fit = training_rows(train, fault, fully_known & (fault == 1))
    validation_fit = validation[fully_known[validation]]
    test_fit = test[fully_known[test]]
    retained = np.concatenate([train_fit, validation_fit, test_fit])
    remap = np.full(len(fault), -1, dtype=np.int64)
    remap[retained] = np.arange(len(retained), dtype=np.int64)
    rgfn_examples = {
        key: (np.asarray(value)[retained] if np.asarray(value).ndim and len(np.asarray(value)) == len(fault) else value)
        for key, value in z.items()
    }
    rgfn_examples["y_mechanism"] = y_axes["mechanism"][retained]
    rgfn_examples["y_component"] = y_axes["component"][retained]
    rgfn_examples["mechanism_label_names"] = np.asarray(MECH, dtype=object)
    rgfn_examples["component_label_names"] = np.asarray(COMP, dtype=object)
    rgfn_splits = {
        "train": remap[train_fit],
        "validation": remap[validation_fit],
        "test": remap[test_fit],
    }
    saved = torch.load(
        ROOT / "data/hourly_detection/models/reason_codes/rgfn/hourly_reason_codes_random.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = HourlyReasonCodeRgfnConfig(**saved["configuration"])
    fitted, status = _fit_reason_code_rgfn_split(
        rgfn_examples,
        rgfn_splits,
        "blocked",
        config,
        time.monotonic() + 3600.0,
    )
    if fitted is None:
        raise RuntimeError(f"RGFN did not complete: {status}")
    (out / "models/rgfn").mkdir()
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in fitted.model.state_dict().items()},
            "configuration": asdict(config),
            "thresholds": fitted.thresholds,
            "label_names": fitted.label_names,
            "training": fitted.training,
        },
        out / "models/rgfn/reason_codes_blocked.pt",
    )
    for partition, indices, local_indices in (
        ("validation", validation_fit, rgfn_splits["validation"]),
        ("test", test_fit, rgfn_splits["test"]),
    ):
        tensor_partition = _reason_code_rgfn_test_partition(rgfn_examples, fitted, local_indices)
        probabilities = predict_reason_code_rgfn(fitted.model, tensor_partition)
        rgfn_scores = {
            "mechanism": probabilities[:, : len(MECH)],
            "component": probabilities[:, len(MECH):],
        }
        rgfn_thresholds = {
            "mechanism": fitted.thresholds[: len(MECH)],
            "component": fitted.thresholds[len(MECH):],
        }
        axis_targets = {axis: values[indices] for axis, values in y_axes.items()}
        summaries.extend(
            policy_summaries("RGFN", partition, rgfn_scores, rgfn_thresholds, axis_targets, groups[indices], policies)
        )

    frame = pd.DataFrame(summaries)
    frame.to_csv(out / "summary.csv", index=False)
    primary = frame.loc[frame.policy.eq("mixed")].copy()
    validation_primary = primary.loc[primary.partition.eq("validation")]
    selected = (
        validation_primary.groupby("method", as_index=False).macro_f1.mean()
        .sort_values(["macro_f1", "method"], ascending=[False, True])
        .iloc[0]["method"]
    )
    report = {
        "design": {
            "targets": "redesigned current-hour reason labels",
            "threshold_selection": "grouped OOF within training only",
            "model_selection": "February validation mixed-policy mean mechanism/component macro-F1",
            "test": "March-April comparison only",
            "unknown_reason_faults": "excluded, not converted to negatives",
        },
        "selected_by_validation": selected,
        "summary": frame.to_dict("records"),
        "rgfn_training": status,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)), encoding="utf-8")
    print(frame.to_string(index=False), flush=True)
    print(f"selected_by_validation={selected}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/eval/blocked_reason_model_comparison_20260916")
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
