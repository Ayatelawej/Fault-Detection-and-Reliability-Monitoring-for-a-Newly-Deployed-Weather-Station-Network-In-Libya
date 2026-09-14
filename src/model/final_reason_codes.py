"""Frozen, current-hour EF-HGB reason heads behind the selected binary detector.

Training references are weak/evidence-derived. Inference never reads labels or
episode IDs. Scores are model scores, not calibrated diagnostic probabilities.
"""
from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from src.model.reason_code_rebuild import (
    ROOT, MECH, COMP, SEED, aligned_labels, event_weights, feature_views,
    features_for_station, fit_estimator, probability, select_policy, sha,
)
from src.model.hourly_baseline import load_hourly_tensor, _fault_groups

VERSION = "current-hour-ef-hgb-v1"
OUTPUT_POLICY_VERSION = "mixed-reason-output-v2"
LEGACY_THRESHOLD_POLICY = {"mechanism": "threshold", "component": "threshold"}
MIXED_OUTPUT_POLICY = {"mechanism": "minimum_one", "component": "threshold"}
FREEZE = pd.Timestamp("2026-06-30T23:00:00Z")
MODEL_DIR = ROOT / "data/model/reason_codes/final"
JULY_OUTPUT = ROOT / "data/eval/july_2026_reason_codes"
JULY_MIXED_OUTPUT = ROOT / "data/eval/july_2026_reason_codes_mixed_v2"
JULY_RAW = ROOT / "data/staging/station_hourly_combined_through_july.csv"
JULY_REFS = ROOT / "data/eval/july_2026_features/external_residuals.parquet"
JULY_GATE = ROOT / "data/eval/one_hour_candidate/july_ef_hgb_binary_predictions.parquet"
REF_GROUPS = dict(pressure="barometer", temp="thermo_hygrometer",
                  dewpoint="thermo_hygrometer", wind="anemometer", solar="light_uv")


def load_observations(raw_path, reference_path, cutoff):
    raw = pd.read_csv(raw_path)
    raw["hour_utc"] = pd.to_datetime(raw.hour_utc, utc=True)
    raw = raw.loc[raw.hour_utc.le(cutoff)].copy()
    refs = pd.read_parquet(reference_path, columns=["station_id", "time_utc"] +
                           ["r_" + k for k in REF_GROUPS])
    refs = refs.rename(columns={"time_utc": "hour_utc", **{
        "r_" + k: f"reference__{v}__{k}" for k, v in REF_GROUPS.items()}})
    refs["hour_utc"] = pd.to_datetime(refs.hour_utc, utc=True)
    if raw.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("Duplicate raw station-hours")
    return raw.merge(refs, on=["station_id", "hour_utc"], how="left", validate="one_to_one")


def build_features(raw):
    frames, evidence = [], {}
    for station, rows in raw.groupby("station_id", sort=True):
        frame, ev = features_for_station(rows.set_index("hour_utc"))
        frame["station_id"] = station
        frame["hour"] = frame.index
        frames.append(frame.set_index(["station_id", "hour"]))
        evidence[station] = ev
    if not frames:
        raise ValueError("No observations available")
    return pd.concat(frames).sort_index(), evidence


def train(model_dir=MODEL_DIR):
    """Refit June development data; select policy with grouped OOF, never July."""
    model_dir = Path(model_dir)
    if model_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen release: {model_dir}")
    inputs = dict(raw=ROOT / "data/merged/station_hourly_merged.csv",
                  references=ROOT / "data/features/external_residuals.parquet",
                  episodes=ROOT / "data/labels/episode_labels.csv",
                  statistical=ROOT / "data/labels/statistical_anomaly_review.csv",
                  tensor=ROOT / "data/hourly_detection/one_hour_final/hourly_detection_01h.npz")
    hashes = {key: sha(path) for key, path in inputs.items()}
    raw = load_observations(inputs["raw"], inputs["references"], FREEZE)
    features, evidence = build_features(raw)
    z = load_hourly_tensor(inputs["tensor"])
    hours = pd.to_datetime(z["hour"], utc=True)
    if hours.max() > FREEZE:
        raise ValueError("Training tensor extends beyond the June freeze")
    index = pd.MultiIndex.from_arrays([z["station_id"], hours], names=["station_id", "hour"])
    if not index.isin(features.index).all():
        raise ValueError("Training rows lack observation history")
    names = features.columns.tolist()
    x = features.reindex(index).to_numpy("float32")
    ep = pd.read_csv(inputs["episodes"])
    for c in ("start_hour", "end_hour"):
        ep[c] = pd.to_datetime(ep[c], utc=True)
    if ep.end_hour.max() > FREEZE:
        raise ValueError("Training episodes extend beyond June")
    stat = pd.read_csv(inputs["statistical"])
    stat["hour_utc"] = pd.to_datetime(stat.hour_utc, utc=True)
    stat = stat.loc[stat.hour_utc.le(FREEZE)]
    y = aligned_labels(raw, evidence, ep, stat, index)
    fault = np.asarray(z["y_binary"], int)
    assert not y[fault == 0].any()
    groups = np.array([f"normal:{s}:{t:%Y-%m-%d}" for s, t in index], dtype=object)
    for key, ii in _fault_groups(fault, z["source_episode_ids"]).items():
        groups[ii] = "event:" + key
    heads, selections = {}, []
    with threadpool_limits(limits=2):
        for axis, labels, offset in (("mechanism", MECH, 0), ("component", COMP, len(MECH))):
            resolved = y[:, offset:offset + len(labels)].any(axis=1)
            positive = np.flatnonzero((fault == 1) & resolved)
            negative = np.flatnonzero(fault == 0)
            if len(negative) > 16000:
                negative = np.sort(np.random.default_rng(SEED).choice(negative, 16000, replace=False))
            fitrows = np.sort(np.r_[positive, negative])
            for j, label in enumerate(labels):
                target = y[:, offset + j]
                events = len(np.unique(groups[fitrows[target[fitrows] == 1]]))
                if events < 3:
                    selections.append(dict(axis=axis, label=label, events=events, status="unsupported"))
                    continue
                views = feature_views(names, axis, label)
                oof = np.zeros((len(fitrows), 3))
                foldids = np.zeros(len(fitrows), int)
                for fold, (fi, oi) in enumerate(GroupKFold(3).split(fitrows, groups=groups[fitrows])):
                    assert not set(groups[fitrows[fi]]) & set(groups[fitrows[oi]])
                    foldids[oi] = fold
                    for branch, cols in enumerate(views):
                        m = fit_estimator(x[fitrows[fi]][:, cols], target[fitrows[fi]], groups[fitrows[fi]])
                        oof[oi, branch] = probability(m, x[fitrows[oi]][:, cols])
                mix, threshold, cv = select_policy(target[fitrows], oof, event_weights(groups[fitrows]), foldids)
                models = [fit_estimator(x[fitrows][:, cols], target[fitrows], groups[fitrows]) for cols in views]
                heads[f"{axis}:{label}"] = dict(models=models, views=views, weights=mix, threshold=threshold)
                selections.append(dict(axis=axis, label=label, events=events, status="fitted",
                    threshold=threshold, full_weight=mix[0], context_weight=mix[1], rules_weight=mix[2],
                    selection_oof_event_f1=cv))
                print(f"Frozen head {axis}/{label}: {events} training events", flush=True)
    assert hashes == {key: sha(path) for key, path in inputs.items()}
    manifest = dict(version=VERSION, trained_through=str(hours.max()), seed=SEED, threads=2,
        input_hashes=hashes, feature_count=len(names), training_hours=len(index),
        training_fault_hours=int(fault.sum()), resolved_fault_hours=int(y.any(axis=1).sum()),
        policy="per-code grouped-OOF threshold; multilabel; abstain if none clears threshold",
        selection="3-fold event-disjoint OOF on June development; refit all development; not held-out accuracy",
        interpretation="Likely current-hour patterns/components, not verified hardware diagnoses",
        external_reference="Archived same-hour residuals; real-time arrival availability remains unverified",
        status="frozen historical-replay release; July excluded from fitting and selection")
    model_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(dict(version=VERSION, feature_names=names, heads=heads, manifest=manifest), model_dir / "reason_heads.joblib")
    manifest["model_sha256"] = sha(model_dir / "reason_heads.joblib")
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(selections).to_csv(model_dir / "selection.csv", index=False)
    print(f"Saved frozen release: {model_dir}", flush=True)


def apply_output_policy(scores, thresholds, gate, mode="threshold"):
    """Convert saved per-code scores to labels without changing the binary gate.

    ``threshold`` retains the historical per-code behavior. ``minimum_one`` first
    retains every threshold-clearing code, then assigns the highest finite score
    when an alerted row would otherwise have no code.
    """
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    gate = np.asarray(gate, dtype=bool)
    if scores.ndim != 2 or thresholds.shape != (scores.shape[1],) or gate.shape != (scores.shape[0],):
        raise ValueError("Output-policy inputs have incompatible shapes")
    if mode not in {"threshold", "minimum_one"}:
        raise ValueError(f"Unsupported reason output policy: {mode}")
    result = (scores >= thresholds) & gate[:, None]
    if mode == "minimum_one":
        safe = np.where(np.isfinite(scores), scores, -np.inf)
        best = safe.max(axis=1)
        fill = gate & ~result.any(axis=1) & np.isfinite(best)
        result[np.flatnonzero(fill), safe[fill].argmax(axis=1)] = True
    return result


def predict(features, detections, bundle, output_policy=None):
    """Only operational gate columns are consumed; each row uses its own hour."""
    if bundle["version"] != VERSION:
        raise ValueError("Unsupported reason-code model version")
    if set(features.columns) != set(bundle["feature_names"]):
        raise ValueError("Reason-code feature schema mismatch")
    out = detections[["station_id", "hour_utc", "random_probability", "random_prediction"]].copy()
    out["hour_utc"] = pd.to_datetime(out.hour_utc, utc=True)
    if out.duplicated(["station_id", "hour_utc"]).any():
        raise ValueError("Duplicate detection station-hours")
    if out.empty or not out.random_prediction.isin([0, 1]).all():
        raise ValueError("Expected a nonempty binary detection ledger")
    index = pd.MultiIndex.from_arrays([out.station_id, out.hour_utc], names=["station_id", "hour"])
    if not index.isin(features.index).all():
        raise ValueError("Detection rows lack observation history")
    x = features.reindex(index)[bundle["feature_names"]].to_numpy("float32")
    gate = out.random_prediction.eq(1).to_numpy()
    policy = dict(LEGACY_THRESHOLD_POLICY if output_policy is None else output_policy)
    if set(policy) != {"mechanism", "component"}:
        raise ValueError("Output policy must define mechanism and component")
    with threadpool_limits(limits=2):
        for axis, labels in (("mechanism", MECH), ("component", COMP)):
            axis_scores, thresholds = [], []
            for j, label in enumerate(labels):
                scores = np.full(len(out), np.nan)
                head = bundle["heads"].get(f"{axis}:{label}")
                if head is not None and gate.any():
                    scores[gate] = np.column_stack([probability(m, x[gate][:, cols])
                        for m, cols in zip(head["models"], head["views"], strict=True)]) @ head["weights"]
                out[f"{axis}_score__{label}"] = scores
                axis_scores.append(scores)
                thresholds.append(np.inf if head is None else head["threshold"])
            predictions = apply_output_policy(
                np.column_stack(axis_scores), np.asarray(thresholds), gate, policy[axis]
            )
            out[f"likely_{axis}s"] = [" | ".join(np.asarray(labels)[row]) for row in predictions]
            out[f"{axis}_status"] = np.where(~gate, "not_applicable",
                np.where(predictions.any(axis=1), "likely", "insufficient_evidence"))
    out["reason_model_version"] = VERSION
    out["reason_output_policy_version"] = (
        OUTPUT_POLICY_VERSION if policy == MIXED_OUTPUT_POLICY else "legacy-per-code-threshold-v1"
    )
    return out


def rescore_saved_july_mixed(output=JULY_MIXED_OUTPUT, source=JULY_OUTPUT):
    """Publish the mixed July policy from immutable saved scores; never refit."""
    output, source = Path(output), Path(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite scored release: {output}")
    source_ledger = source / "reason_code_predictions.parquet"
    source_manifest_path = source / "scoring_manifest.json"
    frozen_manifest_path = MODEL_DIR / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    model_path = MODEL_DIR / "reason_heads.joblib"
    model_hash, gate_hash = sha(model_path), sha(JULY_GATE)
    if model_hash != frozen_manifest.get("model_sha256"):
        raise ValueError("Frozen reason model checksum mismatch")
    if source_manifest.get("model_sha256") != model_hash:
        raise ValueError("Saved-score manifest names a different reason model")
    if source_manifest.get("input_hashes", {}).get("detections") != gate_hash:
        raise ValueError("Saved-score manifest names a different binary gate")
    frame = pd.read_parquet(source_ledger)
    binary = pd.read_parquet(JULY_GATE,
        columns=["station_id", "hour_utc", "random_probability", "random_prediction"])
    if not binary.equals(frame[list(binary.columns)]):
        raise ValueError("Saved reason scores do not match the frozen July binary gate")
    gate = frame["random_prediction"].eq(1).to_numpy()
    bundle = joblib.load(model_path)
    for axis, labels in (("mechanism", MECH), ("component", COMP)):
        scores = frame[[f"{axis}_score__{label}" for label in labels]].to_numpy(float)
        thresholds = np.asarray([bundle["heads"][f"{axis}:{label}"]["threshold"] for label in labels])
        pred = apply_output_policy(scores, thresholds, gate, MIXED_OUTPUT_POLICY[axis])
        frame[f"likely_{axis}s"] = [" | ".join(np.asarray(labels)[row]) for row in pred]
        frame[f"{axis}_status"] = np.where(~gate, "not_applicable",
            np.where(pred.any(axis=1), "likely", "insufficient_evidence"))
    frame["reason_output_policy_version"] = OUTPUT_POLICY_VERSION

    truth_path = ROOT / "data/eval/july_2026_reason_code_evaluation/aligned_reference.parquet"
    truth = pd.read_parquet(truth_path)
    keys = ["station_id", "hour_utc"]
    if not truth[keys].equals(frame[keys]):
        raise ValueError("July truth and saved-score ledger keys differ")
    known = (~truth["truth_fault"].astype(bool) | truth[MECH + COMP].any(axis=1)).to_numpy()
    metrics = {}
    for axis, labels in (("mechanism", MECH), ("component", COMP)):
        actual = truth.loc[known, labels].to_numpy(bool)
        predicted = np.zeros((known.sum(), len(labels)), bool)
        emitted = frame.loc[known, f"likely_{axis}s"].fillna("")
        for j, label in enumerate(labels):
            predicted[:, j] = emitted.str.split(" | ", regex=False).apply(lambda values: label in values).to_numpy()
        tp = int((actual & predicted).sum()); fp = int((~actual & predicted).sum()); fn = int((actual & ~predicted).sum())
        metrics[axis] = dict(micro_f1=2 * tp / (2 * tp + fp + fn), tp=tp, fp=fp, fn=fn,
            assigned_alert_hours=int(frame[f"{axis}_status"].eq("likely").sum()))
    output.mkdir(parents=True, exist_ok=False)
    frame.to_parquet(output / "reason_code_predictions.parquet", index=False)
    manifest = dict(model_version=VERSION, output_policy_version=OUTPUT_POLICY_VERSION,
        output_policy=MIXED_OUTPUT_POLICY, source_saved_scores=str(source_ledger.relative_to(ROOT)),
        source_saved_scores_sha256=sha(source_ledger), model_sha256=sha(MODEL_DIR / "reason_heads.joblib"),
        source_scoring_manifest_sha256=sha(source_manifest_path),
        binary_gate_sha256=sha(JULY_GATE), scored_hours=len(frame), alert_hours=int(gate.sum()),
        original_eligible_metric_hours=int(known.sum()), unknown_fault_hours_excluded=int((~known).sum()),
        evaluation_truth_sha256=sha(truth_path), evaluation_truth_read_after_assignment=True,
        metrics=metrics, binary_health_weather_unchanged=True, retraining_performed=False,
        disclosure="Policy chosen after exploratory July evaluation; these are not independent policy-validation metrics.")
    (output / "scoring_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def score(raw_path, reference_path, detection_path, output, model_dir=MODEL_DIR, output_policy=None):
    """Score a period; direct API callers default to historical thresholds."""
    output, model_dir = Path(output), Path(model_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite scored release: {output}")
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    model_path = model_dir / "reason_heads.joblib"
    if sha(model_path) != manifest["model_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    bundle = joblib.load(model_path)
    detections = pd.read_parquet(detection_path, columns=["station_id", "hour_utc", "random_probability", "random_prediction"])
    detections["hour_utc"] = pd.to_datetime(detections.hour_utc, utc=True)
    raw = load_observations(raw_path, reference_path, detections.hour_utc.max())
    features, _ = build_features(raw)
    policy = LEGACY_THRESHOLD_POLICY if output_policy is None else output_policy
    result = predict(features, detections, bundle, output_policy=policy)
    # Delete-the-future check on every station at a middle scoring timestamp.
    cutoff = detections.hour_utc.sort_values().iloc[len(detections) // 2]
    past, _ = build_features(raw.loc[raw.hour_utc.le(cutoff)])
    early = predict(past, detections.loc[detections.hour_utc.le(cutoff)], bundle, output_policy=policy)
    pd.testing.assert_frame_equal(result.loc[result.hour_utc.le(cutoff)], early)
    assert sha(model_path) == manifest["model_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(output / "reason_code_predictions.parquet", index=False)
    alert = result.random_prediction.eq(1)
    report = dict(version=VERSION, model_sha256=manifest["model_sha256"],
        output_policy_version=(OUTPUT_POLICY_VERSION if policy == MIXED_OUTPUT_POLICY else "legacy-per-code-threshold-v1"),
        output_policy=dict(policy),
        input_hashes={"raw": sha(Path(raw_path)), "references": sha(Path(reference_path)), "detections": sha(Path(detection_path))},
        first_hour=str(result.hour_utc.min()), last_hour=str(result.hour_utc.max()),
        scored_hours=len(result), stations=int(result.station_id.nunique()), alert_hours=int(alert.sum()),
        mechanism_assigned_hours=int(result.mechanism_status.eq("likely").sum()),
        component_assigned_hours=int(result.component_status.eq("likely").sum()),
        both_assigned_hours=int((result.mechanism_status.eq("likely") & result.component_status.eq("likely")).sum()),
        delete_future_invariant=True, prefix_check_cutoff=str(cutoff), labels_read_at_inference=False,
        interpretation="Assignment counts are coverage, not accuracy; no July truth used")
    (output / "scoring_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "july", "july-mixed", "score"])
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-policy", choices=["mixed", "threshold"], default="mixed",
        help="CLI release policy (default: mixed); Python APIs retain threshold defaults")
    args = parser.parse_args(argv)
    if args.action == "train":
        train(args.model_dir)
    elif args.action == "july":
        policy = MIXED_OUTPUT_POLICY if args.output_policy == "mixed" else LEGACY_THRESHOLD_POLICY
        default_output = JULY_MIXED_OUTPUT if args.output_policy == "mixed" else JULY_OUTPUT
        score(JULY_RAW, JULY_REFS, JULY_GATE, args.output or default_output, args.model_dir, policy)
    elif args.action == "july-mixed":
        rescore_saved_july_mixed(args.output or JULY_MIXED_OUTPUT)
    else:
        if not all((args.raw, args.references, args.detections, args.output)):
            parser.error("score requires --raw, --references, --detections and --output")
        policy = MIXED_OUTPUT_POLICY if args.output_policy == "mixed" else LEGACY_THRESHOLD_POLICY
        score(args.raw, args.references, args.detections, args.output, args.model_dir, policy)
