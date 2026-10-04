"""Frozen EF-HGB reason heads behind the selected binary detector.

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
EPISODE_VERSION = "episode-target-ef-hgb-v2"
SELECTED_VERSION = "validation-selected-episode-reasons-v3"
EPISODE_OUTPUT_POLICY = {"mechanism": "minimum_one", "component": "minimum_one"}
EPISODE_MODEL_DIR = ROOT / "data/model/reason_codes/episode_v2"
ACTIVE_MODEL_DIR = ROOT / "data/model/final_system_20260924/reasons"
ACTIVE_JULY_DIR = ROOT / "data/eval/final_system_release_20260924"
JULY_EPISODE_OUTPUT = ROOT / "data/eval/july_2026_reason_codes_episode_v2"
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
    if bundle["version"] not in {VERSION, EPISODE_VERSION, SELECTED_VERSION}:
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
    default_policy = EPISODE_OUTPUT_POLICY if bundle["version"] in {EPISODE_VERSION, SELECTED_VERSION} else LEGACY_THRESHOLD_POLICY
    policy = dict(default_policy if output_policy is None else output_policy)
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
    out["reason_model_version"] = bundle["version"]
    out["reason_output_policy_version"] = (
        "minimum-one-both-v3" if policy == EPISODE_OUTPUT_POLICY else
        OUTPUT_POLICY_VERSION if policy == MIXED_OUTPUT_POLICY else "legacy-per-code-threshold-v1"
    )
    return out


def score(raw_path, reference_path, detection_path, output, model_dir=ACTIVE_MODEL_DIR, output_policy=None):
    """Score a period using version-aware defaults; preserve legacy reproduction."""
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
    default_policy = EPISODE_OUTPUT_POLICY if bundle["version"] in {EPISODE_VERSION, SELECTED_VERSION} else LEGACY_THRESHOLD_POLICY
    policy = default_policy if output_policy is None else output_policy
    result = predict(features, detections, bundle, output_policy=policy)

    cutoff = detections.hour_utc.sort_values().iloc[len(detections) // 2]
    past, _ = build_features(raw.loc[raw.hour_utc.le(cutoff)])
    early = predict(past, detections.loc[detections.hour_utc.le(cutoff)], bundle, output_policy=policy)
    pd.testing.assert_frame_equal(result.loc[result.hour_utc.le(cutoff)], early)
    assert sha(model_path) == manifest["model_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    result.to_parquet(output / "reason_code_predictions.parquet", index=False)
    alert = result.random_prediction.eq(1)
    report = dict(version=bundle["version"], model_sha256=manifest["model_sha256"],
        output_policy_version=str(result.reason_output_policy_version.iloc[0]),
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
    parser = ArgumentParser(description="Score the final reason heads. For refitting use scripts/system.py stage reasons.")
    parser.add_argument("action", choices=["july", "score"])
    parser.add_argument("--model-dir", type=Path, default=ACTIVE_MODEL_DIR)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.action == "july":
        score(JULY_RAW, JULY_REFS, args.detections or ACTIVE_JULY_DIR / "binary_predictions.parquet",
              args.output or ACTIVE_JULY_DIR / "reasons", args.model_dir, EPISODE_OUTPUT_POLICY)
    else:
        if not all((args.raw, args.references, args.detections, args.output)):
            parser.error("score requires --raw, --references, --detections and --output")
        score(args.raw, args.references, args.detections, args.output, args.model_dir, EPISODE_OUTPUT_POLICY)
