# Final system: current-hour reason codes

The final July replay now combines the unchanged selected one-hour EF-HGB fault
detector, separately fitted EF-HGB mechanism/component heads, causal health,
availability, and the existing frozen health forecasts. Reason predictions
explain alerts; they do not modify the detector decision or double-count fault
burden in health scores.

## Frozen release and operation

- Implementation: `src/model/final_reason_codes.py`; shared feature and training
  primitives: `src/model/reason_code_rebuild.py`.
- Frozen model: `data/model/reason_codes/final/reason_heads.joblib`, with model
  checksum, input hashes and freeze date in `manifest.json`.
- Active July deployment: `data/eval/july_2026_reason_codes_mixed_v2/reason_code_predictions.parquet`
  and `scoring_manifest.json`. The previous `july_2026_reason_codes` release is
  immutable and retained for reproduction.
- Dashboard: `python -m streamlit run scripts/run_dashboard.py`. No fitting or
  inference occurs in Streamlit. It checks that the reason ledger matches the
  selected binary detector and clips reason histories to the simulated hour.

For a fresh model release, run `python scripts/final_reason_codes.py train`, then
`python scripts/final_reason_codes.py july`; the CLI defaults to the active mixed
policy and its versioned output directory. For another period, `score` also defaults
to mixed; use `--output-policy threshold` only for historical reproduction. Direct
Python calls to `predict()` or `score()` retain threshold-only defaults for backward
compatibility. Reproduce the active July output without inference or refitting from
validated saved scores with `python scripts/final_reason_codes.py july-mixed`.
Existing releases cannot be
overwritten: use `--model-dir` and `--output` for a separately versioned run.
For other periods use `score --raw <history.csv> --references <residuals.parquet>
--detections <binary.parquet> --output <new-directory>` with the same frozen model.
Supply prior history as well as the scoring period (at least 168 hours for full
rolling context), the canonical channel schema, and same-hour reference residuals.

## Final policy

All development station-hours are through 30 June. Targets use current-hour
evidence inside accepted fault episodes; unresolved fault-hour codes are unknown,
not negative examples. Fit rows combine resolved faults with a deterministic
sample of non-fault negatives. Weights balance events and classes. Three grouped
OOF folds select each head's convex fusion and threshold, followed by a full
development refit. July is never used in model fitting or per-code threshold selection. The selected
binary detector and its 0.30 threshold remain frozen and unchanged.

Reason-code assignments are restricted to binary-positive hours by the output
gate; this describes the output contract rather than a requirement on how many
rows an implementation computes internally. The active mixed
policy retains every code crossing its frozen threshold. If no mechanism clears
its threshold, the highest finite mechanism score is emitted, guaranteeing at
least one mechanism per alert. Components retain the original per-code thresholds
and may say `insufficient_evidence`; binary-negative hours say `not_applicable`.
Multiple codes remain possible. Numeric outputs are model scores, not calibrated
probabilities. Legacy callers retain threshold-only defaults unless they pass an
explicit output policy.
Full-outage snapshot rows suppress sensor-fault reasons and retain outage status.

The 322 feature columns are rebuilt on a continuous hourly clock with no forward
fill of event labels. Component heads have sensor-local views; wind direction has
explicit speed/calm context. Scoring reads only raw history, archived residuals,
the frozen model, and four operational binary columns. It does not read truth,
episode IDs or calibration adjudications. A delete-the-future check compares
predictions after truncating raw history at a mid-period hour.

## Interpretation and retained evidence

July scoring covers all 13,565 hours in the existing binary ledger
(25 stations; the dashboard retains all 26 registry stations, including outages),
from 1 July 00:00 to 31 July 21:00 UTC. All 1,833 detector-positive hours receive
a mechanism; 862 receive at least one above-threshold component and the other 971
retain component `insufficient_evidence`. On the original reason metric population
of 12,722 resolved-fault-or-normal hours, with 843 unknown fault hours excluded,
the mixed output measures **71.0254% mechanism micro-F1** and **71.8318% component
micro-F1**.

These are retrospective July measurements, not independent policy-validation
results: the mixed policy was chosen after exploratory July evaluation. No model
was retrained and no binary, health, weather, or per-code threshold output changed.
The scoring manifest records the source-score, source-manifest, model, binary-ledger
and evaluation-truth hashes and this disclosure. Labels are assigned before
evaluation truth is read; truth is used only for retrospective metrics, not inference.

This is deployed into the **historical July replay**, not connected to a live
station service. External-reference arrival-time availability remains unverified.
The labels are evidence-derived likelihood references, not independently verified
hardware failures. Calibration has only four development events and wind vane
twelve; their generalisation evidence remains limited.

The [redesign experiment](reason_code_redesign_experiment.md) retains its random,
grouped and chronological held-out results. Those are not held-out scores for
this new all-development refit. `selection.csv` reports OOF policy-selection
scores, not independent test accuracy. Ordinary scoring manifests report
assignment coverage without evaluation truth. The active saved-score mixed-policy
publication manifest additionally records retrospective July metrics and the
post-exploration policy-selection disclosure; these are not independent
policy-validation scores. Existing source data, final detector/forecast artifacts
and completed research evidence are preserved.

The obsolete quick-test runner/output and aborted first redesign run were moved
out of active paths into `private_notes/retired_reason_codes_20260912/` for recovery.
Legacy seven-hour research code remains available under `reason-codes-legacy`
because historical report reproduction still depends on it. It is not used by
the final reason pipeline or dashboard. The canonical delivered Word report is
`docs/report/Ayat_Elawej_EC499_Report_FINAL.docx`; its release process backs up the
previous version under `private_notes/release_report_20260914/backups` before
replacement. Report-specific release notes are maintained separately in
`docs/report/RELEASE_NOTES.md`.
