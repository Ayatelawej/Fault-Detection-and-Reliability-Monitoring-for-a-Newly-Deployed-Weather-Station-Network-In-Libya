# June–July contract audit, weather notes and one temperature specialist

## Outcome

The preparation difference is now traced to concrete normalization/history
boundaries. June's retained labels and raw data did not change. Only the
score-neutral weather annotation was added to the July dashboard. The tested
temperature specialist was **not promoted**, because July fault-detection F1
fell from 77.02% to 73.56%.

## What the audit verified

### Labels and observations

- All **4,799 pre-July episodes** in the July label package match the canonical
  records in every field: state, binary target, mechanisms, components, fired
  channels, interval boundaries, duration, station and period.
- All **166,017 shared raw station-hour rows** match in every compared column.
- The July adjudication file records 877 automatic decisions confirmed, 57
  retained exclusions and 40 cases changed to benign. The manifest specifies
  contextual/external evidence requirements for resolving benign cases and
  records that labels were frozen before prediction access. Its stored label
  checksum still matches.
- Continuing calibration conditions were deliberately recorded separately,
  without creating new July-only episodes. That is a documented target-contract
  choice, not a change in the EF-HGB algorithm.
- The original automatic July source package named in the freeze manifest is
  absent locally. Thus this verifies retained records, recorded adjudication and
  checksums—not a complete regeneration proving identical generating code and
  parameters for every June/July label.

### Statistical preprocessing: exact baseline boundary found

For temperature average, wind-speed average and solar-high z-scores, training's
stored tensor matches `data/processed/statistical_anomaly_scores.parquet` exactly.
Tracing that score cache against raw-derived median/MAD baselines finds:

| Scored period | Training score cache matches | July replay score cache matches |
|---|---|---|
| Before June | Baselines fitted through May | Baselines fitted through June |
| June | Baselines fitted through June | Baselines fitted through June |

The relevant comparisons cover 318,017 pre-June channel-hour rows and 43,040
June channel-hour rows, with zero mismatches for each stated baseline match.
Example: INUQAT9 temperature used median/MAD **17.7625 / 3.7625** before June,
then **18.4 / 4.4** in June. The July-frozen baseline is also 18.4 / 4.4.

Thus July's frozen baselines are being applied consistently within that replay,
but the training history contains more than one normalization generation.
Preserving previous-month caches is not equivalent to recomputing all history
with the latest frozen baseline.

### External preprocessing: June history reset reproduced

Station temperature, reference temperature and their raw residual match across
the two external artifacts. Differences occur in derived baselines/spreads and
z-scores around June.

Recomputing temperature and pressure normalization using **June-only history**
reproduces the old June `base_temp`, `bmad_temp`, `z_temp`, `base_pressure` and
`bmad_pressure` fields exactly across **18,000 station-hours**. The July rebuild
instead carries earlier history into June, explaining its different June
normalization/warmup values. No raw-temperature change is needed to reproduce
this discrepancy.

These findings explain the preparation mismatch; they do not prove it causes
the full July score drop. The earlier controlled same-model/same-test check
gave 92.02% F1 with original features and 92.08% with replay-prepared features.
No cache, label or existing model was overwritten to force parity.

## Score-neutral weather annotation: added to dashboard

The Network and Station views now show a note on **204 July alert hours**:

> Temperature pattern consistent with recent history, reference weather and
> neighbours. Alert retained; sensor fault not ruled out.

The note uses the stricter already-tested weather-consistency condition. It is
not a diagnosis of a healthy sensor and can occur on labelled fault hours.
It neither suppresses alerts nor changes fault probabilities, reason codes,
health scores or F1. Full-outage and non-alert snapshot rows suppress the note;
the join uses the exact replay hour. The dashboard loads only operational note
fields and verifies that their binary gate matches the selected detector.

The published note ledger is
`data/eval/july_2026_weather_annotations/weather_annotations.parquet`, with
source/gate checksums in its manifest. `scripts/export_weather_annotations.py`
publishes only the notes—not experimental specialist decisions—and refuses to
overwrite an existing release.

## Targeted numerical refinement: tested and rejected

One small HGB temperature classifier was trained on **577 pre-July eligible
temperature-statistical hours**, including 273 reference fault hours. It uses
20 features: recent same-hour temperature behaviour, external/peer disagreement,
raw temperature/residual levels and changes, coverage, calendar phase, and
prior 6/24-hour corroborating evidence. Continuity uses **observations and prior
evidence**, not true episode IDs, fault labels or future-confirmed outcomes.

Three event-disjoint June-development OOF folds selected threshold **0.45** using
mean event-weighted candidate F1. The OOF selection score was 75.97%, versus
57.93% for always retaining the temperature candidate. July was loaded for
evaluation only after selection/refit. This is a conditional specialist-selection
score, not independently held-out full-pipeline accuracy.

The specialist only reconsidered existing temperature-only alerts without
physical/current-confirmed-stuck or other-channel statistical evidence; it did
not add alerts or change the other detector paths or reason heads.

| July policy | Precision | Recall | Binary F1 |
|---|---:|---:|---:|
| Existing detector | 74.25% | 80.01% | **77.02%** |
| Annotation only | 74.25% | 80.01% | **77.02%** |
| Temperature specialist | 77.49% | 70.02% | 73.56% |

It removed 126 false alerts but lost 170 labelled fault hours. It therefore fails
the intended fault-detection improvement criterion and was left isolated.
Minimum-one reason micro F1 would become 72.11% mechanism / 70.75% component,
but this must not mask the deterioration in binary detection; reason metrics
exclude unknown-reason fault hours. The reason policy itself was not switched.

## Limits and verification

- The annotation and new continuity features passed prefix checks. Two unit
  tests verify score-neutral snapshot integration and prior-only continuity;
  the relevant regression suite passed **62 tests**, and Streamlit smoke tests
  passed with the new column visible.
- The upstream baseline detector/statistical flags are still the inherited
  frozen artifacts. This is not a fresh chronological full-pipeline evaluation.
  Archived reference/peer arrival-time availability and legacy backfilled stuck
  flags remain limitations of the historical replay.
- No July threshold retuning or second specialist search followed the poor
  result. The original detector, reason heads, labels and health artifacts are
  unchanged. Only the independent weather-note output and dashboard presentation
  were added.

Audit outputs: `data/eval/july_contract_audit_v2/`. Specialist outputs and
annotation source: `data/eval/temperature_specialist_check/`. Reproduction:
`scripts/audit_july_contracts.py --output <new-directory>` and
`scripts/test_temperature_refinement.py --output <new-directory>`.
