# Previous reason release: original episode targets

Superseded by `data/model/final_system_20260924/reasons/`. See
[complete model-selection results](final_model_selection_results.md). The paths
and scores below describe the preserved previous release, not the active dashboard.

The active reason model is `data/model/reason_codes/episode_v2/reason_heads.joblib`.
Its July ledger is `data/eval/july_2026_reason_codes_episode_v2/reason_code_predictions.parquet`.
This replaces the mixed-policy current-hour release in the local July replay.
It is not a newly connected live station service.

## Training and output

EF-HGB retains the 322 observation/history/reference features and three feature
views. It now learns original episode mechanism/component targets, including all
development reference fault hours, with sampled normal negatives. Fusion weights
and thresholds are copied from February validation selection in the blocked
pilot; trees are refitted on development data through June. July is not used
for fitting or threshold selection. The release manifest records input hashes.

Each binary-positive hour receives at least one mechanism and component, while
retaining multiple above-threshold codes. Binary-negative hours have no reasons.
Existing full-outage dashboard suppression still applies. These are predicted
event-associated labels, not assertions that each corresponding evidence rule
fires at the displayed hour. No truth or episode identifier is read at inference.
Episode reference labels can be established retrospectively; archived weather
reference arrival-time availability remains unverified.

## Held-out experiment results (not final-refit scores)

| Split | Fault hours | Mechanism micro-F1 | Component micro-F1 |
|---|---:|---:|---:|
| Random | 1,389 | 89.36% | 82.08% |
| Grouped | 1,382 | 76.74% | 61.80% |
| March–April blocked holdout | 1,686 | 73.69% | 80.75% |

These evaluate all reference fault hours against episode labels, without a binary
gate. Grouped is the reason-specific grouped split, not the binary spaced split.
The blocked split trains on both sides of the held-out block. Its configuration
was used for the final refit because temporal holdout is the designated primary
comparison, not because it wins every test metric. No above scores are July scores.

## Operations and rollback

Release checks passed: 13,565 July ledger rows, unchanged binary gate, both outputs
assigned on all 1,833 binary-positive rows and suppressed on binary-negative rows.
The saved scoring prefix is unchanged when future observations are removed.
Thirty targeted reason/dashboard tests passed, including the local app smoke test.
The release uses all 9,260 development reference fault hours.

`python scripts/final_reason_codes.py train` refits this release using the saved
validated selections. `python scripts/final_reason_codes.py july` generates its
July ledger. Both refuse to overwrite existing releases; alternative output/model
directories must be explicitly supplied for a new version. A successful scoring
manifest records the delete-the-future invariant and matching binary gate hash.

Preserved rollback artifacts: `data/model/reason_codes/final/` and
`data/eval/july_2026_reason_codes_mixed_v2/`. Restore the previous dashboard reason
path to roll back; do not delete or overwrite either ledger. Legacy `july-mixed`
still operates on those historical artifacts. Binary detector, health scores,
forecast release and weather notes are unchanged. No Word report was edited.
