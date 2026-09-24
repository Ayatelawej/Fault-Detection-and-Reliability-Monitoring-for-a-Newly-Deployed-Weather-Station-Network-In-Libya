# Feature mix and rule reliance: 23 September 2026

HGB only; production unchanged. All three variants use the same blocked
training/February-validation/March-April-test membership, checked by hashes.
Seven class weights and eleven thresholds use the original validation maximin
selection rule. No test results select the variant. All variants retain hard
and suspect checks, use endpoint-only stuck flags, rebuild group/suspect
indicators, and contain one copy of each rule indicator instead of any/rate pairs.

| Variant | Columns | Validation F1 | Test precision | Test recall | Test F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Corrected, deduplicated inputs | 48 | 54.69% | 75.97% | 68.45% | 72.01% | 96.45% |
| Add raw readings and presence indicators | 72 | 51.83% | 74.00% | 73.25% | 73.62% | 96.50% |
| Add past changes and unchanged-run lengths | 120 | 60.75% | 80.94% | 78.59% | 79.75% | 97.34% |

The third variant is selected by validation (class weight 2, threshold 0.40).
Removing duplicate columns alone reproduced the previous 67-column corrected
HGB metrics. This does not establish that duplication would have no effect on
RGFN, which was not trained in this experiment.

Raw measurements cover temperature, humidity, dew point, pressure, wind speed,
gust, direction (sine/cosine), rain rate/total, solar radiation and UV. Each has
a presence indicator. History features are differences at 1/3/6 hours and the
consecutive count of unchanged observations so far (absolute tolerance 1e-6,
capped at 168 hours). Missing observations break a run; there is no interpolation
or backwards filling. Station histories are placed on a complete hourly grid.
Actual-data prefix checks and a separate missing-value/run-reset example passed.

## Effect of individual rules on the selected variant

All numbers below are decreases in validation F1, in percentage points. Positive
means performance fell; negative means it improved. Shuffle is the mean of five
within-station permutations on validation. Removal means remove that one column,
retrain at the selected fixed settings, and evaluate on validation. These are
different questions: current fitted-model reliance versus fixed-setting retraining
sensitivity. Neither is a causal effect or a percentage of training attributable
to a feature. Correlated inputs and derived group flags remain, so effects are
not additive and zero does not prove the underlying evidence is useless.

| Indicator | Active validation hours | Shuffle | Remove and retrain |
|---|---:|---:|---:|
| Average wind speed stuck | 209 | 0.00 | 0.00 |
| Average gust stuck | 1 | 0.00 | 0.00 |
| Maximum wind speed stuck | 0 | 0.00 | 0.00 |
| Maximum gust stuck | 0 | 0.00 | 0.00 |
| Wind direction cosine stuck | 43 | 0.00 | 0.00 |
| Wind direction sine stuck | 43 | 0.00 | 0.00 |
| Solar hard limit | 0 | 0.00 | 0.00 |
| Solar suspect limit | 0 | 0.00 | 0.00 |
| Rainfall-total hard limit | 5 | 0.50 | 0.00 |
| Rain-rate hard limit | 4 | 0.00 | 0.00 |
| Pressure-trend hard limit | 4 | 0.03 | 1.84 |
| UV suspect limit | 0 | 0.00 | 0.00 |
| Anemometer group | 371 | -0.16 | 0.40 |
| Barometer group | 147 | 2.59 | -1.12 |
| Light/UV group | 10 | 0.01 | 0.05 |
| Other group | 0 | 0.00 | 0.00 |
| Rain-gauge group | 8 | 0.00 | 3.68 |
| Temperature/humidity group | 16 | -0.14 | -0.55 |
| Wind-vane group | 66 | 0.00 | 0.00 |

Several indicators are absent or rare in validation. The one-feature refits are
diagnostics, not separately tuned candidates; do not use them to claim a robust
feature ranking or automatically prune features. The original-label dependence,
cached statistical-detector fitting and existing continuous features remain
unchanged. This repairs the identified stuck-backfill problem but is not a full
leakage-free validation. The test period has also been inspected previously.

Artifacts: `data/eval/causal_feature_mix_20260923/` contains the comparison,
validation grids, selected model, test predictions, full rule importance CSV and
design manifest. Reproduction: `python scripts/experiment_causal_feature_mix.py`
with a fresh output directory.
