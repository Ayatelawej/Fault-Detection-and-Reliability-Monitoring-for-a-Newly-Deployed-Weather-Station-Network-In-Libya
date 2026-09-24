# Causal stuck and rule-family ablation

Quick HGB-only experiment, 23 September 2026. Production models, labels and
dashboard outputs were not changed.

All runs use the same original blocked split: 84,377 training rows, 10,411
February validation rows and 25,260 March-April test rows. Membership hashes
match the previous no-rule experiment. The original rule inputs were reproduced
exactly before modification. Each new variant uses the original seven class
weights and eleven decision thresholds, selected by validation maximin precision,
recall and F1. No test results select settings.

| Input variant | Columns | Validation F1 | Test precision | Test recall | Test F1 | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Original retrospective rule inputs | 67 | 89.51% | 87.42% | 86.54% | 86.97% | 98.27% |
| Causal stuck; retain hard/suspect checks; rebuild groups | 67 | 54.69% | 75.97% | 68.45% | 72.01% | 96.45% |
| Causal stuck plus statistical-only groups | 55 | 53.23% | 71.48% | 70.46% | 70.97% | 96.15% |
| Statistical-only groups | 43 | 53.23% | 71.48% | 70.46% | 70.97% | 96.15% |
| No rule indicators | 29 | 50.61% | 67.87% | 68.27% | 68.07% | 95.72% |

Every variant retains the same 29 non-rule inputs. The original and no-rule
controls come from the previously verified comparison. Statistical-only groups
combine cached robust-z and Isolation Forest flags, excluding physical and stuck
flags. Variants using these groups also exclude standalone hard/suspect columns.

Causal stuck flags require 24 hourly observations with variance below 1e-6,
retain the original zero/skip exceptions, and are set at the current endpoint
only. Missing hourly slots are explicitly reindexed. Historical flags are never
backfilled. Synthetic and actual station/channel prefix checks passed: deleting
future observations did not alter earlier stuck flags. In the causal-all variant,
group flags and suspect gating were rebuilt so they cannot retain the old stuck
signal indirectly.

The 14.96-percentage-point test-F1 reduction after the causal replacement shows
that retrospective feature construction materially benefited the original
current-hour evaluation. It is not evidence that predefined physical checks
are inherently unusable. Keeping statistical groups also performs better here
than removing every indicator, but differences are descriptive, not significance
tests.

Labels remain retrospective and unchanged, so an hour labelled faulty from the
beginning of a flat episode can precede when a causal 24-hour detector can fire.
This is a real recognition delay under the current label definition. This test
does not rebuild or independently verify the labels, continuous features, or the
fitting of cached statistical detectors. It is not a complete leakage-free
evaluation, and the test period has been inspected previously. EF-HGB and RGFN
were not retrained in this quick check.

Reproduction: `python scripts/experiment_causal_rule_indicators.py` (requires a
fresh output directory). Results, feature lists, validation grids and audit:
`data/eval/causal_rule_indicators_blocked_20260923/`.
