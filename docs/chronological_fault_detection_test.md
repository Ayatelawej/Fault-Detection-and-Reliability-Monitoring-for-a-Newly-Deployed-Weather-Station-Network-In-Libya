# Bounded chronological EF-HGB test — 14 September 2026

The final binary EF-HGB configuration was refitted on earlier months and tested
on later months. This is an exploratory temporal-transfer diagnostic using the
existing feature tensor and evidence-derived reference labels, not a rebuilt,
fully causal labelling/preprocessing pipeline. Deployment and the Word report
were not changed.

## Design

Fresh, unfitted clones of all three saved HGB estimators were trained only on the
training partition. All 67 feature names, HGB hyperparameters and fusion weights
(full 0.9, context 0, rules 0.1) matched the saved final random-split EF-HGB.
No hyperparameter or fusion search was performed.

The existing reason experiment's chronological membership was independently
reconstructed and verified exactly. Connected fault groups crossing the April
or May boundaries were removed, together with seven-day post-boundary gaps.
There was no connected-group overlap between any pair of partitions.

| Partition | Dates | Hours | Reference faults |
|---|---|---:|---:|
| Training | 15 June 2025–31 March 2026 | 84,051 | 5,419 |
| Validation | 8–30 April 2026 | 11,199 | 581 |
| Test | 8 May–30 June 2026 | 27,689 | 2,005 |

The test includes 25 stations. Boundary/group exclusions remove 7,279 hours.
All binary reference fault hours are retained within each partition, including
hours whose reason labels are unknown; this is not a reason-only metric.

The primary policy retained the deployed threshold of 0.30. A secondary policy
selected a threshold using April validation only and the established grid
0.30–0.80 in steps of 0.05, maximizing the minimum of precision, recall and F1,
then F1, precision and recall. The choice was saved before test scoring.
April selected 0.30, so both policies have identical results. Thresholds below
0.30 were not tested; this is not a claim that 0.30 is globally optimal.

## Results

| Metric | April validation | May–June test |
|---|---:|---:|
| Precision | 89.55% | 88.14% |
| Recall | 67.81% | 60.80% |
| F1 | 77.18% | 71.96% |
| Accuracy | 97.92% | 96.57% |
| AUROC | 0.9910 | 0.9854 |
| Average precision (AUPRC summary) | 0.8946 | 0.8828 |
| True positives | 394 | 1,219 |
| False positives | 46 | 164 |
| False negatives | 187 | 786 |
| True negatives | 10,572 | 25,520 |

For context, the original random-test F1 was 92.02%. This time-separated test is
20.06 percentage points lower, but the difference is not a controlled estimate
of a single cause: training population, test population and event separation
all changed. The principal weakness at the retained threshold is recall:
786 of 2,005 reference faults were missed, while 164 normal hours were flagged.

## Is it worth doing?

Yes as a robustness check: it provides information the random split does not,
and it did not require a days-long pipeline rebuild. The measured experiment
runtime was about 14.4 seconds after Python imports, with two computational
threads. This does not estimate the runtime of a full causal backtest.

It is not evidence for replacing the deployed model or for claiming 92% F1 on
future months. Strong score ranking alongside lower recall makes a lower
validation-selected threshold a plausible small follow-up, but no improvement
has been measured and no threshold should be chosen from May–June results.
The observed decline does not establish heat as its cause.

## Limitations and verification

- Existing historical features and weak reference labels were reused. Inherited
  statistical adjudication, preprocessing fit periods and backfilled stuck flags
  have timing limitations; estimator fitting alone is chronological here.
- The architecture, hyperparameters and fusion weights were previously selected
  using development data that includes later months. This is therefore not an
  independent prospective evaluation of model selection.
- Labels represent the project's evidence framework, not verified hardware faults.
- Three targeted tests passed: chronological gaps/crossing-group removal,
  rejection of shared groups, and validation-only threshold selection.
- Serialized model predictions replayed exactly on 100 test rows; saved test
  confusion counts were checked against predictions.
- SHA-256 checks confirmed that the original binary model, feature tensor,
  existing split manifest, deployed reason model, mixed July ledger and dashboard
  source files were unchanged.

Artifacts: `data/eval/chronological_ef_hgb_20260914/` contains the pre-fit plan,
validation threshold table, selected threshold, metrics, prediction ledger,
split membership, isolated experimental model and final audit (`report.json`).

Reproduce into a new directory (existing outputs are never overwritten):

```powershell
python scripts/experiment_chronological_fault_detection.py --output data/eval/chronological_ef_hgb_new_run
python -m pytest tests/test_chronological_fault_detection.py -q
```
