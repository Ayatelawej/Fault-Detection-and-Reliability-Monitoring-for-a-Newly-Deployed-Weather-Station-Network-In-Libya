# March–April blocked EF-HGB holdout — 14 September 2026

This requested follow-up holds out March–April 2026 while including May–June
2026 in training. It is a blocked-period classification test, not chronological
future prediction. The earlier May–June chronological result remains unchanged.

## Fixed design

- Training: June 2025–January 2026 and May–June 2026, excluding boundary gaps and
  crossing groups; 84,377 hours, including 5,866 reference faults.
- Validation: 8–28 February 2026; 10,411 hours, including 577 reference faults.
- Test: 8 March–30 April 2026; 25,260 hours across 25 stations, including 1,686
  reference faults. March 1–7 is a boundary gap, not part of the scored test.
- Seven-day post-boundary exclusions at February 1, March 1 and May 1. Entire
  connected fault groups crossing partition boundaries are removed. There is
  no shared group or row across partitions; 10,170 hours are excluded overall.
- Fresh clones of the final binary EF-HGB's three HGB estimators; same 67
  features, hyperparameters and fusion weights (0.9 full, 0 context, 0.1 rules).
  No March–April rows enter estimator fitting or threshold selection.
- Primary threshold 0.30. Secondary threshold chosen using February only from
  the established 0.30–0.80 grid in 0.05 steps, maximizing minimum precision,
  recall and F1, then F1/precision/recall. February selected 0.35 before test scoring.

## March–April test results

| Metric | Fixed threshold 0.30 | February-selected 0.35 |
|---|---:|---:|
| F1 | 87.01% | 87.32% |
| Precision | 86.65% | 88.69% |
| Recall | 87.37% | 86.00% |
| Accuracy | 98.26% | 98.33% |
| True positives | 1,473 | 1,450 |
| False positives | 227 | 185 |
| False negatives | 213 | 236 |
| True negatives | 23,347 | 23,389 |

AUROC is 0.9938 and average precision is 0.9553 for both thresholds. February
validation F1 is 87.31% at 0.30 and 89.69% at its selected 0.35.

## Interpretation

This is useful supplementary evidence: the model can classify a held-out
two-month block substantially better here than in the earlier chronological
May–June experiment (71.96% F1). At the unchanged 0.30 threshold, the numerical
difference is 15.05 percentage points. This is not a controlled estimate of a
seasonal effect: both training composition and test period changed, and later
months are now available for training. It neither proves that heat caused the
previous decline nor demonstrates 87% future-month F1. Retain both experiments
and name their different designs explicitly.

Existing feature preprocessing and weak reference labels were reused, including
their previously documented retrospective timing limitations. Architecture,
hyperparameters and fusion weights were already selected using development
data containing the now-held-out months. This remains exploratory, not a fresh
independent evaluation of the complete pipeline or confirmed hardware diagnoses.

## Verification and artifacts

Five targeted tests passed, including blocked-period membership, boundary gaps,
crossing-event removal and validation threshold selection. The saved experimental
model exactly replayed 100 test scores; confusion counts were checked. Protected
artifact hashes were unchanged: deployed binary/reason models, mixed July ledger,
original tensor, original split manifest and dashboard sources. No Word report
edits or deployment changes were made. Runtime was about 7.8 seconds after
imports with a two-thread limit; no training job remains running.

Saved output: `data/eval/blocked_ef_hgb_20260914/` contains the pre-fit plan,
threshold selection, prediction ledger, split membership, isolated model,
metrics and final audit. The prior chronological output was not overwritten.

```powershell
python scripts/experiment_chronological_fault_detection.py --mode blocked --output data/eval/blocked_ef_hgb_new_run
python -m pytest tests/test_chronological_fault_detection.py -q
```
