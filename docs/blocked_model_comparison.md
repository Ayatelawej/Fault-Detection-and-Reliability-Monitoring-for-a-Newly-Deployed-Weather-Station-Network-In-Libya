# Blocked-period model comparison

This experiment compares HGB, RGFN and EF-HGB using the same blocked-period design: training uses June 2025-January 2026 and May-June 2026, February 2026 is validation, and March-April 2026 is the held-out test period. Seven-day boundary gaps and crossing fault groups are removed. Configuration and threshold choices use training/February only; March-April is scored after those choices are frozen.

## Binary fault detection

| Model | February F1 | March-April precision | March-April recall | March-April F1 |
|---|---:|---:|---:|---:|
| HGB | 89.51% | 87.42% | 86.54% | **86.97%** |
| RGFN | 88.31% | 84.56% | 85.98% | 85.15% |
| EF-HGB | **89.91%** | 87.67% | 86.06% | 86.86% |

February therefore selects EF-HGB. HGB is 0.11 percentage points higher on the held-out test F1, which is too small to justify reversing a validation-frozen choice. The earlier 87.32% blocked EF-HGB result used the already selected development configuration and a February-selected threshold; the 86.86% result above comes from repeating the full class-weight, fusion-weight and threshold selection grid for a fair three-model comparison.

## Current-hour reason classifiers

Unknown-reason fault hours are excluded rather than treated as negative labels. HGB uses the full reason-specific feature view, EF-HGB fuses full/context/rule views, and RGFN uses its gated sensor/evidence branches. Thresholds are selected using grouped out-of-fold training predictions. The table reports conditional macro-F1 on fault hours with supported labels.

| Policy | Model | February mechanism | February component | March-April mechanism | March-April component |
|---|---|---:|---:|---:|---:|
| Minimum-one both axes | HGB | 99.63% | 73.82% | 98.65% | 89.78% |
| Minimum-one both axes | RGFN | 98.14% | **77.51%** | 97.44% | 75.79% |
| Minimum-one both axes | EF-HGB | 99.63% | 69.40% | **98.65%** | **90.22%** |
| Mixed: minimum-one mechanism, threshold component | HGB | 99.63% | 68.47% | 98.65% | 85.61% |
| Mixed: minimum-one mechanism, threshold component | RGFN | 98.14% | **72.97%** | 97.44% | 64.71% |
| Mixed: minimum-one mechanism, threshold component | EF-HGB | 99.63% | 68.70% | **98.65%** | **85.78%** |

February alone selects RGFN under the mixed-policy mean across both axes, but RGFN's component performance does not transfer to March-April. EF-HGB gives the strongest March-April component result under both output policies and ties HGB for mechanism macro-F1. Together with the earlier random and spaced results, this supports retaining EF-HGB rather than replacing it with RGFN.

## Interpretation

The blocked test is a stronger period-transfer assessment than a random split, but a final test period should not be reused for tuning or retroactive model selection. If March-April is used to choose the model, a later untouched period is needed for final evaluation. For the present project, the defensible choice is to retain EF-HGB based on validation-frozen selection plus consistency across random, spaced and blocked assessments, while reporting the blocked comparison as additional robustness evidence.

Logistic regression would be a useful inexpensive classical baseline for the redesigned reason labels. A separate GRU baseline is not necessary here: the reason-code RGFN already uses a GRU sensor encoder, while the one-hour binary input contains no temporal sequence for a standalone GRU to exploit.
