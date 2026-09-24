# Validation-only improvement search

The requested target (validation precision, recall and F1 all above 0.80) was
not reached. No models were deployed, no labels or splits were changed, and no
test predictions were calculated by this search.

The previous corrected 120-feature representation was compared with a
436-feature extension. Added features include current readings for further
channels, 1/3/6/12/24-hour differences, trailing 3/6/12/24-hour variance, range
and flat fractions, and cyclic hour-of-day values. Duplicate newly proposed
features were removed. Actual-data prefix checks passed for the new features.
Missing hourly slots remain missing; there is no backward filling.

Twelve HGB configurations varied tree capacity, iteration count and class weight.
Four CatBoost configurations compared both feature sets and varied depth/weight.
CatBoost numerical imputation medians were fitted on training rows only. Decision
thresholds were selected from validation scores using the minimum of precision,
recall and F1. A 21-point HGB/CatBoost weight grid was subsequently evaluated on
validation only.

| Candidate | Validation precision | Validation recall | Validation F1 |
|---|---:|---:|---:|
| Previous selected corrected HGB | 62.30% | 59.27% | 60.75% |
| Best new HGB | 63.83% | 63.60% | 63.72% |
| Best CatBoost | 67.47% | 67.94% | 67.70% |
| Best validation blend | 67.76% | 67.76% | 67.76% |

The blend weights are 0.05 HGB and 0.95 CatBoost. Its selected threshold is
0.2280371885. These are experimental settings, not production replacements.

Diagnostic: 230 of 577 February reference fault hours had at least one original
retrospective stuck indicator but no corresponding causal stuck indicator. This
does not prove such hours cannot be detected from other signals, but quantifies
the timing mismatch. The best CatBoost could reach precision at least 80% only
with recall at most 57.54%; demanding recall at least 80% limited precision to
54.28%. Threshold adjustment alone therefore did not meet both requirements.

Limitations: the labels remain retrospective, the old continuous evidence and
cached statistical-detector fitting were retained, and February has now been
repeatedly used for tuning. This is not a full leakage-free or independent
hardware-fault validation. Test results were not used by this search; the
March-April period was already inspected in earlier experiments. A fresh
evaluation is required before making deployment claims. Returning to backfilled
stuck features would restore unavailable future information, not resolve it.

Artifacts: `data/eval/causal_validation_search_20260923/`, including trial grids,
feature cache, candidate models, validation predictions and blend selection.
Scripts: `scripts/search_causal_binary_validation.py`,
`scripts/search_causal_catboost_validation.py`,
`scripts/check_causal_validation_blend.py`.

## Historical window-length figure

`docs/figures/hgb_window_comparison_with_temporal.png` adds the original one-hour
blocked HGB result (89.51% validation F1; 86.97% test F1) to the random/spaced
history-window comparison. No saved blocked 3/5/7/12/24-hour results were found;
these are explicitly marked not evaluated rather than interpolated. All plotted
results are historical and retain the original retrospective feature limitation.
