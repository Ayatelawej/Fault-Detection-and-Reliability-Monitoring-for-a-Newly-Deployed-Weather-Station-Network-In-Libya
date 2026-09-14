# Why July reason-code F1 fell, and what to fix

## Main finding

The reason heads did not generally lose their ability to distinguish faults.
With the chosen minimum-one policy and known, resolved fault hours, July micro
F1 remains **98.01% mechanisms / 94.31% components**. The lower **71.03% / 69.33%**
is a detector-to-reason result: false fault alerts get false explanations, and
missed faults never reach the reason heads. These are different evaluation scopes.

On a more comparable development cascade using the frozen binary detector,
minimum-one micro F1 was 88.33% / 88.24%. Thus a real future-period drop remains,
but comparing the conditional 98% result directly to the July cascade exaggerates
the reason heads' apparent degradation. The reason models also differ between
the development split-specific experiment and the final June-wide refit.

## 1. The biggest loss comes from the binary detector

| Frozen detector | Development random test | July |
|---|---:|---:|
| Precision | 91.47% | 74.25% |
| Recall | 92.58% | 80.01% |
| F1 | 92.02% | 77.02% |
| False-positive rate on reference-normal hours | 0.66% | 3.98% |

The false-positive rate is about six times higher. July has 472 false alerts
and 340 missed reference fault hours. Of the missed hours, 109 have supported
current-hour reasons and enter the timed reason metrics; the other 231 are
unresolved under that reference contract.

Exact minimum-one error decomposition on the previous timed evaluation population:

| Axis | False labels on normal hours | Wrong extra labels on resolved faults | Missed labels due to binary detector | Missed labels after detector alert |
|---|---:|---:|---:|---:|
| Mechanisms | 472 | 14 | 109 | 21 |
| Components | 474 | 43 | 109 | 35 |

These are label counts, not necessarily distinct hours: a multilabel alert can
produce more than one incorrect component. All 472 false alerts get a statistical
anomaly mechanism under minimum-one. Of their component assignments, 380 are
thermo-hygrometer, 46 anemometer, 45 barometer and 3 light/UV.

Physical spikes and stuck/flatline mechanisms remain perfect on the supported
July target hours. The statistical route is the principal bottleneck. Light/UV
illustrates why low component F1 need not mean a broken component classifier:
the detector misses 9 of its 10 positive hours, and the component head correctly
labels the single positive hour that reaches it.

## 2. Evidence points to statistical-feature drift and insufficient context

The binary model uses 67 features. Its sensor-group flags are OR summaries of
physical, stuck, robust-z and Isolation Forest flags (`_sensor_group_flags` in
`src/rules/feature_matrix.py`, using the `flag` union from `src/rules/score.py`).
They do not retain which channel/detector combination fired inside a group.

The robust-z baseline uses a station or pooled median/MAD, with no month/hour
conditioning in `src/rules/baselines.py`. The July replay freezes those values
through June. Descriptive evidence from reference-normal rows:

- Median temperature statistical z-score shifts from **0.25 in training** to
  **2.42 in July**, and is **4.48 among July false alerts**.
- **72.67% of false-alert hours** have that temperature score outside the
  training-normal 1st-to-99th-percentile range.
- All 472 false alerts contain at least one sensor-group rule flag. A grouped
  permutation of those flags on pre-July validation reduces fixed-model F1 from
  92.44% to 7.34%, indicating very strong reliance on those inputs. This is a
  dependence diagnostic, not proof that removing them would help; they also
  encode essential genuine-fault evidence.
- Several 24-hour variance inputs are unavailable more often: the temperature
  variance input is missing on 38.79% of training-normal hours versus 64.67% of
  July-normal hours. The binary inputs have little information about rolling
  window coverage beyond the single-hour presence/gap features.

Together these findings are consistent with seasonal/operating-distribution
shift affecting statistical alerts. They do **not** establish that every false
alert is normal summer weather or identify a verified physical cause.

False alerts are concentrated but not confined to one station: INUQAT9 has 75,
IJABAL15 41, IBIRAL3 34 and ITRIPO33 32. The issue needs station-aware analysis,
not a global decision to suppress temperature faults.

## 3. Preprocessing is not identical between development and replay

On all 130,218 shared development station-hours, **118,081** have at least one
changed engineered input in the combined July tensor. Examples:

- Temperature statistical z-score: 97,903 changed rows.
- Wind-speed statistical z-score: 63,004 changed rows.
- Solar statistical z-score: 57,502 changed rows.
- Several external normalization fields and sensor-group flags also differ.

The underlying temperature, wind-speed and solar raw observations match exactly
on all 166,017 shared raw station-hours. This confirms a feature-artifact/version
or preprocessing discrepancy rather than changed observations for those channels.
The historical origin of each difference has not been fully traced.

A controlled check keeps the model, labels, hours and threshold unchanged and
changes only the development inputs to their July-pipeline versions:

| Same development test | Original features | July-pipeline features |
|---|---:|---:|
| F1 | 92.02% | 92.08% |
| False positives | 120 | 90 |
| False negatives | 103 | 127 |

There are 136 decision flips. **The mismatch needs fixing for consistency, but
this test does not show it explains the July F1 collapse**: aggregate development
F1 is almost unchanged. Do not promise that aligning these features alone will
recover July's lost score.

## 4. Development splits gave an easier view than a future-month test

Of the binary detector's 1,389 random-test fault hours, **1,314 (94.60%) share a
connected fault event with training**. This is event dependence across the random
split, not evidence that labels were directly passed to the model. It makes the
random result less informative about unseen future events. The final reason
heads' grouped-OOF selection is separate and should not be confused with this
binary-detector random split.

There is enough fault support to investigate a proper chronological pipeline:
April has 1,027 labelled fault hours, May 1,331 and June 1,103 before any purging.
Events crossing boundaries and feature/label lookahead still require handling.

## 5. Threshold and reference-contract effects remain important

The saved dashboard still uses the earlier abstaining reason policy. On the 403
detected hours labelled statistical anomaly, the median head score is 0.433,
below its 0.70 threshold; 71.96% fall below that threshold. This explains much of
the previously reported low abstaining-policy recall. It does not explain the
remaining 71.03% / 69.33% minimum-one result, because minimum-one already bypasses
that lack of a threshold-crossing code. Raising or lowering all reason thresholds
is therefore not a sufficient fix for the main problem.

The reference is evidence-derived, not hardware ground truth. Two continuing
pressure-offset conditions were deliberately recorded separately rather than
inserted as new July fault labels. **41 of the 472 false-alert hours overlap those
recorded conditions**. This is an identifiable label-contract ambiguity, not
permission to relabel those predictions as correct after seeing the result.
It also cannot explain the other 431 false alerts.

The 843 fault hours without supported current-hour reasons remain unknown; the
administrator-routing experiment showed that changing the output task does not
repair specific-reason classification. Calibration-offset and wind-vane positive
support is absent in this July reference, and light/UV has only ten positive hours.

## Proposed fix order — not implemented in this diagnosis

1. **Make preprocessing a versioned part of the detector release.** Use one
   feature builder and train-fitted baseline objects for development and replay;
   assert that appending future data cannot alter a historical feature prefix.
   Rebuild training features and refit the detector if its original artifact's
   preprocessing cannot be reproduced. Trace raw, normalization and flag stages
   separately; do not just patch the final July tensor to resemble training.
2. **Re-evaluate the full detector chronologically.** For example, train through
   April, select policies on May and test once on June, with event-disjoint
   boundary purging. Fit preprocessing inside each training period, not on all
   June data before testing earlier months. Use additional rolling origins to
   check stability. July is now an inspected diagnostic period, not a fresh
   confirmatory holdout; retain untouched later data for final validation.
3. **Improve the statistical-fault inputs and decision rule first.** Preserve
   the strong physical/stuck evidence, but distinguish per-channel z-score,
   Isolation Forest and sustained-flatline flags instead of only their group OR.
   Add causal hour-of-day/season context, trailing normal behaviour, reference
   and neighbour disagreement, persistence, and window coverage. Reuse appropriate
   causal primitives from the reason redesign rather than blindly copying all
   322 features. Add partial-window continuous variability as a separate feature
   with explicit coverage; do not relax the strict stuck-label confirmation rule
   merely to reduce missingness.
4. **Include evidence-reviewed hard negatives and ambiguous cases.** Use normal
   high-temperature/high-z hours from pre-test periods, with station/event-balanced
   sampling. Keep July error inspection separate from final evaluation. Review
   continuing-condition target semantics without silently revising the frozen
   baseline truth to reward the current model.
5. **Select the complete operating policy on validation.** Measure binary
   precision/recall and mechanism/component cascade F1 together; compare
   minimum-one and abstention explicitly. A higher detector threshold can reduce
   false explanations but also worsen already-missed light/UV and statistical
   faults. Do not tune against July or assume one threshold improves both axes.

No retraining, threshold changes, label edits, or deployment changes were made.
These are evidence-backed priorities, not measured gains from implemented fixes.

## Reproduction

`python scripts/diagnose_july_reason_drop.py --output <new-directory>` writes
fixed-model diagnostic tables. The complete run is
`data/eval/july_reason_drop_diagnosis_v2/`: error attribution, binary comparisons,
station/hour diagnostics, head-score distributions, feature drift, preprocessing
parity, validation permutation reliance and checksum audit. Original July model
probabilities were reproduced exactly and protected inputs were unchanged.
