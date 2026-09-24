# Episode-label reason-classifier pilot — 24 September 2026

Isolated EF-HGB comparison using the same 322 reason features, fixed HGB settings,
saved blocked split membership and February-selected fusion weights/thresholds
for both training-target versions. Unlike the original experiment, this compact
pilot does not use three training-only OOF folds. It took 66 seconds on two threads.
March–April is the test block; May–June observations are included in training.

## Coverage

Every reference fault hour in this split has an original mechanism and component
label: 5,866 training, 577 validation and 1,686 test hours. Current-hour reconstruction
retains reasons for 3,165, 307 and 1,157 hours respectively. Thus all 529 previously
excluded test hours have episode-level reasons, although those are not necessarily
supported at each individual timestamp.

## Matched evaluation against original episode reasons

Both models below are newly fitted pilot models, not the deployed artifact.
Both return at least one mechanism and component. Evaluation assumes a known
reference fault and includes all 1,686 test fault hours.

| Training labels | Mechanism macro-F1 | Mechanism micro-F1 | Component macro-F1 | Component micro-F1 |
|---|---:|---:|---:|---:|
| Current-hour reasons | 57.82% | 66.60% | 45.65% | 64.09% |
| Original episode reasons | 58.57% | 73.69% | 61.30% | 80.75% |

On the previously excluded 529 hours, mechanism micro-F1 rises from 39.24% to
74.47%, and component micro-F1 from 32.73% to 74.73%.

Under the mixed output policy, mechanism scores are unchanged, but component
micro-F1 on all 1,686 hours is 60.85% for current-hour training and 68.25% for
episode training (component macro-F1 44.37% and 53.50%).

## Interpretation

Broader training labels improve agreement with the broader episode-level task,
especially on hours excluded by the current-hour reconstruction. This does not
mean the historical high conditional scores were scores on all fault hours.
Nor does it establish that episode labels are correct real-time explanations:
an episode can include reasons confirmed later or at only some timestamps.
Current-hour features include physical/stuck evidence also used in constructing
targets; these are reference-agreement metrics, not independent hardware diagnoses.
Macro-F1 averages only categories with positive test support. No binary gate is
applied in this conditional evaluation. Results are exploratory on an already
inspected holdout, and no deployment change is justified by this pilot alone.

Runner: `scripts/experiment_episode_reason_pilot.py`.
Detailed results, per-class support, selections and audit:
`data/eval/episode_reason_pilot_20260924/`.
Deployed reason model and July ledger hashes were unchanged. No report or dashboard
was edited, and the test process has finished.
