# July reason-label exclusion audit — 24 September 2026

The audit rebuilt the saved current-hour targets exactly. No models, deployment
ledgers, dashboard or report were changed. The 843 excluded station-hours span
21 stations and overlap 163 accepted fault episodes. They represent 49.56% of
the 1,701 July reference fault hours; the evaluated 858 represent 50.44%.

## Original fired mechanisms on excluded hours (mutually exclusive combinations)

| Original reasons | Excluded hours |
|---|---:|
| Statistical anomaly only | 576 |
| Stuck/flatline only | 138 |
| Statistical anomaly and stuck | 67 |
| Physical breach and statistical anomaly | 26 |
| Physical breach and stuck | 23 |
| Physical breach only | 13 |
| Total | 843 |

All exclusions were reproduced as absent/failed same-hour evidence checks, not
unknown mechanism/component mappings. This establishes the computational cause,
not the correctness of every exclusion.

## Reconstructed check failures

457 hours had only failed statistical checks; 43 only absent statistical records;
76 had a combination of failed/absent statistical records across their channels.
These sum to the 576 statistical-only hours. Across all reason combinations,
533 unique excluded hours had at least one `context_not_outlier` failure, 186
had at least one absent statistical record, and 60 had ERA5 disagreement.
These last counts overlap and must not be summed.

138 hours had only failed causal stuck checks, and 13 only absent physical
breaches. Remaining hours had combinations of these failures. Stuck failures
include incomplete trailing observation windows as well as non-flat windows;
not every case can be described simply as future backfilling.

Concrete example: ITAHLI1 episode july2026_000857 runs from July 24 14:00 to
July 25 14:00 UTC. The first 23 hours (through July 25 12:00) have no retained
causal stuck reason; the final two retain a reason. This is a verified example
of the difference between event membership and causal confirmation timing.

The largest station contributions are IJANZO2 (192 hours), INUQAT9 (96),
ITAHLI1 (89), ITRIPO33 (71), and IJABAL15 (46).

## What is and is not established

The reason-evaluation script reconstructs July statistical evidence, while the
binary reference uses previously adjudicated episodes. The original freeze
manifest names `data/eval/july_2026_labels_external_seam` as its source package;
that directory is not present at the recorded path. This audit therefore does
not establish whether each statistical discrepancy is solely event-to-hour
mapping or partly a changed reference/preparation contract. Recover/locate the
original channel-hour evidence before declaring these exclusions necessary.

The reported reason-only scores are legitimate descriptions of the retained
label population, not established all-fault explanation performance. Retaining
this narrow evaluation requires explicit coverage and a description of the
target definition. It is not sufficient to justify all exclusions as avoiding
future information or to call the original reasons simply unknown.

Recommended diagnostic fix: reconcile each original event reason with its
original timestamped evidence and preserve that provenance; then choose an
explicit event-explanation or current-hour-evidence target independently of
model scores. Do not automatically restore every event reason at every hour,
discard original fault labels, or convert all missing labels to an unknown
class. Any target revision requires re-evaluation; none was performed here.

## Files

### Follow-up: physical evidence and original statistical provenance

All 62 excluded hours having a physical-breach reason (including combined-reason
hours) refer to pressure_trend_hpa. Their observed values range from -5.10 to
5.08, below the unchanged absolute hard threshold of 20. The archived statistical
score file also retains per-hour physical flags; inspected examples are false.

Verified event july2026_000003, station I90583612, July 3 15:00–19:00 UTC:
pressure trend is -5.08, 0, 0, 0, 20.32 respectively. The event carries
spike_impossible plus statistical_anomaly reasons, but the pressure hard breach
is at 19:00, not at the excluded 16:00–18:00 hours. This directly establishes
event-to-hour propagation for this example, rather than disappearance of a hard
limit or its raw evidence.

Repository/retired-file searches and git history lookup did not recover the
original july_2026_labels_external_seam package. Original adjudication summaries,
raw measurements, statistical scores/flags, and reconstructed statistical evidence
remain. Inspected automatic statistical adjudication rows state evidence was
verified but have empty context_outlier_hours; those summaries alone cannot
reconstruct the original per-hour contextual gate decisions. This is a provenance
gap, not proof that the original labels were wrong or that files were deleted.

`data/eval/july_reason_exclusion_audit_20260924/excluded_hours.csv` lists all 843
station/timestamp pairs and their source episode IDs, mechanisms, components,
failed-check categories and statistical failure reasons. `evidence_checks.csv`
contains per-channel details; `episodes.csv` contains intervals;
`breakdown.csv` contains station-level counts. Runner:
`scripts/audit_july_reason_exclusions.py`.
