# Source checkout and replay release checklist

Public snapshot note (14 September 2026): this publication includes the system
source, tests and experiment notes, but intentionally omits README.md, REPO_MAP.md,
the report directory and the report-generation script pending author edits.
References below to those materials describe the separately retained local/private
release, not files available in this public checkout. Public history and previously
published result evidence are preserved; no private-backup history is merged.
The exact public snapshot passed 424 tests with one explicit replay integration
skip in 73.74 seconds under two-thread numerical limits. The skip is expected:
the six ignored replay inputs are not included in this source publication.

## Scope

The delivered UI is a frozen July 2026 replay, not a live-feed service. It reads
saved operational tables and performs no fitting or inference. Training, tuning,
legacy episode reason codes, blocked/chronological experiments, temperature
specialists, and incident-risk forecasts remain research/reproduction workflows.
Preserve their source and compact evidence; they are not startup prerequisites.

The canonical delivered report is
`docs/report/Ayat_Elawej_EC499_Report_FINAL.docx`. Report replacement, its backup,
and report-specific release notes are managed separately from source hygiene.
Do not publish duplicate local illustrated originals or private report backups.

## Source-checkout gate

- Use a dedicated Python 3.11 environment and install `requirements.txt`.
- Run `python -m pip check`. Requirements are mostly unpinned: this is a source
  dependency list, not an exact reproducibility lock. Preserve the release's
  Python/platform and resolved dependency inventory with the external archive.
  Do not assume newly resolved packages can load old serialized models.
- In PowerShell, set numerical thread limits before starting Python:

  ```powershell
  $env:OMP_NUM_THREADS = "2"
  $env:MKL_NUM_THREADS = "2"
  $env:OPENBLAS_NUM_THREADS = "2"
  $env:NUMEXPR_NUM_THREADS = "2"
  python -m pytest -q -ra
  ```

- Keep the tracked `data/merged/` dataset/registry and `data/processed/`
  availability/audit evidence. Tests that depend on tracked evidence must fail
  if that evidence is missing; do not skip them or regenerate it during cleanup.
- The real dashboard smoke test explicitly skips if one of its six required
  ignored replay inputs is absent. The skip lists missing paths. Synthetic
  dashboard tests still run; this skip is not proof of a deployable replay.
  Present but malformed inputs, mismatched gates, and app exceptions still fail.
- `pytest.ini` collects `tests/`, not executable research runners named
  `scripts/test_*.py`. The normal suite does not launch full training workflows.

## Restore the frozen replay

Obtain a matching frozen artifact archive from the project owner. No downloadable
release bundle or automated restoration command is currently specified here.
Restore files at their exact repository-relative paths; do not retrain merely
to make the smoke test run. All six required tables below are ignored by Git.

| Required path | Purpose |
| --- | --- |
| `data/eval/july_2026_health/station_health_scores_through_july.parquet` | Health history including pre-July context for causal projections. |
| `data/eval/july_2026_health_forecast/july_health_forecast_predictions.parquet` | Selected saved forecast policies. |
| `data/eval/one_hour_candidate/july_ef_hgb_binary_predictions.parquet` | Frozen selected binary detector ledger. |
| `data/eval/july_2026_features/statistical_anomaly_scores.parquet` | Detector-evidence scores. |
| `data/eval/july_2026_features/spatial_neighbors.csv` | Neighbour graph. |
| `data/eval/july_2026_reason_codes_mixed_v2/reason_code_predictions.parquet` | Active mixed-policy reason ledger; not the old threshold-only directory. |

The registry `data/merged/station_registry.csv` is tracked and required.
For the full delivered presentation, also restore the optional
`data/eval/july_2026_weather_annotations/weather_annotations.parquet` and its
`manifest.json`; without them, weather notes are omitted but alerts remain.

Restore companion manifests/audits with their tables, especially the active
reason `scoring_manifest.json`. Record archive provenance, release/commit ID,
relative filenames, sizes, and SHA-256 hashes before distribution. Verify hashes
against the owner's frozen inventory (PowerShell `Get-FileHash -Algorithm SHA256`).
The dashboard checks ledger/gate equality, not archive authenticity or every
manifest hash. Never substitute a different experiment's output based only on
matching filenames.

After restoration, run `python -m pytest tests/test_dashboard_app.py -q -ra`.
Require the app-start/hour-400 test to **pass, not skip**, then launch
`python -m streamlit run scripts/run_dashboard.py`. Check all 26 stations and the
Network, Station, and Evidence views. No model archive is needed just to replay.

## Additional reproduction artifacts (not needed just to replay)

- `data/model/reason_codes/final/`: `reason_heads.joblib`, `manifest.json`, and
  selection evidence. Load serialized models only from a trusted archive.
- `data/eval/july_2026_reason_codes/`: immutable source scores and manifest for
  `final_reason_codes.py july-mixed`. That command also requires the original
  frozen model, binary gate, and
  `data/eval/july_2026_reason_code_evaluation/aligned_reference.parquet` for
  retrospective metrics after assignment. It is not a truth-free archive copier.
- `data/hourly_detection/`, `data/features/`, `data/model/`, and relevant
  `data/eval/` experiment trees: exact tensors, split manifests, fitted detector
  and health policies, forecasts, and development evidence. Keep the health
  model manifest authoritative; a normal checkout does not contain these files.
- `data/external/`: archived exact-hour reference caches and optional public
  five-minute source data for feature rebuilding. These do not replace the
  provider's July observations/history needed for July rescoring. Follow the
  README's input boundary and explicit `--five-min-dir` instructions.

Use new versioned output directories for deliberate rebuilds. Keep legacy and
active ledgers and research evidence intact; never overwrite the frozen release.

## Git and confidentiality handoff

- Review the dirty tree and intended diff before staging; preserve other work.
  Do not add ignored artifacts, local caches, private notes, or `.env` variants.
  `.env.example` is the non-secret tracked template.
- Inspect secret-like patterns with filename-only output and review unexpected
  configuration files before publication. A pattern scan is not a complete
  secret audit of history, binaries, or external artifacts.
- Confirm the exact destination before publishing. At the 2026-09-14 inspection,
  `main` tracked `private-backup/main`; `@{push}` resolved there too. Cached refs
  showed 13 ahead/0 behind that upstream but 204 ahead/6 behind `origin/main`.
  Those were local-ref comparisons. A subsequent `git fetch private-backup`
  succeeded and did not change its tracking ref before the release commit.
  A default push does not publish to `origin`; do not force-push the divergent
  public history. This release uses a normal fast-forward push to
  `private-backup/main`; public `origin` is left untouched.

## Local validation record (2026-09-14)

The working-tree suite passed **425 tests in 88.94 seconds** with two-thread
numerical limits, including the real replay integration test. No release model
was retrained. A temporary source-checkout copy containing tracked data plus the
current tracked/untracked source and tests, but no ignored release artifacts,
passed **424 tests with 1 explicit dashboard skip in 76.48 seconds**, under the
same thread limits. An AST scan of 132 Python files found no syntax errors or
missing absolute local-module targets across 405 local imports. `git diff
--check` found no whitespace errors. These checks do not certify unexecuted
research workflows or a clean dependency installation.

Python was 3.11.9; key installed packages were pytest 9.0.3,
pandas 2.3.3, NumPy 1.26.4, PyArrow 24.0.0, SciPy 1.11.3, scikit-learn 1.3.2,
CatBoost 1.2.8, PyTorch 2.5.1+cu124, and Streamlit 1.58.0. This records the tested
environment, not a portable lock. The shared interpreter's `pip check` reported
an unrelated pre-existing `qiskit-aer` requirement on missing `qiskit`; neither is
a project requirement. Recheck dependencies in the dedicated release environment.
