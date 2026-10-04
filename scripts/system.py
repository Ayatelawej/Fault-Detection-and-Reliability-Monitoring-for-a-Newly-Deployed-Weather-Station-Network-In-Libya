"""Small command index for the submission; importing it does not train models."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "package": ["scripts/package_reproduction.py"],
    "availability": ["scripts/build_reliability_foundations.py"],
    "labels": ["scripts/build_labels.py"],
    "features": ["scripts/rebuild_detection_features.py"],
    "dataset": ["scripts/build_hourly_dataset.py"],
    "binary": ["scripts/train_hourly_detection.py"],
    "binary-temporal": ["-m", "src.workflows.compare_binary_temporal"],
    "reasons": ["-m", "src.workflows.compare_reasons"],
    "reason-splits": ["-m", "src.workflows.compare_reason_splits"],
    "health": ["scripts/build_station_health.py"],
    "linear-selection": ["-m", "src.workflows.select_linear_forecasts"],
    "forecasts": ["-m", "src.workflows.compare_forecasts"],
    "stage": ["scripts/stage_final_selected_system.py"],
    "verify": ["scripts/verify_final_system_release.py"],
    "publish": ["scripts/publish_final_system_release.py"],
    "tables": ["scripts/build_final_selection_tables.py"],
    "figures": ["scripts/generate_report_assets.py", "--set", "current"],
    "dashboard": ["-m", "streamlit", "run", "scripts/run_dashboard.py"],
    "test": ["-m", "pytest", "-q"],
}

SETUP_HELP = r"""
Setup (Windows, Python 3.11; run from the repository root):
  py -3.11 -m venv .venv
  .venv\Scripts\python -m pip install -r requirements-pinned.txt

Restore the companion data/model ZIP into a fresh checkout:
  1. Obtain the trusted artifact package identified in config/artifact_release.json.
     Compare its SHA-256 with that manifest before using it.
  2. Check every artifact inside the ZIP:
     .venv\Scripts\python scripts/system.py package --verify C:\path\artifacts.zip
  3. Extract it into the repository root, preserving its data/ paths.
     Do not overwrite an existing working data tree. Some files also exist in
     the source checkout; compare those with the manifest before replacing them.
     Checksums detect changes, but do not authenticate the sender. Only load
     pickle/joblib models from a trusted source.

Run with the installed environment (no activation required):
  .venv\Scripts\python scripts/system.py test
  .venv\Scripts\python scripts/system.py verify
  .venv\Scripts\python scripts/system.py dashboard
Run tests and verification sequentially to limit memory use. The dashboard uses
saved predictions; retraining is not needed. Open the URL printed by Streamlit.

Reproduction notes:
  Keep the tracked outputs/figures/ files; audit tests check their presence.
  The artifact package preserves hourly inputs and reference labels, not raw
  acquisition or manual adjudication. Rebuilding five-minute stuck checks also
  needs the separate Mozn source folder, set through MOZN_FIVE_MIN_DIR.
  Training/release commands can be expensive and may refuse existing outputs.
  Rerun them in a separate reproduction copy, preserving the frozen originals.
  To create a new companion ZIP outside the repository:
    .venv\Scripts\python scripts/system.py package --output C:\path\new-artifacts.zip
"""


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python scripts/system.py COMMAND [arguments]\n")
        print("Commands: " + ", ".join(COMMANDS))
        print("Training/release commands are explicit and may refuse existing outputs.")
        print(SETUP_HELP)
        return 0
    if args[0] not in COMMANDS:
        raise SystemExit("Unknown command: " + args[0])
    return subprocess.call([sys.executable, *COMMANDS[args[0]], *args[1:]], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
