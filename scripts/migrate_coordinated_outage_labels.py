"""Remove timing subtypes without altering outage membership or durations."""
from pathlib import Path
import sys
import shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.config.paths import AVAILABILITY_EVENTS_PATH, NETWORK_OUTAGE_WINDOWS_PATH, STATION_RELIABILITY_SUMMARY_PATH
from src.availability.build_station_reliability_summary import write_station_reliability_summary


def main():
    backup = ROOT / 'data/eval/coordinated_outage_migration_20260920'
    backup.mkdir(exist_ok=False)
    paths = [AVAILABILITY_EVENTS_PATH, NETWORK_OUTAGE_WINDOWS_PATH, STATION_RELIABILITY_SUMMARY_PATH]
    for path in paths:
        shutil.copy2(path, backup / path.name)
    events = pd.read_parquet(AVAILABILITY_EVENTS_PATH)
    windows = pd.read_csv(NETWORK_OUTAGE_WINDOWS_PATH)
    mapping = {'network_midnight': 'coordinated', 'network_other': 'coordinated'}
    for frame, path in [(events, AVAILABILITY_EVENTS_PATH), (windows, NETWORK_OUTAGE_WINDOWS_PATH)]:
        original = frame.copy()
        frame['outage_class'] = frame.outage_class.replace(mapping)
        pd.testing.assert_frame_equal(frame.drop(columns='outage_class'), original.drop(columns='outage_class'))
        if path.suffix == '.parquet':
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
    write_station_reliability_summary()
    print(events.outage_class.value_counts().to_string())
    print(f'{len(windows)} coordinated windows retained; backups: {backup}')


if __name__ == '__main__':
    main()
