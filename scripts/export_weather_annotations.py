"""Publish only score-neutral notes; never promote experimental predictions."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.model.final_reason_codes import ROOT, JULY_GATE
from src.model.reason_code_rebuild import sha


def main():
    source = ROOT / 'data/eval/temperature_specialist_check/weather_annotations.parquet'
    destination = ROOT / 'data/eval/july_2026_weather_annotations'
    if destination.exists():
        raise FileExistsError(destination)
    gate_columns = ['station_id', 'hour_utc', 'random_probability', 'random_prediction']
    notes = pd.read_parquet(source, columns=gate_columns + ['weather_consistent_temperature', 'weather_note'])
    gate = pd.read_parquet(JULY_GATE, columns=gate_columns)
    pd.testing.assert_frame_equal(notes[gate_columns], gate)
    assert not notes.loc[notes.random_prediction.eq(0), 'weather_consistent_temperature'].any()
    destination.mkdir(parents=True, exist_ok=False)
    notes.to_parquet(destination / 'weather_annotations.parquet', index=False)
    (destination / 'manifest.json').write_text(json.dumps(dict(
        source=str(source.relative_to(ROOT)), source_sha256=sha(source), gate_sha256=sha(JULY_GATE),
        annotation_hours=int(notes.weather_consistent_temperature.sum()),
        decisions_and_probabilities_unchanged=True,
        meaning='Weather-consistent pattern, not a benign diagnosis. Existing alert retained.'), indent=2), encoding='utf-8')
    print(f'Published {notes.weather_consistent_temperature.sum()} score-neutral weather notes')


if __name__ == '__main__':
    main()
