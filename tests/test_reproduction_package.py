import json
import zipfile
import pytest
from scripts.package_reproduction import build, inventory, verify


def test_package_round_trip(tmp_path):
    root = tmp_path / 'repo'
    (root / 'data').mkdir(parents=True)
    (root / 'data/example.txt').write_text('frozen example')
    config = dict(version='test', purpose='test', include=['data/example.txt'])
    package = tmp_path / 'artifacts.zip'
    manifest = build(root, config, package)
    assert verify(package) == manifest
    assert manifest['files'][0]['path'] == 'data/example.txt'
    with pytest.raises(FileExistsError):
        build(root, config, package)


def test_reject_outside_roots(tmp_path):
    with pytest.raises(ValueError):
        inventory(tmp_path, {'include': ['../secret']})


def test_corrupt_payload_rejected(tmp_path):
    path = tmp_path / 'bad.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data/a', 'x')
        archive.writestr('artifact-manifest.json', json.dumps({'files': [
            {'path': 'data/a', 'bytes': 1, 'sha256': 'wrong'}]}))
    with pytest.raises(ValueError, match='Checksum'):
        verify(path)
