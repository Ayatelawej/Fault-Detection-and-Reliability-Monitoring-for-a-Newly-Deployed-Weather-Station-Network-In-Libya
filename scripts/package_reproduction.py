"""Build/verify an allowlisted artifact ZIP; never modifies source artifacts."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        result.update(chunk)
    return result.hexdigest()


def inventory(root, config):
    paths = set()
    for name in config['include']:
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()) or not name.startswith('data/'):
            raise ValueError(f'Unsafe artifact root: {name}')
        if not target.exists():
            raise FileNotFoundError(target)
        for path in ([target] if target.is_file() else target.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'Symlink not permitted: {path}')
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                paths.add(path)
    return sorted(paths)


def build(root, config, output):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError('Place the artifact package outside the repository.')
    files = inventory(root, config)
    rows = []
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            with path.open('rb') as source:
                checksum = digest(source)
            rows.append(dict(path=name, bytes=path.stat().st_size, sha256=checksum))
            archive.write(path, name)
        manifest = dict(version=config['version'], purpose=config['purpose'], files=rows)
        archive.writestr('artifact-manifest.json', json.dumps(manifest, indent=2))
    verify(output)
    return manifest


def verify(package):
    with zipfile.ZipFile(package) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP entries')
        manifest = json.loads(archive.read('artifact-manifest.json'))
        expected = {'artifact-manifest.json'}
        for row in manifest['files']:
            name = row['path']
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name or not name.startswith('data/'):
                raise ValueError(f'Unsafe archive path: {name}')
            if name in expected:
                raise ValueError(f'Duplicate manifest path: {name}')
            expected.add(name)
            if archive.getinfo(name).file_size != row['bytes']:
                raise ValueError(f'Size mismatch: {name}')
            with archive.open(name) as source:
                if digest(source) != row['sha256']:
                    raise ValueError(f'Checksum mismatch: {name}')
        if set(names) != expected:
            raise ValueError('Unexpected archive contents')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path, help='New ZIP outside repository')
    group.add_argument('--verify', type=Path, help='Verify every stored artifact hash')
    args = parser.parse_args()
    if args.verify:
        manifest = verify(args.verify)
    else:
        config = json.loads((ROOT / 'config/reproduction_package.json').read_text())
        manifest = build(ROOT, config, args.output)
    print(f"Verified {len(manifest['files'])} artifacts: {manifest['version']}")


if __name__ == '__main__':
    main()
