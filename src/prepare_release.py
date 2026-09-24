"""Assemble authored experiment sources and paper assets in a separate directory."""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

from project_paths import ROOT


def portable(value, key, source, settings):
    if isinstance(value, dict):
        return {name: portable(item, name, source, settings) for name, item in value.items()
                if name not in settings['omit_keys']}
    if isinstance(value, list):
        return [portable(item, key, source, settings) for item in value]
    if key in settings['config_values']:
        return settings['config_values'][key]
    if isinstance(value, str) and value in settings['model_paths']:
        return settings['model_paths'][value]
    if isinstance(value, str) and value.startswith(str(source.parents[1]) + '/'):
        return os.path.relpath(value, source)
    if isinstance(value, str) and value.startswith(source.parents[1].name + '/'):
        return os.path.relpath(source.parents[1] / value.split('/', 1)[1], source)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection', type=Path, default=ROOT / 'configs/schema' / 'release_files.json')
    args = parser.parse_args()
    source = ROOT
    settings = json.loads(args.selection.read_text())
    settings['model_paths'] = {
        json.loads((source / filename).read_text())[key]: settings['config_values'][key]
        for filename, key in settings['model_configs'].items()}
    paths = sorted({path for pattern in settings['files'] for path in source.glob(pattern) if path.is_file()})
    for path in paths:
        target = args.output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            value = portable(json.loads(path.read_text()), '', source, settings)
            target.write_text(json.dumps(value, indent=2) + '\n')
        else:
            shutil.copyfile(path, target)
    for name in settings['repo_files']:
        target = args.output.parents[1] / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.parents[1] / name, target)
    measurements = [source / name for name in settings['measurements']]
    archive_path = args.output / 'results/measurements.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for path in measurements:
            if path.suffix == '.jsonl':
                contents = ''.join(json.dumps(portable(json.loads(line), '', source, settings)) + '\n'
                                   for line in path.read_text().splitlines())
            else:
                contents = json.dumps(portable(json.loads(path.read_text()), '', source, settings), indent=2) + '\n'
            data = contents.encode()
            entry = tarfile.TarInfo(str(path.relative_to(source)))
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    print(json.dumps({'files': len(paths), 'measurement_files': len(measurements),
                      'measurement_archive_bytes': archive_path.stat().st_size, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
