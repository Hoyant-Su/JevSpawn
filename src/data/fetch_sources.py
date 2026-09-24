"""Download the small source files used by the paper's preparation scripts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

from project_paths import ROOT


def fetch(item, output, timeout):
    name, url = item
    with urllib.request.urlopen(url, timeout=timeout) as response:
        content = response.read()
    (output / name).write_bytes(content)
    return {'file': name, 'url': url, 'bytes': len(content)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/data/remote_files.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--files', nargs='+')
    args = parser.parse_args()
    sources = json.loads(args.manifest.read_text())
    selected = args.files if args.files is not None else sources
    args.output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(lambda name: fetch((name, sources[name]), args.output, args.timeout), selected))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
