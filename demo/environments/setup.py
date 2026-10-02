import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import urllib.request

from demo.environments.common import ROOT


def acquire(entry, timeout_seconds):
    with urllib.request.urlopen(entry['url'], timeout=timeout_seconds) as response:
        data = response.read()
    destination = ROOT / 'external' / entry['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return entry['path']


def main():
    parser = argparse.ArgumentParser(description='Acquire the exact upstream environment sources.')
    parser.parse_args()
    manifest = json.loads((ROOT / 'sources.json').read_text())
    with ThreadPoolExecutor(manifest['acquisition']['workers']) as pool:
        for path in pool.map(partial(acquire, timeout_seconds=manifest['acquisition']['timeout_seconds']), manifest['files']):
            print(path)


if __name__ == '__main__':
    main()
