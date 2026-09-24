import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fetch_super_pages import fetch_rows


def fetch(item):
    path, url = item
    with urllib.request.urlopen(url, timeout=45) as response:
        content = response.read()
        headers = dict(response.headers)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {'file': str(path), 'url': url, 'bytes': len(content), 'headers': headers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--request-interval-seconds', type=float, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    requests = []
    for name, fields in [('medxpertqa_text', ['dev', 'test']), ('bright_pony', ['examples', 'documents'])]:
        source = config[name]
        for field in fields:
            url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{source[field]}"
            filename = f'{field}.parquet' if name == 'bright_pony' else Path(source[field]).name
            requests.append((args.output / name / filename, url))
    pending = [item for item in requests if not (args.resume and item[0].exists())]
    with ThreadPoolExecutor(config['download_workers']) as pool:
        for result in pool.map(fetch, pending):
            with (args.output / 'downloads.jsonl').open('a') as output:
                output.write(json.dumps(result) + '\n')
    fetch_rows(config['supergpqa'], args.output, args.request_interval_seconds)
    print(json.dumps({'requested_fixed_files': len(requests), 'downloaded_fixed_files': len(pending),
                      'total_bytes': sum(path.stat().st_size for path, _ in requests)}))


if __name__ == '__main__':
    main()
