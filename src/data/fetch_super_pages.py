import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def fetch_rows(config, output, request_interval):
    folder = output / 'supergpqa'
    pages = output / 'supergpqa_pages'
    folder.mkdir(parents=True, exist_ok=True)
    pages.mkdir(parents=True, exist_ok=True)
    remaining = sorted(index for index in config['evaluation_indices'] + config['feasibility_indices']
                       if not (folder / f'{index}.json').exists())
    while remaining:
        start = remaining[0]
        selected = [index for index in remaining if index - start < 100]
        parameters = {'dataset': config['repository'], 'config': config['viewer_config'],
                      'split': config['viewer_split'], 'offset': start, 'length': selected[-1] - start + 1}
        url = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(parameters)
        started = time.monotonic()
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
            headers = dict(response.headers)
        assert headers['x-revision'] == config['revision']
        raw_path = pages / f'{start}_{parameters["length"]}.json'
        raw_path.write_bytes(content)
        payload = json.loads(content)
        rows = {row['row_idx']: row for row in payload['rows']}
        for index in selected:
            assert not rows[index]['truncated_cells']
            (folder / f'{index}.json').write_text(json.dumps({**payload, 'rows': [rows[index]]}) + '\n')
        record = {'file': str(raw_path), 'url': url, 'bytes': len(content), 'headers': headers,
                  'selected_indices': selected}
        with (output / 'downloads.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        remaining = remaining[len(selected):]
        print(json.dumps({'remaining_selected_rows': len(remaining), 'page_rows': len(rows)}), flush=True)
        time.sleep(max(0, request_interval - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--request-interval-seconds', type=float, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())['supergpqa']
    fetch_rows(config, args.output, args.request_interval_seconds)


if __name__ == '__main__':
    main()
