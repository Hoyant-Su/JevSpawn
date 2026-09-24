import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    target = Path(config['directory'])
    target.mkdir(parents=True, exist_ok=True)
    for record in config['files']:
        path = target / record['name']
        if path.exists() and path.stat().st_size == record['bytes']:
            continue
        partial = path.with_name(path.name + '.partial')
        url = f"https://huggingface.co/{config['repo']}/resolve/{config['revision']}/{config['subfolder']}/{record['name']}"
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--continue-at', '-',
                        '--connect-timeout', str(config['connect_timeout_seconds']),
                        '--max-time', str(config['file_timeout_seconds']),
                        '--speed-limit', str(config['minimum_bytes_per_second']),
                        '--speed-time', str(config['slow_window_seconds']),
                        '--output', str(partial), url], check=True)
        assert partial.stat().st_size == record['bytes'], record['name']
        partial.replace(path)
        print(json.dumps({'file': record['name'], 'bytes': path.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
