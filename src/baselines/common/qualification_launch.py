import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for phase in manifest['phases']:
        print(json.dumps({'phase': phase['name'], 'arguments': phase['arguments']}), flush=True)
        subprocess.run(['bash', 'scripts/run.sh', phase['devices'], *phase['arguments']], check=True)


if __name__ == '__main__':
    main()
