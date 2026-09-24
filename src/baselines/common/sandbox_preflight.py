import argparse
import json
from pathlib import Path

from baselines.common.sandbox_protocol import preflight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.environment.read_text())
    preflight(settings['sandbox'], args.directory)
    record = {'sandbox_preflight': 'passed', 'sandbox': settings['sandbox'],
              'environment': str(args.environment), 'scope': 'Real sandbox bootstrap before model loading'}
    (args.directory / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
