"""Return schema-constrained decisions from frozen Qwen weights without text decoding."""

import argparse
import json
from pathlib import Path

from jev_spawn.backend import Backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--mode', choices=['shared', 'streamed', 'independent'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    backend = Backend(json.loads(args.config.read_text()))
    result = backend.score_fields([json.dumps(state, ensure_ascii=False) for state in request['states']],
                                  request['fields'], mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
