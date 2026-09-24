#!/usr/bin/env bash
set -euo pipefail
seed=$(../../environ/qwen35_spawn/bin/python - "$1" <<'PY'
import json
from pathlib import Path
import sys
import yaml

settings = json.loads(Path(sys.argv[1]).read_text())
seeds = {yaml.safe_load(json.loads((Path(source['run']) / 'protocol.json').read_text())[
    'shared_config_text'])['runtime']['seed'] for source in settings['runs']}
seed, = seeds
print(seed)
PY
)
export PYTHONHASHSEED="$seed"
exec bash scripts/run.sh '' tests/infra/native_parallel_evaluation/qualify.py --config "$1"
