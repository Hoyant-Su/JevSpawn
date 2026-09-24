#!/usr/bin/env bash
set -euo pipefail
read -r workers devices seed entrypoint < <(../../environ/qwen35_spawn/bin/python - "$1" <<'PY'
import json
from pathlib import Path
import sys
import yaml

config = json.loads(Path(sys.argv[1]).read_text())
shared = yaml.safe_load(Path(config['shared_config']).read_text())
print(shared['runtime']['world_size'], ','.join(map(str, config['devices'])),
      shared['runtime']['seed'], config['entrypoint'])
PY
)
export PYTHONHASHSEED="$seed"
exec bash scripts/run.sh "$devices" -m torch.distributed.run --standalone \
    --nproc_per_node="$workers" "$entrypoint" \
    --config "$1"
