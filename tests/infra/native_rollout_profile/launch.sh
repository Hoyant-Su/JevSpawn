#!/usr/bin/env bash
set -euo pipefail
read -r workers devices seed < <(../../environ/qwen35_spawn/bin/python - "$1" <<'PY'
import json
from pathlib import Path
import sys
import yaml

config = json.loads(Path(sys.argv[1]).read_text())
spec = json.loads(Path(config['specification']).read_text())
shared = yaml.safe_load(Path(spec['shared_config']).read_text())
print(shared['runtime']['world_size'], ','.join(map(str, config['devices'])),
      shared['runtime']['seed'])
PY
)
export PYTHONHASHSEED="$seed"
exec bash scripts/run.sh "$devices" -m torch.distributed.run --standalone \
    --nproc_per_node="$workers" tests/infra/native_rollout_profile/profile_rollout.py \
    --config "$1"
