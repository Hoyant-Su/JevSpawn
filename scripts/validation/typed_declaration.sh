#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
read -r workers devices seed < <(../../environ/qwen35_spawn/bin/python - "$1" <<'PY'
import json
from pathlib import Path
import sys
import yaml

specification = json.loads(Path(sys.argv[1]).read_text())
shared = yaml.safe_load(Path(specification['shared_config']).read_text())
workers = shared['runtime']['world_size']
print(workers, ','.join(map(str, range(workers))), shared['runtime']['seed'])
PY
)
export PYTHONHASHSEED="$seed"
exec bash scripts/run.sh "$devices" -m torch.distributed.run --standalone \
    --nproc_per_node="$workers" tests/infra/typed_declaration_qualification.py \
    --specification "$1" --output "$2"
