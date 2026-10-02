#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")/.."
exec "${JEV_PYTHON:-python3}" -m demo.server --config demo/configs/demo.json "$@"
