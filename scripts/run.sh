#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
: "${PYTHONHASHSEED:?Set PYTHONHASHSEED from the experiment shared_config before launching.}"
exec bash ../../runtime/official_lats/launch.sh "$@"
