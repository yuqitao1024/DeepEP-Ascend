#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${ASCEND_HOME_PATH:?source CANN set_env.sh first}/set_env.sh"

python3 "$SCRIPT_DIR/run.py" "$@"
