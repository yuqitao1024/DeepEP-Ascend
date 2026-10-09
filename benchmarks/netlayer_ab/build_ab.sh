#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CANN_HOME=${CANN_HOME:-${ASCEND_HOME_PATH:?source CANN set_env.sh first}}

for layer in 0 1; do
    echo "==> Building netlayer ${layer}"
    NETLAYER_AB_LAYER="$layer" BUILD_DIR="$SCRIPT_DIR/build-layer${layer}" \
        CANN_HOME="$CANN_HOME" "$SCRIPT_DIR/build.sh"
done

echo "Built both scenarios:"
echo "  layer 0: $SCRIPT_DIR/build-layer0/netlayer_ab"
echo "  layer 1: $SCRIPT_DIR/build-layer1/netlayer_ab"
