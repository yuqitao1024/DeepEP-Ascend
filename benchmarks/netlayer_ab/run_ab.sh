#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
: "${ASCEND_HOME_PATH:?source CANN set_env.sh first}"

PAYLOAD_BYTES=${PAYLOAD_BYTES:-$((256 * 1024 * 1024))}
CHUNK_BYTES=${CHUNK_BYTES:-$((16 * 1024 * 1024))}
ITERATIONS=${ITERATIONS:-20}
WARMUP=${WARMUP:-5}
WORLD_SIZE=${WORLD_SIZE:-8}
TIMEOUT=${TIMEOUT:-600}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/results-ab"}

mkdir -p "$OUTPUT_DIR"
for layer in 0 1; do
    binary="$SCRIPT_DIR/build-layer${layer}/netlayer_ab"
    if [[ ! -x "$binary" ]]; then
        echo "missing $binary; run ./build_ab.sh first" >&2
        exit 2
    fi
    echo "==> Running netlayer ${layer}: ${PAYLOAD_BYTES} bytes/peer, ${CHUNK_BYTES} bytes/WQE"
    "$SCRIPT_DIR/run.py" \
        --binary "$binary" \
        --world-size "$WORLD_SIZE" \
        --layer-index "$layer" \
        --payload-bytes "$PAYLOAD_BYTES" \
        --chunk-bytes "$CHUNK_BYTES" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --timeout "$TIMEOUT" \
        --output "$OUTPUT_DIR/layer${layer}.txt"
done

python3 "$SCRIPT_DIR/compare_ab.py" \
    --layer0 "$OUTPUT_DIR/layer0.txt" \
    --layer1 "$OUTPUT_DIR/layer1.txt" \
    --output "$OUTPUT_DIR/summary.txt"
cat "$OUTPUT_DIR/summary.txt"
