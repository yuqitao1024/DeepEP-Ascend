#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BINARY=${BINARY:-"$SCRIPT_DIR/build/netlayer_ab"}
WORLD_SIZE=${WORLD_SIZE:-8}
ITERATIONS=${ITERATIONS:-20}
WARMUP=${WARMUP:-5}
TIMEOUT=${TIMEOUT:-600}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/results-layer0-matrix"}
PROFILE=${PROFILE:-1}
PROFILE_ITERATIONS=${PROFILE_ITERATIONS:-5}
PROFILE_WARMUP=${PROFILE_WARMUP:-1}

mkdir -p "$OUTPUT_DIR"
if [[ ! -x "$BINARY" ]]; then
    echo "missing $BINARY; run ./build_ab.sh first" >&2
    exit 2
fi

run_case() {
    local payload=$1
    local chunk=$2
    local name="payload-${payload}-chunk-${chunk}"
    echo "==> Netlayer 0: ${payload} bytes/peer, ${chunk} bytes/WQE"
    python3 "$SCRIPT_DIR/run.py" \
        --binary "$BINARY" \
        --world-size "$WORLD_SIZE" \
        --layer-index 0 \
        --payload-bytes "$payload" \
        --chunk-bytes "$chunk" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --timeout "$TIMEOUT" \
        --output "$OUTPUT_DIR/${name}.txt"
}

run_profile_case() {
    local payload=$1
    local chunk=$2
    local name="payload-${payload}-chunk-${chunk}"
    local output="$OUTPUT_DIR/${name}-msprof"
    rm -rf "$output"
    mkdir -p "$output"
    echo "==> Netlayer 0 msprof: ${payload} bytes/peer, ${chunk} bytes/WQE"
    msprof \
        --output="$output" \
        --application="python3 $SCRIPT_DIR/run.py --binary $BINARY --world-size $WORLD_SIZE --layer-index 0 --payload-bytes $payload --chunk-bytes $chunk --warmup $PROFILE_WARMUP --iterations $PROFILE_ITERATIONS --timeout $TIMEOUT" \
        --task-time=l0 \
        --ai-core=on \
        --ascendcl=on \
        --runtime-api=off \
        --ge-api=off \
        --hccl=off \
        --parse=on \
        --type=text >/dev/null
    python3 "$SCRIPT_DIR/parse_msprof.py" "$output" \
        --payload-bytes "$payload" \
        --peer-count $((WORLD_SIZE - 1)) \
        --output "$OUTPUT_DIR/${name}-msprof.txt"
}

run_case $((8 * 1024 * 1024)) $((256 * 1024))
run_case $((16 * 1024 * 1024)) $((1 * 1024 * 1024))
run_case $((64 * 1024 * 1024)) $((4 * 1024 * 1024))
run_case $((128 * 1024 * 1024)) $((8 * 1024 * 1024))
run_case $((256 * 1024 * 1024)) $((16 * 1024 * 1024))

cat "$OUTPUT_DIR"/payload-*-chunk-*.txt

if [[ "$PROFILE" == "1" ]]; then
    run_profile_case $((8 * 1024 * 1024)) $((256 * 1024))
    run_profile_case $((16 * 1024 * 1024)) $((1 * 1024 * 1024))
    run_profile_case $((64 * 1024 * 1024)) $((4 * 1024 * 1024))
    run_profile_case $((128 * 1024 * 1024)) $((8 * 1024 * 1024))
    run_profile_case $((256 * 1024 * 1024)) $((16 * 1024 * 1024))
fi
