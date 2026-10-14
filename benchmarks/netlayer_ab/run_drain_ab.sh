#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BINARY=${BINARY:-"$SCRIPT_DIR/build/netlayer_ab"}
WORLD_SIZE=${WORLD_SIZE:-8}
PAYLOAD_BYTES=${PAYLOAD_BYTES:-67108864}
CHUNK_BYTES=${CHUNK_BYTES:-4194304}
ITERATIONS=${ITERATIONS:-20}
WARMUP=${WARMUP:-5}
TIMEOUT=${TIMEOUT:-600}
PROFILE_ITERATIONS=${PROFILE_ITERATIONS:-5}
PROFILE_WARMUP=${PROFILE_WARMUP:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/results-drain-ab"}

mkdir -p "$OUTPUT_DIR"

run_mode() {
    local mode=$1
    python3 "$SCRIPT_DIR/run.py" \
        --binary "$BINARY" \
        --world-size "$WORLD_SIZE" \
        --layer-index 0 \
        --payload-bytes "$PAYLOAD_BYTES" \
        --chunk-bytes "$CHUNK_BYTES" \
        --iterations "$ITERATIONS" \
        --warmup "$WARMUP" \
        --timeout "$TIMEOUT" \
        --drain-mode "$mode" \
        --output "$OUTPUT_DIR/drain-${mode}.txt"
}

run_mode per-peer
run_mode post-all

run_profile_mode() {
    local mode=$1
    local output="$OUTPUT_DIR/drain-${mode}-msprof"
    rm -rf "$output"
    mkdir -p "$output"
    msprof \
        --output="$output" \
        --application="python3 $SCRIPT_DIR/run.py --binary $BINARY --world-size $WORLD_SIZE --layer-index 0 --payload-bytes $PAYLOAD_BYTES --chunk-bytes $CHUNK_BYTES --warmup $PROFILE_WARMUP --iterations $PROFILE_ITERATIONS --timeout $TIMEOUT --drain-mode $mode" \
        --task-time=l0 \
        --ai-core=on \
        --ascendcl=on \
        --runtime-api=off \
        --ge-api=off \
        --hccl=off \
        --parse=on \
        --type=text >/dev/null
    python3 "$SCRIPT_DIR/parse_msprof.py" "$output" \
        --payload-bytes "$PAYLOAD_BYTES" \
        --peer-count $((WORLD_SIZE - 1)) \
        --drain-mode "$mode" \
        --output "$OUTPUT_DIR/drain-${mode}-msprof.txt"
}

run_profile_mode per-peer
run_profile_mode post-all
cat "$OUTPUT_DIR"/drain-*.txt
