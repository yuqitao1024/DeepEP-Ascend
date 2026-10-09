#!/usr/bin/env bash
set -euo pipefail

# NETLAYER_AB_LAYER selects the protocol used by this build:
#   0: UBC_CTP (NPU8P-compatible layer-0 data path)
#   1: UB_MEM (official DeepEP-Ascend layer-1 data path)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CANN_HOME=${CANN_HOME:-${ASCEND_HOME_PATH:?ASCEND_HOME_PATH or CANN_HOME is required}}
BUILD_DIR=${BUILD_DIR:-"$SCRIPT_DIR/build"}
LAYER=${NETLAYER_AB_LAYER:-0}
if [[ "$LAYER" != "0" && "$LAYER" != "1" ]]; then
    echo "NETLAYER_AB_LAYER must be 0 or 1" >&2
    exit 2
fi
ARCH_DIR=${ARCH_DIR:-"$CANN_HOME/$(uname -m)-linux"}
CXX=${CXX:-g++}
if [[ "$LAYER" == "1" ]]; then
    COMPILE_DEFINITIONS="-DNETLAYER_AB_USE_UB_MEM"
else
    COMPILE_DEFINITIONS="-DNETLAYER_AB_USE_UB_CTP"
fi

mkdir -p "$BUILD_DIR"

# CANN's ASC language support performs the device compilation, host stub
# generation, and mixed-object packaging.  This avoids treating the AICore ELF
# as a normal host object.
cmake \
  -G Ninja \
  -S "$SCRIPT_DIR" \
  -B "$BUILD_DIR/cmake" \
  -DASCEND_HOME_PATH="$CANN_HOME" \
  -DDEEP_EP_NETLAYER_AB_ARCH_ROOT="$ARCH_DIR" \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510 \
  -DCMAKE_CXX_FLAGS="$COMPILE_DEFINITIONS" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR/cmake" --target netlayer_ab

"$CXX" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -Werror \
  -I"$SCRIPT_DIR" \
  -I"$CANN_HOME/include" \
  -I"$ARCH_DIR/asc/include" \
  $COMPILE_DEFINITIONS \
  "$SCRIPT_DIR/launcher_main.cpp" \
  "$BUILD_DIR/cmake/netlayer_ab.so" \
  -Wl,-rpath,"$BUILD_DIR/cmake" \
  -L"$CANN_HOME/lib64" \
  -L"$ARCH_DIR/lib64" \
  -Wl,-rpath,"$CANN_HOME/lib64" \
  -Wl,-rpath,"$ARCH_DIR/lib64" \
  -lascendcl \
  -lhcomm \
  -o "$BUILD_DIR/netlayer_ab"

echo "built $BUILD_DIR/netlayer_ab"
