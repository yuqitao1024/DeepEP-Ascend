# Ascend EPv2 Phase 3F FP8 Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact E4M3 payload and four-byte scale-factor transport to synchronous Ascend EPv2 dispatch while retaining BF16 combine.

**Architecture:** Extend the existing dispatch token record and arguments with exact scale-factor geometry and strides. Keep the accepted dispatch transport/control protocol unchanged, validate FP8 collectively and at the C++ boundary, and copy payload/SF bitwise in the producer and epilogue.

**Tech Stack:** Python, PyTorch C++ extension, C++17, AscendC `.asc`, pytest/unittest, CANN 9.2.0, TaskQueue.

**Spec:** `docs/superpowers/specs/2026-08-19-ascend-epv2-fp8-runtime-design.md`

## Global Constraints

- Preserve all public EPv2 Python and C++ signatures.
- Support contiguous NPU `torch.float8_e4m3fn` payloads only.
- Support NPU float32 and packed UE8M0x4 int32 scale factors with four-byte packs.
- Preserve BF16 combine input/output and existing BF16 record geometry.
- Keep async events, communication-stream allocation, FP8 combine, and performance tuning out of scope.
- Do not change CUDA implementation files or execute GPU tests.
- NPU validation must use the repository TaskQueue policy.

---

### Task 1: FP8 Record Geometry And Runtime Admission

**Files:**
- Modify: `csrc/backends/ascend/elastic/layout.hpp`
- Modify: `csrc/backends/ascend/elastic/tiling.hpp`
- Modify: `csrc/backends/ascend/elastic/runtime.cpp`
- Modify: `tests/ascend/production_layout_probe.cpp`
- Modify: `tests/ascend/core_runtime_contract_probe.cpp`

**Interfaces:**
- Consumes: `CoreTilingInput.element_kind`, `num_scale_factor_packs`, and `scale_factor_pack_bytes`.
- Produces: `SymmetricWindowInput.scale_factor_bytes`; FP8 dispatch tiling with one-byte payloads, exact SF bytes, and BF16 combine records.

- [ ] **Step 1: Write failing layout and runtime probe checks**

Add an FP8 layout case equivalent to:

```cpp
auto fp8 = direct_input;
fp8.element_bytes = 1;
fp8.scale_factor_bytes = 8;
SymmetricWindowLayout fp8_layout{};
CHECK(build_symmetric_window_layout(fp8, &fp8_layout).ok());
CHECK(fp8_layout.dispatch_record_bytes < direct_layout.dispatch_record_bytes);
CHECK(fp8_layout.combine_record_bytes == direct_layout.combine_record_bytes);
```

Change the runtime probe so a valid FP8 dispatch reaches argument validation,
an FP8 dispatch without SF pointers is rejected, and FP8 combine remains
unsupported.

- [ ] **Step 2: Run probes and verify the expected failures**

Run:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py \
  -k 'production_layout or core_runtime'
```

Expected: failure because production layout rejects one-byte elements and the
runtime rejects all FP8 tiling before argument validation.

- [ ] **Step 3: Implement split dispatch/combine geometry**

Add `scale_factor_bytes` to `SymmetricWindowInput`, append it after dispatch
payload bytes, and calculate combine payload bytes with checked
`hidden * sizeof(uint16_t)`. Pass `tiling.token_layout.scale_factor_bytes` from
`build_core_tiling`. In runtime validation accept FP8 only for dispatch and
require input/output SF pointers exactly when FP8 is selected.

- [ ] **Step 4: Run focused and full host contract tests**

Run:

```bash
python3 -m pytest -q tests/ascend/test_core_operator_contract.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add csrc/backends/ascend/elastic/layout.hpp \
  csrc/backends/ascend/elastic/tiling.hpp \
  csrc/backends/ascend/elastic/runtime.cpp \
  tests/ascend/production_layout_probe.cpp \
  tests/ascend/core_runtime_contract_probe.cpp
git commit -m "feat: admit Ascend FP8 dispatch geometry"
```

### Task 2: Public Buffer Sizing And C++ Tensor Contract

**Files:**
- Modify: `csrc/backends/ascend/elastic_buffer.hpp`
- Modify: `csrc/backends/ascend/elastic/kernels.hpp`
- Modify: `tests/ascend/cann_transport_probe.cpp`
- Modify: `tests/ascend/test_python_api.py`

**Interfaces:**
- Consumes: FP8-capable layout and runtime validation from Task 1.
- Produces: buffer size hints for FP8; dispatch arguments with `sf_token_stride`, `sf_pack_stride`, `recv_sf_token_stride`, and `recv_sf_pack_stride` in element units.

- [ ] **Step 1: Write failing host API tests**

Update the fake extension so FP8 size calculation no longer raises. Add tests
that assert an Ascend FP8 size request reaches `_C.calculate_elastic_buffer_size`
with `use_fp8_dispatch=True`. Add C++ testing-probe cases for an FP8 hint and
for C++ rejection of invalid FP8/SF pairings.

- [ ] **Step 2: Run tests and verify FP8 is still gated**

Run:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py \
  tests/ascend/test_transport_contract.py
```

Expected: FP8 hint tests fail with the current `does not support FP8` gate.

- [ ] **Step 3: Implement sizing, validation, output allocation, and arguments**

Use the conservative sizing geometry:

```cpp
const auto element_bytes = use_fp8_dispatch ? 1U : 2U;
const auto sf_bytes = use_fp8_dispatch ?
    ((static_cast<std::uint64_t>(hidden) + 31U) / 32U) * 4U : 0U;
```

For dispatch, accept BF16 without SF or E4M3 with 2D float32/int32 SF. Validate
same device, token count, positive pack count/strides, and the conservative SF
bound. Allocate SF output with:

```cpp
const auto token_stride = use_tma_aligned_col_major_sf ? 1 : num_sf_packs;
const auto pack_stride = use_tma_aligned_col_major_sf ?
    align(output_tokens, 4) : 1;
recv_sf = torch::empty_strided(
    {output_tokens, num_sf_packs}, {token_stride, pack_stride}, sf->options());
```

Build FP8 tiling with exact pack geometry, populate SF pointers/strides, return
the narrowed SF tensor, and leave combine tiling BF16.

- [ ] **Step 4: Run focused host API tests**

Run:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py \
  tests/ascend/test_transport_contract.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add csrc/backends/ascend/elastic_buffer.hpp \
  csrc/backends/ascend/elastic/kernels.hpp \
  tests/ascend/cann_transport_probe.cpp tests/ascend/test_python_api.py
git commit -m "feat: expose Ascend FP8 dispatch runtime"
```

### Task 3: Strided SF Producer And Epilogue

**Files:**
- Modify: `csrc/backends/ascend/elastic/dispatch.asc`
- Modify: `tests/ascend/production_dispatch_state_probe.cpp`
- Modify: `tests/ascend/test_stub_source.py`

**Interfaces:**
- Consumes: SF pointers and element-unit strides from Task 2.
- Produces: bitwise input-record-output SF copies for normal, expanded, cached, and zero-padded dispatch.

- [ ] **Step 1: Add failing device/source contract checks**

Add checks that the launch ABI forwards all four SF strides and that zero
padding writes SF slots. Extend the production dispatch-state probe with a
small host helper for checked byte offsets:

```cpp
CHECK(scale_factor_byte_offset(2, 3, 5, 1, 4) == 52);
```

- [ ] **Step 2: Run the focused tests and observe missing stride behavior**

Run:

```bash
python3 -m pytest -q tests/ascend/test_stub_source.py \
  tests/ascend/test_core_operator_contract.py
```

Expected: failure because the kernel indexes SF as contiguous rows and does
not clear expanded SF padding.

- [ ] **Step 3: Implement pack-wise strided copies**

For each pack and byte, calculate input/output byte offsets from element-unit
strides and four-byte packs. Copy SF into the contiguous record in the
producer; write it through output strides in both regular and expanded
epilogues. Clear every SF pack in expanded alignment-only slots when zero
padding is requested. Forward the four strides through the `.asc` launch ABI.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m pytest -q tests/ascend/test_stub_source.py \
  tests/ascend/test_core_operator_contract.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add csrc/backends/ascend/elastic/dispatch.asc \
  tests/ascend/production_dispatch_state_probe.cpp \
  tests/ascend/test_stub_source.py
git commit -m "feat: transport strided Ascend FP8 scales"
```

### Task 4: Collective Python FP8 Preflight

**Files:**
- Modify: `deep_ep/buffers/elastic.py`
- Modify: `tests/ascend/test_python_api.py`

**Interfaces:**
- Consumes: public tuple input and cached `EPHandle`.
- Produces: normalized collective contract fields `x_dtype`, `sf_dtype`, `sf_shape`, `sf_layout`, and `column_major_sf` before C++ allocation/launch.

- [ ] **Step 1: Write failing Python behavior tests**

Add fake tensor strides and cases for float32 SF, int32 SF, noncontiguous SF,
column-major selection, missing SF, BF16 with SF, wrong SF dtype, wrong token
count, zero pack count, nonpositive stride, and asymmetric cross-rank contract
mismatch. Assert accepted calls reach the fake runtime with tuple output.

- [ ] **Step 2: Run tests and verify the BF16-only preflight failure**

Run:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py -k 'dispatch or fp8'
```

Expected: valid FP8 cases fail as `invalid_x_tensor` or
`unsupported_dispatch_mode`.

- [ ] **Step 3: Implement normalized FP8 preflight**

Add an SF-specific tensor contract that checks NPU device, rank, dtype,
shape, and positive strides without requiring contiguity. Accept exactly the
two payload/SF pairings from the spec, include normalized SF fields in the
collective contract, and pass `use_tma_aligned_col_major_sf` into preflight.

- [ ] **Step 4: Run Python and platform suites**

Run:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py tests/platform
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add deep_ep/buffers/elastic.py tests/ascend/test_python_api.py
git commit -m "feat: preflight Ascend FP8 dispatch"
```

### Task 5: Independent Production FP8 Matrix

**Files:**
- Create: `tests/ascend/production/run_fp8_dispatch_combine.py`
- Modify: `tests/ascend/test_python_api.py`
- Modify: `tests/ascend/benchmark/workloads.py`
- Modify: `tests/ascend/benchmark/runtime.py`
- Modify: `docs/ascend-design/epv2-ascend-roadmap.md`

**Interfaces:**
- Consumes: public `ElasticBuffer.dispatch` and BF16 `combine`.
- Produces: a rank-parameterized public NPU acceptance matrix and enables only synchronous FP8 benchmark cases.

- [ ] **Step 1: Add a failing source-contract test for the production matrix**

The contract test must parse the script and require public dispatch/combine,
independent `all_gather` reference construction, exact payload/SF checks,
normal/expanded/padded/cached/weighted/empty/asymmetric/`-1` cases, distributed
failure aggregation, repeated generations, and `finally`-protected teardown.

- [ ] **Step 2: Run the source-contract test and verify the script is absent**

Run:

```bash
python3 -m pytest -q tests/ascend/test_python_api.py -k fp8_matrix
```

Expected: failure because the production script does not exist.

- [ ] **Step 3: Implement the independent matrix and benchmark materialization**

Build exact expected payload/SF results only from gathered source tensors and
routes; never call production helpers from the reference. Exercise FP32 and
packed int32 factors, both SF output layouts, cached route reuse, BF16 combine
after dispatch, malformed inputs, 100 generations, and rank-derived expert
placement. Update benchmark materialization to quantize FP8 cases independently
and mark synchronous FP8 performance cases supported while leaving event and
comm-stream cases deferred.

- [ ] **Step 4: Run host source and benchmark tests**

Run:

```bash
python3 -m pytest -q tests/ascend tests/platform \
  tests/utils/test_ep_benchmark_core.py \
  tests/utils/test_ep_benchmark_manifest.py
```

Expected: all runnable tests pass; accelerator-only tests skip.

- [ ] **Step 5: Update roadmap status after runtime evidence exists**

Record implementation status first. Mark Phase 3F complete only after Task 6
passes the selected two-, four-, and eight-rank runtime matrix.

- [ ] **Step 6: Commit**

```bash
git add tests/ascend/production/run_fp8_dispatch_combine.py \
  tests/ascend/test_python_api.py tests/ascend/benchmark/workloads.py \
  tests/ascend/benchmark/runtime.py docs/ascend-design/epv2-ascend-roadmap.md
git commit -m "test: add Ascend FP8 runtime matrix"
```

### Task 6: Build, NPU Acceptance, And Final Regression

**Files:**
- Modify: `docs/ascend-design/epv2-ascend-roadmap.md`
- Verify without planned edits: all implementation and test files from Tasks 1-5.

**Interfaces:**
- Consumes: complete Phase 3F implementation and production matrix.
- Produces: fresh build, dependency, host, BF16 regression, and multi-NPU evidence.

- [ ] **Step 1: Run the complete runnable host suite**

```bash
python3 -m pytest -q tests/ascend tests/platform \
  tests/utils/test_ep_benchmark_core.py \
  tests/utils/test_ep_benchmark_manifest.py
```

- [ ] **Step 2: Submit a clean CANN production build through TaskQueue**

Use `pto-task-operations` to stage tracked sources, build with
`DEEP_EP_PLATFORM=ascend` and `DEEP_EP_ASCEND_TESTING=0`, import the extension,
and audit `ldd` output for CUDA, NCCL, and NVSHMEM dependencies.

- [ ] **Step 3: Run focused two-rank FP8 and BF16 matrices through TaskQueue**

Run the new FP8 matrix and existing barrier, BF16 dispatch, and BF16 combine
production scripts on one serialized two-NPU allocation.

- [ ] **Step 4: Run four-rank and eight-rank FP8 smoke matrices**

Use the same built artifact and rank-parameterized script. Confirm exact
payload/SF results, BF16 combine, repeated generation, and teardown.

- [ ] **Step 5: Run fresh final verification**

```bash
git diff --check origin/main...HEAD
python3 -m pytest -q tests/ascend tests/platform \
  tests/utils/test_ep_benchmark_core.py \
  tests/utils/test_ep_benchmark_manifest.py
```

Inspect `git status --short`, review every spec requirement against evidence,
and record any physical multi-host or performance limitation without claiming
it complete.

- [ ] **Step 6: Commit final fixes and evidence**

```bash
git add docs/ascend-design/epv2-ascend-roadmap.md
git commit -m "docs: record Ascend Phase 3F qualification"
```
