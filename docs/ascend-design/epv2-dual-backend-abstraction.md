# EPv2 dual-backend abstraction

## Status

This document defines phase 1 of the Ascend 950 migration. Phase 1 adds a
compile-time platform boundary around the EPv2 `ElasticBuffer` API while keeping
the existing CUDA implementation intact. The Ascend backend is a compileable
stub in this phase. It does not implement communication or device kernels.

## Goals

Phase 1 must:

- preserve the current CUDA behavior, exported Python names, and kernel path;
- expose the same `deep_ep._C` extension name for CUDA and Ascend builds;
- select exactly one platform at compile time;
- define a stable C++ and PyBind boundary for EPv2 `dispatch` and `combine`;
- allow the Ascend extension to build without CUDA, NCCL, NVSHMEM, CANN, or
  `torch_npu`;
- allow Python to import the Ascend build without running CUDA JIT or NCCL
  initialization;
- return a clear phase-1 error when an unimplemented Ascend operation is used.

## Non-goals

Phase 1 does not:

- implement Ascend C kernels;
- integrate ACL, HCCL, or `torch_npu`;
- migrate the legacy `Buffer` API;
- migrate Engram, pipeline-parallel send/receive, context-parallel operations,
  or AGRS;
- change CUDA kernel algorithms, launch parameters, metadata layouts, or
  communication behavior;
- load CUDA and Ascend backends in the same process or shared object.

## Existing structure and constraints

The public Python class is `deep_ep.buffers.elastic.ElasticBuffer`. It delegates
most work to `deep_ep._C.ElasticBuffer`. The current C++ implementation in
`csrc/elastic/buffer.hpp` contains both host orchestration and CUDA-specific
details:

- CUDA streams and events;
- CUDA memory allocation and mapped host memory;
- NCCL Gin communicator and symmetric-memory setup;
- CUDA JIT runtime queries;
- device checks such as `Tensor::is_cuda()`;
- launch calls for dispatch, copy epilogue, combine, and reduction epilogue;
- CUDA-only experimental operations on the same class.

`csrc/python_api.cpp` also registers CUDA JIT and the legacy buffer
unconditionally. Python package initialization checks NCCL and initializes the
CUDA JIT before importing the public classes.

Because CUDA types cross the existing class boundary, phase 1 will not try to
make `csrc/elastic/buffer.hpp` platform-neutral internally. It will isolate the
file behind a compile-time selection layer. This keeps the CUDA implementation
unchanged while establishing the boundary needed by the later Ascend runtime.

## Platform selection

### Build setting

The build-facing setting is:

```text
DEEP_EP_PLATFORM=cuda|ascend
```

If the setting is absent, the build defaults to `cuda` for backward
compatibility.

The build translates the setting into exactly one C++ macro:

```text
DEEP_EP_PLATFORM_CUDA=1
DEEP_EP_PLATFORM_ASCEND=1
```

`csrc/platform/config.hpp` enforces the following rules with preprocessor
errors:

- defining both macros is invalid;
- defining neither macro is invalid inside C++ sources;
- each platform build has a stable string name, `"cuda"` or `"ascend"`.

The environment variable is a build-time input only. It does not select a
backend after `deep_ep._C` has been loaded.

### Extension identity

Both builds produce the same Python extension:

```text
deep_ep._C
```

A wheel or installation therefore targets one platform. Installing a second
platform build over the first replaces the extension rather than adding a
runtime-selectable backend.

The extension exports:

```python
_C.get_platform() -> str
```

Python code uses this value to choose platform-specific initialization and
helpers. It must not infer the platform from CUDA availability.

## Source layout

Phase 1 adds the following structure:

```text
csrc/
  platform/
    config.hpp
  elastic/
    api.hpp
    binding.hpp
    buffer.hpp
  backends/
    ascend/
      elastic_buffer.hpp
```

Responsibilities:

- `platform/config.hpp` validates macros and exposes the compiled platform.
- `elastic/api.hpp` is the only EPv2 registration entry included by
  `python_api.cpp`. It selects the platform implementation.
- `elastic/binding.hpp` owns the common EPv2 PyBind names and method bindings.
- `elastic/buffer.hpp` remains the CUDA implementation during phase 1.
- `backends/ascend/elastic_buffer.hpp` implements the same binding contract with
  phase-1 stubs.

The physical move of CUDA code into `backends/cuda/` is deferred. Moving a large
header while changing its registration path would increase regression risk
without improving the phase-1 interface.

## Registration architecture

`csrc/python_api.cpp` becomes platform-aware only at the registration level:

```text
register common module metadata
register get_platform
register common EPv2 API

if CUDA:
    register JIT API
    register legacy Buffer API
    register CUDA-only EPv2 auxiliary APIs

if Ascend:
    do not include or register CUDA JIT, legacy, NCCL, or NVSHMEM code
```

The platform selection remains inside C++ headers. `python_api.cpp` must not
contain separate implementations of buffer operations.

`EventHandle` registration currently lives in the legacy buffer registration
function. Phase 1 moves ownership of that Python-visible name into the selected
platform registration. The CUDA platform registers the existing event type, and
the Ascend platform registers its stub event type. Legacy registration consumes
the CUDA type but no longer owns its PyBind class registration. This prevents a
duplicate registration in CUDA builds and keeps `EventHandle` available when
Legacy is absent from Ascend builds.

## EPv2 API boundary

### Core public operations

The cross-platform phase-1 contract contains:

- `ElasticBuffer` construction;
- `destroy`;
- `dispatch`;
- `combine`;
- `calculate_elastic_buffer_size`;
- `get_physical_domain_size`;
- `get_logical_domain_size`;
- `get_comm_stream` or its phase-1 placeholder representation;
- `barrier`, as a runtime primitive required by later initialization;
- `EventHandle`, sufficient to preserve Python signatures;
- `get_platform`.

The `dispatch` and `combine` PyBind argument names, default values, and return
arity must match the current CUDA binding. Python callers must not need a
platform-specific call signature.

The common binding layer is the contract. It binds a selected implementation
type rather than requiring both implementations to inherit from a virtual base
class. A single-platform shared object does not need runtime polymorphism.

### CUDA-only operations

The following methods remain available in CUDA builds but are outside the
cross-platform phase-1 contract:

- `engram_write` and `engram_fetch`;
- `pp_set_config`, `pp_send`, and `pp_recv`;
- AGRS session methods and `all_gather`;
- all legacy `Buffer` methods;
- NCCL communicator creation and destruction helpers;
- CUDA JIT initialization and compilation helpers.

For compatibility, the CUDA build must keep exporting these names exactly as it
does before the refactor.

The Ascend build does not expose low-level NCCL or CUDA JIT functions. At the
Python wrapper level, all existing CUDA-only high-level methods remain present.
They fail before accessing a missing `_C` symbol.

## Ascend phase-1 stub

The Ascend class must match the common binding contract and avoid CUDA headers
and libraries.

### Construction

Construction validates ordinary scalar arguments that do not require a device
runtime. It stores enough state for platform and domain queries. It does not
allocate device memory, create a communicator, or initialize a stream.

The constructor's current `int64_t nccl_comm` argument becomes an `int64_t
comm_handle` in the common contract without changing its Python position. CUDA
continues to interpret the value as an NCCL communicator. The Ascend phase-1
stub requires the value to be zero and otherwise rejects it. The later HCCL
runtime will define the Ascend handle representation behind this integer ABI.
Python callers never construct this value directly; the platform helper owns
it.

The normal Python constructor calls `calculate_elastic_buffer_size` when
`num_bytes` is absent. Since the Ascend size calculation is not implemented in
phase 1, Ascend stub construction requires an explicit positive `num_bytes`.
Omitting it reports the buffer-size phase-1 error before construction.

### Method behavior

`dispatch`, `combine`, `barrier`, and runtime-dependent stream/event operations
raise Python `NotImplementedError` with this format:

```text
DeepEP Ascend backend: <operation> is not implemented in phase 1
```

The C++ implementation sets `PyExc_NotImplementedError` with
`PyErr_SetString(...)` and then throws `pybind11::error_already_set`. It does not
use a nonexistent pybind11 exception wrapper.

`destroy` is idempotent for the stub and releases any host-only state.

`get_physical_domain_size` and `get_logical_domain_size` raise the phase-1 error.
Rank count alone is not enough to infer an Ascend hardware or logical topology.

`calculate_elastic_buffer_size` throws a phase-1 error because the correct size
depends on the Ascend metadata layout and communication runtime. Returning the
CUDA size would turn an implementation detail into a false platform contract.

### Event and stream types

CUDA `EventHandle` and `at::cuda::CUDAStream` cannot appear in headers compiled
for Ascend. The common binding therefore treats event dependencies and stream
results as platform-owned binding types:

- CUDA keeps its existing event behavior and returned `torch::Stream` object.
- Ascend registers an `EventHandle` stub with the same Python-visible name.
- calling `current_stream_wait` on the Ascend event stub reports the phase-1
  error;
- `get_comm_stream` on the Ascend buffer reports the phase-1 error.

The common C++ layer does not attempt to create a universal native stream type.
Python-visible compatibility is required; native CUDA and future Ascend stream
representations remain backend-owned.

## Python package behavior

Package initialization follows this sequence:

```text
import deep_ep._C
platform = _C.get_platform()

if platform == "cuda":
    check NCCL shared object
    initialize CUDA JIT
    import CUDA legacy and elastic helpers

if platform == "ascend":
    skip NCCL checks
    skip CUDA discovery and JIT initialization
    import the common ElasticBuffer wrapper
```

The public name `deep_ep.ElasticBuffer` remains unchanged.

Platform-specific operations live behind a small Python platform helper rather
than repeated `_C.get_platform()` checks. The helper owns:

- the compiled platform name;
- synchronization hooks;
- event conversion;
- stream conversion;
- communicator-handle acquisition;
- device-type validation.

During phase 1, the Ascend helper does not import `torch_npu`. Runtime-dependent
methods raise the standard phase-1 error.

The Python `ElasticBuffer` constructor currently obtains an NCCL handle before
calling `_C.ElasticBuffer`. That path must be conditional. CUDA continues to use
the existing NCCL helper. Ascend passes zero and does not call any `_C` NCCL
function.

CUDA-only Python methods check platform support before calling the extension.
Users therefore receive a `NotImplementedError` naming the unsupported feature,
not an `AttributeError` caused by a missing binding.

## Build-system changes

### setuptools

`setup.py` selects the extension type and source set from `DEEP_EP_PLATFORM`:

- CUDA uses the current `CUDAExtension`, source files, include paths, link
  libraries, architecture flags, and persistent environment settings.
- Ascend uses `CppExtension` with `csrc/python_api.cpp` and header-only stub
  sources. It does not discover NVSHMEM or NCCL and does not add CUDA include or
  link paths.

The CUDA path remains the default. Refactoring must not change the existing CUDA
build summary or environment-variable behavior except for reporting the selected
platform.

### CMake

CMake gains a cached `DEEP_EP_PLATFORM` string with allowed values `cuda` and
`ascend`.

- CUDA keeps CUDA as a project language and retains the current dependencies.
- Ascend configures a C++-only project with Torch and pybind11.
- each branch defines one platform macro on the `_C` target;
- platform-invalid source files are not parsed or linked.

CMake is a supported structural check even though setuptools remains the normal
installation path.

## Device-kernel boundary for later phases

Phase 1 does not implement kernels, but the abstraction must leave room for five
Ascend kernel stages behind `dispatch` and `combine`:

```text
dispatch
  deterministic prologue (optional)
  communication and routing
  copy/layout epilogue

combine
  reverse communication
  reduction/layout epilogue
```

These are backend implementation stages, not five new Python APIs. Metadata
layout is backend-owned unless a field is part of the existing Python
`EPHandle` contract. The future Ascend backend may fuse or split these stages as
needed, provided it preserves public tensor shapes, dtypes, handle semantics,
and ordering guarantees.

## Error handling

Configuration errors fail early:

- invalid `DEEP_EP_PLATFORM` fails in the build script;
- missing or conflicting platform macros fail in preprocessing;
- importing an extension whose platform metadata is unknown fails during Python
  initialization.

Unsupported Ascend runtime calls use `NotImplementedError`. Invalid common
arguments use the same assertion or exception category as CUDA when validation
does not require a device runtime.

No Ascend stub may silently return empty tensors, CUDA-derived size estimates,
fake topology values, or a successful no-op for communication.

## Testing

### Platform configuration tests

Compile small translation units that include `platform/config.hpp` and verify:

- CUDA-only macro succeeds and reports `"cuda"`;
- Ascend-only macro succeeds and reports `"ascend"`;
- both macros fail compilation;
- no macro fails compilation.

### Binding contract tests

For both built extensions:

- `_C.get_platform()` returns the selected platform;
- `_C.ElasticBuffer` exists;
- the core method names exist;
- Python signatures and default arguments remain aligned where PyBind exposes
  them;
- `EventHandle` remains importable.

The expected core API list is stored once in a test helper so CUDA and Ascend
checks cannot drift independently.

### Ascend stub tests

In a Python environment with CPU PyTorch but without CUDA, NCCL, NVSHMEM, CANN,
or `torch_npu`:

- build and import the Ascend extension;
- import `deep_ep` without CUDA initialization;
- verify the platform result;
- construct the host-only stub with valid scalar arguments;
- pass an explicit positive `num_bytes` and a zero communication handle during
  construction;
- verify that `dispatch`, `combine`, `barrier`, stream access, and buffer-size
  calculation return the specified `NotImplementedError`;
- verify that CUDA-only high-level methods return a named unsupported-feature
  error rather than missing-symbol failures.

### CUDA regression tests

On the existing CUDA test environment:

- build with no new environment variable and confirm the CUDA default;
- build with `DEEP_EP_PLATFORM=cuda`;
- compare exported symbols and PyBind names with the pre-refactor extension;
- run the existing EPv2 dispatch/combine correctness tests;
- run the current import, JIT initialization, legacy, and experimental API
  smoke tests used by the project;
- compare representative dispatch/combine performance to detect accidental host
  path changes. Phase 1 introduces no intentional performance difference.

## Acceptance criteria

Phase 1 is complete when all of the following are true:

1. The default build is CUDA and existing CUDA commands remain valid.
2. Exactly one platform macro is present in every C++ build.
3. CUDA exports all existing APIs plus `get_platform()`.
4. The CUDA EPv2 correctness suite passes without code-path or metadata changes.
5. The Ascend extension builds without CUDA, NCCL, NVSHMEM, CANN, or `torch_npu`.
6. The Ascend package imports without executing CUDA initialization.
7. Both extensions expose the common EPv2 class and method names.
8. Ascend runtime-dependent methods fail with the documented phase-1 error.
9. CUDA-only high-level methods fail clearly on Ascend rather than reaching a
   missing extension symbol.
10. The source tree has one platform selection point for EPv2 registration and
    does not spread platform conditionals through the CUDA kernel implementation.

## Follow-up work

After phase 1, the Ascend implementation can proceed behind the fixed boundary:

1. add the Ascend runtime, stream/event, memory, and HCCL context;
2. define Ascend metadata and workspace layouts;
3. implement the dispatch stages in Ascend C;
4. implement the combine stages in Ascend C;
5. validate functional parity on Ascend 950;
6. tune communication, AICore occupancy, and overlap for the target topology.

Each follow-up may refine backend-owned internals. Changes to the common Python
signature or handle semantics require a separate design review.
