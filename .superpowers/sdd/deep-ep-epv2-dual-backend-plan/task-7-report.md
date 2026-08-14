# Task 7 Report

## Status

Complete. The Ascend extension purity test now accounts for PyTorch's real
`CppExtension` CPU defaults before asserting DeepEP-owned dependencies.

## Files Changed

- `tests/build/test_platform_selection.py`
- `.superpowers/sdd/deep-ep-epv2-dual-backend-plan/task-7-report.md`

## Design

The test constructs a real dependency-free `torch.utils.cpp_extension.CppExtension`
baseline and removes only values present in that baseline from the Ascend
extension. Normalization covers all dependency-capable standard extension
fields present in this environment: `sources`, `include_dirs`, `libraries`,
`library_dirs`, `extra_compile_args`, `extra_link_args`, `extra_objects`,
`define_macros`, `runtime_library_dirs`, `depends`, `export_symbols`,
`swig_opts`, and `undef_macros`. It then asserts the exact DeepEP source,
include, Ascend macro, and host compile flags, empty non-owned fields, and
absence of CUDA, NCCL, NVSHMEM, CANN, and HCCL tokens from the normalized
dependency surface. The existing PyTorch-unavailable skip remains intact.

## Commands / Output

- `python3 -m unittest discover -s tests/build -p 'test_*.py' -v`
  - `Ran 5 tests in 0.070s`
  - `OK (skipped=1)`; the PyTorch-dependent test skipped because local PyTorch
    is unavailable.
- `git diff --check`
  - Passed with no whitespace errors.

## Commit

See the focused commit recorded after this report is written.

## Self-Review

- Production build code was not changed.
- PyTorch include, library, and library-directory values are derived from the
  real baseline rather than installation-specific paths.
- All required standard dependency fields are normalized and scanned.
- Remote PyTorch execution remains the final environment-specific validation
  because the local environment cannot exercise the non-skipped path.
