# EP Benchmark Automation Final Fix Report

Date: 2026-08-20

Base commit: `78db61d0b5126d75b96c05603f33806e42e9c1d5`

Implementation commit: `f51be4c` (`benchmark: harden automation report identity`)

Status: PASS

## Changes By Finding

### 1. Comparison output cannot alias either JSON input

- `tests/benchmark/compare_ep.py` now resolves CUDA, Ascend, and output paths
  before loading either report.
- Normalized path equality rejects textual aliases such as `directory/../file`.
- `Path.samefile` rejects existing filesystem aliases, including hard links.
- Filesystem errors other than a nonexistent output path are propagated rather
  than treated as proof that two paths differ.
- The CLI emits a field-specific `--output must not alias --cuda` or
  `--output must not alias --ascend` error before any write.
- Four executable CLI tests cover normalized and hard-link aliases for both
  inputs and verify that both source byte streams and the alias remain unchanged.

### 2. Comparison Markdown includes the required workload table

- The existing backend workload formatting was extracted into one shared
  `_workload_metadata_lines` helper.
- Comparison Markdown now renders Provenance, Workload, then Detail.
- The canonical test asserts tokens `4096`, hidden `7168`, top-k `6`, experts
  `256`, seed `0`, warmups `30`, and iterations `30`.
- The same test directly asserts that the non-canonical warning is absent.

### 3. Schema-v2 execution identity attests multiple reduction

- Benchmark report `SCHEMA_VERSION` is now `2`.
- Every report contains the exact top-level object:

  ```json
  {"execution_protocol": {"allow_multiple_reduction": 0}}
  ```

  or the same object with integer value `1`.
- `validate_execution_protocol` requires exactly one key and an actual integer
  `0` or `1`; JSON booleans, missing fields, additional fields, and other values
  are rejected with a field-specific error.
- CUDA parity passes `args.allow_multiple_reduction` directly into report
  construction. Ascend runtime does the same.
- `BenchmarkReport.to_dict`, object-level `validate_comparable`, the runtime-free
  raw JSON comparator, strict automation validation, profile identification,
  comparison rows, and offline comparison now consume execution identity.
- Both fixed automation profiles require integer `1`. A matching pair of
  direct-run reports with value `0` still fails automation identification,
  validation, and comparison with
  `execution_protocol.allow_multiple_reduction`.
- The general direct-report comparator accepts either valid value only when the
  two reports match; it rejects mismatches as report identity mismatches.
- Synthetic fixtures and schema tests were updated to version 2.
- `allow_multiple_reduction` remains absent from `workload`, the workload
  fingerprint inputs, and `timing_protocol`.

Compatibility rationale: changing reduction mode changes how reduced combine
executes, so a formal performance report must attest it. It is not routing data
and does not describe timing mechanics, so a separate execution identity object
is more accurate than changing the workload fingerprint or timing protocol.
Schema-v1 automation reports are intentionally rejected. No pre-merge migration
or compatibility reader is provided.

### 4. Canonical warning absence is explicit

- `test_canonical_comparison_renders_workload_before_detail_without_warning`
  directly asserts that `NON-CANONICAL AUTOMATION VALIDATION` is absent from a
  canonical comparison.

### 5. Two-artifact publication failure injection

- Added failure injection at the real final `os.replace` boundary for:
  - failure of the first final promotion; and
  - failure of JSON promotion after Markdown has been promoted.
- Both cases verify no final `benchmark.json` or `benchmark.md` remains, the
  staging JSON remains complete, `workload.json` remains valid, `run.log`
  contains the diagnostic and exit code, and pending Markdown is removed.
- Both tests passed on their first run. Existing `run_ep.py` rollback behavior
  already met the contract, so production orchestration was intentionally left
  unchanged.

## Files Changed

| File | Change |
| --- | --- |
| `tests/benchmark/compare_ep.py` | Resolve paths and reject normalized/existing output aliases. |
| `tests/benchmark/report_markdown.py` | Schema-v2/profile validation, execution identity, shared workload rendering. |
| `tests/benchmark/test_automation.py` | Alias, workload, canonical warning, schema-v2, strict reduction, and publication tests. |
| `tests/ascend/benchmark/report.py` | Schema version 2, exact execution protocol, serialization and comparability. |
| `tests/ascend/benchmark/compare.py` | Supported schema and execution identity checks for raw reports. |
| `tests/ascend/benchmark/runtime.py` | Populate execution protocol from the Ascend CLI argument. |
| `tests/elastic/test_ep.py` | Populate execution protocol from the CUDA parity CLI argument. |
| `tests/ascend/test_benchmark_contract.py` | Schema-v2 fixtures and report/comparator identity tests. |
| `tests/ascend/benchmark/README.md` | Schema-v2, alias, identity, and compatibility documentation. |
| `docs/superpowers/specs/2026-08-18-ascend-epv2-benchmark-parity-design.md` | Factual schema and comparison contract corrections. |
| `.superpowers/sdd/ep-benchmark-automation-plan/final-fix-report.md` | This verification report. |

The `/tmp/ep-benchmark-automation-plan.md` file was not edited.

## TDD Evidence

### Output alias protection

RED:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'output_aliasing_either_input'
FFFF                                                                     [100%]
4 failed, 76 deselected in 25.18s
```

All four cases returned success before the fix; the normalized CUDA and Ascend
cases replaced their input paths.

GREEN:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'output_aliasing_either_input'
....                                                                     [100%]
4 passed, 76 deselected in 0.24s
```

### Comparison workload and canonical warning

RED:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'canonical_comparison_renders_workload'
F                                                                        [100%]
1 failed, 80 deselected in 8.78s
```

The failure was `ValueError: substring not found` for `## Workload`.

GREEN:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'canonical_comparison_renders_workload or backend_markdown'
...                                                                      [100%]
3 passed, 78 deselected in 11.29s
```

### Core schema and host comparator identity

RED:

```text
$ PYTHONPATH=$PWD pytest -q tests/ascend/test_benchmark_contract.py -k 'schema_v2 or execution_protocol or schema_v1'
FFFFFF                                                                   [100%]
6 failed, 34 deselected in 0.09s
```

Failures showed the missing factory argument and comparators that did not
reject execution mismatches or equal schema-v1 pairs.

GREEN:

```text
$ PYTHONPATH=$PWD pytest -q tests/ascend/test_benchmark_contract.py -k 'schema_v2 or execution_protocol or schema_v1'
......                                                                   [100%]
6 passed, 34 deselected in 0.04s
```

### Strict automation identification, validation, and comparison

RED:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'identify_profile_rejects_schema_v1 or disabled_reduction or exact_enabled_execution_protocol or matching_disabled_reduction'
FFFFFF                                                                   [100%]
6 failed, 83 deselected in 5.11s
```

Identification did not reject schema-v1 or disabled reports, while validation
still stopped on the obsolete schema version.

GREEN:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'identify_profile_rejects_schema_v1 or disabled_reduction or exact_enabled_execution_protocol or matching_disabled_reduction'
......                                                                   [100%]
6 passed, 83 deselected in 3.71s
```

### Exact integer protocol value

RED:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py tests/ascend/test_benchmark_contract.py -k 'boolean_execution_protocol or exact_enabled_execution_protocol'
.F..FF                                                                   [100%]
3 failed, 3 passed, 128 deselected in 0.15s
```

Python equality allowed JSON `true` to masquerade as integer `1` in all three
validation paths.

GREEN:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py tests/ascend/test_benchmark_contract.py -k 'boolean_execution_protocol or exact_enabled_execution_protocol'
......                                                                   [100%]
6 passed, 128 deselected in 0.06s
```

### Publication transaction characterization

The finding explicitly required production to remain unchanged if injected
tests exposed no defect. The first run was therefore recorded honestly as a
passing characterization rather than a fabricated RED:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'injected_publication_failures'
..                                                                       [100%]
2 passed, 89 deselected in 0.11s
```

## Final Verification

Focused coverage for all five findings:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py tests/ascend/test_benchmark_contract.py -k 'output_aliasing_either_input or canonical_comparison_renders_workload or identify_profile_rejects_schema_v1 or disabled_reduction or exact_enabled_execution_protocol or schema_v2_serializes or execution_protocol_mismatch or schema_v1_pair or boolean_execution_protocol or injected_publication_failures'
.......................                                                  [100%]
23 passed, 111 deselected in 15.30s
```

Mandated host suite:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py tests/ascend/test_benchmark_contract.py tests/elastic/test_ep_benchmark_parity.py tests/utils/test_ep_benchmark_manifest.py tests/utils/test_ep_benchmark_core.py tests/ascend/test_core_operator_contract.py tests/ascend/test_python_api.py tests/platform tests/build
308 passed, 3 skipped, 76 subtests passed in 100.59s (0:01:40)
```

Compilation:

```text
$ python3 -m compileall -q tests/benchmark tests/utils tests/ascend/benchmark tests/elastic/test_ep.py
exit 0, no output
```

Clean-environment CLI imports:

```text
$ env -i PATH="$PATH" python3 tests/benchmark/run_ep.py --help
exit 0; displayed the backend/profile/output/manifest help

$ env -i PATH="$PATH" python3 tests/benchmark/compare_ep.py --help
exit 0; displayed the CUDA/Ascend/output help
```

Runtime-free import boundary:

```text
$ PYTHONPATH=$PWD pytest -q tests/benchmark/test_automation.py -k 'comparison_modules_are_runtime_free'
1 passed, 91 deselected in 0.02s
```

Diff hygiene:

```text
$ git diff --check
exit 0, no output

$ git diff --cached --check
exit 0, no output
```

`ruff` was not installed. `command -v ruff` returned 1 and
`python3 -m ruff --version` returned `No module named ruff`; it was not installed.

## Live Inventory And Local Simulation

The clean-environment live inventory command exited 0. A separate standard
library assertion parsed the exact same host-only command output and printed:

```text
inventory total=144 supported=144 deferred=0
```

The fixture-based offline smoke flow generated schema-v2 CUDA and Ascend JSON,
invoked `tests/benchmark/compare_ep.py` in an empty child environment, and
asserted the output workload section, smoke warning, and row count:

```text
offline smoke comparison rows=720 workload=yes warning=yes
```

An additional invariant assertion printed:

```text
cases=144 unique=144 operations_per_case=5 rows=720 execution_protocol=1
```

It also asserted `allow_multiple_reduction` is absent from both `workload` and
`timing_protocol`.

## Self-Review

- Read the binding SPEC, `/tmp` plan, current implementations, affected tests,
  and complete reviewed diff context before edits.
- Confirmed all 144 case IDs and their enumeration order are unchanged.
- Confirmed five operations per case and 720 rows remain unchanged.
- Confirmed CUDA automatic SM/QP command behavior is unchanged.
- Confirmed Ascend runtime still uses fixed `num_sms=1`, `num_qps=0`.
- Confirmed profile manifests, routing inputs, workload fingerprints, timing
  protocol, correctness gates, and logical-byte formulas are unchanged.
- Confirmed comparison modules remain free of Torch, Torch-NPU, DeepEP, and the
  runtime adapter.
- Confirmed comparison alias rejection occurs before report loading or output.
- Confirmed both publication failure points leave no completed artifact pair.
- Confirmed current Phase 3F documentation still states 144/144/0 and preserves
  historical evidence.
- Confirmed no H800, NPU, torchrun, TaskQueue, build, dependency installation,
  cluster submission, model, or network-service operation was run.

## Concerns

- Hardware smoke and canonical acceptance were outside authorization and remain
  required before making performance claims.
- Schema-v1 benchmark automation artifacts are deliberately incompatible and
  must be regenerated with schema version 2.
- Ruff results are unavailable because Ruff is not installed in this workspace.

No unresolved host-side correctness concern was found in the final diff.
