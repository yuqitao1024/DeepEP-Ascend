# HCOMM team/window lifecycle patch

This directory archives the source patch required by repeated Ascend
`ElasticBuffer` lifecycles. Binary HCOMM artifacts are intentionally not
committed.

## Baseline and artifacts

- HCOMM repository baseline: `8c5d5ad081e763f981c237d8dfdb15faea292d6e`
- Patch: `hcomm-team-window-deregister.patch`
- Patch SHA256: `7394982ec1c5432b3fe15898974e64441ba5a24a2bf5c110e49bb758174a9329`
- Compiler: `GCC: (Do-Compiler V100R001C30B0032) 7.3.0`
- Patched `libhcomm.so` SHA256: `afb65298169b7810269322a32576429bcd67798a3336718a2642d2fb97332e77`
- Full package SHA256: `33432305ed72929415d8aef825fd31927ffbda87972bed541b21b3f43cad5da9`

Apply the patch from a clean checkout at the baseline commit:

```bash
git apply /path/to/hcomm-team-window-deregister.patch
```

## Fix boundary

The patch treats a registered memory set as a versioned communication-domain
generation:

- team and window destruction remove their `CommMems` tag bindings, including
  team synchronization memory, while retaining endpoint registrations needed
  by live channel objects;
- every alias of a deregistered window is removed and duplicate window handles
  are destroyed once;
- window deregistration is a two-phase operation: HCOMM teardown is marked
  complete before `CommMems` is unregistered, and owner/alias records are
  removed only after both steps succeed, so either failure can be retried;
- windows whose teardown has started are excluded from reuse and pending
  remote-memory exchange;
- team destruction stops on a window-deregistration error and retains the
  manager state required for a later retry;
- communicator destruction closes a shared recursive lifecycle gate before
  unloading binaries or releasing team resources, waits for in-flight team
  operations, and prevents new world-team, subteam, window, channel, and
  destruction operations from entering;
- world-team teardown makes its subteams unavailable immediately, and a team
  handle already removed by communicator teardown is an idempotent no-op
  rather than a second raw HCOMM destruction;
- newly allocated team synchronization memory is zero-initialized;
- endpoint registration lookup tracks the current handle by tag while retaining
  historical handles until `MyRank` teardown, preventing channel use-after-free;
- shared-jetty channel acquisition includes the memory version in its key; and
- the standard `HcclChannelAcquire` path maps
  `(engine, memoryVersion, logicalIndex)` to a physical channel generation and
  includes the memory version in its socket tag.

When the memory version changes, a new AIV URMA channel and socket are created.
Older physical generations remain owned by `EndpointPair` until communication
domain teardown. This is required because AIV URMA does not implement in-place
`UpdateMemInfo`; reusing the old channel would silently retain the first team's
synchronization buffer.

The lifecycle gate adds mutex and state fields to the concrete C++ `CollComm`
layout. The patched HCOMM library and all C++ consumers must therefore be
rebuilt together; mixing this library with consumers built against an older
`CollComm` layout is unsupported.

## Runtime overlay

The library was built and exercised on NPU8P through TaskQueue using this
isolated overlay:

```text
/home/pyptouser/yuqitao/Ascend/hcomm-deepep-window-fix-gcc73/hybrid/cann
```

Only `lib64/libhcomm.so` resolves to the patched GCC 7.3 build. Other overlay
entries resolve to the user-installed weekly HCOMM package. The weekly package
was not overwritten.

The validated full package is installed at:

```text
/home/pyptouser/yuqitao/Ascend/hcomm-deepep-window-fix-7394982e/cann
```

NPU8P tasks use the stable entry point
`/home/pyptouser/yuqitao/Ascend/hcomm-deepep-current/cann`. It resolves to the
versioned installation above. Weekly and experimental HCOMM installations are
not selected by DeepEP tasks.

## Validation record

- `task_20260816_051703_20128429487`: shared-channel memory-version and endpoint
  historical-handle regressions passed, 2/2.
- `task_20260816_054041_21644664861`: standard endpoint channel-generation RED
  test failed because the API was absent.
- `task_20260816_055207_22977648492`: standard endpoint generation reuse and
  failed-creation rollback regressions passed, 2/2.
- `task_20260816_055433_231065214872`: final HCOMM build completed with the
  compiler and library SHA256 recorded above.
- `task_20260816_055733_232816611172`: descriptor validation proved the second
  lifecycle changed channel sync buffer `0x12000003c000` to
  `0x1200000b1000`; three create/destroy cycles and 100 production barriers
  then passed on devices 6 and 7.
- `task_20260816_055958_233726719296`: Phase 2D `barrier-repeat` and `teardown`
  primitives passed against the same patched library.
- `task_20260816_060059_234502631235`: 100 production barriers plus injected,
  rank-qualified completion-timeout diagnostics passed.
- `task_20260816_060752_249507332449`: clean production extension build with
  `DEEP_EP_ASCEND_TESTING=0` passed Ascend 53/53, platform 15/15, and build
  10/10 tests, followed by another 100 two-rank barriers against the initial
  lifecycle patch.
- `task_20260816_063418_285283725074`: two-phase window-deregistration manager
  test failed at compile time because the retry state API was not implemented.
- `task_20260816_063655_28660119437`: the same manager state test passed after
  the retry state API was added.
- `task_20260816_064244_296893832236`: alias removal, deregistration state,
  endpoint generation, and shared-memory-version regressions passed, 6/6.
- `task_20260816_064748_303329514222`: the final HCOMM artifact passed compiler,
  library, and package checks with the `libhcomm.so` SHA256 recorded above.
- `task_20260816_065144_30427631477`: final devices 6 and 7 validation passed
  Ascend 53/53, platform 15/15, and build 10/10 tests, injected diagnostics with
  100 barriers per rank, and a clean `DEEP_EP_ASCEND_TESTING=0` run with another
  100 barriers per rank. The clean extension SHA256 is
  `afd502907cf06389a079d30344340217db1c5a120f37ccf46744096183d581e1`;
  `ldd` resolved the final patched HCOMM overlay and contained no CUDA, NCCL,
  or NVSHMEM dependency.
- `task_20260816_172307_23474721340`: the final team/window lifecycle manager
  suite passed 10/10, including close-gate rejection, in-flight cleanup
  serialization, parent/subteam teardown, retry state, and unknown-team
  idempotency.
- `task_20260816_172952_2492414446`: the coordinated final HCOMM rebuild and
  package completed with CANN GCC 7.3 and produced the `libhcomm.so` SHA256
  recorded above.
- `task_20260816_173926_27925920094`: final devices 6 and 7 validation after
  the lifecycle-gate, unknown-team idempotency, and retry-state fixes passed
  Ascend 53/53, platform 15/15, and build 10/10 tests. Both the injected and
  clean production extensions passed 100 barriers per rank; the clean
  extension SHA256 remained
  `afd502907cf06389a079d30344340217db1c5a120f37ccf46744096183d581e1`, and
  the task completed with exit code 0.
- `task_20260817_214332_36511077985`: rebuilt the full HCOMM run package from
  baseline `8c5d5ad` plus the archived patch; package SHA256 is recorded above.
- `task_20260817_214627_368375922298`: installed the package into the versioned
  user directory and reproduced the recorded patched `libhcomm.so` SHA256.
- `task_20260817_214749_36883025604`: the installed package passed ABI and
  dependency audits, build 11/11, platform 15/15, Ascend 78/78, 100 two-rank
  barriers, dispatch 14/14, and combine 23/23 on devices 6 and 7.
- `task_20260817_215327_370689110251`: promoted the validated versioned
  installation to the stable `hcomm-deepep-current` entry point.

The AArch64 mockcpp trampoline cannot intercept a shared-library-internal C
call reliably, so the endpoint test uses real generation lookup and real input
failure paths. Successful physical-channel creation is covered by the
serialized two-NPU descriptor and production tests.

The archived patch is byte-for-byte the final HCOMM `git diff` used to build
the recorded library.
