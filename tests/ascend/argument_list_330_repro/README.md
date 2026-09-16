# Ascend compiler error 330: kernel argument-list repro

This is a minimal host-runtime + ASC kernel example derived from the dispatch
launch change in commits `987cd6f` and `03e2817`.

`packed_kernel` passes a large ABI as `KernelArguments` and keeps the launch
thread count separate. To reproduce the old runtime behavior, configure with
`-DDEEP_EP_REPRO_LEGACY_ARGS=ON`; the host wrapper will instead launch
`legacy_kernel`, whose equivalent fields are individual kernel parameters.

Build on an Ascend host with CANN and the ASC compiler installed:

```bash
cmake -S tests/ascend/argument_list_330_repro -B build/argument_list_330_repro
cmake --build build/argument_list_330_repro -j
./build/argument_list_330_repro/argument_list_330_repro
```

The packed runtime check expects `packed kernel output=42`. On an affected
runtime, launching the legacy build reports error 330 instead of completing.
