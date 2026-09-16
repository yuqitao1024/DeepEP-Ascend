# Combine kernel parameter ABI repro

This is a small standalone reproduction of the Ascend combine-kernel failure
fixed in `de9519d`. The device kernel has a one-byte enum (`Stage`) beside a
large aggregate containing 64-bit fields (`Tiling`), matching the relevant
shape of `DirectCombineStage` and `CoreTiling`.

The fixed form passes `Tiling` before `Stage`. The old form passes `Stage`
before `Tiling` and can be decoded differently by the device ABI lowering;
the resulting bad `local_window_base`/offset values lead to an invalid GM
address (runtime 507035, device error 264) in the real combine path.

Build the fixed form on an Ascend host with CANN:

```bash
cmake -S tests/ascend/combine_kernel_abi_repro \
  -B build/combine_kernel_abi_repro
cmake --build build/combine_kernel_abi_repro -j
./build/combine_kernel_abi_repro/combine_kernel_abi_repro
```

To compile the pre-fix order:

```bash
cmake -S tests/ascend/combine_kernel_abi_repro \
  -B build/combine_kernel_abi_repro_bad \
  -DDEEP_EP_REPRO_BAD_ORDER=ON
cmake --build build/combine_kernel_abi_repro_bad -j
```

The fixed build must print `output=42`. On an affected CANN/Ascend toolchain,
the bad build may fail during stream synchronization with an invalid-GM
device error (the standalone probe produced error 334, while the full
combine path produced error 264 and host wrapper error 507035). A toolchain
that happens to accept both layouts still provides a
compile-time comparison, but is not evidence that the old order is safe in
DeepEP's full `CoreTiling` ABI.
