# Vulkan Q6_K M4 weight reuse

The FC1/FC2 configuration described below is the existing 4B profile (K/N=2560/10240 and 10240/2560). The 1.7B shapes now use the per-shape loader, reuse and geometry choices documented in [1.7B M4 configurations](vulkan-s17-m4-configs.md).

The Intel M4 Q6_K FC1/FC2 pipelines use the measured `five_words_reuse4` implementation:

- Load four consecutive 32-bit words when aligned, or five shared words and reconstruct the four packed words when offset by two bytes.
- Prepare each output row's unpacked weights and scale once per K iteration, then reuse them across the four token columns.
- Keep the existing activation quantization, integer dot products, floating-point operation order, SG8/WG8, and dispatch conditions.
- Keep FC1 at two output rows per workgroup and FC2 at four output rows. Both pipelines specialize the same shader module.

Only the existing Intel M4 fast path changes. Q4_K, Q8_0, M1, generic fallback paths, and the `GGML_VK_DISABLE_M4_QUANT_LOAD_OPT` escape hatch retain their previous behavior. No weight repacking or new storage layout is used.

## Measured implementation

The reference is the frozen `ce58821a1` experiment [20261113-175040-vulkan-mmvq-full-combinations](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261113-175040-vulkan-mmvq-full-combinations/README.md), variant `five_words_reuse4`. On Core Ultra 7 255H / Arc 140T, GPU time including activation quantization decreased by 20.02% / 6.21% for FC1 / FC2; independent host time decreased by 16.95% / 4.34%. The full experiment passed 216 complete-output bitwise admission checks across all candidates. These are fixed-input operator measurements, not complete-model throughput results.

The integration checks are recorded in [20261114-214633-vulkan-q6-reuse-integration](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261114-214633-vulkan-q6-reuse-integration/validation/integration.json):

- The complete preprocessed Q6 shader matches the GPU-tested winner after renaming three private identifiers.
- Six pre-existing shader configurations have unchanged preprocessed token streams.
- The production dispatch conditions and specialization values are unchanged.
- The shader generator passes C++17 syntax checks.
- The reader checks pass ASan/UBSan, including 18,688 bounded Q6 slices.

These integration checks do not constitute a new Windows build or GPU run. Performance numbers above belong to the frozen measured implementation. No new full-model measurement was performed for this integration.
