# Vulkan Q8_0 M4 K16 kernel

The 1.7B QKV shape (K=2048, N=3072) also has a K16/rows4 profile with SG8/WG16. Attention out and the vocabulary head retain SG8/WG8. See [1.7B M4 configurations](vulkan-s17-m4-configs.md) for the complete precision table and composition results.

The Intel M4 Q8_0 path uses the measured K16/rows4 kernel for K=2048 and N=2048 or N=131072. These are the two Q8_0 matrix shapes in `Spark-X2.6-1.7B-Q4_K_M.gguf`.

Each lane processes 16 K elements per iteration and accumulates four packed integer dot products before applying the scales. Each workgroup computes four output channels for all four token columns. The kernel uses SG8/WG8, the original Q8_0 weight layout, and the existing activation quantizer.

Pipeline creation uses the existing Intel integer-dot, subgroup arithmetic, subgroup-size-control and full-subgroup checks. `GGML_VK_DISABLE_M4_QUANT_LOAD_OPT=1` disables this fast path together with the existing M4 optimizations. Other shapes, token counts, and unsupported devices use their existing paths. The Q4_K/Q6_K and 4B Q8_0 pipelines retain their behavior.

## Numerical behavior

K16 changes the floating-point scaling and accumulation order. It is not bitwise equivalent to the K8 baseline. In the frozen operator experiment, the selected candidate passed all full-output comparisons with NRMSE <= 1e-5 and maximum absolute difference <= 1e-4. The observed worst differences for this candidate were NRMSE 1.4721e-7 and maximum absolute difference 3.8147e-6, including the diagnostic cases.

The separate FP64 reference checks, zero-activation cases, subnormal-scale cases and output stability checks also passed. This is operator-level evidence from deterministic synthetic values at actual GGUF shapes; it does not establish full-model quality or throughput.

## Measured implementation

Reference: [20261114-220205-vulkan-q8-s17-blocks](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261114-220205-vulkan-q8-s17-blocks/analysis/summary.json), candidate `k16_r4`, on Core Ultra 7 255H / Intel Arc 140T.

| Shape N x K | Baseline GPU us | K16/rows4 GPU us | GPU latency reduction | Host latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 2048 x 2048 | 113.759 | 94.681 | 16.77% | 9.12% |
| 131072 x 2048 | 4963.871 | 3892.664 | 21.58% | 21.90% |

GPU intervals include activation quantization. Host intervals measure an independent graph execution plus synchronization. Each shape and mode uses three independent processes with 24 warmup and 192 measured iterations. Each process passed the latency gate for the selected candidate. The original experiment used `ce58821a1` plus the prior Q6 integration; the present integration is on `dd872cfb1`, which also contains the Perfetto support.

## Integration checks

Evidence is recorded in [20261115-030927-vulkan-q8-s17-integration](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261115-030927-vulkan-q8-s17-integration/validation/integration.json).

The checks compare the complete preprocessed shader to the actual GPU-tested candidate, check existing shader configurations and pipeline specializations, compile and exercise the production dispatch and capability gates, and check generator syntax and integer/layout bounds. The existing `test-backend-ops` suite also gains Q8_0 2048 x 2048 cases for one and four token columns.

These local checks do not constitute a new Windows build, GPU run, or full-model validation. The newly added backend-ops cases have not been executed on Vulkan as part of this integration.
