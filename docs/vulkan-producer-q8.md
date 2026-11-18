# Experimental producer-to-Q8_1 fusions

These Intel Vulkan experiments extend the [FFN RMSNorm fusion](vulkan-ffn-rms-q8.md). Set each switch before backend initialization. The [Arc 140T fixed-D3 server configuration](vulkan-optimizations.md) enables the GEGLU fusion by default. Set `GGML_VK_GLU_Q8_1=0` to opt out. The three attention and head switches remain opt-in.

| Position | Switch | Activation shape | Projection output shape |
| --- | --- | --- | --- |
| GEGLU -> FFN down | `GGML_VK_GLU_Q8_1=1` | Q8_1 `[M,6656]` | FP32 `[M,2048]` |
| Attention gate multiplication -> Wo | `GGML_VK_ATTN_MUL_Q8_1=1` | Q8_1 `[M,2048]` | FP32 `[M,2048]` |
| Attention RMSNorm -> QKV | `GGML_VK_ATTN_RMS_Q8_1=1` | Q8_1 and FP32 `[M,2048]` | QKV FP32 `[M,3072]`; gate FP32 `[M,8]` |
| Final RMSNorm -> LM head | `GGML_VK_HEAD_RMS_Q8_1=1` | Q8_1 `[R,2048]` | Logits FP32 `[R,131072]` |

Shapes describe Spark-X2.6-1.7B Q4_K_M in token-first notation. GGML stores the feature dimension first. R is the number of rows selected for logits. Q8_1 storage uses the existing packed x4 layout.

The producer writes quantized activations into the existing matrix input workspace. Matrix kernels and their precision policy remain unchanged. The attention norm also writes FP32 for the BF16 gate projection. The final norm drops its embedding output marker only when embeddings are disabled; requested embeddings retain the original path. DSpark's exported layer inputs precede this norm and remain available.

The matcher requires contiguous FP32 activations, aligned buffers, supported Q4_K/Q6_K/Q8_0 weights, 1-8 rows, and an existing Q8_1 matrix-vector choice. M1 remains subject to the separate existing M1 Q8 policy and normally falls back in this configuration. Larger batches, precision overrides, floating-point projections, unsupported layouts, and exported intermediates retain the original path. Attention RMSNorm permits its normalized FP32 result to remain an output.

GEGLU supports separate gate/up inputs and preserves the existing GELU formula. Attention multiplication supports a contiguous `[head_dim,heads,M]` input and `[1,heads,M]` gate with a zero-offset reshape before Wo. Norm fusion supports 2048-4096 features. The device must provide subgroup arithmetic, clustered operations, full subgroups, integer dot products, subgroup size 8-32, and a 512-invocation workgroup with sufficient shared memory.

Graph optimization keeps each producer and projection adjacent and extends external input lifetimes through the fused result. Existing trailing matrix-add fusions remain available. Output masks and alias checks retain the backend's graph lifetime rules. All skipped matrix work remains in the submission FLOP budget.

Validation uses the same binary with individual switches off/on. It includes CPU-reference backend tests, complete normal/zero/tiny/changed output comparisons, real-model logits and embeddings, embedding mode changes in one context, selected logits, quantized KV, and the layer inputs used by DSpark. This does not measure a complete DSpark round or acceptance behavior.

The fixed performance plan measures independent graphs for each position at M2/M4/M8, an inactive LM-head M1 control, and a full FFN M4 combination with the previous FC1 fusion enabled in both modes. Each case has 12 independent off/on process pairs, 100 warmups and 1000 clean samples per process. Loading, compilation, transfers, readback and profiling are excluded from these timings. These measurements do not establish whole-model throughput gains.

Each timing covers graph compute and completion synchronization for the full local graph, including activation quantization. The Wo graph also includes gate sigmoid; the QKV norm graph includes the BF16 gate projection and sigmoid. The FFN combination covers normalization, gate/up, GEGLU, down and residual addition. These are graph-level timing comparisons rather than sums of separate operator timestamps.

On Arc 140T, the fixed 12-pair experiment measured these paired geometric mean time reductions at M4:

| Position | Time reduction | Paired bootstrap 95% interval | Faster pairs |
| --- | --- | --- | --- |
| GEGLU -> down | 5.34% | 0.10% to 10.73% | 10/12 |
| Gate multiplication -> Wo | 7.04% | 4.78% to 9.31% | 11/12 |
| Attention RMSNorm -> QKV | 6.01% | 3.60% to 8.52% | 10/12 |
| Final RMSNorm -> head | 0.27% | -0.74% to 1.31% | 7/12 |

Wo also reduced M2 time by 10.77% with an interval of 6.39% to 14.59%. The other M2/M8 intervals crossed zero. Intervals are descriptive and have no multiple-comparison correction; the GEGLU M4 lower bound is close to zero.

Adding GEGLU fusion to the complete M4 FFN with the previous FC1 fusion enabled in both modes reduced paired time by 11.83%, interval 3.45% to 17.37%, with 10/12 faster pairs. A separate same-binary trace confirmed four command-buffer submissions without FC2 fusion and three with it; the independent GEGLU graph changed from two to one. Thus the graph-level effect also includes submission changes, and is not a measurement of saved memory traffic alone. The complete gain has not been separated into kernel, cache and submission costs.

CPU and Vulkan each passed 26 new backend cases. All 672 formal complete-output comparisons and the real-model checks were bitwise equal. The full evidence audit passed, including 3987 frozen source files, four executables and 1994 raw output hashes. See the [full results, all shapes and retained regressions](../../../reports/igpu中耗时分布分析/20261117-153952-vulkan-producer-q8/README.md).
