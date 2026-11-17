# Experimental FFN RMSNorm + Q8_1

Set `GGML_VK_FFN_RMS_Q8_1=1` before backend initialization to try this fusion. It is disabled by default.

The Intel Vulkan path recognizes RMSNorm, norm-weight multiplication, and two adjacent expanding projections that share the normalized input. A single normalization kernel writes the existing x4-packed Q8_1 format directly into the activation workspace. Both projections retain their selected matrix kernels and reuse that workspace; the FP32 normalized activation is not materialized.

Both projections remain in the submission FLOP budget even though the graph loop skips their individual nodes. The fused group is submitted as one unit.

The experiment requires contiguous FP32 input, a vector of FP32 norm weights, 512-4096 input features divisible by 128, and 2-8 token rows. Both Q4_K/Q6_K/Q8_0 projections must already select Q8_1 under the existing device and precision policy. Subgroup arithmetic, clustered operations, full subgroups, and sufficient shared memory are required.

The quantizer derives its index from subgroup ID and lane ID so clustered reductions cover consecutive activation blocks. Vulkan does not guarantee a direct mapping from local invocation IDs to subgroup lanes; see [SubgroupLocalInvocationId](https://docs.vulkan.org/refpages/latest/refpages/source/SubgroupLocalInvocationId.html).

An exported normalized activation, another consumer, a floating-point projection, or a precision override that disallows Q8 retains the existing path. M1 and larger batches retain the existing path. No GGML tensor type or model graph API is changed.

Validation compares the same binary with the switch unset/set, including complete projection outputs, repeated input updates, and fallback cases. Performance claims require the complete normalization-plus-projection interval and separate clean host timings.

The initial R6 Arc 140T experiment passed 100 complete-output off/on comparisons. For the independent normalization-plus-gate/up graph, M2/M4 reduced mean host-wall time by 8.99%/10.96%, with improvement in all three process pairs. Full FFN means improved by 3.97-7.58%, but each tested row count had a process pair that regressed. See the [initial report](../../../reports/igpu中耗时分布分析/20261117-133933-vulkan-ffn-rms-budget/README.md).

The same-binary follow-up used 12 off/on process pairs per full FFN shape, plus 6 pairs for the M4 prefix, with 100 warmups and 1000 samples per process. Full FFN M2/M4/M8 paired geometric mean time reductions were 4.39%/7.62%/4.56%; their paired bootstrap 95% intervals were above zero. M2/M4 improved in 12/12 pairs and M8 in 10/12. All 168 complete-output comparisons passed. The switch remains off by default; these independent-graph timings do not establish whole-model throughput gains. See the [expanded report and output-shape table](../../../reports/igpu中耗时分布分析/20261117-140746-vulkan-ffn-rms-repeat/README.md).

The subsequent [producer fusion experiment](vulkan-producer-q8.md) adds GEGLU -> down, gated attention -> Wo, attention norm -> QKV and final norm -> head. With this FC1 switch enabled in both modes, adding GEGLU fusion reduced complete M4 FFN paired time by 11.83% in 12 process pairs, with a 3.45% to 17.37% bootstrap interval. This is an incremental result from a separate frozen revision; it should not be added directly to the percentages above.
