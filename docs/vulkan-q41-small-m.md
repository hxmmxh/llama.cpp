# Q4_1 small-M profiles

The Intel Vulkan backend enables 8 measured Q4_1 profiles for Spark 1.7B QKV W[N,K]=[3072,2048] and FFN down [2048,6656]. QKV uses the new profile at M1; down uses it at M1, M2 and M4..M8. Remaining signatures retain their previous dispatch.

The kernels read standard 20-byte Q4_1/G32 blocks directly. M1 processes a complete G32 block and gathers its payload from aligned uvec4 reads, including the 20-byte-stride realignment. Q8_1-activation paths reuse weight metadata and unpacked nibbles across input positions, with measured K8/K16 steps. Other selected cases use existing Q4_1 shaders with measured row/subgroup/workgroup constants.

| Shape | M | Kernel family | Rows | Subgroup | Workgroup |
| --- | ---: | --- | ---: | ---: | ---: |
| attn_qkv | 1 | f32_vec | 4 | 8 | 8 |
| ffn_down | 1 | f32_vec | 2 | 16 | 16 |
| ffn_down | 2 | q8_k16 | 2 | 32 | 32 |
| ffn_down | 4 | generic | 4 | 8 | 32 |
| ffn_down | 5 | generic | 4 | 8 | 32 |
| ffn_down | 6 | q8_k8 | 2 | 32 | 32 |
| ffn_down | 7 | generic | 2 | 16 | 16 |
| ffn_down | 8 | generic | 4 | 8 | 8 |

Profiles are selected by exact weight type, activation type, K, N and M. They inherit the existing Intel integer-dot/subgroup capability checks and small-M, D1, M4 and M8 gates. Set `GGML_VK_DISABLE_Q41_OPT=1` to retain the previous Q4_1 paths; clear the variable to restore these defaults. M1 uses F32 activation; M2..M8 use Q8_1 activation.

## Validation

Core Ultra 7 255H / Arc 140T, Windows Vulkan, driver 32.0.101.6313. Six matched ON/OFF process pairs per timing mode, identical case order and Q6_K/Q4_1 format mix, 16 warmups and 128 measured calls per case. GPU timestamps and host graph-compute+synchronize measurements use separate processes. All 776 complete-output cases and 141,312 formal samples pass. New versus retained Q4_1 output satisfies NRMSE <= 1e-5 and maximum absolute difference <= 1e-4. Retained profiles, M9 and Q6_K controls match bitwise between flag states. Actual dispatch names and specialization constants are checked.

M1 down GPU time is 203.23 -> 140.54 us (30.85% lower); host time is 338.55 -> 262.71 us (22.40% lower). Q6_K GPU time in this build is 198.18 us.

These are hot resident-operator measurements using real layer-0 weights and deterministic synthetic input. Loading, compilation, input upload and output readback are outside timing. Full-model throughput and task quality were not measured. Initial candidates and all valid slow samples remain in the evidence; five weak or inconsistent profiles were removed before this final build.

[Complete report](../../../reports/igpu中耗时分布分析/20261117-163512-vulkan-s17-q41-port/README.md) / [Output and dispatch validation](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261117-193622-vulkan-s17-q41-confirm/analysis/full-output-validation.json) / [Samples and process means](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261117-193622-vulkan-s17-q41-confirm/analysis/summary.json) / [Focused production patch](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261117-163512-vulkan-s17-q41-port/provenance/production-q41.patch)
