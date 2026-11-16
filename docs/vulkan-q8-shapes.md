# Vulkan Q8_0 projection configurations

The Intel Vulkan backend enables the measured K16 and output-row grouping configurations below by default. They reuse the existing Q8_0 shaders and the original weight layout. No extra weight copies or persistent GPU buffers are required.

`K` is the weight input dimension, `N` is the output dimension, and `M` is the token-column count. Every configuration uses SG8/WG8. Each workgroup computes `rows` output channels for all `M` columns.

| Projection | K | N | M | rows | Activation / kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| Markov | 256 | 131072 | 1 | 8 | F32, existing kernel |
| Attention output | 2048 | 2048 | 2 | 8 | Q8_1, K16 |
| Attention output | 2048 | 2048 | 3 | 4 | Q8_1, K16 |
| Draft FFN gate/up | 2048 | 6656 | 3 | 8 | Q8_1, K16 |
| Shared vocabulary head | 2048 | 131072 | 2 | 4 | Q8_1, K16 |
| Shared vocabulary head | 2048 | 131072 | 3 | 4 | Q8_1, K16 |
| Draft FFN down | 6656 | 2048 | 3 | 4 | Q8_1, K16 |
| Draft feature projection | 10240 | 2048 | 2 | 4 | Q8_1, K16 |
| Draft feature projection | 10240 | 2048 | 4 | 4 | Q8_1, K16 |

The M3 vocabulary-head configuration was tested first. The other eight signatures are the later `primary` selection. The M1 Markov configuration retains F32 activations; K16 does not change weight precision to 16 bits. Shapes not listed retain their previous dispatch.

## Dispatch and fallback

The path requires Intel integer-dot support, subgroup arithmetic, subgroup-size control, full subgroups, subgroup size 8, and sufficient workgroup limits. Matching also requires Q8_0 weights and the exact K/N/M tuple. M1 requires F32 activations; the other entries require the existing Q8_1 activation path. Unsupported capabilities and missing pipelines fall back to the existing backend choices.

`GGML_VK_DISABLE_Q8_SHAPE_OPT=1` disables this entire table, including the M3 head entry. `GGML_VK_DISABLE_M4_QUANT_LOAD_OPT=1` also disables its M4 entry. The experimental `GGML_VK_HEAD_M3_K16` and `GGML_VK_Q8_ALL_MODE` flags are not required. Q4_K, Q6_K, BF16, and the existing M4 profiles are unchanged.

Pipelines with the same M and row count share their handle. The shader loader revisits the owning entry on every load so that lazy compilation also works through an alias.

## Numerical and performance evidence

K16 and row grouping can change floating-point reduction order. The operator checks used real model weights and full output comparisons against the retained backend, with NRMSE <= 1e-5 and maximum absolute error <= 1e-4. They also checked finite outputs and a sampled FP64 reference. This is not a bitwise-equivalence guarantee across configurations.

On Core Ultra 7 255H / Arc 140T, driver 32.0.101.6313, the earlier isolated M3 head experiment reduced GPU latency by 22.82% and improved same-output full D3 throughput from 40.490 to 41.585 token/s. The later eight-seed experiment compared this retained M3 head with the additional eight signatures: 46.034 to 46.820 token/s (+1.71%), with 4 of 8 seeds faster; mean host cost per actual round was 49.172 to 48.186 ms (-2.00%). Acceptance was 42.107% to 41.906%. Output and acceptance trajectories differed across configurations, so the full throughput difference is not a fixed-work kernel speedup. The seed-bootstrap interval crosses zero; task accuracy was not evaluated.

Both experiments used Spark1.7B, Q4_K_M target, Q8_0 backbone draft with a shared target head, D3/async1, CPU acceptance/rejection, F16 KV, flash attention, and cooperative matrices disabled. The multi-seed input was the complete 3667-token gen.txt fixture, with 24 warmup tokens and up to 512 formal tokens in three independent processes per seed/configuration.

- [Head experiment](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261115-194131-vulkan-q8-head-m3/)
- [All-shape operator experiment](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261115-202009-vulkan-q8-all-shapes/)
- [Eight-seed report](../../../reports/igpu中耗时分布分析/20261116-111820-vulkan-q8-primary-seeds/README.md)

Integration validation is recorded in `vulkan-q8-shapes-validation.json`. The added `test-backend-ops` cases cover all nine tuples and nearby unchanged shapes against the CPU backend.
