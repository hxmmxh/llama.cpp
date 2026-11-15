# Vulkan 1.7B M4 configurations

The Intel MMVQ path selects the measured configuration by weight type and exact `(K, N)` for `X[4,K] * W[N,K]^T`. The table is in `vk_m4_quant_s17_configs` in `ggml-vulkan.cpp`. Each cell below is **SG / WG / output rows per workgroup**; every workgroup handles all four token columns.

| Projection | K | N | Q4_K | Q6_K | Q8_0 | BF16 (existing default) |
| --- | ---: | ---: | --- | --- | --- | --- |
| QKV | 2048 | 3072 | 8/8/2 | 8/8/2 | 8/16/4 | 16/64/2 |
| Head gate | 2048 | 8 | 8/64/1 | 16/256/1 | 8/128/2 | 16/64/2 |
| Attention out | 2048 | 2048 | 8/8/2 | 8/8/2 | 8/8/4 | 16/64/2 |
| FC1 (gate/up) | 2048 | 6656 | 16/16/2 | 8/8/4 | 8/32/8 | 16/64/2 |
| FC2 | 6656 | 2048 | 8/16/4 | 8/8/2 | 8/32/8 | 16/64/2 |
| Vocabulary head | 2048 | 131072 | 16/16/2 | 16/32/4 | 8/8/4 | 16/16/2 |

Q4_K uses the vector-128 loader except for the narrow head gate. Q6_K uses the five-word loader with four-token weight reuse for QKV, attention out and FC2; FC1 uses the branchless loader without reuse, the vocabulary head uses branchless loading with reuse, and the head gate uses the generic kernel without reuse. Q8_0 uses K16/rows4 for QKV, attention out and the vocabulary head; the other three shapes use the generic K8 kernel. BF16 retains the backend's existing heuristics.

Only the existing Intel integer-dot M4 path is eligible. Pipeline creation checks subgroup arithmetic, subgroup-size control, full-subgroup support, subgroup limits, workgroup limits, maximum workgroup subgroups and shared-memory capacity. Unsupported configurations retain the existing fallback. `GGML_VK_DISABLE_M4_QUANT_LOAD_OPT=1` disables these profiles and the earlier M4 optimizations. The existing 4B fast paths remain first in dispatch. Configurations with the same shader, row count and geometry share pipelines.

The change adds four compile-time variants of existing shader arithmetic: Q4 vector loads with shared reduction, Q6 branchless loading, Q6 branchless loading plus reuse with shared reduction, and Q8 K16 with shared reduction. The shader algorithm source files are unchanged.

## Evidence and selection

Q4_K/Q6_K/BF16 use the accepted results from [the earlier 24-group SG/WG sweep](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261114-222248-vulkan-s17-optimized-sgwg/analysis/recommendations.json). Q8_0 uses the [new K16 composition sweep](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261115-031615-vulkan-s17-config-integration/analysis/recommendations.json), frozen after the concurrent K16 integration completed at 2026-11-15 03:15 +08:00. Both experiments used Core Ultra 7 255H / Intel Arc 140T, driver 32.0.101.6313, with cooperative matrices disabled.

The Q8 sweep tested six shapes with 15 SG/WG choices for K16/rows4, the current production path, and the previous recommended Q8 kernel/configuration: 36 formal processes and 127,296 timing samples. GPU intervals include activation quantization; host graph timings are independent. An update must improve GPU time by more than 2% in each of three paired processes and avoid host regression in each. The incumbent is the new production K16/SG8/WG8 for attention out and the vocabulary head, and the previous recommendation for other shapes.

Q8 QKV selects K16/rows4 SG8/WG16: 122.663 us versus 127.179 us for the previous recommendation (3.55% less GPU time, 4.23% less host time). Its additional gain over K16/SG8/WG8 is only 0.62% in pooled means and is not independently established as a stable SG/WG gain. Attention out and the vocabulary head retain K16/SG8/WG8. The other Q8 shapes retain their previous recommended configurations. The absolute fastest individual pooled result is not automatically adopted.

The actual 1.7B Q4_K_M model uses Q8_0 only for attention out and the vocabulary head. The Q8 QKV experiment covers the user's requested alternative precision; it does not imply a 3.55% full-model improvement.

## Validation

[Static integration checks](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261115-031615-vulkan-s17-config-integration/validation/integration/checks.json) compare all 18 selected quantized shader token streams to their measured winners, verify existing shader paths are unchanged, and exercise the production pipeline creation and dispatch under 16 capability/escape-hatch scenarios, with and without integer-dot compilation, using ASan/UBSan.

[Native production validation](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261115-034359-vulkan-s17-config-native/analysis/summary.json) uses the final production GGML source without private benchmark dispatch or timing hooks. It covers the six shapes in four precisions, the disable switch, M=1/2/3/5, the existing 4B paths, non-target shapes/output tails, and zero activations. Every M4 target is compared bitwise to the selected measured winner; all outputs are also checked against full FP64 references. See the linked run status for completion. No full-model throughput or quality evaluation is implied by these operator checks.
