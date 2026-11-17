# Vulkan D1 M1/M2 profiles

These profiles remain as fallback entries. The measured [M1..M8 table](vulkan-small-m.md) takes precedence when its type, activation precision, and shape match.

The Intel Vulkan backend uses seven measured Spark 1.7B profiles for target M2 and draft M1. The shader source and the weight layout are unchanged. M1 keeps F32 activations; M2 keeps the existing Q8 activation quantizer.

| Weight type | K | N | M | Output rows | SG / WG |
| --- | ---: | ---: | ---: | ---: | --- |
| q8_0 | 2048 | 131072 | 2 | 4 | 8/8 |
| q6_k | 6656 | 2048 | 2 | 4 | 8/8 |
| q4_k | 2048 | 6656 | 2 | 2 | 8/8 |
| q4_k | 2048 | 3072 | 2 | 2 | 8/8 |
| q4_k | 6656 | 2048 | 2 | 2 | 8/8 |
| q8_0 | 256 | 131072 | 1 | 8 | 16/16 |
| q8_0 | 2048 | 131072 | 1 | 4 | 16/16 |

Set `GGML_VK_DISABLE_D1_QUANT_OPT=1` before process startup to retain the previous M1/M2 paths. Other shapes, M3/M4, and unsupported device capabilities retain their existing paths. The M4 profiles and their disable switch are unchanged.

On Core Ultra 7 255H / Arc 140T, driver 32.0.101.6313, the final production build measured 33.692 to 36.675 token/s (+8.86%) for D1, gen.txt, seed 0. Three independent processes per mode produced identical complete 512-token outputs and draft counts. This is a single-input performance result, not a dataset quality evaluation.

The 17 real-weight operator cases use synthetic F32 activations. All seven selected profiles passed complete-output bitwise checks. The production build passed 34 real-weight on/off checks and four M3/M4 fallback pairs using public GGML APIs.

[Full report](../../../reports/igpu中耗时分布分析/20261116-112032-vulkan-d1-m2-m1-port/README.md) - [Production build identity](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261116-112032-vulkan-d1-m2-m1-port/version-production.json) - [Native numerical checks](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261116-112032-vulkan-d1-m2-m1-port/validation/native-production.json)
