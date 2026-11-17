# Vulkan small-token projection configurations

The Intel Vulkan backend extends the existing Q4_K vector loads, Q6_K word reconstruction and weight reuse, Q8_0 K16, and subgroup/workgroup tuning to M=1..8. M is the number of positions in one forward, as in X[M,K] * W[N,K]^T. A draft length of 3 can produce an M4 target forward; draft length and M are separate values.

The campaign covers 10 actual main/draft shapes, four weight types (Q4_K, Q6_K, Q8_0, BF16), and every M from 1 to 8: 320 strict cases. 184 cases have a newly selected default configuration. The other cases retain their existing dispatch. A separate M1 Q8_1 study covers 30 quantized-weight cases, with 3 selected opt-in configurations.

## Methods

| Method | Purpose |
| --- | --- |
| Q4_K uvec4 loads | Read the packed quants and block header in aligned vectors, retaining the GGUF layout. |
| Q4_K M1 uvec2 loads | Compare the previously measured contiguous F32-activation loader at workgroups of at least 16 threads. This campaign retained the faster generic F32 configurations. |
| Q6_K word loads | Reconstruct unaligned four-word slices of 210-byte blocks from four or five adjacent words. |
| Q6_K reuse | Decode weights/scales once per output row and reuse them across the specialized NUM_COLS token columns. |
| Q6_K branchless loads | Select only the measured shapes where this alternative wins. |
| Q8_0 K16 | Process 16 K elements per lane iteration. K16 is an iteration width, not FP16 precision. |
| Output-row tiles | Compute 1/2/4/8 output channels per workgroup to balance reuse, parallelism, and register use. |
| SG/WG tuning | Select subgroup size, workgroup size, and subgroup-only or shared reduction per K/N/M/type. |

The shader modules with `m4` or `REUSE4` in their existing names use NUM_COLS specialization constants. Those names do not limit the modules to four columns. The integration keeps their measured arithmetic and weight layouts. It adds no persistent weight copies.

## Shape coverage

| Position | K | N |
| --- | ---: | ---: |
| Main fused QKV | 2048 | 3072 |
| Attention head gate | 2048 | 8 |
| Attention output / separate draft Q | 2048 | 2048 |
| FFN gate and up, each | 2048 | 6656 |
| FFN down | 6656 | 2048 |
| Shared vocabulary head | 2048 | 131072 |
| Separate draft K and V, each | 2048 | 512 |
| Draft feature fusion | 10240 | 2048 |
| Markov projection | 256 | 131072 |
| Confidence projection | 2304 | 1 |

The GGUF inventory maps all 169 main projections and all 43 own projections in each BF16/Q4/Q8 draft recipe to these signatures. Each signature is tested in all four weight types, including type conversions used only for operator coverage. Native type matches use the original GGUF bytes. No model weights are rewritten by this integration. Embedding lookup, Markov W1 lookup, F32 norm vectors, and scalar bias are recorded separately in the inventory.

## Activations and opt-in M1 Q8_1

Default M1 retains the existing activation policy, which uses F32 on the measured Arc 140T. Quantized weights at M2..M8 retain the existing Q8_1 activation quantizer; BF16 weights retain their floating activation path. F16 inputs and widened BF16 inputs keep their existing precision rules. An explicit source precision that prohibits Q8 is respected.

Q8_0 stores 32 INT8 values and an FP16 scale in 34 bytes. Q8_1 adds an FP16 scaled sum in 36 bytes. The sum accelerates offset terms in quantized dot products. Q8_0 commonly stores weights; Q8_1 is the temporary activation representation here. Q8_1 is not a higher-precision grade of Q8_0.

Set `GGML_VK_ENABLE_M1_Q8=1` before startup to enable only the measured M1 Q8_1 signatures. This path includes an extra activation quantization step and changes the numerical contract. Unselected M1 signatures retain their floating path. BF16 weights do not use this opt-in path.

## Dispatch and fallback

Configurations require Intel integer-dot support, subgroup arithmetic, subgroup-size control, full subgroups, and sufficient workgroup/shared-memory limits. Matching requires the exact weight type, activation type, K, N, and M. Unsupported devices, missing pipelines, and unmatched signatures retain their existing paths. M9 is outside the table and has boundary regression coverage.

Identical specialization configurations share pipeline handles. The owning entry is revisited during lazy shader compilation, including when a later shape uses an alias.

- `GGML_VK_DISABLE_SMALL_M_OPT=1` disables the entire new table, including M1 Q8_1, and restores the retained backend choices.
- `GGML_VK_DISABLE_M4_QUANT_LOAD_OPT=1` also disables new M4 entries.
- `GGML_VK_DISABLE_Q8_SHAPE_OPT=1` also disables new Q8_0 entries.
- `GGML_VK_DISABLE_D1_QUANT_OPT=1` also disables new M1/M2 entries.
- `GGML_VK_DISABLE_MMVQ=1` continues to take precedence over opt-in M1 activation quantization.

## Numerical and performance evidence

The Arc 140T campaign uses three independent GPU processes and three independent host processes for each study. Cases share compiled device modules within a process; each case has 8 warmup and 36 measured cyclic/reversed observations per variant. Loading, weight conversion, and compilation are outside timing. GPU timestamps and host graph+synchronize clocks are separate.

Strict candidates require full-output NRMSE <= 1e-5 and maximum absolute error <= 1e-4 against retained dispatch, finite outputs, a sampled-channel FP64 reference, within-process output stability, and identical complete-output hashes across processes. M1 A8 candidates instead require full-output NRMSE <= 0.015, maximum error / max(1,max_abs_control) <= 0.02, and sampled FP64 NRMSE < 0.015. All 30 A8 controls match the original F32 outputs bitwise.

Adoption requires GPU combined latency (or floating matvec latency) to decrease by more than 2% in each of three processes, with no host graph regression in any of three processes. For M1 A8 adoption, the same gate is also applied against the newly selected F32 configuration using each study's retained-control ratios. Unsuccessful candidates remain in the evidence and are not enabled. Strict type preservation does not guarantee bitwise equivalence when reduction order changes. Operator checks do not establish dataset accuracy.

Run regression cases with `test-backend-ops -b Vulkan0 -o SMALL_M_PROJECTION,SMALL_M_ATTENTION,SPARK_DECODE_NORM`. These include all matrix signatures at M1..M9, F16/BF16 activation fallbacks, M1..M8 normalization, and F16/Q8_0 GQA attention at 512/4096 keys.

[Complete report and per-shape latency tables](../../../reports/igpu中耗时分布分析/20261117-112752-vulkan-s17-m1-m7/README.md). Raw evidence is retained in the linked experiment, including unsuccessful harness attempts, input/source/build identities, all samples, and selection scripts.

## Initial M1..M7 validation

The integrated Windows build passes 310 real-weight complete-output hash comparisons against the measured winners/fallbacks and 718 Vulkan/CPU regression cases across default and opt-in modes. Each backend run registers 373 cases: 359 pass and 14 half-activation cases are unsupported by the CPU reference (two are also unsupported by Vulkan). A separate public-API helper validates all 14 Vulkan-supported F16/BF16 activation cases per mode against complete FP64 dot products, for 28 numerical checks; opt-in/default half-activation outputs match bitwise. BF16 weights with F16 activation remain unsupported. Nine independent fixed-state main-model processes cover M1..M8 with 512 history tokens, 3 warmups and 12 measured forwards per M. Full logits are finite, stable across repeats and identical across independent processes for each mode; opt-in and default logits match bitwise at M2..M8, and retained/default M8 match bitwise. Full-model reduction-order and M1 activation-quantization errors are reported separately. These are fixed-token forward results, not sampled generation or dataset accuracy measurements.

## M8 extension

M8 integration passes 390 real-weight complete-output hashes (310 unchanged prior cases and 40 each optimized/retained M8 cases), 404 CPU/Vulkan numerical cases, and 21 complete-output FP64 half-activation cases. The backend registers 418 cases; 14 half-activation cases lack CPU reference support, including two also unsupported by Vulkan. An initial random regression has two unchanged M1 scalar-reference failures; controlled near-zero FP64 experiments show CPU activation-quantization error is much larger than the GPU error. Only the scalar fixture was conditioned, retaining the original NMSE threshold and signed weights/activations; all other random fixtures remain unchanged. The initial failure and all four diagnostic results are retained. Native results are reused only after verifying that all previously tested core executable hashes are unchanged. Six independent model processes use the exact previous serialized 512-token prefix and cover M8 plus the M9 fallback boundary. All logits are finite and stable within/across processes for each mode; retained M8 matches the previous complete logits, and M9 remains bitwise unchanged. Fixed-token forwards do not establish sampled generation speed, task accuracy, or DSpark acceptance.

The follow-up selects 32 additional M8 profiles across the same ten shapes and four weight types. `GGML_VK_DISABLE_M8_OPT=1` disables only these profiles. M1..M7 and the M1 Q8_1 opt-in profiles remain unchanged. [Follow-up evidence](../../../cpu_infer/experiments/windows-cpu/windows_core_ultra_7_255h/20261117-135017-vulkan-s17-m8/analysis/summary.json).

For the 1.7B Q4_K_M main model with the same 512-token prefix state, M8 forward+synchronize time decreases from 50.834 ms to 35.904 ms (29.37%). All three independent processes improve, by 26.32% to 33.03%. Complete-logit NRMSE is 6.80e-8, maximum absolute difference is 1.14e-5, and all eight row argmax values agree. M9 remains bitwise unchanged; its observed mean time increases by 2.73%, with no hardware-counter attribution.

## Q4_1 profiles

A separate real-weight QKV/down study adds eight Q4_1 profiles: QKV M1 and down M1, M2, M4..M8. The final build uses six matched ON/OFF rounds with Q6_K controls. See [Q4_1 profiles and validation](vulkan-q41-small-m.md).
