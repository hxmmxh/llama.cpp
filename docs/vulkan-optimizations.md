# Vulkan optimization inventory

This inventory describes the `vulkan_opt` implementation, including the integrated 1.7B kernels and the optional DSpark asynchronous handoff. The older `openvino_opt` checkout does not contain these Vulkan kernel changes.

## Matrix-vector kernels

- The earlier `ce58821a1` change adds Q4_K vector reads including block headers, Q6_K word and branchless reads, and larger output-channel groups for four-token matrix-vector operations.
- [Q6_K weight reuse](vulkan-q6-m4-reuse.md) loads adjacent words and reuses unpacked weights across four token columns. The 4B FC1/FC2 paths retain their existing shapes and dispatch geometry.
- [Q8_0 K16](vulkan-q8-m4-k16.md) processes 16 K elements per lane iteration and four output channels per workgroup for selected 1.7B shapes. It changes floating-point accumulation order; its numerical contract is tolerance based rather than bitwise equivalence to the old kernel.
- [1.7B shape configurations](vulkan-s17-m4-configs.md) select the measured loader, reuse, output-channel count, subgroup and workgroup for six projection shapes in Q4_K/Q6_K/Q8_0. BF16 keeps existing choices. Pipeline creation checks device limits, shares identical configurations and supports lazy compilation.

These original M4 profiles are selected automatically for eligible Intel integer-dot devices. `GGML_VK_DISABLE_M4_QUANT_LOAD_OPT=1` disables the fast paths. Within this legacy table, other token counts and unsupported devices retain their existing paths. Maximum draft length 3 permits a four-position target verification forward; the actual executed shapes still determine dispatch.

## M1..M8 coverage

[Small-token configurations](vulkan-small-m.md) extend the measured loaders, reuse and geometry choices to all ten 1.7B main/draft linear shapes and Q4_K/Q6_K/Q8_0/BF16 at M1..M8. Every tuple is tested; only stable improvements are selected. The new table precedes the retained M4, D1 and Q8-shape tables. Default activation precision is preserved; measured M1 Q8_1 paths are available through `GGML_VK_ENABLE_M1_Q8=1`.

RMSNorm/scale/residual fusion, GELU-times-up, RoPE and KV writes already accept multiple rows. Existing GQA, scalar attention, and split-K selection also cover small M; the regression matrix now exercises F16/Q8_0 KV attention and normalization throughout M1..M8. Their algorithms and KV formats are unchanged by the projection integration.

## Producer activation quantization

The optional [FFN RMSNorm fusion](vulkan-ffn-rms-q8.md) writes Q8_1 activations directly for gate/up. Additional [producer fusions](vulkan-producer-q8.md) cover GEGLU before down, attention gate multiplication before Wo, attention RMSNorm before QKV, and final RMSNorm before the LM head. Each has a separate switch and is disabled by default.

These paths retain existing matrix kernels and precision choices. Attention RMSNorm also writes FP32 for the BF16 gate projection; final RMSNorm preserves the embedding path when requested. Measurements use independent graphs and do not establish whole-model throughput gains.

## DSpark buffers and scheduling

Device DSpark uses a persistent backend host buffer for token and probability readback. On Vulkan this permits asynchronous copies into the backend host allocation. This buffer change also applies when asynchronous handoff is disabled.

[Asynchronous handoff](dspark-async.md) additionally defers host collection until after the target forward, passes draft token IDs between device graphs, caches token embeddings on the device when needed, and orders producer and consumer with backend events. Server teardown drains pending results; final decode timing includes outstanding draft work.

The handoff requires `LLAMA_DSPARK_DEVICE=1`, `LLAMA_DSPARK_DENSE_Q=1` and `LLAMA_DSPARK_ASYNC=1`. It is disabled by default. It retains the host rejection sampler and does not run independent target and draft forwards concurrently. Its previous 4B end-to-end comparison did not establish a throughput gain: [gen/test-c comparison](../../../reports/igpu-benchmark/20261114-203157-vulkan-async-ab-gen-testc/README.md).

## Diagnostics and validation

The `dd872cfb1` commit provides optional Perfetto CPU/GPU timelines. Profiling is disabled for clean throughput measurements.

The integrated GGML source matches the production source used by [108 native correctness cases](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261115-034359-vulkan-s17-config-native/analysis/summary.json), including 24 winner comparisons, enabled/disabled pairs, fallback token counts and FP64 reference checks. The backend tests also cover Q8_0 M1/M4 multiplication and device token handoff with bounds and reuse checks.

Published operator latency reductions belong to their fixed shapes and measurement scopes. They do not establish whole-model throughput or task accuracy. A version comparison must freeze the complete source, build Vulkan, verify device placement and use matching inference settings. A CPU build of `openvino_opt` cannot measure these changes.
