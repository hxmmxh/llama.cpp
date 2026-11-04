# CPU optimization branch

`cpu_opt` is the CPU inference optimization branch, starting from `22c0ae1871b60123c5afc008a78fe9c6fffa2858`. Each optimization is kept in a separate commit with its tests and validation notes.

**GQA four-head decode and scheduling**

The first change ports the independently tested GQA4 attention patch from `a34de8618b25dfc32ee43749db47d767a9f0a25d`. Its workspace source is `cpu_infer/op_optimizer/gqa4_attention/patches/gqa4-attention.patch`.

Four query heads sharing one KV head traverse each KV chunk together. Jobs are distributed over KV chunks and head groups. Each head keeps its own query, F16 accumulator, softmax maximum and sum. The original `ggml_vec_dot_f16` call, per-head arithmetic order, partial-result layout and final reduction are retained.

The fast path requires:

- One query token and one sequence, with at least 512 KV positions.
- F32 queries, F16 keys and values, and 256-dimensional key/value heads.
- Exactly four query heads per KV head, with matching K/V head counts.
- No attention sinks, ALiBi bias or logit softcap.
- No mask, or an F16 mask shared by all heads and sequences.

Other shapes and features keep the existing paths. The CPU reference path is unchanged. This commit does not include GEGLU, activation quantization, Q8 KV or other optimization candidates.

**Validation**

Validation uses Apple M5 Pro, Apple Clang 21, Release/native CPU builds, with Metal, BLAS, Accelerate and OpenMP disabled. The control is the unmodified branch starting commit, built with the same configuration.

- Added 33 cases to `test-backend-ops`; the selected attention regression set passes 73/73 in both builds. It covers the 511/512/513 boundary, uneven chunks, contiguous/interleaved KV, sparse masks, large logits and fallback conditions.
- A deterministic external attention harness compares 168 cases across 1/3/8 threads, KV lengths 511 through 32768, four mask patterns and both KV layouts. All output bytes match the control.
- 166/168 cases pass the independent F32-reference NMSE threshold of `5e-4`. Both 32K, single-thread, unmasked layouts have the same existing error in the control and candidate: NMSE `5.53826004e-4`. The threshold is unchanged and these failures are retained.
- Spark-X2.6-4B Q4_K_M model checks at 2K and 8K history compare 16 decode steps each: all 4,194,304 logits are finite and bitwise identical to the control.

Decode timing uses 8 threads, F16/F16 KV, flash attention, compact SWA, context 8448, batch 256 and microbatch 128. Each process restores a baseline-generated state, warms up for 128 tokens and records three 128-token samples. Process order is baseline/candidate/candidate/baseline (ABBA). Throughput is total timed tokens divided by total decode time; loading, prefill, state restore and sampling are excluded.

| History | Baseline (tokens/s) | GQA4 (tokens/s) | Observed change |
| --- | ---: | ---: | ---: |
| 2048 | 40.04 | 46.16 | +15.27% |
| 8192 | 19.87 | 24.97 | +25.62% |

These are indicative local measurements, not stable speedup estimates. Between the two processes of each variant, throughput changes by +6.70% / +10.49% (baseline / candidate) at 2K, and +25.24% / -6.44% at 8K. In particular, the 8K baseline drift is comparable to the observed gain, so this run cannot establish its magnitude reliably. Windows performance has not been remeasured for this port.

The regression set can be run from a build with `test-backend-ops` enabled:

```sh
./build/bin/test-backend-ops -b CPU -o FLASH_ATTN_EXT \
  -p 'hsk=(128|256),hsv=(128|256),nh=4,nr23=\[(2|4),[12]\],kv=(511|512|513|768|2048),nb=(1|2),'
```

The workspace validation artifacts are in `cpu_infer/experiments/cpu-opt-gqa4-20261104-111129/`, including the source diff, build settings, library hashes, loaded-library logs and raw results. Historical speedups from the older source revision are not treated as measurements of this port.

**GEGLU single-row column parallelism**

This change ports `cpu_infer/op_optimizer/geglu_parallel/patches/geglu-parallel.patch`, originally based on `a34de8618b25dfc32ee43749db47d767a9f0a25d`. Its control is `80da68ddbff46bce475380c85aaebc85202fe424`, which already contains GQA4.

For F32 GEGLU with one row, at least 4096 output elements and multiple threads, workers process disjoint column ranges aligned to 16 floats. Each element still uses the existing GEGLU vector function. Packed input, swapped halves and separate gate/input tensors keep their existing semantics. Multiple rows, short rows, single-thread execution, other data types and the CPU reference path retain their existing scheduling.

The original experimental patch starts column splitting at 1024 elements. This port raises the threshold to 4096 because historical i5 measurements showed regressions for 1024/2560 elements at 8 threads. Those lengths now retain the original row path. This is a conservative scope restriction, not a claim that 4096 is optimal for every CPU and thread count.

Validation uses the same native CPU build configuration described above:

- Added 48 cases to `test-backend-ops`, covering short rows, the 4095/4096/4097 threshold, uneven tails, packed/split/swapped inputs, padded views and multi-row fallback. Both variants pass all 84 selected GEGLU tests.
- A deterministic external harness checks 480 cases at 1/3/8/16 threads, ten widths from 1 to 10241, one/two rows, three input layouts and contiguous/padded views. Inputs include GELU branch boundaries and signed zero; outputs are cleared to NaN before each candidate calculation to detect unwritten elements. All outputs are finite and bitwise equal to the CPU reference and the control.
- Spark-X2.6-4B Q4_K_M checks at 2K and 8K history compare 16 decode steps per depth. All 4,194,304 logits are finite and bitwise identical to the GQA4 control.

The single-node microbenchmark uses persistent thread pools and ABBA process order. Each configuration warms up for 20 calls, calibrates a batch to at least 50 ms and records five batches. Timing includes graph execution and synchronization, and excludes allocation and initialization. All four process output dumps are bitwise identical.

| Elements | Threads | Control (us) | GEGLU parallel (us) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 4096 | 3 | 2.815 | 1.676 | 1.68x |
| 10240 | 3 | 5.825 | 2.613 | 2.23x |
| 4096 | 8 | 3.726 | 1.986 | 1.88x |
| 10240 | 8 | 7.209 | 2.173 | 3.32x |

The 8-thread, 10240-element control/candidate latency drift is +3.25% / -2.17%. The unchanged single-thread and short-row cases serve as controls; their timing fluctuations are not optimization gains. The 16-thread measurements have much larger process drift (up to 58.52%), so their observed speedup is not reported here as a stable estimate. Full records include all 24 configurations and their drift.

Whole-model timing uses the same 8-thread, F16/F16 KV, compact-SWA protocol as the GQA4 measurements above, with newly saved control states. An initial ABBA run showed small negative changes at both depths within substantial process drift, so a BAAB run was added using identical binaries, states and parameters.

| Order | History | GQA4 control (tokens/s) | GQA4 + GEGLU (tokens/s) | Observed change | Control / candidate throughput drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| ABBA | 2048 | 46.98 | 45.87 | -2.38% | -14.38% / -0.37% |
| ABBA | 8192 | 36.70 | 35.63 | -2.93% | +8.73% / +10.96% |
| BAAB | 2048 | 51.54 | 51.41 | -0.26% | +1.84% / -1.36% |
| BAAB | 8192 | 38.06 | 40.35 | +6.04% | -1.02% / -0.76% |

The reversed run is effectively neutral at 2K and positive at 8K. Because the two rounds disagree in direction at 8K and the initial round has substantial drift, these measurements do not establish a stable whole-model speedup. The demonstrated benefit of this commit is the wide-row GEGLU operator; its speedup must not be substituted for model throughput.

The backend regression set can be reproduced with:

```sh
./build/bin/test-backend-ops -b CPU -o GEGLU
```

Build settings, source and library hashes, exact-output dumps and raw timing records are retained in `cpu_infer/experiments/cpu-opt-geglu-20261104-115621/`. Windows has not been remeasured for this port.
