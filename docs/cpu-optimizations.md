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
