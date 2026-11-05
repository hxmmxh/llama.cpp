# DSpark CPU optimizations

The local `cpu_opt` branch includes the validated Spark-X2.6-4B DSpark optimizations developed in the workspace experiments. This integration starts at `7a9121b51`, retaining the existing GQA attention, GEGLU, NEON Q8_K quantization and Q8 V optimizations. It adds no new model format, public API or speculative decoding algorithm.

| Change | Implementation and scope |
| --- | --- |
| ARM BF16 dot product | `ggml/src/ggml-cpu/vec.cpp`: AArch64 NEON F32 products with F64 accumulation; other architectures keep their existing paths. Primarily relevant to the original BF16 draft and the small BF16 tensors retained after quantization. |
| Shared vocabulary head padding | `src/models/dflash.cpp`: for the single-sequence DSpark shared Q8_0 CPU_REPACK head with activation `[2560,7]` and weights `[2560,131072]`, append one zero activation row, project eight rows, then retain the first seven. The model block, attention, KV and Markov chain retain seven positions. |
| Q4_K/Q6_K tail3 | `ggml/src/ggml-cpu/repack.cpp` and `arch/arm/tail3-q4.inc`, `tail3-q6.inc`: AArch64 NEON/DOTPROD, Q8_K activations, 8x8 layout and M7. The last three GEMV rows share weight decoding while retaining per-row arithmetic order. |
| Q4_K/Q6_K GEMM8 | `ggml/src/ggml-cpu/arch/arm/repack.cpp`: NEON/I8MM, non-SVE, eight activation rows and four output columns. Other row counts keep their existing GEMM paths. |
| M7/M8 activation preparation | `ggml/src/ggml-cpu/repack.cpp`: AArch64 NEON, Q8_K packed 4x8 preparation, K >= 1024 and multiple threads. Divide complete 256-element quantization blocks among workers before the existing barrier. This also covers the selected Q5_K 8x8 route. |
| ARGMAX | `ggml/src/ggml-cpu/ops.cpp` and `ggml-cpu.c`: AArch64 NEON scan; single rows with at least 16384 values use per-thread vocabulary ranges and cache-line scratch. Ties retain the last maximum; any NaN falls back to the original scalar scan. Small inputs and other architectures retain their existing paths. |
| Draft mixed quantization | The explicit [tensor type recipe](dspark-targetmix-tensor-types.txt) is used with `llama-quantize`; this changes weights and is separate from the runtime code. |

The ARM kernels apply to all eligible tensors, not only DSpark tensors. Unsupported shapes, types and instruction sets retain the existing paths. Shared-head padding requires the exact DSpark/CPU_REPACK conditions above; set `DSPARK_HEAD_PAD_DISABLE=1` before starting a process to disable it. Do not change that setting during graph reuse.

The integration omits the experiments' tensor capture callbacks, stage timers, tail3 trace and per-group selection switches. The three later Q4 M8 variants with no reliable benefit are not included. Cross-operator activation reuse, top-k reuse and Markov fusion have not been implemented.

## Probability-path integration (2026-11-05)

This repository also includes the subsequent probability-path changes. These complement the CPU kernels above and do not modify the existing GGUF files:

| Change | Implementation and scope |
| --- | --- |
| Parallel Markov projection | `common/speculative-dspark.h` replaces the per-token scalar loop with a persistent CPU backend and thread pool. Non-Q8-compatible weights retain a packed FP32 fallback; losslessly packable fallback weights use compact codes and scales. |
| Integer W2 dot product | When W2 can be reconstructed as Q8_0 with exact original F32 values, use native Q8_0 dot products. Products and block sums use integer arithmetic, with floating-point scales and output accumulation. W1 input quantization is lossy relative to the older FP32 projection. |
| Cached W1 lookup | Quantize the static table once during loading using the CPU backend's native Q8_0 quantizer, then select Q8 rows directly by predecessor token. Release the probability path's F32 copy only after cache creation succeeds. For this model the auxiliary table is 34 MiB instead of 128 MiB; original BF16 model storage is unchanged. |
| Draft sampling | Skip identity temperature scaling at T=1; separate the prefix selection scan from probability normalization while preserving RNG calls and division semantics. |
| Target rejection sampling | Validate that the full q array is in token-ID order before indexing it directly. Sparse or reordered q retains the original scatter behavior. |
| Conditional sampler copy | Clone the server sampler only if the context's rollback capability could require checkpoint restoration. |
| Readable W2 storage | Retain the existing loader fix that keeps Markov W2 out of CPU_REPACK, whose buffer does not support host tensor reads. |

The latest W1-cache experiment preserved all ten tested output sequences and proposal/acceptance counts relative to the integer W2 version. Its 94 MiB auxiliary-table reduction is established; a stable speed improvement is not. The earlier FP32-to-integer change can alter proposal distributions and generated sequences. Historical measurements and their comparison limits are in the workspace [probability report](../../experiments/dpsark_optimizer/prob-round2-20261105/REPORT.md), [integer W2 report](../../experiments/dpsark_optimizer/prob-int8-20261105/REPORT.md) and [W1 cache report](../../experiments/dpsark_optimizer/prob-w1-cache-20261105/REPORT.md). Those results are not new performance measurements of this integrated branch.

The current generation experiment settings are **thinking low, target T=1 / top_p=0.95, draft T=1 / top_p=1, probabilistic sampling**, with top_k=0, min_p=0 and neutral penalties. Historical greedy commands below describe the older validation only. For the probability setup, use the same CPU/model/KV arguments with:

```sh
--spec-type draft-dspark --spec-draft-sampling probabilistic \
--spec-draft-n-max 7 --spec-draft-p-min 0 --no-spec-draft-backend-sampling \
--jinja --reasoning on --reasoning-effort low
```

The completion request must also set `temperature: 1.0`, `top_p: 0.95`, `top_k: 0`, `min_p: 0.0`, `repeat_penalty: 1.0`, `frequency_penalty: 0.0` and `presence_penalty: 0.0`; render the prompt with thinking enabled and `reasoning_effort=low`. The DSpark full-vocabulary draft sampler uses the target temperature and sets draft top_p=1, top_k=0 and min_p=0. The CPU probability path disables backend sampling. Startup logs identify `Q8_0 integer dot` and the W1 cache size; incompatible W2 weights report `FP32 fallback`.

Native integration validation uses the separate `build-dspark-prob-cpu` directory with the CPU-only Release options below. Build `llama-server`, `llama-cli`, `test-speculative-dspark`, `test-sampling`, `test-speculative-rejection-sampling` and its `test-llama-archs` fixture, then run:

```sh
ctest --test-dir build-dspark-prob-cpu --output-on-failure -j 1 \
  -R '^(test-speculative-dspark|test-sampling|test-speculative-rejection-sampling)$'
```

The native build and all four selected CTest entries (three tests plus the generated-model fixture) passed. Five fixed probability requests produced exactly the same output tokens, generation counts and proposal/acceptance counts as the validated W1-cache experimental version. The server exited normally, and loader records confirm that the CPU and application libraries came from the new integrated build. These are correctness checks, not a new throughput benchmark. Windows and other CPU platforms were not tested by this port.

The [machine-readable probability validation](dspark-probability-validation.json) and workspace [port report](../../experiments/dpsark_optimizer/port-xmhuang6-prob-20261105/REPORT.md) record the source and binary identities. The existing local CPU kernel changes and the previously modified loader are preserved. The records retain the working-tree status at the time of validation; they do not describe a new distribution release.

## Draft quantization recipe

For the five-layer draft, gate/up use Q4_K. Q/K/V and FFN down use Q6_K in layers 2 and 4 and Q4_K elsewhere. Attention output, FC and Markov W2 use Q8_0. Markov W1, attention gates and confidence projection retain BF16; F32 norms remain unchanged. The vocabulary head is shared with the main model and is not stored in the draft file.

From this repository root, after building `llama-quantize`, choose a new output path:

```sh
./build-dspark-cpu/bin/llama-quantize \
  --tensor-type-file docs/dspark-targetmix-tensor-types.txt \
  ../../models/spark_2_6_opensource/Spark-X2.6-4B-DSpark.gguf \
  ../../models/spark_2_6_opensource/Spark-X2.6-4B-DSpark-Q4_K_M-targetmix-new.gguf \
  Q4_K_M 8
```

The already available model does not need to be quantized again. Historical model identities are:

| Model | SHA256 |
| --- | --- |
| Main Q4_K_M | `91cb2811aebb47ac916aa619d27d4cdd7f4143106b7917accb8ead4cd7f5cbd9` |
| Original BF16 draft | `2ee073c18870c66d7fbc8e124fd8d6c2364c8a83e4bbbc3ebaeaa9660eedeac8` |
| Mixed draft | `3ad60e51f1f17578b0c8c5c4bfd9ef01b6bed2527a6b97f3de4eaca261808e82` |
| Q8_0 backbone draft | `360bb19f510d7be4171fc4de3657b1000f85f2cf081da37425e67f181b88fd66` |

Quantization is lossy. The historical three prompts retained the same final output as the BF16 draft, but proposal acceptance changed; there is no broad task-quality equivalence claim. Retaining small BF16 projections also means the BF16 dot path can still execute, without implying that the original BF16 draft's speedup transfers to the quantized draft.

An alternative [Q8_0 backbone recipe](dspark-q8-backbone-tensor-types.txt) converts every layer's FFN gate/up/down and Q/K/V to Q8_0, alongside the existing Q8_0 attention output, FC and Markov W2. Markov W1, attention gates and confidence projection remain BF16, and F32 tensors retain their original bytes. The recipe contains 37 Q8_0 weights; the produced GGUF contains 37 Q8_0, seven BF16 and 14 F32 tensors. It is 700,596,512 bytes, versus 486,293,792 bytes for the mixed draft.

Generate this variant directly from the original BF16 draft, selecting a new output path:

```sh
./build-dspark-cpu/bin/llama-quantize \
  --tensor-type-file docs/dspark-q8-backbone-tensor-types.txt \
  ../../models/spark_2_6_opensource/Spark-X2.6-4B-DSpark.gguf \
  ../../models/spark_2_6_opensource/Spark-X2.6-4B-DSpark-Q8_0-backbone-new.gguf \
  Q8_0 8
```

The generated local model is `Spark-X2.6-4B-DSpark-Q8_0-backbone.gguf`. Select it with `-md` in the server command below. The main model and its shared output head retain their existing weights. [Local Q8 measurements](dspark-q8-20261104.md) report 27 formal requests, draft cost, acceptance and the limitations of the historical mixed-draft reference.

## Build and run

Build a native Release CPU configuration from this repository root:

```sh
cmake -S . -B build-dspark-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF -DGGML_OPENMP=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF
cmake --build build-dspark-cpu \
  --target llama-server llama-cli llama-quantize test-backend-ops test-quantize-fns -j 2
```

In the parent workspace, builds and model checks use `../op_optimizer/_shared/experiment_lock.py` to avoid overlapping other cooperating measurements. The validation configuration uses the workspace Ninja toolchain; record compiler and CPU flags when reproducing elsewhere.

Start the CPU-only greedy DSpark server using the existing mixed draft:

```sh
./build-dspark-cpu/bin/llama-server \
  -m ../../models/spark_2_6_opensource/Spark-X2.6-4B-Q4_K_M.gguf \
  -md ../../models/spark_2_6_opensource/Spark-X2.6-4B-DSpark-Q4_K_M-targetmix.gguf \
  -ngl 0 -ngld 0 -t 8 -tb 8 -td 8 -tbd 8 \
  -ctk f16 -ctv f16 -ctkd f16 -ctvd f16 -fa on \
  -c 2048 -np 1 -b 512 -ub 512 --fit off \
  --spec-type draft-dspark --spec-draft-sampling greedy \
  --spec-draft-n-max 7 --spec-draft-p-min 0 \
  --jinja --reasoning off --host 127.0.0.1 --port 8080
```

These parameters reproduce the tested short-context setup. A context capacity of 2048 does not mean the prompts contained 2048 tokens: the three historical inputs contained 34, 35 and 44 tokens.

## Validation and historical performance

The original experiments and exact patch provenance remain in the workspace [catalog](../../experiments/dpsark_optimizer/README.md) and [second-round report](../../experiments/dpsark_optimizer/round2-20261104/REPORT.md). Each experimental combination transition checked 75,281,408 floats bitwise; the new two-change combination retained output and acceptance counts for 72 uninstrumented requests.

On the original M5 Pro experimental branch, preparation parallelism plus ARGMAX improved aggregate DSpark generation throughput by 7.3% / 5.6% in ABBA / BAAB orders, draft time reached about 14 ms per round, and separately instrumented M8 target verification reached about 50 ms. Background process drift limits the precision of those gains. These are historical measurements, not a performance measurement of the combined `cpu_opt` branch.

DSpark has a per-round draft and verification cost, so benefit depends on accepted tokens per round. The historical same-round Python request gained about 23% over ordinary generation; the two low-acceptance natural-language requests remained slower. The existing Chinese ordinary-versus-DSpark divergence at token 8 is not fixed by this integration. Same-mode output equivalence does not establish equivalence between those two generation modes.

Regression entry points:

```sh
./build-dspark-cpu/bin/test-backend-ops -b CPU -o ARGMAX
./build-dspark-cpu/bin/test-backend-ops -b CPU -o MUL_MAT \
  -p '^type_a=(q[456]_K|bf16),type_b=f32,m=16,n=[1-9],k=(256|257|1024|2560|4096|10240),'
./build-dspark-cpu/bin/test-quantize-fns -v
```

Integration checks on 2026-11-04 passed with the new binaries:

- All five requested build targets completed successfully. The runtime loader logs identify the integrated CPU library for every test process.
- Backend regression sets passed ARGMAX 23/23, selected matrix multiplication 113/113, GEGLU 84/84 and attention 79/79; the complete quantization suite reported zero failures.
- An independent ARGMAX oracle checked 6600 synthetic rows and 2200 real-logit rows, including ties, signed zero, infinities, NaNs, noncontiguous rows and 1/2/3/8/16 threads.
- Q4/Q6 and Q5 integration checks retained bitwise-identical outputs and effective quantization workspaces compared with the audited experimental candidate: 8,432,592 and 554,400 bytes respectively. This also checks compatibility with the branch's existing single-row NEON quantization.
- All 12,582,912 main-model logits at matching M1/M2/M4/M8 boundaries were bitwise equal to the experimental candidate.
- The three greedy DSpark requests generated 96 tokens each with identical output tokens and proposal/acceptance counts to the historical quantized-draft gold. The integrated server exited normally.

The [machine-readable validation summary](dspark-cpu-validation.json) records the 2026-11-04 source, runtime and model hashes. Full logs, request output, exact-output dumps and the validation runner remain in `build-dspark-cpu/validation/`. Those checks covered local macOS CPU greedy behavior. The probability-path port has a separate record above; Windows, other CPUs and long context need their own validation. Historical percentages are not added to the existing CPU optimization gains.
