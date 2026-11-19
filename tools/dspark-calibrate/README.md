# DSpark startup cost calibration

`llama-dspark-calibrate` loads one target and one draft model and builds a D0-D7 cost table before serving requests. It is a standalone entry point for archiving and offline analysis. The server can run the same measurement API on every model load with [--spec-adaptive](../server/README-adaptive-draft.md).

```sh
cmake --build build --target llama-dspark-calibrate
LLAMA_DSPARK_DEVICE=1 LLAMA_DSPARK_DENSE_Q=1 \
build/bin/llama-dspark-calibrate \
    -m target.gguf -md draft.gguf \
    --device Vulkan0 --device-draft Vulkan0 -ngl 99 -ngld 99 \
    -t 12 -c 2048 -b 512 -ub 128 -fa on \
    -ctk q4_0 -ctv q4_0 \
    --calibration-history 512 --calibration-warmup 2 \
    --calibration-repetitions 5 --calibration-output costs.json
```

Use the device IDs reported by the current build's `--list-devices`. On Windows set the two environment variables before invoking the executable. The device path is optional; without it the existing CPU Markov/sampler path is used. Target and draft KV types retain their independent common CLI settings. No model is downloaded by this example.

## Measurement

The supplied prompt text (`-p` or `-f`) is tokenized and repeated to fill the requested history. A BOS token is included when available. One additional token is kept as the anchor. This is a fixed-prefix cost measurement, not a chat-quality evaluation. The complete token fixture is saved in JSON.

The target prefix is evaluated once and its extracted features populate the draft cache. Both KV states are saved. D7 reuses the initial speculator; D1-D6 use fresh speculators against the same loaded weights and contexts. Each speculator has its actual draft maximum set to D, so the draft forward uses D rows and the target verification uses D+1 rows. Non-anchor-first drafters, trained limits below seven, synthetic acceptance, confidence truncation and multiple sequences are rejected.

Before each warmup or measured round, both prefixes and the sampling RNGs are reset. The real probabilistic DSpark path executes these stages:

| JSON field | Scope |
| --- | --- |
| `draft` | Draft forward, Markov chain, sampling and probability collection, through completion |
| `draft_cleanup` | Remove temporary draft KV rows |
| `target` | Build the D+1 token batch and complete target verification |
| `context` | Inject all D+1 target feature rows into the draft cache, through completion |
| `sampling` | Target rejection/residual/bonus sampling |
| `rollback` | Acceptance bookkeeping, remove rejected target/draft suffixes and drain both contexts |
| `round` | Sum of the preceding mutually exclusive host-wall intervals |

Each stage explicitly synchronizes. Async handoff is disabled for calibration even when `LLAMA_DSPARK_ASYNC` is enabled. These are synchronized startup costs, not uninterrupted async generation throughput or GPU-only kernel timings. Model loading, speculator initialization, prefix preparation/restore, sampler reset, warmups and validation are outside the round timings. Total loading and calibration wall times are reported separately.

Sampling defaults are target T=1/top-p=0.95/top-k=0/min-p=0 and draft T=1/top-p=1. A fixed seed gives repeated measurements of the same state and random path, not independent acceptance samples. Per-depth accepted counts are diagnostic; they do not estimate task-wide acceptance rates. A different prefix, KV format, context length, placement or sampling path can change the costs.

Every round checks full target logits for finite values and full draft probabilities for valid values and normalization. Raw samples, token IDs, mean/median/min/max and sample SD are retained. `repeat_tokens_match` records token repeatability without presenting it as task accuracy. Output files must be new; errors return a nonzero exit code without publishing a complete table.

## Calling after model loading

`common/speculative-calibration.h` exposes `common_speculative_calibrate(params, target, draft, prompt, options)`. Both contexts must be empty, support one sequence and have capacity for the prefix plus eight positions. Initialize their output capacities for eight target positions before loading. Pass a tokenized prompt containing the history and one final anchor. There must be no live speculator attached to these contexts.

The function restores the same saved prefix for every measurement and leaves both KV caches empty on success or failure. Instantiate the normal generation speculator after it returns. The returned raw records can be reduced to a cost table by the caller; the tool demonstrates JSON aggregation. This API does not choose or change the serving draft length. Callers can set `include_ordinary=false` and `max_draft` in D1-D7 to measure only the configured speculative depths; defaults retain D0-D7.

## D0 ordinary decode

The default table now includes D0 before initializing the DSpark speculator. D0 executes one target row and ordinary target sampling, with no draft forward, Markov head, probability collection or draft-feature injection. Its output is one token per measured round.

Use `--calibration-ordinary-only` to measure D0 alone. This mode requires only `-m target.gguf`, does not load a draft model, and reports zero draft/context stages. Keep the same target KV types, history, sampling, threads and device when comparing with D1-D7. Output schema version 2 records the selected mode and depth order.

```sh
build/bin/llama-dspark-calibrate -m target.gguf --device Vulkan0 -ngl 99 \
    -t 12 -c 2048 -b 512 -ub 128 -fa on -ctk q4_0 -ctv q4_0 \
    --calibration-ordinary-only --calibration-history 512 \
    --calibration-output ordinary-cost.json
```

D0 is a synchronized fixed-prefix measurement; its token rate is not a continuous-generation benchmark. With default D0-D7 calibration, the target prefix is prepared again after D0 so speculative feature extraction cannot contaminate the D0 timing. The incoming target must be freshly initialized, without an earlier speculator having enabled feature extraction.
