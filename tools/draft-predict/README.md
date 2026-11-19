# Standalone adaptive draft prediction

`llama-draft-predict` fits acceptance observations and recommends a draft depth from D0-D7. D is both the actual generated draft count and the verified draft count; the target processes D+1 rows. This replay tool loads no models. The server can apply the same predictor through the opt-in [runtime feedback loop](../server/README-adaptive-draft.md), which measures fresh costs on every model load instead of reading a saved calibration file.

The reusable C++ interface is `common/speculative-adaptive.h`. Its library target, `llama-speculative-adaptive`, depends only on the C++ standard library. The command-line tool also uses the repository's existing JSON library.

## Build and test

From the repository root:

```sh
cmake -S . -B build-adaptive -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build-adaptive --target llama-draft-predict test-speculative-adaptive
ctest --test-dir build-adaptive -R '^test-speculative-adaptive$' --output-on-failure
build-adaptive/bin/llama-draft-predict --demo --current-draft 3
```

The demo generates a reproducible synthetic fixture with different block distributions and a change from difficult to easy acceptance. It illustrates prediction and forgetting, not actual inference throughput or a measured controller speedup. It does not apply the recommendations to its fixture schedule.

A direct build avoids model/backend dependencies entirely:

```sh
c++ -std=c++17 -O2 -Icommon -Ivendor \
    common/speculative-adaptive.cpp tools/draft-predict/draft-predict.cpp \
    -o /tmp/llama-draft-predict
c++ -std=c++17 -O2 -Icommon \
    common/speculative-adaptive.cpp tests/test-speculative-adaptive.cpp \
    -o /tmp/test-speculative-adaptive
/tmp/test-speculative-adaptive
```

## Replay real records

First obtain a cost report from the separate `llama-dspark-calibrate` tool. This predictor reads its version 1 or 2 `cost_table`: `depth`, `samples`, and `round.mean_us`/`round.sd_us`. Optional row counts must match actual D/D+1 execution. D0 is ordinary generation with exactly zero accepted draft tokens and one committed token; the candidate set is D0-D7. Missing costs stay unavailable and are never guessed.

The calibration report must have `complete: true` and `history_tokens`. Fixed-prefix repetitions and `accepted_total` in that report are never used as independent acceptance observations. Synchronized startup cost measurements remain estimates for an async production path; real complete-round observations update the corresponding cost estimate.

Create a JSONL file with one real generation round per line:

```jsonl
{"draft":3,"accepted":1,"committed":2,"round_us":32000,"context_tokens":512}
{"draft":3,"accepted":3,"committed":4,"round_us":33100,"context_tokens":512}
{"draft":2,"accepted":1,"committed":2,"round_us":26000,"context_tokens":512,"probe":true}
{"draft":3,"accepted":1,"committed":1,"round_us":19000,"context_tokens":512,"censored":true}
```

Then run:

```sh
build-adaptive/bin/llama-draft-predict \
    --calibration costs.json --rounds rounds.jsonl \
    --current-draft 3 --seed 0 > prediction.json
```

`--trace-every 16` emits a compact JSON report every 16 records and a final report. Replay evaluates the logged depths; it does not pretend that a recommendation was executed. Without `--current-draft`, comparisons use the last logged depth, or D3 before any rounds.

| Record field | Meaning |
| --- | --- |
| `draft` | Actual generated/verified D, from 1 through the configured maximum |
| `accepted` | Observed continuous accepted prefix, from 0 through D |
| `committed` | Actual submitted token count; omission infers A+1 for a normal round |
| `round_us` | Positive complete cycle wall time, including waits/cache work; omission supplies no cost observation |
| `censored` | No rejection is known after the observed accepted prefix, for example due to a terminal limit; requires explicit positive `committed` |
| `probe` | Mark a real exploration round so its entire elapsed time is included in probe diagnostics; requires a timing |
| `context_tokens` | Optional cost-bucket label; when present it must equal the calibration's `history_tokens` |

For D3/A1, position 1 supplies a success, position 2 a failure, and position 3 no conditional observation. A censored D3/A1 supplies only the position-1 success. A shortened terminal round updates acceptance successes and the token clock, but not normal-round cost or the direct-round switching threshold.

One instance owns one explicitly chosen context-cost bucket and one model/draft precision, KV format, backend and sampling configuration. Prepare separate reports/instances for different configurations and context buckets. The tool does not select buckets or interpolate/extrapolate context costs. Future callers must map real context lengths to calibrated buckets deliberately. Do not mix adjacent-round overlapping host phase timers and call their sum a complete cycle time.

## Acceptance model

For each actual D and position i:

```text
c[D,i] = P(A >= i | A >= i-1, actual depth D)
logit(c[D,i]) = a + b*(i-1) + u[D] + v[D,i]
S[D,i] = product(c[D,1..i])
E[A_D] = sum(S[D,1..D])
```

The shared intercept and position trend transfer information to unseen configurations. Depth and cell residuals retain block-specific differences. Proper zero-centered Gaussian priors on the residuals regularize this otherwise redundant parameterization. No constant conditional acceptance or decreasing conditional-rate assumption is imposed. The survival products are automatically nonincreasing.

Each observation updates only its actual D. A longer-block record is not copied into a shorter-block row: changing the actual block can change the prefix distribution. Measured data determine observed cells; predictions for unmeasured cells keep residual uncertainty even after many observations at other depths.

The implementation fits a regularized logistic MAP with damped Newton steps and samples a joint Laplace approximation using the full parameter precision. All candidate depths share each posterior draw, preserving their parameter correlation in gain comparisons. Intervals describe uncertainty about expected acceptance/throughput, not the range of individual realized accepted lengths. These are approximate model intervals, not a proven frequentist coverage guarantee; validate their coverage on independent real traces before enabling automatic serving decisions.

Observations decay by committed-token age with a default half-life of 512 tokens. `--half-life-tokens 0` disables likelihood/cost-weight forgetting for stationary replay/tests; the separate 512-token switch-qualification window still expires old rounds. The average accepted length is a derived quantity, not another independent label. A prior acceptance of 0.65 is an initial location, not an assumption that all positions have that probability.

## Costs and decisions

Mean, population variance V and effective count w of complete-round costs are maintained per actual D. Joint model draws are combined with positive lognormal cost draws. Their relative variance is `V / (max(1,w-1) * mean^2) + rho^2 / clamp(w,rho^2,1)`, with rho defaulting to 3%. Aging preserves V and reduces w. Even a zero-scatter prior loses precision once w falls below one window; the aging floor term is capped at 1 and fresh measurements can restore precision. Independent cost errors across depths are an explicit approximation; shared timing correlation and systematic startup/production bias are not inferred. Reported token rate is `(1 + E[A_D]) / (cycle_time + round_overhead_us)` within each parameter draw. Terminal rounds are outside that normal-round reward formula.

The default interval uses the 5th and 95th percentiles, a central 90% approximate interval. A switch requires the candidate's gain lower bound to exceed 3% and at least eight uncensored rounds at that actual D ending within the last 512 committed tokens. This integer qualification count is separate from the decayed likelihood weights. Positive-depth upward and downward moves use the same rule; all calibrated depths are considered.

A candidate receives a `probe` recommendation when its mean relative gain exceeds 3% and its probability of exceeding 3% is at least 60%. Candidates are ranked by mean gain. From a positive incumbent, if the usual gain rules do not qualify, positive candidates with no live cost or a cost aged at least 128 committed tokens can trigger `refresh_cost`: gain upper bound >3%, probability above 3% at least 20%, and E[max(gain−3%,0)] >3%. Rank these by expected positive excess gain. The upper bound alone is insufficient; cooldown exclusions and reentry cost still apply. The previous 2% timing allowance and `--probe-credit-us` option have been removed. Output schema version 2 previously replaced `probe_credit_us` with `runtime_us`/`probe_us` and adds `recent_rounds` for each candidate.

`evaluate()` is read-only and does not execute a probe or apply a recommendation. Its optional excluded-depth list lets a caller enforce cooldown or focus on one ongoing probe. Its third argument, `round_overhead_us`, supplies a nonnegative shared per-round overhead for throughput comparisons; reported `round_us` remains the inference cost alone. The standalone CLI leaves this argument at zero and reports that value in its output. The server supplies smoothed boundary work divided by its 16-round normal window for every candidate, including those in eight-round probes. The server also owns the 16-round episode limit and per-candidate cooldown/backoff; the standalone replay tool does not simulate those state transitions or turn recommendations into observations. It evaluates the actual logged records only. From D0, gain-triggered probing requires probability at least 80%, and uncertainty-only cost refresh is disabled. The server additionally requires two changed ordinary-cost windows before reentry and can promote a positive probe at 16 rounds using a 75% probability threshold plus measured-rate checks; these runtime state rules are not simulated by this standalone predictor.

The server records acceptance per round and uses `observe_cost_window()` after draining a fixed-D window. Each complete window supplies one timing observation; the first replaces the startup prior, later windows use weighted updates. Cancellation and incomplete windows still contribute to runtime/probe accounting, but not cost learning. Between-window decision work is accounted separately by the server. Aging reduces cost evidence weight without shrinking population variance, preventing uncertainty collapse when a candidate goes unobserved. A `need_cost` result preserves the current D until its cost is supplied.

The output contains conditional/prefix probabilities, weighted success/failure counts, recent normal-round qualification counts, accepted-length/cost/rate/gain intervals, probability of exceeding the margin, observed runtime/probe durations, `round_overhead_us` and the recommended action. Input errors return a nonzero status. Model and timing settings are fixed within one process; replay the original records to reproduce a fit.

## Validation scope

`test-speculative-adaptive` checks right censoring, recovery of the supplied heterogeneous D3 survival curve, retained unseen-tail uncertainty, distinct actual-block distributions, upward/downward decisions, direct-evidence gating, missing costs, gain-triggered probing without credit, recent evidence and expiry, stale cost uncertainty, time accounting, token-based forgetting, phase changes, input validation and reproducibility. It uses no model files, GPU or remote machine. `test-server-component` additionally checks the runtime window scheduler. Full-model speed measurements remain separate work.

## D0 and schema version 3

Predictions now include D0 at index 0, so array indices equal actual D. `--current-draft 0` and D0 round records are supported. D0 has no position model; ordinary rounds age the positive-depth evidence without inventing rejections. Old calibration tables lacking D0 remain readable, but D0 then has no available cost and cannot be selected.

A D0 fallback uses threshold `-margin/(1+margin)`: its mean gain must exceed that threshold with probability at least 60%; eight recent ordinary rounds qualify promotion. This prefers ordinary generation when DSpark does not retain the margin, without requiring D0 itself to be 3% faster. Positive candidates retain the 95% lower-bound switch condition. Schema 3 replaces `probability_gain_gt_margin` with `gain_threshold` and `probability_gain_gt_threshold`.

`evaluate()` also accepts a restart cost and token horizon. From D0, positive candidates' cost per committed token includes `reentry_cost_us/gain_horizon_tokens`. The CLI reports zero restart cost because a round log alone does not describe the paused live contexts; the server computes it from its resume state. `observe_cost_window()` can subtract measured one-time transition work from cost learning while retaining the entire duration in runtime/probe totals.
