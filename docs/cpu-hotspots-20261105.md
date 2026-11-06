# CPU hotspot optimizations

This change starts from `ce1e5fd2c6955e2e54536adf39959f603057b495` and preserves model formats, quantization, sampling parameters and public APIs.

- Q4_K GEMV uses four VNNI integer accumulation chains for the 8-channel repacked layout. The specialization requires AVX-VNNI, excludes AVX512, and handles one output row. Existing paths remain for other cases. Float scaling and FMA order are unchanged.
- F16 attention extends the existing GQA4 helper to M7/M8. It groups queries by token and KV head while preserving each head's arithmetic order. The specialization requires D256, GQA4, one sequence, at least 512 KV entries and enough jobs for all workers. NUMA, sinks, ALiBi, softcap and other unsupported configurations retain the existing path.
- Top-p retains its 256-candidate heuristic. When more candidates are needed, it sorts only the probability buckets required to complete the nucleus and avoids rescanning that completed prefix. Token ordering, probabilities and RNG behavior are preserved.
- The concurrently integrated [Q8 position reuse](q8-reuse43-20261105.md) is included in the combined validation.

On Core Ultra 7 255H, ordinary generation measured 21.2214 tokens/s versus the same-configuration historical reference of 20.2682; probabilistic DSpark measured 23.9722 versus 22.7101. These are cross-session observations of +4.70% and +5.56%, not an interleaved baseline experiment. Against the historical Q8-only implementation, DSpark changes by only +0.52%, which does not establish a stable incremental improvement.

All 16 generated token sequences match their respective historical references. DSpark proposal, acceptance and round counts also match. Kernel, sampler and full-logit checks passed. All slow forward samples are retained: the combined D512 M8-all mean is 1.41% slower than the historical reference, with a shared third-process slowdown across several candidates. Single-kernel improvements must not be substituted for whole-model gains.

The [experiment report](../../experiments/cpu-hotspots-ce1e5fd2-20261105/REPORT.md) contains configurations, isolated results, raw evidence, fallback compilation, source hashes, the candidate package and the independent audit. No distribution release or remote daily entry point was replaced.
