# CPU hotspot round 2 (2026-11-06)

The baseline is the previous combined working tree, including its uncommitted Q4 GEMV, Q8 tiling, attention and sampling changes. The new x86 paths share packed Q4_K weights across two activation rows for M4/M8 VNNI GEMM and share F16 key conversion across two GQA4 query heads. Each output retains its accumulation order; ISA and shape fallbacks remain in place.

## Combined integration

The commit collects the retained changes from both hotspot rounds and the Q8 integration on top of `ce1e5fd2c6955e2e54536adf39959f603057b495`.

| Optimization | Implementation | Scope |
| --- | --- | --- |
| Q8 M7/M8 position reuse | [sgemm.cpp](../ggml/src/ggml-cpu/llamafile/sgemm.cpp) | Two output channels share weights over 4+3 or 4+4 positions; signed-byte VNNI, F16C, 16 vector registers, with a worker-occupancy guard |
| Q4_K single-row integer dot | [repack.cpp](../ggml/src/ggml-cpu/arch/x86/repack.cpp) | Four VNNI accumulation chains on the 8-channel repacked GEMV path; non-AVX512 |
| Q4_K M4/M8 weight reuse | [repack.cpp](../ggml/src/ggml-cpu/arch/x86/repack.cpp) | Two activation rows share weight loads and nibble unpacking; VNNI, non-AVX512 |
| F16 GQA4 M7/M8 scheduling | [ops.cpp](../ggml/src/ggml-cpu/ops.cpp) | Token/head-group jobs for D256, one sequence, at least 512 KV entries and enough jobs for all workers |
| F16 GQA4 pair dot | [ops.cpp](../ggml/src/ggml-cpu/ops.cpp) | Two query heads share key loads and half conversion; AVX2/F16C, non-AVX512 |
| Top-p nucleus sorting | [llama-sampler.cpp](../src/llama-sampler.cpp) | Stop sorting lower-probability buckets once the nucleus is complete, preserving the existing 256-candidate heuristic |

The four implementation files match the [tested combined source hashes](../../experiments/cpu-hotspots-round2-20261106/source-identity.json). The commit also includes attention and sampling regression cases. Earlier committed Q6 M8, Q8 PV and other CPU work remains in the parent history. Rejected Q4 reuse8, host-buffer reuse and split exp/sum experiments are not part of this integration. Model formats, quantization, sampling parameters and public APIs are unchanged.

Validation includes the existing Q4 numerical/oracle/canary checks, ISA fallback builds, 646-case attention comparisons, sampler differential tests and fixed-logit checks, followed by complete-output and acceptance checks in ordinary and DSpark generation. Committing this exact tested source does not add another performance run or establish ARM throughput gains.

## Performance observations

Ultra 7 historical comparisons observe ordinary generation 21.2214 to 21.2358 token/s (+0.07%, effectively unchanged), and probabilistic DSpark 23.9722 to 24.9493 token/s (+4.08%). All 16 requests match previous output tokens and applicable acceptance counts. These are cross-session observations with no baseline rerun; the two DSpark processes differ by about 5.9%. Some M1 forward means and large-M microkernel samples regress, so this is not a universal performance claim.

Four additional DSpark processes with the same frozen runtime produced 25.9924, 26.1635, 26.1632 and 26.2607 token/s (aggregate 26.1446; max/min−1 1.03%; sample CV 0.43%). All six candidate processes aggregate to 25.7336 token/s, with max/min−1 8.28% and sample CV 2.96%, retaining the original slow sample. The six-process +7.35% versus the historical reference remains a cross-session observation, not a stable causal gain. All 16 additional requests match complete tokens and acceptance counts; no source changes or baseline rerun were made. [Follow-up measurements and independent audit](../../experiments/windows-cpu/windows_core_ultra_7_255h/20261106-hotspots-round2-repeat/REPORT.md).

[Full results, raw samples, failed candidates and package](../../experiments/cpu-hotspots-round2-20261106/REPORT.md), [incremental patch](../../experiments/cpu-hotspots-round2-20261106/combined/round2.patch), [independent code review](../../experiments/cpu-hotspots-round2-20261106/host/KERNEL-REVIEW.md). Performance validation used the working tree before this combined commit; the recorded uncommitted state in those experiment manifests is historical. No published distribution was replaced.
