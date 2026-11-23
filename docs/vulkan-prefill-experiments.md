# Vulkan packed V prefill

Compatible token32 V4/V8 caches on Arc 140T use direct packed attention for 9..1024 query rows without a precision tail. Prefill also classifies mask tiles by default. This avoids materializing the full historical V cache before attention. Wider batches retain the gather path; decode and precision-tail dispatch keep their existing behavior.

Dirty V tiles read their exact F16 staging values, matching the gather path. The backend checks the prefill marker, tensor layouts, device capabilities and buffer limits before dispatch. Prefill uses one KV partition with the existing grouped-head kernel.

Set `LLAMA_VPLANE_PREFILL=0` to restore the gather path. Set `LLAMA_VPLANE_PREFILL_MASK_TILES=0` to disable only prefill mask classification. Both settings are read once per process. The Vulkan launcher defaults to batch512/ubatch512; the engine already had an ubatch512 default.

## Validation and measurement

`test-vulkan-vplane --contract` covers the query boundaries and explicit backend dispatch marker. `--gpu-prefill` checks 24 V4/V8 cases against an independent FP64 reference, including dirty staging, optional mask classification, repeated outputs and malformed markers. Existing `--gpu` checks retain decode coverage.

On Arc 140T, the initial cold full-prompt screen at batch2048/ubatch1024 measured 445.72 token/s at 32768 tokens and 284.06 token/s at 65536 tokens with packed V and mask classification. Matched controls measured 295.46 and 167.80 token/s. These are single-process screens; final model logits matched the controls bitwise. They do not establish throughput when appending 512 tokens to an existing history.

The follow-up ten-depth benchmark uses existing histories 0/512/1024/2048/4096/8192/16384/32768/65536/131072 and appends 512 tokens. Each depth runs three independent processes, each with four native timings excluding the first. A fourth process is reported separately when initial throughput spread exceeds 5%. Full128K requires capacity for 131072 historical plus512 appended tokens; inputs must not be shortened.

`llama-bench --prefill-check-prefix PATH` uses history seed4321 and query seed1234+repeat_index, saving the complete final vocabulary logits outside timing. Loading, native warmup, history construction/restoration and file writes are excluded. `--kv-tail-tokens N` records a configured precision tail in native output; the selected packed-prefill benchmark uses no precision tail.
