# Vulkan KV precision tail

Use `--kv-tail-tokens N` with a single-sequence causal Spark context. The default is
`0`. The initial implementation supports Q4_0, Q4_1 and Q8_0 for either K or V,
and stores the tail in F16. Enable Flash Attention explicitly.

```sh
llama-cli -m model.gguf -ngl 99 -fa on -ctk q4_0 -ctv q4_0 --kv-tail-tokens 1024
```

For the server, select `--parallel 1`. API callers can set
`llama_context_params::kv_tail_tokens`.
On Arc 140T, the same flag can be combined with the
[token32 packed cache](vulkan-vplane.md) for Q4_0 K/V or Q8_0 K/V.

Each layer keeps its existing quantized K/V plus a GPU-resident F16 ring. K and V
enter both caches after any Hadamard transforms used by the cache format. The allocation in each
cache group is capped at its body capacity and holds up to `N + ubatch - 1` rows.
This lets the first query still see its complete tail after the current microbatch
has written its rows. The SWA group caps N at its attention window. For the
token32 cache, a standard bounded SWA window lives entirely in F16 when N
covers the window and no explicit SWA cache types are set. Set
`--cache-type-k-swa f16 --cache-type-v-swa f16` to select F16 SWA explicitly.

The graph gives each query a mapping from body rows to F16 slots. The
`FLASH_ATTN_EXT_TAIL` operation first applies the existing causal/SWA mask,
then reads one K/V representation per visible row. It consumes quantized blocks
directly and accumulates one FP32 online softmax and weighted value sum. The
existing inverse V rotation follows attention. The CPU version implements the
same operation for numerical comparison. Vulkan supports single-stream heads
through dimension 512. The fused operation runs on the KV cache device.

Suffix removal frees ring slots for the next append and preserves surviving
exact rows. Position changes invalidate affected attention history in the ring,
and the current state format restores the quantized body. After restore, new
writes refill the tail.
