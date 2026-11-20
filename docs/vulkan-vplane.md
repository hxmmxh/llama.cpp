# Token32 V8/V4 attention

The Vulkan backend includes the measured packed token32 V consumer for Intel Arc
140T (device 0x7d51). It has an inline F16 decode path and an integer PV path.
The private constructor and input checks are in
`ggml/src/ggml-vulkan/ggml-vulkan-vplane.h`. The operator also reads the experimental token32 V8 and V4 model caches. Compatible single-sequence Spark Q8_0/Q8_0 and Q4_0/Q4_0 caches on Arc 140T use token32 by default. SWA K/V remain F16 by default. Other devices, model shapes and cache formats use the ordinary layout. Add `--no-kv-v-token32` to select the ordinary channel-block layout explicitly.

Build with Vulkan integer-dot support. Run with `GGML_VK_DISABLE_COOPMAT=1`.
Call `ggml_vk_flash_attn_vplane(ctx, q, k, packed_v, mask, bits, scale)`, then
check `ggml_backend_supports_op()` on the selected Vulkan backend before
building or scheduling the graph. A null result means the input contract is
unsupported. A device rejection must be handled by the caller; this private
packed input has no CPU or other-backend fallback.

## Layout and scope

- Q: F32, D=256, eight query heads, interleaved heads before permutation.
- K: Q8_0 for V8 or Q4_0 for V4, D=256, two KV heads, channel32 quantization.
- V: contiguous I8 storage `[chunk_bytes, kv/32, 2]`, with 8704 bytes per V8
  chunk or 4608 bytes per V4 chunk. Each chunk is 32 tokens for one KV head.
  The first 512 bytes contain 256 F16 scales. The rest contain signed Q8 codes
  or Q4_0 nibbles (decoded as code minus eight).
- Quantize each fixed channel over 32 consecutive tokens using the canonical
  GGML Q8_0/Q4_0 reference quantizer. Store scales and codes separately.
- With channel `d`, token `t`, and `G = config.groups`, let `g=t%G`, `c=t/G`.
  Q8's byte offset is `512+(g*64+d/4)*(128/G)+(c/4)*16+(d%4)*4+c%4`.
  Q4's byte offset is `512+(g*64+d/4)*(64/G)+(d%4)*(16/G)+c/2`, with nibble
  shift `4*(c%2)`. Scales are stored at byte `2*d`.
- Obtain `config` from `ggml_vk_vplane_tuning(kv, rows, bits)` before packing;
  changing groups requires repacking. Do not relabel existing channel32 data.
- V8 and V4 accept M=1 through 8.
  KV lengths are multiples of 32 through 1048576. Q/K use the exact
  interleaved-head strides checked by the helper. Nonzero view offsets,
  sparse hints, sinks, ALiBi and softcaps are rejected. Optional masks are
  contiguous F16 `[kv, M]`. Accumulation precision is fixed to F32.

For the prepacked benchmark, KV>=4096 uses 64 threads per workgroup and split 64/M.
V4 uses row2; M1 and shorter-context M2 decode V to F16, while M4/M8 and
M2 at KV>=65536 use integer PV. V8 normally uses row2 and F16 decode;
M8 below KV65536 uses integer PV. At KV>=65536, V8 M2/M8 use row1 and
F16 decode. The requested split is rounded to full 32-token blocks by the
existing reduction path. Short caches retain WG128/row4 and split16/M.

## Incremental V8 and V4 cache

Use `-ctk q8_0 -ctv q8_0 -fa on` for V8 or
`-ctk q4_0 -ctv q4_0 -fa on` for V4. `--kv-v-token32` remains accepted and `--no-kv-v-token32` disables the packed path. Set
`GGML_VK_DISABLE_COOPMAT=1` for a single-sequence Spark D256/GQA4 context
with KV layers on an Arc 140T. SWA K/V default to F16, so only global layers are packed. The explicit flags `--cache-type-k-swa f16 --cache-type-v-swa f16` preserve that selection. `llama-bench` uses the same defaults and accepts both token32 switches.
Packed V supports 1 to 8 queries per microbatch, including target verification
with 1 to 7 speculative draft tokens. The packed model cache keeps the same
four-way layout across query widths, including at 64K and longer contexts.
Historical V is stored in the selected packed format. Each layer retains up to eight recent
32-token tiles in F16 so small appends and speculative rollback can reuse exact
values without re-quantizing them repeatedly. This staging size is independent
of context length. K uses channel32 Q8_0 or Q4_0 storage and Hadamard rotation.
V is quantized across tokens without Hadamard rotation.

Per-query mask classification is enabled by default for token32 K4V4 decode without a precision tail.
The shader skips per-element checks for fully visible or fully masked blocks.
Set `LLAMA_VPLANE_DECODE_MASK_TILES=0` to disable it or `=1` to enable it explicitly on either packed format.

The GPU merges new V values and previous values into temporary dirty tiles,
then updates the packed cache. For a dirty tile crossing the query's mask
boundary, packed attention reads the merged values directly so future
candidates cannot affect visible values through the shared scale. Wider
microbatches unpack one layer into graph workspace for standard attention.
The state format saves packed tiles and recent staging rows. Restore copies
packed bytes when cell placement preserves the tile boundaries. It repacks
when the destination groups differ and can import F16 rows from the original
token32 cache. V4 and V8 have distinct packed state identifiers.

Add `--kv-tail-tokens 1024` for a rolling F16 K/V tail. Each query uses its
own position to select tail rows after applying the causal or SWA mask. The
mixed-precision Vulkan kernel reads older V directly from token32 storage.
Only the configured tail and the recent tile staging use persistent F16.
Decode and short verification batches use split-K packed attention with
per-row F16 selection. Set `LLAMA_VPLANE_TAIL_TILED_PREFILL=1` to use
that path for prefill microbatches up to 128 queries. Larger microbatches
use the precision-tail reference kernel.
If the tail covers the full sliding window, the bounded SWA cache stores
one F16 copy and uses native Flash Attention. The global F16 tail refills
after state restore.

## Evidence and validation

The measured source is experiment `20261117-193011-vulkan-s17-kv-five-v4`,
SHA256 `7fe32d31a8fb746f50098266648ae6d9437a73b5286692416f8d8acdaa8cc67c`.
Its three processes cover M1/2/4/8 at 32K/128K plus short-mask controls.
All eight winning V4 configurations beat every valid K4/V16 control by more
than 2% in each process. These are complete SDPA host times with prepacked
caches; they exclude cache production and are not model throughput results.
Other accepted lengths are functional extensions of the same block layout,
not additional measured performance claims.

`test-vulkan-vplane --contract` checks malformed inputs and exhaustive nibble
expansion on CPU. `test-vulkan-vplane --gpu [reference.bin]` checks 20 packed
V8/V4 cases on Arc140T, exact upload/readback, repeated outputs, an independent
FP64 reference, and optional full historical GPU outputs. It also checks six mixed
V8/F16 cases against an independent FP64 reference. The reference file
contains consecutive F32 outputs in test shape order, V8 then V4 per shape.
Numerical limits are NRMSE 3% / max absolute 0.02 against decoded-input FP64,
and NRMSE 1% / max absolute 0.01 against historical GPU outputs.

The private constructor marks the attention operation, not its input tensor.
The packed dispatch is validated separately from ordinary Flash Attention;
no process-wide experimental tuning variables are required. Plain F16 and
channel-quantized caches retain the original dispatch and shader path.
