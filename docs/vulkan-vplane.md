# Token32 V8/V4 attention

The Vulkan backend includes the measured packed token32 V consumer for Intel Arc
140T (device 0x7d51). It has an inline F16 decode path and an integer PV path.
The private constructor and input checks are in
`ggml/src/ggml-vulkan/ggml-vulkan-vplane.h`. This is a prepacked operator entry;
ordinary `-ctv q8_0` / `-ctv q4_0` caches retain their channel-block layout.
The model cache writer, partial-token blocks, rollback and state serialization
do not yet produce this layout.

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
- M=1/2/4/8 and multiples of 32 KV positions through 131328 are accepted.
  The KV768/M3 sliding-mask regression is also accepted. Q/K use the exact
  interleaved-head strides checked by the helper. Nonzero view offsets,
  sparse hints, sinks, ALiBi and softcaps are rejected. Optional masks are
  contiguous F16 `[kv, M]`. Accumulation precision is fixed to F32.

For KV>=4096, workgroups have 64 threads and requested split is 64/M.
V4 uses row2; M1 and shorter-context M2 decode V to F16, while M4/M8 and
M2 at KV>=65536 use integer PV. V8 normally uses row2 and F16 decode;
M8 below KV65536 uses integer PV. At KV>=65536, V8 M2/M8 use row1 and
F16 decode. The requested split is rounded to full 32-token blocks by the
existing reduction path. Short caches retain WG128/row4 and split16/M.

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
FP64 reference, and optional full historical GPU outputs. The reference file
contains consecutive F32 outputs in test shape order, V8 then V4 per shape.
Numerical limits are NRMSE 3% / max absolute 0.02 against decoded-input FP64,
and NRMSE 1% / max absolute 0.01 against historical GPU outputs.

The private constructor marks the attention operation, not its input tensor.
The packed dispatch is validated separately from ordinary Flash Attention;
no process-wide experimental tuning variables are required. Plain F16 and
channel-quantized caches retain the original dispatch and shader path.
