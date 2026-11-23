#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>

// Private packed-cache contract. Ordinary Q4_0/Q8_0 tensors keep their layout.
constexpr int32_t GGML_VK_VPLANE_TAG = 0x56504c31;
constexpr uint32_t GGML_VK_FA_VPLANE = 128u;
constexpr uint32_t GGML_VK_FA_VPLANE_INTEGER = 256u;
constexpr uint32_t GGML_VK_FA_VPLANE_Q4 = 512u;
constexpr uint32_t GGML_VK_FA_VPLANE_MASKED_F16 = 1024u;
constexpr uint32_t GGML_VK_FA_VPLANE_F16_TILE_MAP = 2048u;
constexpr uint32_t GGML_VK_FA_VPLANE_TAIL = 4096u;
constexpr uint32_t GGML_VK_FA_VPLANE_MASK_TILES = 32768u;
constexpr uint32_t GGML_VK_FA_VPLANE_PREFILL = 65536u;
constexpr int64_t GGML_VK_VPLANE_TILE_RECORD = 36;
constexpr int64_t GGML_VK_VPLANE_TILE_BYTES = 8704;
constexpr int64_t GGML_VK_VPLANE_V4_TILE_BYTES = 4608;
constexpr int32_t GGML_VK_VPLANE_TILE_HAS_TAIL = 1;
// Packed buffers and F16 gather offsets remain below the 32-bit shader address limit.
constexpr int64_t GGML_VK_VPLANE_MAX_KV = 1024 * 1024;

static inline constexpr int64_t ggml_vk_vplane_tile_bytes(int bits) {
    return bits == 4 ? GGML_VK_VPLANE_V4_TILE_BYTES : GGML_VK_VPLANE_TILE_BYTES;
}

// Encoded mask class for a block of 32 F16 entries.
static inline int32_t ggml_vk_vplane_tile_mask(const ggml_fp16_t * mask) {
    uint16_t any = 0;
    uint16_t all = UINT16_MAX;
    for (int col = 0; col < 32; ++col) {
        any |= mask[col];
        all &= mask[col];
    }
    if ((any & 0x7fffu) == 0) {
        return 4; // MASK_OPT_ALL_ZERO << 1
    }
    return any == 0xfc00u && all == 0xfc00u ? 2 : 0; // MASK_OPT_ALL_NEG_INF << 1
}

static constexpr int64_t GGML_VK_VPLANE_MAX_QUERY_ROWS = 8;
static constexpr int64_t GGML_VK_VPLANE_MAX_TAIL_ROWS = 128;
static constexpr int64_t GGML_VK_VPLANE_MAX_PREFILL_ROWS = 1024;
static constexpr uint32_t GGML_VK_VPLANE_CACHE_GROUPS = 4;

struct ggml_vk_vplane_config {
    uint32_t groups;
    uint32_t workgroup_size;
    uint32_t row_split;
    uint32_t splits;
    bool integer;
};

static inline ggml_vk_vplane_config ggml_vk_vplane_tuning(
        int64_t kv, int64_t rows, int bits, uint32_t groups = 0) {
    if (rows <= 0 || rows > GGML_VK_VPLANE_MAX_PREFILL_ROWS) {
        return {};
    }
    const bool long_kv = kv >= 4096;
    if (rows > GGML_VK_VPLANE_MAX_QUERY_ROWS) {
        return {GGML_VK_VPLANE_CACHE_GROUPS, long_kv ? 64u : 128u, long_kv ? 2u : 4u, long_kv ? 4u : 1u, false};
    }
    const bool row1 = bits == 8 && groups != GGML_VK_VPLANE_CACHE_GROUPS && kv >= 65536 && (rows == 2 || rows == 8);
    return {row1 ? 8u : 4u, long_kv ? 64u : 128u, long_kv ? (row1 ? 1u : 2u) : 4u,
            uint32_t((long_kv ? 64 : 16) / rows),
            long_kv && (bits == 4 ? (rows >= 4 || (rows == 2 && kv >= 65536)) : (rows == 8 && kv < 65536))};
}

static inline bool ggml_vk_vplane_inputs(const ggml_tensor * q, const ggml_tensor * k,
                                         const ggml_tensor * v, const ggml_tensor * mask, int bits,
                                         int64_t max_rows = GGML_VK_VPLANE_MAX_QUERY_ROWS) {
    if (!q || !k || !v || (bits != 4 && bits != 8) || q->type != GGML_TYPE_F32 ||
        k->type != (bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0) || v->type != GGML_TYPE_I8) {
        return false;
    }
    const int64_t kv = k->ne[1], rows = q->ne[1];
    if (kv < 32 || kv > GGML_VK_VPLANE_MAX_KV || kv % 32 != 0 || rows < 1 || rows > max_rows ||
        q->ne[0] != 256 || q->ne[2] != 8 || q->ne[3] != 1 ||
        k->ne[0] != 256 || k->ne[2] != 2 || k->ne[3] != 1 ||
        v->ne[0] != ggml_vk_vplane_tile_bytes(bits) || v->ne[1] != kv / 32 || v->ne[2] != 2 || v->ne[3] != 1) {
        return false;
    }
    const size_t k_row = 8 * ggml_type_size(k->type);
    if (q->nb[0] != 4 || q->nb[2] != 1024 || q->nb[1] != 8192 || q->nb[3] != 8192 * size_t(rows) ||
        k->nb[0] != ggml_type_size(k->type) || k->nb[2] != k_row || k->nb[1] != 2 * k_row ||
        k->nb[3] < 2 * k_row * size_t(kv) ||
        v->nb[0] != 1 || v->nb[1] != size_t(v->ne[0]) ||
        v->nb[2] < v->nb[1] * size_t(v->ne[1]) || v->nb[2] % v->nb[1] != 0 ||
        v->nb[3] != 2 * v->nb[2]) {
        return false;
    }
    return !mask || (mask->type == GGML_TYPE_F16 && mask->ne[0] == kv && mask->ne[1] == rows &&
                     mask->ne[2] == 1 && mask->ne[3] == 1 && ggml_is_contiguous(mask));
}

static inline bool ggml_vk_vplane_masked_v_inputs(
        const ggml_tensor * v, const ggml_tensor * map, const ggml_tensor * k, const ggml_tensor * mask) {
    if (!v) {
        return !map;
    }
    if (!mask || v->type != GGML_TYPE_F16) {
        return false;
    }
    if (map) {
        const int64_t tiles = k->ne[1] / 32;
        return map->type == GGML_TYPE_I32 && (map->ne[0] == tiles || map->ne[0] == tiles * (1 + mask->ne[1])) &&
            ggml_is_vector(map) && ggml_is_contiguous(map) &&
            v->ne[0] == 512 && v->ne[1] % 32 == 0 && v->ne[2] == 1 && v->ne[3] == 1 &&
            ggml_is_contiguous(v);
    }
    return v->ne[0] == 256 && v->ne[1] == k->ne[1] && v->ne[2] == 2 && v->ne[3] == 1 &&
        v->nb[0] == 2 && v->nb[1] == 1024 && v->nb[2] == 512 && v->nb[3] >= 1024 * size_t(k->ne[1]);
}

static inline bool ggml_vk_vplane_integer_pv(
        ggml_vk_vplane_config config, int bits, const ggml_tensor * masked_v, const ggml_tensor * tile_map) {
    return config.integer && (!masked_v || (bits == 4 && tile_map));
}

// Returns null for an unsupported input contract. Check Vulkan supports_op too.
// V is [chunk_bytes, kv / 32, 2] I8 storage. A model cache keeps its packing across query widths.
static inline ggml_tensor * ggml_vk_flash_attn_vplane(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                                     ggml_tensor * v, ggml_tensor * mask, int bits, float scale,
                                                     uint32_t groups = 0, ggml_tensor * masked_v = nullptr,
                                                     ggml_tensor * tile_map = nullptr, bool with_tail = false,
                                                     bool prefill = false) {
    const int64_t max_rows = prefill ? GGML_VK_VPLANE_MAX_PREFILL_ROWS :
        (with_tail ? GGML_VK_VPLANE_MAX_TAIL_ROWS : GGML_VK_VPLANE_MAX_QUERY_ROWS);
    if (!ctx || !std::isfinite(scale) || !ggml_vk_vplane_inputs(q, k, v, mask, bits, max_rows) ||
        !ggml_vk_vplane_masked_v_inputs(masked_v, tile_map, k, mask)) {
        return nullptr;
    }
    const auto config = ggml_vk_vplane_tuning(k->ne[1], q->ne[1], bits, groups);
    if (groups != 0 && config.groups != groups) {
        return nullptr;
    }
    auto * out = ggml_flash_attn_ext(ctx, q, k, k, mask, scale, 0.0f, 0.0f);
    out->src[2] = v;
    out->src[5] = masked_v;
    out->src[6] = tile_map;
    ggml_prec_set_acc(out, GGML_PREC_F32);
    out->op_params[5] = GGML_VK_VPLANE_TAG;
    out->op_params[6] = bits;
    out->op_params[7] = config.groups;
    out->op_params[8] = ggml_vk_vplane_integer_pv(config, bits, masked_v, tile_map);
    out->op_params[9] = prefill ? 1 : 0;
    return out;
}

static inline bool ggml_vk_vplane_tail_inputs(
        const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask,
        const ggml_tensor * kt, const ggml_tensor * vt, const ggml_tensor * slots,
        const ggml_tensor * scratch, const ggml_tensor * map, int bits, bool tiled = false) {
    if (!mask || !scratch || !map || !kt || !vt || !slots ||
        !ggml_vk_vplane_inputs(q, k, v, mask, bits, tiled ? GGML_VK_VPLANE_MAX_TAIL_ROWS : INT32_MAX) ||
        !ggml_vk_vplane_masked_v_inputs(scratch, map, k, mask) || map->ne[0] != k->ne[1] / 32 ||
        slots->type != GGML_TYPE_I32 || !ggml_is_contiguous(slots) ||
        slots->ne[0] != k->ne[1] + (tiled ? k->ne[1] / 32 : 0) ||
        slots->ne[1] != q->ne[1] || slots->ne[2] != 1 || slots->ne[3] != 1) {
        return false;
    }
    const auto valid = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F16 && t->ne[0] == 256 && t->ne[2] == 2 && t->ne[3] == 1 &&
            t->ne[1] > 0 && t->ne[1] <= GGML_VK_VPLANE_MAX_KV && t->nb[0] == 2 && t->nb[1] == 1024 && t->nb[2] == 512;
    };
    return valid(kt) && valid(vt) && kt->ne[1] == vt->ne[1];
}

static inline ggml_tensor * ggml_vk_flash_attn_vplane_tail(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
        ggml_tensor * kt, ggml_tensor * vt, ggml_tensor * slots, ggml_tensor * scratch,
        ggml_tensor * map, int bits, float scale, bool tiled = false) {
    if (!ctx || !std::isfinite(scale) || !ggml_vk_vplane_tail_inputs(q, k, v, mask, kt, vt, slots, scratch, map, bits, tiled)) {
        return nullptr;
    }
    if (tiled) {
        auto * out = ggml_vk_flash_attn_vplane(ctx, q, k, v, mask, bits, scale,
                                              GGML_VK_VPLANE_CACHE_GROUPS, scratch, map, true);
        if (out) {
            out->src[7] = kt;
            out->src[8] = vt;
            out->src[9] = slots;
        }
        return out;
    }
    auto * out = ggml_flash_attn_ext_tail(ctx, q, k, k, mask, kt, vt, slots, scale, 0.0f, 0.0f);
    out->src[2] = v;
    out->src[8] = scratch;
    out->src[9] = map;
    out->op_params[5] = GGML_VK_VPLANE_TAG;
    out->op_params[6] = bits;
    return out;
}


// Update four-way token32 tiles in-place. The packed tile width selects V4 or V8.
static inline ggml_tensor * ggml_vk_vplane_pack_v8(
        ggml_context * ctx, ggml_tensor * values, ggml_tensor * tiles, ggml_tensor * packed, ggml_tensor * recent = nullptr) {
    auto * result = ggml_view_tensor(ctx, packed);
    result->op = GGML_OP_VPLANE_PACK_V8;
    result->src[0] = values;
    result->src[1] = tiles;
    result->src[2] = packed;
    result->src[3] = recent;
    return result;
}

// Without incoming rows, gather all visible tiles for the wide prefill path.
static inline ggml_tensor * ggml_vk_vplane_gather_v8(
        ggml_context * ctx, ggml_tensor * packed, ggml_tensor * meta, ggml_tensor * scratch,
        ggml_tensor * incoming = nullptr) {
    const int64_t tiles = incoming ? meta->ne[1] : meta->ne[0];
    auto * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, 32 * tiles);
    result->op = GGML_OP_VPLANE_GATHER_V8;
    result->src[0] = packed;
    result->src[1] = meta;
    result->src[2] = scratch;
    result->src[3] = incoming;
    return result;
}
