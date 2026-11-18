#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>

// Private packed-cache contract. Ordinary Q4_0/Q8_0 tensors keep their layout.
constexpr int32_t GGML_VK_VPLANE_TAG = 0x56504c31;
constexpr uint32_t GGML_VK_FA_VPLANE = 128u;
constexpr uint32_t GGML_VK_FA_VPLANE_INTEGER = 256u;
constexpr uint32_t GGML_VK_FA_VPLANE_Q4 = 512u;

struct ggml_vk_vplane_config {
    uint32_t groups;
    uint32_t workgroup_size;
    uint32_t row_split;
    uint32_t splits;
    bool integer;
};

static inline ggml_vk_vplane_config ggml_vk_vplane_tuning(int64_t kv, int64_t rows, int bits) {
    if (rows <= 0 || rows > 8) {
        return {};
    }
    const bool long_kv = kv >= 4096;
    const bool row1 = bits == 8 && kv >= 65536 && (rows == 2 || rows == 8);
    return {row1 ? 8u : 4u, long_kv ? 64u : 128u, long_kv ? (row1 ? 1u : 2u) : 4u,
            uint32_t((long_kv ? 64 : 16) / rows),
            long_kv && (bits == 4 ? (rows >= 4 || (rows == 2 && kv >= 65536)) : (rows == 8 && !row1))};
}

static inline bool ggml_vk_vplane_inputs(const ggml_tensor * q, const ggml_tensor * k,
                                         const ggml_tensor * v, const ggml_tensor * mask, int bits) {
    if (!q || !k || !v || (bits != 4 && bits != 8) || q->type != GGML_TYPE_F32 ||
        k->type != (bits == 4 ? GGML_TYPE_Q4_0 : GGML_TYPE_Q8_0) || v->type != GGML_TYPE_I8) {
        return false;
    }
    const int64_t kv = k->ne[1], rows = q->ne[1];
    if (kv < 32 || kv > 131328 || kv % 32 != 0 ||
        !(rows == 1 || rows == 2 || rows == 4 || rows == 8 || (kv == 768 && rows == 3)) ||
        q->ne[0] != 256 || q->ne[2] != 8 || q->ne[3] != 1 ||
        k->ne[0] != 256 || k->ne[2] != 2 || k->ne[3] != 1 ||
        v->ne[0] != (bits == 4 ? 4608 : 8704) || v->ne[1] != kv / 32 || v->ne[2] != 2 || v->ne[3] != 1) {
        return false;
    }
    const size_t k_row = 8 * ggml_type_size(k->type);
    if (q->nb[0] != 4 || q->nb[2] != 1024 || q->nb[1] != 8192 || q->nb[3] != 8192 * size_t(rows) ||
        k->nb[0] != ggml_type_size(k->type) || k->nb[2] != k_row || k->nb[1] != 2 * k_row ||
        k->nb[3] != 2 * k_row * size_t(kv) || !ggml_is_contiguous(v)) {
        return false;
    }
    return !mask || (mask->type == GGML_TYPE_F16 && mask->ne[0] == kv && mask->ne[1] == rows &&
                     mask->ne[2] == 1 && mask->ne[3] == 1 && ggml_is_contiguous(mask));
}

// Returns null for an unsupported input contract. Check Vulkan supports_op too.
// V is [chunk_bytes, kv / 32, 2] I8 storage, packed for vplane_tuning().groups.
static inline ggml_tensor * ggml_vk_flash_attn_vplane(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
                                                     ggml_tensor * v, ggml_tensor * mask, int bits, float scale) {
    if (!ctx || !std::isfinite(scale) || !ggml_vk_vplane_inputs(q, k, v, mask, bits)) {
        return nullptr;
    }
    const auto config = ggml_vk_vplane_tuning(k->ne[1], q->ne[1], bits);
    auto * out = ggml_flash_attn_ext(ctx, q, k, k, mask, scale, 0.0f, 0.0f);
    out->src[2] = v;
    ggml_prec_set_acc(out, GGML_PREC_F32);
    out->op_params[5] = GGML_VK_VPLANE_TAG;
    out->op_params[6] = bits;
    out->op_params[7] = config.groups;
    out->op_params[8] = config.integer;
    return out;
}
