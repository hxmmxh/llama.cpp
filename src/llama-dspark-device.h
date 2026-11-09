#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <vector>

struct llama_dspark_device_sampling {
    std::vector<float> uniforms;
    float inverse_temperature = 1.0f;
    ggml_backend_t backend = nullptr;
};

struct llama_dspark_device_features {
    std::vector<ggml_tensor *> layers;
    std::vector<int32_t> row_indices;
    size_t capacity = 0;
    size_t source_rows = 0;
    size_t row_offset = 0;
    size_t n_rows = 0;
    ggml_backend_t backend = nullptr;
};

inline ggml_tensor * llama_dspark_cdf_index(ggml_context * ctx, ggml_tensor * cdf, ggml_tensor * uniform) {
    auto * mask = ggml_step(ctx, ggml_sub(ctx, cdf, uniform));
    auto * count = ggml_clamp(ctx, ggml_sum_rows(ctx, mask), 1.0f, (float) cdf->ne[0]);
    return ggml_scale_bias(ctx, count, -1.0f, (float) cdf->ne[0]);
}

// Bound prefix-sum error for large vocabularies by sampling a group, then a token in that group.
inline ggml_tensor * llama_dspark_sample_probs(ggml_context * ctx, ggml_tensor * probs, ggml_tensor * uniform) {
    constexpr int64_t group_size = 256;
    ggml_tensor * index;
    if (probs->ne[1] == 1 && probs->ne[0] > group_size && probs->ne[0] % group_size == 0) {
        const int64_t groups = probs->ne[0]/group_size;
        auto * grouped = ggml_reshape_2d(ctx, probs, group_size, groups);
        auto * mass = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, grouped), groups, 1);
        auto * cdf = ggml_cumsum(ctx, mass);
        auto * group = ggml_cast(ctx, llama_dspark_cdf_index(ctx, cdf, uniform), GGML_TYPE_I32);
        group = ggml_reshape_1d(ctx, group, 1);
        auto * preceding = ggml_reshape_2d(ctx, ggml_sub(ctx, cdf, mass), 1, groups);
        auto * offset = ggml_get_rows(ctx, preceding, group);
        auto * local = ggml_get_rows(ctx, grouped, group);
        index = llama_dspark_cdf_index(ctx, ggml_cumsum(ctx, local), ggml_sub(ctx, uniform, offset));
        index = ggml_add(ctx, index, ggml_scale(ctx, ggml_cast(ctx, group, GGML_TYPE_F32), (float) group_size));
    } else {
        index = llama_dspark_cdf_index(ctx, ggml_cumsum(ctx, probs), uniform);
    }
    return ggml_reshape_1d(ctx, ggml_cast(ctx, index, GGML_TYPE_I32), probs->ne[1]);
}

inline void llama_dspark_copy_feature_rows(ggml_backend_t backend, ggml_tensor * src, ggml_tensor * dst, size_t row_offset) {
    GGML_ASSERT(src && dst && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src->ne[0] == dst->ne[0] && row_offset <= (size_t) src->ne[1]);
    GGML_ASSERT((size_t) dst->ne[1] <= (size_t) src->ne[1] - row_offset);
    ggml_context_ptr views(ggml_init({ ggml_tensor_overhead(), nullptr, true }));
    GGML_ASSERT(views);
    auto * view = ggml_view_2d(views.get(), src, src->ne[0], dst->ne[1], src->nb[1], row_offset*src->nb[1]);
    ggml_backend_view_init(view);
    GGML_ASSERT(ggml_backend_buffer_get_type(view->buffer) == ggml_backend_buffer_get_type(dst->buffer));
    ggml_backend_tensor_copy_async(backend, backend, view, dst);
}
