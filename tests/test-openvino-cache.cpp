#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

static context_ptr make_context() {
    ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
    context_ptr ctx(ggml_init(params), ggml_free);
    if (!ctx) {
        throw std::runtime_error("context allocation failed");
    }
    return ctx;
}

static buffer_ptr allocate(ggml_context * ctx, ggml_backend_t backend) {
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("buffer allocation failed");
    }
    return buffer;
}

static void check(ggml_backend_t backend, ggml_cgraph * graph, ggml_tensor * output,
                  const std::vector<float> & expected) {
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("graph compute failed");
    }
    ggml_backend_synchronize(backend);
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < expected.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-3f * (1 + std::abs(expected[i]))) {
            std::fprintf(stderr, "element %zu: got %g, expected %g\n", i, actual[i], expected[i]);
            throw std::runtime_error("incorrect output");
        }
    }
}

static void test_scale(ggml_backend_t backend, int width) {
    auto ctx = make_context();
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 4);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    auto * output = ggml_scale(ctx.get(), input, 2.0f);
    ggml_set_name(output, "output");
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    auto buffer = allocate(ctx.get(), backend);
    std::vector<float> values(width * 4), expected(values.size());
    for (int round = 0; round < 4; ++round) {
        // Change op_params on the same graph and change input contents on hits.
        float scale = round < 2 ? 2.0f : 3.0f;
        std::memcpy(output->op_params, &scale, sizeof(scale));
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = float(int(i % 11) - 5) * float(round + 1);
            expected[i] = scale * values[i];
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        check(backend, graph, output, expected);
    }
}

static void test_matmul(ggml_backend_t first, ggml_backend_t second, ggml_type type, backend_ptr * owner = nullptr) {
    constexpr int k = 256, n = 32, m = 4;
    auto weight_ctx = make_context();
    auto * weight = ggml_new_tensor_2d(weight_ctx.get(), type, k, n);
    ggml_set_name(weight, "weight");
    auto weight_buffer = allocate(weight_ctx.get(), first);
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    auto ctx = make_context();
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    auto * output = ggml_mul_mat(ctx.get(), weight, input);
    ggml_set_name(output, "output");
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    auto buffer = allocate(ctx.get(), first);
    std::vector<float> weights(k * n), values(k * m), expected(n * m);
    std::vector<unsigned char> packed(ggml_nbytes(weight));
    for (int round = 0; round < 4; ++round) {
        // Replacing constants at the same address must invalidate both requests.
        const float weight_value = round < 2 ? 1.0f : 2.0f;
        if (round == 0 || round == 2) {
            std::fill(weights.begin(), weights.end(), weight_value);
            if (type == GGML_TYPE_F32) {
                std::memcpy(packed.data(), weights.data(), packed.size());
            } else {
                ggml_quantize_chunk(type, weights.data(), packed.data(), 0, n, k, nullptr);
            }
            ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
        }
        for (int row = 0; row < m; ++row) {
            const float value = float(row + round + 1) / 16;
            std::fill(values.begin() + row * k, values.begin() + (row + 1) * k, value);
            std::fill(expected.begin() + row * n, expected.begin() + (row + 1) * n, k * value * weight_value);
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        check(first, graph, output, expected);
        check(second, graph, output, expected);
        check(first, graph, output, expected);
    }
    if (owner) {
        // Destruction must leave the second backend's existing request usable.
        owner->reset();
        check(second, graph, output, expected);
    }
}

static void test_dynamic_features(ggml_backend_t backend, bool copy_input) {
    auto weight_ctx = make_context();
    auto * weight = ggml_new_tensor_2d(weight_ctx.get(), GGML_TYPE_F32, 32, 16);
    ggml_set_name(weight, "projection.weight");
    auto weight_buffer = allocate(weight_ctx.get(), backend);
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> weights(32 * 16, 0.25f);
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size() * sizeof(float));
    const int lengths[] = {2, 128, 124, 4, 4, 4};
    for (size_t round = 0; round < sizeof(lengths) / sizeof(lengths[0]); ++round) {
        const int length = lengths[round];
        const bool extract_features = round != 4;
        auto ctx = make_context();
        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, length);
        ggml_set_name(input, "external_features");
        ggml_set_input(input);
        auto * features = ggml_mul_mat(ctx.get(), weight, copy_input ? ggml_dup(ctx.get(), input) : input);
        ggml_set_name(features, "features");
        if (extract_features) {
            ggml_set_output(features);
        }
        auto * output = features;
        // Cross the small-graph threshold to exercise the model request cache.
        for (int i = 0; i < 20; ++i) {
            output = ggml_scale(ctx.get(), output, 1.0f);
        }
        ggml_set_name(output, "output");
        ggml_set_output(output);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        auto buffer = allocate(ctx.get(), backend);
        std::vector<float> values(32 * length), expected(16 * length);
        for (int row = 0; row < length; ++row) {
            const float value = float(row + 1) / 8;
            std::fill(values.begin() + row * 32, values.begin() + (row + 1) * 32, value);
            std::fill(expected.begin() + row * 16, expected.begin() + (row + 1) * 16, 8 * value);
        }
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        std::vector<float> sentinel(expected.size(), -1234.0f);
        ggml_backend_tensor_set(features, sentinel.data(), 0, sentinel.size() * sizeof(float));
        check(backend, graph, output, expected);
        if (extract_features) {
            std::vector<float> actual(expected.size());
            ggml_backend_tensor_get(features, actual.data(), 0, actual.size() * sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > 1e-3f * (1 + std::abs(expected[i]))) {
                    throw std::runtime_error("explicit intermediate feature output was not updated");
                }
            }
        }
    }
}

static void test_q4_minimum(ggml_backend_t backend) {
    constexpr int k = 256, n = 32, m = 4;
    auto weight_ctx = make_context();
    auto * weight = ggml_new_tensor_2d(weight_ctx.get(), GGML_TYPE_Q4_K, k, n);
    ggml_set_name(weight, "offset.weight");
    auto weight_buffer = allocate(weight_ctx.get(), backend);
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto ctx = make_context();
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    auto * output = ggml_mul_mat(ctx.get(), weight, input);
    ggml_set_name(output, "output");
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    auto buffer = allocate(ctx.get(), backend);
    const int indices[m] = {0, 31, 64, 255};
    std::vector<float> values(k * m, 0.0f);
    for (int row = 0; row < m; ++row) {
        values[row * k + indices[row]] = 1.0f;
    }
    ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
    for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<block_q4_K> blocks(n);
        std::vector<float> expected(n * m), decoded(k);
        for (int row = 0; row < n; ++row) {
            auto & block = blocks[row];
            // Fractional zero point, zero point above 15, and constant negative group.
            block.d = ggml_fp32_to_fp16(pattern == 0 ? 1.0f : pattern == 1 ? 0.03125f : 0.0f);
            block.dmin = ggml_fp32_to_fp16(pattern == 0 ? 0.25f : pattern == 1 ? 0.75f : 1.0f);
            std::fill_n(block.scales, 8, uint8_t(1));
            std::fill_n(block.scales + 8, 4, uint8_t(0x11));
            for (size_t i = 0; i < sizeof(block.qs); ++i) {
                block.qs[i] = uint8_t(((i + row) & 15) | (((i * 3 + row) & 15) << 4));
            }
            dequantize_row_q4_K(&block, decoded.data(), k);
            for (int col = 0; col < m; ++col) {
                expected[col * n + row] = decoded[indices[col]];
            }
        }
        ggml_backend_tensor_set(weight, blocks.data(), 0, blocks.size() * sizeof(block_q4_K));
        check(backend, graph, output, expected);
    }
}

static void test_distinct_kv_indices(ggml_backend_t backend, int extra_ops) {
    auto ctx = make_context();
    auto * full = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 8);
    auto * window = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 4);
    auto * values = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 16, 1);
    auto * full_indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    auto * window_indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    ggml_set_name(full, "full_rows");
    ggml_set_name(window, "window_rows");
    ggml_set_name(values, "values");
    ggml_set_name(full_indices, "attn_inp_k_idxs");
    ggml_set_name(window_indices, "attn_inp_k_idxs");
    for (auto * input : {full, window, values, full_indices, window_indices}) {
        ggml_set_input(input);
    }
    auto * full_update = ggml_set_rows(ctx.get(), full, values, full_indices);
    auto * window_update = ggml_set_rows(ctx.get(), window, values, window_indices);
    ggml_set_name(full_update, "full_update");
    ggml_set_name(window_update, "window_update");
    ggml_set_output(full_update);
    ggml_set_output(window_update);
    auto * output = full_update;
    for (int i = 0; i < extra_ops; ++i) {
        output = ggml_scale(ctx.get(), output, 1.0f);
    }
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, window_update);
    ggml_build_forward_expand(graph, output);
    auto buffer = allocate(ctx.get(), backend);
    // Scheduler graph views omit the leaf list, so input identity must also
    // work when only nodes and their source pointers are available.
    auto view = *graph;
    view.n_leafs = 0;
    std::vector<float> data(16), zeros_full(16 * 8, 0.0f), zeros_window(16 * 4, 0.0f);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = float(i + 1);
    }
    ggml_backend_tensor_set(values, data.data(), 0, data.size() * sizeof(float));
    for (int round = 0; round < 2; ++round) {
        const int64_t full_index = 3 - round;
        const int64_t window_index = 1 - round;
        ggml_backend_tensor_set(full, zeros_full.data(), 0, zeros_full.size() * sizeof(float));
        ggml_backend_tensor_set(window, zeros_window.data(), 0, zeros_window.size() * sizeof(float));
        ggml_backend_tensor_set(full_indices, &full_index, 0, sizeof(full_index));
        ggml_backend_tensor_set(window_indices, &window_index, 0, sizeof(window_index));
        auto expected_full = zeros_full;
        auto expected_window = zeros_window;
        std::copy(data.begin(), data.end(), expected_full.begin() + 16 * full_index);
        std::copy(data.begin(), data.end(), expected_window.begin() + 16 * window_index);
        check(backend, &view, output, expected_full);
        std::vector<float> actual(expected_window.size());
        ggml_backend_tensor_get(window, actual.data(), 0, actual.size() * sizeof(float));
        if (actual != expected_window) {
            throw std::runtime_error("distinct KV index tensors with identical names were aliased");
        }
    }
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_load_all();
        const char * name = argc > 1 ? argv[1] : "OPENVINO";
        auto reg = ggml_backend_reg_by_name(name);
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
            throw std::runtime_error("requested backend is unavailable");
        }
        auto dev = ggml_backend_reg_dev_get(reg, 0);
        // OpenVINO registers all devices for discovery; only the selected one
        // advertises GPU type (including when GGML_OPENVINO_DEVICE=CPU).
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
            auto candidate = ggml_backend_reg_dev_get(reg, i);
            if (ggml_backend_dev_type(candidate) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                dev = candidate;
                break;
            }
        }
        std::printf("device=%s description=%s\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
        backend_ptr first(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
        backend_ptr second(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
        if (!first || !second) {
            throw std::runtime_error("backend initialization failed");
        }
        test_scale(first.get(), 32);
        test_scale(second.get(), 64);
        test_matmul(first.get(), second.get(), GGML_TYPE_F32);
        test_matmul(first.get(), second.get(), GGML_TYPE_Q4_K, &first);
        test_scale(second.get(), 32);
        test_dynamic_features(second.get(), false);
        test_dynamic_features(second.get(), true);
        test_q4_minimum(second.get());
        test_distinct_kv_indices(second.get(), 0);
        test_distinct_kv_indices(second.get(), 20);
        std::printf("%s cache regression: PASS\n", name);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
