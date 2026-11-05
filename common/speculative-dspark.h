#pragma once

#include "common.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

// Use native Q8 dot products when W2 is losslessly representable; otherwise use the float fallback.
class common_speculative_dspark_markov {
public:
    common_speculative_dspark_markov(const std::vector<float> & w2, int32_t rank, int32_t vocab,
                                    const common_cpu_params & cpu, const std::vector<float> * w1 = nullptr) : rank(rank), vocab(vocab) {
        GGML_ASSERT(rank > 0 && vocab > 0 && w2.size() / rank == (size_t) vocab && w2.size() % rank == 0);
        GGML_ASSERT(!w1 || w1->size() == w2.size());
        if (init_integer(w2, cpu, w1)) {
            return;
        }
        const size_t blocks = ((size_t) vocab + width - 1) / width;
        if (!pack_quantized(w2)) {
            weights.resize(blocks * rank * width, 0.0f);
            for (int32_t token = 0; token < vocab; ++token) {
                float * block = weights.data() + (size_t) (token / width) * rank * width;
                for (int32_t r = 0; r < rank; ++r) {
                    block[(size_t) r * width + token % width] = w2[(size_t) token * rank + r];
                }
            }
        }

        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev || cpu.n_threads <= 1) {
            return;
        }
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * new_pool = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        auto * free_pool = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        auto * set_pool = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        auto * set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (!new_pool || !free_pool || !set_pool || !set_threads) {
            return;
        }
        auto params = ggml_threadpool_params_from_cpu_params(cpu);
        params.n_threads = std::min<int32_t>(params.n_threads, blocks);
        params.poll = 0; // Do not spin while the target or draft backbone uses its own pool.
        pool = pool_ptr(new_pool(&params), free_pool);
        backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!pool || !backend) {
            throw std::runtime_error("failed to initialize DSpark Markov CPU workers");
        }
        set_threads(backend.get(), params.n_threads);
        set_pool(backend.get(), pool.get());

        ctx.reset(ggml_init({ 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4, false), nullptr, true }));
        if (!ctx) {
            throw std::runtime_error("failed to allocate DSpark Markov graph");
        }
        input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
        output = ggml_map_custom1_inplace(ctx.get(), input, compute, GGML_N_TASKS_MAX, this);
        graph = ggml_new_graph_custom(ctx.get(), 4, false);
        ggml_build_forward_expand(graph, output);
    }

    common_speculative_dspark_markov(const common_speculative_dspark_markov &) = delete;
    common_speculative_dspark_markov & operator=(const common_speculative_dspark_markov &) = delete;
    common_speculative_dspark_markov(common_speculative_dspark_markov &&) = delete;
    common_speculative_dspark_markov & operator=(common_speculative_dspark_markov &&) = delete;

    bool uses_quantized_weights() const { return !integer_weights.empty() || !quantized_weights.empty(); }
    bool uses_integer_dot() const { return !integer_weights.empty(); }
    size_t w1_cache_bytes() const { return integer_w1.size(); }

    void apply_token(float * logits, int32_t token) {
        GGML_ASSERT(!integer_w1.empty() && token >= 0 && token < vocab);
        apply_integer(logits, integer_w1.data() + (size_t) token * ggml_row_size(GGML_TYPE_Q8_0, rank));
    }

    void apply(float * logits, const float * w1) {
        if (uses_integer_dot()) {
            GGML_ASSERT(integer_w1.empty());
            apply_integer(logits, w1);
            return;
        }
        if (!backend) {
            apply_rows(logits, w1, 0, 1);
            return;
        }
        row = w1;
        input->data = output->data = logits;
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DSpark Markov CPU graph failed");
        }
    }

private:
    static constexpr int width = 16;
    using pool_ptr = std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)>;
    pool_ptr pool{ nullptr, nullptr };
    ggml_backend_ptr backend;
    ggml_context_ptr ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    std::vector<float> weights;
    std::vector<int8_t> quantized_weights;
    std::vector<float> quantized_scales;
    std::vector<uint8_t> integer_weights;
    std::vector<uint8_t> integer_w1;
    std::vector<float> integer_bias;
    ggml_tensor * integer_input = nullptr;
    const float * row = nullptr;
    int32_t rank;
    int32_t vocab;

    void apply_integer(float * logits, const void * w1) {
        integer_input->data = const_cast<void *>(w1);
        input->data = output->data = logits;
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DSpark integer Markov CPU graph failed");
        }
    }

    void cache_w1(const std::vector<float> & w1) {
        integer_w1.resize(ggml_row_size(GGML_TYPE_Q8_0, rank) * (size_t) vocab);
        ggml_context_ptr prepare_ctx(ggml_init({ 2 * ggml_tensor_overhead() + ggml_graph_overhead_custom(4, false), nullptr, true }));
        if (!prepare_ctx) {
            throw std::runtime_error("failed to allocate DSpark W1 preparation graph");
        }
        auto * source = ggml_new_tensor_2d(prepare_ctx.get(), GGML_TYPE_F32, rank, vocab);
        source->data = const_cast<float *>(w1.data());
        // CPU cast uses the same native quantizer as the dynamic Q8 dot input conversion.
        auto * packed = ggml_cast(prepare_ctx.get(), source, GGML_TYPE_Q8_0);
        packed->data = integer_w1.data();
        auto * prepare_graph = ggml_new_graph_custom(prepare_ctx.get(), 4, false);
        ggml_build_forward_expand(prepare_graph, packed);
        if (ggml_backend_graph_compute(backend.get(), prepare_graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("DSpark W1 quantization failed");
        }
    }

    bool init_integer(const std::vector<float> & w2, const common_cpu_params & cpu, const std::vector<float> * w1) {
        if (rank % ggml_blck_size(GGML_TYPE_Q8_0) != 0) {
            return false;
        }
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) {
            return false;
        }
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * new_pool = (decltype(ggml_threadpool_new) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        auto * free_pool = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        auto * set_pool = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        auto * set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (!new_pool || !free_pool || !set_pool || !set_threads) {
            return false;
        }
        for (float value : w2) {
            if (!std::isfinite(value)) {
                return false;
            }
        }
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, rank);
        std::vector<uint8_t> packed(row_bytes * (size_t) vocab);
        GGML_ASSERT(ggml_quantize_chunk(GGML_TYPE_Q8_0, w2.data(), packed.data(), 0, vocab, rank, nullptr) == packed.size());
        std::vector<float> restored(rank);
        const auto to_float = ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float;
        for (int32_t token = 0; token < vocab; ++token) {
            to_float(packed.data() + (size_t) token * row_bytes, restored.data(), rank);
            if (std::memcmp(restored.data(), w2.data() + (size_t) token * rank, rank * sizeof(float)) != 0) {
                return false;
            }
        }
        // Preserve W2 exactly; use the CPU backend's native Q8 input quantizer for W1.
        auto params = ggml_threadpool_params_from_cpu_params(cpu);
        params.n_threads = std::max<int32_t>(1, std::min<int32_t>(params.n_threads, vocab));
        params.poll = 0;
        pool = pool_ptr(new_pool(&params), free_pool);
        backend.reset(ggml_backend_dev_init(dev, nullptr));
        if (!pool || !backend) {
            throw std::runtime_error("failed to initialize DSpark integer Markov workers");
        }
        set_threads(backend.get(), params.n_threads);
        set_pool(backend.get(), pool.get());
        if (w1) {
            cache_w1(*w1);
        }
        ctx.reset(ggml_init({ 5 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true }));
        if (!ctx) {
            throw std::runtime_error("failed to allocate DSpark integer Markov graph");
        }
        integer_weights = std::move(packed);
        integer_bias.resize(vocab);
        auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, rank, vocab);
        weight->data = integer_weights.data();
        integer_input = ggml_new_tensor_1d(ctx.get(), integer_w1.empty() ? GGML_TYPE_F32 : GGML_TYPE_Q8_0, rank);
        auto * bias = ggml_mul_mat(ctx.get(), weight, integer_input);
        bias->data = integer_bias.data();
        input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, vocab);
        output = ggml_add_inplace(ctx.get(), input, bias);
        graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        return true;
    }

    bool pack_quantized(const std::vector<float> & w2) {
        if (rank % 32 != 0) {
            return false;
        }
        const size_t blocks = ((size_t) vocab + width - 1) / width;
        std::vector<int8_t> codes(blocks * rank * width, 0);
        std::vector<float> scales(blocks * (rank / 32) * width, 0.0f);
        for (int32_t token = 0; token < vocab; ++token) {
            for (int32_t r = 0; r < rank; r += 32) {
                const float * values = w2.data() + (size_t) token * rank + r;
                float amax = 0.0f;
                for (int j = 0; j < 32; ++j) {
                    if (!std::isfinite(values[j])) {
                        return false;
                    }
                    amax = std::max(amax, std::abs(values[j]));
                }
                const float scale = amax / 127.0f;
                scales[((size_t) (token / width) * (rank / 32) + r / 32) * width + token % width] = scale;
                for (int j = 0; j < 32; ++j) {
                    const float q = scale > 0.0f ? std::round(values[j] / scale) : 0.0f;
                    if (q < -127.0f || q > 127.0f) {
                        return false;
                    }
                    const int8_t code = (int8_t) q;
                    const float restored = (float) code * scale;
                    if (std::memcmp(&restored, values + j, sizeof(float)) != 0) {
                        return false;
                    }
                    codes[((size_t) (token / width) * rank + r + j) * width + token % width] = code;
                }
            }
        }
        quantized_weights = std::move(codes);
        quantized_scales = std::move(scales);
        return true;
    }

    void apply_quantized(float * logits, const float * w1, int ith, int nth) const {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        const int64_t blocks = ((int64_t) vocab + width - 1) / width;
        for (int64_t block = blocks * ith / nth; block < blocks * (ith + 1) / nth; ++block) {
            const int8_t * codes = quantized_weights.data() + (size_t) block * rank * width;
            const float * scales = quantized_scales.data() + (size_t) block * (rank / 32) * width;
            float sums[width] = {};
            for (int32_t r = 0; r < rank; r += 32) {
                const float * scale = scales + (size_t) (r / 32) * width;
                for (int j = 0; j < 32; ++j) {
                    for (int lane = 0; lane < width; ++lane) {
                        const float weight = (float) codes[(size_t) (r + j) * width + lane] * scale[lane];
                        sums[lane] += weight * w1[r + j];
                    }
                }
            }
            const int64_t token = block * width;
            for (int lane = 0; lane < width && token + lane < vocab; ++lane) {
                logits[token + lane] += sums[lane];
            }
        }
    }

    static void compute(ggml_tensor * dst, const ggml_tensor *, int ith, int nth, void * userdata) {
        auto * self = static_cast<common_speculative_dspark_markov *>(userdata);
        self->apply_rows(static_cast<float *>(dst->data), self->row, ith, nth);
    }

    void apply_rows(float * logits, const float * w1, int ith, int nth) const {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
        if (!quantized_weights.empty()) {
            apply_quantized(logits, w1, ith, nth);
            return;
        }
        const int64_t blocks = ((int64_t) vocab + width - 1) / width;
        const int64_t begin = blocks * ith / nth;
        const int64_t end = blocks * (ith + 1) / nth;
        for (int64_t block = begin; block < end; ++block) {
            const float * w = weights.data() + (size_t) block * rank * width;
            float sums[width] = {};
            for (int32_t r = 0; r < rank; ++r) {
                for (int lane = 0; lane < width; ++lane) {
                    sums[lane] += w[(size_t) r * width + lane] * w1[r];
                }
            }
            const int64_t token = block * width;
            for (int lane = 0; lane < width && token + lane < vocab; ++lane) {
                logits[token + lane] += sums[lane];
            }
        }
    }
};
