#include "speculative-dspark.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>

static_assert(!std::is_copy_constructible<common_speculative_dspark_markov>::value, "graph owns a self pointer");
static_assert(!std::is_move_constructible<common_speculative_dspark_markov>::value, "graph owns a self pointer");

static void check_lossless(int rank, int vocab, int threads) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    std::mt19937 rng(4567);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> w2((size_t) rank * vocab), w1(rank), base(vocab);
    for (int token = 0; token < vocab; ++token) {
        for (int r = 0; r < rank; r += 32) {
            // These are decoded Q8 blocks, including all-zero blocks and signed extrema.
            const float scale = (token + r / 32) % 5 == 0 ? 0.0f : (1 + (token + r) % 7) / 1024.0f;
            for (int j = 0; j < 32; ++j) {
                const int code = j == 0 ? 127 : j == 1 ? -127 : (int) (rng() % 255) - 127;
                w2[(size_t) token * rank + r + j] = scale == 0.0f ? 0.0f : scale * code;
            }
        }
    }
    for (auto & x : base) { x = dist(rng); }
    common_cpu_params cpu;
    cpu.n_threads = threads;
    common_speculative_dspark_markov kernel(w2, rank, vocab, cpu);
    GGML_ASSERT(kernel.uses_quantized_weights());
    ggml_backend_ptr reference_backend;
    ggml_context_ptr reference_ctx;
    ggml_cgraph * reference_graph = nullptr;
    std::vector<uint8_t> packed;
    std::vector<float> reference_bias(vocab);
    auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (dev) {
        GGML_ASSERT(kernel.uses_integer_dot());
        reference_backend.reset(ggml_backend_dev_init(dev, nullptr));
        auto * set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
        GGML_ASSERT(reference_backend && set_threads);
        set_threads(reference_backend.get(), 1);
        reference_ctx.reset(ggml_init({ 3 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true }));
        packed.resize(ggml_row_size(GGML_TYPE_Q8_0, rank) * (size_t) vocab);
        GGML_ASSERT(ggml_quantize_chunk(GGML_TYPE_Q8_0, w2.data(), packed.data(), 0, vocab, rank, nullptr) == packed.size());
        auto * weight = ggml_new_tensor_2d(reference_ctx.get(), GGML_TYPE_Q8_0, rank, vocab);
        weight->data = packed.data();
        auto * activation = ggml_new_tensor_1d(reference_ctx.get(), GGML_TYPE_F32, rank);
        activation->data = w1.data();
        auto * output = ggml_mul_mat(reference_ctx.get(), weight, activation);
        output->data = reference_bias.data();
        reference_graph = ggml_new_graph_custom(reference_ctx.get(), 8, false);
        ggml_build_forward_expand(reference_graph, output);
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        for (auto & x : w1) { x = dist(rng); }
        if (reference_backend) {
            GGML_ASSERT(ggml_backend_graph_compute(reference_backend.get(), reference_graph) == GGML_STATUS_SUCCESS);
        }
        auto expected = base;
        std::vector<float> actual(vocab + 2, 123456.0f);
        std::copy(base.begin(), base.end(), actual.begin() + 1);
        for (int token = 0; token < vocab; ++token) {
            float bias = 0.0f;
            if (reference_backend) {
                bias = reference_bias[token];
            } else {
                for (int r = 0; r < rank; ++r) {
                    bias += w2[(size_t) token * rank + r] * w1[r];
                }
            }
            expected[token] += bias;
        }
        kernel.apply(actual.data() + 1, w1.data());
        GGML_ASSERT(actual.front() == 123456.0f && actual.back() == 123456.0f);
        GGML_ASSERT(std::memcmp(expected.data(), actual.data() + 1, vocab * sizeof(float)) == 0);
    }
    if (dev) {
        std::vector<float> table(w2.size());
        for (auto & x : table) { x = dist(rng); }
        for (size_t offset = 0; offset < table.size(); offset += 32) {
            // Exercise native rounding at half-integers, which differs from the file quantizer.
            table[offset + 0] = 127.0f;
            table[offset + 1] = 0.5f;
            table[offset + 2] = 1.5f;
            table[offset + 3] = -0.5f;
            table[offset + 4] = -1.5f;
        }
        const std::vector<int> tokens = { vocab - 1, 0, vocab / 2, vocab - 1 };
        std::vector<std::vector<float>> selected;
        for (int token : tokens) {
            selected.emplace_back(table.begin() + (size_t) token * rank, table.begin() + (size_t) (token + 1) * rank);
        }
        common_speculative_dspark_markov cached(w2, rank, vocab, cpu, &table);
        GGML_ASSERT(cached.w1_cache_bytes() == ggml_row_size(GGML_TYPE_Q8_0, rank) * (size_t) vocab);
        std::vector<float>().swap(table);
        for (size_t i = 0; i < tokens.size(); ++i) {
            auto expected = base;
            std::vector<float> actual(vocab + 2, 123456.0f);
            std::copy(base.begin(), base.end(), actual.begin() + 1);
            kernel.apply(expected.data(), selected[i].data());
            cached.apply_token(actual.data() + 1, tokens[i]);
            GGML_ASSERT(actual.front() == 123456.0f && actual.back() == 123456.0f);
            GGML_ASSERT(std::memcmp(expected.data(), actual.data() + 1, vocab * sizeof(float)) == 0);
        }
    }
    printf("q8 projection/cache rank=%d vocab=%d threads=%d integer=%d check=pass\n", rank, vocab, threads, kernel.uses_integer_dot());
}

static void check_fallback() {
    common_cpu_params cpu;
    cpu.n_threads = 1;
    for (float special : { -0.0f, 0.1234567f, std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::quiet_NaN() }) {
        std::vector<float> w2(32, 127.0f);
        std::vector<float> w1(32, 1.0f);
        w2[7] = special;
        common_speculative_dspark_markov kernel(w2, 32, 1, cpu, &w1);
        GGML_ASSERT(!kernel.uses_quantized_weights());
        GGML_ASSERT(kernel.w1_cache_bytes() == 0);
    }
    std::vector<float> w1(7, 1.0f);
    common_speculative_dspark_markov odd(std::vector<float>(7, 0.0f), 7, 1, cpu, &w1);
    GGML_ASSERT(!odd.uses_quantized_weights());
    GGML_ASSERT(odd.w1_cache_bytes() == 0);
    printf("lossless fallback check=pass\n");
}

static void reference(float * logits, const float * w1, const std::vector<float> & w2, int rank, int vocab) {
    for (int token = 0; token < vocab; ++token) {
        float bias = 0.0f;
        for (int r = 0; r < rank; ++r) {
            bias += w2[(size_t) token * rank + r] * w1[r];
        }
        logits[token] += bias;
    }
}

static void check(int rank, int vocab, int threads, bool bench) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> w2((size_t) rank * vocab), w1(rank), base(vocab);
    for (auto & x : w2) { x = dist(rng); }
    for (auto & x : base) { x = dist(rng); }
    common_cpu_params cpu;
    cpu.n_threads = threads;
    common_speculative_dspark_markov kernel(w2, rank, vocab, cpu);
    double old_ms = 0.0, new_ms = 0.0;
    const int repeats = bench ? 12 : 3;
    for (int i = 0; i < repeats; ++i) {
        for (auto & x : w1) { x = dist(rng); }
        auto expected = base;
        std::vector<float> actual(vocab + 2, 123456.0f);
        std::copy(base.begin(), base.end(), actual.begin() + 1);
        const auto a = std::chrono::steady_clock::now();
        reference(expected.data(), w1.data(), w2, rank, vocab);
        const auto b = std::chrono::steady_clock::now();
        kernel.apply(actual.data() + 1, w1.data());
        const auto c = std::chrono::steady_clock::now();
        GGML_ASSERT(actual.front() == 123456.0f && actual.back() == 123456.0f);
        if (std::memcmp(expected.data(), actual.data() + 1, vocab * sizeof(float)) != 0) {
            for (int token = 0; token < vocab; ++token) {
                const float error = std::abs(expected[token] - actual[token + 1]);
                // The original scalar tail may contract multiply/add on some compilers.
                const bool bad = rank == 256 ? std::memcmp(&expected[token], &actual[token + 1], sizeof(float)) != 0
                                            : error > 1e-6f * (1.0f + std::abs(expected[token]));
                if (bad) {
                    fprintf(stderr, "mismatch rank=%d vocab=%d threads=%d token=%d expected=%a actual=%a\n",
                            rank, vocab, threads, token, expected[token], actual[token + 1]);
                    GGML_ABORT("DSpark Markov mismatch");
                }
            }
        }
        if (i > 0) {
            old_ms += std::chrono::duration<double, std::milli>(b - a).count();
            new_ms += std::chrono::duration<double, std::milli>(c - b).count();
        }
    }
    printf("rank=%d vocab=%d threads=%d check=pass reference_ms=%.4f candidate_ms=%.4f\n",
           rank, vocab, threads, old_ms / (repeats - 1), new_ms / (repeats - 1));
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    if (!bench) {
        check_fallback();
    }
    for (int threads : { 1, 3, 8 }) {
        if (!bench) {
            for (int rank : { 32, 64, 256 }) {
                for (int vocab : { 1, 15, 16, 17, 63, 65 }) {
                    check_lossless(rank, vocab, threads);
                }
            }
            for (int rank : { 1, 7, 256 }) {
                for (int vocab : { 1, 7, 8, 9, 63, 65 }) {
                    check(rank, vocab, threads, false);
                }
            }
        }
        check(256, 131072, threads, bench);
    }
    return 0;
}
