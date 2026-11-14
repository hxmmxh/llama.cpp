#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-trace.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    const char * device_name = argc > 1 ? argv[1] : "CPU";
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 40;
    if (repeats < 1 || repeats > 10000) { return 1; }
    ggml_backend_load_all();
    auto device = ggml_backend_dev_by_name(device_name);
    if (!device) { std::fprintf(stderr, "device not found: %s\n", device_name); return 1; }
    auto backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) { return 1; }
    auto ctx = ggml_init({16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true});
    auto a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    auto b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 4);
    auto result = ggml_mul_mat(ctx, a, b);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer || !ggml_backend_supports_op(backend, result)) { return 1; }
    std::vector<float> weights(64 * 64), input(64 * 4), output(64 * 4);
    for (size_t i = 0; i < weights.size(); ++i) { weights[i] = ((int) (i % 19) - 9) / 16.0f; }
    ggml_backend_tensor_set(a, weights.data(), 0, weights.size() * sizeof(float));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        ggml_trace_context_guard guard({(uint64_t) repeat + 1, (uint64_t) repeat + 10001});
        ggml_trace_scope round("probe.round", "round");
        for (size_t i = 0; i < input.size(); ++i) { input[i] = ((int) ((i + repeat) % 13) - 6) / 8.0f; }
        {
            ggml_trace_scope scope("probe.set_input", "copy");
            ggml_backend_tensor_set_async(backend, b, input.data(), 0, input.size() * sizeof(float));
        }
        if (ggml_backend_graph_compute_async(backend, graph) != GGML_STATUS_SUCCESS) { return 1; }
        {
            ggml_trace_scope scope("probe.readback", "readback");
            ggml_backend_tensor_get_async(backend, result, output.data(), 0, output.size() * sizeof(float));
            ggml_backend_synchronize(backend);
        }
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 64; ++row) {
                double reference = 0;
                for (int k = 0; k < 64; ++k) { reference += (double) weights[row * 64 + k] * input[col * 64 + k]; }
                if (!std::isfinite(output[col * 64 + row]) || std::abs(output[col * 64 + row] - reference) > 1e-5) {
                    std::fprintf(stderr, "incorrect output: repeat %d, row %d, col %d\n", repeat, row, col);
                    return 1;
                }
            }
        }
    }
    const auto context = ggml_trace_get_context();
    if (context.round_id || context.graph_id) { return 1; }
    if (ggml_trace_enabled()) {
        std::thread worker([]() {
            ggml_trace_context_guard guard({123, 456});
            ggml_trace_scope scope("escaped \"label\"\n\\", "test");
        });
        worker.join();
    }
    ggml_backend_synchronize(backend);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    ggml_trace_flush();
    uint64_t digest = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < output.size() * sizeof(float); ++i) {
        digest = (digest ^ reinterpret_cast<unsigned char *>(output.data())[i]) * UINT64_C(1099511628211);
    }
    std::printf("output %016llx\n", (unsigned long long) digest);
    return 0;
}
