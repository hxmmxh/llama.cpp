#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

// Model-backed integration check. Supply the same target/draft options as the server.
// Compare ordinary logits and resumed draft distributions against continuous feature
// injection, including a gap long enough to overwrite a small sliding window.
int main(int argc, char ** argv) {
    try {
        common_params params;
        params.n_parallel = 1;
        params.n_ctx = 1024;
        params.n_batch = params.n_ubatch = 128;
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
        params.speculative.draft.probabilistic = true;
        params.speculative.draft.n_max = 7;
        params.sampling.seed = 42;
        common_init();
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) { return 1; }
        params.warmup = params.fit_params = false;
        params.n_outputs_max = params.n_outputs_max_per_seq = 8;
        llama_backend_init();
        {
            auto target = common_init_from_params(params);
            require(target && target->context(), "target load failed");
            auto draft_params = common_base_params_to_speculative(params);
            auto draft = common_speculative_init_from_params(draft_params, target->model(), target->context());
            require(draft && draft->context(), "draft load failed");
            auto * ctx = target->context();
            auto * dft = draft->context();
            params.speculative.draft.ctx_tgt = ctx;
            params.speculative.draft.ctx_dft = dft;
            const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(target->model()));
            const auto seed = common_tokenize(ctx, "Explain how a computer stores information.\n", false, true);
            require(!seed.empty(), "empty seed prompt");
            std::vector<std::vector<float>> reference_logits;
            std::vector<std::vector<float>> reference_q;
            for (bool pause : {false, true}) {
                llama_memory_clear(llama_get_memory(ctx), false);
                llama_memory_clear(llama_get_memory(dft), false);
                common_speculative_ptr spec(common_speculative_init(params.speculative, 1));
                require(bool(spec), "speculator init failed");
                common_batch batch(ctx);
                llama_tokens prompt;
                auto feed = [&](int count, bool compare) {
                    for (int i = 0; i < count; ++i) {
                        batch.clear();
                        batch.add(seed[prompt.size() % seed.size()], (llama_pos) prompt.size(), 0, true);
                        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decode failed");
                        require(common_speculative_process(spec.get(), batch), "feature injection/record failed");
                        const float * logits = llama_get_logits_ith(ctx, 0);
                        if (compare) {
                            if (!pause) {
                                reference_logits.emplace_back(logits, logits + vocab);
                            } else {
                                const auto & expected = reference_logits.at(prompt.size() - 32);
                                for (int v = 0; v < vocab; ++v) {
                                    require(std::isfinite(logits[v]) && std::abs(logits[v] - expected[v]) < 1e-3f,
                                            "ordinary/resumed target logits differ from continuous path");
                                }
                            }
                        }
                        prompt.push_back(batch.tokens[0].id);
                    }
                };
                feed(32, false);
                common_speculative_begin(spec.get(), 0, prompt);
                int episode = 0;
                for (int gap : {1, 8, 513}) {
                    const auto pos = (llama_pos) prompt.size();
                    if (pause) {
                        require(common_speculative_set_active(spec.get(), false, 0, pos), "pause failed");
                    }
                    const auto frozen_pos = llama_memory_seq_pos_max(llama_get_memory(dft), 0);
                    feed(gap, true);
                    if (pause) {
                        require(llama_memory_seq_pos_max(llama_get_memory(dft), 0) == frozen_pos,
                                "D0 still injected draft KV");
                        require(common_speculative_set_active(spec.get(), true, 0, (llama_pos) prompt.size()), "resume failed");
                    }
                    feed(1, true);
                    llama_tokens candidates;
                    std::vector<std::vector<llama_token_data>> q;
                    common_sampler_draft_probs dense;
                    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
                    dp = {};
                    dp.drafting = true;
                    dp.n_max = 3;
                    dp.actual_draft_length = true;
                    dp.pos0 = (llama_pos) prompt.size();
                    dp.id_last = seed[prompt.size() % seed.size()];
                    dp.prompt = &prompt;
                    dp.result = &candidates;
                    dp.result_q = &q;
                    dp.result_q_dense = &dense;
                    dp.temp = 1.0f;
                    dp.seed = 42;
                    common_speculative_draft(spec.get());
                    common_speculative_draft_finish(spec.get());
                    require(candidates.size() == 3, "resumed draft length differs");
                    std::vector<float> probs(3 * vocab, 0.0f);
                    for (int r = 0; r < 3; ++r) {
                        if (!dense.empty()) {
                            std::copy(dense.row(r), dense.row(r) + vocab, probs.begin() + r * vocab);
                        } else {
                            for (const auto & token : q.at(r)) { probs[r * vocab + token.id] = token.p; }
                        }
                    }
                    if (!pause) {
                        reference_q.push_back(probs);
                    } else {
                        for (size_t i = 0; i < probs.size(); ++i) {
                            require(std::isfinite(probs[i]) && std::abs(probs[i] - reference_q[episode][i]) < 1e-3f,
                                    "resumed draft distribution differs from continuous path");
                        }
                    }
                    require(llama_memory_seq_rm(llama_get_memory(dft), 0, (llama_pos) prompt.size(), -1), "draft trim failed");
                    ++episode;
                }
                require(common_speculative_set_active(spec.get(), false, 0, (llama_pos) prompt.size()), "final pause failed");
                feed(1, false);
                common_speculative_invalidate_resume(spec.get());
                require(!common_speculative_set_active(spec.get(), true, 0, (llama_pos) prompt.size()), "stale checkpoint resumed");
                require(common_speculative_reset_active(spec.get()), "request cleanup did not invalidate cached prefix");
                require(common_speculative_is_active(spec.get()), "next request feature hooks remain disabled");
            }
        }
        llama_backend_free();
        std::puts("ordinary pause/resume: logits, draft distributions, frozen KV and request cleanup PASS");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ordinary pause/resume: %s\n", error.what());
        return 1;
    }
    return 0;
}
