#include "sampling.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <random>
#include <limits>
#include <stdexcept>
#include <unordered_map>

// Tests that common_sampler_sample_and_accept_n_rejection preserves the target
// distribution regardless of what the draft distribution Q looks like.

static void check_dense_q_storage() {
    common_sampler_draft_probs q;
    GGML_ASSERT(q.empty());
    q.resize(7, 131072);
    float * allocation = q.row(0);
    for (size_t row = 0; row < q.size(); ++row) {
        for (size_t token = 0; token < q.vocab_size(); ++token) {
            q.row(row)[token] = float(row*131072 + token);
        }
    }
    q.clear();
    GGML_ASSERT(q.empty());
    q.resize(2, 131072);
    GGML_ASSERT(q.row(0) == allocation && q.row(1)[131071] == 262143.0f);
    q.resize(7, 131072);
    GGML_ASSERT(q.row(0) == allocation && q.row(6)[131071] == 917503.0f);
    q.truncate(3);
    bool rejected_row = false, rejected_growth = false, rejected_overflow = false, rejected_vocab = false;
    try { (void) q.row(3); } catch (const std::out_of_range &) { rejected_row = true; }
    try { q.truncate(4); } catch (const std::out_of_range &) { rejected_growth = true; }
    try { q.resize(std::numeric_limits<size_t>::max(), 131072); } catch (const std::length_error &) { rejected_overflow = true; }
    try { q.resize(1, 0); } catch (const std::length_error &) { rejected_vocab = true; }
    GGML_ASSERT(rejected_row && rejected_growth && rejected_overflow && rejected_vocab);
    GGML_ASSERT(q.size() == 3 && q.vocab_size() == 131072 && q.row(0) == allocation);
    q.resize(0, 131072);
    GGML_ASSERT(q.empty());
    printf("dense-q storage vocab=131072 growth/reuse/truncate/empty/overflow PASS\n");
}

static std::string grammar_literal(const std::string & piece) {
    GGML_ASSERT(!piece.empty());
    std::string result = "\"";
    for (char ch : piece) {
        if (ch == '\n') { result += "\\n"; }
        else if (ch == '\r') { result += "\\r"; }
        else if (ch == '\t') { result += "\\t"; }
        else {
            if (ch == '\\' || ch == '"') { result += '\\'; }
            result += ch;
        }
    }
    return result + "\"";
}

static void check_sampler_candidates(common_sampler * a, common_sampler * b) {
    const auto * left = common_sampler_get_candidates(a, false);
    const auto * right = common_sampler_get_candidates(b, false);
    GGML_ASSERT(left->size == right->size && left->selected == right->selected && left->sorted == right->sorted);
    for (size_t i = 0; i < left->size; ++i) {
        GGML_ASSERT(left->data[i].id == right->data[i].id);
        GGML_ASSERT(std::memcmp(&left->data[i].p, &right->data[i].p, sizeof(float)) == 0);
        GGML_ASSERT(std::memcmp(&left->data[i].logit, &right->data[i].logit, sizeof(float)) == 0);
    }
}

static void test_dense_q_replay(llama_model * model, llama_context * ctx) {
    check_dense_q_storage();
    const auto * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    GGML_ASSERT(n_vocab >= 16);
    constexpr llama_token accepted_token = 10, other_token = 11, rejected_token = 12;
    llama_memory_clear(llama_get_memory(ctx), true);
    common_batch batch(ctx);
    for (int row = 0; row < 8; ++row) {
        batch.add(accepted_token, row, 0, true);
    }
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    std::vector<float> logits(n_vocab, -INFINITY);
    size_t comparisons = 0;
    for (int grammar = 0; grammar < 3; ++grammar) {
        const bool grammar_first = grammar == 2;
        for (int rows : {0, 1, 3, 7}) {
            for (int scenario = 0; scenario < 7; ++scenario) {
                common_sampler_draft_probs dense;
                if (rows) { dense.resize(rows, n_vocab); }
                std::vector<std::vector<llama_token_data>> legacy(rows);
                llama_tokens draft(rows, accepted_token);
                std::vector<int> indices(rows + 1);
                logits[accepted_token] = std::log(grammar ? 0.12f : 0.6f);
                logits[other_token] = std::log(grammar ? 0.08f : 0.4f);
                logits[rejected_token] = grammar ? std::log(0.8f) : -INFINITY;
                for (int row = 0; row <= rows; ++row) {
                    indices[row] = row;
                    std::memcpy(llama_get_logits_ith(ctx, row), logits.data(), (size_t) n_vocab*sizeof(float));
                }
                for (int row = 0; row < rows; ++row) {
                    float * q = dense.row(row);
                    std::fill(q, q + n_vocab, 0.0f);
                    q[accepted_token] = 0.05f;
                    q[other_token] = 0.95f;
                    if ((scenario == 1 && row == 0) || (scenario == 2 && row == rows - 1) || (scenario == 6 && row == rows/2)) {
                        draft[row] = rejected_token;
                        q[accepted_token] = q[other_token] = 0.1f;
                        q[rejected_token] = 0.8f;
                    }
                    if (scenario == 3) { std::fill(q, q + n_vocab, 0.0f); }
                    if (scenario == 4) {
                        // Exercise the zero-residual fallback without an acceptance draw.
                        std::fill(q, q + n_vocab, 1.0f);
                        draft[row] = n_vocab;
                    }
                    if (scenario == 5) {
                        q[accepted_token] = 0.8f;
                        q[other_token] = 0.2f;
                    }
                    for (llama_token token = 0; token < n_vocab; ++token) {
                        legacy[row].push_back({token, 0.0f, q[token]});
                    }
                    if (scenario == 5) {
                        // Retain the sparse/unordered legacy scatter path as a control.
                        legacy[row].erase(std::remove_if(legacy[row].begin(), legacy[row].end(), [](const llama_token_data & e) { return e.p == 0.0f; }), legacy[row].end());
                        std::reverse(legacy[row].begin(), legacy[row].end());
                        legacy[row].push_back({n_vocab, 0.0f, 0.25f});
                    }
                }
                for (uint32_t seed : {1u, 42u, 9123u}) {
                    common_params_sampling params;
                    params.seed = seed;
                    params.temp = 1.0f;
                    params.samplers = {COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE};
                    params.top_p = 0.95f;
                    if (grammar) {
                        params.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= (" + grammar_literal(common_token_to_piece(vocab, accepted_token)) + " | " + grammar_literal(common_token_to_piece(vocab, other_token)) + ")*"};
                    }
                    common_sampler_ptr old(common_sampler_init(model, params));
                    common_sampler_ptr candidate(common_sampler_clone(old.get()));
                    for (int repeat = 0; repeat < 8; ++repeat) {
                        const auto before = common_sampler_sample_and_accept_n_rejection(old.get(), ctx, indices, draft, legacy, grammar_first);
                        const auto after = common_sampler_sample_and_accept_n_rejection(candidate.get(), ctx, indices, draft, dense, grammar_first);
                        GGML_ASSERT(before == after);
                        check_sampler_candidates(old.get(), candidate.get());
                        if (scenario == 0) { GGML_ASSERT(after.size() == (size_t) rows + 1); }
                        if (rows && (scenario == 1 || scenario == 3 || scenario == 4)) { GGML_ASSERT(after.size() == 1); }
                        if (rows && scenario == 2) { GGML_ASSERT(after.size() == (size_t) rows); }
                        if (rows && scenario == 6) { GGML_ASSERT(after.size() == (size_t) rows/2 + 1); }
                        if (grammar) {
                            for (llama_token token : after) { GGML_ASSERT(token == accepted_token || token == other_token); }
                        }
                        // The next ordinary draws and rejection rounds probe both RNG streams.
                        for (int probe = 0; probe < 3; ++probe) {
                            const llama_token a = common_sampler_sample_logits(old.get(), logits.data(), n_vocab, grammar_first);
                            const llama_token b = common_sampler_sample_logits(candidate.get(), logits.data(), n_vocab, grammar_first);
                            GGML_ASSERT(a == b);
                            check_sampler_candidates(old.get(), candidate.get());
                            common_sampler_accept(old.get(), a, true);
                            common_sampler_accept(candidate.get(), b, true);
                        }
                        ++comparisons;
                    }
                }
            }
        }
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    printf("dense-q replay comparisons=%zu rows=0,1,3,7 grammar=none,last,first future_rng=target,rejection PASS\n", comparisons);
    printf("dense-q replay exact PASS\n");
}

static void test_probabilistic_sampling(llama_model * model, llama_context * ctx) {
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    GGML_ASSERT(n_vocab >= 3);
    std::vector<float> logits(n_vocab, -INFINITY);
    logits[0] = std::log(0.1f);
    logits[1] = std::log(0.3f);
    logits[2] = std::log(0.6f);

    common_params_sampling params;
    params.seed = 42;
    params.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    for (float temp : {0.5f, 1.0f, 2.0f}) {
        params.temp = temp;
        common_sampler_ptr sampler(common_sampler_init(model, params));
        common_sampler_sample_logits(sampler.get(), logits.data(), n_vocab);
        const auto * candidates = common_sampler_get_candidates(sampler.get(), false);
        double sum = 0.0;
        double norm = 0.0;
        for (int32_t i = 0; i < 3; ++i) {
            norm += std::exp((double) logits[i] / temp);
        }
        for (size_t i = 0; i < candidates->size; ++i) {
            const auto & token = candidates->data[i];
            const double expected = std::exp((double) logits[token.id] / temp) / norm;
            GGML_ASSERT(std::abs(token.p - expected) < 1e-6);
            sum += token.p;
        }
        GGML_ASSERT(std::abs(sum - 1.0) < 1e-6);
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    common_batch batch(ctx);
    batch.add(0, 0, 0, true);
    batch.add(1, 1, 0, true);
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

    const float target_probs[] = {0.75f, 0.2f, 0.05f};
    for (int32_t row = 0; row < 2; ++row) {
        float * target_logits = llama_get_logits_ith(ctx, row);
        std::fill(target_logits, target_logits + n_vocab, -INFINITY);
        for (int32_t i = 0; i < 3; ++i) {
            target_logits[i] = std::log(target_probs[i]);
        }
    }

    params.temp = 1.0f;
    common_sampler_ptr draft(common_sampler_init(model, params));
    params.seed = 123;
    common_sampler_ptr target(common_sampler_init(model, params));
    int counts[3] = {};
    const int n_samples = 12000;
    for (int i = 0; i < n_samples; ++i) {
        const llama_token token = common_sampler_sample_logits(draft.get(), logits.data(), n_vocab);
        const auto * candidates = common_sampler_get_candidates(draft.get(), true);
        const std::vector<std::vector<llama_token_data>> q = {
            {candidates->data, candidates->data + candidates->size}
        };
        const auto accepted = common_sampler_sample_and_accept_n_rejection(target.get(), ctx, {0, 1}, {token}, q);
        GGML_ASSERT(accepted[0] >= 0 && accepted[0] < 3);
        ++counts[accepted[0]];
    }
    for (int i = 0; i < 3; ++i) {
        GGML_ASSERT(std::abs((double) counts[i] / n_samples - target_probs[i]) < 0.025);
    }
    const std::vector<std::vector<llama_token_data>> outside_vocab = {{{n_vocab, 0.0f, 1.0f}}};
    for (int i = 0; i < 100; ++i) {
        const auto accepted = common_sampler_sample_and_accept_n_rejection(target.get(), ctx, {0, 1}, {n_vocab}, outside_vocab);
        GGML_ASSERT(accepted.size() == 1 && accepted[0] >= 0 && accepted[0] < 3);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
}

static int test_speculative_rejection_sampling(common_params & params, int n_trials, bool dense_q_only) {
    params.sampling.backend_sampling = false;
    if (dense_q_only) {
        params.warmup = false;
    }
    auto llama_init = common_init_from_params(params);

    auto * model_tgt = llama_init->model();
    auto * ctx_tgt   = llama_init->context();
    if (!model_tgt || !ctx_tgt) {
        LOG_ERR("failed to initialize target model and context\n");
        return 1;
    }

    test_dense_q_replay(model_tgt, ctx_tgt);
    if (dense_q_only) {
        return 0;
    }
    test_probabilistic_sampling(model_tgt, ctx_tgt);

    const llama_tokens prompt = { 1, 2, 3, 1, 2 };

    // Process the prompt with the target model to get its true distribution for
    // the position right after the prompt.
    common_batch batch = common_batch_get_one(ctx_tgt, prompt);
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        LOG_ERR("failed to decode the test prompt\n");
        return 1;
    }

    common_params_sampling sampling;
    sampling.seed     = 18;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    sampling.temp     = 1.0f;

    // Initialize the sampler for the target model.
    common_sampler_ptr smpl(common_sampler_init(model_tgt, sampling));
    if (!smpl) {
        LOG_ERR("failed to initialize target sampler\n");
        return 1;
    }

    // Sample once so the sampler chain runs and populates cur_p. We don't need
    // the returned token id itself so the return value is ignored.
    common_sampler_sample(smpl.get(), ctx_tgt, -1);

    // Get the target candidates distribution in sorted order.
    const llama_token_data_array * dist_tgt = common_sampler_get_candidates(smpl.get(), true);

    // Clone smpl before the trial loop starts. We need the dist_tgt pointer to stay valid.
    common_sampler_ptr smpl_verify(common_sampler_clone(smpl.get()));
    if (!smpl_verify) {
        LOG_ERR("failed to clone target sampler\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::mt19937 rng(18);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);

    std::unordered_map<llama_token, int> token_counts;

    std::vector<llama_token_data> q(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        q[i].id = i;
        q[i].logit = 0.0f;
    }

    for (int trial = 0; trial < n_trials; ++trial) {
        float q_sum = 0.0f;

        for (auto & td : q) {
            td.p = uni(rng);
            q_sum += td.p;
        }

        const float inv_sum = 1.0f / q_sum;
        for (auto & td : q) {
            td.p *= inv_sum;
        }

        float u = uni(rng);

        // Fallback to the last bucket to absorb residual probability from FP precision loss
        llama_token draft_tok = q.back().id;
        for (const auto & e : q) {
            u -= e.p;
            if (u <= 0.0f) {
                draft_tok = e.id;
                break;
            }
        }

        // Both indices reuse the prompt's last output logits. The second entry
        // is the bonus draw slot; it is always stale and its result is ignored.
        const llama_tokens draft = { draft_tok };
        const std::vector<std::vector<llama_token_data>> draft_q = { q };
        const std::vector<int> idxs = { -1, -1 };

        const std::vector<llama_token> result =
            common_sampler_sample_and_accept_n_rejection(smpl_verify.get(), ctx_tgt, idxs, draft, draft_q, false);

        token_counts[result[0]]++;
    }

    LOG("\n");
    LOG("Empirical vs target distribution after %d trials:\n", n_trials);

    bool   ok      = true;
    size_t covered = 0;

    auto get_count_for = [&](llama_token id) -> int {
        const auto it = token_counts.find(id);
        return it != token_counts.end() ? it->second : 0;
    };

    // Iterate over the target distribution
    for (size_t i = 0; i < dist_tgt->size; ++i) {
        const llama_token id  = dist_tgt->data[i].id;
        const float       p_t = dist_tgt->data[i].p;

        const int   count = get_count_for(id);
        const float p_e   = (float) count / n_trials;
        const float diff  = std::fabs(p_t - p_e);

        // We only deal with ids in dist_tgt, and if each trial result was one
        // of dist_tgt's candidates then covered ends up equal to the number of
        // trials. If not this inconsistency tells us how many trials produced
        // a token outside of the target's distribution.
        covered += count;

        const float standard_error = std::sqrt(p_t * (1.0f - p_t) / n_trials);
        const float tolerance      = 6.0f * standard_error;

        LOG("token %3d : target = %f, empirical = %f, diff = %f, tol = %f\n", id, p_t, p_e, diff, tolerance);

        if (diff > tolerance) {
            LOG_ERR("token %d: empirical probability %f too far from target %f (diff %f > tol %f)\n",
                    id, p_e, p_t, diff, tolerance);
            ok = false;
        }
    }

    if (covered != (size_t) n_trials) {
        LOG_ERR("%zu of %d trials produced a token outside dist_tgt's candidates\n",
                (size_t) n_trials - covered, n_trials);
        ok = false;
    }

    if (!ok) {
        return 1;
    }

    return 0;
}

struct parsed_args {
    int n_trials;
    bool dense_q_only = false;
    std::vector<char *> argv;
};

static std::optional<parsed_args> parse_n_trials_arg(int argc, char ** argv, int default_n_trials) {
    parsed_args result;
    result.n_trials = default_n_trials;
    result.argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dense-q-only") == 0) {
            result.dense_q_only = true;
        } else if (strcmp(argv[i], "--n-trials") == 0) {
            if (i + 1 >= argc) {
                LOG_ERR("%s: --n-trials requires a number argument\n", __func__);
                return std::nullopt;
            }
            int value = std::atoi(argv[i + 1]);
            if (value <= 0) {
                LOG_ERR("%s: --n-trials requires a positive number argument\n", __func__);
                return std::nullopt;
            }
            result.n_trials = value;
            i++;
        } else {
            result.argv.push_back(argv[i]);
        }
    }
    result.argv.push_back(nullptr);

    return result;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 256;

    common_init();

    auto parsed = parse_n_trials_arg(argc, argv, 5000);
    if (!parsed) {
        return 1;
    }
    const int fargc = (int) parsed->argv.size() - 1;

    if (!common_params_parse(fargc, parsed->argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("Using target model '%s'\n", params.model.path.c_str());

    llama_backend_init();

    const int result = test_speculative_rejection_sampling(params, parsed->n_trials, parsed->dense_q_only);
    if (result == 0) {
        LOG("\nAll tests passed.\n");
    } else {
        LOG("\nSome tests failed.\n");
    }

    llama_backend_free();
    return result;
}
