#include "speculative-calibration.h"

#include "log.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace {

struct calibration_cache_guard {
    llama_context * target;
    llama_context * draft;

    ~calibration_cache_guard() {
        llama_synchronize(target);
        if (draft) { llama_synchronize(draft); }
        llama_memory_clear(llama_get_memory(target), false);
        if (draft) { llama_memory_clear(llama_get_memory(draft), false); }
    }
};

std::vector<uint8_t> save_prefix(llama_context * ctx) {
    llama_synchronize(ctx);
    std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
    if (state.empty() || llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != state.size()) {
        throw std::runtime_error("failed to save calibration prefix");
    }
    return state;
}

void restore_prefix(llama_context * ctx, const std::vector<uint8_t> & state) {
    if (llama_state_seq_set_data(ctx, state.data(), state.size(), 0) != state.size()) {
        throw std::runtime_error("failed to restore calibration prefix");
    }
    llama_synchronize(ctx);
}

void remove_suffix(llama_context * ctx, llama_pos pos) {
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, pos, -1)) {
        throw std::runtime_error("calibration requires partial KV suffix removal");
    }
}

void validate_sample(
        llama_context * target,
        int n_vocab,
        const common_speculative_calibration_sample & sample,
        const std::vector<std::vector<llama_token_data>> & sparse_q,
        const common_sampler_draft_probs & dense_q) {
    if (sample.draft_tokens.size() != (size_t) sample.draft ||
            sample.output_tokens.empty() || sample.output_tokens.size() > (size_t) sample.draft + 1) {
        throw std::runtime_error("unexpected calibration draft/output count");
    }
    for (llama_token token : sample.draft_tokens) {
        if (token < 0 || token >= n_vocab) {
            throw std::runtime_error("calibration draft token outside vocabulary");
        }
    }
    for (llama_token token : sample.output_tokens) {
        if (token < 0 || token >= n_vocab) {
            throw std::runtime_error("calibration output token outside vocabulary");
        }
    }
    if ((!dense_q.empty() && (dense_q.size() != (size_t) sample.draft || dense_q.vocab_size() != (size_t) n_vocab)) ||
            (dense_q.empty() && sparse_q.size() != (size_t) sample.draft)) {
        throw std::runtime_error("unexpected calibration probability shape");
    }
    for (int row = 0; row < sample.draft; ++row) {
        double sum = 0.0;
        if (!dense_q.empty()) {
            for (int token = 0; token < n_vocab; ++token) {
                const float p = dense_q.row(row)[token];
                if (!std::isfinite(p) || p < 0.0f) {
                    throw std::runtime_error("invalid calibration draft probability");
                }
                sum += p;
            }
        } else {
            for (const auto & candidate : sparse_q[row]) {
                if (candidate.id < 0 || candidate.id >= n_vocab || !std::isfinite(candidate.p) || candidate.p < 0.0f) {
                    throw std::runtime_error("invalid calibration draft candidate");
                }
                sum += candidate.p;
            }
        }
        if (std::abs(sum - 1.0) > 1e-3) {
            throw std::runtime_error("calibration draft probabilities are not normalized");
        }
    }
    for (int row = 0; row < sample.target_rows; ++row) {
        const float * logits = llama_get_logits_ith(target, row);
        if (!logits) {
            throw std::runtime_error("missing calibration target logits");
        }
        for (int token = 0; token < n_vocab; ++token) {
            if (!std::isfinite(logits[token])) {
                throw std::runtime_error("non-finite calibration target logits");
            }
        }
    }
}

std::vector<common_speculative_calibration_sample> calibrate_ordinary(
        const common_params & params, llama_context * target, const llama_tokens & prompt,
        const common_speculative_calibration_params & calibration) {
    if (!target || !llama_get_memory(target) || params.n_parallel != 1 || llama_n_seq_max(target) != 1 ||
            llama_memory_seq_pos_max(llama_get_memory(target), 0) >= 0 || prompt.size() < 2 ||
            prompt.size() > llama_n_ctx_seq(target) || calibration.warmup < 1 || calibration.warmup > 1000 ||
            calibration.repetitions < 1 || calibration.repetitions > 10000 ||
            !std::isfinite(params.sampling.temp) || params.sampling.temp <= 0 ||
            params.sampling.seed == LLAMA_DEFAULT_SEED || params.sampling.backend_sampling) {
        throw std::invalid_argument("ordinary calibration requires an empty single-sequence context and valid sizes/sampling");
    }
    calibration_cache_guard guard{target, nullptr};
    const auto * model = llama_get_model(target);
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (auto token : prompt) {
        if (token < 0 || token >= vocab) { throw std::invalid_argument("invalid ordinary calibration prompt token"); }
    }
    const llama_tokens history(prompt.begin(), prompt.end() - 1);
    const llama_pos pos = (llama_pos) history.size();
    common_batch batch(target);
    const size_t chunk = std::min(llama_n_batch(target), llama_n_ubatch(target));
    for (size_t offset = 0; offset < history.size(); offset += chunk) {
        batch.clear();
        for (size_t i = offset; i < std::min(history.size(), offset + chunk); ++i) {
            batch.add(history[i], (llama_pos) i, 0, false);
        }
        if (llama_process(target, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("ordinary calibration prefill failed");
        }
    }
    const auto prefix = save_prefix(target);
    auto sampling = params.sampling;
    common_sampler_ptr sampler(common_sampler_init(model, sampling));
    if (!sampler) { throw std::runtime_error("ordinary calibration sampler failed"); }
    std::vector<common_speculative_calibration_sample> samples;
    const std::vector<std::vector<llama_token_data>> sparse_q;
    const common_sampler_draft_probs dense_q;
    for (int repeat = -calibration.warmup; repeat < calibration.repetitions; ++repeat) {
        restore_prefix(target, prefix);
        if (llama_memory_seq_pos_max(llama_get_memory(target), 0) != pos - 1) {
            throw std::runtime_error("ordinary calibration prefix was not restored");
        }
        common_sampler_reset(sampler.get());
        for (auto token : history) { common_sampler_accept(sampler.get(), token, false); }
        common_speculative_calibration_sample sample;
        sample.repetition = repeat;
        sample.target_rows = 1;
        const int64_t start = ggml_time_us();
        batch.clear();
        batch.add(prompt.back(), pos, 0, true);
        if (llama_process(target, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("ordinary calibration decode failed");
        }
        llama_synchronize(target);
        const int64_t after_target = ggml_time_us();
        const auto token = common_sampler_sample(sampler.get(), target, 0);
        common_sampler_accept(sampler.get(), token, true);
        sample.output_tokens.push_back(token);
        const int64_t after_sampling = ggml_time_us();
        remove_suffix(target, pos + 1);
        llama_synchronize(target);
        const int64_t end = ggml_time_us();
        sample.target_us = after_target - start;
        sample.sampling_us = after_sampling - after_target;
        sample.rollback_us = end - after_sampling;
        sample.round_us = end - start;
        validate_sample(target, vocab, sample, sparse_q, dense_q);
        if (sample.round_us <= 0) { throw std::runtime_error("invalid ordinary calibration timing"); }
        if (repeat >= 0) { samples.push_back(std::move(sample)); }
    }
    LOG_INF("calibration: D0 ordinary decode, draft rows=0, target rows=1, samples=%d\n", calibration.repetitions);
    return samples;
}

} // namespace

std::vector<common_speculative_calibration_sample> common_speculative_calibrate(
        const common_params & params,
        llama_context * target,
        llama_context * draft,
        const llama_tokens & prompt,
        const common_speculative_calibration_params & calibration) {
    if (calibration.ordinary_only) { return calibrate_ordinary(params, target, prompt, calibration); }
    const int maximum = calibration.max_draft;
    if (maximum < 1 || maximum > COMMON_SPECULATIVE_CALIBRATION_MAX_DRAFT) {
        throw std::invalid_argument("calibration maximum draft must be D1-D7");
    }
    if (!target || !draft || !llama_get_memory(target) || !llama_get_memory(draft)) {
        throw std::invalid_argument("calibration requires target and draft KV contexts");
    }
    if (params.speculative.types != std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK} ||
            !params.speculative.draft.probabilistic || params.speculative.draft.p_min != 0.0f ||
            params.speculative.has_synth()) {
        throw std::invalid_argument("calibration requires probabilistic DSpark without synthetic acceptance or confidence truncation");
    }
    if (params.n_parallel != 1 || llama_n_seq_max(target) != 1 || llama_n_seq_max(draft) != 1 ||
            llama_memory_seq_pos_max(llama_get_memory(target), 0) >= 0 ||
            llama_memory_seq_pos_max(llama_get_memory(draft), 0) >= 0) {
        throw std::invalid_argument("calibration requires empty single-sequence contexts");
    }
    if (calibration.warmup < 1 || calibration.warmup > 1000 ||
            calibration.repetitions < 1 || calibration.repetitions > 10000 ||
            prompt.size() < 2 || prompt.size() + maximum > llama_n_ctx_seq(target) ||
            prompt.size() + maximum > llama_n_ctx_seq(draft) ||
            llama_n_batch(target) < (uint32_t) maximum + 1 || llama_n_ubatch(target) < (uint32_t) maximum + 1 ||
            llama_n_batch(draft) < (uint32_t) maximum + 1 || llama_n_ubatch(draft) < (uint32_t) maximum + 1 ||
            !std::isfinite(params.sampling.temp) || params.sampling.temp <= 0.0f || params.sampling.seed == LLAMA_DEFAULT_SEED) {
        throw std::invalid_argument("invalid calibration sizes, temperature or seed");
    }
    const auto * model = llama_get_model(target);
    const auto * model_draft = llama_get_model(draft);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (!common_speculative_are_compatible(model, model_draft)) {
        throw std::invalid_argument("incompatible calibration vocabularies");
    }
    for (llama_token token : prompt) {
        if (token < 0 || token >= n_vocab) {
            throw std::invalid_argument("calibration prompt token outside vocabulary");
        }
    }
    char anchor[16] = {};
    if (llama_model_meta_val_str(model_draft, "dflash.sample_from_anchor", anchor, sizeof(anchor)) >= 0 &&
            std::string(anchor) != "true") {
        throw std::invalid_argument("calibration currently requires anchor-first DSpark");
    }

    // This capability probe clears KV; run it only while both contexts are empty.
    const bool allow_async = calibration.runtime_pipeline &&
            common_context_can_seq_rm(target) == COMMON_CONTEXT_SEQ_RM_TYPE_PART &&
            common_context_can_seq_rm(draft) == COMMON_CONTEXT_SEQ_RM_TYPE_PART;
    auto samples = calibration.include_ordinary ? calibrate_ordinary(params, target, prompt, calibration) :
            std::vector<common_speculative_calibration_sample>{};
    calibration_cache_guard cache_guard{target, draft};
    common_params_speculative spec_params = params.speculative;
    spec_params.draft.ctx_tgt = target;
    spec_params.draft.ctx_dft = draft;
    spec_params.draft.n_max = maximum;
    spec_params.draft.n_min = 0;
    common_speculative_ptr spec(common_speculative_init(spec_params, 1));
    if (!spec || common_speculative_n_max(spec.get()) != maximum) {
        throw std::runtime_error("draft model cannot calibrate the configured maximum length");
    }

    const llama_tokens history(prompt.begin(), prompt.end() - 1);
    const llama_pos pos = (llama_pos) history.size();
    common_batch batch(target);
    const size_t chunk = std::min(llama_n_batch(target), llama_n_ubatch(target));
    for (size_t offset = 0; offset < history.size(); offset += chunk) {
        batch.clear();
        for (size_t i = offset; i < std::min(history.size(), offset + chunk); ++i) {
            batch.add(history[i], (llama_pos) i, 0, false);
        }
        if (llama_process(target, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 ||
                !common_speculative_process(spec.get(), batch)) {
            throw std::runtime_error("failed to prepare calibration prefix");
        }
    }
    const auto target_prefix = save_prefix(target);
    const auto draft_prefix = save_prefix(draft);
    auto sampling_params = params.sampling;
    common_sampler_ptr sampler(common_sampler_init(model, sampling_params));
    if (!sampler) {
        throw std::runtime_error("failed to initialize calibration sampler");
    }

    samples.reserve((maximum + 1) * calibration.repetitions);
    // Reuse the prefix-building speculator for the maximum, then measure the shorter blocks.
    for (int index = 0; index < maximum; ++index) {
        const int depth = index == 0 ? maximum : index;
        if (index > 0) {
            spec.reset();
            spec_params.draft.n_max = depth;
            spec.reset(common_speculative_init(spec_params, 1));
        }
        if (!spec || common_speculative_n_max(spec.get()) != depth) {
            throw std::runtime_error("calibration draft length was clamped");
        }
        std::vector<std::vector<llama_token_data>> sparse_q;
        common_sampler_draft_probs dense_q;
        for (int repeat = -calibration.warmup; repeat < calibration.repetitions; ++repeat) {
            restore_prefix(target, target_prefix);
            restore_prefix(draft, draft_prefix);
            if (llama_memory_seq_pos_max(llama_get_memory(target), 0) != pos - 1 ||
                    llama_memory_seq_pos_max(llama_get_memory(draft), 0) != pos - 1) {
                throw std::runtime_error("calibration prefix position was not restored");
            }
            common_sampler_reset(sampler.get());
            for (llama_token token : history) {
                common_sampler_accept(sampler.get(), token, false);
            }
            common_speculative_begin(spec.get(), 0, history);
            common_speculative_calibration_sample sample;
            sample.draft = depth;
            sample.repetition = repeat;
            sample.draft_rows = depth;
            sample.target_rows = depth + 1;
            sparse_q.clear();
            dense_q.clear();
            auto & dp = common_speculative_get_draft_params(spec.get(), 0);
            dp = {};
            dp.drafting = true;
            dp.n_max = depth;
            dp.pos0 = pos;
            dp.id_last = prompt.back();
            dp.prompt = &history;
            dp.result = &sample.draft_tokens;
            dp.result_q = &sparse_q;
            dp.result_q_dense = &dense_q;
            dp.temp = params.sampling.temp;
            dp.seed = params.sampling.seed;
            dp.actual_draft_length = calibration.runtime_pipeline;
            dp.allow_async = allow_async;
            if (llama_memory_seq_pos_max(llama_get_memory(target), 0) != pos - 1 ||
                    llama_memory_seq_pos_max(llama_get_memory(draft), 0) != pos - 1) {
                throw std::runtime_error("calibration setup changed the restored prefix");
            }

            const int64_t start = ggml_time_us();
            common_speculative_draft(spec.get());
            if (!calibration.runtime_pipeline) {
                common_speculative_draft_finish(spec.get());
                llama_synchronize(draft);
            }
            if (sample.draft_tokens.size() != (size_t) depth) {
                throw std::runtime_error("calibration did not execute the requested draft length");
            }
            const int64_t after_draft = ggml_time_us();
            remove_suffix(draft, pos);
            if (!calibration.runtime_pipeline) { llama_synchronize(draft); }
            const int64_t after_cleanup = ggml_time_us();
            batch.clear();
            batch.add(prompt.back(), pos, 0, true);
            for (size_t i = 0; i < sample.draft_tokens.size(); ++i) {
                batch.add(sample.draft_tokens[i], pos + 1 + (llama_pos) i, 0, true);
            }
            if (llama_process(target, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
                throw std::runtime_error("calibration target verification failed");
            }
            if (calibration.runtime_pipeline) { common_speculative_draft_finish(spec.get()); }
            else { llama_synchronize(target); }
            if (sample.draft_tokens.size() != (size_t) depth ||
                    (!dense_q.empty() ? dense_q.size() : sparse_q.size()) != (size_t) depth) {
                throw std::runtime_error("calibration did not complete the requested draft length");
            }
            const int64_t after_target = ggml_time_us();
            if (!common_speculative_process(spec.get(), batch)) {
                throw std::runtime_error("calibration draft feature injection failed");
            }
            if (!calibration.runtime_pipeline) { llama_synchronize(draft); }
            const int64_t after_context = ggml_time_us();
            std::vector<int> rows(depth + 1);
            std::iota(rows.begin(), rows.end(), 0);
            sample.output_tokens = !dense_q.empty() ?
                    common_sampler_sample_and_accept_n_rejection(sampler.get(), target, rows, sample.draft_tokens, dense_q) :
                    common_sampler_sample_and_accept_n_rejection(sampler.get(), target, rows, sample.draft_tokens, sparse_q);
            if (sample.output_tokens.empty() || sample.output_tokens.size() > (size_t) depth + 1) {
                throw std::runtime_error("calibration rejection sampler returned an invalid count");
            }
            const int64_t after_sampling = ggml_time_us();
            sample.accepted = (int) sample.output_tokens.size() - 1;
            common_speculative_accept(spec.get(), 0, (uint16_t) sample.accepted);
            remove_suffix(target, pos + 1 + sample.accepted);
            remove_suffix(draft, pos + 1 + sample.accepted);
            llama_synchronize(target);
            llama_synchronize(draft);
            const int64_t end = ggml_time_us();
            sample.draft_us = after_draft - start;
            sample.draft_cleanup_us = after_cleanup - after_draft;
            sample.target_us = after_target - after_cleanup;
            sample.context_us = after_context - after_target;
            sample.sampling_us = after_sampling - after_context;
            sample.rollback_us = end - after_sampling;
            if (calibration.runtime_pipeline) {
                // Overlapped phases cannot be attributed using host enqueue timestamps.
                sample.draft_us = sample.draft_cleanup_us = sample.target_us = 0;
                sample.context_us = sample.sampling_us = sample.rollback_us = 0;
            }
            sample.round_us = end - start;
            validate_sample(target, n_vocab, sample, sparse_q, dense_q);
            if (sample.round_us <= 0) {
                throw std::runtime_error("invalid calibration timing");
            }
            if (repeat >= 0) {
                samples.push_back(std::move(sample));
            }
        }
        LOG_INF("calibration: D%d, draft rows=%d, target rows=%d, samples=%d\n",
                depth, depth, depth + 1, calibration.repetitions);
    }
    std::sort(samples.begin(), samples.end(), [](const common_speculative_calibration_sample & a,
                                               const common_speculative_calibration_sample & b) {
        return a.draft != b.draft ? a.draft < b.draft : a.repetition < b.repetition;
    });
    return samples;
}
