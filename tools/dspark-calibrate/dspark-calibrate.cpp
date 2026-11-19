#include "arg.h"
#include "common.h"
#include "log.h"
#include "speculative.h"
#include "speculative-calibration.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

using json = nlohmann::ordered_json;

static void print_usage(int, char ** argv) {
    LOG("\nUsage: %s -m target.gguf -md draft.gguf [common options] [calibration options]\n", argv[0]);
    LOG("  --calibration-history N       cached prefix tokens (default: 512)\n");
    LOG("  --calibration-warmup N        warmup rounds per depth (default: 2, minimum: 1)\n");
    LOG("  --calibration-repetitions N   measured rounds per depth (default: 5)\n");
    LOG("  --calibration-output FILE     write JSON to a new file (default: stdout)\n");
    LOG("  --calibration-ordinary-only  measure D0 without loading a draft model\n");
    LOG("D0-D7 use one model load, identical saved prefixes and synchronized stage timings.\n");
}

static int parse_count(const std::string & value) {
    size_t end = 0;
    const int count = std::stoi(value, &end);
    if (end != value.size()) {
        throw std::invalid_argument("calibration count must be an integer");
    }
    return count;
}

static json stage_stats(const std::vector<int64_t> & values) {
    double sum = 0.0;
    for (auto value : values) {
        sum += value;
    }
    const double mean = sum / values.size();
    double squares = 0.0;
    for (auto value : values) {
        squares += (value - mean) * (value - mean);
    }
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const size_t middle = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[middle] :
            ((double) sorted[middle - 1] + sorted[middle]) / 2.0;
    return {{"mean_us", mean}, {"median_us", median},
            {"min_us", sorted.front()}, {"max_us", sorted.back()},
            {"sd_us", values.size() > 1 ? std::sqrt(squares / (values.size() - 1)) : 0.0}};
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    bool backend_initialized = false;
    try {
        common_speculative_calibration_params calibration;
        int history_tokens = 512;
        std::string output;
        std::vector<char *> common_args{argv[0]};
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--calibration-ordinary-only") {
                calibration.ordinary_only = true;
                continue;
            }
            if (arg == "--calibration-history" || arg == "--calibration-warmup" ||
                    arg == "--calibration-repetitions" || arg == "--calibration-output") {
                if (++i >= argc) {
                    throw std::invalid_argument("missing value for " + arg);
                }
                if (arg == "--calibration-history") {
                    history_tokens = parse_count(argv[i]);
                } else if (arg == "--calibration-warmup") {
                    calibration.warmup = parse_count(argv[i]);
                } else if (arg == "--calibration-repetitions") {
                    calibration.repetitions = parse_count(argv[i]);
                } else {
                    output = argv[i];
                }
            } else {
                common_args.push_back(argv[i]);
            }
        }
        if (history_tokens < 1 || history_tokens > 1048576 ||
                calibration.warmup < 1 || calibration.warmup > 1000 ||
                calibration.repetitions < 1 || calibration.repetitions > 10000) {
            throw std::invalid_argument("calibration history/warmup/repetitions out of range");
        }
        if (!output.empty() && std::filesystem::exists(output)) {
            throw std::invalid_argument("calibration output already exists: " + output);
        }
        common_params params;
        params.n_ctx = 2048;
        params.n_batch = 512;
        params.n_ubatch = 128;
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
        params.speculative.draft.n_max = COMMON_SPECULATIVE_CALIBRATION_MAX_DRAFT;
        params.speculative.draft.probabilistic = true;
        params.sampling.seed = 0;
        params.sampling.temp = 1.0f;
        params.sampling.top_p = 0.95f;
        params.sampling.top_k = 0;
        params.sampling.min_p = 0.0f;
        common_init();
        if (!common_params_parse((int) common_args.size(), common_args.data(), params, LLAMA_EXAMPLE_SPECULATIVE, print_usage)) {
            return 1;
        }
        if (params.model.path.empty() || (!calibration.ordinary_only && params.speculative.draft.mparams.path.empty()) ||
                params.n_parallel != 1 || params.speculative.draft.n_max != COMMON_SPECULATIVE_CALIBRATION_MAX_DRAFT ||
                params.speculative.draft.n_min != 0 || params.speculative.has_synth()) {
            throw std::invalid_argument("supply target/draft files, one sequence and an untruncated seven-token draft maximum");
        }
        if (calibration.ordinary_only) {
            params.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE};
            params.speculative.draft.mparams = {};
        }
        params.warmup = false;
        params.fit_params = false;
        params.n_outputs_max = calibration.ordinary_only ? 1 : COMMON_SPECULATIVE_CALIBRATION_MAX_DRAFT + 1;
        params.n_outputs_max_per_seq = params.n_outputs_max;
        if (params.n_ctx > 0 && history_tokens + params.n_outputs_max > params.n_ctx) {
            throw std::invalid_argument("context capacity is smaller than history + 8");
        }
        llama_backend_init();
        backend_initialized = true;
        llama_numa_init(params.numa);
        {
            const int64_t startup_start = ggml_time_us();
            auto target = common_init_from_params(params);
            if (!target || !target->model() || !target->context()) {
                throw std::runtime_error("failed to load target");
            }
            auto draft_params = common_base_params_to_speculative(params);
            common_speculative_init_result_ptr draft;
            if (!calibration.ordinary_only) { draft = common_speculative_init_from_params(draft_params, target->model(), target->context()); }
            if (!calibration.ordinary_only && (!draft || !draft->context())) {
                throw std::runtime_error("failed to load draft");
            }
            const int64_t loaded = ggml_time_us();
            const std::string text = params.prompt.empty() ?
                    "Explain how a computer stores information and processes instructions.\n" : params.prompt;
            const auto seed_tokens = common_tokenize(target->context(), text, false, true);
            if (seed_tokens.empty()) {
                throw std::invalid_argument("calibration prompt has no tokens");
            }
            llama_tokens prompt;
            prompt.reserve(history_tokens + 1);
            const llama_token bos = llama_vocab_bos(llama_model_get_vocab(target->model()));
            if (bos != LLAMA_TOKEN_NULL) {
                prompt.push_back(bos);
            }
            for (size_t i = 0; prompt.size() < (size_t) history_tokens + 1; ++i) {
                prompt.push_back(seed_tokens[i % seed_tokens.size()]);
            }
            const auto samples = common_speculative_calibrate(params, target->context(), draft ? draft->context() : nullptr, prompt, calibration);
            const int64_t calibrated = ggml_time_us();
            json report = {
                {"schema_version", 2}, {"complete", true},
                {"mode", calibration.ordinary_only ? "fixed_prefix_synchronized_ordinary_round" : "fixed_prefix_synchronized_calibration"},
                {"history_tokens", history_tokens}, {"prefix_source", "repeated_prompt_tokens"},
                {"prefix_token_ids", prompt}, {"warmup_per_depth", calibration.warmup},
                {"repetitions_per_depth", calibration.repetitions}, {"depth_order", calibration.ordinary_only ? json{0} : json{0, 7, 1, 2, 3, 4, 5, 6}},
                {"target_model", params.model.path}, {"draft_model", params.speculative.draft.mparams.path},
                {"target_k", ggml_type_name(params.cache_type_k)}, {"target_v", ggml_type_name(params.cache_type_v)},
                {"draft_k", ggml_type_name(draft_params.cache_type_k)}, {"draft_v", ggml_type_name(draft_params.cache_type_v)},
                {"target_context", llama_n_ctx(target->context())}, {"draft_context", (draft ? llama_n_ctx(draft->context()) : 0)},
                {"target_batch", llama_n_batch(target->context())}, {"target_ubatch", llama_n_ubatch(target->context())},
                {"draft_batch", (draft ? llama_n_batch(draft->context()) : 0)}, {"draft_ubatch", (draft ? llama_n_ubatch(draft->context()) : 0)},
                {"target_threads", llama_n_threads(target->context())}, {"draft_threads", (draft ? llama_n_threads(draft->context()) : 0)},
                {"seed", params.sampling.seed}, {"target_temperature", params.sampling.temp},
                {"target_top_p", params.sampling.top_p}, {"target_top_k", params.sampling.top_k},
                {"draft_temperature", params.sampling.temp}, {"draft_top_p", 1.0},
                {"load_us", loaded - startup_start}, {"calibration_us", calibrated - loaded},
                {"validation", "finite_target_logits_and_normalized_draft_probabilities"},
                {"excluded_from_round", {"model_load", "prefix_prefill", "prefix_restore", "speculator_setup", "warmup", "sampler_reset", "validation"}},
                {"samples", json::array()}, {"cost_table", json::array()}
            };
            for (const auto & sample : samples) {
                report["samples"].push_back({
                    {"draft", sample.draft}, {"repetition", sample.repetition}, {"accepted", sample.accepted},
                    {"draft_rows", sample.draft_rows}, {"target_rows", sample.target_rows},
                    {"draft_us", sample.draft_us}, {"draft_cleanup_us", sample.draft_cleanup_us},
                    {"target_us", sample.target_us}, {"context_us", sample.context_us},
                    {"sampling_us", sample.sampling_us}, {"rollback_us", sample.rollback_us},
                    {"round_us", sample.round_us}, {"draft_tokens", sample.draft_tokens},
                    {"output_tokens", sample.output_tokens}
                });
            }
            LOG_INF("\n D | draft ms | target ms | context ms | sample ms | round ms\n");
            for (int depth = 0; depth <= (calibration.ordinary_only ? 0 : COMMON_SPECULATIVE_CALIBRATION_MAX_DRAFT); ++depth) {
                json row = {{"depth", depth}, {"draft_rows", depth}, {"target_rows", depth + 1},
                            {"samples", 0}, {"accepted_total", 0}, {"repeat_tokens_match", true}};
                std::vector<common_speculative_calibration_sample> group;
                for (const auto & sample : samples) {
                    if (sample.draft == depth) {
                        group.push_back(sample);
                        row["accepted_total"] = row["accepted_total"].get<int>() + sample.accepted;
                    }
                }
                if (group.size() != (size_t) calibration.repetitions) {
                    throw std::runtime_error("incomplete calibration table");
                }
                row["samples"] = group.size();
                for (const auto & sample : group) {
                    if (sample.draft_tokens != group.front().draft_tokens || sample.output_tokens != group.front().output_tokens) {
                        row["repeat_tokens_match"] = false;
                    }
                }
                const std::pair<const char *, int64_t common_speculative_calibration_sample::*> stages[] = {
                    {"draft", &common_speculative_calibration_sample::draft_us},
                    {"draft_cleanup", &common_speculative_calibration_sample::draft_cleanup_us},
                    {"target", &common_speculative_calibration_sample::target_us},
                    {"context", &common_speculative_calibration_sample::context_us},
                    {"sampling", &common_speculative_calibration_sample::sampling_us},
                    {"rollback", &common_speculative_calibration_sample::rollback_us},
                    {"round", &common_speculative_calibration_sample::round_us}
                };
                for (const auto & stage : stages) {
                    std::vector<int64_t> times;
                    for (const auto & sample : group) {
                        times.push_back(sample.*stage.second);
                    }
                    row[stage.first] = stage_stats(times);
                }
                LOG_INF("%2d | %8.3f | %9.3f | %10.3f | %9.3f | %8.3f\n", depth,
                        row["draft"]["mean_us"].get<double>() / 1000.0,
                        row["target"]["mean_us"].get<double>() / 1000.0,
                        row["context"]["mean_us"].get<double>() / 1000.0,
                        row["sampling"]["mean_us"].get<double>() / 1000.0,
                        row["round"]["mean_us"].get<double>() / 1000.0);
                report["cost_table"].push_back(std::move(row));
            }
            const std::string serialized = report.dump(2) + "\n";
            if (output.empty()) {
                std::fwrite(serialized.data(), 1, serialized.size(), stdout);
            } else {
                std::ofstream file(output);
                file << serialized;
                if (!file) {
                    throw std::runtime_error("failed to write calibration output");
                }
            }
        }
        llama_backend_free();
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("calibration: %s\n", e.what());
        if (backend_initialized) {
            llama_backend_free();
        }
        return 1;
    }
}
