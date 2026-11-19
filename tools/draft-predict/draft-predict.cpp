#include "speculative-adaptive.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

using json = nlohmann::ordered_json;

static int integer(const json & value) {
    if (!value.is_number_integer() ||
            (value.is_number_unsigned() && value.get<uint64_t>() > (uint64_t) std::numeric_limits<int>::max())) {
        throw std::invalid_argument("expected an integer in int range");
    }
    const auto number = value.get<int64_t>();
    if (number < std::numeric_limits<int>::min() || number > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("integer out of range");
    }
    return (int) number;
}

static double real(const json & value) {
    if (!value.is_number() || !std::isfinite(value.get<double>())) {
        throw std::invalid_argument("expected a finite number");
    }
    return value.get<double>();
}

static json read_json(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() > 64 * 1024 * 1024) {
        throw std::invalid_argument("cannot read JSON file or file exceeds 64 MiB: " + path);
    }
    file.seekg(0);
    json value;
    file >> value;
    file >> std::ws;
    if (!file.eof()) { throw std::invalid_argument("trailing data in JSON file: " + path); }
    return value;
}

static int load_costs(common_speculative_adaptive & model, const json & input, int maximum) {
    if (!input.is_object() || !input.at("complete").get<bool>() || !input.at("cost_table").is_array()) {
        throw std::invalid_argument("calibration must be a complete cost table");
    }
    const int version = integer(input.at("schema_version"));
    const int context = integer(input.at("history_tokens"));
    if ((version != 1 && version != 2) || context < 0 || context > 1048576 || input.at("cost_table").size() > 8) {
        throw std::invalid_argument("unsupported calibration schema, context or table size");
    }
    std::array<bool, 8> seen{};
    bool loaded = false;
    for (const auto & row : input.at("cost_table")) {
        const int draft = integer(row.at("depth"));
        if (draft < 0 || draft > 7 || seen[draft]) { throw std::invalid_argument("invalid or duplicate calibration depth"); }
        seen[draft] = true;
        if (draft > maximum) { continue; }
        if ((row.contains("draft_rows") && integer(row.at("draft_rows")) != draft) ||
                (row.contains("target_rows") && integer(row.at("target_rows")) != draft + 1)) {
            throw std::invalid_argument("calibration does not have actual D/D+1 rows");
        }
        model.set_cost(draft, real(row.at("round").at("mean_us")), real(row.at("round").at("sd_us")), integer(row.at("samples")));
        loaded = true;
    }
    if (!loaded) { throw std::invalid_argument("calibration contains no D0-D7 cost measurements"); }
    return context;
}

static json interval(const common_speculative_adaptive_interval & value) {
    return {{"mean", value.mean}, {"low", value.low}, {"high", value.high}};
}

static json report(
        const common_speculative_adaptive & model, const common_speculative_adaptive_params & params,
        int current, int context, uint64_t records, const std::string & mode) {
    const auto result = model.evaluate(current);
    const char * action = "hold";
    switch (result.action) {
        case COMMON_SPECULATIVE_ADAPTIVE_SWITCH: action = "switch"; break;
        case COMMON_SPECULATIVE_ADAPTIVE_PROBE: action = "probe"; break;
        case COMMON_SPECULATIVE_ADAPTIVE_NEED_COST: action = "need_cost"; break;
        case COMMON_SPECULATIVE_ADAPTIVE_HOLD: break;
    }
    json output = {
        {"schema_version", 3}, {"estimates_only", true},
        {"model", "logit(c[D,i]) = a + b*(i-1) + u[D] + v[D,i]"},
        {"uncertainty", "joint_regularized_logistic_laplace_and_lognormal_cost"},
        {"interval_probability", 1.0 - 2.0 * params.interval_tail},
        {"history_tokens", context}, {"calibration_mode", mode},
        {"round_records", records}, {"committed_tokens", model.committed_tokens()},
        {"gain_margin", params.gain_margin}, {"half_life_tokens", params.half_life_tokens},
        {"evidence_window_tokens", params.evidence_window_tokens}, {"probe_probability", params.probe_probability},
        {"reentry_probability", params.reentry_probability},
        {"refresh_cost", result.refresh_cost}, {"refresh_cost_tokens", params.refresh_cost_tokens},
        {"refresh_probability", params.refresh_probability}, {"refresh_expected_gain", params.refresh_expected_gain},
        {"current_draft", current}, {"action", action}, {"recommended_draft", result.draft},
        {"runtime_us", result.runtime_us}, {"probe_us", result.probe_us},
        {"round_overhead_us", result.round_overhead_us}, {"reentry_cost_us", result.reentry_cost_us},
        {"gain_horizon_tokens", result.gain_horizon_tokens}, {"predictions", json::array()}
    };
    for (const auto & prediction : result.predictions) {
        json row = {
            {"draft", prediction.draft}, {"direct_rounds", prediction.direct_rounds},
            {"recent_rounds", prediction.recent_rounds},
            {"live_cost_windows", prediction.live_cost_windows}, {"cost_age_tokens", prediction.cost_age_tokens},
            {"expected_positive_gain", prediction.expected_positive_gain},
            {"accepted", interval(prediction.accepted)}, {"positions", json::array()}
        };
        for (int i = 1; i <= prediction.draft; ++i) {
            const auto count = model.counts(prediction.draft, i);
            row["positions"].push_back({
                {"position", i}, {"conditional", interval(prediction.conditional[i - 1])},
                {"survival", interval(prediction.survival[i - 1])},
                {"success_weight", count.success}, {"failure_weight", count.failure}
            });
        }
        row["round_us"] = prediction.cost_available ? interval(prediction.round_us) : json(nullptr);
        row["tokens_per_second"] = prediction.cost_available ? interval(prediction.tokens_per_second) : json(nullptr);
        const bool comparison = prediction.cost_available && result.action != COMMON_SPECULATIVE_ADAPTIVE_NEED_COST;
        row["relative_gain"] = comparison ? interval(prediction.relative_gain) : json(nullptr);
        row["gain_threshold"] = prediction.gain_threshold;
        row["probability_gain_gt_threshold"] = comparison ? json(prediction.probability_gain) : json(nullptr);
        output["predictions"].push_back(std::move(row));
    }
    return output;
}

static void usage(const char * name) {
    std::fprintf(stderr,
            "Usage: %s --calibration costs.json [--rounds rounds.jsonl] [options]\n"
            "       %s --demo [options]\n"
            "  --current-draft N       comparison depth (default: last round, or 3)\n"
            "  --max-draft N           candidate maximum, 1-7 (default: 7)\n"
            "  --half-life-tokens N    observation half-life; 0 retains history (default: 512)\n"
            "  --draws N               posterior samples, 128-16384 (default: 1024)\n"
            "  --seed N                reproducible posterior RNG seed (default: 0)\n"
            "  --trace-every N         emit a JSON line every N records (default: final only)\n"
            "The demo uses synthetic observations. No model is loaded or executed.\n", name, name);
}

int main(int argc, char ** argv) {
    try {
        common_speculative_adaptive_params params;
        std::string calibration_path;
        std::string rounds_path;
        bool demo = false;
        bool current_set = false;
        int current = 3;
        int trace_every = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") { usage(argv[0]); return 0; }
            if (key == "--demo") { demo = true; continue; }
            if (++i >= argc) { throw std::invalid_argument("missing value for " + key); }
            const std::string value = argv[i];
            if (key == "--calibration") { calibration_path = value; }
            else if (key == "--rounds") { rounds_path = value; }
            else if (key == "--current-draft") { current = integer(json::parse(value)); current_set = true; }
            else if (key == "--max-draft") { params.max_draft = integer(json::parse(value)); }
            else if (key == "--half-life-tokens") { params.half_life_tokens = real(json::parse(value)); }
            else if (key == "--draws") { params.draws = integer(json::parse(value)); }
            else if (key == "--trace-every") { trace_every = integer(json::parse(value)); }
            else if (key == "--seed") {
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
                    throw std::invalid_argument("seed must be an unsigned integer");
                }
                params.seed = std::stoull(value);
            } else { throw std::invalid_argument("unknown option: " + key); }
        }
        if (trace_every < 0 || (demo && (!calibration_path.empty() || !rounds_path.empty())) || (!demo && calibration_path.empty())) {
            throw std::invalid_argument("choose --demo or --calibration; trace interval must be nonnegative");
        }
        if (!current_set) { current = std::min(current, params.max_draft); }
        common_speculative_adaptive model(params);
        if (current < 0 || current > params.max_draft) { throw std::invalid_argument("current draft out of range"); }
        int context = 512;
        uint64_t records = 0;
        std::string mode = "synthetic_demo";
        auto observe = [&](const common_speculative_adaptive_observation & observation) {
            model.observe(observation);
            ++records;
            if (!current_set) { current = observation.draft; }
            if (trace_every > 0 && records % trace_every == 0) {
                std::puts(report(model, params, current, context, records, mode).dump().c_str());
            }
        };
        if (demo) {
            model.set_cost(0, 20000, 300, 5);
            const std::array<double, 7> cost = {18000, 24000, 28000, 33000, 37000, 40000, 37000};
            for (int d = 1; d <= params.max_draft; ++d) { model.set_cost(d, cost[d - 1], 0.015 * cost[d - 1], 5); }
            std::mt19937_64 rng(params.seed);
            std::uniform_real_distribution<double> uniform;
            std::normal_distribution<double> timing(1.0, 0.015);
            for (int t = 0; t < 840; ++t) {
                const int draft = t % params.max_draft + 1;
                const std::array<double, 7> probability = t < 420 ?
                        std::array<double, 7>{0.607, 0.6402, 0.5955, 0.55, 0.50, 0.45, 0.40} :
                        std::array<double, 7>{0.8496, 0.8242, 0.8141, 0.79, 0.77, 0.74, 0.70};
                int accepted = 0;
                for (int i = 0; i < draft; ++i) {
                    const double eta = std::log(probability[i] / (1.0 - probability[i])) + 0.06 * (draft - 3);
                    if (uniform(rng) >= 1.0 / (1.0 + std::exp(-eta))) { break; }
                    ++accepted;
                }
                common_speculative_adaptive_observation observation;
                observation.draft = draft;
                observation.accepted = accepted;
                observation.round_us = cost[draft - 1] * std::max(0.5, timing(rng));
                observe(observation);
            }
        } else {
            const auto calibration = read_json(calibration_path);
            context = load_costs(model, calibration, params.max_draft);
            mode = calibration.value("mode", std::string("unspecified_calibration"));
            if (!rounds_path.empty()) {
                std::ifstream file(rounds_path, std::ios::binary | std::ios::ate);
                if (!file || file.tellg() < 0 || file.tellg() > 64 * 1024 * 1024) {
                    throw std::invalid_argument("cannot read rounds file or file exceeds 64 MiB");
                }
                file.seekg(0);
                std::string line;
                while (std::getline(file, line)) {
                    if (line.size() > 1024 * 1024) { throw std::invalid_argument("round record exceeds 1 MiB"); }
                    if (line.find_first_not_of(" \t\r") == std::string::npos) { continue; }
                    const auto row = json::parse(line);
                    if (!row.is_object() || (row.contains("context_tokens") && integer(row.at("context_tokens")) != context)) {
                        throw std::invalid_argument("round context does not match the calibration bucket");
                    }
                    common_speculative_adaptive_observation observation;
                    observation.draft = integer(row.at("draft"));
                    observation.accepted = integer(row.at("accepted"));
                    if ((row.contains("draft_rows") && integer(row.at("draft_rows")) != observation.draft) ||
                            (row.contains("target_rows") && integer(row.at("target_rows")) != observation.draft + 1)) {
                        throw std::invalid_argument("round does not have actual D/D+1 rows");
                    }
                    observation.committed = row.contains("committed") ? integer(row.at("committed")) : 0;
                    observation.censored = row.value("censored", false);
                    observation.probe = row.value("probe", false);
                    observation.round_us = row.contains("round_us") ? real(row.at("round_us")) : 0.0;
                    observe(observation);
                }
                if (!file.eof()) { throw std::runtime_error("failed to read rounds file"); }
            }
        }
        const std::string output = report(model, params, current, context, records, mode).dump(trace_every > 0 ? -1 : 2);
        std::puts(output.c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "draft-predict: %s\n", error.what());
        return 1;
    }
}
