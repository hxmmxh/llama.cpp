#include "speculative-adaptive.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace {

bool finite_range(double value, double low, double high) {
    return std::isfinite(value) && value >= low && value <= high;
}

double sigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exp_value = std::exp(value);
    return exp_value / (1.0 + exp_value);
}

double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

std::vector<double> cholesky(const std::vector<double> & matrix, int size) {
    std::vector<double> lower(size * size, 0.0);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = matrix[i * size + j];
            for (int k = 0; k < j; ++k) {
                value -= lower[i * size + k] * lower[j * size + k];
            }
            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value)) {
                    throw std::runtime_error("adaptive posterior precision is not positive definite");
                }
                lower[i * size + j] = std::sqrt(value);
            } else {
                lower[i * size + j] = value / lower[j * size + j];
            }
        }
    }
    return lower;
}

std::vector<double> solve_upper(const std::vector<double> & lower, std::vector<double> value) {
    const int size = (int) value.size();
    for (int i = size - 1; i >= 0; --i) {
        for (int j = i + 1; j < size; ++j) {
            value[i] -= lower[j * size + i] * value[j];
        }
        value[i] /= lower[i * size + i];
    }
    return value;
}

std::vector<double> solve_precision(const std::vector<double> & lower, std::vector<double> value) {
    const int size = (int) value.size();
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < i; ++j) {
            value[i] -= lower[i * size + j] * value[j];
        }
        value[i] /= lower[i * size + i];
    }
    return solve_upper(lower, value);
}

common_speculative_adaptive_interval summarize(std::vector<double> values, double tail) {
    common_speculative_adaptive_interval result;
    for (double value : values) {
        result.mean += value;
    }
    result.mean /= values.size();
    std::sort(values.begin(), values.end());
    auto quantile = [&](double probability) {
        const double index = probability * (values.size() - 1);
        const size_t first = (size_t) index;
        const size_t last = std::min(first + 1, values.size() - 1);
        return values[first] + (index - first) * (values[last] - values[first]);
    };
    result.low = quantile(tail);
    result.high = quantile(1.0 - tail);
    return result;
}

} // namespace

common_speculative_adaptive::common_speculative_adaptive(const common_speculative_adaptive_params & params) : params(params) {
    if (params.max_draft < 1 || params.max_draft > 7 ||
            !finite_range(params.half_life_tokens, 0.0, 1e9) ||
            !finite_range(params.prior_acceptance, 0.001, 0.999) ||
            !finite_range(params.intercept_sd, 0.001, 10.0) ||
            !finite_range(params.slope_sd, 0.001, 10.0) ||
            !finite_range(params.depth_sd, 0.001, 10.0) ||
            !finite_range(params.position_sd, 0.001, 10.0) ||
            !finite_range(params.cost_relative_sd, 0.001, 1.0) ||
            !finite_range(params.gain_margin, 0.0, 1.0) ||
            !finite_range(params.interval_tail, 0.001, 0.25) ||
            params.min_direct_rounds < 1 || params.min_direct_rounds > 1000000 ||
            params.evidence_window_tokens < (uint64_t) params.min_direct_rounds || params.evidence_window_tokens > 1000000 ||
            !finite_range(params.probe_probability, 0.5, 1.0) ||
            !finite_range(params.reentry_probability, params.probe_probability, 1.0) ||
            params.refresh_cost_tokens < 1 || params.refresh_cost_tokens > 1000000 ||
            !finite_range(params.refresh_probability, 0.0, 1.0) ||
            !finite_range(params.refresh_expected_gain, 0.0, 1.0) ||
            !finite_range(params.probe_score_tolerance, 0.0, 1.0) ||
            params.draws < 128 || params.draws > 16384) {
        throw std::invalid_argument("invalid adaptive draft parameters");
    }
    for (double floor : params.cost_relative_sd_by_draft) {
        if (floor != 0.0 && !finite_range(floor, 0.001, 1.0)) {
            throw std::invalid_argument("invalid per-depth cost uncertainty");
        }
    }
    observations.resize(params.max_draft * (params.max_draft + 1) / 2);
    rounds.resize(params.max_draft + 1, 0.0);
    recent_rounds.resize(params.max_draft + 1);
    costs.resize(params.max_draft + 1);
}

int common_speculative_adaptive::cell(int draft, int position) const {
    if (draft < 1 || draft > params.max_draft || position < 1 || position > draft) {
        throw std::invalid_argument("adaptive draft/position out of range");
    }
    return draft * (draft - 1) / 2 + position - 1;
}

void common_speculative_adaptive::set_cost(int draft, double mean_us, double sd_us, int samples) {
    if (draft < 0 || draft > params.max_draft) { throw std::invalid_argument("adaptive draft out of range"); }
    if (!finite_range(mean_us, 1e-6, 1e12) || !finite_range(sd_us, 0.0, 1e12) || samples < 1 || samples > 1000000000) {
        throw std::invalid_argument("invalid adaptive cost measurement");
    }
    costs[draft] = { 0, tokens, (double) samples, mean_us, sd_us * sd_us * (samples - 1) / samples };
}

void common_speculative_adaptive::update_cost(int draft, double sample) {
    auto & cost = costs[draft];
    // A live window supersedes the startup-prefix prior, which may have a different context cost.
    cost.last_live_token = tokens;
    if (cost.live_windows++ == 0) {
        cost.weight = 1.0;
        cost.mean = sample;
        cost.variance = 0.0;
        return;
    }
    const double delta = sample - cost.mean;
    const double weight = cost.weight + 1.0;
    cost.mean += delta / weight;
    cost.variance = std::max(0.0, (cost.weight * cost.variance + delta * (sample - cost.mean)) / weight);
    cost.weight = weight;
}

void common_speculative_adaptive::observe(const common_speculative_adaptive_observation & observation) {
    if (observation.draft < 0 || observation.draft > params.max_draft) { throw std::invalid_argument("adaptive draft out of range"); }
    const int draft = observation.draft;
    const int accepted = observation.accepted;
    if (accepted < 0 || accepted > draft || observation.committed < 0 || observation.committed > draft + 1 ||
            (observation.censored && observation.committed == 0) ||
            !finite_range(observation.round_us, 0.0, 1e12) || (observation.probe && observation.round_us == 0.0)) {
        throw std::invalid_argument("invalid adaptive round observation");
    }
    const int committed = observation.committed > 0 ? observation.committed : accepted + 1;
    if ((!observation.censored && committed != accepted + 1) || committed > accepted + 1 ||
            tokens > std::numeric_limits<uint64_t>::max() - (uint64_t) committed) {
        throw std::invalid_argument("invalid adaptive committed token count");
    }

    const double discount = params.half_life_tokens > 0.0 ? std::exp2(-committed / params.half_life_tokens) : 1.0;
    for (auto & count : observations) {
        count.success *= discount;
        count.failure *= discount;
    }
    for (double & count : rounds) {
        count *= discount;
    }
    for (auto & cost : costs) {
        cost.weight *= discount;
    }
    tokens += committed;
    for (auto & recent : recent_rounds) {
        while (!recent.empty() && tokens - recent.front().tokens >= params.evidence_window_tokens) {
            recent.pop_front();
        }
    }

    for (int position = 1; position <= accepted; ++position) {
        observations[cell(draft, position)].success += 1.0;
    }
    if (!observation.censored && accepted < draft) {
        observations[cell(draft, accepted + 1)].failure += 1.0;
    }
    if (!observation.censored) {
        rounds[draft] += 1.0;
        recent_rounds[draft].push_back({tokens, accepted});
        if (observation.round_us > 0.0) {
            update_cost(draft, observation.round_us);
        }
    }
    elapsed_us += observation.round_us;
    if (observation.probe) {
        probe_us += observation.round_us;
    }
}

void common_speculative_adaptive::refresh_acceptance(int draft, int retained_rounds) {
    if (draft < 1 || draft > params.max_draft || retained_rounds < 1 || retained_rounds > 1000000) {
        throw std::invalid_argument("invalid adaptive acceptance refresh");
    }
    auto & recent = recent_rounds[draft];
    while (recent.size() > (size_t) retained_rounds) { recent.pop_front(); }
    rounds[draft] = 0.0;
    for (int position = 1; position <= draft; ++position) { observations[cell(draft, position)] = {}; }
    for (const auto & round : recent) {
        const double weight = params.half_life_tokens > 0.0 ?
                std::exp2(-(double) (tokens - round.tokens) / params.half_life_tokens) : 1.0;
        rounds[draft] += weight;
        for (int position = 1; position <= round.accepted; ++position) {
            observations[cell(draft, position)].success += weight;
        }
        if (round.accepted < draft) { observations[cell(draft, round.accepted + 1)].failure += weight; }
    }
}

void common_speculative_adaptive::observe_cost_window(int draft, double total_us, int round_count, bool probe, bool complete, double transition_us) {
    if (draft < 0 || draft > params.max_draft) { throw std::invalid_argument("adaptive draft out of range"); }
    if (!finite_range(total_us, 1e-6, 1e12) || !finite_range(transition_us, 0.0, total_us) || round_count < 1 || round_count > 1000000) {
        throw std::invalid_argument("invalid adaptive cost window");
    }
    if (complete && total_us > transition_us) {
        update_cost(draft, (total_us - transition_us) / round_count);
    }
    elapsed_us += total_us;
    if (probe) { probe_us += total_us; }
}

common_speculative_adaptive_counts common_speculative_adaptive::counts(int draft, int position) const {
    return observations[cell(draft, position)];
}

uint64_t common_speculative_adaptive::committed_tokens() const {
    return tokens;
}

double common_speculative_adaptive::runtime_time_us() const {
    return elapsed_us;
}

double common_speculative_adaptive::probe_time_us() const {
    return probe_us;
}

void common_speculative_adaptive::fit(std::vector<double> & mean, std::vector<double> & precision_cholesky) const {
    const int size = 2 + params.max_draft + (int) observations.size();
    std::vector<double> prior_mean(size, 0.0);
    prior_mean[0] = std::log(params.prior_acceptance / (1.0 - params.prior_acceptance));
    std::vector<double> prior_precision(size, 1.0 / (params.position_sd * params.position_sd));
    prior_precision[0] = 1.0 / (params.intercept_sd * params.intercept_sd);
    prior_precision[1] = 1.0 / (params.slope_sd * params.slope_sd);
    for (int draft = 1; draft <= params.max_draft; ++draft) {
        prior_precision[1 + draft] = 1.0 / (params.depth_sd * params.depth_sd);
    }

    auto objective = [&](const std::vector<double> & theta, std::vector<double> * gradient, std::vector<double> * precision) {
        double loss = 0.0;
        if (gradient) { gradient->assign(size, 0.0); }
        if (precision) { precision->assign(size * size, 0.0); }
        for (int i = 0; i < size; ++i) {
            const double delta = theta[i] - prior_mean[i];
            loss += 0.5 * prior_precision[i] * delta * delta;
            if (gradient) { (*gradient)[i] = prior_precision[i] * delta; }
            if (precision) { (*precision)[i * size + i] = prior_precision[i]; }
        }
        for (int draft = 1; draft <= params.max_draft; ++draft) {
            for (int position = 1; position <= draft; ++position) {
                const auto & count = observations[cell(draft, position)];
                const double trials = count.success + count.failure;
                const std::array<int, 4> index = { 0, 1, 1 + draft, 2 + params.max_draft + cell(draft, position) };
                const std::array<double, 4> feature = { 1.0, (double) position - 1.0, 1.0, 1.0 };
                double eta = 0.0;
                for (int i = 0; i < 4; ++i) { eta += theta[index[i]] * feature[i]; }
                loss += trials * softplus(eta) - count.success * eta;
                const double probability = sigmoid(eta);
                for (int i = 0; i < 4; ++i) {
                    if (gradient) { (*gradient)[index[i]] += feature[i] * (trials * probability - count.success); }
                    if (precision) {
                        for (int j = 0; j < 4; ++j) {
                            (*precision)[index[i] * size + index[j]] += trials * probability * (1.0 - probability) * feature[i] * feature[j];
                        }
                    }
                }
            }
        }
        return loss;
    };

    mean = prior_mean;
    std::vector<double> gradient;
    std::vector<double> precision;
    for (int iteration = 0; iteration < 30; ++iteration) {
        const double loss = objective(mean, &gradient, &precision);
        const auto step = solve_precision(cholesky(precision, size), gradient);
        double scale = 1.0;
        std::vector<double> next(size);
        bool improved = false;
        for (int backtrack = 0; backtrack < 20; ++backtrack) {
            for (int i = 0; i < size; ++i) { next[i] = mean[i] - scale * step[i]; }
            if (objective(next, nullptr, nullptr) <= loss) {
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if (!improved) { break; }
        double change = 0.0;
        for (int i = 0; i < size; ++i) { change = std::max(change, std::abs(next[i] - mean[i])); }
        mean = std::move(next);
        if (change < 1e-7) { break; }
    }
    objective(mean, nullptr, &precision);
    precision_cholesky = cholesky(precision, size);
}

common_speculative_adaptive_result common_speculative_adaptive::evaluate(int current_draft, const std::vector<int> & excluded_drafts, double round_overhead_us, double reentry_cost_us, int gain_horizon_tokens) const {
    if (current_draft < 0 || current_draft > params.max_draft) { throw std::invalid_argument("adaptive draft out of range"); }
    if (!finite_range(round_overhead_us, 0.0, 1e12)) {
        throw std::invalid_argument("invalid adaptive round overhead");
    }
    std::vector<bool> excluded(params.max_draft + 1, false);
    for (int draft : excluded_drafts) {
        if (draft < 0 || draft > params.max_draft) { throw std::invalid_argument("adaptive draft out of range"); }
        excluded[draft] = true;
    }
    common_speculative_adaptive_result result;
    result.draft = current_draft;
    result.runtime_us = runtime_time_us();
    result.probe_us = probe_time_us();
    if (!finite_range(reentry_cost_us, 0.0, 1e18) || gain_horizon_tokens < 1) { throw std::invalid_argument("invalid adaptive reentry cost or horizon"); }
    result.round_overhead_us = round_overhead_us;
    result.reentry_cost_us = reentry_cost_us;
    result.gain_horizon_tokens = gain_horizon_tokens;
    std::vector<double> mean;
    std::vector<double> lower;
    fit(mean, lower);
    const int size = (int) mean.size();
    const int depths = params.max_draft;
    std::vector<std::vector<double>> accepted(depths + 1), time(depths + 1), rate(depths + 1);
    std::vector<std::vector<std::vector<double>>> conditional(depths + 1), survival(depths + 1);
    for (int draft = 0; draft <= depths; ++draft) {
        conditional[draft].resize(draft);
        survival[draft].resize(draft);
    }
    std::mt19937_64 rng(params.seed);
    std::normal_distribution<double> normal;
    for (int draw = 0; draw < params.draws; ++draw) {
        std::vector<double> noise(size);
        for (double & value : noise) { value = normal(rng); }
        // H = L L^T, so L^-T z has covariance H^-1.
        noise = solve_upper(lower, std::move(noise));
        for (int i = 0; i < size; ++i) { noise[i] += mean[i]; }
        for (int draft = 0; draft <= depths; ++draft) {
            const int d = draft;
            double prefix = 1.0;
            double total = 0.0;
            for (int position = 1; position <= draft; ++position) {
                const double probability = sigmoid(noise[0] + noise[1] * (position - 1) +
                        noise[1 + draft] + noise[2 + depths + cell(draft, position)]);
                prefix *= probability;
                total += prefix;
                conditional[d][position - 1].push_back(probability);
                survival[d][position - 1].push_back(prefix);
            }
            accepted[d].push_back(total);
            const auto & cost = costs[d];
            if (cost.mean > 0.0) {
                const double relative_sd = params.cost_relative_sd_by_draft[d] > 0.0 ?
                        params.cost_relative_sd_by_draft[d] : params.cost_relative_sd;
                const double floor_variance = relative_sd * relative_sd;
                // Discounted evidence below one window also weakens a zero-scatter prior.
                const double aged_floor = floor_variance / std::clamp(cost.weight, floor_variance, 1.0);
                const double relative_variance = cost.variance / std::max(1.0, cost.weight - 1.0) / (cost.mean * cost.mean) +
                        aged_floor;
                const double log_variance = std::log1p(relative_variance);
                const double cost_draw = std::exp(std::log(cost.mean) - 0.5 * log_variance + std::sqrt(log_variance) * normal(rng));
                time[d].push_back(cost_draw);
                const double restart_per_token = current_draft == 0 && draft > 0 ? reentry_cost_us / gain_horizon_tokens : 0.0;
                rate[d].push_back(1e6 / ((cost_draw + round_overhead_us) / (1.0 + total) + restart_per_token));
            }
        }
    }
    for (int draft = 0; draft <= depths; ++draft) {
        const int d = draft;
        common_speculative_adaptive_prediction prediction;
        prediction.draft = draft;
        prediction.gain_threshold = draft == 0 && current_draft > 0 ? -params.gain_margin / (1.0 + params.gain_margin) : params.gain_margin;
        prediction.direct_rounds = rounds[d];
        prediction.recent_rounds = (int) recent_rounds[d].size();
        prediction.cost_available = !time[d].empty();
        prediction.live_cost_windows = costs[d].live_windows;
        prediction.cost_age_tokens = tokens - costs[d].last_live_token;
        prediction.accepted = summarize(accepted[d], params.interval_tail);
        for (int position = 0; position < draft; ++position) {
            prediction.conditional.push_back(summarize(conditional[d][position], params.interval_tail));
            prediction.survival.push_back(summarize(survival[d][position], params.interval_tail));
        }
        if (prediction.cost_available) {
            prediction.round_us = summarize(time[d], params.interval_tail);
            prediction.tokens_per_second = summarize(rate[d], params.interval_tail);
            if (!rate[current_draft].empty()) {
                std::vector<double> gain(params.draws);
                for (int draw = 0; draw < params.draws; ++draw) {
                    gain[draw] = rate[d][draw] / rate[current_draft][draw] - 1.0;
                    prediction.expected_positive_gain += std::max(0.0, gain[draw] - params.gain_margin) / params.draws;
                    if (gain[draw] > prediction.gain_threshold) { prediction.probability_gain += 1.0; }
                }
                prediction.probability_gain /= params.draws;
                prediction.relative_gain = summarize(std::move(gain), params.interval_tail);
            }
        }
        result.predictions.push_back(std::move(prediction));
    }
    if (!result.predictions[current_draft].cost_available) {
        result.action = COMMON_SPECULATIVE_ADAPTIVE_NEED_COST;
        return result;
    }
    double best_switch = params.gain_margin;
    for (const auto & prediction : result.predictions) {
        if (prediction.draft > 0 && prediction.draft != current_draft && !excluded[prediction.draft] && prediction.cost_available &&
                prediction.recent_rounds >= params.min_direct_rounds && prediction.relative_gain.low > best_switch) {
            result.action = COMMON_SPECULATIVE_ADAPTIVE_SWITCH;
            result.draft = prediction.draft;
            best_switch = prediction.relative_gain.low;
        }
    }
    if (result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH) { return result; }
    const auto & ordinary = result.predictions[0];
    // Prefer ordinary decoding when speculation cannot retain the required margin.
    if (current_draft > 0 && !excluded[0] && ordinary.cost_available &&
            ordinary.recent_rounds >= params.min_direct_rounds && ordinary.relative_gain.mean > ordinary.gain_threshold &&
            ordinary.probability_gain >= params.probe_probability) {
        result.action = COMMON_SPECULATIVE_ADAPTIVE_SWITCH;
        result.draft = 0;
        return result;
    }

    double best_probe = params.gain_margin;
    for (const auto & prediction : result.predictions) {
        if (prediction.draft == 0 || prediction.draft == current_draft || excluded[prediction.draft] || !prediction.cost_available ||
                prediction.probability_gain < (current_draft == 0 ? params.reentry_probability : params.probe_probability)) {
            continue;
        }
        const double score = prediction.relative_gain.mean;
        if (score > best_probe) {
            result.action = COMMON_SPECULATIVE_ADAPTIVE_PROBE;
            result.draft = prediction.draft;
            best_probe = score;
        }
    }
    if (result.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE) {
        // Near-equal upside does not justify a longer, more expensive probe.
        for (const auto & prediction : result.predictions) {
            if (prediction.draft > 0 && prediction.draft != current_draft && !excluded[prediction.draft] &&
                    prediction.cost_available && prediction.relative_gain.mean > params.gain_margin &&
                    prediction.relative_gain.mean >= best_probe - params.probe_score_tolerance &&
                    prediction.probability_gain >= (current_draft == 0 ? params.reentry_probability : params.probe_probability) &&
                    prediction.round_us.mean < result.predictions[result.draft].round_us.mean) {
                result.draft = prediction.draft;
            }
        }
    }
    if (result.action == COMMON_SPECULATIVE_ADAPTIVE_HOLD && current_draft > 0 && !excluded[0] &&
            ordinary.cost_available && ordinary.relative_gain.mean > ordinary.gain_threshold &&
            ordinary.probability_gain >= params.probe_probability) {
        result.action = COMMON_SPECULATIVE_ADAPTIVE_PROBE;
        result.draft = 0;
    }
    // Exploration is tied to unresolved, meaningful upside, never a time quota.
    if (result.action == COMMON_SPECULATIVE_ADAPTIVE_HOLD && current_draft > 0) {
        double best_upside = params.refresh_expected_gain;
        for (const auto & prediction : result.predictions) {
            if (prediction.draft == 0 || prediction.draft == current_draft || excluded[prediction.draft] ||
                    !prediction.cost_available ||
                    (prediction.live_cost_windows > 0 && prediction.cost_age_tokens < params.refresh_cost_tokens) ||
                    prediction.relative_gain.high <= params.gain_margin ||
                    prediction.probability_gain < params.refresh_probability) { continue; }
            if (prediction.expected_positive_gain > best_upside) {
                result.action = COMMON_SPECULATIVE_ADAPTIVE_PROBE;
                result.draft = prediction.draft;
                result.refresh_cost = true;
                best_upside = prediction.expected_positive_gain;
            }
        }
        if (result.refresh_cost) {
            for (const auto & prediction : result.predictions) {
                if (prediction.draft > 0 && prediction.draft != current_draft && !excluded[prediction.draft] &&
                        prediction.cost_available &&
                        (prediction.live_cost_windows == 0 || prediction.cost_age_tokens >= params.refresh_cost_tokens) &&
                        prediction.relative_gain.high > params.gain_margin && prediction.probability_gain >= params.refresh_probability &&
                        prediction.expected_positive_gain > params.refresh_expected_gain &&
                        prediction.expected_positive_gain >= best_upside - params.probe_score_tolerance &&
                        prediction.round_us.mean < result.predictions[result.draft].round_us.mean) {
                    result.draft = prediction.draft;
                }
            }
        }
    }
    return result;
}
