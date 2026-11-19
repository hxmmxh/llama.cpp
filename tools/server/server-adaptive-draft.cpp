#include "server-adaptive-draft.h"

#include "log.h"
#include "speculative-calibration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

using json = common_json;

void server_adaptive_draft::validate(const common_params & params) {
    const int maximum = params.speculative.draft.n_max;
    if (params.n_parallel != 1 || maximum < 1 || maximum > 7 || params.speculative.has_synth() ||
            std::count(params.speculative.types.begin(), params.speculative.types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK) != 1 ||
            std::any_of(params.speculative.types.begin(), params.speculative.types.end(), [](common_speculative_type type) {
                return type != COMMON_SPECULATIVE_TYPE_NONE && type != COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
            }) ||
            !params.speculative.draft.probabilistic || params.speculative.draft.n_min != 0 ||
            params.speculative.draft.p_min != 0.0f || !std::isfinite(params.sampling.temp) || params.sampling.temp <= 0.0f) {
        throw std::invalid_argument("adaptive depth requires single-sequence probabilistic DSpark, D0-D7, n-min=0, p-min=0 and no synthetic acceptance");
    }
}

server_adaptive_draft_costs server_adaptive_draft::calibrate(
        const common_params & params, llama_context * target, llama_context * draft) {
    validate(params);
    if (!target || !draft) { throw std::invalid_argument("adaptive calibration requires target and draft contexts"); }
    const int maximum = params.speculative.draft.n_max;
    int64_t capacity = std::min(llama_n_ctx_seq(target), llama_n_ctx_seq(draft));
    capacity = std::min<int64_t>(capacity, llama_model_n_ctx_train(llama_get_model(target)));
    if (params.kv_unified_per_slot > 0) { capacity = std::min<int64_t>(capacity, params.kv_unified_per_slot); }
    const int history = (int) std::min<int64_t>(512, capacity - maximum - 1);
    if (history < 1) { throw std::invalid_argument("adaptive calibration context is too small"); }
    const auto seed_tokens = common_tokenize(target, "Explain how a computer stores information and processes instructions.\n", false, true);
    if (seed_tokens.empty()) { throw std::runtime_error("adaptive calibration prompt has no tokens"); }
    llama_tokens prompt;
    prompt.reserve(history + 1);
    const auto bos = llama_vocab_bos(llama_model_get_vocab(llama_get_model(target)));
    if (bos != LLAMA_TOKEN_NULL) { prompt.push_back(bos); }
    for (size_t i = 0; prompt.size() < (size_t) history + 1; ++i) {
        prompt.push_back(seed_tokens[i % seed_tokens.size()]);
    }
    auto measured_params = params;
    measured_params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK};
    // Repeated seeds stabilize timing only; calibration acceptance is never learned.
    if (measured_params.sampling.seed == LLAMA_DEFAULT_SEED) { measured_params.sampling.seed = 0; }
    common_speculative_calibration_params options;
    options.include_ordinary = true;
    options.runtime_pipeline = true;
    options.max_draft = maximum;
    options.repetitions = 7;
    LOG_INF("adaptive calibration: D0-D%d, history=%d, warmup=%d, repetitions=%d, runtime_pipeline=1\n",
            maximum, history, options.warmup, options.repetitions);
    const auto samples = common_speculative_calibrate(measured_params, target, draft, prompt, options);
    server_adaptive_draft_costs result;
    result.history_tokens = history;
    result.temperature = params.sampling.temp;
    result.top_p = params.sampling.top_p;
    result.top_k = params.sampling.top_k;
    result.min_p = params.sampling.min_p;
    for (int depth = 0; depth <= maximum; ++depth) {
        std::vector<double> timings;
        for (const auto & sample : samples) {
            if (sample.draft != depth) { continue; }
            timings.push_back(sample.round_us);
        }
        const auto cost = summarize_cost(depth, std::move(timings));
        if (cost.samples != options.repetitions) { throw std::runtime_error("adaptive calibration did not measure every depth"); }
        result.rows.push_back(cost);
        LOG_INF("adaptive calibration: D%d round=%.3f ms sd=%.3f ms n=%d runtime_pipeline=1 statistic=median_mad\n",
                depth, cost.mean_us / 1000.0, cost.sd_us / 1000.0, cost.samples);
    }
    return result;
}

server_adaptive_draft_cost server_adaptive_draft::summarize_cost(int draft, std::vector<double> samples) {
    if (draft < 0 || draft > 7 || samples.empty() || samples.size() > 1000000 ||
            std::any_of(samples.begin(), samples.end(), [](double sample) {
                return !std::isfinite(sample) || sample <= 0.0 || sample > 1e12;
            })) { throw std::invalid_argument("invalid adaptive calibration samples"); }
    const auto median = [](std::vector<double> & values) {
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        return values.size() % 2 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
    };
    const double center = median(samples);
    for (double & sample : samples) { sample = std::abs(sample - center); }
    return {draft, (int) samples.size(), center, 1.4826 * median(samples)};
}

server_adaptive_draft::server_adaptive_draft(const server_adaptive_draft_costs & data, const common_params & params) {
    validate(params);
    maximum = params.speculative.draft.n_max;
    history_tokens = data.history_tokens;
    if (history_tokens < 1 || history_tokens > 1048576 || data.rows.size() != (size_t) maximum + 1) {
        throw std::invalid_argument("adaptive depth requires fresh costs for every configured depth");
    }
    for (const auto & entry : std::vector<std::pair<double, double>>{
            {data.temperature, params.sampling.temp}, {data.top_p, params.sampling.top_p},
            {data.top_k, params.sampling.top_k}, {data.min_p, params.sampling.min_p}}) {
        if (!std::isfinite(entry.first) || !std::isfinite(entry.second) || std::abs(entry.first - entry.second) > 1e-6) {
            throw std::invalid_argument("adaptive sampling must match the current load's calibration");
        }
    }
    std::array<bool, 8> seen{};
    std::array<double, 8> mean{};
    std::array<double, 8> relative_sd{};
    for (const auto & cost : data.rows) {
        const int depth = cost.draft;
        if (depth < 0 || depth > maximum || seen[depth]) { throw std::invalid_argument("invalid or duplicate adaptive cost depth"); }
        seen[depth] = true;
        if (cost.samples < 1 || !std::isfinite(cost.mean_us) || cost.mean_us <= 0.0 || cost.mean_us > 1e12 ||
                !std::isfinite(cost.sd_us) || cost.sd_us < 0.0 || cost.sd_us > 1e12) {
            throw std::invalid_argument("invalid adaptive cycle cost");
        }
        mean[depth] = cost.mean_us;
        relative_sd[depth] = std::max(0.03, std::min(1.0, cost.sd_us / cost.mean_us));
    }
    common_speculative_adaptive_params options;
    options.max_draft = maximum;
    options.draws = 512;
    options.cost_relative_sd_by_draft = relative_sd;
    options.seed = params.sampling.seed;
    model = std::make_unique<common_speculative_adaptive>(options);
    for (int depth = 0; depth <= maximum; ++depth) {
        if (!seen[depth]) { throw std::invalid_argument("adaptive cost table must cover every configured depth"); }
        // Startup repetitions are one weak timing prior; live drained windows replace it.
        model->set_cost(depth, mean[depth], 0.0, 1);
    }
    ordinary_round_us = mean[0];
    incumbent = active = std::min(3, maximum);
}

uint64_t server_adaptive_draft::cooldown_remaining(int draft) const {
    const auto & history = probes[draft];
    const uint64_t age = model->committed_tokens() - history.tokens;
    return history.cooldown > age ? history.cooldown - age : 0;
}

void server_adaptive_draft::cool_down(int draft) {
    auto & history = probes[draft];
    history.tokens = model->committed_tokens();
    history.cooldown = std::min<uint64_t>(1024, std::max<uint64_t>(128, history.cooldown * 2));
}

bool server_adaptive_draft::ordinary_supported() const {
    // A posterior promotion must not contradict fresh, directly observed throughput.
    const auto & ordinary = live[0];
    const auto & current = live[incumbent];
    const auto now = model->committed_tokens();
    return ordinary.rate > 0.0 && current.rate > 0.0 &&
            now - ordinary.tokens < 512 && now - current.tokens < 512 &&
            ordinary.rate * 1.03 >= current.rate;
}

double server_adaptive_draft::measured_ordinary_saving() const {
    const auto & older = incumbent_windows[0];
    const auto & newer = incumbent_windows[1];
    const double rate = live[0].rate;
    if (rate <= 0.0 || older.rate <= 0.0 || newer.rate <= 0.0 ||
            model->committed_tokens() - older.tokens >= 256 ||
            older.rate > rate * 1.03 || newer.rate > rate * 1.03) {
        return 0.0;
    }
    const double positive_us = 1e6 * (older.committed / older.rate + newer.committed / newer.rate) /
            (older.committed + newer.committed);
    const double ordinary_us = 1e6 / rate;
    return positive_us >= ordinary_measured_gain * ordinary_us ? positive_us - ordinary_us : 0.0;
}

bool server_adaptive_draft::positive_supported(int draft) const {
    const auto & current = live[incumbent];
    if (probe_tokens == 0 || probe_elapsed_us <= 0.0 || current.rate <= 0.0 ||
            model->committed_tokens() - current.tokens >= 512) { return false; }
    const double replay_per_token = incumbent == 0 ? probe_reentry_us / gain_horizon_tokens : 0.0;
    const double episode_rate = 1e6 / (probe_elapsed_us / probe_tokens + replay_per_token);
    const double latest_rate = 1e6 / (1e6 / live[draft].rate + replay_per_token);
    return episode_rate > current.rate * 1.03 && latest_rate > current.rate * 1.03;
}

void server_adaptive_draft::account_boundary(int64_t now_us) {
    if (window_boundary < 0) { return; }
    if (now_us < window_boundary) { throw std::logic_error("adaptive boundary clock moved backwards"); }
    const double elapsed = (double) (now_us - window_boundary);
    boundary_us += elapsed;
    boundary_mean_us = boundary_count++ == 0 ? elapsed : boundary_mean_us + 0.25 * (elapsed - boundary_mean_us);
    window_boundary = -1;
}

int server_adaptive_draft::begin(int limit, int64_t now_us, int remaining_tokens) {
    if (pending >= 0 || now_us < 0) { throw std::logic_error("adaptive draft round already open or invalid clock"); }
    gain_horizon_tokens = std::clamp(remaining_tokens, 1, 512);
    account_boundary(now_us);
    pending = std::min(active, std::max(0, limit));
    // Capacity-limited tails keep the existing hooks and need no resume checkpoint.
    mode_transition_allowed = limit > 0;
    if (window_start < 0) { window_start = now_us; }
    if (window_depth < 0) { window_depth = pending; }
    if (pending != active || pending != window_depth) { window_valid = false; }
    return pending;
}

void server_adaptive_draft::account_transition(int64_t elapsed_us) {
    if (pending < 0 || elapsed_us < 0) { throw std::logic_error("invalid adaptive transition accounting"); }
    transition_us += elapsed_us;
    window_transition_us += elapsed_us;
    if (pending == 0 && window_rounds == 0 && window_valid) {
        // Charge the first ordinary forward, but learn steady cost after it drains.
        window_warmup = true;
        window_limit = 1;
    }
    if (probing && incumbent == 0 && pending > 0) {
        probe_reentry_us = std::max(probe_reentry_us, (double) elapsed_us);
    } else if (pending > 0 && incumbent > 0) {
        ordinary_restore_us = elapsed_us;
    }
}

bool server_adaptive_draft::observe(int draft, int accepted, int committed, bool terminal) {
    if (pending < 0 || draft != pending) { throw std::logic_error("adaptive draft did not execute its requested block"); }
    common_speculative_adaptive_observation observation;
    observation.draft = draft;
    observation.accepted = terminal ? std::min(accepted, committed) : accepted;
    observation.committed = committed;
    observation.censored = terminal;
    model->observe(observation);
    ordinary_tokens = draft == 0 ? ordinary_tokens + committed : 0;
    gain_horizon_tokens = std::max(1, gain_horizon_tokens - committed);
    pending = -1;
    ++total_rounds;
    ++window_rounds;
    window_tokens += committed;
    if (probing && !terminal && !window_warmup && draft == active) { ++probe_rounds; }
    if (terminal) { window_valid = false; }
    return terminal || window_rounds >= window_limit;
}

void server_adaptive_draft::drain(int64_t now_us, bool terminal) {
    if (pending >= 0) { throw std::logic_error("adaptive cost window closed before acceptance"); }
    if (window_start < 0) {
        if (terminal) { account_boundary(now_us); }
        return;
    }
    if (window_rounds == 0 && (!terminal || now_us == window_start)) { window_start = -1; return; }
    if (now_us <= window_start) { throw std::logic_error("adaptive cost clock did not advance"); }
    const int timing_rounds = std::max(1, window_rounds);
    last_round_us = (double) (now_us - window_start) / timing_rounds;
    last_transition_us = window_transition_us;
    last_depth = window_depth;
    last_rounds = window_rounds;
    const bool complete_window = window_valid && window_rounds == window_limit && !terminal;
    last_warmup = window_warmup;
    last_valid = complete_window && !last_warmup;
    if (complete_window && last_warmup) {
        ordinary_entry_us = std::max(window_transition_us, (double) (now_us - window_start) - ordinary_round_us);
    }
    const int timing_depth = window_depth >= 0 ? window_depth : active;
    const auto previous_incumbent = live[incumbent];
    model->observe_cost_window(timing_depth, now_us - window_start, timing_rounds, probing, last_valid, window_transition_us);
    if (last_valid && timing_depth == 0) {
        ordinary_round_us = ((double) (now_us - window_start) - window_transition_us) / timing_rounds;
    }
    if (last_valid) {
        const double elapsed = (double) (now_us - window_start) - window_transition_us +
                timing_rounds * boundary_mean_us / normal_window_rounds;
        live[timing_depth] = { 1e6 * window_tokens / std::max(1.0, elapsed), model->committed_tokens(), window_tokens, timing_rounds };
        if (probing) {
            probe_elapsed_us += elapsed;
            probe_tokens += window_tokens;
        } else if (incumbent == 0 && timing_depth == 0) {
            ordinary_slow_windows = ordinary_reference_rate > live[0].rate * reentry_cost_change ?
                    std::min(2, ordinary_slow_windows + 1) : 0;
        } else if (timing_depth == incumbent) {
            incumbent_windows[0] = incumbent_windows[1];
            incumbent_windows[1] = live[timing_depth];
        }
    }
    window_tokens = 0;
    window_transition_us = 0.0;
    window_warmup = false;
    ++windows;
    window_rounds = 0;
    window_depth = -1;
    window_valid = true;
    window_start = -1;
    window_boundary = terminal ? -1 : now_us;
    action = "hold";
    reason = "none";
    predictions = json::array();
    if (last_warmup && complete_window) {
        window_limit = probing ? ordinary_probe_window_rounds : normal_window_rounds;
        action = probing ? "probe" : "hold";
        if (probing) { probe_outcome = "warmup"; }
        return;
    }
    if (terminal || !last_valid) {
        if (probing) {
            if (!terminal) { cool_down(active); }
            probe_outcome = terminal ? "terminal" : "invalid_window";
        }
        probing = false;
        active = incumbent;
        window_limit = normal_window_rounds;
        return;
    }
    if (ordinary_only) {
        reentry_cost_us = 0.0;
        return;
    }

    bool acceptance_changed = false;
    if (!probing && incumbent > 0 && timing_depth == incumbent) {
        const auto & current = live[incumbent];
        if (acceptance_reference.rounds == 0) { acceptance_reference = current; }
        const double reference_reward = (double) acceptance_reference.committed / acceptance_reference.rounds;
        const double reward = (double) current.committed / current.rounds;
        acceptance_slow_windows = reward <= recovery_reward_ratio * reference_reward &&
                current.rate <= recovery_rate_ratio * acceptance_reference.rate ? acceptance_slow_windows + 1 : 0;
        if (promotion_windows > 0) { --promotion_windows; }
        if (acceptance_slow_windows >= 2) {
            model->refresh_acceptance(incumbent, 2 * recovery_window_rounds);
            ++acceptance_refreshes;
            acceptance_changed = true;
            acceptance_reference = current;
            acceptance_slow_windows = 0;
        }
        window_limit = promotion_windows > 0 || acceptance_slow_windows > 0 ? recovery_window_rounds : normal_window_rounds;
    }

    // D0 cannot observe changed acceptance. Cost aging alone must not restart DSpark.
    const bool block_reentry = !probing && incumbent == 0 && ordinary_slow_windows < 2;
    if (block_reentry) {
        ++reentry_blocked_windows;
        reason = "ordinary_no_new_evidence";
    }
    std::vector<int> excluded;
    for (int draft = 0; draft <= maximum; ++draft) {
        if ((probing && draft != active) || (!probing && cooldown_remaining(draft) > 0) ||
                (block_reentry && draft > 0)) {
            excluded.push_back(draft);
        }
    }
    reentry_cost_us = incumbent == 0 ? (probing ? probe_reentry_us : ordinary_tokens * ordinary_round_us) : 0.0;
    auto result = model->evaluate(incumbent, excluded, boundary_mean_us / normal_window_rounds,
            reentry_cost_us, gain_horizon_tokens);
    if (!probing && incumbent == 0 && result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH) {
        // Previous direct rounds predate the fallback; verify the new cost regime first.
        result.action = COMMON_SPECULATIVE_ADAPTIVE_PROBE;
    }
    if (probing && active > 0) {
        const auto & candidate = result.predictions[active];
        if (probe_rounds >= probe_max_rounds && candidate.relative_gain.mean > 0.03 &&
                candidate.probability_gain >= bounded_probe_probability && positive_supported(active)) {
            result.action = COMMON_SPECULATIVE_ADAPTIVE_SWITCH;
            result.draft = active;
            reason = "bounded_probe_gain";
        }
        if (result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && !positive_supported(active)) {
            result.action = COMMON_SPECULATIVE_ADAPTIVE_HOLD;
            reason = "positive_rate_guard";
        }
    }
    if (!probing && incumbent > 0 && cooldown_remaining(0) > 0 &&
            live[incumbent].tokens > live[0].tokens && ordinary_supported()) {
        auto reconsider = excluded;
        reconsider.erase(std::remove(reconsider.begin(), reconsider.end(), 0), reconsider.end());
        const auto fresh = model->evaluate(incumbent, reconsider, boundary_mean_us / normal_window_rounds,
                reentry_cost_us, gain_horizon_tokens);
        if (fresh.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && fresh.draft == 0) {
            result = fresh;
            ++ordinary_cooldown_releases;
            reason = "fresh_ordinary_evidence";
        }
    }
    if (result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && result.draft == 0 && !ordinary_supported()) {
        result.action = COMMON_SPECULATIVE_ADAPTIVE_HOLD;
        reason = "ordinary_rate_guard";
    }
    const bool positive_probe = result.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && result.draft > 0;
    bool start_ordinary_recheck = false;
    if (probing && ordinary_recheck) {
        // Only the new steady D0 window can confirm a recheck.
        const bool confirmed = measured_ordinary_saving() > 0.0;
        result.action = confirmed ? COMMON_SPECULATIVE_ADAPTIVE_SWITCH : COMMON_SPECULATIVE_ADAPTIVE_HOLD;
        result.draft = 0;
        reason = confirmed ? "ordinary_recheck_gain" : "ordinary_recheck_no_gain";
    } else if (!probing && incumbent > 0 && live[0].rate > 0.0 &&
            (result.action == COMMON_SPECULATIVE_ADAPTIVE_HOLD || positive_probe)) {
        const auto age = model->committed_tokens() - live[0].tokens;
        if (age < 512) {
            const double saving = measured_ordinary_saving();
            const double entry_us = std::max(ordinary_entry_us, ordinary_round_us);
            const int recheck_tokens = 1 + ordinary_probe_window_rounds;
            const double restore_us = ordinary_restore_us > 0.0 ? ordinary_restore_us : recheck_tokens * ordinary_round_us;
            if (age <= ordinary_fresh_tokens && gain_horizon_tokens * saving > entry_us) {
                if (cooldown_remaining(0) > 0) { ++ordinary_cooldown_releases; }
                result.action = COMMON_SPECULATIVE_ADAPTIVE_SWITCH;
                result.draft = 0;
                reason = "ordinary_recent_gain";
            } else if (age > ordinary_fresh_tokens && cooldown_remaining(0) == 0 &&
                    (gain_horizon_tokens - recheck_tokens) * saving > entry_us + restore_us) {
                result.action = COMMON_SPECULATIVE_ADAPTIVE_PROBE;
                result.draft = 0;
                start_ordinary_recheck = true;
                reason = "ordinary_recheck";
            } else if (result.refresh_cost && live[0].rate >= ordinary_measured_gain * live[incumbent].rate) {
                result.action = COMMON_SPECULATIVE_ADAPTIVE_HOLD;
                reason = "ordinary_refresh_deferred";
            }
        }
    } else if (!probing && incumbent > 0 && positive_probe && live[0].rate == 0.0 &&
            cooldown_remaining(0) == 0 && gain_horizon_tokens > ordinary_probe_window_rounds + 1 &&
            timing_depth == incumbent && previous_incumbent.rate > 0.0 &&
            model->committed_tokens() - previous_incumbent.tokens <= ordinary_probe_baseline_tokens &&
            live[incumbent].rate <= ordinary_probe_rate_ratio * previous_incumbent.rate) {
        // If the incumbent slowed, measure the better predicted alternative first.
        const auto & ordinary = result.predictions[0];
        const auto & candidate = result.predictions[result.draft];
        if (ordinary.probability_gain >= ordinary_probe_probability && ordinary.relative_gain.mean > 0.03 &&
                ordinary.relative_gain.mean >= candidate.relative_gain.mean + ordinary_probe_gain_lead) {
            result.draft = 0;
            result.refresh_cost = false;
            reason = "ordinary_probe_priority";
        }
    }
    if (!probing && incumbent > 0 && acceptance_slow_windows == 1 &&
            result.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && result.draft > 0) {
        result.action = COMMON_SPECULATIVE_ADAPTIVE_HOLD;
        reason = "acceptance_recheck";
    }
    bool start_recovery = false;
    const bool measured_fallback = result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH &&
            result.draft == 0 && measured_ordinary_saving() > 0.0;
    if (acceptance_changed && !measured_fallback) {
        const auto & older = incumbent_windows[0];
        const auto & newer = incumbent_windows[1];
        const double current_rate = 1e6 * (older.committed + newer.committed) /
                (1e6 * older.committed / older.rate + 1e6 * newer.committed / newer.rate);
        double best_saving = 0.0;
        double best_rate = 0.0;
        int candidate = -1;
        for (int draft = 0; draft <= maximum; ++draft) {
            const auto & prediction = result.predictions[draft];
            const bool measured = live[draft].rate > 0.0 &&
                    model->committed_tokens() - live[draft].tokens < recovery_fresh_tokens;
            if (draft == incumbent || !prediction.cost_available || prediction.relative_gain.mean <= 0.03 ||
                    prediction.probability_gain < recovery_probability ||
                    (cooldown_remaining(draft) > 0 && (!measured || probes[draft].tokens >= older.tokens))) { continue; }
            // Old speed cannot override the posterior after the acceptance regime changed.
            const double rate = measured ? std::min(live[draft].rate, prediction.tokens_per_second.mean) :
                    prediction.tokens_per_second.mean;
            if (rate <= recovery_measured_gain * std::max(current_rate, newer.rate)) { continue; }
            const int probe_budget = recovery_window_rounds * (draft + 1) + (draft == 0 ? 1 : 0);
            const double transition = draft == 0 ? std::max(ordinary_entry_us, ordinary_round_us) +
                    std::max(ordinary_restore_us, probe_budget * ordinary_round_us) : 0.0;
            const double saving = (gain_horizon_tokens - probe_budget) * (1e6 / current_rate - 1e6 / rate) - transition;
            if (gain_horizon_tokens > probe_budget && saving > best_saving) {
                candidate = draft;
                best_saving = saving;
                best_rate = rate;
            }
        }
        if (candidate >= 0) {
            result.action = COMMON_SPECULATIVE_ADAPTIVE_PROBE;
            result.draft = candidate;
            result.refresh_cost = false;
            recovery_baseline_rate = current_rate;
            recovery_expected_rate = best_rate;
            recovery_expected_saving_us = best_saving;
            start_recovery = true;
            reason = "acceptance_recovery_probe";
        } else {
            result.action = COMMON_SPECULATIVE_ADAPTIVE_HOLD;
            result.draft = incumbent;
            result.refresh_cost = false;
            reason = "acceptance_refreshed";
        }
    }
    if (probing && recovery_probe) {
        const bool confirmed = probe_tokens > 0 && probe_elapsed_us > 0.0 &&
                1e6 * probe_tokens / probe_elapsed_us > recovery_measured_gain * recovery_baseline_rate &&
                live[active].rate > recovery_measured_gain * recovery_baseline_rate;
        result.action = confirmed ? COMMON_SPECULATIVE_ADAPTIVE_SWITCH : COMMON_SPECULATIVE_ADAPTIVE_HOLD;
        result.draft = active;
        reason = confirmed ? "acceptance_recovery_gain" : "acceptance_recovery_no_gain";
    }
    if (reason == "none") {
        reason = result.refresh_cost ? "refresh_cost" :
                result.action == COMMON_SPECULATIVE_ADAPTIVE_HOLD ? "insufficient_gain" : "gain";
    }
    for (const auto & p : result.predictions) {
        predictions.push_back({{"draft", p.draft}, {"cost_us", p.round_us.mean},
            {"gain", p.relative_gain.mean}, {"gain_low", p.relative_gain.low}, {"gain_high", p.relative_gain.high},
            {"probability_gain", p.probability_gain}, {"expected_positive_gain", p.expected_positive_gain},
            {"recent_rounds", p.recent_rounds}, {"live_cost_windows", p.live_cost_windows},
            {"cost_age_tokens", p.cost_age_tokens}, {"live_rate", live[p.draft].rate},
            {"predicted_rate", p.tokens_per_second.mean}});
    }
    if (result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH) {
        if (recovery_probe && probing) { ++recovery_switches; }
        cool_down(incumbent);
        probes[result.draft] = {};
        incumbent = active = result.draft;
        incumbent_windows = {};
        acceptance_reference = live[incumbent];
        if (probing && probe_tokens > 0 && probe_elapsed_us > 0.0 && probe_rounds > 0) {
            acceptance_reference = {1e6 * probe_tokens / probe_elapsed_us, model->committed_tokens(), probe_tokens, probe_rounds};
        }
        acceptance_slow_windows = 0;
        promotion_windows = incumbent > 0 ? 2 : 0;
        if (incumbent == 0) {
            ordinary_reference_rate = live[0].rate;
            ordinary_slow_windows = 0;
        }
        probing = false;
        window_limit = promotion_windows > 0 ? recovery_window_rounds : normal_window_rounds;
        ++switches;
        probe_outcome = "switched";
        action = "switch";
    } else if (probing) {
        if (result.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && probe_rounds < probe_max_rounds) {
            window_limit = std::min(active == 0 ? ordinary_probe_window_rounds : probe_window_rounds,
                    probe_max_rounds - probe_rounds);
            probe_outcome = "continue";
            action = "probe";
        } else {
            cool_down(active);
            if (incumbent == 0) {
                ordinary_reference_rate = live[0].rate;
                ordinary_slow_windows = 0;
            }
            probing = false;
            active = incumbent;
            window_limit = promotion_windows > 0 || acceptance_slow_windows > 0 ? recovery_window_rounds : normal_window_rounds;
            probe_outcome = probe_rounds >= probe_max_rounds ? "round_limit" : "no_gain";
        }
    } else if (result.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE) {
        probe_reentry_us = reentry_cost_us;
        active = probe_draft = result.draft;
        probing = true;
        recovery_probe = start_recovery;
        if (start_recovery) { ++recovery_probes; }
        ordinary_recheck = start_ordinary_recheck && !start_recovery;
        probe_rounds = 0;
        probe_elapsed_us = 0.0;
        probe_tokens = 0;
        ++probe_episodes;
        window_limit = active == 0 ? ordinary_probe_window_rounds : probe_window_rounds;
        probe_outcome = "started";
        action = "probe";
    }
}

void server_adaptive_draft::abandon(int64_t now_us) {
    const bool was_probing = probing;
    pending = -1;
    window_valid = false;
    drain(now_us, true);
    if (was_probing) { probe_outcome = "cancelled"; }
    probing = false;
    active = incumbent;
    action = "hold";
}

void server_adaptive_draft::stay_ordinary() {
    if (pending >= 0) { throw std::logic_error("cannot lock ordinary mode during a round"); }
    ordinary_only = true;
    incumbent = active = 0;
    probing = false;
    window_valid = false;
    window_limit = normal_window_rounds;
    action = "hold";
    probe_outcome = "context_shift";
}

json server_adaptive_draft::status() const {
    json cooldowns = json::array();
    for (int draft = 0; draft <= maximum; ++draft) { cooldowns.push_back(cooldown_remaining(draft)); }
    return {{"draft", active}, {"ordinary_only", ordinary_only}, {"cooldown_first_draft", 0}, {"incumbent", incumbent}, {"probe", probing}, {"action", action},
            {"reason", reason}, {"predictions", predictions}, {"ordinary_cooldown_releases", ordinary_cooldown_releases},
            {"ordinary_reference_rate", ordinary_reference_rate}, {"ordinary_slow_windows", ordinary_slow_windows},
            {"reentry_blocked_windows", reentry_blocked_windows},
            {"acceptance_refreshes", acceptance_refreshes}, {"acceptance_slow_windows", acceptance_slow_windows},
            {"acceptance_reference_rate", acceptance_reference.rate},
            {"acceptance_reference_reward", acceptance_reference.rounds > 0 ? (double) acceptance_reference.committed / acceptance_reference.rounds : 0.0},
            {"recovery_probes", recovery_probes}, {"recovery_switches", recovery_switches},
            {"recovery_probe", probing && recovery_probe}, {"recovery_baseline_rate", recovery_baseline_rate},
            {"recovery_expected_rate", recovery_expected_rate}, {"recovery_expected_saving_us", recovery_expected_saving_us},
            {"probe_tokens", probe_tokens}, {"probe_elapsed_us", probe_elapsed_us},
            {"rounds", total_rounds}, {"windows", windows}, {"round_us", last_round_us},
            {"committed_tokens", model->committed_tokens()},
            {"measured_draft", last_depth}, {"measured_rounds", last_rounds}, {"cost_usable", last_valid},
            {"measured_warmup", last_warmup},
            {"measured_transition_us", last_transition_us},
            {"runtime_us", model->runtime_time_us() + boundary_us}, {"probe_us", model->probe_time_us()},
            {"transition_us", transition_us}, {"reentry_cost_us", reentry_cost_us}, {"gain_horizon_tokens", gain_horizon_tokens},
            {"boundary_us", boundary_us}, {"probe_rounds", probe_rounds}, {"probe_draft", probe_draft},
            {"boundary_mean_us", boundary_mean_us}, {"round_overhead_us", boundary_mean_us / normal_window_rounds},
            {"probe_episodes", probe_episodes}, {"switches", switches}, {"probe_outcome", probe_outcome},
            {"cooldown_tokens", cooldowns}, {"calibration_history_tokens", history_tokens}};
}
