#pragma once

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

struct common_speculative_adaptive_params {
    int max_draft = 7;
    double half_life_tokens = 512.0; // 0 retains all observations.
    double prior_acceptance = 0.65;
    double intercept_sd = 1.5;
    double slope_sd = 0.35;
    double depth_sd = 0.45;
    double position_sd = 0.60;
    double cost_relative_sd = 0.03;
    std::array<double, 8> cost_relative_sd_by_draft{}; // 0 uses the shared floor.
    uint64_t refresh_cost_tokens = 128;
    double refresh_probability = 0.20;
    double refresh_expected_gain = 0.03;
    double probe_score_tolerance = 0.02;
    double gain_margin = 0.03;
    double interval_tail = 0.05;
    int min_direct_rounds = 8;
    uint64_t evidence_window_tokens = 512;
    double probe_probability = 0.60;
    double reentry_probability = 0.80;
    int draws = 1024;
    uint64_t seed = 0;
};

struct common_speculative_adaptive_observation {
    int draft = 0; // Actual generated and verified draft count; 0 is ordinary decoding.
    int accepted = 0;
    int committed = 0; // 0 infers accepted + 1 for an uncensored round.
    bool censored = false; // No rejection is known after the accepted prefix.
    bool probe = false;
    double round_us = 0.0; // 0 omits the cost observation.
};

struct common_speculative_adaptive_counts {
    double success = 0.0;
    double failure = 0.0;
};

struct common_speculative_adaptive_interval {
    double mean = 0.0;
    double low = 0.0;
    double high = 0.0;
};

struct common_speculative_adaptive_prediction {
    int draft = 0;
    double direct_rounds = 0.0;
    int recent_rounds = 0;
    bool cost_available = false;
    uint64_t live_cost_windows = 0;
    uint64_t cost_age_tokens = 0;
    double expected_positive_gain = 0.0;
    std::vector<common_speculative_adaptive_interval> conditional;
    std::vector<common_speculative_adaptive_interval> survival;
    common_speculative_adaptive_interval accepted;
    common_speculative_adaptive_interval round_us;
    common_speculative_adaptive_interval tokens_per_second;
    common_speculative_adaptive_interval relative_gain;
    double probability_gain = 0.0;
    double gain_threshold = 0.0;
};

enum common_speculative_adaptive_action {
    COMMON_SPECULATIVE_ADAPTIVE_HOLD,
    COMMON_SPECULATIVE_ADAPTIVE_SWITCH,
    COMMON_SPECULATIVE_ADAPTIVE_PROBE,
    COMMON_SPECULATIVE_ADAPTIVE_NEED_COST,
};

struct common_speculative_adaptive_result {
    common_speculative_adaptive_action action = COMMON_SPECULATIVE_ADAPTIVE_HOLD;
    int draft = 0;
    bool refresh_cost = false;
    double runtime_us = 0.0;
    double probe_us = 0.0;
    double round_overhead_us = 0.0;
    double reentry_cost_us = 0.0;
    int gain_horizon_tokens = 512;
    std::vector<common_speculative_adaptive_prediction> predictions;
};

// One instance owns one model, sampling/KV configuration and context-cost bucket.
// This component does not execute models or change a live speculative context.
class common_speculative_adaptive {
public:
    explicit common_speculative_adaptive(const common_speculative_adaptive_params & params = {});

    // Fixed-prefix repetitions supply costs only, never acceptance observations.
    void set_cost(int draft, double mean_us, double sd_us, int samples);
    void observe(const common_speculative_adaptive_observation & observation);
    // Rebuild one depth from its latest uncensored rounds after a measured regime change.
    void refresh_acceptance(int draft, int retained_rounds);
    // One drained fixed-depth window supplies one independent cost observation.
    // One-time transitions stay in runtime/probe totals but are excluded from recurring cost.
    void observe_cost_window(int draft, double total_us, int round_count, bool probe, bool complete = true, double transition_us = 0.0);
    // Shared overhead affects throughput comparisons; round_us predictions remain inference-only.
    // From D0, amortize reentry cost across the supplied future token horizon.
    common_speculative_adaptive_result evaluate(int current_draft, const std::vector<int> & excluded_drafts = {},
            double round_overhead_us = 0.0, double reentry_cost_us = 0.0, int gain_horizon_tokens = 512) const;
    common_speculative_adaptive_counts counts(int draft, int position) const;
    uint64_t committed_tokens() const;
    double runtime_time_us() const;
    double probe_time_us() const;

private:
    struct cost_stats {
        uint64_t live_windows = 0;
        uint64_t last_live_token = 0;
        double weight = 0.0;
        double mean = 0.0;
        double variance = 0.0; // Weighted population variance; unchanged by aging alone.
    };

    common_speculative_adaptive_params params;
    std::vector<common_speculative_adaptive_counts> observations;
    std::vector<double> rounds;
    struct recent_round {
        uint64_t tokens;
        int accepted;
    };
    std::vector<std::deque<recent_round>> recent_rounds;
    std::vector<cost_stats> costs;
    uint64_t tokens = 0;
    double elapsed_us = 0.0;
    double probe_us = 0.0;

    int cell(int draft, int position) const;
    void update_cost(int draft, double sample);
    void fit(std::vector<double> & mean, std::vector<double> & precision_cholesky) const;
};
