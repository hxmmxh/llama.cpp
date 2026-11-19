#pragma once

#include "common.h"
#include "json.h"
#include "speculative-adaptive.h"

#include <memory>
#include <array>

struct server_adaptive_draft_cost {
    int draft = 0;
    int samples = 0;
    double mean_us = 0.0;
    double sd_us = 0.0;
};

struct server_adaptive_draft_costs {
    int history_tokens = 0;
    float temperature = 0.0f;
    float top_p = 1.0f;
    int top_k = 0;
    float min_p = 0.0f;
    std::vector<server_adaptive_draft_cost> rows;
};

class server_adaptive_draft {
public:
    static void validate(const common_params & params);
    // Run before creating the serving speculator; both contexts must be fresh and empty.
    static server_adaptive_draft_costs calibrate(const common_params & params, llama_context * target, llama_context * draft);
    static server_adaptive_draft_cost summarize_cost(int draft, std::vector<double> samples);
    server_adaptive_draft(const server_adaptive_draft_costs & costs, const common_params & params);

    int begin(int limit, int64_t now_us, int remaining_tokens = 512);
    void account_transition(int64_t elapsed_us);
    bool observe(int draft, int accepted, int committed, bool terminal);
    // Call only after both model contexts have drained.
    void drain(int64_t now_us, bool terminal = false);
    void abandon(int64_t now_us);
    bool ordinary_pending() const { return pending == 0; }
    bool needs_mode_transition(bool speculative_active) const {
        return pending >= 0 && mode_transition_allowed && speculative_active != (pending > 0);
    }
    // A context shift invalidates the saved DSpark resume point for this request.
    void stay_ordinary();
    common_json status() const;

private:
    static constexpr int normal_window_rounds = 16;
    static constexpr int probe_window_rounds = 8;
    static constexpr int ordinary_probe_window_rounds = 8;
    static constexpr int probe_max_rounds = 16;
    static constexpr int recovery_window_rounds = 8;
    static constexpr double recovery_reward_ratio = 0.80;
    static constexpr double recovery_rate_ratio = 0.85;
    static constexpr double recovery_measured_gain = 1.05;
    static constexpr double recovery_probability = 0.60;
    static constexpr uint64_t recovery_fresh_tokens = 256;
    static constexpr double bounded_probe_probability = 0.75;
    static constexpr double reentry_cost_change = 1.10;
    static constexpr double ordinary_measured_gain = 1.05;
    static constexpr uint64_t ordinary_fresh_tokens = 64;
    static constexpr uint64_t ordinary_probe_baseline_tokens = 128;
    static constexpr double ordinary_probe_rate_ratio = 0.80;
    static constexpr double ordinary_probe_probability = 0.90;
    static constexpr double ordinary_probe_gain_lead = 0.05;
    struct probe_history {
        uint64_t tokens = 0;
        uint64_t cooldown = 0;
    };

    std::unique_ptr<common_speculative_adaptive> model;
    std::array<probe_history, 8> probes{};
    struct live_window {
        double rate = 0.0;
        uint64_t tokens = 0;
        uint64_t committed = 0;
        int rounds = 0;
    };
    std::array<live_window, 8> live{};
    std::array<live_window, 2> incumbent_windows{};
    live_window acceptance_reference{};
    int acceptance_slow_windows = 0;
    int promotion_windows = 0;
    uint64_t acceptance_refreshes = 0;
    uint64_t recovery_probes = 0;
    uint64_t recovery_switches = 0;
    bool recovery_probe = false;
    double recovery_baseline_rate = 0.0;
    double recovery_expected_rate = 0.0;
    double recovery_expected_saving_us = 0.0;
    bool ordinary_recheck = false;
    double ordinary_entry_us = 0.0;
    double ordinary_restore_us = 0.0;
    uint64_t window_tokens = 0;
    uint64_t ordinary_cooldown_releases = 0;
    double ordinary_reference_rate = 0.0;
    int ordinary_slow_windows = 0;
    uint64_t reentry_blocked_windows = 0;
    common_json predictions = common_json::array();
    std::string reason = "none";
    int incumbent = 3;
    int active = 3;
    int maximum = 7;
    int pending = -1;
    int window_depth = -1;
    int window_rounds = 0;
    int window_limit = normal_window_rounds;
    int64_t window_start = -1;
    int64_t window_boundary = -1;
    bool window_valid = true;
    bool mode_transition_allowed = false;
    bool window_warmup = false;
    bool last_warmup = false;
    bool probing = false;
    bool ordinary_only = false;
    uint64_t total_rounds = 0;
    uint64_t windows = 0;
    int history_tokens = 0;
    double last_round_us = 0.0;
    double last_transition_us = 0.0;
    int last_depth = 0;
    int last_rounds = 0;
    bool last_valid = false;
    double boundary_us = 0.0;
    double boundary_mean_us = 0.0;
    double transition_us = 0.0;
    double window_transition_us = 0.0;
    double ordinary_round_us = 0.0;
    uint64_t ordinary_tokens = 0;
    double reentry_cost_us = 0.0;
    double probe_reentry_us = 0.0;
    double probe_elapsed_us = 0.0;
    uint64_t probe_tokens = 0;
    int gain_horizon_tokens = 512;
    uint64_t boundary_count = 0;
    int probe_rounds = 0;
    int probe_draft = -1;
    uint64_t probe_episodes = 0;
    uint64_t switches = 0;
    std::string probe_outcome = "none";
    std::string action = "hold";

    uint64_t cooldown_remaining(int draft) const;
    void cool_down(int draft);
    bool ordinary_supported() const;
    double measured_ordinary_saving() const;
    bool positive_supported(int draft) const;
    void account_boundary(int64_t now_us);
};
