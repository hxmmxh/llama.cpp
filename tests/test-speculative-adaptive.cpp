#include "speculative-adaptive.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static void near(double actual, double expected, double tolerance, const char * message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

static common_speculative_adaptive_params stationary(int maximum = 7) {
    common_speculative_adaptive_params params;
    params.max_draft = maximum;
    params.half_life_tokens = 0.0;
    params.evidence_window_tokens = 1000000;
    params.draws = 2048;
    params.seed = 20261117;
    return params;
}

static void observe(common_speculative_adaptive & model, int draft, int accepted, int repeats) {
    for (int i = 0; i < repeats; ++i) {
        common_speculative_adaptive_observation round;
        round.draft = draft;
        round.accepted = accepted;
        model.observe(round);
    }
}

static void check_censoring() {
    common_speculative_adaptive model(stationary());
    observe(model, 3, 1, 1);
    near(model.counts(3, 1).success, 1.0, 0.0, "accepted first position missing");
    near(model.counts(3, 2).failure, 1.0, 0.0, "first rejection missing");
    near(model.counts(3, 3).failure, 0.0, 0.0, "unreached conditional position counted as failure");
    near(model.counts(2, 1).success, 0.0, 0.0, "long-D observation copied into a different actual block");
    observe(model, 3, 3, 1);
    near(model.counts(3, 3).success, 1.0, 0.0, "all-accepted prefix missing");
    near(model.counts(7, 4).failure, 0.0, 0.0, "unseen suffix counted as failure");
    common_speculative_adaptive_observation eos;
    eos.draft = 3;
    eos.accepted = 1;
    eos.committed = 1;
    eos.censored = true;
    eos.round_us = 1e9;
    model.set_cost(3, 1000.0, 0.0, 5);
    model.observe(eos);
    near(model.counts(3, 2).failure, 1.0, 0.0, "EOS incorrectly added a rejection");
    const auto result = model.evaluate(3);
    near(result.predictions[3].direct_rounds, 2.0, 0.0, "censored round qualified as direct switching evidence");
    require(result.predictions[3].recent_rounds == 2, "censored round counted toward recent qualification");
    require(result.predictions[3].round_us.mean < 1100.0, "shortened terminal round contaminated ordinary cycle cost");
}

static void check_empirical_curve() {
    common_speculative_adaptive model(stationary());
    // Exact histogram for S1=.6070, S2=.3886, S3=.2314.
    observe(model, 3, 0, 3930);
    observe(model, 3, 1, 2184);
    observe(model, 3, 2, 1572);
    observe(model, 3, 3, 2314);
    const auto result = model.evaluate(3);
    const auto & depth = result.predictions[3];
    const double survival[] = {0.6070, 0.3886, 0.2314};
    const double conditional[] = {0.6070, 0.3886 / 0.6070, 0.2314 / 0.3886};
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        near(depth.survival[i].mean, survival[i], 0.012, "survival estimate does not recover observed prefix frequencies");
        near(depth.conditional[i].mean, conditional[i], 0.012, "conditional estimate assumes a constant acceptance rate");
        sum += depth.survival[i].mean;
    }
    near(depth.accepted.mean, sum, 1e-10, "mean accepted length violates the survival identity");
    near(depth.accepted.mean, 1.227, 0.025, "accepted length differs from the supplied D3 curve");
    const auto & unseen = result.predictions[7];
    require(unseen.conditional[6].high - unseen.conditional[6].low > 0.25, "unmeasured tail became falsely certain");
    require(unseen.accepted.high - unseen.accepted.low > depth.accepted.high - depth.accepted.low,
            "unseen D7 has less uncertainty than the measured D3");
    require(result.predictions[2].direct_rounds == 0.0, "cross-depth prediction counted as a real observation");
}

static void check_depth_effects() {
    auto params = stationary(3);
    common_speculative_adaptive model(params);
    observe(model, 1, 0, 2000);
    observe(model, 3, 3, 2000);
    const auto result = model.evaluate(3);
    require(result.predictions[1].accepted.mean < 0.05, "D1 could not learn its own prefix distribution");
    require(result.predictions[3].accepted.mean > 2.9, "D3 could not learn a different block distribution");
    for (const auto & prediction : result.predictions) {
        double previous = 1.0;
        for (const auto & position : prediction.survival) {
            require(position.low >= 0.0 && position.high <= 1.0 && position.mean <= previous,
                    "invalid or increasing prefix survival probability");
            previous = position.mean;
        }
        require(prediction.accepted.low >= 0.0 && prediction.accepted.high <= prediction.draft,
                "accepted length outside the physical draft bound");
    }
}

static void check_switches() {
    for (bool upward : {false, true}) {
        common_speculative_adaptive model(stationary(3));
        model.set_cost(1, upward ? 12000.0 : 10000.0, 0.0, 10);
        model.set_cost(2, 90000.0, 0.0, 10);
        model.set_cost(3, upward ? 18000.0 : 30000.0, 0.0, 10);
        observe(model, 1, upward ? 0 : 1, 500);
        observe(model, 3, upward ? 3 : 0, 500);
        const int current = upward ? 1 : 3;
        const auto result = model.evaluate(current);
        require(result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH, "clear measured improvement did not switch");
        require(result.draft == (upward ? 3 : 1), "selected the wrong direction of depth change");
        near(result.predictions[current].relative_gain.low, 0.0, 0.0, "current arm comparison is not paired with itself");
    }
    common_speculative_adaptive missing(stationary(3));
    missing.set_cost(1, 10000.0, 0.0, 5);
    require(missing.evaluate(3).action == COMMON_SPECULATIVE_ADAPTIVE_NEED_COST, "missing incumbent cost silently extrapolated");

    auto params = stationary(3);
    common_speculative_adaptive unmeasured(params);
    unmeasured.set_cost(1, 1000.0, 0.0, 5);
    unmeasured.set_cost(3, 30000.0, 0.0, 5);
    observe(unmeasured, 3, 0, 200);
    const auto probe = unmeasured.evaluate(3);
    require(probe.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && probe.draft == 1,
            "predicted short D was promoted without actual short-block evidence");
}

static void check_gain_trigger() {
    common_speculative_adaptive model(stationary(4));
    for (int d = 1; d <= 4; ++d) { model.set_cost(d, 1000.0, 0.0, 1); }
    require(model.committed_tokens() == 0 && model.counts(1, 1).success == 0.0,
            "startup cost repetitions created acceptance evidence");
    observe(model, 3, 3, 128);
    const auto proposal = model.evaluate(3);
    require(proposal.runtime_us == 0.0 && proposal.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && proposal.draft == 4,
            "positive expected gain still requires earned timing credit");
    require(model.evaluate(3, {4}).action == COMMON_SPECULATIVE_ADAPTIVE_HOLD, "excluded candidate was scheduled");

    common_speculative_adaptive uncertain(stationary(2));
    uncertain.set_cost(1, 1000.0, 0.0, 1);
    uncertain.set_cost(2, 1300.0, 0.0, 1);
    const auto optimistic = uncertain.evaluate(1);
    require(optimistic.predictions[2].relative_gain.high > 0.03 && optimistic.predictions[2].relative_gain.mean < 0.03,
            "fixture does not distinguish optimistic from expected gain");
    require(optimistic.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && optimistic.refresh_cost,
            "unmeasured cost with meaningful upside was not refreshed");
    uncertain.observe_cost_window(2, 1300.0 * 8, 8, true);
    require(uncertain.evaluate(1).action == COMMON_SPECULATIVE_ADAPTIVE_HOLD,
            "fresh live cost was repeatedly explored on uncertainty alone");
    observe(uncertain, 0, 0, 128);
    require(uncertain.evaluate(1).refresh_cost, "stale live cost could not be refreshed");
    require(uncertain.evaluate(1, {2}).action == COMMON_SPECULATIVE_ADAPTIVE_HOLD,
            "cost refresh ignored cooldown");
    uncertain.set_cost(2, 1000000.0, 0.0, 1);
    require(uncertain.evaluate(1).action == COMMON_SPECULATIVE_ADAPTIVE_HOLD,
            "cost refresh probed a candidate without useful upside");

    auto noisy_params = stationary(2);
    noisy_params.cost_relative_sd = 1.0;
    common_speculative_adaptive noisy(noisy_params);
    noisy.set_cost(1, 1000.0, 0.0, 1);
    noisy.set_cost(2, 2000.0, 0.0, 1);
    const auto unlikely = noisy.evaluate(1);
    require(unlikely.predictions[2].relative_gain.mean > 0.03 && unlikely.predictions[2].probability_gain < 0.60,
            "fixture does not distinguish mean gain from gain probability");
    require(unlikely.action != COMMON_SPECULATIVE_ADAPTIVE_SWITCH,
            "uncertain gain was promoted without direct evidence");
    noisy.set_cost(0, 600.0, 0.0, 1);
    const auto reentry = noisy.evaluate(0);
    require(reentry.predictions[1].relative_gain.high > 0.03 && reentry.predictions[1].probability_gain < 0.80,
            "fixture must have unresolved but unlikely reentry upside");
    require(reentry.action == COMMON_SPECULATIVE_ADAPTIVE_HOLD && !reentry.refresh_cost,
            "D0 restarted on cost uncertainty without credible net gain");
}

static void check_recent_evidence() {
    common_speculative_adaptive_params params;
    params.max_draft = 2;
    common_speculative_adaptive model(params);
    model.set_cost(1, 1000.0, 0.0, 1);
    model.set_cost(2, 10000.0, 0.0, 1);
    observe(model, 2, 0, 16);
    observe(model, 1, 1, 8);
    const auto result = model.evaluate(2);
    require(result.predictions[1].direct_rounds < 8.0 && result.predictions[1].recent_rounds == 8,
            "recent rounds were confused with decayed evidence");
    require(result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && result.draft == 1,
            "eight recent rounds could not qualify a clearly better candidate");
    observe(model, 2, 0, 512);
    const auto aged = model.evaluate(2);
    require(aged.predictions[1].recent_rounds == 0 && aged.action != COMMON_SPECULATIVE_ADAPTIVE_SWITCH,
            "stale direct rounds still qualify a permanent switch");
}

static void check_stale_cost_uncertainty() {
    common_speculative_adaptive_params params;
    params.max_draft = 3;
    common_speculative_adaptive model(params);
    model.set_cost(1, 1000.0, 400.0, 10);
    model.set_cost(3, 1000.0, 0.0, 1);
    auto previous = model.evaluate(3).predictions[1].round_us;
    for (int batch : {128, 1872}) {
        observe(model, 3, 3, batch);
        const auto stale = model.evaluate(3).predictions[1].round_us;
        require(stale.high - stale.low >= previous.high - previous.low,
                "aging unobserved costs made their mean more certain");
        previous = stale;
    }
    model.observe_cost_window(1, 1200.0, 1, true);
    near(model.evaluate(3).predictions[1].round_us.mean, 1200.0, 10.0,
            "fresh cost failed to replace aged evidence");
}

static void check_zero_scatter_cost_aging() {
    for (double half_life : {512.0, 0.0}) {
        auto params = stationary(3);
        params.half_life_tokens = half_life;
        common_speculative_adaptive model(params);
        model.set_cost(1, 1000.0, 0.0, 1);
        model.set_cost(3, 1000.0, 0.0, 1);
        const auto fresh = model.evaluate(3).predictions[1].round_us;
        observe(model, 3, 3, 128);
        const auto aged = model.evaluate(3).predictions[1].round_us;
        if (half_life == 0.0) {
            near(aged.high - aged.low, fresh.high - fresh.low, 0.0, "disabled forgetting changed cost evidence");
            continue;
        }
        require(aged.high - aged.low > 1.25 * (fresh.high - fresh.low),
                "unvisited zero-scatter startup cost retained false certainty");
        observe(model, 3, 3, 20000);
        const auto stale = model.evaluate(3).predictions[1].round_us;
        require(std::isfinite(stale.mean) && stale.low > 0.0 && stale.high - stale.low > aged.high - aged.low,
                "very old costs lost finite, positive uncertainty");
        model.observe_cost_window(1, 1000.0, 1, true);
        const auto refreshed = model.evaluate(3).predictions[1].round_us;
        near(refreshed.high - refreshed.low, fresh.high - fresh.low, 1.0,
                "fresh measurement failed to restore zero-scatter cost precision");
    }
}

static void check_shared_round_overhead() {
    common_speculative_adaptive model(stationary(2));
    model.set_cost(1, 1000.0, 0.0, 10);
    model.set_cost(2, 4000.0, 0.0, 10);
    observe(model, 1, 0, 500);
    observe(model, 2, 2, 500);
    const auto bare = model.evaluate(1);
    require(bare.action == COMMON_SPECULATIVE_ADAPTIVE_HOLD, "fixture should prefer D1 without boundary overhead");
    const auto adjusted = model.evaluate(1, {}, 2000.0);
    require(adjusted.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && adjusted.draft == 2,
            "shared boundary overhead was omitted from throughput comparison");
    near(adjusted.predictions[2].round_us.mean, bare.predictions[2].round_us.mean, 0.0,
            "boundary overhead contaminated learned inference cost");
    near(adjusted.predictions[1].relative_gain.low, 0.0, 0.0, "shared overhead broke self-comparison");
    bool rejected = false;
    try { model.evaluate(1, {}, -1.0); } catch (const std::invalid_argument &) { rejected = true; }
    require(rejected, "invalid boundary overhead accepted");
}

static void check_forgetting() {
    auto params = stationary(3);
    params.half_life_tokens = 4.0;
    common_speculative_adaptive model(params);
    observe(model, 3, 1, 1);
    observe(model, 1, 1, 1);
    near(model.counts(3, 1).success, std::exp2(-0.5), 1e-12, "forgetting does not use committed-token age");
    require(model.committed_tokens() == 4, "committed clock is incorrect");

    params.half_life_tokens = 32.0;
    common_speculative_adaptive drift(params);
    observe(drift, 3, 3, 300);
    require(drift.evaluate(3).predictions[3].accepted.mean > 2.7, "easy phase was not learned");
    observe(drift, 3, 0, 300);
    require(drift.evaluate(3).predictions[3].accepted.mean < 0.2, "recent rejections could not replace stale easy-phase evidence");
}

static void check_per_depth_cost_uncertainty() {
    auto params = stationary(3);
    params.cost_relative_sd_by_draft[2] = 0.50;
    common_speculative_adaptive model(params);
    for (int d = 1; d <= 3; ++d) { model.set_cost(d, 1000.0, 0.0, 1); }
    auto predictions = model.evaluate(3).predictions;
    require(predictions[2].round_us.high - predictions[2].round_us.low > 1000.0,
            "noisy depth lost its own cost uncertainty");
    require(predictions[3].round_us.high - predictions[3].round_us.low < 120.0,
            "another depth's noise contaminated the incumbent cost");
}

static void check_drained_cost_windows() {
    common_speculative_adaptive model(stationary(3));
    model.set_cost(3, 1000.0, 0.0, 1);
    observe(model, 3, 0, 16);
    model.observe_cost_window(3, 32000.0, 16, false);
    const auto result = model.evaluate(3);
    near(result.predictions[3].round_us.mean, 2000.0, 10.0, "first live window did not supersede startup-prefix cost");
    near(result.runtime_us, 32000.0, 1e-9, "window time was not accounted");
    model.observe_cost_window(1, 1000.0, 1, true, false);
    const auto charged = model.evaluate(3);
    require(!charged.predictions[1].cost_available, "incomplete window learned a misleading normal cost");
    near(charged.probe_us, 1000.0, 1e-9, "incomplete probe window was not accounted");
    near(charged.runtime_us, 33000.0, 1e-9, "incomplete window was omitted from runtime");
}

static void check_validation_and_reproducibility() {
    common_speculative_adaptive model(stationary(3));
    observe(model, 3, 1, 4);
    const auto before = model.committed_tokens();
    int rejected = 0;
    for (int which = 0; which < 6; ++which) {
        common_speculative_adaptive_observation invalid;
        invalid.draft = 3;
        if (which == 0) { invalid.draft = 4; }
        if (which == 1) { invalid.accepted = 4; }
        if (which == 2) { invalid.censored = true; }
        if (which == 3) { invalid.round_us = std::numeric_limits<double>::quiet_NaN(); }
        if (which == 4) { invalid.committed = 3; }
        if (which == 5) { invalid.probe = true; }
        try { model.observe(invalid); } catch (const std::invalid_argument &) { ++rejected; }
    }
    require(rejected == 6 && model.committed_tokens() == before, "invalid observations mutated the model");
    near(model.counts(3, 2).failure, 4.0, 0.0, "validation failure altered existing evidence");
    const auto first = model.evaluate(3);
    const auto second = model.evaluate(3);
    near(first.predictions[3].accepted.mean, second.predictions[3].accepted.mean, 0.0, "fixed posterior seed is not reproducible");
    near(model.counts(3, 2).failure, 4.0, 0.0, "prediction added its own output as evidence");
    require(first.predictions[3].accepted.high > first.predictions[3].accepted.low, "downward estimates lost all uncertainty");
}

static void check_ordinary_candidate() {
    auto params = stationary(1);
    params.cost_relative_sd = 0.001;
    common_speculative_adaptive model(params);
    model.set_cost(0, 1000.0, 0.0, 10);
    model.set_cost(1, 1500.0, 0.0, 10);
    observe(model, 1, 0, 500);
    auto result = model.evaluate(1);
    require(result.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && result.draft == 0,
            "ordinary fallback must first collect live evidence");
    const auto before = model.counts(1, 1);
    observe(model, 0, 0, 8);
    result = model.evaluate(1);
    require(result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && result.draft == 0,
            "slower DSpark did not fall back to ordinary");
    const auto & ordinary = result.predictions[0];
    require(ordinary.conditional.empty() && ordinary.survival.empty(), "D0 invented draft positions");
    near(ordinary.accepted.high, 0.0, 0.0, "ordinary accepted length is not exactly zero");
    near(ordinary.tokens_per_second.mean, 1000.0, 1.0, "ordinary reward must be one token");
    near(model.counts(1, 1).failure, before.failure, 0.0, "ordinary observation fabricated rejection evidence");
    require(model.evaluate(1, {0}).action == COMMON_SPECULATIVE_ADAPTIVE_HOLD, "D0 ignored cooldown");
    model.set_cost(1, 1010.0, 0.0, 10);
    require(model.evaluate(1).draft == 0, "ordinary should win within the speculation margin");
    require(model.evaluate(0).action == COMMON_SPECULATIVE_ADAPTIVE_HOLD, "D0 should not probe a slower draft");
    model.set_cost(1, 500.0, 0.0, 10);
    result = model.evaluate(0);
    require(result.action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && result.draft == 1,
            "fresh profitable DSpark could not resume from D0");

    require(model.evaluate(0, {}, 0.0, 1e9, 128).action == COMMON_SPECULATIVE_ADAPTIVE_HOLD,
            "expensive replay did not suppress an unamortized restart");
    require(model.evaluate(0, {}, 0.0, 1000.0, 512).draft == 1,
            "affordable replay blocked profitable restart");
    auto uncertain_params = stationary(1);
    common_speculative_adaptive tied(uncertain_params);
    tied.set_cost(0, 1000.0, 0.0, 10);
    tied.set_cost(1, 1000.0, 0.0, 10);
    observe(tied, 1, 0, 2000);
    observe(tied, 0, 0, 8);
    require(tied.evaluate(1).action == COMMON_SPECULATIVE_ADAPTIVE_SWITCH && tied.evaluate(1).draft == 0,
            "3 percent cost floor permanently prevented a no-benefit fallback");
    common_speculative_adaptive transition(stationary(1));
    transition.observe_cost_window(0, 104000, 4, true, true, 100000);
    near(transition.runtime_time_us(), 104000, 0, "transition missing from runtime");
    near(transition.probe_time_us(), 104000, 0, "transition missing from probe time");
    near(transition.evaluate(0).predictions[0].round_us.mean, 1000, 5,
            "one-time transition inflated short-window recurring cost");
    auto aging_params = params;
    aging_params.half_life_tokens = 8;
    aging_params.evidence_window_tokens = 8;
    common_speculative_adaptive aging(aging_params);
    aging.set_cost(0, 1000, 0, 1);
    aging.set_cost(1, 500, 0, 1);
    observe(aging, 1, 0, 8);
    observe(aging, 0, 0, 8);
    const auto aged = aging.evaluate(0);
    require(aged.predictions[1].recent_rounds == 0, "ordinary tokens did not age DSpark direct evidence");
    require(aged.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && aged.draft == 1,
            "stale DSpark must be probed again before resuming");
    int rejected = 0;
    for (int which = 0; which < 3; ++which) {
        common_speculative_adaptive_observation invalid;
        if (which == 0) { invalid.accepted = 1; }
        if (which == 1) { invalid.committed = 2; }
        if (which == 2) { invalid.draft = -1; }
        try { aging.observe(invalid); } catch (const std::invalid_argument &) { ++rejected; }
    }
    require(rejected == 3 && aging.committed_tokens() == 16, "invalid D0 feedback mutated evidence");
}

static void check_acceptance_refresh() {
    for (int draft = 1; draft <= 7; ++draft) {
        auto params = stationary();
        params.half_life_tokens = 512;
        common_speculative_adaptive model(params);
        model.set_cost(draft, 1000, 0, 1);
        observe(model, draft, draft, 40);
        observe(model, draft, 0, 16);
        const auto tokens = model.committed_tokens();
        model.observe_cost_window(draft, 16000, 16, false);
        const auto before = model.evaluate(draft).predictions[draft];
        model.refresh_acceptance(draft, 16);
        const auto after = model.evaluate(draft).predictions[draft];
        near(model.counts(draft, 1).success, 0, 0, "refresh retained old accepted prefixes");
        require(model.counts(draft, 1).failure > 15, "refresh dropped recent rejections");
        require(after.recent_rounds == 16 && after.accepted.mean < before.accepted.mean,
                "recent acceptance refresh did not respond to regression");
        require(model.committed_tokens() == tokens && model.runtime_time_us() == 16000 && model.probe_time_us() == 0,
                "refresh recounted token or timing totals");
        near(after.round_us.mean, before.round_us.mean, 0, "acceptance refresh changed timing evidence");
    }
}

static void check_probe_ties() {
    for (bool refresh : {false, true}) {
        auto params = stationary();
        params.probe_probability = refresh ? 1.0 : 0.60;
        params.reentry_probability = refresh ? 1.0 : 0.80;
        common_speculative_adaptive model(params);
        model.set_cost(3, 1000, 0, 1);
        observe(model, 3, 1, 16);
        // Set two unmeasured depths to nearly equal predicted rates.
        model.set_cost(2, 1000, 0, 1);
        model.set_cost(6, 1000, 0, 1);
        const auto initial = model.evaluate(3);
        const double reward2 = initial.predictions[2].accepted.mean + 1;
        const double reward6 = initial.predictions[6].accepted.mean + 1;
        model.set_cost(2, reward2 * 400, 0, 1);
        model.set_cost(6, reward6 * (refresh ? 410 : 397), 0, 1);
        const auto result = model.evaluate(3);
        require(result.action == COMMON_SPECULATIVE_ADAPTIVE_PROBE && result.refresh_cost == refresh,
                "near-tie fixture did not request the expected probe");
        require(result.draft == 2, "near-tie probe ignored the cheaper measured round cost");
        model.set_cost(6, reward6 * 300, 0, 1);
        require(model.evaluate(3).draft == 6, "probe tie tolerance suppressed a clear gain");
    }
}

int main() {
    try {
        check_acceptance_refresh();
        check_probe_ties();
        check_ordinary_candidate();
        check_censoring();
        check_empirical_curve();
        check_depth_effects();
        check_switches();
        check_gain_trigger();
        check_recent_evidence();
        check_stale_cost_uncertainty();
        check_zero_scatter_cost_aging();
        check_shared_round_overhead();
        check_forgetting();
        check_drained_cost_windows();
        check_per_depth_cost_uncertainty();
        check_validation_and_reproducibility();
        std::puts("adaptive draft: censoring, curves, gain triggers, recent evidence, costs, drift and validation passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "adaptive draft test: %s\n", error.what());
        return 1;
    }
}
