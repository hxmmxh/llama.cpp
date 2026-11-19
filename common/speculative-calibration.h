#pragma once

#include "common.h"

static constexpr int COMMON_SPECULATIVE_CALIBRATION_MAX_DRAFT = 7;

struct common_speculative_calibration_params {
    bool ordinary_only = false;
    // Match the serving pipeline. Stage fields are unavailable (0); round_us remains synchronized.
    bool runtime_pipeline = false;
    int warmup = 2;
    int repetitions = 5;
    bool include_ordinary = true;
    int max_draft = COMMON_SPECULATIVE_CALIBRATION_MAX_DRAFT;
};

struct common_speculative_calibration_sample {
    int draft = 0;
    int repetition = 0;
    int accepted = 0;
    int draft_rows = 0;
    int target_rows = 0;
    int64_t draft_us = 0;
    int64_t draft_cleanup_us = 0;
    int64_t target_us = 0;
    int64_t context_us = 0;
    int64_t sampling_us = 0;
    int64_t rollback_us = 0;
    int64_t round_us = 0;
    llama_tokens draft_tokens;
    llama_tokens output_tokens;
};

// Requires fresh, empty single-sequence contexts and no live speculator. Leaves both caches empty.
// ordinary_only permits a null draft context and executes no draft work.
// Prefix restore, sampler reset, validation and speculator setup are outside the synchronized round timings.
std::vector<common_speculative_calibration_sample> common_speculative_calibrate(
        const common_params & params,
        llama_context * target,
        llama_context * draft,
        const llama_tokens & prompt,
        const common_speculative_calibration_params & calibration = {});
