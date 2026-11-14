#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_trace_context {
    uint64_t round_id;
    uint64_t graph_id;
};

// Opt in with GGML_TRACE=<new-file.json>. Timestamps use the platform monotonic clock.
GGML_API bool     ggml_trace_enabled(void);
GGML_API bool     ggml_trace_active(void);
GGML_API uint64_t ggml_trace_time_ns(void);
GGML_API uint64_t ggml_trace_host_time_ns(uint64_t ticks);
GGML_API uint64_t ggml_trace_next_id(void);
GGML_API uint64_t ggml_trace_cpu_track(void);
GGML_API uint64_t ggml_trace_new_track(const char * name);
GGML_API struct ggml_trace_context ggml_trace_get_context(void);
GGML_API void ggml_trace_set_context(struct ggml_trace_context context);
GGML_API void ggml_trace_slice(const char * name, const char * category, uint64_t track,
                              uint64_t begin_ns, uint64_t end_ns, struct ggml_trace_context context,
                              uint64_t id, const char * value_name, uint64_t value);
GGML_API void ggml_trace_instant(const char * name, const char * value_name, uint64_t value);
GGML_API void ggml_trace_diagnostic(const char * name, const char * value_name, uint64_t value);
GGML_API void ggml_trace_submission(uint64_t begin_ns, uint64_t end_ns, struct ggml_trace_context context, uint64_t id);
GGML_API void ggml_trace_gpu(const char * name, uint64_t track, uint64_t begin_ns, uint64_t end_ns,
                            struct ggml_trace_context context, uint64_t id, uint64_t clock_error_ns,
                            uint64_t submit_ns, uint64_t submit_track);
// Flush after backend/context teardown. Also called on normal process exit.
GGML_API void ggml_trace_flush(void);

#ifdef __cplusplus
}

class ggml_trace_scope {
public:
    explicit ggml_trace_scope(const char * name, const char * category = "cpu", uint64_t id = 0,
                              const char * value_name = nullptr, uint64_t value = 0) :
        name(name), category(category), id(id), value_name(value_name), value(value) {
        if (ggml_trace_active()) {
            track = ggml_trace_cpu_track();
            context = ggml_trace_get_context();
            begin_ns = ggml_trace_time_ns();
        }
    }
    ~ggml_trace_scope() {
        if (begin_ns) {
            ggml_trace_slice(name, category, track, begin_ns, ggml_trace_time_ns(), context, id, value_name, value);
        }
    }
    ggml_trace_scope(const ggml_trace_scope &) = delete;
    ggml_trace_scope & operator=(const ggml_trace_scope &) = delete;
private:
    const char * name;
    const char * category;
    uint64_t id, track = 0, begin_ns = 0;
    ggml_trace_context context {};
    const char * value_name;
    uint64_t value;
};

class ggml_trace_context_guard {
public:
    explicit ggml_trace_context_guard(ggml_trace_context context) : previous(ggml_trace_get_context()) {
        ggml_trace_set_context(context);
    }
    ~ggml_trace_context_guard() { ggml_trace_set_context(previous); }
    ggml_trace_context_guard(const ggml_trace_context_guard &) = delete;
    ggml_trace_context_guard & operator=(const ggml_trace_context_guard &) = delete;
private:
    ggml_trace_context previous;
};
#endif
