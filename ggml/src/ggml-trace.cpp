#include "ggml-trace.h"

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <time.h>
#endif

namespace {
struct trace_event {
    std::string name, category, value_name;
    char phase;
    uint64_t track, begin, end, id, value;
    ggml_trace_context context;
};

struct trace_track {
    uint64_t id;
    std::string name;
};

static uint64_t trace_env_count(const char * name, uint64_t fallback, uint64_t maximum) {
    const char * value = std::getenv(name);
    if (!value || !*value) {
        return fallback;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (*value == '-' || errno || *end || parsed > maximum) {
        std::fprintf(stderr, "ggml_trace: invalid %s; using %llu\n", name, (unsigned long long) fallback);
        return fallback;
    }
    return parsed;
}

struct trace_state {
    FILE * file = nullptr;
    std::mutex mutex;
    std::atomic<uint64_t> next_id {1};
    std::atomic<bool> closed {false};
    uint64_t origin = 0, start = 0, stop = UINT64_MAX, dropped = 0;
    size_t capacity = 0;
    std::vector<trace_event> events;
    std::vector<trace_track> tracks;

    trace_state() {
        const char * path = std::getenv("GGML_TRACE");
        if (!path || !*path) {
            return;
        }
        // Existing evidence must not be overwritten.
        file = std::fopen(path, "wx");
        if (!file) {
            std::fprintf(stderr, "ggml_trace: cannot create %s (errno %d); tracing disabled\n", path, errno);
            return;
        }
        origin = ggml_trace_time_ns();
        start = origin + trace_env_count("GGML_TRACE_START_MS", 0, 86400000) * 1000000;
        const uint64_t duration = trace_env_count("GGML_TRACE_DURATION_MS", 0, 86400000);
        if (duration) {
            stop = start + duration * 1000000;
        }
        capacity = (size_t) trace_env_count("GGML_TRACE_MAX_EVENTS", 200000, 2000000);
        events.reserve(capacity);
        std::atexit(ggml_trace_flush);
        std::fprintf(stderr, "ggml_trace: recording to %s (max %zu events)\n", path, capacity);
    }
};

// Keep the recorder alive while backend static destructors run.
static trace_state & state() {
    static trace_state * recorder = new trace_state;
    return *recorder;
}

thread_local ggml_trace_context trace_context {};

static void append(trace_event event, bool selected = false) {
    auto & s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.file || s.closed || event.begin < s.origin || (!selected && (event.begin < s.start || event.begin >= s.stop))) {
        return;
    }
    if (s.events.size() >= s.capacity) {
        ++s.dropped;
        return;
    }
    s.events.push_back(std::move(event));
}

static void write_string(FILE * file, const std::string & value) {
    std::fputc('"', file);
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') {
            std::fputc('\\', file);
            std::fputc(c, file);
        } else if (c < 0x20) {
            std::fprintf(file, "\\u%04x", c);
        } else {
            std::fputc(c, file);
        }
    }
    std::fputc('"', file);
}
}

uint64_t ggml_trace_host_time_ns(uint64_t ticks) {
#ifdef _WIN32
    static const uint64_t frequency = []() {
        LARGE_INTEGER value;
        QueryPerformanceFrequency(&value);
        return (uint64_t) value.QuadPart;
    }();
    return (uint64_t) ((long double) ticks * 1000000000.0L / frequency);
#else
    return ticks;
#endif
}

uint64_t ggml_trace_time_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return ggml_trace_host_time_ns((uint64_t) value.QuadPart);
#else
    timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (uint64_t) value.tv_sec * 1000000000 + value.tv_nsec;
#endif
}

bool ggml_trace_enabled(void) {
    auto & s = state();
    return s.file && !s.closed.load(std::memory_order_relaxed);
}

bool ggml_trace_active(void) {
    if (!ggml_trace_enabled()) {
        return false;
    }
    const uint64_t now = ggml_trace_time_ns();
    return now >= state().start && now < state().stop;
}

uint64_t ggml_trace_next_id(void) {
    return state().next_id.fetch_add(1, std::memory_order_relaxed);
}

uint64_t ggml_trace_new_track(const char * name) {
    auto & s = state();
    const uint64_t id = ggml_trace_next_id();
    std::lock_guard<std::mutex> guard(s.mutex);
    s.tracks.push_back({id, name});
    return id;
}

uint64_t ggml_trace_cpu_track(void) {
    thread_local const uint64_t track = []() {
        const uint64_t number = ggml_trace_next_id();
        return ggml_trace_new_track(("CPU " + std::to_string(number)).c_str());
    }();
    return track;
}

ggml_trace_context ggml_trace_get_context(void) { return trace_context; }
void ggml_trace_set_context(ggml_trace_context context) { trace_context = context; }

void ggml_trace_slice(const char * name, const char * category, uint64_t track,
                      uint64_t begin_ns, uint64_t end_ns, ggml_trace_context context,
                      uint64_t id, const char * value_name, uint64_t value) {
    if (end_ns >= begin_ns) {
        append({name, category, value_name ? value_name : "", 'X', track, begin_ns, end_ns, id, value, context});
    }
}

void ggml_trace_instant(const char * name, const char * value_name, uint64_t value) {
    if (ggml_trace_active()) {
        const uint64_t now = ggml_trace_time_ns();
        append({name, "trace", value_name ? value_name : "", 'i', ggml_trace_cpu_track(), now, now, 0, value, trace_context});
    }
}

void ggml_trace_diagnostic(const char * name, const char * value_name, uint64_t value) {
    if (ggml_trace_enabled()) {
        const uint64_t now = ggml_trace_time_ns();
        append({name, "trace", value_name ? value_name : "", 'i', ggml_trace_cpu_track(), now, now, 0, value, trace_context}, true);
    }
}

void ggml_trace_submission(uint64_t begin_ns, uint64_t end_ns, ggml_trace_context context, uint64_t id) {
    append({"vulkan.queue_submit", "submit", "", 'X', ggml_trace_cpu_track(), begin_ns, end_ns, id, 0, context}, true);
}

void ggml_trace_gpu(const char * name, uint64_t track, uint64_t begin_ns, uint64_t end_ns,
                    ggml_trace_context context, uint64_t id, uint64_t clock_error_ns,
                    uint64_t submit_ns, uint64_t submit_track) {
    if (end_ns < begin_ns) {
        return;
    }
    auto & s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.file || s.closed || begin_ns < s.origin) {
        return;
    }
    if (s.capacity - s.events.size() < 3) {
        s.dropped += 3;
        return;
    }
    s.events.push_back({name, "gpu", "calibration_max_deviation_ns", 'G', track, begin_ns, end_ns, id, clock_error_ns, context});
    if (begin_ns >= submit_ns) {
        s.events.push_back({"submit_to_gpu", "submit", "", 's', submit_track, submit_ns, submit_ns, id, 0, context});
        s.events.push_back({"submit_to_gpu", "submit", "", 'f', track, begin_ns, begin_ns, id, 0, context});
    } else {
        s.events.push_back({"vulkan.clock_order_uncertain", "trace", "ordering_error_ns", 'i', track, begin_ns, begin_ns,
                            id, submit_ns - begin_ns, context});
    }
}

void ggml_trace_flush(void) {
    auto & s = state();
    std::lock_guard<std::mutex> guard(s.mutex);
    if (!s.file || s.closed.exchange(true)) {
        return;
    }
    FILE * file = s.file;
    // Color overlapping GPU envelopes onto lanes before writing complete slices.
    std::vector<trace_event *> gpu_events;
    for (auto & event : s.events) {
        if (event.phase == 'G') { gpu_events.push_back(&event); }
    }
    std::sort(gpu_events.begin(), gpu_events.end(), [](const trace_event * a, const trace_event * b) {
        return a->begin < b->begin;
    });
    std::map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> lanes;
    std::map<uint64_t, uint64_t> submission_tracks;
    for (auto * event : gpu_events) {
        auto & queue_lanes = lanes[event->track];
        size_t lane = 0;
        while (lane < queue_lanes.size() && queue_lanes[lane].first > event->begin) { ++lane; }
        if (lane == queue_lanes.size()) {
            uint64_t track = event->track;
            if (lane > 0) {
                std::string name = "GPU";
                for (const auto & item : s.tracks) { if (item.id == event->track) { name = item.name; break; } }
                track = ggml_trace_next_id();
                s.tracks.push_back({track, name + " / overlap " + std::to_string(lane)});
            }
            queue_lanes.push_back({0, track});
        }
        queue_lanes[lane].first = event->end;
        event->track = queue_lanes[lane].second;
        submission_tracks[event->id] = event->track;
    }
    for (auto & event : s.events) {
        if (event.phase == 'f' && submission_tracks.count(event.id)) { event.track = submission_tracks[event.id]; }
    }
    std::fprintf(file, "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[");
    bool first = true;
    for (const auto & track : s.tracks) {
        std::fprintf(file, "%s{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":%llu,\"args\":{\"name\":",
                     first ? "" : ",", (unsigned long long) track.id);
        write_string(file, track.name);
        std::fputs("}}", file);
        first = false;
    }
    for (const auto & event : s.events) {
        std::fprintf(file, "%s{\"name\":", first ? "" : ",");
        first = false;
        write_string(file, event.name);
        std::fputs(",\"cat\":", file);
        write_string(file, event.category);
        std::fprintf(file, ",\"ph\":\"%c\",\"pid\":1,\"tid\":%llu,\"ts\":%.3f",
                     event.phase == 'G' ? 'X' : event.phase, (unsigned long long) event.track, (event.begin - s.origin) / 1000.0);
        if (event.phase == 'X' || event.phase == 'G') {
            std::fprintf(file, ",\"dur\":%.3f", (event.end - event.begin) / 1000.0);
        }
        if (event.phase == 'b' || event.phase == 'e' || event.phase == 's' || event.phase == 'f') {
            std::fprintf(file, ",\"id\":\"%llu\"", (unsigned long long) event.id);
        }
        if (event.phase == 'f') { std::fputs(",\"bp\":\"e\"", file); }
        if (event.phase == 'i') { std::fputs(",\"s\":\"t\"", file); }
        std::fprintf(file, ",\"args\":{\"round_id\":%llu,\"graph_id\":%llu,\"id\":%llu",
                     (unsigned long long) event.context.round_id, (unsigned long long) event.context.graph_id,
                     (unsigned long long) event.id);
        if (!event.value_name.empty()) {
            std::fputc(',', file);
            write_string(file, event.value_name);
            std::fprintf(file, ":%llu", (unsigned long long) event.value);
        }
        std::fputs("}}", file);
    }
    std::fprintf(file, "],\"ggml_trace\":{\"version\":1,\"events\":%zu,\"dropped_events\":%llu,\"clock\":\"monotonic\"}}\n",
                 s.events.size(), (unsigned long long) s.dropped);
    if (std::fclose(file) != 0) {
        std::fprintf(stderr, "ggml_trace: failed to write trace (errno %d)\n", errno);
    }
}
