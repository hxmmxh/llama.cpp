#include "ggml-vulkan-trace.h"

#include <cstring>

namespace {
constexpr uint32_t trace_slot_count = 512;

struct trace_slot {
    bool used = false, submitted = false;
    uint64_t id = 0, track = 0;
    ggml_trace_context context {};
    uint64_t device_ticks = 0, host_ns = 0, error_ns = 0, recorded_ns = 0;
    uint32_t valid_bits = 0;
    bool transfer = false;
    uint64_t submit_ns = 0, submit_track = 0;
};

static long double timestamp_delta(uint64_t timestamp, uint64_t reference, uint32_t bits) {
    const uint64_t mask = bits == 64 ? UINT64_MAX : (UINT64_C(1) << bits) - 1;
    const uint64_t delta = (timestamp - reference) & mask;
    if (delta & (UINT64_C(1) << (bits - 1))) {
        return -(long double) ((reference - timestamp) & mask);
    }
    return (long double) delta;
}
}

struct vk_trace_state {
    std::mutex mutex;
    vk::QueryPool pool;
    PFN_vkGetCalibratedTimestampsEXT calibrate = nullptr;
    VkTimeDomainEXT host_domain;
    uint64_t device_ticks = 0, host_ns = 0, error_ns = 0, last_calibration = 0;
    uint64_t dropped = 0;
    bool calibrated = false;
    uint32_t compute_bits = 0, transfer_bits = 0;
    uint64_t compute_track = 0, transfer_track = 0;
    trace_slot slots[trace_slot_count];
};

const char * ggml_vk_trace_extension(const std::vector<vk::ExtensionProperties> & extensions) {
    if (!ggml_trace_enabled()) {
        return nullptr;
    }
    for (const char * name : {"VK_KHR_calibrated_timestamps", "VK_EXT_calibrated_timestamps"}) {
        for (const auto & extension : extensions) {
            if (std::strcmp(extension.extensionName, name) == 0) {
                return name;
            }
        }
    }
    return nullptr;
}

static bool trace_calibrate(vk_device_struct & device, vk_trace_state & trace) {
    ggml_trace_scope scope("vulkan.clock_calibration", "trace");
    VkCalibratedTimestampInfoEXT infos[2] = {};
    infos[0].sType = infos[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
    infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
    infos[1].timeDomain = trace.host_domain;
    uint64_t best_error = UINT64_MAX, best[2] = {};
    for (int attempt = 0; attempt < 3; ++attempt) {
        uint64_t values[2], error;
        if (trace.calibrate(device.device, 2, infos, values, &error) == VK_SUCCESS && error < best_error) {
            best_error = error;
            best[0] = values[0];
            best[1] = values[1];
        }
    }
    trace.calibrated = best_error != UINT64_MAX;
    trace.last_calibration = ggml_trace_time_ns();
    if (trace.calibrated) {
        trace.device_ticks = best[0];
        trace.host_ns = ggml_trace_host_time_ns(best[1]);
        trace.error_ns = best_error;
        ggml_trace_diagnostic("vulkan.clock_calibrated", "max_deviation_ns", best_error);
    } else {
        ggml_trace_diagnostic("vulkan.clock_calibration_failed", nullptr, 0);
    }
    return trace.calibrated;
}

void ggml_vk_trace_init(vk_device & device, vk::Instance instance, const char * extension, bool host_query_reset) {
    if (!ggml_trace_enabled()) {
        return;
    }
    if (!host_query_reset) {
        GGML_LOG_WARN("ggml_trace: %s has no host query reset; CPU trace only\n", device->name.c_str());
        ggml_trace_diagnostic("vulkan.gpu_trace_unavailable", "reason", 6);
        return;
    }
    if (!extension) {
        GGML_LOG_WARN("ggml_trace: %s has no calibrated timestamps; CPU trace only\n", device->name.c_str());
        ggml_trace_diagnostic("vulkan.gpu_trace_unavailable", "reason", 1);
        return;
    }
    const bool khr = std::strcmp(extension, "VK_KHR_calibrated_timestamps") == 0;
    auto get_domains = reinterpret_cast<PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT>(
        instance.getProcAddr(khr ? "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR" : "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT"));
    auto calibrate = reinterpret_cast<PFN_vkGetCalibratedTimestampsEXT>(device->device.getProcAddr(
        khr ? "vkGetCalibratedTimestampsKHR" : "vkGetCalibratedTimestampsEXT"));
    uint32_t count = 0;
    if (!get_domains || !calibrate || get_domains(device->physical_device, &count, nullptr) != VK_SUCCESS || !count) {
        ggml_trace_diagnostic("vulkan.gpu_trace_unavailable", "reason", 2);
        return;
    }
    std::vector<VkTimeDomainEXT> domains(count);
    if (get_domains(device->physical_device, &count, domains.data()) != VK_SUCCESS) {
        ggml_trace_diagnostic("vulkan.gpu_trace_unavailable", "reason", 3);
        return;
    }
#ifdef _WIN32
    const VkTimeDomainEXT host_domain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT;
#else
    const VkTimeDomainEXT host_domain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT;
#endif
    if (std::find(domains.begin(), domains.end(), host_domain) == domains.end() ||
        std::find(domains.begin(), domains.end(), VK_TIME_DOMAIN_DEVICE_EXT) == domains.end()) {
        ggml_trace_diagnostic("vulkan.gpu_trace_unavailable", "reason", 4);
        return;
    }
    auto trace = std::make_shared<vk_trace_state>();
    trace->calibrate = calibrate;
    trace->host_domain = host_domain;
    const auto families = device->physical_device.getQueueFamilyProperties();
    trace->compute_bits = families[device->compute_queue->queue_family_index].timestampValidBits;
    trace->transfer_bits = families[device->transfer_queue->queue_family_index].timestampValidBits;
    if (!trace->compute_bits || !trace_calibrate(*device, *trace)) {
        ggml_trace_diagnostic("vulkan.gpu_trace_unavailable", "reason", 5);
        return;
    }
    trace->pool = device->device.createQueryPool({{}, vk::QueryType::eTimestamp, 2 * trace_slot_count});
    trace->compute_track = ggml_trace_new_track((device->name + " / compute submission envelopes").c_str());
    trace->transfer_track = device->transfer_queue->handle == device->compute_queue->handle ? trace->compute_track :
        ggml_trace_new_track((device->name + " / transfer submission envelopes").c_str());
    device->trace = trace;
    GGML_LOG_INFO("ggml_trace: calibrated GPU timeline on %s (timestamp bits %u/%u)\n",
                  device->name.c_str(), trace->compute_bits, trace->transfer_bits);
}

static void trace_collect(vk_device_struct & device, vk_trace_state & trace) {
    for (uint32_t index = 0; index < trace_slot_count; ++index) {
        auto & slot = trace.slots[index];
        if (!slot.submitted) {
            continue;
        }
        // Each query returns a timestamp followed by availability. Never wait here.
        uint64_t result[4] = {};
        const vk::Result status = device.device.getQueryPoolResults(trace.pool, 2 * index, 2,
            sizeof(result), result, 2 * sizeof(uint64_t),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (status != vk::Result::eSuccess || !result[1] || !result[3]) {
            continue;
        }
        const long double period = device.properties.limits.timestampPeriod;
        const long double start = slot.host_ns + timestamp_delta(result[0], slot.device_ticks, slot.valid_bits) * period;
        const long double duration = timestamp_delta(result[2], result[0], slot.valid_bits) * period;
        const long double wrap_half_ns = std::ldexp(1.0L, slot.valid_bits - 1) * period;
        const uint64_t now = ggml_trace_time_ns();
        if (start < 0 || start > UINT64_MAX || duration < 0 || start + duration > UINT64_MAX ||
            now - slot.recorded_ns >= wrap_half_ns || now - slot.host_ns >= wrap_half_ns) {
            ++trace.dropped;
            ggml_trace_diagnostic("vulkan.invalid_timestamp_interval", "submit_id", slot.id);
        } else {
            const uint64_t begin_ns = (uint64_t) start;
            ggml_trace_gpu(slot.transfer ? "Vulkan transfer" : "Vulkan compute", slot.track,
                begin_ns, begin_ns + (uint64_t) duration, slot.context, slot.id, slot.error_ns,
                slot.submit_ns, slot.submit_track);
        }
        slot = {};
    }
}

void ggml_vk_trace_collect(vk_device_struct & device) {
    if (device.trace) {
        ggml_trace_scope scope("vulkan.collect_timestamps", "trace");
        std::lock_guard<std::mutex> guard(device.trace->mutex);
        trace_collect(device, *device.trace);
    }
}

void ggml_vk_trace_begin(vk_device & device, vk_queue & queue, vk_command_buffer & buffer) {
    buffer.trace_slot = -1;
    if (!device->trace || !ggml_trace_active()) {
        return;
    }
    auto & trace = *device->trace;
    std::lock_guard<std::mutex> guard(trace.mutex);
    const uint64_t now = ggml_trace_time_ns();
    if (now - trace.last_calibration > 500000000) {
        trace_calibrate(*device, trace);
    }
    const uint32_t bits = queue.transfer_only ? trace.transfer_bits : trace.compute_bits;
    if (!trace.calibrated || !bits) {
        ++trace.dropped;
        ggml_trace_diagnostic("vulkan.gpu_submission_untraced", nullptr, 0);
        return;
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t index = 0; index < trace_slot_count; ++index) {
            auto & slot = trace.slots[index];
            if (slot.used) {
                continue;
            }
            slot.used = true;
            slot.id = ggml_trace_next_id();
            slot.context = ggml_trace_get_context();
            slot.device_ticks = trace.device_ticks;
            slot.host_ns = trace.host_ns;
            slot.error_ns = trace.error_ns;
            slot.recorded_ns = now;
            slot.valid_bits = bits;
            slot.transfer = queue.transfer_only;
            slot.track = queue.transfer_only ? trace.transfer_track : trace.compute_track;
            // Host reset prevents stale availability before this command buffer executes.
            device->device.resetQueryPool(trace.pool, 2 * index, 2);
            buffer.buf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, trace.pool, 2 * index);
            buffer.trace_slot = (int32_t) index;
            return;
        }
        // Normal recovery follows an existing synchronize. Poll here only when full.
        if (pass == 0) { trace_collect(*device, trace); }
    }
    ++trace.dropped;
    ggml_trace_diagnostic("vulkan.query_capacity_exhausted", nullptr, 0);
}

void ggml_vk_trace_end(vk_device & device, vk_command_buffer & buffer) {
    if (buffer.trace_slot >= 0) {
        buffer.buf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe,
            device->trace->pool, 2 * (uint32_t) buffer.trace_slot + 1);
    }
}

void ggml_vk_trace_submit(vk_device & device, vk_command_buffer & buffer, uint64_t begin_ns, uint64_t end_ns) {
    if (buffer.trace_slot >= 0) {
        auto & trace = *device->trace;
        std::lock_guard<std::mutex> guard(trace.mutex);
        auto & slot = trace.slots[buffer.trace_slot];
        slot.submitted = true;
        slot.submit_ns = begin_ns;
        slot.submit_track = ggml_trace_cpu_track();
        ggml_trace_submission(begin_ns, end_ns, slot.context, slot.id);
        buffer.trace_slot = -1;
    }
}

void ggml_vk_trace_destroy(vk_device_struct & device) {
    if (!device.trace) {
        return;
    }
    ggml_vk_trace_collect(device);
    auto & trace = *device.trace;
    uint64_t pending = 0;
    for (const auto & slot : trace.slots) {
        pending += slot.used;
    }
    if (pending || trace.dropped) {
        GGML_LOG_WARN("ggml_trace: incomplete GPU trace: %llu pending, %llu dropped\n",
                      (unsigned long long) pending, (unsigned long long) trace.dropped);
        ggml_trace_diagnostic("vulkan.pending_at_teardown", "count", pending);
        ggml_trace_diagnostic("vulkan.dropped_submissions", "count", trace.dropped);
    }
    device.device.destroyQueryPool(trace.pool);
    device.trace.reset();
}
