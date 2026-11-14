#pragma once

#include "ggml-vulkan-types.h"
#include "ggml-trace.h"

const char * ggml_vk_trace_extension(const std::vector<vk::ExtensionProperties> & extensions);
void ggml_vk_trace_init(vk_device & device, vk::Instance instance, const char * extension, bool host_query_reset);
void ggml_vk_trace_begin(vk_device & device, vk_queue & queue, vk_command_buffer & buffer);
void ggml_vk_trace_end(vk_device & device, vk_command_buffer & buffer);
void ggml_vk_trace_submit(vk_device & device, vk_command_buffer & buffer, uint64_t begin_ns, uint64_t end_ns);
void ggml_vk_trace_collect(vk_device_struct & device);
void ggml_vk_trace_destroy(vk_device_struct & device);
