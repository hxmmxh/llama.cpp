# CPU and Vulkan submission timelines

This opt-in tracer writes Chrome Trace JSON, which can be opened in
[Perfetto](https://ui.perfetto.dev). No Perfetto SDK or separate capture service
is required. It targets one request on one Vulkan device, including DSpark D3.
Tracing is disabled unless `GGML_TRACE` names a new output file.

With `GGML_TRACE` unset, no trace file, event buffer, GPU query pool, timestamp
commands or calibration calls are created. The instrumentation still performs
CPU enable checks and trace-context bookkeeping. This is a runtime opt-in
implementation, not a compile-time removal of all instrumentation; zero overhead
relative to a build without these hooks has not been established by a controlled
throughput comparison.

## Capture

Build the normal Vulkan server from this branch. Keep your existing model,
sampling, KV, FA and draft settings. In the PowerShell session that starts it:

```powershell
$env:GGML_TRACE = 'C:\traces\dspark-d3-001.json'
# Optional: start 20 seconds after recorder initialization and capture 10 seconds.
$env:GGML_TRACE_START_MS = '20000'
$env:GGML_TRACE_DURATION_MS = '10000'
# Default: 200000 events. Increase only if the trace reports dropped events.
$env:GGML_TRACE_MAX_EVENTS = '200000'
```

The parent directory must exist. Existing files are refused. The start delay is
relative to recorder initialization, not the start of decode. For an initial
short correctness capture, omit the delay and duration. After warmup, issue a
short request with your existing D3 configuration, then stop the server normally
with Ctrl+C. The file is written at normal process exit, after context cleanup.
An embedded caller can call `ggml_trace_flush()` after freeing its contexts and
backends. A forced process termination cannot produce a complete trace.

Use a separate process with all `GGML_TRACE*` variables removed for the clean
reference. Also remove `GGML_VK_PERF_LOGGER`, `GGML_VK_PERF_LOGGER_CONCURRENT`,
`GGML_VK_OP_PROFILE`, `GGML_VK_SERIALIZE_SUBMISSIONS` and device validation flags;
do not use a build with `GGML_VULKAN_CHECK_RESULTS`. Those diagnostic modes can
change the scheduling being measured. Preserve any compatibility setting needed
by your actual device, such as the existing cooperative-matrix setting.

## Read the timeline

CPU tracks are logical application tracks. They show wall intervals, including
time when a thread was descheduled; they are not OS CPU-running samples.

| Event | Meaning |
| --- | --- |
| `server.iteration` | One server scheduling iteration, including idle/prefill iterations; `round_id` identifies the iteration, not necessarily a completed speculative round |
| `server.draft`, `dspark.draft` | Draft preparation, backbone and device Markov sampling, followed by required readback and CPU probability handling |
| `server.target_batch` | Target batch execution and its existing completion wait; inspect `tokens` and the actual request phase to distinguish prefill from verification |
| `server.draft_process` | Target feature handoff and draft context processing |
| `dspark.rejection_sampling`, `server.accept_and_rollback` | CPU verification, acceptance and rollback bookkeeping |
| `llama.ubatch` | One graph execution instance; `graph_id` changes even when its graph structure is reused |
| `llama.graph_reused`, `llama.build_graph`, `llama.allocate_graph`, `llama.set_inputs` | Graph reuse, rebuild, allocation and input preparation |
| `sched.compute_splits`, backend-named `split` slices | Backend placement and split execution |
| `backend.get_async`, `backend.set_async`, `sched.copy_input`, `dspark.copy_dense_q` | Enqueue/copy wall cost; `bytes` records payload size, not completion time |
| `vulkan.record_graph` | Host graph recording, including nested submits and tracing housekeeping |
| `vulkan.submit`, `vulkan.driver_submit` | Submission preparation and actual Vulkan driver call; the outer scope also contains existing queue-lock costs |
| `vulkan.synchronize`, `vulkan.wait_for_fence` | Existing completion handling and wait intervals; nested trace collection is visible separately |
| `vulkan.clock_calibration`, `vulkan.collect_timestamps` | Tracing overhead |

GPU slices are **command-buffer submission envelopes**: a TOP_OF_PIPE timestamp
at recording start and a BOTTOM_OF_PIPE timestamp at recording end. They can
include semaphore waits, barriers and multiple dispatches. They are not individual
shader times, engine occupancy or proof that the GPU is busy throughout the
interval. Overlapping envelopes are put on extra lanes at export time. `submit_to_gpu`
flow IDs link a host submission to its GPU envelope. GPU lanes are logical tracks named after the device and queue. Clock-order
uncertainty is marked explicitly and suppresses the affected flow arrow.

Use `round_id`, `graph_id` and the submission `id` to connect events. A zero ID
means that no enclosing server iteration/graph was active, for example model
loading or post-graph output handling. CPU scopes are nested; do not add parent
and child durations. Likewise, calculate GPU interval unions rather than adding
overlapping envelopes when examining coverage.

Look first for gaps between consecutive GPU envelopes that coincide with CPU
graph preparation, dense-q copying or acceptance work. A long CPU fence wait
while GPU work continues is not itself a removable scheduling bubble. A long
submission envelope needs additional analysis to distinguish useful work from
device-side dependencies or stalls. GPUView can provide complementary system
queue, residency and preemption evidence.

## Clock and lifetime contract

The device must expose `VK_KHR_calibrated_timestamps` or
`VK_EXT_calibrated_timestamps`, a device time domain and the platform host domain:
QPC on Windows, `CLOCK_MONOTONIC` on POSIX. QPC values are converted using its
actual frequency. Devices lacking these capabilities produce CPU-only traces
and an explicit diagnostic; GPU alignment is never inferred from a host wait.

Calibration is refreshed when starting a command buffer after 500 ms. Each
submission retains its calibration anchor and reported
`calibration_max_deviation_ns`. This describes the calibration sample's bound;
it does not bound subsequent clock drift or timestamp-stage placement. Avoid
interpreting gaps near that uncertainty as precise scheduling costs.

In the 2026-11-14 Arc 140T validation (driver 32.0.101.6313), the short D3
capture passed numerical/output and Perfetto import checks, but some mapped GPU
starts preceded their CPU submit markers by up to 0.536 ms. The calibration API
reported a maximum deviation of only 42.4 us. These ordering conflicts are
explicit diagnostics with omitted flows; they are evidence of alignment limits,
not real GPU execution before submission. This device needs further clock
alignment validation before attributing sub-millisecond scheduling gaps.

The tracer uses a fixed 512-slot query pool. A slot is reused only after both
query results are available, using reads **without** `WAIT_BIT`. No tracing fence,
queue idle call, semaphore dependency or per-operator memory barrier is added.
Results are collected after existing completion handling, or when all slots are
in use; ordinary command-buffer recording does not poll every outstanding query.
Timestamp commands still have a cost, so compare clean and observed processes.
Query exhaustion drops submissions instead of stalling inference. Queue-family
`timestampValidBits` is used for modular arithmetic; intervals that cannot be
unambiguously decoded are rejected. Unsupported transfer-queue timestamps and
unsubmitted command buffers are reported as incomplete coverage.

CPU scopes are selected when they begin. GPU batches are selected when command
buffer recording begins. Their end and flow events may extend past the requested
window so that selected intervals remain complete. The in-memory event cap is
bounded; `ggml_trace.dropped_events` and `vulkan.*` diagnostics must be checked
before treating a capture as complete. Memory allocation, locking, calibration
and query polling contribute observer overhead. This tracer does not establish
a performance improvement by itself.

## Validation

```powershell
cmake --build build-vulkan --config Release --target test-trace
python tests/test-trace.py build-vulkan/bin/test-trace.exe CPU
python tests/test-trace.py build-vulkan/bin/test-trace.exe Vulkan0
```

The test uses 600 graph executions with changing inputs, compares every output
against an independent reference, and compares clean/observed output signatures.
It also checks trace JSON, round/graph context, GPU query reuse, event limits,
string escaping, delayed capture and preservation of existing output files.
Vulkan validation requires actual calibrated GPU events; CPU-only fallback is
not a passing GPU result. Follow with a short, same-input D3 off/on model check
and retain tokens, proposed/accepted counts, process exits and raw traces.
