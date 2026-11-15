# DSpark asynchronous submission

This experimental path builds on the device DSpark implementation described in [dspark-vulkan.md](dspark-vulkan.md). Set `LLAMA_DSPARK_DEVICE=1`, `LLAMA_DSPARK_DENSE_Q=1`, and `LLAMA_DSPARK_ASYNC=1` before starting the server. The existing model, sampling, and draft-length arguments still apply.

Draft token IDs and proposal probabilities use a persistent backend host buffer. On Vulkan, the host buffer permits asynchronous readback instead of an immediate synchronous staging copy. The server defers collection until the target forward has completed.

The target uploads the anchor token and copies the draft IDs directly from the draft's device tensor. Verification uses a GPU-resident copy of the original token embedding table so embedding lookup also stays on device. The copy preserves the table dtype and bytes; for the tested 4B BF16 table it adds 640 MiB, allocated once at startup. Prefill retains the original embedding path. A backend event orders the producer and consumer. Host draft IDs are placeholders until `common_speculative_draft_finish()` replaces them; the server collects them before context injection, rejection sampling, or slot teardown. The event is recycled after the consumer has finished, before the draft graph is reused.

Startup prints `DSpark asynchronous submission and device token handoff enabled`. The first actual handoff prints `DSpark asynchronous handoff active: draft=3 target=4` for D3. A Vulkan allocation should print `DSpark readback buffer: Vulkan_Host`. The startup message alone does not prove the deferred path was used.

The current path requires a Spark target, one sequence, target and draft on the same GPU, and partial KV removal for both contexts. The anchor and draft block must fit in one target microbatch. Reordered draft outputs are rejected. Callers that do not opt in through `allow_async` collect synchronously. No public `llama.h` ABI is changed; the staging API and common C++ structures require recompilation.

This changes scheduling and storage, while retaining the existing RNG calls and rejection sampler. It does not introduce concurrent independent target and draft forwards. The existing draft context injection is retained. Before sending the final response, the server drains the draft context and updates the decode clock to include the final outstanding update. Disabling `LLAMA_DSPARK_ASYNC` restores synchronous result collection; device DSpark still uses the backend host output buffer.

Measure total decode time. Deferred draft work can move waiting into the target phase, so a lower draft host time alone is not evidence of a speedup. Tests and raw measurements for the first candidate are stored in `../experiments/windows-cpu/windows_core_ultra_7_255h/20261113-165614-vulkan-dspark-async/` relative to the repository root.
