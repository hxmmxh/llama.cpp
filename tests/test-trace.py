"""Check real recorder output and independent graph results in separate processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable = str(Path(sys.argv[1]).resolve())
    device = sys.argv[2] if len(sys.argv) > 2 else "CPU"
    with tempfile.TemporaryDirectory(prefix="ggml-trace-") as directory:
        root = Path(directory)
        environment = {k: v for k, v in os.environ.items() if not k.startswith("GGML_TRACE")}

        def run(extra):
            return subprocess.run([executable, device, "600"], env={**environment, **extra},
                                  capture_output=True, text=True, check=True)

        clean = run({})
        path = root / "trace.json"
        observed = run({"GGML_TRACE": str(path)})
        assert clean.stdout == observed.stdout, "tracing changed graph output"
        trace = json.loads(path.read_text())
        assert trace["ggml_trace"]["dropped_events"] == 0
        events = trace["traceEvents"]
        assert any(e["name"] == 'escaped "label"\n\\' for e in events)
        rounds = [e for e in events if e["name"] == "probe.round"]
        assert len(rounds) == 600
        assert all(e["args"]["round_id"] and e["args"]["graph_id"] and e["dur"] >= 0 for e in rounds)
        starts = {str(e["args"]["id"]): e for e in events if e.get("cat") == "gpu"}
        for start in starts.values():
            assert start["dur"] >= 0
            assert "calibration_max_deviation_ns" in start["args"]
        if device != "CPU":
            assert len(starts) >= 600, "GPU timeline missing; inspect calibrated timestamp support"
            flows = {e["id"] for e in events if e["ph"] == "s"}
            uncertain = {str(e["args"]["id"]) for e in events if e["name"] == "vulkan.clock_order_uncertain"}
            assert starts.keys() <= flows | uncertain
        limited = root / "limited.json"
        run({"GGML_TRACE": str(limited), "GGML_TRACE_MAX_EVENTS": "2"})
        limited_trace = json.loads(limited.read_text())
        assert limited_trace["ggml_trace"]["dropped_events"] > 0
        assert limited_trace["ggml_trace"]["events"] == 2
        existing = root / "existing.json"
        existing.write_text("existing evidence")
        refused = run({"GGML_TRACE": str(existing)})
        assert existing.read_text() == "existing evidence"
        assert "tracing disabled" in refused.stderr
        assert refused.stdout == clean.stdout
        delayed = root / "delayed.json"
        run({"GGML_TRACE": str(delayed), "GGML_TRACE_START_MS": "86400000"})
        assert not any(e["ph"] == "X" for e in json.loads(delayed.read_text())["traceEvents"])
        print(f"PASS: {device}; unchanged output; 600 rounds; {len(starts)} GPU submissions; capacity, escaping, context and file checks")


if __name__ == "__main__":
    main()
