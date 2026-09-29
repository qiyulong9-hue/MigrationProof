#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from migrationproof.benchmark_analysis import analyze_log


ITERATION = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iteration_window(text):
    rows = [(int(m.group(1)), int(m.group(2))) for m in ITERATION.finditer(text)]
    if not rows:
        raise ValueError("no benchmark iterations found")
    return {
        "first_logged_iteration": rows[0][0],
        "last_logged_iteration": rows[-1][0],
        "logged_iteration_count": len(rows),
        "configured_total_iterations": rows[0][1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_log", type=Path)
    parser.add_argument("optimized_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline_text = args.baseline_log.read_text(encoding="utf-8", errors="replace")
    optimized_text = args.optimized_log.read_text(encoding="utf-8", errors="replace")
    baseline = analyze_log(baseline_text)
    optimized = analyze_log(optimized_text)
    baseline_window = iteration_window(baseline_text)
    optimized_window = iteration_window(optimized_text)
    comparable = all(
        baseline.get(key) == optimized.get(key)
        for key in ("warmup_iterations", "repetitions", "global_batch_size")
    ) and baseline_window["logged_iteration_count"] == optimized_window["logged_iteration_count"]
    comparison = {"comparable": comparable}
    if comparable:
        bp50 = baseline["step_time_ms_p50"]
        op50 = optimized["step_time_ms_p50"]
        comparison.update({
            "comparison_basis": "equal newly executed iteration windows after checkpoint restore",
            "p50_step_time_improvement_percent": round((1.0 - op50 / bp50) * 100.0, 6),
            "throughput_improvement_percent": round(
                (optimized["throughput_samples_per_second"] / baseline["throughput_samples_per_second"] - 1.0) * 100.0,
                6,
            ),
        })
    else:
        comparison["reason"] = "measured_window_mismatch"
    document = {
        "schema_version": "migrationproof.performance-comparison.resumed.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"warmup_iterations": 1, "measured_iterations": 20, "effective_batch_size": 16},
        "baseline_iteration_window": baseline_window,
        "optimized_iteration_window": optimized_window,
        "baseline": baseline,
        "optimized": optimized,
        "comparison": comparison,
        "source_evidence": {
            "baseline_log": str(args.baseline_log),
            "baseline_log_sha256": file_sha256(args.baseline_log),
            "optimized_log": str(args.optimized_log),
            "optimized_log_sha256": file_sha256(args.optimized_log),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.open("x", encoding="utf-8", newline="\n").write(json.dumps(document, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
