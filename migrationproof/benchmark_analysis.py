import math
import re
import statistics


_ITERATION = re.compile(
    r"iteration\s+(\d+)\s*/\s*(\d+).*?elapsed time per iteration \(ms\):\s*"
    r"([0-9.]+).*?global batch size:\s*(\d+)",
    re.IGNORECASE,
)


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def analyze_log(text, warmup_iterations=1):
    rows = [
        {"iteration": int(match.group(1)), "total_iterations": int(match.group(2)),
         "step_time_ms": float(match.group(3)), "global_batch_size": int(match.group(4))}
        for match in _ITERATION.finditer(text)
    ]
    if not rows:
        raise ValueError("no benchmark iterations found")
    totals = {row["total_iterations"] for row in rows}
    batches = {row["global_batch_size"] for row in rows}
    if len(totals) != 1 or len(batches) != 1:
        raise ValueError("benchmark protocol changed within the log")
    measured = rows[warmup_iterations:]
    if not measured:
        raise ValueError("no measured iterations remain after warmup")
    times = [row["step_time_ms"] for row in measured]
    batch = measured[0]["global_batch_size"]
    p50 = statistics.median(times)
    return {
        "warmup_iterations": warmup_iterations,
        "repetitions": len(times),
        "total_iterations": rows[0]["total_iterations"],
        "global_batch_size": batch,
        "step_time_ms_p50": round(p50, 6),
        "step_time_ms_p95": round(_percentile(times, 0.95), 6),
        "step_time_ms_mean": round(statistics.mean(times), 6),
        "step_time_ms_stdev": round(statistics.stdev(times), 6) if len(times) > 1 else 0.0,
        "throughput_samples_per_second": round(batch * 1000.0 / p50, 12),
        "measured_step_times_ms": times,
    }


def compare_runs(baseline, optimized):
    comparable = all(
        baseline.get(key) == optimized.get(key)
        for key in ("warmup_iterations", "repetitions", "total_iterations", "global_batch_size")
    )
    if not comparable:
        return {"comparable": False, "reason": "protocol_mismatch"}
    baseline_p50 = baseline["step_time_ms_p50"]
    optimized_p50 = optimized["step_time_ms_p50"]
    baseline_throughput = baseline["throughput_samples_per_second"]
    optimized_throughput = optimized["throughput_samples_per_second"]
    return {
        "comparable": True,
        "p50_step_time_improvement_percent": round((1.0 - optimized_p50 / baseline_p50) * 100.0, 6),
        "throughput_improvement_percent": round((optimized_throughput / baseline_throughput - 1.0) * 100.0, 6),
    }
