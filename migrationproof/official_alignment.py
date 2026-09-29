import hashlib
import math
import re
import statistics
from pathlib import Path


ITERATION_RE = re.compile(
    r"iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*"
    r"([0-9.]+).*?global batch size:\s*(\d+).*?loss:\s*([0-9.Ee+\-]+|nan|inf)"
    r".*?grad norm:\s*([0-9.Ee+\-]+|nan|inf)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_scalar(text: str, key: str):
    match = re.search(rf"(?:^|\n)\s*(?:training\.)?{re.escape(key)}:\s*([^\s#]+)", text)
    return match.group(1) if match else None


def parse_training_log(path: Path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for match in ITERATION_RE.finditer(text):
        step, total, step_ms, gbs, loss, grad = match.groups()
        row = {
            "step": int(step), "total": int(total), "step_time_ms": float(step_ms),
            "global_batch_size": int(gbs), "loss": float(loss), "grad_norm": float(grad),
        }
        if not all(math.isfinite(row[key]) for key in ("step_time_ms", "loss", "grad_norm")):
            raise ValueError(f"non-finite training value at iteration {row['step']}")
        rows.append(row)
    if not rows:
        raise ValueError("no training iterations found")
    return {
        "path": str(path), "sha256": _sha256(path), "rows": rows,
        "protocol": {
            "global_batch_size": rows[0]["global_batch_size"],
            "micro_batch_size": _extract_scalar(text, "micro_batch_size"),
            "gradient_accumulation_steps": _extract_scalar(text, "gradient_accumulation_steps"),
            "seed": _extract_scalar(text, "seed"), "lr": _extract_scalar(text, "lr"),
            "cutoff_len": _extract_scalar(text, "cutoff_len"),
        },
    }


def _pearson(left, right):
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else (1.0 if left == right else 0.0)


def compare_accuracy_logs(ascendc_path, triton_path, *, expected_iterations=100,
                          last_window=10, max_last_mean_rel=0.01,
                          max_step_rel=0.02, min_pearson=0.99) -> dict:
    ascendc = parse_training_log(Path(ascendc_path))
    triton = parse_training_log(Path(triton_path))
    left, right = ascendc["rows"], triton["rows"]
    reasons = []
    if len(left) != expected_iterations or len(right) != expected_iterations:
        reasons.append("iteration_count_mismatch")
    if [row["step"] for row in left] != [row["step"] for row in right]:
        reasons.append("iteration_identity_mismatch")
    if ascendc["protocol"]["global_batch_size"] != triton["protocol"]["global_batch_size"]:
        reasons.append("global_batch_size_mismatch")
    controlled = ("micro_batch_size", "gradient_accumulation_steps", "seed", "lr", "cutoff_len")
    mismatches = [key for key in controlled if ascendc["protocol"].get(key) != triton["protocol"].get(key)]
    reasons.extend(f"{key}_mismatch" for key in mismatches)
    pair_count = min(len(left), len(right))
    left_losses = [row["loss"] for row in left[:pair_count]]
    right_losses = [row["loss"] for row in right[:pair_count]]
    rel_diffs = [abs(a - b) / max(abs(a), 1e-12) for a, b in zip(left_losses, right_losses)]
    tail = min(last_window, pair_count)
    left_tail = statistics.mean(left_losses[-tail:])
    right_tail = statistics.mean(right_losses[-tail:])
    tail_rel = abs(left_tail - right_tail) / max(abs(left_tail), 1e-12)
    pearson = _pearson(left_losses, right_losses)
    if max(rel_diffs, default=float("inf")) > max_step_rel:
        reasons.append("max_step_loss_difference_exceeded")
    if tail_rel > max_last_mean_rel:
        reasons.append("last_window_loss_difference_exceeded")
    if pearson < min_pearson:
        reasons.append("loss_curve_correlation_below_gate")
    return {
        "schema_version": "migrationproof.official-alignment.v1",
        "sources": {"ascendc": {k: v for k, v in ascendc.items() if k != "rows"},
                    "triton": {k: v for k, v in triton.items() if k != "rows"}},
        "protocol": {"comparable": not any(reason.endswith("_mismatch") for reason in reasons),
                     "expected_iterations": expected_iterations},
        "metrics": {"max_step_loss_relative_difference": max(rel_diffs, default=None),
                    "last_window_loss_relative_difference": tail_rel,
                    "loss_curve_pearson": pearson,
                    "ascendc_last_window_mean_loss": left_tail,
                    "triton_last_window_mean_loss": right_tail},
        "gate": {"passed": not reasons, "reasons": reasons,
                 "thresholds": {"max_step_relative_difference": max_step_rel,
                                "max_last_window_mean_relative_difference": max_last_mean_rel,
                                "minimum_pearson": min_pearson}},
    }

