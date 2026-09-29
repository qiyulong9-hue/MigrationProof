import hashlib
import json
from pathlib import Path


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_history(root: Path) -> dict:
    root = Path(root)
    candidates = []
    for path in sorted(root.glob("evidence/**/comparison_v2.json")):
        data = _read_json(path)
        candidates.append({
            "source": str(path),
            "source_sha256": _sha256(path),
            "baseline_run_id": data.get("baseline_run_id"),
            "optimized_run_id": data.get("optimized_run_id"),
            "protocol": data.get("measurement_gate", {}),
            "accuracy_gate": data.get("loss_alignment_gate", {}),
            "performance": data.get("performance", {}),
            "final_status": data.get("overall_status"),
        })
    for path in sorted(root.glob("runs/**/acceptance_report.json")):
        data = _read_json(path)
        baseline = data.get("baseline", {})
        optimized = data.get("optimized", {})
        candidates.append({
            "source": str(path),
            "source_sha256": _sha256(path),
            "baseline_run_id": Path(baseline.get("run_dir", "unknown")).name,
            "optimized_run_id": Path(optimized.get("run_dir", "unknown")).name,
            "protocol": {
                "baseline_warmup": baseline.get("warmup_count"),
                "optimized_warmup": optimized.get("warmup_count"),
                "baseline_measurements": baseline.get("measurement_count"),
                "optimized_measurements": optimized.get("measurement_count"),
                "baseline_batch": baseline.get("global_batch_size"),
                "optimized_batch": optimized.get("global_batch_size"),
            },
            "accuracy_gate": data.get("comparison", {}).get("accuracy_gate"),
            "performance": {
                "baseline": baseline,
                "optimized": optimized,
                "comparison": data.get("comparison", {}),
            },
            "final_status": data.get("final_status"),
        })
    pairs = {(item["baseline_run_id"], item["optimized_run_id"]) for item in candidates}
    status = "single_run" if len(pairs) == 1 else "multiple_distinct_runs"
    if not candidates:
        status = "insufficient"
    return {
        "schema_version": "migrationproof.history_audit.1",
        "history_root": str(root),
        "performance_claim_status": status,
        "candidate_runs": candidates,
        "decision": (
            "Do not merge metrics across run pairs. Use a newly executed B4 formal run "
            "for semifinal claims; retain all B3 pairs as historical evidence."
        ),
    }
