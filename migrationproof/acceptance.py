def decide(task, accuracy, baseline, candidate, evidence):
    required_repetitions = task["performance_gate"]["repetitions"]
    if not evidence.get("complete"):
        return {"final_status": "insufficient_evidence", "reason": "evidence_missing"}
    if baseline.get("global_batch_size") != candidate.get("global_batch_size"):
        return {"final_status": "insufficient_evidence", "reason": "batch_mismatch"}
    if min(baseline.get("repetitions", 0), candidate.get("repetitions", 0)) < required_repetitions:
        return {"final_status": "insufficient_evidence", "reason": "repetitions_insufficient"}

    metric = task["accuracy_gate"]["metric"]
    if accuracy.get(metric) is None:
        return {"final_status": "insufficient_evidence", "reason": "accuracy_missing"}
    if accuracy[metric] < task["accuracy_gate"]["minimum"]:
        return {"final_status": "fail", "reason": "accuracy_gate_failed"}

    primary = task["performance_gate"]["primary_metric"]
    old = baseline.get(primary)
    new = candidate.get(primary)
    if old is None or new is None or old <= 0:
        return {"final_status": "insufficient_evidence", "reason": "performance_missing"}
    if primary == "throughput":
        improvement = (new / old - 1.0) * 100.0
    else:
        improvement = (1.0 - new / old) * 100.0
    threshold = task["performance_gate"]["minimum_improvement_percent"]
    status = "pass" if improvement >= threshold else "conditional_pass"
    return {
        "final_status": status,
        "reason": "all_gates_passed" if status == "pass" else "performance_gate_not_met",
        "performance_improvement_percent": improvement,
    }
