REQUIRED_ARTIFACTS = (
    "environment", "model_identity", "dataset_identity", "fine_tuning",
    "accuracy", "baseline", "optimized", "ledger",
)


def build_report(task, artifacts, decision):
    missing = [name for name in REQUIRED_ARTIFACTS if name not in artifacts]
    accepted = []
    rejected = []
    untested = []
    if decision.get("final_status") == "pass":
        accepted.append("optimized Ascend build passes the declared accuracy-constrained performance gate")
    if artifacts.get("fine_tuning", {}).get("level") == "supported":
        untested.append("fine-tuning improvement has historical support but requires a formal B4 rerun")
    if missing:
        rejected.append("a complete acceptance claim cannot be made while required evidence is missing")
    next_action = (
        "Run one formal B4 fine-tuning and validation job using the frozen v0.4 dataset and config."
        if artifacts.get("fine_tuning", {}).get("level") == "supported"
        else "Freeze the evidence ledger and generate the semifinal submission package."
    )
    return {
        "schema_version": "migrationproof.competition-report.v1",
        "task_id": task.get("task_id"),
        "task": task,
        "environment": artifacts.get("environment"),
        "model_identity": artifacts.get("model_identity"),
        "dataset_identity": artifacts.get("dataset_identity"),
        "fine_tuning": artifacts.get("fine_tuning"),
        "accuracy": artifacts.get("accuracy"),
        "baseline_performance": artifacts.get("baseline"),
        "optimized_performance": artifacts.get("optimized"),
        "evidence_ledger": artifacts.get("ledger", []),
        "missing_evidence": missing,
        "accepted_hypotheses": accepted,
        "rejected_hypotheses": rejected,
        "untested_hypotheses": untested,
        "final_status": decision.get("final_status", "insufficient_evidence"),
        "exact_next_action": next_action,
    }
