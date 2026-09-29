def _issue(code, severity, reason, required_evidence, remediation):
    return {
        "code": code,
        "severity": severity,
        "reason": reason,
        "required_evidence": required_evidence,
        "remediation": remediation,
    }


def diagnose(evidence):
    issues = []
    required = set(evidence.get("required_evidence", []))
    present = set(evidence.get("artifacts", {}))
    missing = sorted(required - present)
    if missing:
        issues.append(_issue(
            "MP_EVIDENCE_MISSING", "error", "required evidence is absent", missing,
            "Generate the missing artifacts in a new run directory and record their SHA-256 hashes.",
        ))

    reference_hash = evidence.get("reference_model_sha256")
    candidate_hash = evidence.get("candidate_model_sha256")
    if reference_hash and candidate_hash and reference_hash != candidate_hash:
        issues.append(_issue(
            "MP_MODEL_IDENTITY_MISMATCH", "error", "model identities differ", ["model_identity"],
            "Use the same immutable model revision for the baseline and optimized candidate.",
        ))

    baseline = evidence.get("baseline") or {}
    candidate = evidence.get("candidate") or {}
    if baseline and candidate and baseline.get("global_batch_size") != candidate.get("global_batch_size"):
        issues.append(_issue(
            "MP_PROTOCOL_MISMATCH", "error", "global batch sizes differ", ["baseline_config", "candidate_config"],
            "Rerun both builds with an identical global batch size and preserve both configs.",
        ))

    accuracy = evidence.get("accuracy") or {}
    if accuracy.get("value") is not None and accuracy.get("minimum") is not None:
        if accuracy["value"] < accuracy["minimum"]:
            issues.append(_issue(
                "MP_ACCURACY_GATE_FAILED", "error", "accuracy is below the declared threshold", ["accuracy_output"],
                "Reject the performance claim and improve quality before benchmarking again.",
            ))

    if baseline and candidate and min(baseline.get("repetitions", 0), candidate.get("repetitions", 0)) < 20:
        issues.append(_issue(
            "MP_REPETITIONS_INSUFFICIENT", "warning", "fewer than 20 measured repetitions", ["baseline_log", "candidate_log"],
            "Run at least 20 post-warmup measurements for each comparable build.",
        ))
    return issues
