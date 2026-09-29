def evaluate_evidence(evidence):
    if "environment" not in evidence:
        return "insufficient_evidence", ["missing_environment"]
    for field, issue in (("model_sha256", "model_identity_missing"),
                         ("dataset_sha256", "dataset_identity_missing")):
        if field not in evidence:
            return "insufficient_evidence", [issue]
        value = str(evidence[field])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
            return "insufficient_evidence", ["invalid_sha256"]
    baseline = evidence.get("baseline", {})
    candidate = evidence.get("candidate", {})
    if "NVIDIA" in str(baseline.get("device", "")):
        return "insufficient_evidence", ["wrong_primary_baseline"]
    if baseline.get("global_batch_size") != candidate.get("global_batch_size"):
        return "insufficient_evidence", ["unfair_comparison"]
    if min(baseline.get("repetitions", 0), candidate.get("repetitions", 0)) < 20:
        return "insufficient_evidence", ["insufficient_repetitions"]
    smoke = str(evidence.get("smoke", "")).lower()
    if "failed" in smoke or "exception" in smoke:
        return "fail", ["smoke_failed"]
    accuracy = evidence.get("accuracy", {})
    if accuracy.get("value", 0) < accuracy.get("minimum", 0):
        return "fail", ["accuracy_gate_failed"]
    performance = evidence.get("performance", {})
    if performance.get("improvement_percent", 0) < performance.get("minimum_percent", 0):
        return "fail", ["performance_gate_failed"]
    if evidence.get("application_accuracy_pending"):
        return "conditional_pass", ["missing_application_accuracy"]
    return "pass", []


def evaluate_corpus(cases):
    mismatches = []
    decision_hits = 0
    issue_hits = 0
    for case in cases:
        status, issues = evaluate_evidence(case["evidence"])
        expected = case["expected"]
        decision_hits += int(status == expected["status"])
        issue_hits += int(issues == expected["issues"])
        if status != expected["status"] or issues != expected["issues"]:
            mismatches.append({"id": case["id"], "expected": expected,
                               "actual": {"status": status, "issues": issues}})
    count = len(cases)
    return {
        "sample_count": count,
        "decision_accuracy": decision_hits / count if count else 0.0,
        "exact_issue_match": issue_hits / count if count else 0.0,
        "mismatches": mismatches,
    }
