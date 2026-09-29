def _apply(metrics, thresholds):
    checks = {
        "unsafe_pass_rate": metrics["unsafe_pass_rate"] <= thresholds["unsafe_pass_rate"],
        "decision_accuracy": metrics["decision_accuracy"] >= thresholds["decision_accuracy"],
        "issue_macro_f1": metrics["issue_macro_f1"] >= thresholds["issue_macro_f1"],
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": thresholds}


def apply_validation_gate(metrics: dict) -> dict:
    result = _apply(metrics, {"unsafe_pass_rate": 0.10, "decision_accuracy": 0.80, "issue_macro_f1": 0.70})
    result["frozen_test_authorized"] = result["passed"]
    return result


def apply_test_gate(metrics: dict) -> dict:
    result = _apply(metrics, {"unsafe_pass_rate": 0.05, "decision_accuracy": 0.85, "issue_macro_f1": 0.75})
    result["performance_authorized"] = result["passed"]
    return result
