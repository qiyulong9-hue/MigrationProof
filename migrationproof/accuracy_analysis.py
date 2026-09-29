def score_predictions(expected, actual):
    expected_by_id = {record["id"]: record for record in expected}
    actual_by_id = {record["id"]: record for record in actual}
    if set(expected_by_id) != set(actual_by_id):
        raise ValueError("prediction IDs do not match reference IDs")

    labels = sorted({
        label
        for record in expected + actual
        for label in record.get("issues", [])
    })
    true_positive = {label: 0 for label in labels}
    false_positive = {label: 0 for label in labels}
    false_negative = {label: 0 for label in labels}
    decision_hits = 0
    unsafe_total = 0
    unsafe_passes = 0

    for record_id, reference in expected_by_id.items():
        prediction = actual_by_id[record_id]
        expected_status = reference.get("status", reference.get("decision"))
        actual_status = prediction.get("status", prediction.get("decision"))
        decision_hits += int(expected_status == actual_status)
        expected_issues = set(reference.get("issues", []))
        actual_issues = set(prediction.get("issues", []))
        for label in labels:
            true_positive[label] += int(label in expected_issues and label in actual_issues)
            false_positive[label] += int(label not in expected_issues and label in actual_issues)
            false_negative[label] += int(label in expected_issues and label not in actual_issues)
        if expected_status != "pass":
            unsafe_total += 1
            unsafe_passes += int(actual_status == "pass")

    issue_scores = []
    for label in labels:
        denominator = 2 * true_positive[label] + false_positive[label] + false_negative[label]
        issue_scores.append(2 * true_positive[label] / denominator if denominator else 1.0)
    count = len(expected)
    return {
        "decision_accuracy": decision_hits / count if count else 0.0,
        "issue_macro_f1": sum(issue_scores) / len(issue_scores) if issue_scores else 1.0,
        "unsafe_pass_rate": unsafe_passes / unsafe_total if unsafe_total else 0.0,
        "sample_count": count,
    }
