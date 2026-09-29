import json
import math
import statistics


def build_frozen_split(test_cases, seed=42, train_size=200, validation_size=40):
    test_ids = [case["id"] for case in test_cases]
    return {
        "seed": seed,
        "train": [f"MPBTR-{seed}-{index:04d}" for index in range(1, train_size + 1)],
        "validation": [f"MPBVA-{seed}-{index:04d}" for index in range(1, validation_size + 1)],
        "test": test_ids,
    }


def build_sft_records(split, count, seed=42):
    scenarios = (
        ("pass", [], "All required evidence is present and both gates pass."),
        ("conditional_pass", ["missing_application_accuracy"],
         "Core migration evidence passes, but the non-core explanation review is pending."),
        ("fail", ["accuracy_gate_failed"],
         "The declared accuracy is below its threshold; rerun evaluation after correction."),
        ("insufficient_evidence", ["unfair_comparison"],
         "Baseline and optimized global batch sizes differ; rerun with identical controls."),
        ("insufficient_evidence", ["invalid_sha256"],
         "The model identity hash is incomplete; provide a full SHA-256 digest."),
        ("fail", ["smoke_failed"],
         "The backward smoke test failed; repair the migration before acceptance."),
    )
    records = []
    for offset in range(count):
        index = (offset + seed) % len(scenarios)
        status, issues, rationale = scenarios[index]
        evidence = {
            "case_variant": offset + 1,
            "environment": "2x Ascend 910B4-1; CANN 8.2.RC1",
            "accuracy": "pass" if status not in {"fail"} else "below threshold",
            "protocol": "identical" if "unfair_comparison" not in issues else "batch mismatch",
            "model_sha256": "a" * 64 if "invalid_sha256" not in issues else "abc123",
            "smoke": "pass" if "smoke_failed" not in issues else "backward failed",
        }
        answer = {"status": status, "issues": issues, "rationale": rationale}
        records.append({
            "id": f"MPB{split[:2].upper()}-{seed}-{offset + 1:04d}",
            "images": [],
            "messages": [
                {"role": "system", "content": "Audit Ascend migration evidence. Return JSON only."},
                {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, sort_keys=True)},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, sort_keys=True)},
            ],
        })
    return records


def aggregate_independent_pairs(pairs):
    if len(pairs) < 3:
        raise ValueError("at least three independent run pairs are required")
    improvements = [
        (1.0 - pair["optimized"] / pair["baseline"]) * 100.0
        for pair in pairs
    ]
    mean = statistics.mean(improvements)
    standard_error = statistics.stdev(improvements) / math.sqrt(len(improvements))
    critical = 4.302652729911275 if len(improvements) == 3 else 1.96
    margin = critical * standard_error
    return {
        "pair_count": len(pairs),
        "improvement_percent_by_pair": improvements,
        "improvement_percent_mean": mean,
        "improvement_percent_stdev": statistics.stdev(improvements),
        "improvement_percent_ci95": [mean - margin, mean + margin],
    }


def score_explanation(text, expected_status, expected_issues):
    normalized = text.lower().replace("_", " ")
    status_present = expected_status.replace("_", " ") in normalized
    issue_hits = sum(issue.replace("_", " ") in normalized for issue in expected_issues)
    issue_coverage = 1.0 if not expected_issues else issue_hits / len(expected_issues)
    fidelity = min(5, 1 + int(status_present) * 2 + round(issue_coverage * 2))
    words = text.split()
    has_action = any(token in normalized for token in ("rerun", "provide", "verify", "补充", "重新", "核验"))
    readability = 5 if 5 <= len(words) <= 80 and has_action else 4 if len(words) <= 120 else 2
    return {
        "decision_preserved": status_present,
        "issue_coverage": issue_coverage,
        "fidelity": fidelity,
        "readability": readability,
    }
