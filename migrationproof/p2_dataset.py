import hashlib
import json
import random
from collections import Counter


DECISION_CYCLE = ("pass", "conditional_pass", "fail", "insufficient_evidence")
SCENARIOS = (
    ("pass", "none", "accuracy=0.91", "unsafe_pass_rate=0.02", False),
    ("conditional_pass", "missing_evidence", "accuracy=0.86", "memory_log=missing", False),
    ("fail", "accuracy_gate_failed", "accuracy=0.79", "accuracy_gate=0.85", True),
    ("fail", "unsafe_pass", "unsafe_pass_rate=0.12", "unsafe_pass_gate=0.05", True),
    ("insufficient_evidence", "unfair_comparison", "baseline_batch=4", "optimized_batch=8", True),
    ("insufficient_evidence", "contradictory_evidence", "report_accuracy=0.88", "log_accuracy=0.61", True),
    ("insufficient_evidence", "invalid_model_identity", "model_sha256=abc123", "required_sha256_length=64", True),
    ("conditional_pass", "missing_evidence", "test_split=frozen", "review_status=pending", False),
)


def _record(split: str, index: int, seed: int, scenario: tuple) -> dict:
    decision, issue, fact_a, fact_b, adversarial = scenario
    record_id = f"MP6-{split[:2].upper()}-{seed}-{index:04d}"
    evidence = [fact_a, fact_b]
    input_payload = {
        "model": "Qwen3.5-0.8B",
        "device": "Ascend 910B4-1",
        "facts": evidence,
        "instruction": "Return exactly decision, issue_type, evidence, explanation as JSON.",
    }
    label = {
        "decision": decision,
        "issue_type": issue,
        "evidence": evidence,
        "explanation": f"Decision is {decision} because {fact_a} and {fact_b}.",
    }
    return {
        "id": record_id,
        "input": input_payload,
        "label": label,
        "tags": ["adversarial" if adversarial else "standard", issue],
        "images": [],
        "messages": [
            {"role": "system", "content": "Audit migration evidence and return one JSON object with exactly four ordered fields: decision, issue_type, evidence, explanation."},
            {"role": "user", "content": json.dumps(input_payload, ensure_ascii=False, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ],
    }


def build_dataset(seed: int = 20260917) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    sizes = {"train": 360, "validation": 72, "test": 60}
    result = {}
    for split, size in sizes.items():
        records = []
        for index in range(1, size + 1):
            desired = DECISION_CYCLE[(index - 1) % len(DECISION_CYCLE)]
            options = [scenario for scenario in SCENARIOS if scenario[0] == desired]
            scenario = options[(index + seed) % len(options)]
            records.append(_record(split, index, seed, scenario))
        rng.shuffle(records)
        result[split] = records
    return result


def validate_dataset(splits: dict) -> dict:
    ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    overlap = sorted((ids["train"] & ids["validation"]) | (ids["train"] & ids["test"]) | (ids["validation"] & ids["test"]))
    if overlap:
        raise ValueError(f"split leakage: {overlap[:3]}")
    for rows in splits.values():
        for row in rows:
            if list(row["label"]) != ["decision", "issue_type", "evidence", "explanation"]:
                raise ValueError("invalid label schema")
            source = json.dumps(row["input"], ensure_ascii=False)
            if not all(item in source for item in row["label"]["evidence"]):
                raise ValueError("ungrounded evidence")
    payload = json.dumps(splits, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "valid": True,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "id_overlap": overlap,
        "train_decision_counts": dict(Counter(row["label"]["decision"] for row in splits["train"])),
        "adversarial_train_count": sum("adversarial" in row["tags"] for row in splits["train"]),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
