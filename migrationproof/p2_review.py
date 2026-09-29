import random
from collections import defaultdict


def stratified_review_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[record["decision"]].append(record)
    rng = random.Random(seed)
    decisions = sorted(groups)
    per_group = n // len(decisions)
    selected = []
    for decision in decisions:
        pool = list(groups[decision])
        rng.shuffle(pool)
        selected.extend(pool[:per_group])
    for record in selected:
        record = record
        record.setdefault("reviewer_a", {})
        record.setdefault("reviewer_b", {})
    return selected


def review_status(rows: list[dict]) -> dict:
    complete = all(row.get("reviewer_a") and row.get("reviewer_b") for row in rows)
    return {"status": "READY_FOR_AGREEMENT" if complete else "AUTHOR_REVIEW_NEEDED", "sample_count": len(rows)}
