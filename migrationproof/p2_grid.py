import itertools


def build_candidate_grid() -> list[dict]:
    candidates = []
    combinations = itertools.product(("full", "lora"), (1e-6, 3e-6), (16, 32))
    for index, (method, learning_rate, effective_batch_size) in enumerate(combinations, 1):
        candidates.append({
            "candidate_id": f"P2-C{index:02d}",
            "method": method,
            "learning_rate": learning_rate,
            "effective_batch_size": effective_batch_size,
            "weight_decay": 0.1 if method == "full" else 0.01,
            "epochs": 2 if effective_batch_size == 16 else 3,
            "stages": {"smoke_steps": 5, "short_steps": 30, "full_steps": 120},
        })
    return candidates


def rank_candidates(metrics: list[dict]) -> dict:
    eligible = [row for row in metrics if row["unsafe_pass_rate"] <= 0.10]
    if not eligible:
        return {"selected_candidate_id": None, "reason": "no safety-eligible candidate"}
    selected = min(eligible, key=lambda row: (
        row["unsafe_pass_rate"], -row["decision_accuracy"], -row["issue_macro_f1"], -row["fidelity"]
    ))
    return {"selected_candidate_id": selected["candidate_id"], "reason": "safety-first validation ranking"}
