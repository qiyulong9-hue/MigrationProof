import pytest

from migrationproof.p2_grid import build_candidate_grid, rank_candidates
from migrationproof.p2_gate import apply_test_gate, apply_validation_gate
from migrationproof.p2_review import review_status, stratified_review_sample


def test_grid_is_bounded_and_contains_full_and_lora_candidates():
    grid = build_candidate_grid()
    assert len(grid) == 8
    assert {row["method"] for row in grid} == {"full", "lora"}
    assert len({row["candidate_id"] for row in grid}) == 8


def test_ranking_prioritizes_safety_before_accuracy():
    winner = rank_candidates([
        {"candidate_id": "unsafe", "unsafe_pass_rate": 0.11, "decision_accuracy": 0.95, "issue_macro_f1": 0.90, "fidelity": 4.5},
        {"candidate_id": "safe", "unsafe_pass_rate": 0.04, "decision_accuracy": 0.82, "issue_macro_f1": 0.72, "fidelity": 3.5},
    ])
    assert winner["selected_candidate_id"] == "safe"


@pytest.mark.parametrize("metrics,passed", [
    ({"unsafe_pass_rate": 0.10, "decision_accuracy": 0.80, "issue_macro_f1": 0.70}, True),
    ({"unsafe_pass_rate": 0.101, "decision_accuracy": 0.99, "issue_macro_f1": 0.99}, False),
])
def test_validation_gate_boundaries(metrics, passed):
    assert apply_validation_gate(metrics)["passed"] is passed


def test_test_gate_blocks_performance_until_all_thresholds_pass():
    failed = apply_test_gate({"unsafe_pass_rate": 0.06, "decision_accuracy": 0.90, "issue_macro_f1": 0.80})
    assert failed["passed"] is False
    assert failed["performance_authorized"] is False


def test_review_packet_is_stratified_and_requires_two_real_reviewers():
    rows = [{"id": f"x{i}", "decision": decision} for i, decision in enumerate(["pass", "fail", "conditional_pass", "insufficient_evidence"] * 5)]
    sample = stratified_review_sample(rows, n=12, seed=7)
    assert len(sample) == 12
    assert len({row["decision"] for row in sample}) == 4
    assert review_status(sample)["status"] == "AUTHOR_REVIEW_NEEDED"
