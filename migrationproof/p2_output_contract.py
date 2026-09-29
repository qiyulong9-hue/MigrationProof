import json
from dataclasses import dataclass


VALID_DECISIONS = {"pass", "conditional_pass", "fail", "insufficient_evidence"}
VALID_ISSUES = {
    "none", "accuracy_gate_failed", "unsafe_pass", "unfair_comparison",
    "missing_evidence", "contradictory_evidence", "invalid_model_identity",
}
FIELDS = ["decision", "issue_type", "evidence", "explanation"]


class OutputContractError(ValueError):
    pass


@dataclass(frozen=True)
class StructuredDecision:
    decision: str
    issue_type: str
    evidence: tuple[str, ...]
    explanation: str


def parse_model_output(text: str) -> StructuredDecision:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputContractError("output must be exactly one JSON object") from exc
    if not isinstance(payload, dict) or list(payload) != FIELDS:
        raise OutputContractError("output must contain the four ordered fields only")
    if payload["decision"] not in VALID_DECISIONS:
        raise OutputContractError("unknown decision")
    if payload["issue_type"] not in VALID_ISSUES:
        raise OutputContractError("unknown issue_type")
    if not isinstance(payload["evidence"], list) or not payload["evidence"] or not all(
        isinstance(item, str) and item for item in payload["evidence"]
    ):
        raise OutputContractError("evidence must be a non-empty string list")
    if not isinstance(payload["explanation"], str) or not payload["explanation"].strip():
        raise OutputContractError("explanation must be non-empty")
    return StructuredDecision(
        payload["decision"], payload["issue_type"], tuple(payload["evidence"]),
        payload["explanation"].strip(),
    )
