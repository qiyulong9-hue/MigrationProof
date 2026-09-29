import json


STATUSES = {"pass", "conditional_pass", "fail", "insufficient_evidence"}


def _case(index, status, issues, mutation, tags):
    evidence = {
        "environment": "2x Ascend 910B4-1; CANN 8.2.RC1; torch 2.6.0; torch_npu 2.6.0",
        "model_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "smoke": "forward/backward/update/save completed",
        "accuracy": {"value": 0.95, "minimum": 0.90},
        "baseline": {"device": "Ascend 910B4-1", "global_batch_size": 8, "repetitions": 20},
        "candidate": {"device": "Ascend 910B4-1", "global_batch_size": 8, "repetitions": 20},
        "performance": {"improvement_percent": 10.0, "minimum_percent": 5.0},
    }
    mutation(evidence)
    return {
        "id": f"MPBX-{index:04d}",
        "request": "Audit this Qwen3.5 Ascend migration delivery and issue one acceptance status.",
        "evidence": evidence,
        "expected": {"status": status, "issues": issues},
        "tags": tags,
    }


def build_expanded_cases():
    specs = []
    noop = lambda evidence: None
    for i in range(10):
        specs.append(("pass", [], noop, ["control", "complete_evidence"]))
    for i in range(10):
        def missing_app(evidence, variant=i):
            evidence["application_accuracy_pending"] = True
            evidence["non_core_item"] = f"explanation_review_{variant}"
        specs.append(("conditional_pass", ["missing_application_accuracy"], missing_app,
                      ["boundary", "non_core_pending"]))
    fail_specs = [
        ("accuracy_gate_failed", lambda e: e["accuracy"].update(value=0.899)),
        ("accuracy_gate_failed", lambda e: e["accuracy"].update(value=0.50)),
        ("performance_gate_failed", lambda e: e["performance"].update(improvement_percent=4.999)),
        ("performance_gate_failed", lambda e: e["performance"].update(improvement_percent=-3.0)),
        ("smoke_failed", lambda e: e.update(smoke="backward exception")),
        ("smoke_failed", lambda e: e.update(smoke="optimizer update failed")),
    ]
    for repeat in range(2):
        for issue, mutation in fail_specs:
            specs.append(("fail", [issue], mutation, ["adversarial", "proven_failure", f"variant_{repeat+1}"]))
    insufficient_specs = [
        ("missing_environment", lambda e: e.pop("environment")),
        ("invalid_sha256", lambda e: e.update(model_sha256="abc123")),
        ("invalid_sha256", lambda e: e.update(dataset_sha256="not-a-sha")),
        ("wrong_primary_baseline", lambda e: e["baseline"].update(device="NVIDIA GPU")),
        ("unfair_comparison", lambda e: e["candidate"].update(global_batch_size=4)),
        ("insufficient_repetitions", lambda e: e["candidate"].update(repetitions=19)),
        ("insufficient_repetitions", lambda e: e["baseline"].update(repetitions=5)),
        ("model_identity_missing", lambda e: e.pop("model_sha256")),
        ("dataset_identity_missing", lambda e: e.pop("dataset_sha256")),
    ]
    for repeat in range(2):
        for issue, mutation in insufficient_specs:
            specs.append(("insufficient_evidence", [issue], mutation,
                          ["adversarial", "boundary", f"variant_{repeat+1}"]))
    cases = [
        _case(index, status, issues, mutation, tags)
        for index, (status, issues, mutation, tags) in enumerate(specs, start=1)
    ]
    return cases


def validate_corpus(cases):
    errors = []
    ids = [case.get("id") for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case id")
    for case in cases:
        if case.get("expected", {}).get("status") not in STATUSES:
            errors.append(f'{case.get("id")}: invalid status')
        if not isinstance(case.get("expected", {}).get("issues"), list):
            errors.append(f'{case.get("id")}: issues must be a list')
        if not case.get("evidence"):
            errors.append(f'{case.get("id")}: evidence is required')
        if not case.get("tags"):
            errors.append(f'{case.get("id")}: tags are required')
    return errors


def to_jsonl(cases):
    return "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in cases)


def to_utf8_bytes(cases):
    return to_jsonl(cases).encode("utf-8")
