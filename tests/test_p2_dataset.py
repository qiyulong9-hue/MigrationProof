import json

import pytest

from migrationproof.p2_dataset import build_dataset, validate_dataset


def test_dataset_is_deterministic_balanced_and_leak_free():
    first = build_dataset(seed=20260917)
    second = build_dataset(seed=20260917)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    report = validate_dataset(first)
    assert report["valid"] is True
    assert report["split_sizes"] == {"train": 360, "validation": 72, "test": 60}
    assert report["id_overlap"] == []
    assert min(report["train_decision_counts"].values()) >= 80
    assert report["adversarial_train_count"] >= 120


def test_every_label_has_exact_structured_fields_and_grounded_evidence():
    dataset = build_dataset(seed=7)
    for split in dataset.values():
        for record in split:
            label = record["label"]
            assert list(label) == ["decision", "issue_type", "evidence", "explanation"]
            assert isinstance(label["evidence"], list) and label["evidence"]
            source = json.dumps(record["input"], ensure_ascii=False)
            assert all(item in source for item in label["evidence"])
            assert record["images"] == []
            assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]
            assert json.loads(record["messages"][-1]["content"]) == label


def test_validation_rejects_split_leakage():
    dataset = build_dataset(seed=11)
    dataset["validation"][0]["id"] = dataset["train"][0]["id"]
    with pytest.raises(ValueError, match="split leakage"):
        validate_dataset(dataset)
