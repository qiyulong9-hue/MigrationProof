import argparse
import hashlib
import json
from pathlib import Path

from migrationproof.p0p1_protocol import build_frozen_split, build_sft_records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open("x", encoding="utf-8", newline="\n").write(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    test_cases = [json.loads(line) for line in args.test_cases.read_text(encoding="utf-8").splitlines() if line]
    split = build_frozen_split(test_cases, seed=42)
    train = build_sft_records("train", 240, seed=42)
    validation = build_sft_records("validation", 48, seed=1042)
    requests = []
    references = []
    for case in test_cases:
        requests.append({
            "id": case["id"],
            "generation": {"max_new_tokens": 160, "temperature": 0.0},
            "messages": [
                {"role": "system", "content": "Audit Ascend migration evidence. Return JSON only with status, issues, and rationale."},
                {"role": "user", "content": json.dumps(case["evidence"], ensure_ascii=False, sort_keys=True)},
            ],
        })
        references.append({"id": case["id"], **case["expected"]})
    outputs = {
        "train.json": json.dumps(train, ensure_ascii=False, indent=2) + "\n",
        "validation.json": json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        "test_requests.jsonl": "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in requests),
        "test_references.jsonl": "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in references),
        "split_ids.json": json.dumps(split, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    }
    for name, payload in outputs.items():
        write_new(args.output / name, payload)
    manifest = {
        "name": "MigrationProof-Bench",
        "version": "v0.5-p0p1",
        "seed": 42,
        "counts": {"train": len(train), "validation": len(validation), "test": len(test_cases)},
        "files": {name: digest(args.output / name) for name in outputs},
        "leakage_policy": "test IDs and records are excluded from generated train and validation sets",
    }
    write_new(args.output / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
