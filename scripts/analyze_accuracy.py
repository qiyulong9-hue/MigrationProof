#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from migrationproof.accuracy_analysis import score_predictions


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metrics = score_predictions(read_jsonl(args.reference), read_jsonl(args.predictions))
    thresholds = {"decision_accuracy_min": 0.90, "issue_macro_f1_min": 0.80,
                  "unsafe_pass_rate_max": 0.05}
    passed = (metrics["decision_accuracy"] >= thresholds["decision_accuracy_min"]
              and metrics["issue_macro_f1"] >= thresholds["issue_macro_f1_min"]
              and metrics["unsafe_pass_rate"] <= thresholds["unsafe_pass_rate_max"])
    document = {
        "schema_version": "migrationproof.application-accuracy.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "thresholds": thresholds,
        "gate": "pass" if passed else "fail",
        "source_evidence": {
            "reference": str(args.reference), "reference_sha256": sha256(args.reference),
            "predictions": str(args.predictions), "predictions_sha256": sha256(args.predictions),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
