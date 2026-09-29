#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from migrationproof.corpus_evaluator import evaluate_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    cases = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    metrics = evaluate_corpus(cases)
    document = {
        "schema_version": "migrationproof.expanded-bench-evaluation.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "MigrationProof-Bench v0.4",
        "split": "test-seed42-expanded",
        "metrics": metrics,
        "input_path": str(args.input),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "gate": "pass" if not metrics["mismatches"] else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, ensure_ascii=False, indent=2))
    raise SystemExit(0 if document["gate"] == "pass" else 1)


if __name__ == "__main__":
    main()
