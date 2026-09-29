#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from migrationproof.benchmark_corpus import build_expanded_cases, to_utf8_bytes, validate_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    cases = build_expanded_cases()
    errors = validate_corpus(cases)
    if errors:
        raise SystemExit("; ".join(errors))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / "test_cases.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    data = to_utf8_bytes(cases)
    data_path.write_bytes(data)
    statuses = sorted({case["expected"]["status"] for case in cases})
    manifest = {
        "name": "MigrationProof-Bench",
        "version": "v0.4",
        "split": "test-seed42-expanded",
        "sample_count": len(cases),
        "statuses": {status: sum(case["expected"]["status"] == status for case in cases) for status in statuses},
        "boundary_sample_count": sum("boundary" in case["tags"] for case in cases),
        "adversarial_sample_count": sum("adversarial" in case["tags"] for case in cases),
        "test_cases_sha256": hashlib.sha256(data).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
