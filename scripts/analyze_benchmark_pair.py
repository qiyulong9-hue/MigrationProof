#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from migrationproof.benchmark_analysis import analyze_log, compare_runs


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_log", type=Path)
    parser.add_argument("optimized_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    baseline_text = args.baseline_log.read_text(encoding="utf-8", errors="replace")
    optimized_text = args.optimized_log.read_text(encoding="utf-8", errors="replace")
    baseline = analyze_log(baseline_text)
    optimized = analyze_log(optimized_text)
    document = {
        "schema_version": "migrationproof.performance-comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {"warmup_iterations": 1, "measured_iterations": 20},
        "baseline": baseline,
        "optimized": optimized,
        "comparison": compare_runs(baseline, optimized),
        "source_evidence": {
            "baseline_log": str(args.baseline_log),
            "baseline_log_sha256": file_sha256(args.baseline_log),
            "optimized_log": str(args.optimized_log),
            "optimized_log_sha256": file_sha256(args.optimized_log),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
