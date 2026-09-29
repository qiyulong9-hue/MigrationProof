#!/usr/bin/env python3
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from migrationproof.acceptance import decide


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--accuracy", required=True, type=Path)
    parser.add_argument("--performance", required=True, type=Path)
    parser.add_argument("--environment", required=True, type=Path)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    task = load(args.task)
    accuracy_document = load(args.accuracy)
    performance_document = load(args.performance)
    environment_document = load(args.environment)
    smoke_document = load(args.smoke)
    baseline = dict(performance_document["baseline"])
    optimized = dict(performance_document["optimized"])
    baseline["throughput"] = baseline["throughput_samples_per_second"]
    optimized["throughput"] = optimized["throughput_samples_per_second"]
    complete = (
        performance_document["comparison"].get("comparable") is True
        and accuracy_document.get("gate") == "pass"
        and smoke_document.get("status") == "pass"
        and environment_document.get("npu_available") is True
    )
    decision = decide(task, accuracy_document["metrics"], baseline, optimized,
                      {"complete": complete})
    inputs = [args.task, args.accuracy, args.performance, args.environment, args.smoke]
    report = {
        "schema_version": "migrationproof.acceptance-report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_id": task["task_id"],
        "decision": decision,
        "accuracy": accuracy_document["metrics"],
        "performance": performance_document["comparison"],
        "environment": {
            "device_model": environment_document.get("device_model"),
            "device_count": environment_document.get("device_count"),
            "cann_version": task.get("ascend", {}).get("cann_version"),
        },
        "smoke_status": smoke_document.get("status"),
        "evidence_ledger": [sha256(path) for path in inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if decision["final_status"] == "pass" else 1)


if __name__ == "__main__":
    main()
