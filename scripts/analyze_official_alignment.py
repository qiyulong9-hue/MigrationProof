import argparse
import json
from pathlib import Path

from migrationproof.official_alignment import compare_accuracy_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ascendc_log", type=Path)
    parser.add_argument("triton_log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare_accuracy_logs(args.ascendc_log, args.triton_log)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": result["gate"]["passed"]}))


if __name__ == "__main__":
    main()

