import argparse
import json
from pathlib import Path

from migrationproof.p0p1_protocol import score_explanation


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    refs = {row["id"]: row for row in load_jsonl(args.references)}
    rows = []
    for prediction in load_jsonl(args.predictions):
        reference = refs[prediction["id"]]
        rows.append({"id": prediction["id"], **score_explanation(
            prediction.get("rationale", ""), reference["status"], reference.get("issues", []))})
    report = {
        "sample_count": len(rows),
        "decision_preservation_rate": sum(row["decision_preserved"] for row in rows) / len(rows),
        "mean_fidelity": sum(row["fidelity"] for row in rows) / len(rows),
        "mean_readability": sum(row["readability"] for row in rows) / len(rows),
        "items": rows,
        "note": "automatic rubric; manual dual-review remains required for competition claims",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.open("x", encoding="utf-8", newline="\n").write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
