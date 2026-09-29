import argparse
import json
from collections import defaultdict
from pathlib import Path

from migrationproof.p2_gate import apply_validation_gate


parser = argparse.ArgumentParser()
parser.add_argument("--references", required=True, type=Path)
parser.add_argument("--predictions", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
refs = {row["id"]: row for row in map(json.loads, args.references.read_text(encoding="utf-8").splitlines())}
preds = {row["id"]: row for row in map(json.loads, args.predictions.read_text(encoding="utf-8").splitlines())}
ids = sorted(refs)
accuracy = sum(preds[i]["decision"] == refs[i]["decision"] for i in ids) / len(ids)
labels = sorted({row["issue_type"] for row in refs.values()})
f1s = []
for label in labels:
    tp = sum(preds[i]["issue_type"] == label == refs[i]["issue_type"] for i in ids)
    fp = sum(preds[i]["issue_type"] == label != refs[i]["issue_type"] for i in ids)
    fn = sum(preds[i]["issue_type"] != label == refs[i]["issue_type"] for i in ids)
    f1s.append(0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
unsafe_ids = [i for i in ids if refs[i]["decision"] == "fail"]
unsafe_pass = sum(preds[i]["decision"] in {"pass", "conditional_pass"} for i in unsafe_ids) / max(1, len(unsafe_ids))
metrics = {"sample_count": len(ids), "decision_accuracy": accuracy, "issue_macro_f1": sum(f1s) / len(f1s), "unsafe_pass_rate": unsafe_pass, "valid_output_rate": sum(preds[i]["valid_output"] for i in ids) / len(ids)}
result = {"metrics": metrics, "gate": apply_validation_gate(metrics)}
args.output.open("x", encoding="utf-8", newline="\n").write(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
