import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, type=Path)
parser.add_argument("--requests", required=True, type=Path)
parser.add_argument("--references", required=True, type=Path)
args = parser.parse_args()
rows = json.loads(args.dataset.read_text(encoding="utf-8"))
with args.requests.open("x", encoding="utf-8", newline="\n") as requests, args.references.open("x", encoding="utf-8", newline="\n") as references:
    for row in rows:
        requests.write(json.dumps({"id": row["id"], "messages": row["messages"][:-1], "generation": {"max_new_tokens": 192}}, ensure_ascii=False) + "\n")
        references.write(json.dumps({"id": row["id"], **row["label"]}, ensure_ascii=False) + "\n")
