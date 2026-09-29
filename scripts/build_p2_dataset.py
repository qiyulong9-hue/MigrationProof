import argparse
import json
from pathlib import Path

from migrationproof.p2_dataset import build_dataset, validate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    dataset = build_dataset(args.seed)
    report = validate_dataset(dataset)
    args.output.mkdir(parents=True, exist_ok=False)
    for split, rows in dataset.items():
        with (args.output / f"{split}.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (args.output / f"{split}.json").open("x", encoding="utf-8", newline="\n").write(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        )
    (args.output / "manifest.json").open("x", encoding="utf-8", newline="\n").write(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
