import argparse
import json
from pathlib import Path

from migrationproof.p0p1_protocol import aggregate_independent_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    pairs = json.loads(args.input.read_text(encoding="utf-8"))
    result = aggregate_independent_pairs(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.open("x", encoding="utf-8", newline="\n").write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
