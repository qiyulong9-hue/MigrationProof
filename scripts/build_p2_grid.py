import json
from pathlib import Path

from migrationproof.p2_grid import build_candidate_grid


output = Path("configs/p2_candidate_grid_v1.json")
output.open("x", encoding="utf-8", newline="\n").write(json.dumps(build_candidate_grid(), indent=2) + "\n")
print(output)
