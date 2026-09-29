#!/usr/bin/env bash
set -eo pipefail
ROOT="${ROOT_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HISTORY=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
SOURCE_CONFIG="$HISTORY/project/qwen3_5_0_8b_smoke_v4.yaml"
LOAD="$ROOT/checkpoints/p2_P2-C03_short_20260917_021757"
RUN_DIR="$ROOT/runs/p2_candidate_load_smoke_v2_$(date +%Y%m%d_%H%M%S)"
CACHE_DIR="$ROOT/cache/p2_candidate_load_smoke_v2"
DATA_ROOT="$ROOT/data/migrationproof-bench-v0.6.1-p2"
source "$ROOT/env/load_project_env.sh"
mkdir -p "$RUN_DIR" "$CACHE_DIR"
"$MP_PYTHON" - "$SOURCE_CONFIG" "$RUN_DIR/config.yaml" "$CACHE_DIR" "$DATA_ROOT" "$LOAD" <<'PY'
from pathlib import Path
import sys, yaml
source, output, cache, data_root, load = sys.argv[1:6]
c = yaml.safe_load(Path(source).read_text(encoding="utf-8"))
c["parallel"]["fsdp_plan"].update({"num_to_forward_prefetch": 1, "num_to_backward_prefetch": 1})
c["data"]["dataset_param"]["basic_parameters"].update({"dataset_dir": data_root, "dataset": str(Path(data_root) / "train.json"), "cache_dir": cache, "cutoff_len": 1024})
c["features"]["recompute"] = False
c["training"].update({"seed": 20260917, "train_iters": 31, "gradient_accumulation_steps": 8, "save_interval": 0, "save": "", "load": load, "lr": 3.0e-6, "weight_decay": 0.1, "val_interval": 0})
Path(output).open("x", encoding="utf-8").write(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
PY
cd "$HISTORY/source/MindSpeed-MM-torch260-nativefsdp-v2"
set +e
"$MP_TORCHRUN" --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6699 mindspeed_mm/fsdp/train/trainer.py "$RUN_DIR/config.yaml" 2>&1 | tee "$RUN_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/sha256.txt"
printf '%s\n' "$RUN_DIR"
exit "$STATUS"
