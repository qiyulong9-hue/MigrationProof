#!/usr/bin/env bash
set -eo pipefail

MODE="${1:?usage: run_p2_benchmark_once_v2.sh baseline|optimized}"
ROOT="${ROOT_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PAIR="${PAIR_ID:?PAIR_ID required}"
HISTORY=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
SOURCE_CONFIG="$HISTORY/project/qwen3_5_0_8b_smoke_v4.yaml"
BASE_LOAD="$HISTORY/models/Qwen3.5-0.8B-dcp-v1"
CANDIDATE_LOAD="$ROOT/checkpoints/p2_P2-C03_short_20260917_021757"

source "$ROOT/env/load_project_env.sh"
case "$MODE" in
  baseline) RECOMPUTE=true; FP=0; BP=0; LOAD="$BASE_LOAD"; PORT_OFFSET=0 ;;
  optimized) RECOMPUTE=false; FP=1; BP=1; LOAD="$CANDIDATE_LOAD"; PORT_OFFSET=1 ;;
  *) exit 2 ;;
esac
RUN_ID="p2v2_pair${PAIR}_${MODE}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"
CACHE_DIR="$ROOT/cache/p2v2_benchmark_pair${PAIR}_${MODE}"
DATA_ROOT="$ROOT/data/migrationproof-bench-v0.6.1-p2"
mkdir -p "$RUN_DIR" "$CACHE_DIR"
cp -n "$SOURCE_CONFIG" "$RUN_DIR/config.source.yaml"
"$MP_PYTHON" - "$SOURCE_CONFIG" "$RUN_DIR/config.yaml" "$RECOMPUTE" "$FP" "$BP" "$CACHE_DIR" "$DATA_ROOT" "$LOAD" <<'PY'
from pathlib import Path
import sys, yaml
source, output = map(Path, sys.argv[1:3])
recompute, fp, bp, cache, data_root, load = sys.argv[3:9]
c = yaml.safe_load(source.read_text(encoding="utf-8"))
c["parallel"]["fsdp_plan"].update({"num_to_forward_prefetch": int(fp), "num_to_backward_prefetch": int(bp)})
c["data"]["dataset_param"]["basic_parameters"].update({"dataset_dir": data_root, "dataset": str(Path(data_root) / "train.json"), "cache_dir": cache, "cutoff_len": 1024})
c["features"]["recompute"] = recompute.lower() == "true"
c["training"].update({"seed": 20260917, "train_iters": 21, "gradient_accumulation_steps": 8, "save_interval": 0, "save": "", "load": load, "lr": 3.0e-6, "weight_decay": 0.1, "val_interval": 0})
output.open("x", encoding="utf-8").write(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
PY
env | sort > "$RUN_DIR/environment.txt"
sha256sum "$RUN_DIR/config.yaml" "$DATA_ROOT/manifest.json" "$DATA_ROOT/train.json" "$ROOT/scripts/run_p2_benchmark_once_v2.sh" > "$RUN_DIR/input_sha256.txt"
cd "$HISTORY/source/MindSpeed-MM-torch260-nativefsdp-v2"
set +e
"$MP_TORCHRUN" --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port "$((6700 + PAIR * 10 + PORT_OFFSET))" mindspeed_mm/fsdp/train/trainer.py "$RUN_DIR/config.yaml" 2>&1 | tee "$RUN_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/sha256.txt"
printf '%s\n' "$RUN_DIR"
exit "$STATUS"
