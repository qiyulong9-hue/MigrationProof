#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORY=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
SOURCE_CONFIG="$HISTORY/project/qwen3_5_0_8b_smoke_v4.yaml"
source "$ROOT/env/load_project_env.sh"
set -u

CANDIDATE_ID="${1:?candidate id required}"
STAGE="${2:-smoke}"
LR="${3:-1e-6}"
EFFECTIVE_BATCH="${4:-16}"
WEIGHT_DECAY="${5:-0.1}"
case "$STAGE" in
  smoke) STEPS=5 ;;
  short) STEPS=30 ;;
  full) STEPS=120 ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac

RUN_ID="p2_${CANDIDATE_ID}_${STAGE}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"
CHECKPOINT_DIR="$ROOT/checkpoints/$RUN_ID"
CACHE_ROOT="$ROOT/cache/mpbench_v061_${CANDIDATE_ID}_${STAGE}"
DATA_ROOT="$ROOT/data/migrationproof-bench-v0.6.1-p2"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$CACHE_ROOT/train"
cp -n "$SOURCE_CONFIG" "$RUN_DIR/config.source.yaml"

"$MP_PYTHON" - "$SOURCE_CONFIG" "$RUN_DIR/config.yaml" "$DATA_ROOT" "$CACHE_ROOT" "$CHECKPOINT_DIR" "$LR" "$EFFECTIVE_BATCH" "$WEIGHT_DECAY" "$STEPS" <<'PY'
from pathlib import Path
import sys, yaml
source, output, data_root, cache_root, checkpoint = map(Path, sys.argv[1:6])
lr, effective_batch, weight_decay, steps = float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
train = config["data"]["dataset_param"]
train["basic_parameters"].update({"dataset_dir": str(data_root), "dataset": str(data_root / "train.json"), "cache_dir": str(cache_root / "train"), "cutoff_len": 1024})
config["parallel"]["fsdp_plan"].update({"num_to_forward_prefetch": 1, "num_to_backward_prefetch": 1})
config["features"]["recompute"] = False
config["training"].update({"seed": 20260917, "train_iters": steps, "val_interval": 0, "save_interval": steps, "save": str(checkpoint), "lr": lr, "weight_decay": weight_decay, "gradient_accumulation_steps": effective_batch // 2})
output.open("x", encoding="utf-8").write(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
PY

env | sort > "$RUN_DIR/environment.txt"
sha256sum "$RUN_DIR/config.yaml" "$DATA_ROOT/manifest.json" "$DATA_ROOT/train.json" "$ROOT/scripts/run_p2_candidate.sh" > "$RUN_DIR/input_sha256.txt"
cd "$HISTORY/source/MindSpeed-MM-torch260-nativefsdp-v2"
set +e
"$MP_TORCHRUN" --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6237 mindspeed_mm/fsdp/train/trainer.py "$RUN_DIR/config.yaml" 2>&1 | tee "$RUN_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/sha256.txt"
printf '%s\n' "$RUN_DIR"
exit "$STATUS"
