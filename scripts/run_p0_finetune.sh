#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORY=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
SOURCE_CONFIG="$HISTORY/project/qwen3_5_0_8b_smoke_v4.yaml"
source "$ROOT/env/load_project_env.sh"
set -u

RUN_ID="p0_b4_finetune_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"
CHECKPOINT_DIR="$ROOT/checkpoints/$RUN_ID"
CACHE_ROOT="$ROOT/cache/mpbench_v05_seed42"
DATA_ROOT="$ROOT/data/migrationproof-bench-v0.5-p0p1"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$CACHE_ROOT/train" "$CACHE_ROOT/validation"
cp -n "$SOURCE_CONFIG" "$RUN_DIR/config.source.yaml"

"$MP_PYTHON" - "$SOURCE_CONFIG" "$RUN_DIR/config.yaml" "$DATA_ROOT" "$CACHE_ROOT" "$CHECKPOINT_DIR" <<'PY'
from copy import deepcopy
from pathlib import Path
import sys, yaml
source, output, data_root, cache_root, checkpoint = map(Path, sys.argv[1:6])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
train = config["data"]["dataset_param"]
train["basic_parameters"].update({
    "dataset_dir": str(data_root), "dataset": str(data_root / "train.json"),
    "cache_dir": str(cache_root / "train"), "cutoff_len": 1024,
})
validation = deepcopy(train)
validation["basic_parameters"].update({
    "dataset": str(data_root / "validation.json"),
    "cache_dir": str(cache_root / "validation"),
})
config["data"]["val_dataset_param"] = validation
config["parallel"]["fsdp_plan"].update({"num_to_forward_prefetch": 1, "num_to_backward_prefetch": 1})
config["features"]["recompute"] = False
config["training"].update({
    "seed": 42, "train_iters": 100, "val_interval": 50, "val_micro_batch_size": 1,
    "save_interval": 100, "save": str(checkpoint), "lr": 1.0e-5,
})
output.open("x", encoding="utf-8").write(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
PY

env | sort > "$RUN_DIR/environment.txt"
sha256sum "$RUN_DIR/config.source.yaml" "$RUN_DIR/config.yaml" \
  "$DATA_ROOT/manifest.json" "$DATA_ROOT/train.json" "$DATA_ROOT/validation.json" \
  "$ROOT/env/load_project_env.sh" "$ROOT/scripts/run_p0_finetune.sh" > "$RUN_DIR/input_sha256.txt"

cd "$HISTORY/source/MindSpeed-MM-torch260-nativefsdp-v2"
set +e
"$MP_TORCHRUN" --nproc_per_node 2 --nnodes 1 --node_rank 0 \
  --master_addr localhost --master_port 6229 \
  mindspeed_mm/fsdp/train/trainer.py "$RUN_DIR/config.yaml" 2>&1 | tee "$RUN_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/sha256.txt"
printf '%s\n' "$RUN_DIR"
exit "$STATUS"
