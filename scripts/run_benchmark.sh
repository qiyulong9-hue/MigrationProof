#!/usr/bin/env bash
set -eo pipefail

MODE="${1:?usage: run_benchmark.sh baseline|optimized}"
ROOT=/home/89b0d3e7cd994caa/migrationproof_qwen35_semifinal_20260914_v1
HISTORY=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
SOURCE_CONFIG="$HISTORY/project/qwen3_5_0_8b_smoke_v4.yaml"
source "$ROOT/env/load_project_env.sh"
set -u

case "$MODE" in
  baseline)
    RECOMPUTE=true
    FORWARD_PREFETCH=0
    BACKWARD_PREFETCH=0
    ;;
  optimized)
    RECOMPUTE=false
    FORWARD_PREFETCH=1
    BACKWARD_PREFETCH=1
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

TRAIN_ITERS=21
RUN_ID="semifinal_b4_${MODE}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"
CACHE_DIR="$ROOT/cache/official_coco_qwen35_b4_v1"
mkdir -p "$RUN_DIR" "$CACHE_DIR"
cp -n "$SOURCE_CONFIG" "$RUN_DIR/config.source.yaml"

"$MP_PYTHON" - "$SOURCE_CONFIG" "$RUN_DIR/config.yaml" \
  "$TRAIN_ITERS" "$RECOMPUTE" "$FORWARD_PREFETCH" "$BACKWARD_PREFETCH" \
  "$CACHE_DIR" <<'PY'
from pathlib import Path
import sys

source, output = map(Path, sys.argv[1:3])
train_iters, recompute, forward_prefetch, backward_prefetch = sys.argv[3:7]
cache_dir = sys.argv[7]
text = source.read_text(encoding="utf-8")
replacements = {
    "num_to_forward_prefetch: 0": f"num_to_forward_prefetch: {forward_prefetch}",
    "num_to_backward_prefetch: 0": f"num_to_backward_prefetch: {backward_prefetch}",
    "cache_dir: /home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01/cache/smoke_v4_eager":
        f"cache_dir: {cache_dir}",
    "  recompute: true": f"  recompute: {recompute}",
    "  train_iters: 2": f"  train_iters: {train_iters}",
    "  save_interval: 2": "  save_interval: 0",
    "  save: /home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01/checkpoints/smoke_v4_eager_2":
        '  save: ""',
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"required configuration marker missing: {old}")
    text = text.replace(old, new, 1)
output.open("x", encoding="utf-8").write(text)
PY

env | sort > "$RUN_DIR/environment.txt"
sha256sum "$RUN_DIR/config.source.yaml" "$RUN_DIR/config.yaml" \
  "$ROOT/env/load_project_env.sh" "$ROOT/scripts/run_benchmark.sh" \
  > "$RUN_DIR/input_sha256.txt"

cd "$HISTORY/source/MindSpeed-MM-torch260-nativefsdp-v2"
set +e
"$MP_TORCHRUN" --nproc_per_node 2 --nnodes 1 --node_rank 0 \
  --master_addr localhost --master_port 6127 \
  mindspeed_mm/fsdp/train/trainer.py "$RUN_DIR/config.yaml" \
  2>&1 | tee "$RUN_DIR/train.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

printf '%s\n' "$TRAIN_STATUS" > "$RUN_DIR/exit_code.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/sha256.txt"
printf '%s\n' "$RUN_DIR"
exit "$TRAIN_STATUS"
