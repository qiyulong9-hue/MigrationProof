#!/usr/bin/env bash
set -eo pipefail

MODE="${1:?usage: run_p0_benchmark_once.sh baseline|optimized}"
ROOT="${ROOT_OVERRIDE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PAIR="${PAIR_ID:-independent}"
HISTORY=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
SOURCE_CONFIG="$HISTORY/project/qwen3_5_0_8b_smoke_v4.yaml"
source "$ROOT/env/load_project_env.sh"
case "$MODE" in
  baseline) RECOMPUTE=true; FP=0; BP=0; PORT_OFFSET=0 ;;
  optimized) RECOMPUTE=false; FP=1; BP=1; PORT_OFFSET=1 ;;
  *) exit 2 ;;
esac
RUN_ID="p0_pair${PAIR}_${MODE}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"
CACHE_DIR="$ROOT/cache/official_coco_qwen35_b4_v1"
mkdir -p "$RUN_DIR" "$CACHE_DIR"
cp -n "$SOURCE_CONFIG" "$RUN_DIR/config.source.yaml"
"$MP_PYTHON" - "$SOURCE_CONFIG" "$RUN_DIR/config.yaml" "$RECOMPUTE" "$FP" "$BP" "$CACHE_DIR" <<'PY'
from pathlib import Path
import sys, yaml
source, output = map(Path, sys.argv[1:3])
recompute, fp, bp, cache = sys.argv[3:7]
c = yaml.safe_load(source.read_text(encoding="utf-8"))
c["parallel"]["fsdp_plan"].update({"num_to_forward_prefetch": int(fp), "num_to_backward_prefetch": int(bp)})
c["data"]["dataset_param"]["basic_parameters"]["cache_dir"] = cache
c["features"]["recompute"] = recompute.lower() == "true"
c["training"].update({"train_iters": 21, "save_interval": 0, "save": ""})
output.open("x", encoding="utf-8").write(yaml.safe_dump(c, allow_unicode=True, sort_keys=False))
PY
env | sort > "$RUN_DIR/environment.txt"
cd "$HISTORY/source/MindSpeed-MM-torch260-nativefsdp-v2"
set +e
"$MP_TORCHRUN" --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost \
  --master_port "$((6300 + PAIR * 10 + PORT_OFFSET))" \
  mindspeed_mm/fsdp/train/trainer.py "$RUN_DIR/config.yaml" 2>&1 | tee "$RUN_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/sha256.txt"
exit "$STATUS"
