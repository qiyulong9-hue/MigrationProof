#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HISTORY=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
SOURCE_CONFIG="$HISTORY/project/qwen3_5_0_8b_smoke_v4.yaml"
LOAD="$ROOT/checkpoints/p2_P2-C03_short_20260917_021757"
RUN_ID="p2_P2-C04R_continuation_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/$RUN_ID"
CHECKPOINT_DIR="$ROOT/checkpoints/$RUN_ID"
CACHE_DIR="$ROOT/cache/$RUN_ID"
DATA_ROOT="$ROOT/data/migrationproof-bench-v0.6.1-p2"

source "$ROOT/env/load_project_env.sh"
mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$CACHE_DIR"
cp -n "$SOURCE_CONFIG" "$RUN_DIR/config.source.yaml"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$MP_PYTHON" - "$SOURCE_CONFIG" "$RUN_DIR/config.yaml" "$DATA_ROOT" "$CACHE_DIR" "$LOAD" "$CHECKPOINT_DIR" <<'PY'
from pathlib import Path
import sys
import yaml
from migrationproof.p2_continuation import build_continuation_config

source, output, data_root, cache, load, save = map(Path, sys.argv[1:7])
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["data"]["dataset_param"]["basic_parameters"].update(
    {
        "dataset_dir": str(data_root),
        "dataset": str(data_root / "train.json"),
        "cache_dir": str(cache),
        "cutoff_len": 1024,
    }
)
config = build_continuation_config(
    config,
    load=str(load),
    save=str(save),
    resume_iteration=30,
    additional_iterations=30,
    learning_rate=5e-7,
    effective_batch_size=16,
)
config["training"].update(
    {"seed": 20260918, "weight_decay": 0.1, "no_load_optim": True, "no_load_rng": True}
)
output.open("x", encoding="utf-8").write(
    yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
)
PY

env | sort > "$RUN_DIR/environment.txt"
sha256sum "$RUN_DIR/config.yaml" "$DATA_ROOT/manifest.json" "$DATA_ROOT/train.json" "$ROOT/scripts/run_p2_c04r_continuation.sh" "$ROOT/migrationproof/p2_continuation.py" > "$RUN_DIR/input_sha256.txt"
cd "$HISTORY/source/MindSpeed-MM-torch260-nativefsdp-v2"
set +e
"$MP_TORCHRUN" --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 7018 mindspeed_mm/fsdp/train/trainer.py "$RUN_DIR/config.yaml" 2>&1 | tee "$RUN_DIR/train.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
sha256sum "$RUN_DIR"/* > "$RUN_DIR/sha256.txt"
printf '%s\n' "$RUN_DIR"
exit "$STATUS"
