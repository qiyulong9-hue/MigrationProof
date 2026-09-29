#!/usr/bin/env bash

# Session-only environment for the MigrationProof semifinal release.
# This file does not modify shell profiles or system packages.

MP_HISTORY_ROOT=/home/89b0d3e7cd994caa/migrationproof_qwen35_20260716_01
MP_ENV="$MP_HISTORY_ROOT/env/qwen35_py311_torch260_v2"
MP_PYTHON="$MP_ENV/bin/python"
MP_TORCHRUN="$MP_ENV/bin/torchrun"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

export MP_HISTORY_ROOT MP_ENV MP_PYTHON MP_TORCHRUN
export PATH="$MP_ENV/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$MP_ENV/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$MP_ENV/lib/python3.11/site-packages/scikit_learn.libs/libgomp-a49a47f9.so.1.0.0"
export PYTHONPATH="$MP_HISTORY_ROOT/env/vendor_setuptools_80_9:$MP_HISTORY_ROOT/source/Megatron-LM-core_v0.12.1:$MP_HISTORY_ROOT/source/MindSpeed:$MP_HISTORY_ROOT/source/MindSpeed-MM-torch260-validation-v5:${PYTHONPATH:-}"
export NON_MEGATRON=true
export ASCEND_LAUNCH_BLOCKING=0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
