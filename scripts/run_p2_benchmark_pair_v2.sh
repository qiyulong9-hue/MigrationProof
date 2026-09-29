#!/usr/bin/env bash
set -eo pipefail
PAIR="${1:?usage: run_p2_benchmark_pair_v2.sh pair-id}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for MODE in baseline optimized; do
  ROOT_OVERRIDE="$ROOT" PAIR_ID="$PAIR" bash "$ROOT/scripts/run_p2_benchmark_once_v2.sh" "$MODE"
done
