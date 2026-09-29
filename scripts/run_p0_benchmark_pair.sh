#!/usr/bin/env bash
set -eo pipefail

PAIR="${1:?usage: run_p0_benchmark_pair.sh pair-id}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for MODE in baseline optimized; do
  ROOT_OVERRIDE="$ROOT" PAIR_ID="$PAIR" bash "$ROOT/scripts/run_p0_benchmark_once.sh" "$MODE"
done
