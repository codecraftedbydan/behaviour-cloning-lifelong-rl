#!/usr/bin/env bash
# PINNED 200-task null-rule run: scripts/run_pilot.sh with --num-new-tasks 200 (same gating:
# --confirmed, --matmul-precision highest only, --allow-cpu required on CPU; allow_cpu and
# platform recorded). Seeds 500-509 (fresh 600-609); output results/null_200/.
# Usage: scripts/run_null200.sh --confirmed --matmul-precision highest [--allow-cpu]
set -euo pipefail
for a in "$@"; do
  case "$a" in --num-new-tasks*) echo "run_null200.sh: task count is fixed at 200" >&2; exit 2 ;; esac
done
exec bash "$(dirname "$0")/run_pilot.sh" --num-new-tasks 200 "$@"
