#!/usr/bin/env bash
# PINNED 40-task pilot (tars-test-rules v2 §5). DO NOT RUN until Dan confirms the pilot.
# Stream seeds 500-509 (fresh baselines 600-609 derived automatically) are reserved; the
# seed guard refuses them unless --allow-official-seeds is given (passed below on purpose).
#   40 new tasks, K = 100 updates/block, revisit every 3rd new task (53 blocks),
#   lifelong_stream preset, hidden 950 (1,091,555 params), head-reset probe every 5th new task.
#   --num-new-tasks 200: the 200-task null-rule run (266 blocks, 40 head probes), same settings
#   otherwise, output results/null_200 (wrapper: scripts/run_null200.sh). Only 40 or 200.
#
# Usage: scripts/run_pilot.sh --confirmed --matmul-precision highest [--allow-cpu] [--num-new-tasks 40|200]
#   --matmul-precision highest  required; no other value is accepted (fp32 matmuls everywhere)
#   --allow-cpu                 required to run when jax's default backend is not gpu
# Precision, platform, default backend and allow_cpu are recorded in the run record; relative
# metrics refuse to mix platforms or precisions (stream / fresh / head probe), and the analysis
# below uses --require-device-info.
# TARS_PILOT_DRY_RUN=1: do every check, then only print the commands (nothing created or run).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIRMED=0; PREC=""; ALLOW_CPU=0; NTASKS=40
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirmed) CONFIRMED=1; shift ;;
    --allow-cpu) ALLOW_CPU=1; shift ;;
    --num-new-tasks) NTASKS="${2:-}"; shift 2 || shift ;;
    --num-new-tasks=*) NTASKS="${1#*=}"; shift ;;
    --matmul-precision) PREC="${2:-}"; shift 2 || shift ;;
    --matmul-precision=*) PREC="${1#*=}"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$NTASKS" in
  40) DEFAULT_OUT=results/pilot_40 ;;
  200) DEFAULT_OUT=results/null_200 ;;
  *) echo "--num-new-tasks must be 40 (pilot) or 200 (null-rule run), got '$NTASKS'" >&2; exit 2 ;;
esac
if [[ "$PREC" != "highest" ]]; then
  echo "--matmul-precision highest must be given explicitly (got '${PREC}'; only 'highest' is accepted)" >&2
  exit 2
fi
if [[ $CONFIRMED != 1 ]]; then
  echo "Run ($NTASKS new tasks) is pinned but not authorised. Re-run as: $0 --confirmed --matmul-precision highest" >&2
  exit 2
fi
BACKEND=$(.venv/bin/python -c "import jax; print(jax.default_backend())" 2>/dev/null || echo unknown)
CPU_FLAG=()
if [[ "$BACKEND" != "gpu" ]]; then
  if [[ $ALLOW_CPU == 1 && "$BACKEND" == "cpu" ]]; then
    echo "WARNING: running the pilot on CPU (--allow-cpu); recorded as allow_cpu=True" >&2
    CPU_FLAG=(--allow-cpu)
  else
    echo "refusing to run the pilot: jax default backend is '$BACKEND', not 'gpu' (pass --allow-cpu to accept cpu)" >&2
    exit 2
  fi
fi
OUT=${OUT:-$DEFAULT_OUT}
RUN=(); LOG="$OUT/pilot.log"
if [[ "${TARS_PILOT_DRY_RUN:-0}" == 1 ]]; then
  RUN=(echo DRY-RUN:); LOG=/dev/null
  echo "DRY-RUN: mkdir -p $OUT"
else
  mkdir -p "$OUT"
fi
"${RUN[@]}" .venv/bin/python lifelong.py \
  --preset lifelong_stream \
  --hidden-size 950 \
  --seeds 500-509 --allow-official-seeds \
  --num-new-tasks "$NTASKS" --block-updates 100 --revisit-every 3 \
  --head-probe-every 5 \
  --matmul-precision "$PREC" \
  --require-gpu "${CPU_FLAG[@]}" \
  --out "$OUT/pilot.npz" 2>&1 | tee "$LOG"
"${RUN[@]}" .venv/bin/python plot_stream.py "$OUT/pilot.npz" --out-dir "$OUT" --require-device-info
