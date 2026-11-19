#!/usr/bin/env bash
# Gate v2 runs: 4 MinAtar envs x {ours, PureJaxRL} x seeds 200-209 (two vmapped
# groups of 5 for memory), 1e7 env steps. No config changes from the presets.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(pwd)/results/gate_v2; mkdir -p $OUT
REF=${PUREJAXRL_DIR:-../purejaxrl-ref}
for env in Breakout Asterix SpaceInvaders Freeway; do
  for s in 200-204 205-209; do
    [ -f $OUT/ours_${env}_$s.npz ] || .venv/bin/python -W ignore ppo.py --preset purejaxrl_match \
        --env $env-MinAtar --total-timesteps 10000000 --seeds $s --out $OUT/ours_${env}_$s.npz
    [ -f $OUT/ref_${env}_$s.npz ] || (cd $REF && .venv/bin/python -W ignore run_ref.py --preset minatar \
        --env $env-MinAtar --total-timesteps 1e7 --seeds $s --out $OUT/ref_${env}_$s.npz)
  done
done
.venv/bin/python scripts/gate_v2.py --dir results/gate_v2
