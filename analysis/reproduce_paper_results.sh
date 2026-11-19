#!/usr/bin/env bash
# Recompute every number and figure the paper uses from the extracted counted cells, then check the outputs
# against the reviewed copies in paper_results/. CPU only; takes about a minute.
# Usage: bash analysis/reproduce_paper_results.sh [work dir]   (run analysis/extract_bundles.sh first)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); A=$ROOT/analysis; X=${TARS_CELLS:-$ROOT/data/extracted}
W=${1:-$ROOT/data/repro}; mkdir -p "$W"; cd "$W"; export TARS_PPO=$ROOT TARS_CELLS=$X TARS_SWEEPS=${TARS_SWEEPS:-$ROOT/data/lr_sweeps}
PY=${PYTHON:-python}
CELLS="scratch_1M_r2 pretrained_1M frozen_random_1M frozen_pretrained_1M lora_1M ln_scratch_1M ln_pretrained_1M pqn_scratch_1M pqn_pretrained_1M kickstart_1M"
$PY "$A/zero_shot.py" $(for c in $CELLS; do echo "$X/$c/$c"; done) --out zero_shot_counted_700-724.json
$PY "$A/h1prime.py" > /dev/null
$PY "$A/counted_analysis.py" > /dev/null
$PY "$A/addendum.py" > /dev/null
$PY "$A/per_cell_seeds_x_tasks.py" > /dev/null
$PY "$A/revisit_learned_fraction.py" > /dev/null
$PY "$A/rank_lr_raw.py" > /dev/null
$PY "$A/rank_kickstart_raw.py" > /dev/null
# the reviewed JSON records the cell path as "../cells/kickstart_1M", so run it from the same relative layout
mkdir -p cells ks && ln -sfn "$X/kickstart_1M/kickstart_1M" cells/kickstart_1M
(cd ks && $PY "$A/kickstart_windows.py" ../cells/kickstart_1M "$W/kickstart_windows_700-724.json" > /dev/null)
# pilot_40 and null_200 summaries; plot_stream.py lives in the repo root and needs the repo's environment
PR=${TARS_PILOTS:-$ROOT/data/pilot_runs}
for r in pilot_40 null_200; do
  (cd "$ROOT" && $PY plot_stream.py "$PR/$r/pilot.npz" --out-dir "$W/$r" --require-device-info > /dev/null)
  if cmp -s "$ROOT/paper_results/$r/summary.json" "$W/$r/summary.json"; then echo "MATCH  $r/summary.json"; else echo "DIFFER $r/summary.json"; fi
done
cd "$ROOT/paper_results" && for f in *_results.json *_700-724.json *_ranking_raw.json per_cell_seeds_x_tasks.json revisit_learned_fraction.json *.png; do
  if cmp -s "$f" "$W/$f"; then echo "MATCH  $f"; else echo "DIFFER $f"; fi; done
