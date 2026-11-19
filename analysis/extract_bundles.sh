#!/usr/bin/env bash
# Verify and unpack the release assets: the ten counted-cell bundles into data/extracted/<cell>/ and the
# LR sweep trials into data/lr_sweeps/ and the pilot runs into data/pilot_runs/ (the layouts the analysis scripts expect).
# (The scratch_1M_r2 bundle's sha256sums.txt wrongly lists itself, so that line is skipped.)
# Usage: bash analysis/extract_bundles.sh <dir holding the downloaded release assets>
set -euo pipefail
SRC=${1:?bundle dir}; ROOT=$(cd "$(dirname "$0")/.." && pwd); OUT=${TARS_CELLS:-$ROOT/data/extracted}
(cd "$SRC" && sha256sum -c "$ROOT/paper_results/bundles.sha256")
mkdir -p "$OUT"
while read -r _ f; do
  cell=${f%%_review_*}; mkdir -p "$OUT/$cell"; tar xzf "$SRC/$f" -C "$OUT/$cell"
  (cd "$OUT/$cell/$cell" && grep -v " sha256sums.txt$" sha256sums.txt | sha256sum -c --quiet -) && echo "ok $cell"
done < "$ROOT/paper_results/bundles.sha256"
# learning-rate (LR) sweep trials, used only by rank_lr_raw.py and rank_kickstart_raw.py
SW=${TARS_SWEEPS:-$ROOT/data/lr_sweeps}
(cd "$SRC" && grep " lr_sweeps_scrubbed.tgz$" "$ROOT/paper_results/release_assets.sha256" | sha256sum -c -)
mkdir -p "$SW" && tar xzf "$SRC/lr_sweeps_scrubbed.tgz" -C "$SW" && echo "ok lr_sweeps"
# the 40-task pilot and the 200-task null run (Sections 4.1 and 4.2): their pilot.npz arrays
PR=${TARS_PILOTS:-$ROOT/data/pilot_runs}
(cd "$SRC" && grep " pilot_runs.tgz$" "$ROOT/paper_results/release_assets.sha256" | sha256sum -c -)
mkdir -p "$PR" && tar xzf "$SRC/pilot_runs.tgz" -C "$PR" && echo "ok pilot_runs"
