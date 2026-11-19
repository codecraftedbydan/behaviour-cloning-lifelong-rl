#!/usr/bin/env bash
# Stage-1 counted-cell queue for a single-GPU Linux machine. Lives outside the frozen repo.
# Resumable: skips trials whose manifest status is complete; half-done trials
# get a new suffixed out-dir (_r2, _r3, ...). Failures are logged; queue continues.
# Do not change XLA allocator env vars. Do not modify the frozen repo.
# counted (12 Nov, decision 17:48): derived from kickstart_queue.sh (8d169bdd). Differences:
# runner lr_cell_freshlr.py (4098ddd5) with a per-trial --fresh-lr; counted seeds 700-724
# (fresh 800-824); the trial list (the 10 non-reset cells at the picked LRs, kickstart_1M at
# KL 0.01; reset cells wait for Dan); --kickstart-coef only for kickstart_1M; out-dirs under
# $BASE/counted; log counted_queue.log; the lock stays $BASE/sweeps/queue.lock so this can
# never overlap a sweep queue. Usage:
#   counted_queue.sh BC_LIST BC_LIST_SHA256
# BC_LIST is the reviewed BC sha256 list: a first line "# bc_dir=<absolute dir>" then
# `sha256sum` lines "<sha256>  <file name>". Its own sha256 must equal BC_LIST_SHA256 (the
# reviewed and logged value). Before EVERY trial that needs BC the queue
# re-verifies, and EXITS (does not continue) unless every file that cell needs (the 25
# .msgpack + 25 .json for seeds 700-724, plain or _ln) is in the list and matches on disk.
set -u
BASE=${BASE:?set BASE to the run directory holding tars-ppo/, runner/ and the BC checkpoints}
REPO="$BASE/tars-ppo"
RUNNER="$BASE/runner/lr_cell_freshlr.py"
VENV_PY="$REPO/.venv/bin/python"
SWEEPS="$BASE/counted"
LOCKDIR="$BASE/sweeps"
LOG="$SWEEPS/counted_queue.log"
BC_LIST="${1:-}"
BC_LIST_SHA256="${2:-}"
BC_DIR=""   # set only from the reviewed list, below
# The counted runs used runner sha 4098ddd5...; the released copy differs only in comments (reviewer names
# made neutral), which gives the sha below.
EXPECTED_RUNNER_SHA=003959c4d952b799f87dd856d6c9b66bbd509a9dead193132ab2bbe713f24ad5
SEEDS=700-724
CHUNK=5
PRECISION=highest

mkdir -p "$SWEEPS"
exec >>"$LOG" 2>&1
exec 9>"$LOCKDIR/queue.lock"
if ! flock -n 9; then
  echo "FATAL $(date -u '+%Y-%m-%d %H:%M:%S UTC'): another queue holds $LOCKDIR/queue.lock"
  exit 2
fi

ts() { date -u '+%Y-%m-%d %H:%M:%S UTC'; }

echo "==== counted queue start $(ts) pid=$$ script_sha256=$(sha256sum "$0" | awk '{print $1}') bc_list=$BC_LIST bc_list_sha256=$BC_LIST_SHA256 ===="
echo "runner=$RUNNER"
got_sha=$(sha256sum "$RUNNER" | awk '{print $1}')
echo "runner_sha256=$got_sha"
if [[ "$got_sha" != "$EXPECTED_RUNNER_SHA" ]]; then
  echo "FATAL: runner sha mismatch want=$EXPECTED_RUNNER_SHA got=$got_sha"
  exit 2
fi
cd "$REPO" || exit 2
head=$(git rev-parse HEAD)
echo "repo_HEAD=$head"
if [[ "$head" != 33660f6e2f20f289ec6403dc0a2ab4faf365dfd2 ]]; then
  echo "FATAL: unexpected HEAD $head"
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "FATAL: tracked tree dirty"
  git status --porcelain --untracked-files=no
  exit 2
fi
echo "tracked tree clean (untracked results allowed)"

declare -a TRIALS=()
# cell|stream LR (amended-rule pick)|fresh LR|kickstart coef ("-" = flag not passed)
TRIALS+=("scratch_1M|2.5e-4|1e-4|-")
TRIALS+=("pretrained_1M|2.5e-4|1e-4|-")
TRIALS+=("frozen_random_1M|1e-4|1e-4|-")
TRIALS+=("frozen_pretrained_1M|1e-4|1e-4|-")
TRIALS+=("lora_1M|1e-3|1e-4|-")
TRIALS+=("ln_scratch_1M|5e-4|1e-4|-")
TRIALS+=("ln_pretrained_1M|5e-4|1e-4|-")
TRIALS+=("pqn_scratch_1M|1e-4|1e-4|-")
TRIALS+=("pqn_pretrained_1M|1e-4|1e-4|-")
TRIALS+=("kickstart_1M|2.5e-4|1e-4|0.01")

echo "n_trials=${#TRIALS[@]}"
for t in "${TRIALS[@]}"; do echo "planned $t"; done

needs_bc() {
  case "$1" in
    pretrained_1M|lora_1M|frozen_pretrained_1M|ln_pretrained_1M|pqn_pretrained_1M|kickstart_1M) return 0 ;;
    *) return 1 ;;
  esac
}

bc_variant() {
  case "$1" in
    ln_pretrained_1M|pqn_pretrained_1M) echo ln ;;
    *) echo plain ;;
  esac
}

# Exit status 0 only if the reviewed list checks out and every file the cell needs matches.
check_bc() {
  local cell="$1" v s name f want got
  if [[ -z "$BC_LIST" || -z "$BC_LIST_SHA256" || ! -f "$BC_LIST" ]]; then
    echo "BC_GATE: no reviewed BC list given (args: BC_LIST BC_LIST_SHA256)"; return 1
  fi
  got=$(sha256sum "$BC_LIST" | awk '{print $1}')
  if [[ "$got" != "$BC_LIST_SHA256" ]]; then
    echo "BC_GATE: list sha256 $got != reviewed $BC_LIST_SHA256"; return 1
  fi
  BC_DIR=$(sed -n '1s/^# bc_dir=//p' "$BC_LIST")
  if [[ -z "$BC_DIR" || "$BC_DIR" != /* || ! -d "$BC_DIR" ]]; then
    echo "BC_GATE: bad bc_dir line in list: '$BC_DIR'"; BC_DIR=""; return 1
  fi
  v=$(bc_variant "$cell")
  for s in $(seq 700 724); do
    if [[ "$v" == ln ]]; then name="bc_h950_l2_shared_ln_s${s}.msgpack"; else name="bc_h950_l2_shared_s${s}.msgpack"; fi
    for f in "$name" "$name.json"; do
      want=$(awk -v n="$f" '$1 !~ /^#/ && $2 == n {print $1}' "$BC_LIST")
      if [[ -z "$want" || $(printf '%s\n' "$want" | wc -l) -ne 1 ]]; then
        echo "BC_GATE: $f not listed exactly once"; BC_DIR=""; return 1
      fi
      if [[ ! -f "$BC_DIR/$f" ]]; then echo "BC_GATE: missing $BC_DIR/$f"; BC_DIR=""; return 1; fi
      got=$(sha256sum "$BC_DIR/$f" | awk '{print $1}')
      if [[ "$got" != "$want" ]]; then echo "BC_GATE: $f sha256 $got != reviewed $want"; BC_DIR=""; return 1; fi
    done
  done
  echo "BC_GATE OK $cell variant=$v bc_dir=$BC_DIR (50 files match reviewed list)"
  return 0
}

manifest_status() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo ""
    return 0
  fi
  "$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$f" 2>/dev/null || echo ""
}

run_trial() {
  local cell="$1" lr="$2" flr="$3" coef="$4"
  local base="$SWEEPS/${cell}_lr${lr}_flr${flr}"
  [[ "$coef" != - ]] && base="${base}_kl${coef}"
  local out="" status=""

  status=$(manifest_status "$base/manifest.json")
  if [[ "$status" == "complete" ]]; then
    echo "$(ts) SKIP complete $cell lr=$lr flr=$flr kl=$coef out=$base"
    return 0
  fi

  if [[ ! -e "$base/manifest.json" && ! -e "$base/runner_identity.json" ]]; then
    out="$base"
  else
    local n=2
    while true; do
      local cand="${base}_r${n}"
      status=$(manifest_status "$cand/manifest.json")
      if [[ "$status" == "complete" ]]; then
        echo "$(ts) SKIP complete $cell lr=$lr flr=$flr kl=$coef out=$cand"
        return 0
      fi
      if [[ -e "$cand/manifest.json" || -e "$cand/runner_identity.json" ]]; then
        n=$((n+1))
        continue
      fi
      out="$cand"
      break
    done
  fi

  if needs_bc "$cell"; then
    if ! check_bc "$cell"; then
      echo "$(ts) FATAL BC gate failed before $cell lr=$lr flr=$flr kl=$coef: stopping queue (not continuing)"
      exit 3
    fi
  fi

  echo "$(ts) PRECHECK before $cell lr=$lr flr=$flr kl=$coef"
  date -u
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true

  local -a cmd
  cmd=(env PYTHONPATH="$REPO" "$VENV_PY" "$RUNNER" --repo "$REPO" --lr "$lr" --fresh-lr "$flr" --
    --cell "$cell" --seeds "$SEEDS" --chunk-size "$CHUNK" --matmul-precision "$PRECISION"
    --allow-official-seeds --out-dir "$out")
  if needs_bc "$cell"; then
    cmd+=(--bc-dir "$BC_DIR" --critic-warmup-updates 10)
  fi
  if [[ "$coef" != - ]]; then
    cmd+=(--kickstart-coef "$coef")
  fi

  echo "$(ts) START $cell lr=$lr flr=$flr kl=$coef out=$out"
  printf 'CMD '
  printf '%q ' "${cmd[@]}"
  echo

  local start_epoch end_epoch rc
  start_epoch=$(date -u +%s)
  "${cmd[@]}"
  rc=$?
  end_epoch=$(date -u +%s)
  echo "$(ts) END $cell lr=$lr flr=$flr kl=$coef rc=$rc wall_s=$((end_epoch-start_epoch)) out=$out"
  return $rc
}

fail_count=0
ok_count=0
for t in "${TRIALS[@]}"; do
  IFS='|' read -r cell lr flr coef <<<"$t"
  if run_trial "$cell" "$lr" "$flr" "$coef"; then
    ok_count=$((ok_count+1))
  else
    fail_count=$((fail_count+1))
    echo "$(ts) FAILED trial $cell lr=$lr flr=$flr kl=$coef (continuing)"
  fi
done

echo "==== queue done $(ts) ok_or_skip=$ok_count failures=$fail_count ===="
