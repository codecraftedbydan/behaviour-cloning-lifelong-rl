# Reproducing the paper's results

This file covers setup, tests, recomputing every number and figure in the paper from the released run data,
and rerunning the training itself.

## 1. Setup

The counted runs used Python 3.14.4, JAX 0.12.0.1 on one NVIDIA RTX 3060 graphics processing unit (GPU) under
Linux, with matrix-multiply precision `highest` and no XLA (Accelerated Linear Algebra) allocator environment
variables set. `requirements-lock.txt` is the exact package list of that environment.

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt          # GPU; for CPU only, install jax==0.12.0.1 without the cuda12 extras
python -m pytest -q tests
```

The equivalence tests compare against a PureJaxRL checkout. Clone it next to this repo as `../purejaxrl-ref`
(commit 31756b1) or point `PUREJAXRL_DIR` at it; without it those tests skip.

## 2. Recompute the paper's numbers and figure (CPU, about a minute)

Download the release assets (the ten counted-cell bundles, `lr_sweeps_scrubbed.tgz`, `pilot_runs.tgz` and
`SHA256SUMS`) into one directory, then, from this repo's environment:

```bash
bash analysis/extract_bundles.sh <asset dir>    # verifies every SHA-256, unpacks into data/
bash analysis/reproduce_paper_results.sh        # recomputes everything, compares with paper_results/
```

Every line should print `MATCH`. The outputs are compared byte for byte with the files in `paper_results/`:

| File | Produced by | What it holds |
|---|---|---|
| `zero_shot_counted_700-724.json` | `analysis/zero_shot.py` | zero-shot and backward-transfer metrics per cell |
| `h1prime_results.json` | `analysis/h1prime.py` | hypothesis H1′ and per-cell rows |
| `counted_analysis_results.json` | `analysis/counted_analysis.py` | H3, H5, H6 and the paired confidence intervals |
| `addendum_results.json`, `lora_learning_vs_bwt_per_seed_v2.png` | `analysis/addendum.py` | low-rank adaptation (LoRA) addendum and Figure 1 |
| `kickstart_windows_700-724.json` | `analysis/kickstart_windows.py` | per-seed kickstart penalty windows |
| `per_cell_seeds_x_tasks.json`, `revisit_learned_fraction.json` | `analysis/per_cell_seeds_x_tasks.py`, `analysis/revisit_learned_fraction.py` | supporting tables |
| `lr_ranking_raw.json`, `kickstart_ranking_raw.json` | `analysis/rank_lr_raw.py`, `analysis/rank_kickstart_raw.py` | learning-rate (LR) and kickstart coefficient selection |
| `pilot_40/summary.json`, `null_200/summary.json` | `plot_stream.py` on each run's `pilot.npz` | the 40-task pilot and 200-task null run (Sections 4.1 and 4.2) |

`paper_results/cells/<cell>/` holds each counted cell's `manifest.json`, `runner_identity.json` and
`sha256sums.txt`.

### About the released run data

The released bundles differ from the originals in one way: I replaced the absolute run directory on the training
machine with the placeholder `<BASE>` in the JSON and log files (`runner_identity.json`, the queue log snapshots,
and the `sha256sums.txt` files that list them). I did not touch any chunk file (`*.npz`), so every chunk keeps its
original SHA-256, and the `.npz` arrays still hold the original run-directory path in their `bc_dir` field. `paper_results/SCRUB_MANIFEST.json` and `paper_results/SCRUB_MANIFEST_lr_sweeps.json` record, for
every changed file, its old and new SHA-256, and for JSON files a parsed comparison showing that only path
strings changed. `tools/scrub_release.py` is the script that made them.

The pilot and null runs themselves were launched with `scripts/run_pilot.sh` and `scripts/run_null200.sh`;
their launch logs are not included.

## 3. Rerun the training (GPU)

The ten counted cells ran through `runner/counted_queue.sh`, which calls `runner/lr_cell_freshlr.py` on stream
seeds 700 to 724, with fresh-agent seeds 800 to 824, in 5-seed chunks. The queue refuses any runner but the expected
one. The counted runs used runner SHA-256 `4098ddd5…`; the released copy differs from it only in comments, so its
SHA-256 is `003959c4…`, which is what the queue now expects.

1. Pick a run directory and lay it out as `$BASE/tars-ppo` (this repo at the release commit),
   `$BASE/runner/` (the two runner files) and `$BASE/sweeps/` (for the queue lock).
2. Build the behaviour cloning (BC) checkpoints for seeds 700 to 724 with `bc.py` (hidden size 950, two layers,
   plain and layer-norm variants) into `$BASE/tars-ppo/checkpoints/bc_h950_33660f6`. Each checkpoint's
   `.json` metadata records the arguments and task-split hashes it was built with.
3. Edit line 1 of `runner/bc_list_h950_33660f6_700-724.sha256` so `# bc_dir=` holds that absolute directory,
   and check the 50 checkpoint hashes against the list.
4. Run `BASE=<run dir> bash runner/counted_queue.sh <bc list> <sha256 of the bc list>`.

GPU runs are not bitwise reproducible across hardware, drivers or library builds, so a rerun should match the
paper's numbers within seed noise, not byte for byte. Section 2 is the exact check.

## Provenance

All counted cells were produced from commit `33660f6` of this code. This release adds files (licence,
`analysis/`, `runner/`, `tools/`, `paper_results/`, this file), removes run logs, changes one default path
in three files (`scripts/numerical_equivalence.py`, `scripts/run_gate_v2.sh`, `tests/test_equivalence.py`), and
rewords comments, docstrings and two message strings that named internal reviewers. No code logic changed:
`git diff 33660f6` on any source file shows only those lines. Because `provenance.py` hashes all Python and script
files, a run from this release records a different source-tree hash than the counted runs (`6a52b120…`).

## Virtual environment location

The scripts in `scripts/` (for example `run_pilot.sh` and `run_null200.sh`) and two tests call `./.venv/bin/python`.
Create the virtual environment at `./.venv` in the repo root, as in the setup steps above, or symlink `./.venv` to it.
With the environment anywhere else, `test_pilot_script_pinned_and_gated` and `test_null200_script_stubbed_dry_run` fail.

## Hash map: as run and reviewed, then released

The paper cites the hashes of the files that actually ran and were reviewed. The released copies differ from those
files only in code comments and docstrings, the description text (`method`) stored in two results files, and file
paths: input and output paths are now set by environment variables or `BASE`, and `h1prime.py` writes its output
only under `if __name__ == "__main__":`. No computation or number changes. The recompute above checks this by
reproducing every shipped result.

| file | as run or reviewed (SHA-256 prefix) | released | what differs |
|---|---|---|---|
| `runner/lr_cell_freshlr.py` | `4098ddd5` | `003959c4` | comments only |
| `runner/counted_queue.sh` | `9cb4fa55` | `910c299b` | `BASE` from the environment, host name removed, expected runner hash |
| `runner/bc_list_h950_33660f6_700-724.sha256` | `4a5df567` | `bcfc75fb` | run directory replaced by `<BASE>` |
| `analysis/h1prime.py` | `e63d7b30` | `bb65fcab` | paths, output write under `__main__`, comments |
| `analysis/counted_analysis.py` | `beb89648` | `12fe974a` | paths, comments |
| `analysis/rank_lr_raw.py` | `bf77d747` | `d23cddab` | paths |
| `analysis/rank_kickstart_raw.py` | `df0d6b75` | `a9024d38` | paths |
| `analysis/addendum.py` | `f32b519b` | `e3a577e2` | paths, comments |
| `analysis/zero_shot.py` | `248fcb77` | `be4f87dc` | paths, comments |
| `analysis/per_cell_seeds_x_tasks.py` | `c810f063` | `0101e545` | paths, output to current directory, comments |
| `analysis/revisit_learned_fraction.py` | `48aaa72a` | `8a119896` | paths, output to current directory, comments |
| `analysis/kickstart_windows.py` | `a0be1566` | `a0be1566` | none |
| `paper_results/counted_analysis_results.json` | `fa814b9b` | `a2d0641d` | `method` string only (parsed JSON otherwise identical) |
| `paper_results/h1prime_results.json` | `b2afe5e2` | `a44b3b42` | `method` string only (parsed JSON otherwise identical) |

Every analysis script writes its output JSON to the current directory, not to the script's own folder, so run it
from the directory where you want its output.
Every other shipped result file is byte-identical to the reviewed file. Package source files
(`arms.py`, `bc.py`, `cells.py`, `lifelong.py`, `pqn.py`, `provenance.py`, `seeds.py`, `scripts/`, `tests/`) differ
from commit `33660f6` in comments and in the one default path noted above; `git diff 33660f6` shows each change.
Comments in the analysis scripts that cite `fa814b9b` or `e63d7b30` refer to the reviewed files.

Tracked files changed from commit `33660f6` (comments, wording, or the one default path; `git diff 33660f6 -- <file>` shows each):

| file | `33660f6` | released |
|---|---|---|
| `.gitignore` | `31df9755` | `bc5a63f7` |
| `arms.py` | `a393c154` | `8a6d66cd` |
| `bc.py` | `e1f2c28d` | `7ef6deb0` |
| `cells.py` | `8a43ab76` | `3c0b3fdc` |
| `lifelong.py` | `c3a17e0d` | `4c7522bb` |
| `pqn.py` | `2eb7b806` | `6b588858` |
| `provenance.py` | `98151aed` | `ac66bc50` |
| `results/gate_v2.md` | `73e95003` | `7c9be38a` |
| `results/match.md` | `df59bdd9` | `f11e7ee7` |
| `results/match_fresh.md` | `d5d8083a` | `bd9a0f23` |
| `results/stream_dev/README.md` | `8915d18e` | `4fed7cbb` |
| `scripts/bc_dev_checks.py` | `2ac6e703` | `c6e9529f` |
| `scripts/kickstart_sweep.py` | `e5b49b97` | `6e3c1881` |
| `scripts/match.py` | `bd7d8ff4` | `f74dfdce` |
| `scripts/numerical_equivalence.py` | `cfea0cac` | `6e96eb02` |
| `scripts/run_gate_v2.sh` | `0c2c1178` | `9426ee04` |
| `seeds.py` | `556ba141` | `aa49f458` |
| `tests/test_arms.py` | `c1199b75` | `4a264003` |
| `tests/test_cells.py` | `3dc872a9` | `6d0b9694` |
| `tests/test_equivalence.py` | `49d4d147` | `0ca705fa` |
| `tests/test_pqn.py` | `b41ba2f7` | `a322185a` |
| `tests/test_provenance.py` | `78f9ee88` | `4bdc0d65` |
