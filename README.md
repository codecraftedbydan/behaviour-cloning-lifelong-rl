# TARS: does behaviour-cloning pretraining protect a lifelong reinforcement learning agent against primacy bias?

This repository holds the code, results and figures for my pre-registered study of whether a pretrained start helps or hinders an agent that keeps learning over a long stream of tasks.

- **Paper:** [Portable Document Format (PDF) version](paper.pdf)
- **Blog post:** Forthcoming

## Summary

I asked whether starting a reinforcement learning (RL) agent from behaviour cloning (BC) on planner demonstrations softens the collapse in learning that follows the first task switch in a lifelong stream, a pattern known as primacy bias. The agent is trained with proximal policy optimisation (PPO), has about one million parameters, and learns a stream of 40 new 8×8 gridworld tasks with earlier tasks revisited. Ten cells, each on 25 seeds, also vary the update method and the algorithm. No pre-registered hypothesis was supported. The primary pretraining effect on late relative adaptation (hypothesis H1′) was D = +0.045, with a 95% confidence interval (CI) of [+0.000, +0.088]. The lower bound sits at 0 within bootstrap noise, so I do not read the interval as excluding 0, and D is about half the pre-registered minimum gap of 0.093: any benefit was not detected at this resolution. One result reversed significantly (hypothesis H6): with only the heads trained, a frozen random trunk beat a frozen pretrained trunk, −0.109 [−0.149, −0.073], although both ran at 10⁻⁴, the bottom edge of their learning-rate grid. Low-rank adaptation (LoRA) learned late tasks best of the PPO cells but forgot the most, and it ran at a learning rate of 10⁻³ against 2.5 × 10⁻⁴ for full fine-tuning, so that comparison is confounded.

## What is in this repository

All counted runs used the code at commit `33660f6`. This release adds the licence, the analysis and runner scripts, the paper's results files and `REPRODUCE.md`, removes run logs, and changes one default path in three files. Every other source file is byte-identical to `33660f6`. The main pieces are:

| File | What it does |
|---|---|
| `ppo.py` | PPO in JAX, with the whole training loop in one jitted function; checked against PureJaxRL |
| `pqn.py` | The parallelised Q-network (PQN) agent used in the value-based cells |
| `gridworld.py`, `stream.py` | The 8×8 gridworld, the task stream (goal, dynamics and observation shifts, with revisits) and the demonstration pool |
| `expert.py`, `bc.py` | The shortest-path planner that produces demonstrations, and behaviour-cloning pretraining |
| `arms.py`, `cells.py` | The update-method arms (full fine-tuning, LoRA, frozen trunks, kickstarting) and the stage-1 cell runner, one named cell per run |
| `lifelong.py`, `stream_metrics.py` | The lifelong training loop with its fresh-agent baseline, and the stream metrics, including relative adaptation |
| `seeds.py`, `provenance.py` | The reserved seed blocks, and the run records (commit, source hashes, clean-tree check) |
| `runner/` | `lr_cell_freshlr.py`, the wrapper every counted run went through (it runs `cells.py run` unchanged but sets the stream learning rate and, separately, the fresh-baseline learning rate), the queue script `counted_queue.sh`, and the list of behaviour-cloning checkpoint hashes |
| `analysis/` | The scripts that turn the run data into every number and the figure in the paper, plus `extract_bundles.sh` and `reproduce_paper_results.sh` |
| `paper_results/` | The results files (in JavaScript Object Notation (JSON) format) that the paper's numbers come from, the paper's figure `lora_learning_vs_bwt_per_seed_v2.png`, each counted cell's manifest, and the summaries of the 40-task pilot (`pilot_40/`) and the 200-task run (`null_200/`) from Sections 4.1 and 4.2 |
| `scripts/` | Development checks, the PureJaxRL equivalence check, the kickstart sweep, and the launch scripts for the pilot and 200-task runs |
| `tools/scrub_release.py` | The script that replaced the training machine's run directory with a placeholder in the released run data (see `REPRODUCE.md`) |
| `results/` | Outputs of the development and equivalence checks, not used for any counted result |
| `tests/` | The test suite |

The raw run data is too large for the repository, so it is attached to the release as assets: the ten counted-cell bundles, `lr_sweeps_scrubbed.tgz`, `pilot_runs.tgz` and a `SHA256SUMS` file of Secure Hash Algorithm 256-bit (SHA-256) hashes, about 347 megabytes in total. Launch logs and scratch outputs are left out.

## Setup

The counted runs used Python 3.14.4 and JAX 0.12.0.1 on one NVIDIA RTX 3060 graphics processing unit (GPU) under Linux. `requirements-lock.txt` is the exact package list of that environment.

```bash
python3.14 -m venv .venv && . .venv/bin/activate
pip install -r requirements-lock.txt   # GPU; for central processing unit (CPU) only, install jax==0.12.0.1 without the cuda12 extras
python -m pytest -q tests
```

The equivalence tests compare against a PureJaxRL checkout. Clone it next to this repository as `../purejaxrl-ref` (commit `31756b1`) or point `PUREJAXRL_DIR` at it; without it those tests are skipped.

## Reproducing the results

There are two levels. `REPRODUCE.md` has the full detail for both.

**Recompute every number and the figure from the released run data (CPU, about a minute).** Download the release assets into one directory, then:

```bash
bash analysis/extract_bundles.sh <asset dir>   # checks every SHA-256 hash, unpacks into data/
bash analysis/reproduce_paper_results.sh       # recomputes everything, compares with paper_results/
```

Every line should print `MATCH`: the outputs are compared byte for byte with the files in `paper_results/`. The released bundles differ from the originals only in that the training machine's absolute run directory is replaced by a placeholder in a few metadata files. No chunk of run data was changed, and `paper_results/SCRUB_MANIFEST.json` records every changed file with its old and new hash.

**Rerun the training (GPU).** The ten cells ran through `runner/counted_queue.sh`, which calls `runner/lr_cell_freshlr.py` on stream seeds 700–724, with fresh-agent seeds 800–824, in seed batches of five, at the learning rates in the results table below. In outline:

1. Lay out a run directory as `$BASE/tars-ppo` (this repository), `$BASE/runner/` and `$BASE/sweeps/`.
2. Build the behaviour-cloning checkpoints for seeds 700–724 with `bc.py` (hidden size 950, two layers, plain and layer-norm variants).
3. Point line 1 of `runner/bc_list_h950_33660f6_700-724.sha256` at that checkpoint directory and check the 50 checkpoint hashes against the list.
4. Run `BASE=<run dir> bash runner/counted_queue.sh <bc list> <sha256 of the bc list>`.

Because `provenance.py` hashes every source file, a run from this release records a different source-tree hash from the counted runs, as expected.

**Hardware and determinism.** Every counted cell ran on a single NVIDIA RTX 3060 GPU, at full 32-bit matrix precision (the "highest" matrix-multiplication precision setting in JAX), from one queue on one driver version. GPU runs at this precision are not bitwise reproducible from run to run, even with the same commit, seeds and software, so a training rerun should match the paper within seed noise, not bit for bit. The recompute step above is the exact check. Results also depend on how seeds are grouped into batches, so use the recorded seed batches of five.

**Seeds.**

| Use | Stream seeds | Fresh-agent seeds |
|---|---|---|
| Counted runs | 700–724 | 800–824 (fresh seed 800+i paired with stream seed 700+i) |
| Learning-rate and kickstart tuning only | 750–759 | 850–859 |
| Development checks only, never used for ranking | 900 and above | |

## Results

Late relative adaptation is the stream agent's area under the curve (AUC) of normalised training return on new tasks 21–40, minus a fresh agent's AUC on the same task. Higher is better, and 0 would mean learning as fast as a fresh agent. The primary aggregate is the interquartile mean (IQM) over seeds × tasks. The per-seed version (each seed's mean over tasks, then the IQM over seeds, with 95% CIs over seeds) is secondary. The table is descriptive; only the pre-registered contrasts have pass rules.

| Cell | Stream learning rate | Fresh reference | Late AUC_rel, seeds × tasks IQM (primary) | Raw stream late AUC, seeds × tasks IQM (primary) | Late AUC_rel, per-seed IQM [95% CI] (secondary) | Raw stream late AUC, per-seed IQM [95% CI] (secondary) |
|---|---|---|---|---|---|---|
| Scratch | 2.5 × 10⁻⁴ | PPO | −0.860 | 0.016 | −0.718 [−0.764, −0.667] | 0.180 [0.138, 0.238] |
| Pretrained, full fine-tuning | 2.5 × 10⁻⁴ | PPO | −0.815 | 0.063 | −0.692 [−0.730, −0.643] | 0.215 [0.167, 0.269] |
| Frozen pretrained trunk | 10⁻⁴ (edge) | PPO | −0.910 | 0.000 | −0.865 [−0.886, −0.840] | 0.032 [0.014, 0.057] |
| Frozen random trunk | 10⁻⁴ (edge) | PPO | −0.800 | 0.070 | −0.690 [−0.749, −0.626] | 0.213 [0.154, 0.284] |
| LoRA | 10⁻³ | PPO | −0.463 | 0.433 | −0.481 [−0.501, −0.456] | 0.419 [0.394, 0.458] |
| Kickstart (Kullback–Leibler (KL) coefficient 0.01) | 2.5 × 10⁻⁴ | PPO | −0.807 | 0.068 | −0.666 [−0.700, −0.632] | 0.247 [0.205, 0.280] |
| PPO with layer normalisation (LN), scratch | 5 × 10⁻⁴ | PPO+LN | −0.889 | 0.003 | −0.739 [−0.790, −0.692] | 0.153 [0.100, 0.196] |
| PPO+LN, pretrained | 5 × 10⁻⁴ | PPO+LN | −0.871 | 0.010 | −0.741 [−0.787, −0.678] | 0.157 [0.109, 0.198] |
| PQN scratch | 10⁻⁴ (edge) | PQN | −0.075 | 0.095 | −0.036 [−0.070, −0.009] | 0.177 [0.158, 0.205] |
| PQN pretrained | 10⁻⁴ (edge) | PQN | −0.075 | 0.088 | −0.046 [−0.088, −0.009] | 0.179 [0.159, 0.201] |

PQN's relative adaptation is close to 0 because its own fresh agent is weak, not because PQN streams learn well, so the PQN rows are not comparable with the PPO rows on that column. Read the raw AUC columns beside them.

## Limitations

- **One network size.** Every counted result is for about one million parameters. The size line (hypothesis H4) was dropped for compute budget before any run, so I make no claim about how the results scale.
- **One environment family.** All tasks are fully observed 8×8 gridworlds, with 40 new tasks per stream, one wall layout per seed and a fixed demonstration set.
- **Learning rates at the edge of the grid.** Both frozen-trunk cells, both PQN cells and all three fresh baselines chose 10⁻⁴, the bottom of the grid, so some heads and baselines may be under-tuned. The H6 reversal rests on edge learning rates.
- **LoRA's learning rate.** LoRA ran at 10⁻³ against 2.5 × 10⁻⁴ for full fine-tuning, so its better late learning and its extra forgetting cannot be put down to the low-rank update alone.
- **Weak PQN baseline.** PQN's fresh agent learns little in 100 updates, so PQN's relative adaptation says little about plasticity.
- **Post-hoc change to the learning-rate rule.** I changed the selection rule after seeing the sweep and before any counted run, because the original rule rewarded learning rates that weaken the fresh baseline. The original rule's picks are reported as a sensitivity result.
- **Where the collapse sits is not tested.** The reset arms (hypothesis H7) were not run.
- **Not bitwise reproducible.** See "Hardware and determinism" above.

## How the work was done

I designed and directed this study and made every design decision and ruling in it. The code and the analysis were built with artificial intelligence (AI) assistance. Every analysis that counts towards a result was independently re-run and checked before I accepted it, and the counted analyses reproduced their results files byte for byte. The test rules were fixed before the runs they govern, and every later change is dated in the paper with its reason.

## Citation

```bibtex
@misc{shaw2026tars,
  author       = {Daniel Shaw},
  title        = {Does pretraining make a reinforcement learning agent a better or worse lifelong learner? {A} pre-registered test of behaviour cloning against primacy bias},
  year         = {2026},
  howpublished = {\url{https://github.com/codecraftedbydan/behaviour-cloning-lifelong-rl}},
  note         = {Code and paper}
}
```

## Licence

Released under the MIT Licence; see [LICENSE](LICENSE).
