# Match check: tars-ppo vs PureJaxRL (CPU, 2026-10-23)

**Rule (pre-set):** same env and step budget, 5 seeds. **PASS iff my IQM final return
falls inside PureJaxRL's 95% bootstrap CI.**

**Metric (identical for both).** Per update, take the return of the most recently finished
episode in each env, read at the end of the rollout, and average over envs. Final return is the
mean of that over the last 10 updates. IQM drops the top and bottom 25% of seeds (1 of 5 each
side). The CI is a percentile bootstrap over seeds (10,000 resamples).
For PureJaxRL this is `info["returned_episode_returns"][:, -1, :].mean(-1)` from its `LogWrapper`.
Its usual plotting definition (mean over *all* T steps x envs) gives almost the same IQM:
489.8 vs 490.5 (CartPole) and 49.03 vs 48.96 (Breakout).

## Results

| env (budget per seed) | seeds | this repo: final per seed | this repo IQM [95% CI] | PureJaxRL: final per seed | PureJaxRL IQM [95% CI] | rule |
|---|---|---|---|---|---|---|
| CartPole-v1 (499,712) | 5 (key 0) | 500, 500, 500, 500, 500 | **500.0** [500.0, 500.0] | 490.3, 454.8, 500.0, 481.2, 500.0 | 490.5 [463.6, 500.0] | **PASS** (at the CI's upper edge) |
| Breakout-MinAtar (9,994,240) | 5 (key 0) | 25.3, 40.1, 41.5, 33.5, 23.5 | **33.0** [24.1, 41.0] | 50.0, 42.6, 28.9, 54.3, 56.8 | 49.0 [33.5, 55.9] | **FAIL** (0.5 below the lower bound) |

The two rows above are the pre-registered comparison. The rows below are diagnostics run
*after* the FAIL. They are not a re-run to get a pass.

| diagnostic (Breakout-MinAtar) | this repo IQM [CI] | PureJaxRL IQM [CI] | rule |
|---|---|---|---|
| seed set 2 (key 1), 5 seeds each | 60.3 [46.0, 71.5] (74.0, 55.8, 58.8, 39.7, 66.4) | 43.6 [26.6, 59.8] (45.6, 61.8, 55.8, 29.5, 25.1) | FAIL, but this time this repo is *above* |
| pooled 10 vs 10 seeds | 44.9 [32.9, 58.4] (mean 45.9) | 46.3 [35.0, 54.9] (mean 45.0) | PASS; permutation test on means p = 0.55 |
| this repo with truncation bootstrap off, key 0 | 33.0, bit-identical to the main run | n/a | n/a |

Plots: `match_cartpole.png`, `match_minatar.png` (the two pre-registered comparisons),
`match_minatar_seed1.png`, `match_minatar_pooled10.png`. Raw numbers: `match_summary.jsonl`;
PureJaxRL arrays: `purejaxrl_ref/*.npz`.

## Why Breakout FAILed: likely cause is seed variance, not an implementation difference

1. **Time-limit handling (the one deliberate algorithmic difference): ruled out exactly.** With
   `--no-truncation-bootstrap` (PureJaxRL's behaviour), every per-update return and value loss is
   bit-identical to the main run. So no Breakout episode hit the 1000-step limit in these runs,
   and the bootstrap term was always multiplied by zero.
2. **Network, advantage normalisation, value clipping, optimiser: identical by construction.**
   ReLU 64x2 separate actor/critic, same orthogonal gains, flattened obs, per-minibatch advantage
   normalisation, value clipping at 0.2, Adam eps 1e-5, global-norm clip 0.5. The one remaining
   difference is LR-schedule granularity: optax decays per gradient step, while PureJaxRL keeps
   the LR constant within an update. That shifts the LR by less than 1/1220 of its range.
3. **The direction flips with the seed set.** Key 0 puts this repo 16 points below; key 1 puts this repo
   17 points above. Pooled over 10 seeds the IQMs differ by 1.4 and the means by 0.9 (p = 0.55).
   The per-seed spread *within* each method (23–74 for this repo, 25–62 for PureJaxRL) is far larger
   than the difference between them.
4. **The rule is underpowered for a high-variance env.** It treats my 5-seed IQM as exact and only
   accounts for PureJaxRL's sampling noise. On Breakout, a 5-seed IQM moves by more than ±15
   between seed sets of the *same* code (PureJaxRL: 49.0 vs 43.6; this repo: 33.0 vs 60.3). So two
   correct implementations will often FAIL. Suggested fix:
   use 10+ seeds, or test the *difference* (bootstrap CI of IQM_ours - IQM_ref must contain 0),
   or require the two CIs to overlap.

CartPole PASSes but is a weak test, as expected. This repo is 5/5 at 500 and PureJaxRL is 2/5. The
gap may partly come from truncation bootstrapping, which does fire on CartPole every time an
episode reaches 500 steps. I didn't ablate it on CartPole.

## Setup and wall-clock (8-core development CPU, jax 0.12.0.1 CPU, both in separate venvs on identical jax/flax/optax/gymnax versions, runs sequential)

| run | this repo: run + compile | this repo steps/s | PureJaxRL: run + compile | PureJaxRL steps/s |
|---|---|---|---|---|
| CartPole, 5 seeds x 5e5 | 12.6 s + 3.3 s | 199k | 10.6 s + 2.3 s | 235k |
| Breakout, 5 seeds x 1e7 (key 0) | 376.6 s + 3.9 s | 133k | 299.2 s + 2.7 s | 167k |
| Breakout, key 1 | 372.8 s + 4.2 s | 134k | 286.7 s + 2.8 s | 174k |
| Breakout, this repo with bootstrap off | 320.9 s + 4.1 s | 156k | n/a | n/a |

This repo is about 20% slower. About 15 points of that is the extra critic forward pass on the pre-reset
obs that truncation bootstrapping needs; the bootstrap-off run recovers it. The rest is my eval hook
(8 episodes x 1000 steps every 20 updates) and the per-update network-health pass. Steps/s counts
training env steps over all 5 seeds and excludes eval steps.

**Configs.**
* CartPole: PureJaxRL `purejaxrl/ppo.py` `__main__` defaults (4 envs, T=128, lr 2.5e-4 annealed,
  4 epochs x 4 minibatches, tanh 64x2). This repo uses its defaults, which are the same values.
* Breakout: PureJaxRL has no MinAtar script. Its MinAtar config is the "PPO Gymnax MinAtar" cell of
  `examples/brax_minatar.ipynb`: 64 envs, T=128, lr 5e-3 annealed, 4 epochs x 8 minibatches, ReLU
  64x2, ent 0.01, clip 0.2, 1e7 steps. The notebook's env is SpaceInvaders-MinAtar; I swapped in
  Breakout-MinAtar as asked, with no other changes. This repo: `--num-envs 64 --num-steps 128
  --num-minibatches 8 --lr 5e-3 --activation relu --hidden-size 64 --eval-every 20` (so
  1220 updates, exactly PureJaxRL's budget). The full 1e7 budget fitted: about 5.0 min (PureJaxRL) and 6.3 min (this repo) per 5-seed run.

**How PureJaxRL was run.** It was cloned to `../purejaxrl-ref` (commit 31756b1, 2024-09-09),
and `purejaxrl/ppo.py` was **unmodified**. The runner (`scripts/run_purejaxrl_ref.py`, copied
from `../purejaxrl-ref/run_ref.py`) imports its `make_train`, vmaps it over 5 seeds
from `PRNGKey(seed)` (same seed keys as this repo) and jits it. `wrappers.py` imports brax and navix at
module level but only uses them in Brax/Navix wrapper classes that are never touched, so I stubbed
those two imports instead of installing them.

## Caveats
* Seeds are "matched" only in the top-level key. The two codebases consume randomness differently,
  so seed *i* of this repo and seed *i* of PureJaxRL are unrelated runs.
* A 5-seed bootstrap CI is coarse, and on CartPole it sits at the 500 ceiling, so the PASS is at the
  edge of the interval.
* CPU only. GPU timings on the 3060 could rank the two differently.
* Side observation (this repo, Breakout): the tau-dormant fraction (tau=0.1) is 55–66% at init (ReLU on
  sparse binary obs) and still 24–45% at the end. That is already relevant to the plasticity work.
  PureJaxRL doesn't log this, so I can't compare.
