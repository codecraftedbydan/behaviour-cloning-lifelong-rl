# Baseline gate: fresh pre-registered match, Breakout-MinAtar (tars-ppo vs PureJaxRL)

## Pre-registration (written and committed before any fresh-seed run)
* **Rule** (`re-portfolio/plans/tars-test-rules.md` section 0): 10 seeds each.
  **(A)** the 95% bootstrap CI of IQM(this repo) - IQM(PureJaxRL) contains 0, **and**
  **(B)** |IQM(this repo) - IQM(PureJaxRL)| <= 10% of IQM(PureJaxRL). PASS needs both.
* **Seeds:** 100-109, one `PRNGKey(s)` per seed, the same list for both methods. None of these seeds were
  used before; earlier runs used split(PRNGKey(0|1), 5).
* **Env/budget:** Breakout-MinAtar, 1220 updates x 8192 = 9,994,240 env steps per seed for both.
* **This repo:** `python ppo.py --preset purejaxrl_match --env Breakout-MinAtar --total-timesteps 10000000`.
  This preset is the exact config of the earlier Breakout run (checked field by field), packaged under a
  name. Nothing changed or tuned.
* **PureJaxRL:** unmodified `purejaxrl/ppo.py` (commit 31756b1) with its notebook MinAtar config,
  env = Breakout-MinAtar, via `scripts/run_purejaxrl_ref.py --preset minatar`.
* **Metric:** final return per seed = mean over the last 10 updates of the end-of-rollout, env-averaged
  return of the last finished episode. IQM over seeds (middle 6 of 10). Bootstrap: 10,000
  resamples of seeds within each method, percentile CI of the IQM difference (`scripts/match_fresh.py`).
* **Execution:** each method runs as two vmapped groups of 5 seeds (100-104, 105-109) because of development-machine
  memory. Seeds are independent, so grouping does not change what is computed. Runs are sequential.

## Result (runs 11:31–11:53 BST, 23 Oct 2026; analysis by `scripts/match_fresh.py`)

| | this repo (preset `purejaxrl_match`) | PureJaxRL (unmodified) |
|---|---|---|
| final return, seeds 100–109 | 35.0, 41.9, 22.5, 77.5, 60.7, 25.4, 72.4, 38.8, 64.7, 46.5 | 52.0, 42.2, 82.2, 85.8, 26.3, 60.0, 28.9, 85.3, 23.3, 71.9 |
| IQM [95% CI] | **47.9** [33.9, 63.4] | **56.2** [35.2, 76.4] |
| mean | 48.5 | 55.8 |

**IQM difference (this repo - PureJaxRL) = -8.28, 95% bootstrap CI [-33.11, +17.29]**

| criterion | requirement | value | result |
|---|---|---|---|
| (A) CI of the difference contains 0 | -33.11 <= 0 <= +17.29 | contains 0 | **PASS** |
| (B) \|difference\| <= 10% of PureJaxRL's IQM | 8.28 <= 5.62 | gap = 14.7% | **FAIL** |
| **Gate (A and B)** | | | **FAIL** |

Plot: `match_fresh.png`. Raw: `match_fresh.json`, `fresh/*.npz`.

**Wall-clock** (8-core CPU, groups of 5 seeds, sequential): this repo 339.2 s + 4.3 s compile and
344.5 s + 3.9 s (147k / 145k env-steps/s). PureJaxRL 290.3 s + 2.6 s and 280.7 s + 2.6 s
(172k / 178k env-steps/s). Total 21.3 min.

## Explanation of the FAIL (exploratory: no new runs, and the verdict above stands)

The rules require a FAIL to be explained before further runs. The checks below use only runs that
already exist (20 seeds per method: 10 earlier + 10 fresh). They are labelled exploratory because
the earlier 10 seeds were seen before the rule was set.

1. **Criterion B cannot be met reliably at 10 seeds on this env, even by identical code.** Per-seed
   final return sd is about 20 (PureJaxRL 20.2, this repo 17.8) on an IQM of about 50. Splitting
   *PureJaxRL's own* 20 seeds into two random groups of 10, PureJaxRL passes B against itself only
   **33%** of the time (this repo vs itself: 34%). With 10 seeds the bootstrap CI half-width on the
   difference is about 20 points, while the B tolerance is about 5. Getting the CI half-width down
   to the tolerance would take roughly 150+ seeds per method.
2. **There is no sign of a real difference.** Pooled 20 vs 20: IQM 45.9 (this repo) vs 48.9 (PureJaxRL),
   difference -3.0, CI [-17.9, +12.2], 6.2% of PureJaxRL's IQM; permutation test on means
   p = 0.60. The sign of the gap has flipped between seed blocks: -16 on block key 0, +17 on
   block key 1, -8 on seeds 100–109.
3. **Implementation differences already checked.** The network, losses, advantage normalisation,
   value clipping and optimiser are identical by construction. Truncation bootstrapping never
   fires on Breakout (the bootstrap-off run is bit-identical). LR-schedule granularity is
   negligible (under 1/1220).

**Likely cause: seed variance on a high-variance env, with criterion B applied to a point estimate.**
Nothing points to an implementation difference. Options (the gate
is theirs to change, and any change needs a dated log entry plus another fresh seed block):
apply B to the CI rather than the point gap; use a lower-variance gate env or metric (e.g. CartPole
or Acrobot plus a longer budget, or greedy eval over more episodes); or accept A plus a pooled
equivalence check. I have not changed the rule or rerun anything.
