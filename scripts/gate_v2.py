"""Baseline gate v2 (plans/tars-test-rules.md section 0, locked 23 Oct before the run).

purejaxrl_match preset vs unmodified PureJaxRL, 1e7 env steps, seeds 200-209,
on Breakout, Asterix, SpaceInvaders, Freeway (MinAtar).
  * Per env: the 95% bootstrap CI of IQM(ours) - IQM(PureJaxRL) includes 0.
  * Across envs: if ours is lower (IQM diff < 0) on all 4, FAIL even if
    every CI includes 0.
  * PASS needs both.
Definitions are unchanged from scripts/match_fresh.py. Final return per seed is
the mean over the last 10 updates of the end-of-rollout, env-averaged return of
the last finished episode. The bootstrap resamples seeds independently within
each method (10,000 resamples, rng seed 0) and takes a percentile CI of the IQM
difference. Also reported, without changing the verdict: whether ours is HIGHER
on all 4 (the rule's "no consistent sign" wording could be read to include that).

    python scripts/gate_v2.py --dir results/gate_v2
expects <dir>/{ours,ref}_<Env>_{200-204,205-209}.npz
"""
import argparse, glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot import bootstrap_ci, iqm, smooth  # noqa: E402

ENVS = ["Breakout", "Asterix", "SpaceInvaders", "Freeway"]
SEEDS = list(range(200, 210))
LAST, N_BOOT = 10, 10_000


def load(files, key):
    ds = [np.load(f) for f in sorted(files)]
    return (np.concatenate([d[key] for d in ds]), np.concatenate([d["seeds"] for d in ds]),
            ds[0]["env_steps"], sum(float(d["run_seconds"]) for d in ds))


def analyse(of, rf):
    o_iqm, r_iqm = iqm(of), iqm(rf)
    rng = np.random.default_rng(0)
    io = rng.integers(0, len(of), size=(N_BOOT, len(of)))
    ir = rng.integers(0, len(rf), size=(N_BOOT, len(rf)))
    boot = np.array([iqm(of[i]) - iqm(rf[j]) for i, j in zip(io, ir)])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return o_iqm, r_iqm, o_iqm - r_iqm, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/gate_v2")
    a = p.parse_args()
    rows, curves = [], {}
    for env in ENVS:
        o_ret, o_seeds, o_steps, o_run = load(glob.glob(f"{a.dir}/ours_{env}_*.npz"), "metric_episode_return")
        r_ret, r_seeds, r_steps, r_run = load(glob.glob(f"{a.dir}/ref_{env}_*.npz"), "ret_end_of_rollout")
        assert sorted(o_seeds.tolist()) == SEEDS and sorted(r_seeds.tolist()) == SEEDS, (env, o_seeds, r_seeds)
        assert o_steps[-1] == r_steps[-1], f"{env}: budget mismatch"
        of, rf = o_ret[:, -LAST:].mean(1), r_ret[:, -LAST:].mean(1)
        o_iqm, r_iqm, d, lo, hi = analyse(of, rf)
        o_ci, r_ci = bootstrap_ci(of[:, None], n_boot=N_BOOT), bootstrap_ci(rf[:, None], n_boot=N_BOOT)
        rows.append(dict(env=f"{env}-MinAtar", budget=int(o_steps[-1]),
                         ours_iqm=round(float(o_iqm), 2), ours_ci=[round(float(o_ci[0][0]), 2), round(float(o_ci[1][0]), 2)],
                         ref_iqm=round(float(r_iqm), 2), ref_ci=[round(float(r_ci[0][0]), 2), round(float(r_ci[1][0]), 2)],
                         ours_mean=round(float(of.mean()), 2), ref_mean=round(float(rf.mean()), 2),
                         diff=round(float(d), 2), diff_ci=[round(float(lo), 2), round(float(hi), 2)],
                         rel_diff_pct=round(float(100 * d / r_iqm), 1) if r_iqm else None,
                         ci_contains_0=bool(lo <= 0 <= hi),
                         ours_final=[round(float(x), 2) for x in of], ref_final=[round(float(x), 2) for x in rf],
                         ours_run_s=round(o_run, 1), ref_run_s=round(r_run, 1)))
        curves[env] = (o_steps, o_ret, r_steps, r_ret)
    all_ci = all(r["ci_contains_0"] for r in rows)
    all_lower = all(r["diff"] < 0 for r in rows)
    all_higher = all(r["diff"] > 0 for r in rows)
    verdict = "PASS" if (all_ci and not all_lower) else "FAIL"
    out = dict(rule="per-env 95% bootstrap CI of IQM diff contains 0, AND NOT (ours lower on all 4)",
               seeds=SEEDS, envs=rows, all_ci_contain_0=all_ci, ours_lower_on_all=all_lower,
               ours_higher_on_all=all_higher, verdict=verdict)
    json.dump(out, open(f"{a.dir}/gate_v2.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, env, r in zip(axes, ENVS, rows):
        o_steps, o_ret, r_steps, r_ret = curves[env]
        for steps, ret, label, c in ((r_steps, r_ret, "PureJaxRL", "tab:blue"), (o_steps, o_ret, "ours", "tab:red")):
            s = smooth(ret, 10)
            lo, hi = bootstrap_ci(s)
            ax.plot(steps, iqm(s), color=c, lw=2, label=label)
            ax.fill_between(steps, lo, hi, color=c, alpha=0.18)
        ax.set_title(f"{env}: diff {r['diff']:+.1f} [{r['diff_ci'][0]:+.1f}, {r['diff_ci'][1]:+.1f}]", fontsize=10)
        ax.set_xlabel("env steps"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("episode return (IQM, 10 seeds, 95% CI)"); axes[0].legend(fontsize=9)
    fig.suptitle(f"Gate v2: purejaxrl_match vs PureJaxRL, seeds 200-209, 1e7 steps: {verdict}")
    fig.tight_layout(); fig.savefig(f"{a.dir}/gate_v2.png", dpi=130)
    print(json.dumps({k: v for k, v in out.items() if k != "envs"}, indent=1))
    for r in rows:
        print(f"{r['env']:22s} ours {r['ours_iqm']:7.2f} ref {r['ref_iqm']:7.2f} diff {r['diff']:+7.2f} "
              f"CI [{r['diff_ci'][0]:+.2f}, {r['diff_ci'][1]:+.2f}] contains0={r['ci_contains_0']}")


if __name__ == "__main__":
    main()
