"""Baseline gate (plans/tars-test-rules.md, section 0) on a fresh seed block.

PASS iff BOTH:
  (A) the 95% bootstrap CI of IQM(ours) - IQM(PureJaxRL) contains 0, and
  (B) |IQM(ours) - IQM(PureJaxRL)| <= 0.10 * IQM(PureJaxRL).
Final return per seed = mean over the last 10 updates of the end-of-rollout,
env-averaged return of the last finished episode (same definition for both).
Bootstrap: resample seeds with replacement independently within each method
(10,000 resamples), recompute both IQMs, take the difference, percentile CI.
With a single env this is the same as rliable's stratified bootstrap.

    python scripts/match_fresh.py --ours a.npz b.npz --ref c.npz d.npz \
        --out results/match_fresh.png --json results/match_fresh.json
"""
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot import bootstrap_ci, iqm, smooth  # noqa: E402

LAST, N_BOOT, TOL = 10, 10_000, 0.10


def load(files, key):
    ds = [np.load(f) for f in files]
    ret = np.concatenate([d[key] for d in ds])
    seeds = np.concatenate([d["seeds"] for d in ds])
    run_s = [float(d["run_seconds"]) for d in ds]
    comp_s = [float(d["compile_seconds"]) for d in ds]
    sps = [float(d["steps_per_sec"]) for d in ds]
    return ret, seeds, ds[0]["env_steps"], run_s, comp_s, sps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", nargs="+", required=True)
    p.add_argument("--ref", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--json", required=True)
    a = p.parse_args()

    o_ret, o_seeds, o_steps, o_run, o_comp, o_sps = load(a.ours, "metric_episode_return")
    r_ret, r_seeds, r_steps, r_run, r_comp, r_sps = load(a.ref, "ret_end_of_rollout")
    assert o_steps[-1] == r_steps[-1], "budget mismatch"
    assert sorted(o_seeds.tolist()) == sorted(r_seeds.tolist()), "seed lists differ"

    of, rf = o_ret[:, -LAST:].mean(1), r_ret[:, -LAST:].mean(1)
    o_iqm, r_iqm = iqm(of), iqm(rf)
    diff = o_iqm - r_iqm

    rng = np.random.default_rng(0)
    io = rng.integers(0, len(of), size=(N_BOOT, len(of)))
    ir = rng.integers(0, len(rf), size=(N_BOOT, len(rf)))
    boot_diff = np.array([iqm(of[i]) - iqm(rf[j]) for i, j in zip(io, ir)])
    d_lo, d_hi = np.quantile(boot_diff, [0.025, 0.975])
    o_lo, o_hi = bootstrap_ci(of[:, None], n_boot=N_BOOT)
    r_lo, r_hi = bootstrap_ci(rf[:, None], n_boot=N_BOOT)

    crit_a = bool(d_lo <= 0.0 <= d_hi)
    crit_b = bool(abs(diff) <= TOL * r_iqm)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for steps, ret, label, c in ((r_steps, r_ret, "PureJaxRL (unmodified)", "tab:blue"),
                                  (o_steps, o_ret, "ours (preset purejaxrl_match)", "tab:red")):
        s = smooth(ret, 10)
        lo, hi = bootstrap_ci(s)
        ax.plot(steps, iqm(s), color=c, lw=2, label=f"{label}: IQM, {ret.shape[0]} seeds")
        ax.fill_between(steps, lo, hi, color=c, alpha=0.18)
    ax.set_xlabel("environment steps (per seed)")
    ax.set_ylabel("episode return")
    ax.set_title(f"Breakout-MinAtar, fresh seeds {int(o_seeds.min())}-{int(o_seeds.max())}: "
                 f"diff {diff:+.1f} [{d_lo:+.1f}, {d_hi:+.1f}]  "
                 f"A:{'PASS' if crit_a else 'FAIL'} B:{'PASS' if crit_b else 'FAIL'}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)

    out = dict(
        seeds=[int(s) for s in o_seeds], budget=int(o_steps[-1]),
        ours_final=[round(float(x), 2) for x in of], ref_final=[round(float(x), 2) for x in rf],
        ours_iqm=round(float(o_iqm), 2), ours_ci=[round(float(o_lo[0]), 2), round(float(o_hi[0]), 2)],
        ref_iqm=round(float(r_iqm), 2), ref_ci=[round(float(r_lo[0]), 2), round(float(r_hi[0]), 2)],
        ours_mean=round(float(of.mean()), 2), ref_mean=round(float(rf.mean()), 2),
        diff=round(float(diff), 2), diff_ci=[round(float(d_lo), 2), round(float(d_hi), 2)],
        tol=round(float(TOL * r_iqm), 2), rel_gap_pct=round(float(100 * abs(diff) / r_iqm), 1),
        crit_A_ci_contains_0=crit_a, crit_B_within_10pct=crit_b, passed=crit_a and crit_b,
        ours_run_s=o_run, ours_compile_s=o_comp, ours_sps=o_sps,
        ref_run_s=r_run, ref_compile_s=r_comp, ref_sps=r_sps,
    )
    json.dump(out, open(a.json, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
