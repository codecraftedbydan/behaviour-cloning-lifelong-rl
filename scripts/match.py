"""Match check against PureJaxRL (pre-set rule).

Rule: same env + step budget, 5 seeds; PASS iff our IQM final return lies inside
PureJaxRL's 95% bootstrap CI (of the IQM, resampling seeds).
Final return = mean over the last 10 updates of the per-update episode return, where
the per-update value is the most recently finished episode's return at the end of the
rollout, averaged over envs (identical definition for both implementations).

    python scripts/match.py --name cartpole --ours results/cartpole.npz \
        --ref ../purejaxrl-ref/results/ref_cartpole.npz --out results/match_cartpole.png
"""
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot import bootstrap_ci, iqm, smooth  # noqa: E402

LAST = 10


def summarise(curves):
    final = curves[:, -LAST:].mean(axis=1)
    lo, hi = bootstrap_ci(final[:, None], n_boot=10000)
    return final, iqm(final), float(lo[0]), float(hi[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--ours", required=True, nargs="+", help="one or more npz; seeds are pooled")
    p.add_argument("--ref", required=True, nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--smooth", type=int, default=10)
    p.add_argument("--json", help="append summary as JSON here")
    a = p.parse_args()

    os_, rs_ = [np.load(f) for f in a.ours], [np.load(f) for f in a.ref]
    o, r = os_[0], rs_[0]  # timing / step axis from the first file
    o_ret = np.concatenate([x["metric_episode_return"] for x in os_])
    r_ret = np.concatenate([x["ret_end_of_rollout"] for x in rs_])
    r_all = np.concatenate([x["ret_all_steps"] for x in rs_])
    o_steps, r_steps = o["env_steps"], r["env_steps"]
    assert o_steps[-1] == r_steps[-1], f"budget mismatch {o_steps[-1]} vs {r_steps[-1]}"

    of, o_iqm, o_lo, o_hi = summarise(o_ret)
    rf, r_iqm, r_lo, r_hi = summarise(r_ret)
    passed = r_lo <= o_iqm <= r_hi

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for steps, ret, label, c in ((r_steps, r_ret, "PureJaxRL", "tab:blue"),
                                  (o_steps, o_ret, "ours (tars-ppo)", "tab:red")):
        s = smooth(ret, a.smooth)
        lo, hi = bootstrap_ci(s)
        ax.plot(steps, iqm(s), color=c, lw=2, label=f"{label}: IQM over {ret.shape[0]} seeds")
        ax.fill_between(steps, lo, hi, color=c, alpha=0.18)
    ax.set_xlabel("environment steps (per seed)")
    ax.set_ylabel("episode return")
    ax.set_title(f"{a.name}, {o_ret.shape[0]} vs {r_ret.shape[0]} seeds: ours {o_iqm:.1f} vs PureJaxRL {r_iqm:.1f} "
                 f"[{r_lo:.1f}, {r_hi:.1f}] → {'PASS' if passed else 'FAIL'}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)

    summary = dict(
        env=a.name, budget=int(o_steps[-1]),
        ours_final=np.round(of, 2).tolist(), ours_iqm=round(float(o_iqm), 2),
        ours_ci=[round(o_lo, 2), round(o_hi, 2)],
        ref_final=np.round(rf, 2).tolist(), ref_iqm=round(float(r_iqm), 2),
        ref_ci=[round(r_lo, 2), round(r_hi, 2)],
        ref_iqm_all_steps_def=round(float(iqm(r_all[:, -LAST:].mean(1))), 2),
        passed=bool(passed),
        ours_run_s=round(float(o["run_seconds"]), 1), ours_compile_s=round(float(o["compile_seconds"]), 1),
        ours_sps=round(float(o["steps_per_sec"])),
        ref_run_s=round(float(r["run_seconds"]), 1), ref_compile_s=round(float(r["compile_seconds"]), 1),
        ref_sps=round(float(r["steps_per_sec"])),
    )
    print(json.dumps(summary, indent=1))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
