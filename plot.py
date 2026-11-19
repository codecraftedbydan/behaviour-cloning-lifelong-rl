"""Plot a ppo.py results file: per-seed return curves + IQM with a bootstrap
95% CI, and a diagnostics grid (KL, clip fraction, entropy, explained
variance, network health).

    python plot.py results/cartpole.npz --out results/cartpole.png
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def iqm(x, axis=0):
    """Interquartile mean: drop the bottom and top 25% of runs, then average
    the rest (Agarwal et al., 2021). x: (num_seeds, ...)."""
    x = np.sort(x, axis=axis)
    n = x.shape[axis]
    k = int(np.floor(0.25 * n))
    return np.take(x, np.arange(k, n - k), axis=axis).mean(axis=axis)


def bootstrap_ci(x, stat=iqm, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap over seeds. x: (num_seeds, T) -> (lo, hi), each (T,).
    With only 5 seeds the CI is coarse, so read it as indicative."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.stack([stat(x[i], axis=0) for i in idx])  # (n_boot, T)
    return np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)


def smooth(y, w):
    if w <= 1:
        return y
    k = np.ones(w) / w
    pad = np.concatenate([np.repeat(y[..., :1], w - 1, axis=-1), y], axis=-1)
    return np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), -1, pad)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", default="results/cartpole.png")
    p.add_argument("--smooth", type=int, default=10, help="moving-average window (updates)")
    args = p.parse_args()

    d = np.load(args.npz)
    cfg = json.loads(str(d["config"]))
    steps = d["env_steps"]
    ret = d["metric_episode_return"]  # (seeds, updates)
    n_seeds = ret.shape[0]
    ret_s = smooth(ret, args.smooth)
    mid = iqm(ret_s)
    lo, hi = bootstrap_ci(ret_s)

    # ---- main figure: returns -------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in range(n_seeds):
        ax.plot(steps, ret_s[s], lw=0.8, alpha=0.45, label="per seed" if s == 0 else None)
    ax.plot(steps, mid, color="k", lw=2, label=f"IQM over {n_seeds} seeds")
    ax.fill_between(steps, lo, hi, color="k", alpha=0.15, label="95% bootstrap CI (IQM)")
    ev_steps, ev = d["eval_env_steps"], d["eval_returns"][:, :, 0]
    ax.plot(ev_steps, iqm(ev), "o", ms=3, color="tab:red", alpha=0.7, label="eval-hook IQM")
    ax.set_xlabel("environment steps (per seed)")
    ax.set_ylabel("episode return")
    ax.set_title(f"PPO (JAX, from scratch) on {cfg['env_name']}: "
                 f"{d['steps_per_sec']:,.0f} env-steps/s on {d['platform']}")
    ax.set_ylim(0, 520)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")

    # ---- diagnostics figure ---------------------------------------------------
    keys = [("approx_kl", "approx KL"), ("clip_frac", "clip fraction"),
            ("entropy", "policy entropy"), ("explained_variance", "explained variance"),
            ("dormant_frac", f"tau-dormant fraction (tau={cfg['dormant_tau']})"),
            ("weight_norm", "global weight norm")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (k, title) in zip(axes.flat, keys):
        y = smooth(d[f"metric_{k}"], args.smooth)
        for s in range(n_seeds):
            ax.plot(steps, y[s], lw=0.8, alpha=0.5)
        ax.plot(steps, iqm(y), color="k", lw=1.5)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_xlabel("env steps")
    fig.tight_layout()
    diag = args.out.replace(".png", "_diagnostics.png")
    fig.savefig(diag, dpi=130)
    print(f"saved {diag}")

    # ---- summary numbers ------------------------------------------------------
    final = ret[:, -10:].mean(axis=1)
    print("final return per seed (mean of last 10 updates):", np.round(final, 1))
    print(f"final IQM: {iqm(final):.1f}")
    for thr in (475, 500):
        first = [steps[np.argmax(r >= thr)] if (r >= thr).any() else None for r in ret_s]
        print(f"first step where smoothed return >= {thr} per seed:", first)
    print("final eval return per seed:", np.round(ev[:, -1], 1))
    for k in ("dormant_frac", "saturated_frac", "weight_norm", "explained_variance",
              "approx_kl", "clip_frac", "entropy", "approx_kl_first_mb"):
        v = d[f"metric_{k}"]
        print(f"{k:22s} first={np.nanmean(v[:, 0]):.4g}  last10={np.nanmean(v[:, -10:]):.4g}  "
              f"max={np.nanmax(np.abs(v)) if 'first_mb' in k else np.nanmax(v):.4g}")


if __name__ == "__main__":
    main()
