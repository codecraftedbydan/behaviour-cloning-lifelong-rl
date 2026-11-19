"""Lifelong-preset sanity runs (seeds 300-304) vs the non-lifelong parent preset
on the same seeds: return, per-layer dormant fraction, weight norm, approx KL."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot import bootstrap_ci, iqm, smooth  # noqa: E402

D = "results/lifelong"
RUNS = [("cartpole", "default", "CartPole-v1, preset default"),
        ("cartpole", "lifelong", "CartPole-v1, preset lifelong"),
        ("breakout", "pjm", "Breakout-MinAtar, purejaxrl_match"),
        ("breakout", "lifelong", "Breakout-MinAtar, purejaxrl_match,lifelong")]
fig, axes = plt.subplots(2, 4, figsize=(19, 7.5))
for col, env in enumerate(["cartpole", "breakout"]):
    ax_r, ax_d, ax_w, ax_k = axes[0, 2 * col], axes[0, 2 * col + 1], axes[1, 2 * col], axes[1, 2 * col + 1]
    for (e, tag, label), c in zip(RUNS, ["tab:blue", "tab:red"] * 2):
        if e != env:
            continue
        d = np.load(f"{D}/{env}_{tag}.npz")
        x = d["env_steps"]
        s = smooth(d["metric_episode_return"], 10)
        lo, hi = bootstrap_ci(s)
        ax_r.plot(x, iqm(s), color=c, lw=2, label=label); ax_r.fill_between(x, lo, hi, color=c, alpha=0.18)
        layers = sorted(k for k in d.files if k.startswith("metric_dormant_frac."))
        for k, ls in zip(layers, ["-", "--", ":", "-."]):
            ax_d.plot(x, d[k].mean(0), color=c, ls=ls, label=f"{tag}: {k.split('.', 1)[1]}")
        ax_w.plot(x, d["metric_weight_norm"].mean(0), color=c, label=tag)
        ax_k.plot(x, smooth(d["metric_approx_kl"], 10).mean(0), color=c, label=tag)
    ax_r.set_title(f"{env}: training return (IQM, 5 seeds)", fontsize=10); ax_r.legend(fontsize=8)
    ax_d.set_title(f"{env}: dormant fraction per layer (seed mean)", fontsize=10); ax_d.legend(fontsize=7)
    ax_w.set_title(f"{env}: global weight norm", fontsize=10); ax_w.legend(fontsize=8)
    ax_k.set_title(f"{env}: approx KL (k3)", fontsize=10); ax_k.legend(fontsize=8); ax_k.set_yscale("log")
for ax in axes.flat:
    ax.grid(alpha=0.3); ax.set_xlabel("env steps")
fig.suptitle("Lifelong preset sanity runs, seeds 300-304 (dormant tau: 0.1 non-lifelong / 0.025 lifelong)")
fig.tight_layout(); fig.savefig(f"{D}/sanity.png", dpi=120)
print("saved", f"{D}/sanity.png")
