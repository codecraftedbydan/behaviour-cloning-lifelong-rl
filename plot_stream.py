"""Plots + summary for a lifelong stream run (lifelong.py output).

    python plot_stream.py results/stream_dev/size_k60.npz --out-dir results/stream_dev/size_k60
Tasks are numbered from 1 (rules numbering). Primary metric: AUC_rel slope over new
tasks excluding task 1. Writes: adaptation.png (AUC stream vs fresh, AUC_rel, steps), curves.png
(normalised training curves per new task), health.png (probe metrics per block),
forgetting.png, and summary.json.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import stream_metrics as SM  # noqa: E402


def band(ax, x, y, label, color=None):
    """IQM line + 95% bootstrap band over seeds (y: (S, N))."""
    mid = np.array([SM.iqm(c[np.isfinite(c)]) if np.isfinite(c).sum() else np.nan for c in y.T])
    ci = np.array([SM.bootstrap_ci(c[np.isfinite(c)], n_boot=2000) if np.isfinite(c).sum() > 1
                   else (np.nan, np.nan) for c in y.T])
    ax.plot(x, mid, label=label, color=color)
    ax.fill_between(x, ci[:, 0], ci[:, 1], alpha=0.2, color=color)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thr", type=float, default=0.8)
    p.add_argument("--require-device-info", action="store_true",
                   help="refuse files without stream/fresh platform + matmul precision (pilot)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    d = dict(np.load(a.npz))
    r = SM.analyze(d, thr=a.thr, require_device_info=a.require_device_info)
    S, B, K = r["S"], r["B"], r["K"]
    is_new = d["is_new"].astype(bool)

    # Adaptation per new task (by new-task index; all seeds share the same block layout).
    nb = np.flatnonzero(is_new[0])
    x = np.arange(1, len(nb) + 1)  # task number, rules numbering (task 1 = first task)
    sched0 = d["task_of_block"] - 1 if "task_of_block" in d else d["schedule"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    band(ax[0], x, r["auc_stream"][:, nb], "stream agent")
    band(ax[0], x, np.take_along_axis(r["auc_fresh"], sched0[:, nb], 1), "fresh agent")
    ax[0].set(xlabel="task number (new tasks)", ylabel=f"AUC (mean norm. return, first {r['M']} upd.)",
              title="Adaptation speed"); ax[0].legend()
    band(ax[1], x, r["auc_rel"][:, nb], "stream - fresh", "C3")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].axvline(1.5, color="grey", ls=":", lw=1)
    s = r["slope_auc_rel_new_excl_task1_summary"]
    ax[1].set(xlabel="task number (new tasks)", ylabel="AUC_rel",
              title=f"AUC_rel; primary slope (excl. task 1) IQM {s['iqm']:.4f}\n"
                    f"95% CI [{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")
    band(ax[2], x, r["steps_rel"][:, nb], "stream - fresh", "C4")
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set(xlabel="task number (new tasks)", ylabel=f"steps-to-{a.thr} diff (env steps)",
              title="Relative steps-to-threshold (capped at K)")
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/adaptation.png", dpi=120); plt.close(fig)

    # Training curves for a few new tasks (IQM over seeds) + fresh mean.
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.get_cmap("viridis")
    for i, b in enumerate(nb):
        mid = SM.iqm(r["curves_stream"][:, b], axis=0)
        ax.plot(np.arange(1, K + 1), mid, color=cmap(i / max(1, len(nb) - 1)), lw=1,
                label="task 1" if i == 0 else (f"task {len(nb)}" if i == len(nb) - 1 else None))
    ax.plot(np.arange(1, K + 1), np.nanmean(r["curves_fresh"], (0, 1)), "k--", lw=2, label="fresh (mean)")
    ax.set(xlabel="update within block", ylabel="normalised return",
           title="Per-task adaptation curves (colour = task number)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/curves.png", dpi=120); plt.close(fig)

    # Health per block (value at the end of each block; block -1 = init).
    keys = [k for k in ["dormant_frac", "srank.trunk_0", "srank.trunk_1", "erank.trunk_1",
                        "weight_norm", "random_target_mse", "probe_entropy"] if f"stream/probes/{k}" in d]
    fig, ax = plt.subplots(2, 4, figsize=(18, 7))
    health = {}
    for i, k in enumerate(keys + ["switch_entropy"]):
        if k == "switch_entropy":
            y = d["stream/switch_entropy"]; xs = np.arange(1, B + 1)  # at the start of block b
        else:
            y = np.concatenate([d[f"stream/probe0/{k}"][:, None], d[f"stream/probes/{k}"]], 1).astype(float)
            xs = np.arange(0, B + 1)  # 0 = init, b = end of block b
        band(ax.flat[i], xs, y, k)
        ax.flat[i].set(title=k, xlabel="block number (0 = init)")
        sl = SM.ols_slope(y[:, 1:] if k != "switch_entropy" else y[:, 1:])
        health[k] = dict(first=float(np.mean(y[:, 1])), last=float(np.mean(y[:, -1])), slope=SM.summarize(sl))
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/health.png", dpi=120); plt.close(fig)

    # Forgetting
    fig, ax = plt.subplots(figsize=(6, 4))
    for s_rows in r["forgetting"]:
        if s_rows:
            ax.scatter([q[0] for q in s_rows], [q[2] for q in s_rows], s=10, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="block number of the revisit", ylabel="forgetting (norm. eval)", title="Forgetting on revisits")
    fig.tight_layout(); fig.savefig(f"{a.out_dir}/forgetting.png", dpi=120); plt.close(fig)

    summ = {k: v for k, v in r.items() if k.endswith("_summary")}
    summ.update(primary=r["primary"], by_shift=r["by_shift"], health=health, thr=a.thr, M=r["M"],
                K_updates=K, K_env_steps=K * r["batch_size"], S=S, B=B,
                steps_capped_frac_stream=r["steps_capped_frac_stream"],
                steps_capped_frac_fresh=r["steps_capped_frac_fresh"],
                forgetting=[[list(q) for q in rows] for rows in r["forgetting"]],
                seeds=d["seeds"].tolist(),
                fresh_seeds=d["fresh_seeds"].tolist() if "fresh_seeds" in d else None,
                git_commit=str(d.get("git_commit", "unknown")),
                param_count=int(d["param_count"]) if "param_count" in d else None,
                wall_seconds=float(d.get("wall_seconds", np.nan)),
                stream_seconds=float(d["stream_seconds"]), fresh_seconds=float(d["fresh_seconds"]),
                auc_source=r["auc_source"], platform=r["platform"],
                switch_entropy_note="mean policy entropy over the fixed 512-state probe batch at each switch")
    if "head_probe" in r:
        hp = r["head_probe"]
        summ["head_probe"] = {k: v for k, v in hp.items() if k.endswith("_summary")}
        summ["head_probe"].update(tasks=hp["tasks"].tolist(), blocks=hp["blocks"].tolist(),
                                  auc_rel_headprobe=hp["auc_rel_headprobe"].tolist(),
                                  auc_rel_stream=hp["auc_rel_stream"].tolist())
    with open(f"{a.out_dir}/summary.json", "w") as f:
        json.dump(summ, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: (v["iqm"], v["ci95"]) for k, v in summ.items() if k.startswith("slope")}, indent=0))


if __name__ == "__main__":
    main()
