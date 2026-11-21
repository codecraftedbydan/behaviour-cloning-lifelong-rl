"""H1' analysis on the 10 counted cells (tars-test-rules §4, §7, §9; amended 10 Nov rule).
AUC_rel per task = stream AUC - fresh AUC from the family's scratch-cell fresh arrays (decision 10 Nov 22:05,
12 Nov 08:10): PPO <- scratch_1M_r2, PPO-LN <- ln_scratch_1M, PQN <- pqn_scratch_1M; seeds 700-724 / 800-824.
Primary (H1', §9): D = IQM_seeds(late AUC_rel, pretrained_1M) - IQM_seeds(late AUC_rel, scratch_1M), late =
new tasks 21-40, per-seed late mean as in stream_metrics.analyze; 95% CI by paired seed bootstrap (both cells
resampled with the same seed indices), 10k resamples, rng seed 0. Pass: CI excludes 0 AND D >= 0.093.
Sensitivity (report only): IQM over seeds x tasks (25 x 20); IQM of paired per-seed differences.
Also: H1'-b (primacy bias D), AUC_rel(task 2) D, probability of improvement, raw stream AUC, per-cell table."""
import glob, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("TARS_PPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); import stream_metrics as sm
X = os.environ.get("TARS_CELLS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "extracted"))
CELLS = {"scratch_1M": "scratch_1M_r2/scratch_1M_r2", "pretrained_1M": "pretrained_1M/pretrained_1M",
         "frozen_pretrained_1M": "frozen_pretrained_1M/frozen_pretrained_1M", "frozen_random_1M": "frozen_random_1M/frozen_random_1M",
         "lora_1M": "lora_1M/lora_1M", "kickstart_1M": "kickstart_1M/kickstart_1M",
         "ln_scratch_1M": "ln_scratch_1M/ln_scratch_1M", "ln_pretrained_1M": "ln_pretrained_1M/ln_pretrained_1M",
         "pqn_scratch_1M": "pqn_scratch_1M/pqn_scratch_1M", "pqn_pretrained_1M": "pqn_pretrained_1M/pqn_pretrained_1M"}
FAM = {"ppo": "scratch_1M", "ppo_ln": "ln_scratch_1M", "pqn": "pqn_scratch_1M"}
CFAM = {c: ("pqn" if c.startswith("pqn") else "ppo_ln" if c.startswith("ln_") else "ppo") for c in CELLS}
SHARED_T2 = {703, 705, 712, 714, 724}
GAP, NB = 0.093, 10_000
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
problems = []
def load(cell):
    d = os.path.join(X, CELLS[cell]); man = json.load(open(f"{d}/manifest.json"))
    if man["status"] != "complete": problems.append(f"{cell}: status {man['status']}")
    if man["seeds"] != list(range(700, 725)): problems.append(f"{cell}: seeds")
    if man["device_platform"] != "gpu" or man["matmul_precision"] != "highest" or man["git_dirty"]: problems.append(f"{cell}: provenance")
    if not man["source_tree_hash"].startswith("6a52b120"): problems.append(f"{cell}: tree")
    acc = {k: [] for k in ("auc_stream", "auc_fresh", "sched", "is_new", "opt", "seeds")}
    for c in man["chunks"]:
        p = os.path.join(d, os.path.basename(c.get("path", c.get("file"))))
        if sha(p) != c["sha256"]: problems.append(f"{cell}: chunk sha {p}")
        z = dict(np.load(p, allow_pickle=False)); r = sm.analyze(z, require_device_info=True)
        acc["auc_stream"].append(r["auc_stream"]); acc["auc_fresh"].append(r["auc_fresh"])
        acc["sched"].append(np.asarray(z["task_of_block"]) - 1); acc["is_new"].append(np.asarray(z["is_new"]).astype(bool))
        acc["opt"].append(np.asarray(z["optimal_return"])); acc["seeds"].append(np.asarray(z["seeds"]))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    if list(out["seeds"]) != list(range(700, 725)): problems.append(f"{cell}: seed order")
    out["fresh_lr"] = json.loads(man["fresh_config"])["lr"]; out["lr"] = json.loads(man["config"])["lr"]
    return out
D = {c: load(c) for c in CELLS}
for c in CELLS:  # paired design: same task chains per seed
    for k in ("sched", "is_new", "opt"):
        if not np.array_equal(D[c][k], D["scratch_1M"][k]): problems.append(f"{c}: {k} differs from scratch_1M")
    if abs(D[c]["fresh_lr"] - 1e-4) > 1e-12: problems.append(f"{c}: fresh lr {D[c]['fresh_lr']}")
S = 25
def metrics(c):
    d, ref = D[c], D[FAM[CFAM[c]]]
    sched, isn = d["sched"], d["is_new"]
    fres = np.take_along_axis(ref["auc_fresh"], sched, 1)
    rel = d["auc_stream"] - fres
    tn = np.cumsum(isn, 1); late = isn & (tn > int(isn[0].sum()) // 2)
    assert (late.sum(1) == 20).all()
    late_tasks = np.stack([rel[s][late[s]] for s in range(S)])          # (25, 20)
    raw_tasks = np.stack([d["auc_stream"][s][late[s]] for s in range(S)])
    b1 = np.argmax(isn & (tn == 1), 1); b2 = np.argmax(isn & (tn == 2), 1)
    t1, t2 = rel[np.arange(S), b1], rel[np.arange(S), b2]
    own = d["auc_stream"] - np.take_along_axis(d["auc_fresh"], sched, 1)
    return dict(late=late_tasks.mean(1), late_tasks=late_tasks, raw=raw_tasks.mean(1), t1=t1, t2=t2, pb=t2 - t1,
                late_own_fresh=np.stack([own[s][late[s]] for s in range(S)]).mean(1))
M = {c: metrics(c) for c in CELLS}
rng = np.random.default_rng(0); IDX = rng.integers(0, S, size=(NB, S))
def ci(b): return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
def diff(a, b, key, mask=None):
    x, y = M[a][key], M[b][key]
    if mask is not None: x, y = x[mask], y[mask]
    n = len(x); idx = IDX if n == S else np.random.default_rng(0).integers(0, n, size=(NB, n))
    if x.ndim == 1:
        est = sm.iqm(x) - sm.iqm(y); boots = sm.iqm(x[idx], axis=1) - sm.iqm(y[idx], axis=1)
    else:
        est = sm.iqm(x.ravel()) - sm.iqm(y.ravel())
        boots = sm.iqm(x[idx].reshape(NB, -1), axis=1) - sm.iqm(y[idx].reshape(NB, -1), axis=1)
    lo, hi = ci(boots)
    return dict(D=float(est), ci95=[lo, hi], ci_excludes_zero=bool(lo > 1e-12 or hi < -1e-12), meets_gap=bool(est >= GAP), n=int(n))
def poi(a, b, key="late"):
    x, y = M[a][key], M[b][key]
    f = lambda xx, yy: float(((xx[:, None] > yy[None, :]) + 0.5 * (xx[:, None] == yy[None, :])).mean())
    boots = [f(x[i], y[i]) for i in IDX[:2000]]
    return dict(p=f(x, y), ci95=ci(np.array(boots)), boot_n=2000)
def cell_row(c):
    m = M[c]; s = lambda v: dict(iqm=float(sm.iqm(v)), ci95=list(sm.bootstrap_ci(v)))
    return dict(lr=D[c]["lr"], fresh_ref=FAM[CFAM[c]], late_auc_rel=s(m["late"]), raw_late_auc_stream=s(m["raw"]),
                late_auc_rel_own_fresh=s(m["late_own_fresh"]), auc_rel_task1=s(m["t1"]), auc_rel_task2=s(m["t2"]), primacy_bias=s(m["pb"]))
keep = np.array([s not in SHARED_T2 for s in range(700, 725)])
h1 = diff("pretrained_1M", "scratch_1M", "late")
out = dict(method=__doc__, problems=problems,
  H1prime=dict(primary=h1, passes=bool(h1["ci_excludes_zero"] and h1["D"] > 0 and h1["meets_gap"]),
               sens_seeds_x_tasks=diff("pretrained_1M", "scratch_1M", "late_tasks"),
               sens_paired_diff_iqm=dict(iqm=float(sm.iqm(M["pretrained_1M"]["late"] - M["scratch_1M"]["late"])),
                                         ci95=list(sm.bootstrap_ci(M["pretrained_1M"]["late"] - M["scratch_1M"]["late"]))),
               raw_late_auc_D=diff("pretrained_1M", "scratch_1M", "raw"),
               poi_pretrained_gt_scratch=poi("pretrained_1M", "scratch_1M")),
  H1prime_b=dict(primacy_bias_D=diff("pretrained_1M", "scratch_1M", "pb"),
                 sens_excl_shared_task2=diff("pretrained_1M", "scratch_1M", "pb", keep)),
  auc_rel_task2_D=dict(all=diff("pretrained_1M", "scratch_1M", "t2"), excl_shared_task2=diff("pretrained_1M", "scratch_1M", "t2", keep)),
  cells={c: cell_row(c) for c in CELLS})
if __name__ == "__main__":  # importing this module (counted_analysis.py does) must not rewrite the result file
    json.dump(out, open("h1prime_results.json", "w"), indent=1)
print("PROBLEMS", problems)
p = out["H1prime"]; print("H1' D", p["primary"], "PASS" if p["passes"] else "not supported")
for k in ("sens_seeds_x_tasks", "sens_paired_diff_iqm", "raw_late_auc_D", "poi_pretrained_gt_scratch"): print(" ", k, p[k])
print("H1'-b", out["H1prime_b"]); print("task2", out["auc_rel_task2_D"])
for c, r in out["cells"].items():
    print(f"{c:22s} lr {r['lr']:.1e} lateRel {r['late_auc_rel']['iqm']:+.3f} [{r['late_auc_rel']['ci95'][0]:+.3f},{r['late_auc_rel']['ci95'][1]:+.3f}] raw {r['raw_late_auc_stream']['iqm']:.3f} ownFresh {r['late_auc_rel_own_fresh']['iqm']:+.3f} t1 {r['auc_rel_task1']['iqm']:+.3f} t2 {r['auc_rel_task2']['iqm']:+.3f} pb {r['primacy_bias']['iqm']:+.3f}")
