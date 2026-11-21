"""Counted-cell analysis v2 (tars-test-rules §4, §6, §7, §9). Reuses the loader of h1prime.py (e63d7b30,
passed review with one fix 14 Nov) unchanged by import; h1prime.py stays as the passed record.
Fix (review, 14 Nov): §4 primary for per-task metrics = IQM across seeds x tasks with the rliable stratified
bootstrap (for each task column, seeds resampled with replacement; paired comparisons use the same seed draw
for every cell in the comparison), 10k resamples, rng seed 0. Late window = new tasks 21-40 by position (20
columns). Forgetting (H3) = analyze()'s per-revisit forgetting, 13 revisit columns per seed; gap 0.05.
AUC_rel gaps 0.093. Sensitivities: per-seed mean then IQM over seeds, paired seed bootstrap (h1prime.py).
H4 needs the size line, which has no counted cells, so it is not computable. Zero-shot section is report-only
from zero_shot_counted_700-724.json (729c4aba...)."""
import io, contextlib, json, hashlib, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with contextlib.redirect_stdout(io.StringIO()):
    import h1prime as H
sm, D, M, S, NB = H.sm, H.D, H.M, H.S, H.NB
assert not H.problems
# forgetting grid (25 x 13) per cell, from analyze on each chunk
def forget_grid(cell):
    import glob
    d = os.path.join(H.X, H.CELLS[cell]); man = json.load(open(f"{d}/manifest.json")); rows = []
    for c in man["chunks"]:
        z = dict(np.load(os.path.join(d, os.path.basename(c.get("path", c.get("file")))), allow_pickle=False))
        r = sm.analyze(z, require_device_info=True)
        rows += [[x[2] for x in f] for f in r["forgetting"]]
    g = np.array(rows, float); assert g.shape == (25, 13), g.shape; return g
for c in H.CELLS: M[c]["forget"] = forget_grid(c)
rng = np.random.default_rng(0)
def strat_idx(T): return rng.integers(0, S, size=(NB, S, T))   # per resample, per task column: seed draw
def iqm_grid_boot(x, idx):  # x (S,T), idx (NB,S,T) -> (NB,)
    g = np.take_along_axis(np.broadcast_to(x, (NB,) + x.shape), idx, axis=1)
    return sm.iqm(g.reshape(NB, -1), axis=1)
def ci(b): return [float(np.quantile(b, .025)), float(np.quantile(b, .975))]
def verdict(est, lo, hi, gap, sign=+1):
    exc = bool(lo > 1e-12 or hi < -1e-12)
    return dict(D=float(est), ci95=[lo, hi], ci_excludes_zero=exc, meets_gap=bool(sign * est >= gap),
                supported=bool(exc and sign * est > 0 and sign * est >= gap))
def contrast(terms, key, gap, sign=+1):
    """terms: list of (coef, cell). Estimate = sum coef*IQM_seedsxtasks(cell); stratified paired bootstrap."""
    T = M[terms[0][1]][key].shape[1]; idx = strat_idx(T)
    est = sum(c * sm.iqm(M[cell][key].ravel()) for c, cell in terms)
    boots = sum(c * iqm_grid_boot(M[cell][key], idx) for c, cell in terms)
    lo, hi = ci(boots); out = verdict(est, lo, hi, gap, sign)
    # sensitivity: per-seed mean, IQM over seeds, paired seed bootstrap
    pidx = np.random.default_rng(0).integers(0, S, size=(NB, S))
    ps = {cell: M[cell][key].mean(1) for _, cell in terms}
    e2 = sum(c * sm.iqm(ps[cell]) for c, cell in terms); b2 = sum(c * sm.iqm(ps[cell][pidx], axis=1) for c, cell in terms)
    out["sens_per_seed_mean"] = dict(D=float(e2), ci95=ci(b2))
    return out
L = "late_tasks"
res = dict(method=__doc__, inputs=dict(h1prime_py="e63d7b30", zero_shot_json=hashlib.sha256(open(
    "zero_shot_counted_700-724.json", "rb").read()).hexdigest()))
res["H1prime"] = contrast([(1, "pretrained_1M"), (-1, "scratch_1M")], L, H.GAP)
res["H3"] = {"lora_minus_fullFT": contrast([(1, "lora_1M"), (-1, "pretrained_1M")], "forget", 0.05, sign=-1),
             "frozen_pretrained_minus_fullFT": contrast([(1, "frozen_pretrained_1M"), (-1, "pretrained_1M")], "forget", 0.05, sign=-1),
             "note": "forgetting (stream_metrics.forgetting) = eval at end of the task's previous block minus eval just before the revisit; higher = more forgetting; H3 supported if the difference is <= -0.05 with CI excluding 0"}
res["H4"] = "not computable: the size line (0.1M/0.3M/3-10M) has no counted cells"
res["H5"] = contrast([(1, "ln_pretrained_1M"), (-1, "ln_scratch_1M"), (-1, "pqn_pretrained_1M"), (1, "pqn_scratch_1M")], L, H.GAP)
res["H5"]["D_ppo_ln"] = contrast([(1, "ln_pretrained_1M"), (-1, "ln_scratch_1M")], L, H.GAP)
res["H5"]["D_pqn"] = contrast([(1, "pqn_pretrained_1M"), (-1, "pqn_scratch_1M")], L, H.GAP)
res["H6"] = contrast([(1, "frozen_pretrained_1M"), (-1, "frozen_random_1M")], L, H.GAP)
res["forgetting_per_cell"] = {c: float(sm.iqm(M[c]["forget"].ravel())) for c in H.CELLS}
zs = json.load(open("zero_shot_counted_700-724.json"))
keys = ["zero_shot_new_late", "end_of_block_new_late", "bwt_revisit_late", "zero_shot_new", "end_of_block_new", "bwt_final", "zero_shot_revisit"]
res["zero_shot"] = {e["cell"]: {k: e[k]["iqm"] for k in keys} for e in zs}
lo = next(e for e in zs if e["cell"] == "lora_1M")
res["lora_per_seed"] = {k: lo[k]["per_seed"] for k in ("end_of_block_new_late", "bwt_revisit_late", "bwt_final")}
res["lora_per_seed"]["late_auc_rel"] = M["lora_1M"]["late"].tolist()
json.dump(res, open("counted_analysis_results.json", "w"), indent=1, default=float)
# LoRA paired figure, per seed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, k, t in ((ax[0], "bwt_revisit_late", "Backward transfer on revisits (tasks 21-40)"), (ax[1], "bwt_final", "Backward transfer at end of stream")):
    x = np.array(lo["end_of_block_new_late"]["per_seed"], float); y = np.array(lo[k]["per_seed"], float)
    a.scatter(x, y, s=25); a.axhline(0, c="grey", lw=.5); a.set_xlabel("End-of-block score, new tasks 21-40 (per seed)")
    a.set_ylabel("Backward transfer (per seed)"); a.set_title(t, fontsize=10)
fig.suptitle("LoRA ~1M, 25 seeds (700-724): late-task learning vs backward transfer"); fig.tight_layout()
fig.savefig("lora_learning_vs_bwt_per_seed.png", dpi=130)
f = lambda r: f"{r['D']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] sup={r['supported']} | per-seed {r['sens_per_seed_mean']['D']:+.3f} [{r['sens_per_seed_mean']['ci95'][0]:+.3f}, {r['sens_per_seed_mean']['ci95'][1]:+.3f}]"
print("H1'", f(res["H1prime"])); print("H3 lora", f(res["H3"]["lora_minus_fullFT"])); print("H3 frozen", f(res["H3"]["frozen_pretrained_minus_fullFT"]))
print("H5", f(res["H5"]), "\n  D_ln", f(res["H5"]["D_ppo_ln"]), "\n  D_pqn", f(res["H5"]["D_pqn"])); print("H6", f(res["H6"]))
print("forget", {k: round(v, 3) for k, v in res["forgetting_per_cell"].items()})
