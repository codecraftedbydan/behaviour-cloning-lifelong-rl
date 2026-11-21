"""Pick the kickstart KL coefficient for kickstart_1M (stream LR fixed at 2.5e-4, scratch_1M's pick).
Rule as for LR (rank_lr_raw.py bf77d747): argmax over the coef grid of IQM (seeds 750-759) of the stream
agent's RAW late AUC (new tasks 21-40). Edge flagged; tie -> nearest log-centre, then lower value.
Fresh agents are not used for the choice. Same per-trial integrity checks, plus kickstart_coef in
config, fresh kickstart_coef 0, and kickstart effective updates > 0 for every seed. Report only:
scratch_1M / pretrained_1M at 2.5e-4 from the LR sweep for context."""
import hashlib, json, math, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("TARS_PPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stream_metrics as sm

ROOT = os.path.join(os.environ.get("TARS_SWEEPS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lr_sweeps")), "kickstart_20261112")
SWEEP = os.path.join(os.environ.get("TARS_SWEEPS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lr_sweeps")), "sweeps_20261110")
GRID = [0.01, 0.03, 0.1, 0.3]
LR = 2.5e-4

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load(d, lr, coef, problems):
    man = json.load(open(os.path.join(d, "manifest.json")))
    rid = json.load(open(os.path.join(d, "runner_identity.json")))
    cfg, fcfg = json.loads(man["config"]), json.loads(man["fresh_config"])
    chk = dict(status=man["status"] == "complete", platform=man["device_platform"] == "gpu",
               prec=man["matmul_precision"] == "highest", chunk=man["chunk_size"] == 5,
               tree=man["source_tree_hash"].startswith("6a52b120"), clean=not man["git_dirty"],
               lr=math.isclose(cfg["lr"], lr) and math.isclose(fcfg["lr"], lr),
               seeds=man["seeds"] == list(range(750, 760)) and man["fresh_seeds"] == list(range(850, 860)),
               binding=man["ranking_binding"] is True,
               runner=str(rid.get("runner_sha256", "")).startswith("ec4061bc"))
    if coef is not None:
        chk.update(coef=math.isclose(cfg.get("kickstart_coef", -1), coef),
                   fresh_coef=fcfg.get("kickstart_coef", 0.0) == 0.0)
    bad = [k for k, v in chk.items() if not v]
    if bad: problems.append(f"{d}: failed checks {bad}")
    rs, seeds, keff = [], [], []
    for c in man["chunks"]:
        p = os.path.join(d, os.path.basename(c["path"]) if "path" in c else c["file"])
        if sha(p) != c["sha256"]: problems.append(f"{d}: chunk sha mismatch {p}")
        z = dict(np.load(p, allow_pickle=False))
        r = sm.analyze(z, require_device_info=True)
        is_new = np.asarray(z["is_new"]).astype(bool); tn = np.cumsum(is_new, 1)
        lm = is_new & (tn > int(is_new[0].sum()) // 2)
        seeds += list(map(int, z["seeds"]))
        rs += [float(r["auc_stream"][s][lm[s]].mean()) for s in range(len(lm))]
        if not np.allclose(z["stream/metrics/lr"], lr) or not np.allclose(z["fresh/metrics/lr"], lr):
            problems.append(f"{d}: per-update lr metric != {lr}")
        if coef is not None:
            k = np.asarray(z["kickstart_effective_updates"]); keff += k.tolist()
            if (k <= 0).any(): problems.append(f"{d}: zero kickstart effective updates")
    if seeds != list(range(750, 760)): problems.append(f"{d}: seed order {seeds}")
    return rs, keff

def summ(x):
    s = sm.summarize(x); return dict(iqm=float(s["iqm"]), ci95=[float(v) for v in s["ci95"]], mean=float(s["mean"]), n=int(s["n"]))

problems, T, K = [], {}, {}
for c in GRID:
    T[c], K[c] = load(os.path.join(ROOT, f"kickstart_1M_lr2.5e-4_kl{c}"), LR, c, problems)
sc = {c: float(sm.iqm(T[c])) for c in GRID}
centre = math.sqrt(GRID[0] * GRID[-1])
ranked = sorted(GRID, key=lambda c: (-sc[c], abs(math.log(c / centre)), c))
best = ranked[0]
ctx = {cell: summ(load(os.path.join(SWEEP, f"{cell}_lr2.5e-4"), LR, None, problems)[0])
       for cell in ("scratch_1M", "pretrained_1M")}
out = dict(method=__doc__, lr=LR, grid=GRID, chosen_coef=best, ranked_coef=ranked,
           edge=best in (GRID[0], GRID[-1]), tie_broken=sum(sc[c] == sc[best] for c in GRID) > 1,
           trials={str(c): dict(raw_late_auc_stream=summ(T[c]), kickstart_effective_updates=K[c]) for c in GRID},
           report_only_context_lr2_5e_4=ctx, problems=problems)
json.dump(out, open("kickstart_ranking_raw.json", "w"), indent=1)
for c in GRID:
    s = out["trials"][str(c)]["raw_late_auc_stream"]
    print(f"kl {c:<5} IQM {s['iqm']:.4f} CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}] keff {min(K[c])}-{max(K[c])}")
print("chosen", best, "edge", out["edge"], "tie", out["tie_broken"], "ranked", ranked)
for k, v in ctx.items(): print(f"context {k} @2.5e-4 IQM {v['iqm']:.4f} CI [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
print("PROBLEMS", problems)
