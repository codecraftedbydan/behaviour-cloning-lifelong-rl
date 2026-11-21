"""Re-rank LR sweep under the amended rule (Dan approved 10 Nov 2026 22:02 GMT; research-log 22:03).
Stream LR per cell = argmax over its grid of IQM (seeds 750-759) of the stream agent's RAW late AUC
(new tasks 21-40). Fresh LR per fresh config = argmax of IQM of the fresh agent's raw late AUC over the
same grid. Edge winner flagged; tie -> nearest log-centre of grid, then lower value. 10k percentile
bootstrap seed 0 (stream_metrics.summarize). Same per-trial integrity checks as rank_lr.py (ce2bb4d4).
Report only: fresh IQM from every cell sharing a fresh config; AUC_rel of each stream pick against
the tuned fresh baseline (per task, seed-paired 850-859)."""
import hashlib, json, math, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("TARS_PPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stream_metrics as sm

ROOT = os.path.join(os.environ.get("TARS_SWEEPS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lr_sweeps")), "sweeps_20261110")
DEFAULT = [1e-4, 2.5e-4, 5e-4, 1e-3]
GRIDS = {"lora_1M": [2.5e-4, 5e-4, 1e-3, 2e-3]}
CELLS = ["pretrained_1M", "scratch_1M", "frozen_pretrained_1M", "frozen_random_1M", "lora_1M",
         "ln_pretrained_1M", "ln_scratch_1M", "pqn_pretrained_1M", "pqn_scratch_1M"]
FRESH = {"ppo": (["pretrained_1M", "scratch_1M", "frozen_pretrained_1M", "frozen_random_1M", "lora_1M"], "scratch_1M"),
         "ppo_ln": (["ln_pretrained_1M", "ln_scratch_1M"], "ln_scratch_1M"),
         "pqn": (["pqn_pretrained_1M", "pqn_scratch_1M"], "pqn_scratch_1M")}
FAM = {c: f for f, (cs, _) in FRESH.items() for c in cs}
NAMES = {1e-4: "1e-4", 2.5e-4: "2.5e-4", 5e-4: "5e-4", 1e-3: "1e-3", 2e-3: "2e-3"}
fmt = NAMES.__getitem__

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

problems, T = [], {}
for cell in CELLS:
    for lr in GRIDS.get(cell, DEFAULT):
        d = os.path.join(ROOT, f"{cell}_lr{fmt(lr)}")
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
        bad = [k for k, v in chk.items() if not v]
        if bad: problems.append(f"{d}: failed checks {bad}")
        rs, rf, rel, seeds, st_task, fr_task = [], [], [], [], [], []
        for c in man["chunks"]:
            p = os.path.join(d, os.path.basename(c["path"]) if "path" in c else c["file"])
            if sha(p) != c["sha256"]: problems.append(f"{d}: chunk sha mismatch {p}")
            z = dict(np.load(p, allow_pickle=False))
            r = sm.analyze(z, require_device_info=True)
            is_new = np.asarray(z["is_new"]).astype(bool); tn = np.cumsum(is_new, 1)
            lm = is_new & (tn > int(is_new[0].sum()) // 2)
            sched = np.asarray(z["task_of_block"]) - 1
            fres = np.take_along_axis(r["auc_fresh"], sched, 1)
            seeds += list(map(int, z["seeds"])); rel += [float(v) for v in r["late_auc_rel"]]
            for s in range(len(lm)):
                rs.append(float(r["auc_stream"][s][lm[s]].mean())); rf.append(float(fres[s][lm[s]].mean()))
                st_task.append((r["auc_stream"][s][lm[s]], sched[s][lm[s]])); fr_task.append(np.asarray(r["auc_fresh"][s]))
            if not np.allclose(z["stream/metrics/lr"], lr) or not np.allclose(z["fresh/metrics/lr"], lr):
                problems.append(f"{d}: per-update lr metric != {lr}")
        if seeds != list(range(750, 760)): problems.append(f"{d}: seed order {seeds}")
        T[(cell, lr)] = dict(rs=rs, rf=rf, rel=rel, st_task=st_task, fr_task=fr_task, fcfg_lr=fcfg["lr"])

def pick(grid, scores):
    centre = math.sqrt(grid[0] * grid[-1])
    ranked = sorted(grid, key=lambda lr: (-scores[lr], abs(math.log(lr / centre)), lr))
    best = ranked[0]
    return dict(chosen_lr=best, ranked_lr=ranked, edge=best in (grid[0], grid[-1]),
                tie_broken=sum(scores[l] == scores[best] for l in grid) > 1)

def summ(x):
    s = sm.summarize(x); return dict(iqm=float(s["iqm"]), ci95=[float(v) for v in s["ci95"]], mean=float(s["mean"]), n=int(s["n"]))

out = {"method": __doc__, "stream": {}, "fresh": {}}
for fam, (cells, src) in FRESH.items():
    grid = GRIDS.get(src, DEFAULT)
    sc = {lr: float(sm.iqm(T[(src, lr)]["rf"])) for lr in grid}
    res = pick(grid, sc)
    res.update(source_cell=src, grid=grid, trials={fmt(lr): summ(T[(src, lr)]["rf"]) for lr in grid},
               by_cell={c: {fmt(lr): float(sm.iqm(T[(c, lr)]["rf"])) for lr in GRIDS.get(c, DEFAULT)} for c in cells},
               by_cell_pick={c: fmt(pick(GRIDS.get(c, DEFAULT), {lr: float(sm.iqm(T[(c, lr)]["rf"])) for lr in GRIDS.get(c, DEFAULT)})["chosen_lr"]) for c in cells})
    out["fresh"][fam] = res
for cell in CELLS:
    grid = GRIDS.get(cell, DEFAULT)
    sc = {lr: float(sm.iqm(T[(cell, lr)]["rs"])) for lr in grid}
    res = pick(grid, sc)
    fam = FAM[cell]; flr = out["fresh"][fam]["chosen_lr"]; fsrc = out["fresh"][fam]["source_cell"]
    st = T[(cell, res["chosen_lr"])]["st_task"]; fr = T[(fsrc, flr)]["fr_task"]
    rel_tuned = [float((a - fr[s][t]).mean()) for s, (a, t) in enumerate(st)]
    res.update(grid=grid, fresh_family=fam,
               trials={fmt(lr): dict(raw_late_auc_stream=summ(T[(cell, lr)]["rs"]),
                                     late_auc_rel_same_lr_fresh=summ(T[(cell, lr)]["rel"])) for lr in grid},
               report_only_late_auc_rel_vs_tuned_fresh=dict(fresh_lr=flr, fresh_source=fsrc, **summ(rel_tuned)))
    out["stream"][cell] = res
out["problems"] = problems
json.dump(out, open("lr_ranking_raw.json", "w"), indent=1)
for fam, r in out["fresh"].items():
    print(f"FRESH {fam:7s} chosen {fmt(r['chosen_lr'])} edge={r['edge']} tie={r['tie_broken']} | " +
          "  ".join(f"{k}:{v['iqm']:.4f}" for k, v in r["trials"].items()) + f" | per-cell picks {r['by_cell_pick']}")
for cell, r in out["stream"].items():
    print(f"{cell:22s} chosen {fmt(r['chosen_lr']):7s} edge={r['edge']} tie={r['tie_broken']} | " +
          "  ".join(f"{k}:{v['raw_late_auc_stream']['iqm']:.4f}" for k, v in r["trials"].items()) +
          f" | CI {['%.3f'%x for x in r['trials'][fmt(r['chosen_lr'])]['raw_late_auc_stream']['ci95']]} | relTuned {r['report_only_late_auc_rel_vs_tuned_fresh']['iqm']:+.3f}")
print("PROBLEMS", problems)
