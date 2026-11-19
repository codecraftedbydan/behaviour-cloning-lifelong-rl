"""Support table for the paper's results section (14 Nov 2026).
Reads the passed counted chunks read-only (tars-ppo-checks/counted/extracted, sha-checked against each manifest)
and computes, per cell, over the 25 x 13 revisit grid used by counted_analysis.py's forgetting:
  - learned_frac_pos: share of revisits whose task scored > 0 (normalised greedy eval) at the end of its previous block
  - learned_frac_thr: share with that score >= 0.8 (the threshold used elsewhere in the rules)
  - forgetting per seed (mean over the seed's 13 revisits) and IQM over seeds x revisits (cross-check vs fa814b9b)
Writes only results-support/revisit_learned_fraction.json next to this script."""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("TARS_PPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); import stream_metrics as sm
X = os.environ.get("TARS_CELLS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "extracted"))
CELLS = {"scratch_1M": "scratch_1M_r2/scratch_1M_r2", "pretrained_1M": "pretrained_1M/pretrained_1M",
         "frozen_pretrained_1M": "frozen_pretrained_1M/frozen_pretrained_1M", "frozen_random_1M": "frozen_random_1M/frozen_random_1M",
         "lora_1M": "lora_1M/lora_1M", "kickstart_1M": "kickstart_1M/kickstart_1M",
         "ln_scratch_1M": "ln_scratch_1M/ln_scratch_1M", "ln_pretrained_1M": "ln_pretrained_1M/ln_pretrained_1M",
         "pqn_scratch_1M": "pqn_scratch_1M/pqn_scratch_1M", "pqn_pretrained_1M": "pqn_pretrained_1M/pqn_pretrained_1M"}
out, problems = {}, []
for cell, rel in CELLS.items():
    d = os.path.join(X, rel); man = json.load(open(f"{d}/manifest.json"))
    learned, forget, seeds = [], [], []
    for c in man["chunks"]:
        p = os.path.join(d, os.path.basename(c.get("path", c.get("file"))))
        if hashlib.sha256(open(p, "rb").read()).hexdigest() != c["sha256"]: problems.append(f"{cell}: sha {p}")
        z = dict(np.load(p, allow_pickle=True)); r = sm.analyze(z, require_device_info=True)
        sched = np.asarray(z["task_of_block"]) - 1; opt = np.asarray(z["optimal_return"])
        ev = z["stream/eval"] / opt[:, None, :]
        for s, rows in enumerate(r["forgetting"]):
            last, lrow = {}, []
            for b in range(sched.shape[1]):
                t = int(sched[s, b])
                if t in last: lrow.append(float(ev[s, last[t], t]))
                last[t] = b
            learned.append(lrow); forget.append([x[2] for x in rows])
        seeds += list(np.asarray(z["seeds"]))
    L, F = np.array(learned), np.array(forget); assert L.shape == F.shape == (25, 13)
    out[cell] = dict(learned_frac_pos=float((L > 0).mean()), learned_frac_thr=float((L >= 0.8).mean()),
                     n_revisits=int(L.size),
                     forgetting_iqm_seeds_x_revisits=float(sm.iqm(F.ravel())),
                     forgetting_per_seed=F.mean(1).tolist(),
                     forgetting_per_seed_iqm=float(sm.iqm(F.mean(1))),
                     forgetting_per_seed_median=float(np.median(F.mean(1))),
                     forgetting_frac_nonzero=float((np.abs(F) > 1e-9).mean()),
                     seeds=[int(x) for x in seeds])
out["problems"] = problems
json.dump(out, open("revisit_learned_fraction.json", "w"), indent=1)
for c in CELLS: print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[c].items() if k not in ("forgetting_per_seed", "seeds")})
print("problems", problems)
