"""Support table for the paper's results section (14 Nov 2026).
Per-cell point estimates of late AUC_rel and raw stream late AUC as IQM over seeds x tasks (25 x 20, new tasks 21-40),
the section 4 aggregate, using the same AUC_rel definition as h1prime.py (e63d7b30): stream AUC minus the family's
scratch-cell fresh AUC, per task. Re-implemented here (not imported, because importing h1prime.py rewrites its
results file). Cross-check: differences of these IQMs must reproduce the passed contrasts in fa814b9b.
Writes only results-support/per_cell_seeds_x_tasks.json."""
import hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.environ.get("TARS_PPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); import stream_metrics as sm
X = os.environ.get("TARS_CELLS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "extracted"))
CELLS = {"scratch_1M": "scratch_1M_r2/scratch_1M_r2", "pretrained_1M": "pretrained_1M/pretrained_1M",
         "frozen_pretrained_1M": "frozen_pretrained_1M/frozen_pretrained_1M", "frozen_random_1M": "frozen_random_1M/frozen_random_1M",
         "lora_1M": "lora_1M/lora_1M", "kickstart_1M": "kickstart_1M/kickstart_1M",
         "ln_scratch_1M": "ln_scratch_1M/ln_scratch_1M", "ln_pretrained_1M": "ln_pretrained_1M/ln_pretrained_1M",
         "pqn_scratch_1M": "pqn_scratch_1M/pqn_scratch_1M", "pqn_pretrained_1M": "pqn_pretrained_1M/pqn_pretrained_1M"}
FAM = lambda c: "pqn_scratch_1M" if c.startswith("pqn") else "ln_scratch_1M" if c.startswith("ln_") else "scratch_1M"
D = {}
for c, rel in CELLS.items():
    d = os.path.join(X, rel); man = json.load(open(f"{d}/manifest.json")); acc = {k: [] for k in ("s", "f", "sched", "isn")}
    for ch in man["chunks"]:
        p = os.path.join(d, os.path.basename(ch.get("path", ch.get("file"))))
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == ch["sha256"]
        z = dict(np.load(p, allow_pickle=True)); r = sm.analyze(z, require_device_info=True)
        acc["s"].append(r["auc_stream"]); acc["f"].append(r["auc_fresh"])
        acc["sched"].append(np.asarray(z["task_of_block"]) - 1); acc["isn"].append(np.asarray(z["is_new"]).astype(bool))
    D[c] = {k: np.concatenate(v) for k, v in acc.items()}
out = {}
for c in CELLS:
    d, ref = D[c], D[FAM(c)]
    rel = d["s"] - np.take_along_axis(ref["f"], d["sched"], 1)
    tn = np.cumsum(d["isn"], 1); late = d["isn"] & (tn > int(d["isn"][0].sum()) // 2); assert (late.sum(1) == 20).all()
    L = np.stack([rel[s][late[s]] for s in range(25)]); R = np.stack([d["s"][s][late[s]] for s in range(25)])
    out[c] = dict(late_auc_rel_iqm_seeds_x_tasks=float(sm.iqm(L.ravel())), raw_late_auc_stream_iqm_seeds_x_tasks=float(sm.iqm(R.ravel())))
g = lambda c: out[c]["late_auc_rel_iqm_seeds_x_tasks"]
out["crosscheck"] = dict(H1prime=g("pretrained_1M") - g("scratch_1M"), H6=g("frozen_pretrained_1M") - g("frozen_random_1M"),
                         H5=(g("ln_pretrained_1M") - g("ln_scratch_1M")) - (g("pqn_pretrained_1M") - g("pqn_scratch_1M")))
json.dump(out, open("per_cell_seeds_x_tasks.json", "w"), indent=1)
print(json.dumps(out, indent=1))
