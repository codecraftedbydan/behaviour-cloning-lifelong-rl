"""Read-only: per-seed kickstart KL windows from the passed counted kickstart_1M chunks (seeds 700-724).
Window = run of blocks from block 1 with kickstart_on_block True (gate rule: on until first block that is not
shared obs layout + shared action perm; off for good after). Reads arrays only; writes one JSON."""
import glob, hashlib, json, sys
import numpy as np
cell = sys.argv[1]; out = sys.argv[2]
files = sorted(glob.glob(f"{cell}/chunk_*.npz"))
rows, sha = [], {}
for f in files:
    sha[f.split("/")[-1]] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    p = np.load(f, allow_pickle=False)
    on, upd, coef = p["kickstart_on_block"], p["kickstart_effective_updates_block"], p["kickstart_coef_block"]
    tob, new = p["task_of_block"], p["is_new"]
    for i, s in enumerate(p["seeds"]):
        o = on[i].astype(bool); nb = int(o.sum())
        first_off = int(np.argmin(o)) if not o.all() else len(o)
        assert o[:nb].all() and not o[nb:].any(), f"seed {s}: window not contiguous from block 1"
        rows.append(dict(seed=int(s), window_blocks=nb, first_off_block_1based=first_off + 1,
            window_tasks=[int(t) for t in tob[i][:nb]], window_new_blocks=int(new[i][:nb].sum()),
            effective_updates=int(p["kickstart_effective_updates"][i]),
            effective_updates_check=int(upd[i].sum()),
            revisit_effective_updates=int(p["kickstart_effective_updates_revisit"][i]),
            coef_in_window=sorted({float(c) for c in coef[i][:nb]}), coef_outside=sorted({float(c) for c in coef[i][nb:]})))
    rule = str(p["kickstart_gate_rule"])
bad = [r["seed"] for r in rows if r["effective_updates"] != r["effective_updates_check"]]
res = dict(cell=cell, chunk_sha256=sha, gate_rule=rule, n_seeds=len(rows), mismatched_update_sums=bad, per_seed=rows)
json.dump(res, open(out, "w"), indent=1)
w = np.array([r["window_blocks"] for r in rows]); u = np.array([r["effective_updates"] for r in rows])
print("seeds", rows[0]["seed"], "-", rows[-1]["seed"], "n", len(rows), "mismatch", bad)
print("window_blocks counts", dict(zip(*np.unique(w, return_counts=True))))
print("eff_updates counts", dict(zip(*np.unique(u, return_counts=True))), "revisit>0", [r["seed"] for r in rows if r["revisit_effective_updates"]])
print("coef sets", {tuple(r["coef_in_window"]) for r in rows}, {tuple(r["coef_outside"]) for r in rows})
for r in rows: print(r["seed"], r["window_blocks"], r["window_tasks"], r["effective_updates"])
