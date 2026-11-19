"""Addendum to counted_analysis.py (beb89648, passed review 14 Nov); passed files untouched.
1) LoRA figure v2: same data as f9d4bc75, n per panel in the titles (left n=23: seeds 704 and 714 have no
   late revisits; right n=25). 2) Per-seed kickstart effective KL updates on 700-724, read from the
   kickstart_1M counted chunks (field kickstart_effective_updates), chunk shas checked against the manifest."""
import json, os, hashlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C = os.environ.get("TARS_CELLS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "extracted"))
zs = json.load(open("zero_shot_counted_700-724.json"))
assert hashlib.sha256(open("zero_shot_counted_700-724.json", "rb").read()).hexdigest().startswith("729c4aba")
lo = next(e for e in zs if e["cell"] == "lora_1M"); seeds = lo["seeds"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, k, t in ((ax[0], "bwt_revisit_late", "Backward transfer on revisits (tasks 21-40)"), (ax[1], "bwt_final", "Backward transfer at end of stream")):
    x = np.array(lo["end_of_block_new_late"]["per_seed"], float); y = np.array(lo[k]["per_seed"], float)
    m = np.isfinite(x) & np.isfinite(y)
    a.scatter(x[m], y[m], s=25, alpha=0.7); a.axhline(0, c="grey", lw=.5)
    a.set_xlabel("End-of-block score, new tasks 21-40 (per seed)"); a.set_ylabel("Backward transfer (per seed)")
    a.set_title(f"{t}, n = {int(m.sum())}", fontsize=10)
fig.suptitle("LoRA ~1M, seeds 700-724: late-task learning vs backward transfer (points semi-transparent; overlaps darker)")
fig.tight_layout(); fig.savefig("lora_learning_vs_bwt_per_seed_v2.png", dpi=130)
d = f"{C}/kickstart_1M/kickstart_1M"; man = json.load(open(f"{d}/manifest.json")); eff = []
for c in man["chunks"]:
    p = os.path.join(d, os.path.basename(c.get("path", c.get("file"))))
    assert hashlib.sha256(open(p, "rb").read()).hexdigest() == c["sha256"]
    eff += np.asarray(np.load(p, allow_pickle=True)["kickstart_effective_updates"]).ravel().tolist()
assert len(eff) == 25
out = dict(kickstart_effective_updates=dict(zip(map(str, range(700, 725)), eff)), min=min(eff), max=max(eff),
           lora_left_panel_n=23, lora_left_panel_missing=[s for s, v in zip(seeds, lo["bwt_revisit_late"]["per_seed"]) if v != v])
json.dump(out, open("addendum_results.json", "w"), indent=1, default=float); print(out)
