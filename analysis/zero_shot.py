"""Report-only zero-shot and backward-transfer metrics, computed after the run from saved chunks.

Outside the frozen repo (tars-ppo @ 33660f6); reads only arrays the runner already writes:
  stream/eval  (S, B, T)  greedy-config eval on every task of the chain after block b
  stream/eval0 (S, T)     same eval before the first update
  fresh/eval0  (S, T, 1)  fresh agent's eval before its first update (its random init)
  optimal_return (S, T), task_of_block (S, B) 1-based, is_new (S, B)
All scores are normalised by optimal_return, as in stream_metrics.analyze.

Definitions (b, t 0-based internally; written out 1-based):
  before[s, b, t]  = eval0[s, t] if b == 0 else eval[s, b-1, t]
  zero_shot[s, b]  = before[s, b, task(b)]          score on the block's task before its first update
  zero_shot_minus_fresh_init[s, b] = zero_shot - fresh_eval0[s, task(b)]
  bwt_revisit[s, b] (revisit blocks only) = zero_shot[s, b] - eval[s, last(task), task]
      where last(task) is the most recent earlier block that trained the task. Equals
      -stream_metrics.forgetting row for that block (checked).
  bwt_final[s, t] (GEM-style, Lopez-Paz & Ranzato 2017) = eval[s, B-1, t] - eval[s, first(t), t]
      for every task first trained before the final block.
Report-only. Not a pass rule.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np

KEYS = ("stream/eval", "stream/eval0", "fresh/eval0", "optimal_return", "task_of_block", "is_new", "seeds")


sys.path.insert(0, os.environ.get("TARS_PPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import stream_metrics as SM  # frozen repo, read-only import: same IQM as the paper's metrics


def iqm(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float("nan") if x.size == 0 else float(SM.iqm(x))


def compute(ev, ev0, fev0, opt, sched, is_new):
    """ev (S,B,T), ev0 (S,T), fev0 (S,T), opt (S,T), sched (S,B) 0-based, is_new (S,B) bool."""
    S, B, T = ev.shape
    evn, ev0n, fev0n = ev / opt[:, None, :], ev0 / opt, fev0 / opt
    before = np.concatenate([ev0n[:, None, :], evn[:, :-1, :]], axis=1)          # (S,B,T)
    zs = np.take_along_axis(before, sched[..., None], axis=2)[..., 0]            # (S,B)
    zs_rel = zs - np.take_along_axis(fev0n, sched, axis=1)
    bwt_rev = np.full((S, B), np.nan)
    bwt_final = np.full((S, T), np.nan)
    for s in range(S):
        last, first = {}, {}
        for b in range(B):
            t = int(sched[s, b])
            if t in last:
                bwt_rev[s, b] = zs[s, b] - evn[s, last[t], t]
            last[t] = b
            first.setdefault(t, b)
        for t, b in first.items():
            if b < B - 1:
                bwt_final[s, t] = evn[s, B - 1, t] - evn[s, b, t]
    end = np.take_along_axis(evn, sched[..., None], axis=2)[..., 0]               # (S,B) end-of-block score
    # task number of each block's task = order of the task's first appearance (1-based), so a
    # revisit carries the number of the task it revisits (same numbering as H1': new tasks 21-40)
    tnum = np.zeros((S, B), int)
    for s in range(S):
        order = {}
        for b in range(B):
            order.setdefault(int(sched[s, b]), len(order) + 1)
            tnum[s, b] = order[int(sched[s, b])]
    return dict(zero_shot=zs, zero_shot_minus_fresh_init=zs_rel, end_of_block=end, bwt_revisit=bwt_rev,
                bwt_final=bwt_final, is_new=is_new, task_num=tnum)


LATE = (21, 40)  # H1' window: new tasks 21-40 (1-based task numbers)


def summarise(r, late=LATE):
    new, rev = r["is_new"], ~r["is_new"]
    inw = (r["task_num"] >= late[0]) & (r["task_num"] <= late[1])
    per_seed = lambda a, m: np.array([np.nanmean(a[s][m[s]]) if m[s].any() else np.nan for s in range(a.shape[0])])
    out = {}
    for name in ("zero_shot", "zero_shot_minus_fresh_init", "end_of_block"):
        for tag, m in (("new_late", new & inw), ("new", new), ("revisit_late", rev & inw),
                       ("revisit_early", rev & ~inw), ("revisit", rev)):
            v = per_seed(r[name], m); out[f"{name}_{tag}"] = dict(iqm=iqm(v), per_seed=v.tolist())
    for tag, m in (("late", rev & inw), ("early", rev & ~inw), ("", rev)):
        v = per_seed(r["bwt_revisit"], m)
        out["bwt_revisit" + (f"_{tag}" if tag else "")] = dict(iqm=iqm(v), per_seed=v.tolist())
    out["late_window_task_numbers"] = list(late)
    v = np.nanmean(r["bwt_final"], axis=1); out["bwt_final"] = dict(iqm=iqm(v), per_seed=v.tolist())
    return out


def load_cell(cell_dir):
    files = sorted(glob.glob(os.path.join(cell_dir, "chunk_*.npz")))
    if not files:
        raise FileNotFoundError(f"no chunks in {cell_dir}")
    parts = [np.load(f, allow_pickle=False) for f in files]
    for p, f in zip(parts, files):
        missing = [k for k in KEYS if k not in p.files]
        if missing:
            raise KeyError(f"{f}: missing {missing}")
    d = {k: np.concatenate([p[k] for p in parts], axis=0) for k in KEYS}
    cfg = json.loads(str(parts[0]["config"])) if "config" in parts[0].files else {}
    sha = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}
    return d, cfg, sha


def run(cell_dir):
    d, cfg, sha = load_cell(cell_dir)
    r = compute(d["stream/eval"], d["stream/eval0"], d["fresh/eval0"][..., 0], d["optimal_return"],
                np.asarray(d["task_of_block"]) - 1, np.asarray(d["is_new"]).astype(bool))
    return dict(cell=os.path.basename(os.path.normpath(cell_dir)), seeds=d["seeds"].tolist(),
                eval_greedy=cfg.get("eval_greedy"), eval_episodes=cfg.get("eval_episodes"),
                chunk_sha256=sha, report_only=True, **summarise(r))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = [run(c) for c in a.cells]
    json.dump(res, open(a.out, "w"), indent=1)
    for x in res:
        print(f"{x['cell']:32s} greedy={x['eval_greedy']} zs_new21-40={x['zero_shot_new_late']['iqm']:.3f} "
              f"zs_rev21-40={x['zero_shot_revisit_late']['iqm']:.3f} end_new21-40={x['end_of_block_new_late']['iqm']:.3f} "
              f"bwt_rev21-40={x['bwt_revisit_late']['iqm']:.3f} | all: zs_new={x['zero_shot_new']['iqm']:.3f} "
              f"zs_rev={x['zero_shot_revisit']['iqm']:.3f} zs-fresh_new={x['zero_shot_minus_fresh_init_new']['iqm']:.3f} "
              f"end_new={x['end_of_block_new']['iqm']:.3f} bwt_rev={x['bwt_revisit']['iqm']:.3f} bwt_final={x['bwt_final']['iqm']:.3f}")
