"""Stage-1 dev-seed checks for the BC arm (dev seeds 900+ only; reserved seeds are refused;
runs from a clean tracked tree unless --allow-dirty). Stream config: STAGE1_STREAM (shared
layout on). BC per seed from that seed's scratch init (bc.pretrain(seed=s)).

  1. BC per seed: best epoch, held-out (validation demo task) accuracy at the best epoch,
     IQM over seeds with a 95% percentile-bootstrap CI. Pre-set gate (decision, 31 Oct; not
     tuned): best epoch >= 1 on every seed AND held-out IQM >= 0.40 AND CI lower bound
     > 0.30. Also the training-demo accuracy curve (seed mean) and the first epoch at which
     training accuracy reaches 0.90 (fixed-length fallback, cap = max_epochs = 200):
     pass 1 trains each seed until it reaches 0.90 (bc.train_bc(until_train_acc=0.9)),
     pass 2 re-trains every seed for the longest pass-1 length (deterministic, same RNG
     path), so the seed-mean curve is exact.
  2. Critic warm-up re-measure: pretrained (per-seed BC) agent on task 1, critic-only
     updates for --warmup-horizon updates. n* = smallest n after which no later update
     lowers the seed-mean value loss by more than --tol (5%) relative to update n; cap 10
     (10% of K). Per-seed values reported too.
  3. Task-1 AUC_rel = AUC(stream agent, task 1) - AUC(fresh agent, task 1), pretrained
     (per-seed BC + measured warm-up) vs scratch, per seed (measured, not gating).
  Extra (not gating): greedy normalised return on the 5 retention demo tasks, per-seed BC
  params vs the same seed's scratch init; BC accuracy on retention-task demos.

Usage: python scripts/bc_dev_checks.py --seeds 900-904 --hidden-size 950 --out-dir DIR
"""
import argparse
import dataclasses
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bc  # noqa: E402
import lifelong  # noqa: E402
import provenance  # noqa: E402
import ppo  # noqa: E402
import seeds as SD  # noqa: E402
import stream as S  # noqa: E402
import stream_metrics as SM  # noqa: E402


def stop_improving(curve, tol):
    """Smallest n (1-based count of updates) such that min(curve[n:]) >= (1 - tol) * curve[n-1]."""
    c = np.asarray(curve, float)
    for n in range(1, len(c)):
        if np.nanmin(c[n:]) >= (1.0 - tol) * c[n - 1]:
            return n
    return len(c)


def norm_eval(env, network, config, params, tasks_stacked, opt, key, max_steps):
    r = np.asarray(ppo.evaluate(env, network, config, params, tasks_stacked, key, max_steps))
    return r / opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="900-904")
    p.add_argument("--hidden-size", type=int, default=950)
    p.add_argument("--warmup-horizon", type=int, default=30)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=None, help="override the measured warm-up")
    p.add_argument("--fallback-acc", type=float, default=0.90)
    p.add_argument("--allow-dirty", action="store_true", help="dev seeds only; recorded")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    seeds = ppo.parse_seeds(a.seeds)
    if any(s < SD.DEV_MIN_SEED or s in SD.RESERVED_SEEDS for s in seeds):
        raise SystemExit(f"dev seeds >= {SD.DEV_MIN_SEED} only")
    lifelong.check_seeds(seeds, False)
    prov = provenance.check_clean(seeds, a.allow_dirty)
    os.makedirs(a.out_dir, exist_ok=True)
    rep = dict(seeds=list(seeds), hidden_size=a.hidden_size, git_commit=lifelong.git_commit(),
               provenance={k: v for k, v in prov.items() if k != "source_hashes"})
    t00 = time.time()

    config = ppo.Config(**dict(ppo.resolve_presets("lifelong_stream"), hidden_size=a.hidden_size))
    full = S.STAGE1_STREAM                    # registered stream config (BC pool check)
    task1 = dataclasses.replace(full, num_new_tasks=1, head_probe_every=0)
    K = full.block_updates
    cap = 10
    bccfg = bc.BCConfig()
    streams = [S.make_stream(full, s) for s in seeds]
    rep["task1_shared"] = [bool(s.shared_obs[0] and s.shared_action[0]) for s in streams]
    env, _, network, obs_shape = ppo.make_env_and_network(config)
    rep["param_count"] = int(sum(x.size for x in jax.tree.leaves(bc.init_params(config))))

    # (1) BC per seed
    t0 = time.time()
    demos = bc.demo_data(full, streams, bccfg)
    rt = demos[2]
    bcp, per = {}, {}
    for s in seeds:
        params, meta, _ = bc.pretrain(config, full, streams, s, bccfg, demos=demos)
        meta["git_commit"] = rep["git_commit"]
        meta.update(prov)
        bc.save_checkpoint(os.path.join(a.out_dir, bc.checkpoint_name(config, s)), params, meta)
        bcp[s] = params
        per[s] = {k: meta[k] for k in ("best_epoch", "epochs_run", "best_val_loss", "best_val_acc",
                                       "best_val_entropy", "best_train_acc")}
        per[s]["train_acc_curve"] = meta["history"]["train_acc"]
        per[s]["val_acc_curve"] = meta["history"]["val_acc"]
        per[s]["val_loss_curve"] = meta["history"]["val_loss"]
        print("BC seed", s, json.dumps({k: v for k, v in per[s].items() if "curve" not in k}), flush=True)
    acc = np.array([per[s]["best_val_acc"] for s in seeds])
    summ = SM.summarize(acc)
    be = [per[s]["best_epoch"] for s in seeds]
    gate = dict(best_epoch_ge1_all=bool(min(be) >= 1), heldout_iqm=summ["iqm"], heldout_ci95=summ["ci95"],
                iqm_ge_040=bool(summ["iqm"] >= 0.40), ci_lo_gt_030=bool(summ["ci95"][0] > 0.30))
    gate["passed"] = bool(gate["best_epoch_ge1_all"] and gate["iqm_ge_040"] and gate["ci_lo_gt_030"])
    L = min(len(per[s]["train_acc_curve"]) for s in seeds)
    rep["bc"] = dict(per_seed=per, best_epoch=be, heldout_acc_at_best=acc.tolist(), gate=gate,
                     num_train_samples=meta["num_train_samples"], num_val_samples=meta["num_val_samples"],
                     demo_success=meta["demo_success"], retention_tasks_hash=meta["retention_tasks_hash"],
                     es_train_acc_seed_mean_common_prefix=np.mean(
                         [per[s]["train_acc_curve"][:L] for s in seeds], 0).tolist(),
                     seconds=time.time() - t0)
    print("BC GATE", json.dumps(gate), flush=True)

    # fallback length: first epoch with training-demo accuracy >= fallback_acc
    t0 = time.time()
    _, _, _, dtr, dva = demos
    p1 = {}
    for s in seeds:
        _, info = bc.train_bc(config, dtr, dva, bccfg, init=lifelong.scratch_init_params(config, s),
                              until_train_acc=a.fallback_acc)
        ta = info["history"]["train_acc"]
        hit = [e for e, x in zip(info["history"]["epoch"], ta) if x >= a.fallback_acc]
        p1[s] = hit[0] if hit else None
    E = max([v for v in p1.values() if v is not None] or [bccfg.max_epochs])
    curves = []
    for s in seeds:
        _, info = bc.train_bc(config, dtr, dva, dataclasses.replace(bccfg, max_epochs=E),
                              init=lifelong.scratch_init_params(config, s), until_train_acc=2.0)
        curves.append(info["history"]["train_acc"])
    mc = np.mean(curves, 0)
    hit = [e for e in range(len(mc)) if mc[e] >= a.fallback_acc]
    rep["fallback"] = dict(threshold=a.fallback_acc, cap=bccfg.max_epochs, per_seed_first_epoch=p1,
                           seed_mean_first_epoch=hit[0] if hit else None, epochs=E,
                           train_acc_seed_mean_curve=mc.tolist(), seconds=time.time() - t0)
    print("FALLBACK", json.dumps({k: v for k, v in rep["fallback"].items() if "curve" not in k}), flush=True)

    # extra: retention tasks (greedy), BC accuracy on retention-task demos
    ret_stacked = S.stack_tasks(rt, full.max_steps)
    opt = np.array([1.0 - 0.9 * S._bfs_np(t["walls"], tuple(t["goal"]))[tuple(t["start"])] / full.max_steps
                    for t in rt])
    ek = jax.random.PRNGKey(4242)
    dret = bc.make_demos(rt, full, bccfg, jax.random.PRNGKey(4243))
    ex = {}
    for s in seeds:
        si = lifelong.scratch_init_params(config, s)
        ex[s] = dict(
            pretrained=float(norm_eval(env, network, config, bcp[s], ret_stacked, opt, ek, full.max_steps).mean()),
            scratch=float(norm_eval(env, network, config, si, ret_stacked, opt, ek, full.max_steps).mean()),
            retention_demo_acc=float(bc.bc_stats(network, bcp[s], jnp.asarray(dret["obs"]),
                                                 jnp.asarray(dret["label"], jnp.int32))[1]))
    rep["retention_extra"] = ex
    print("RETENTION", json.dumps(ex), flush=True)

    # (2) warm-up re-measure
    H = a.warmup_horizon
    wcfg = dataclasses.replace(task1, block_updates=H)
    w = lifelong.run_condition(bc.pretrained_config(config, H), wcfg, seeds, fresh=False,
                               verbose=False, seed_init_params=[bcp[s] for s in seeds],
                               allow_dirty=a.allow_dirty)
    vl = np.asarray(w["stream/metrics/value_loss"]).reshape(len(seeds), H)
    ev = np.asarray(w["stream/metrics/explained_variance"]).reshape(len(seeds), H)
    per_seed = [stop_improving(v, a.tol) for v in vl]
    mean_n = stop_improving(vl.mean(0), a.tol)
    measured = min(mean_n, cap)
    rep["warmup"] = dict(horizon=H, tol=a.tol, cap=cap, per_seed=per_seed, seed_mean_curve=mean_n,
                         measured=measured, value_loss_mean_curve=vl.mean(0).tolist(),
                         explained_variance_mean_curve=ev.mean(0).tolist())
    print("WARMUP", json.dumps(rep["warmup"]), flush=True)
    W = measured if a.warmup is None else a.warmup

    # (3) task-1 AUC_rel, pretrained vs scratch
    res = {}
    for arm, cfg, sip, rtt in (("pretrained", bc.pretrained_config(config, W), [bcp[s] for s in seeds], rt),
                               ("scratch", config, None, None)):
        t0 = time.time()
        d = lifelong.run_condition(cfg, task1, seeds, fresh=True, verbose=False, seed_init_params=sip,
                                   retention_tasks=rtt, allow_dirty=a.allow_dirty)
        sched = np.asarray(d["task_of_block"]) - 1
        o = np.asarray(d["optimal_return"])
        cs = SM.normalized_curves(np.asarray(d["stream/metrics/rollout_return"]), sched, o)
        cf = np.asarray(d["fresh/metrics/rollout_return"])[:, :, 0] / o[..., None]
        a_s, a_f = SM.auc(cs)[:, 0], SM.auc(cf)[:, 0]
        res[arm] = dict(auc_stream=a_s.tolist(), auc_fresh=a_f.tolist(), auc_rel=(a_s - a_f).tolist(),
                        auc_rel_mean=float((a_s - a_f).mean()), seconds=time.time() - t0,
                        final10_stream=np.nanmean(cs[:, 0, -10:], 1).tolist())
        if rtt is not None:
            r0 = np.asarray(d["stream/retention0"]) / d["retention_optimal_return"]
            r1 = np.asarray(d["stream/retention"])[:, 0] / d["retention_optimal_return"]
            res[arm]["retention_before"] = float(r0.mean())
            res[arm]["retention_after_task1"] = float(r1.mean())
        print(arm, json.dumps(res[arm]), flush=True)
    rep["task1"] = dict(warmup=W, K=K, **res)
    rep["seconds"] = time.time() - t00
    with open(os.path.join(a.out_dir, "bc_dev_checks.json"), "w") as f:
        json.dump(rep, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"done in {rep['seconds']:.0f}s")


if __name__ == "__main__":
    main()
