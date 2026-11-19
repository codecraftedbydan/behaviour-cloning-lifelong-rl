"""Metrics for the lifelong stream, per tars-test-rules v2 (sec. 2 and 5).

Inputs come from lifelong.run_condition. Shapes: S seeds, B blocks, T tasks, K updates.
Tasks are numbered from 1 everywhere in the outputs ("task 1" = the first task).

Per block (task occurrence) of seed s training task n:
  curve[k]        = training-rollout return at update k of the block / R_BFS(n), where
                    R_BFS is the noiseless BFS optimum and random return counts as 0
  AUC             = mean of curve over the first M = K updates (adaptation speed)
  steps-to-thr    = env steps until curve >= 0.8, capped at K (rules sec. 2); the fraction
                    of capped (never reached) values is reported separately
Fresh baseline: the same for a fresh agent trained on n alone for K updates (paired fresh
seed). Relative measure: AUC_rel = stream - fresh; revisits reuse the fresh run.

Slopes: per seed OLS on task number, then IQM + 95% percentile bootstrap CI (10k) over seeds.
  PRIMARY  slope_auc_rel_new_excl_task1: AUC_rel on new tasks, task 1 excluded.
           Pass rule (sec. 5): CI entirely below 0.
  secondary slope_auc_rel_new_incl_task1, raw slopes (slope_auc_raw_*), steps-to-threshold
           slopes (slope_steps_*), slopes over all occurrences (*_all_occ, x = block number),
           per-shift-type results, forgetting, and
  primacy_bias = AUC_rel(task 2) - AUC_rel(task 1): the drop at the first switch.
  late_auc_rel = mean AUC_rel over new tasks in the second half of the stream (sec. 7 D).
Sign: plasticity loss = AUC slope < 0 and steps slope > 0.

Forgetting (revisit of task n): normalised greedy eval on n at the end of n's last earlier
block, minus normalised eval on n at the end of the block before the revisit (before any
training on it). Positive = forgot.
"""
from __future__ import annotations

import numpy as np

SHIFT_NAMES = {0: "init", 1: "goal", 2: "dynamics", 3: "obs", 4: "layout", 5: "revisit"}


def iqm(x, axis=0):
    x = np.sort(np.asarray(x, float), axis=axis)
    n = x.shape[axis]
    k = int(np.floor(0.25 * n))
    return np.take(x, np.arange(k, n - k), axis=axis).mean(axis=axis)


def bootstrap_ci(x, stat=iqm, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap over seeds (x: (S,)). One task setting, so the stratified
    bootstrap of Agarwal et al. reduces to resampling seeds."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def summarize(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:  # e.g. entropy-adjusted slope on a stream too short for 3 coefficients
        return dict(iqm=float("nan"), mean=float("nan"), ci95=(float("nan"), float("nan")), n=0,
                    ci_excludes_zero=False)
    lo, hi = bootstrap_ci(x)
    return dict(iqm=float(iqm(x)), mean=float(x.mean()), ci95=(lo, hi), n=int(len(x)),
                # tolerance so float noise around an exactly-zero effect is not "significant"
                ci_excludes_zero=bool(lo > 1e-12 or hi < -1e-12))


def normalized_curves(rollout_return, schedule, optimal_return):
    """(S, B, K) stream curves normalised by each block's task optimum."""
    opt = np.take_along_axis(optimal_return, schedule, axis=1)  # (S, B)
    return rollout_return / opt[..., None]


def auc(curve, M=None):
    M = M or curve.shape[-1]
    return np.nanmean(curve[..., :M], axis=-1)


def steps_to_threshold(curve, thr, cap_at_k=True):
    """First update (1-based) at which curve >= thr. If never: K (rules: capped at K) or
    K+1 with cap_at_k=False (censoring visible). Multiply by batch size for env steps."""
    hit = curve >= thr
    K = curve.shape[-1]
    return np.where(hit.any(-1), hit.argmax(-1) + 1, K if cap_at_k else K + 1)


def relative(stream_val, fresh_val, schedule):
    """stream_val (S, B) minus fresh_val (S, T) gathered at each block's task."""
    return stream_val - np.take_along_axis(fresh_val, schedule, axis=1)


def ols_slope(y, x=None, mask=None):
    """Per-row OLS slope of y on x. y: (S, B); mask selects points (S, B) bool."""
    y = np.asarray(y, float)
    S, B = y.shape
    x = np.broadcast_to(np.arange(B) if x is None else x, (S, B)).astype(float)
    mask = np.ones((S, B), bool) if mask is None else mask
    out = np.full(S, np.nan)
    for s in range(S):
        m = mask[s] & np.isfinite(y[s])
        if m.sum() >= 2 and np.ptp(x[s][m]) > 0:
            out[s] = np.polyfit(x[s][m], y[s][m], 1)[0]
    return out


def forgetting(eval_norm, eval0_norm, schedule):
    """eval_norm (S, B, T): normalised eval after each block; eval0_norm (S, T) before
    training; schedule (S, B) 0-based task indices. Returns per seed a list of
    (block number, task number, forgetting), both 1-based."""
    S, B, _ = eval_norm.shape
    out = []
    for s in range(S):
        rows, last = [], {}
        for b in range(B):
            t = int(schedule[s, b])
            if t in last:
                before = eval_norm[s, b - 1, t] if b > 0 else eval0_norm[s, t]
                rows.append((b + 1, t + 1, float(eval_norm[s, last[t], t] - before)))
            last[t] = b
        out.append(rows)
    return out


PRIMARY = "slope_auc_rel_new_excl_task1"
# Files written before run records stated the source; the AUC came from the same place.
AUC_SOURCE_LEGACY = "training-rollout return of the stochastic policy (not recorded in file)"


class PlatformMismatch(ValueError):
    pass


def _check_same(d, field, require):
    """stream/, fresh/ and (if present) headprobe/ must agree on `field`."""
    sp, fp = d.get(f"stream/{field}"), d.get(f"fresh/{field}")
    if sp is None or fp is None:
        if require:
            raise PlatformMismatch(f"stream/fresh {field} not recorded")
        return None
    sp, fp = str(np.asarray(sp)), str(np.asarray(fp))
    if sp != fp:
        raise PlatformMismatch(f"stream run has {field}={sp!r} but fresh run has {fp!r}; "
                               "refusing to compute relative metrics across them")
    if "head_probe_tasks" in d:
        hp = d.get(f"headprobe/{field}")
        if hp is None:
            if require:
                raise PlatformMismatch(f"headprobe {field} not recorded")
        elif str(np.asarray(hp)) != sp:
            raise PlatformMismatch(f"head probe has {field}={str(np.asarray(hp))!r} but stream has {sp!r}")
    return sp


def check_same_platform(d, require=False):
    """Relative metrics compare stream and fresh runs; refuse if they ran on different
    platforms (cpu vs gpu numerics/throughput differ). Files written before device
    logging have no info: accepted unless require=True."""
    return _check_same(d, "device_platform", require)


def check_same_precision(d, require=False):
    """Same for jax_default_matmul_precision ("default" vs "highest": TF32 vs fp32 on GPU)."""
    return _check_same(d, "matmul_precision", require)


def entropy_adjusted_slope(rel, task_num, entropy, mask):
    """Per seed OLS of rel on [1, task_num, entropy] over the masked blocks; returns the
    task-number coefficient (NaN if < 4 points or the design is rank-deficient)."""
    S_ = rel.shape[0]
    out = np.full(S_, np.nan)
    for s in range(S_):
        m = mask[s] & np.isfinite(rel[s]) & np.isfinite(entropy[s])
        if m.sum() < 4:
            continue
        X = np.stack([np.ones(m.sum()), task_num[s][m].astype(float), entropy[s][m].astype(float)], 1)
        if np.linalg.matrix_rank(X) < 3:
            continue
        out[s] = np.linalg.lstsq(X, rel[s][m], rcond=None)[0][1]
    return out


def analyze(d, thr=0.8, M=None, require_device_info=False):
    """Full metric set from a run_condition result dict (or loaded npz)."""
    platform = check_same_platform(d, require_device_info)
    precision = check_same_precision(d, require_device_info)
    if "task_of_block" in d:
        sched = np.asarray(d["task_of_block"]) - 1   # 0-based indices for array lookups
    else:  # files written before 1-based numbering
        sched = np.asarray(d["schedule"])
    is_new, opt = np.asarray(d["is_new"]).astype(bool), np.asarray(d["optimal_return"])
    S, B = sched.shape
    T = opt.shape[1]
    batch = int(d["batch_size"])
    cs = normalized_curves(d["stream/metrics/rollout_return"], sched, opt)       # (S, B, K)
    cf = d["fresh/metrics/rollout_return"][:, :, 0] / opt[..., None]            # (S, T, K)
    K = cs.shape[-1]
    task_num = np.cumsum(is_new, axis=1)        # 1-based number of the latest new task
    block_num = np.broadcast_to(np.arange(1, B + 1), (S, B))
    excl1 = is_new & (task_num > 1)
    res = dict(thr=thr, M=M or K, K=K, S=S, B=B, T=T, batch_size=batch, platform=platform,
               matmul_precision=precision,
               task_of_block=sched + 1, primary=PRIMARY)
    st_s, st_f = steps_to_threshold(cs, thr), steps_to_threshold(cf, thr)
    vals = {"auc": (auc(cs, M), auc(cf, M)),
            "steps": (st_s.astype(float) * batch, st_f.astype(float) * batch)}  # env steps
    for name, (sv, fv) in vals.items():
        rel = relative(sv, fv, sched)
        res[f"{name}_stream"], res[f"{name}_fresh"], res[f"{name}_rel"] = sv, fv, rel
        for variant, x, mask in (("new_excl_task1", task_num, excl1),
                                 ("new_incl_task1", task_num, is_new),
                                 ("all_occ", block_num, None)):
            for kind, y in (("rel", rel), ("raw", sv)):
                sl = ols_slope(y, x, mask)
                res[f"slope_{name}_{kind}_{variant}"] = sl
                res[f"slope_{name}_{kind}_{variant}_summary"] = summarize(sl)
    res["steps_capped_frac_stream"] = float((steps_to_threshold(cs, thr, False) > K).mean())
    res["steps_capped_frac_fresh"] = float((steps_to_threshold(cf, thr, False) > K).mean())
    # Secondary: entropy-adjusted primary slope. Per seed, AUC_rel (new tasks, task 1
    # excluded) ~ 1 + task number + probe-batch policy entropy at the switch; the task-number
    # coefficient is the decline NOT explained by entropy collapse. Primary stays unadjusted.
    if "stream/switch_entropy" in d:
        ea = entropy_adjusted_slope(res["auc_rel"], task_num, np.asarray(d["stream/switch_entropy"]), excl1)
        res[f"{PRIMARY}_entropy_adj"] = ea
        res[f"{PRIMARY}_entropy_adj_summary"] = summarize(ea)
    # Primacy bias: AUC_rel of task 2 minus task 1 (the first switch).
    rel = res["auc_rel"]
    b1 = np.argmax(is_new & (task_num == 1), axis=1)
    b2 = np.argmax(is_new & (task_num == 2), axis=1)
    pb = rel[np.arange(S), b2] - rel[np.arange(S), b1]
    res["primacy_bias"], res["primacy_bias_summary"] = pb, summarize(pb)
    # Late-stream AUC_rel (sec. 7): new tasks in the second half of the stream.
    n_new = int(is_new[0].sum())
    late = is_new & (task_num > n_new // 2)
    lr = np.array([rel[s][late[s]].mean() for s in range(S)])
    res["late_auc_rel"], res["late_auc_rel_summary"] = lr, summarize(lr)
    # Shared layout (stage 1): share of new tasks still on the shared obs layout / action
    # perm, overall and in the late window, per seed (logged, sec. 7 interpretation).
    if "task_shared_obs_layout" in d:
        tob = np.asarray(d["task_of_block"]) - 1
        for name in ("obs_layout", "action_perm"):
            sh = np.asarray(d[f"task_shared_{name}"])
            blk = np.take_along_axis(sh, tob, axis=1)
            res[f"shared_{name}_share_new"] = np.array([blk[s][is_new[s]].mean() for s in range(S)])
            res[f"shared_{name}_share_late"] = np.array([blk[s][late[s]].mean() for s in range(S)])
    # Per shift type (new tasks, task 1 excluded): pooled per seed, then IQM/CI.
    shift = np.asarray(d["block_shift"])
    res["by_shift"] = {}
    for code in np.unique(shift):
        nm = SHIFT_NAMES[int(code)]
        if nm in ("init",):
            continue
        m = shift == code
        per_seed = np.array([np.nanmean(rel[s][m[s]]) if m[s].any() else np.nan for s in range(S)])
        entry = dict(count=int(m.sum()), auc_rel=summarize(per_seed))
        if nm != "revisit":
            sl = ols_slope(rel, task_num, m & excl1)
            entry["auc_rel_slope"] = None if np.isnan(sl).all() else summarize(sl)
            per_seed_st = np.array([np.nanmean(res["steps_rel"][s][m[s]]) if m[s].any() else np.nan
                                    for s in range(S)])
            entry["steps_rel"] = summarize(per_seed_st)
        res["by_shift"][nm] = entry
    # Forgetting
    ev = d["stream/eval"] / opt[:, None, :]
    ev0 = d["stream/eval0"] / opt
    f = forgetting(ev, ev0, sched)
    res["forgetting"] = f
    res["forgetting_per_seed"] = np.array([np.mean([r[2] for r in rows]) if rows else np.nan for rows in f])
    if np.isfinite(res["forgetting_per_seed"]).any():
        res["forgetting_summary"] = summarize(res["forgetting_per_seed"])
    # Final per-task return (sec. 2): normalised eval on the block's task at its end.
    res["final_return"] = np.take_along_axis(ev, sched[..., None], axis=2)[..., 0]
    res["curves_stream"], res["curves_fresh"] = cs, cf
    res["auc_source"] = str(np.asarray(d["auc_source"])) if "auc_source" in d else AUC_SOURCE_LEGACY
    # Head-reset probe (plasticity vs exploration): frozen stream trunk + fresh heads.
    if "head_probe_tasks" in d:
        hp_t = np.asarray(d["head_probe_tasks"]) - 1                       # (P,) 0-based
        hp_b = np.asarray(d["head_probe_blocks"]) - 1                      # (P,) 0-based
        ch = d["headprobe/metrics/rollout_return"][:, :, 0] / opt[:, hp_t, None]  # (S, P, K)
        a_h = auc(ch, M)
        a_f = res["auc_fresh"][:, hp_t]
        a_s = res["auc_stream"][:, hp_b]
        res["head_probe"] = dict(
            tasks=hp_t + 1, blocks=hp_b + 1,
            auc_headprobe=a_h, auc_rel_headprobe=a_h - a_f, auc_rel_stream=a_s - a_f,
            # > 0: re-initialised heads on the frozen trunk recover part of the stream deficit
            headprobe_minus_stream=a_h - a_s,
            auc_rel_headprobe_summary=summarize(np.nanmean(a_h - a_f, axis=1)),
            auc_rel_stream_summary=summarize(np.nanmean(a_s - a_f, axis=1)),
            headprobe_minus_stream_summary=summarize(np.nanmean(a_h - a_s, axis=1)))
    return res
