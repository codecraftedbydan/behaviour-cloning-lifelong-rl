"""Kickstarting-coefficient sweep hook (stage-1 cell 12; tars-test-rules sec. 7).

Equal-trial sweep: exactly 4 coefficients (arms.KICKSTART_SWEEP_TRIALS), each a full
kickstart-arm run (BC init per seed, full FT, critic warm-up, KL(pi_BC || pi) to the same
seed's frozen BC policy) on the HELD-OUT sweep stream: stream seeds 750-759 (fresh
850-859), tuning only, never the test seeds. The candidate values are not set by the rules
and must be given. Dev seeds >= 900 are accepted for smoke tests only: such runs are
MECHANICS-ONLY and are marked ranking_binding: false (sweep.json and every trial record).
ranking_binding is true only for stream seeds exactly 750-759 with fresh agents exactly
850-859 (arms.sweep_binding).

Output: one run record per coefficient (kickstart_c<i>.npz) and sweep.json with the
per-trial seed-mean late AUC_rel summary (IQM, mean, 95% CI). Selection metric (set by
decision): IQM over seeds of late AUC_rel (mean AUC_rel over new tasks 21-40 of the 40-task
stream, i.e. the second half) on stream seeds 750-759 with fresh agents on 850-859. Only a
binding run writes "ranked_best" (the top coefficient by that metric); mechanics-only runs
(dev seeds) write NO ranked_best at all. ranked_best is not written anywhere else; adopting
it for cell 12 is a separate, explicit step.

  PYTHONPATH=. python scripts/kickstart_sweep.py --seeds 750-759 --allow-official-seeds \
      --coefs a,b,c,d --bc-dir <ckpts> --critic-warmup-updates W --out-dir <dir> [stream args]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import numpy as np

import arms
import lifelong
import ppo
import provenance
import stream as S
import stream_metrics as SM


SELECTION_METRIC = ("IQM over seeds of late_auc_rel (mean AUC_rel over new tasks 21-40), stream seeds "
                    "750-759, fresh 850-859 (set in advance)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", required=True)
    p.add_argument("--coefs", required=True, help="exactly 4 comma-separated coefficients > 0")
    p.add_argument("--bc-dir", required=True)
    p.add_argument("--critic-warmup-updates", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--preset", default="lifelong_stream")
    p.add_argument("--size", choices=sorted(arms.SIZE_TARGETS), default="1M")
    p.add_argument("--num-new-tasks", type=int, default=lifelong.StreamConfig_default("num_new_tasks"))
    p.add_argument("--block-updates", type=int, default=lifelong.StreamConfig_default("block_updates"))
    p.add_argument("--revisit-every", type=int, default=lifelong.StreamConfig_default("revisit_every"))
    p.add_argument("--pilot-stream", action="store_true",
                   help="use the pilot stream config instead of STAGE1 (shared layout); dev only")
    p.add_argument("--matmul-precision", choices=ppo.MATMUL_PRECISIONS, default="default")
    p.add_argument("--allow-official-seeds", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args(argv)

    ppo.set_matmul_precision(a.matmul_precision)
    seeds = ppo.parse_seeds(a.seeds)
    try:
        coefs, seeds = arms.check_kickstart_sweep(a.coefs.split(","), seeds)
    except ValueError as e:
        raise SystemExit(str(e))
    lifelong.check_seeds(seeds, a.allow_official_seeds)
    prov = provenance.check_clean(seeds, a.allow_dirty)
    base = S.PILOT_STREAM if a.pilot_stream else S.STAGE1_STREAM
    scfg = dataclasses.replace(base, num_new_tasks=a.num_new_tasks, block_updates=a.block_updates,
                               revisit_every=a.revisit_every)
    os.makedirs(a.out_dir, exist_ok=True)
    trials = []
    for i, c in enumerate(coefs):
        config, kw, rec = arms.prepare(ppo.Config(**ppo.resolve_presets(a.preset)), scfg, seeds,
                                       "kickstart", a.size, a.critic_warmup_updates, c, a.bc_dir,
                                       a.allow_dirty)
        t0 = time.perf_counter()
        try:
            res = lifelong.run_condition(config, scfg, seeds, verbose=True,
                                         allow_official_seeds=a.allow_official_seeds,
                                         allow_dirty=a.allow_dirty, **kw)
        except lifelong.KickstartWindowError as e:
            raise SystemExit(f"trial {i} (coef {c}): {e}")
        res.update(rec)
        binding, why = arms.sweep_binding(seeds, res["fresh_seeds"])
        res.update(dict(ranking_binding=binding, ranking_binding_reason=why))
        res.update(dict(wall_seconds=time.perf_counter() - t0, config=json.dumps(ppo.config_record(config)),
                        stream_config=json.dumps(S.config_record(scfg)), sweep_trial=i,
                        sweep_coefs=np.array(coefs)))
        res.update(ppo.device_info())
        np.savez_compressed(os.path.join(a.out_dir, f"kickstart_c{i}.npz"), **res)
        m = SM.analyze(res)
        s = m["late_auc_rel_summary"]
        trials.append(dict(trial=i, coef=c, late_auc_rel=np.asarray(m["late_auc_rel"]).tolist(),
                           kickstart_effective_updates=np.asarray(res["kickstart_effective_updates"]).tolist(),
                           iqm=s["iqm"], mean=s["mean"], ci95=list(s["ci95"])))
        print(f"trial {i}: coef {c}: late AUC_rel IQM {s['iqm']:.4f}", flush=True)
    out = dict(seeds=seeds, fresh_seeds=[int(s) for s in res["fresh_seeds"]],
               ranking_binding=binding, ranking_binding_reason=why,
               coefs=coefs, trials=trials, selection_metric=SELECTION_METRIC,
               stream_config=S.config_record(scfg),
               **{k: v for k, v in prov.items() if k != "source_hashes"})
    if binding:  # dev / mechanics-only runs never carry a ranking
        ranked = sorted(trials, key=lambda t: -t["iqm"] if np.isfinite(t["iqm"]) else np.inf)
        out["ranked_best"] = ranked[0]["coef"]
    with open(os.path.join(a.out_dir, "sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("SWEEP", json.dumps({k: out.get(k) for k in ("coefs", "ranking_binding", "ranked_best")}))
    return out


if __name__ == "__main__":
    main()
