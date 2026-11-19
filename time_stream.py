"""Timing-only dev run: compile vs run time, s/update, peak RAM for stream + fresh agents.

    python time_stream.py --seeds 930 --hidden-size 905 --num-new-tasks 3 --block-updates 100
Dev seeds only (seed guard as in lifelong.py). Results are timing, never metrics.
"""
import argparse
import json
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

import lifelong
import ppo
import stream as S


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # Linux: KB


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", required=True)
    p.add_argument("--hidden-size", type=int, default=905)
    p.add_argument("--num-new-tasks", type=int, default=3)
    p.add_argument("--block-updates", type=int, default=100)
    p.add_argument("--revisit-every", type=int, default=3)
    p.add_argument("--fresh-tasks", type=int, default=None, help="time only this many fresh tasks")
    p.add_argument("--out")
    a = p.parse_args()
    seeds = ppo.parse_seeds(a.seeds)
    fresh = lifelong.check_seeds(seeds)
    config = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], hidden_size=a.hidden_size))
    scfg = S.StreamConfig(num_new_tasks=a.num_new_tasks, block_updates=a.block_updates,
                          revisit_every=a.revisit_every)
    streams = [S.make_stream(scfg, s) for s in seeds]
    B, T, K = streams[0].num_blocks, streams[0].num_tasks, scfg.block_updates
    tasks = S.stack_streams(streams)
    sched = jnp.asarray(np.stack([s.schedule for s in streams]))
    probe = jnp.stack([S.probe_batch(s) for s in streams])
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])
    res = dict(seeds=seeds, fresh_seeds=fresh, hidden_size=a.hidden_size,
               param_count=lifelong.param_count(config), blocks=B, tasks=T, K=K,
               batch_size=config.batch_size, platform=jax.devices()[0].platform,
               git_commit=lifelong.git_commit(), **ppo.device_info())
    print(json.dumps(res), flush=True)

    fn = jax.jit(jax.vmap(lifelong.make_stream_train(config, scfg, B)))
    t0 = time.perf_counter(); c = fn.lower(keys, tasks, sched, probe).compile()
    res["stream_compile_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); out = jax.block_until_ready(c(keys, tasks, sched, probe))
    res["stream_run_s"] = time.perf_counter() - t0
    res["stream_s_per_update"] = res["stream_run_s"] / (B * K)  # all seeds together
    res["rss_after_stream_gb"] = rss_gb()
    del out
    print(json.dumps(res), flush=True)

    ffn = jax.jit(jax.vmap(lifelong.make_stream_train(config, scfg, 1)))
    fk = lifelong.fresh_keys(seeds, range(T))
    z = jnp.zeros((len(seeds), 1), jnp.int32)
    tk0 = jax.tree.map(lambda x: x[:, 0:1], tasks)
    t0 = time.perf_counter(); fc = ffn.lower(fk[:, 0], tk0, z, probe).compile()
    res["fresh_compile_s"] = time.perf_counter() - t0
    nf = a.fresh_tasks or T
    t0 = time.perf_counter()
    for t in range(nf):
        tk = jax.tree.map(lambda x: x[:, t:t + 1], tasks)
        jax.block_until_ready(fc(fk[:, t], tk, z, probe))
    res["fresh_run_s"] = time.perf_counter() - t0
    res["fresh_tasks_timed"] = nf
    res["fresh_s_per_update"] = res["fresh_run_s"] / (nf * K)
    res["peak_rss_gb"] = rss_gb()
    print(json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
