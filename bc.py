"""Behaviour-cloning (BC) pretraining for the stage-1 pretrained arms (review B-B4).

Pipeline (tars-test-rules sec. 7):
  1. Demo tasks. `demo_split` draws fully random tasks with `stream.make_demo_pool`, which
     must be given the run's actual streams: it refuses streams that are not covered by
     the overlap-checked reference set, and the pool shares no walls, (walls, goal), layout,
     obs permutation or task with any reference stream. Passing no streams is refused here.
     The pool is split into train / validation (early stopping) / retention tasks, with
     no wall layout shared between the three splits (so no full task either).
  2. Demos. `expert.generate_demos`: BFS planner with a uniformly random action with prob
     epsilon = 0.15 and planner recovery from wherever the agent ends up; the BC target is
     the planner's action at the visited state. epsilon = 0 is the noiseless ablation.
  3. BC. Cross-entropy of the actor logits against the planner labels. Only trunk + actor
     head get gradients (the loss does not touch critic_out, so Adam leaves it exactly at
     its init). Adam, shuffled minibatches; after every epoch the validation loss on the
     held-out demo TASKS (never stream tasks) is measured, and the params with the lowest
     validation loss are kept; training stops after `patience` epochs without improvement
     (patience fixed once, here). One fixed demo set for all sizes: the pool depends only on
     the stream config, demo_seed and the reference set, not on the network.
     BC accuracy, validation loss and policy entropy are logged per epoch and per size.
     BC per seed (review fix 1): BC for stream seed s starts from that seed's own
     scratch init (lifelong.scratch_init_params, the scratch arm's key path), so the
     pretrained and scratch arms of a seed differ only by BC; zero epochs == scratch(s).
  4. Checkpoint per (network size, seed) (`checkpoint_name`), msgpack params + JSON
     metadata incl. git commit and source-file hashes (provenance.py).
  5. Handover. `pretrained_config` turns on the critic warm-up (value head trained with
     trunk and policy frozen, inside task 1's K-update budget; lifelong.make_stream_train
     refuses a warm-up longer than K) and keeps adam_reset_at_handover=True: RL starts from
     a fresh Adam state, the BC optimiser state is never passed on.
  6. Retention. lifelong.run_condition(..., retention_tasks=...) evaluates the stream agent
     on the 5 held-out retention demo tasks at the start and at every task switch.

Values not fixed by the rules are flags with the defaults in BCConfig (see its comments).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

import expert
import ppo
import stream as S


@dataclasses.dataclass(frozen=True)
class BCConfig:
    epsilon: float = 0.15          # rules sec. 7: random action 15% of the time, planner recovery
    num_train_tasks: int = 200     # not fixed by the rules (flag)
    num_val_tasks: int = 20        # held-out demo tasks for early stopping (flag)
    num_retention_tasks: int = 5   # rules sec. 7: 5 held-out demo tasks
    episodes_per_task: int = 8     # not fixed by the rules (flag)
    max_epochs: int = 200          # cap; early stopping normally ends training first (flag)
    patience: int = 10             # rules: "patience fixed once" -> fixed here (flag)
    batch_size: int = 256
    lr: float = 3e-4
    demo_seed: int = 1             # make_demo_pool RNG (with the stream config's stream_seed)
    data_seed: int = 0             # demo rollouts (expert noise, env resets)
    init_seed: int = 0             # init_params() only; pretrain() uses the seed's scratch init
    shuffle_seed: int = 0


# ----------------------------------------------------------------------------- demo tasks
def demo_split(scfg: S.StreamConfig, streams, bccfg: BCConfig = BCConfig()):
    """(train, val, retention) lists of host-side task dicts for BC on `streams` (the run's
    actual streams; required). No wall layout is shared between the splits."""
    if streams is None or len(streams) == 0:
        raise ValueError("demo_split needs the run's actual streams (stream.make_stream for "
                         "the seeds being run), so make_demo_pool can refuse uncovered ones")
    n = bccfg.num_train_tasks + bccfg.num_val_tasks + bccfg.num_retention_tasks
    pool, _ = S.make_demo_pool(scfg, 2 * n, streams=streams, demo_seed=bccfg.demo_seed)
    chosen, walls = [], set()
    for t in pool:  # deterministic: the first n tasks with a new wall layout
        if S.walls_key(t) not in walls:
            walls.add(S.walls_key(t)); chosen.append(t)
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} wall-distinct demo tasks in a pool of {2 * n}")
    a, b = bccfg.num_train_tasks, bccfg.num_train_tasks + bccfg.num_val_tasks
    return chosen[:a], chosen[a:b], chosen[b:]


def tasks_hash(tasks):
    return hashlib.sha1("".join(S.task_key(t) for t in tasks).encode()).hexdigest()


def make_demos(tasks, scfg, bccfg: BCConfig, key):
    stacked = S.stack_tasks(tasks, scfg.max_steps)
    return expert.generate_demos(stacked, bccfg.episodes_per_task, bccfg.epsilon, key,
                                 max_steps=scfg.max_steps, size=scfg.size)


# ----------------------------------------------------------------------------- BC training
def bc_stats(network, params, obs, labels):
    logits, _, _ = network.apply(params, obs)
    logp = jax.nn.log_softmax(logits, -1)
    loss = -jnp.mean(jnp.take_along_axis(logp, labels[:, None], -1))
    acc = jnp.mean((jnp.argmax(logits, -1) == labels).astype(jnp.float32))
    ent = jnp.mean(ppo.categorical_entropy(logits))
    return loss, acc, ent


def init_params(config: ppo.Config, bccfg: BCConfig = BCConfig()):
    _, _, network, obs_shape = ppo.make_env_and_network(config)
    return network.init(jax.random.PRNGKey(bccfg.init_seed), jnp.zeros(obs_shape))


def train_bc(config: ppo.Config, train_demos, val_demos, bccfg: BCConfig = BCConfig(), verbose=False,
             init=None, until_train_acc=None):
    """BC with early stopping on the validation (held-out demo task) loss. Returns
    (best_params, info). Starts from `init` (default: init_params(config, bccfg)).
    max_epochs = 0 returns the init params unchanged.
    until_train_acc (diagnostic only, default off): ignore patience and stop at the first
    epoch whose training-demo accuracy is >= the value (or at max_epochs); used to measure
    the fixed-length fallback. Same RNG path, so its history prefix equals the default run's."""
    _, _, network, _ = ppo.make_env_and_network(config)
    params = init_params(config, bccfg) if init is None else init
    tx = optax.adam(bccfg.lr)
    opt = tx.init(params)
    X, Y = jnp.asarray(train_demos["obs"]), jnp.asarray(train_demos["label"], jnp.int32)
    Xv, Yv = jnp.asarray(val_demos["obs"]), jnp.asarray(val_demos["label"], jnp.int32)
    bs = min(bccfg.batch_size, int(X.shape[0]))
    nb = int(X.shape[0]) // bs

    def loss_fn(p, x, y):
        return bc_stats(network, p, x, y)[0]

    @jax.jit
    def epoch(params, opt, perm):
        def step(carry, idx):
            p, o = carry
            l, g = jax.value_and_grad(loss_fn)(p, X[idx], Y[idx])
            u, o = tx.update(g, o, p)
            return (optax.apply_updates(p, u), o), l
        (params, opt), losses = jax.lax.scan(step, (params, opt), perm.reshape(nb, bs))
        return params, opt, losses.mean()

    stats = jax.jit(lambda p, x, y: bc_stats(network, p, x, y))
    v0 = [float(x) for x in stats(params, Xv, Yv)]
    hist = dict(epoch=[0], train_loss=[np.nan], val_loss=[v0[0]], val_acc=[v0[1]], val_entropy=[v0[2]],
                train_acc=[float(stats(params, X, Y)[1])])
    best, best_loss, best_epoch, since = params, v0[0], 0, 0
    rng = np.random.default_rng(bccfg.shuffle_seed)
    for e in range(1, bccfg.max_epochs + 1):
        perm = jnp.asarray(rng.permutation(int(X.shape[0]))[:nb * bs])
        params, opt, tl = epoch(params, opt, perm)
        vl, va, ve = [float(x) for x in stats(params, Xv, Yv)]
        hist["epoch"].append(e); hist["train_loss"].append(float(tl)); hist["val_loss"].append(vl)
        hist["val_acc"].append(va); hist["val_entropy"].append(ve)
        hist["train_acc"].append(float(stats(params, X, Y)[1]))
        if verbose:
            print(f"bc epoch {e}: train {float(tl):.4f} val {vl:.4f} acc {va:.3f} ent {ve:.3f}", flush=True)
        if vl < best_loss:
            best, best_loss, best_epoch, since = params, vl, e, 0
        else:
            since += 1
            if until_train_acc is None and since >= bccfg.patience:
                break
        if until_train_acc is not None and hist["train_acc"][-1] >= until_train_acc:
            break
    bl, ba, be = [float(x) for x in stats(best, Xv, Yv)]
    info = dict(best_epoch=best_epoch, epochs_run=hist["epoch"][-1], best_val_loss=bl, best_val_acc=ba,
                best_val_entropy=be, best_train_acc=float(stats(best, X, Y)[1]),
                num_train_samples=int(X.shape[0]), num_val_samples=int(Xv.shape[0]), history=hist)
    return best, info


# ----------------------------------------------------------------------------- handover
def pretrained_config(config: ppo.Config, warmup_updates: int) -> ppo.Config:
    """Stream config of a pretrained arm: critic warm-up for the first `warmup_updates`
    updates (inside task 1's K; lifelong refuses more), Adam reset at the BC->RL handover."""
    return dataclasses.replace(config, critic_warmup_updates=int(warmup_updates),
                               adam_reset_at_handover=True)


# ----------------------------------------------------------------------------- checkpoints
def checkpoint_name(config: ppo.Config, seed: int) -> str:
    """One BC checkpoint per (network size / architecture, stream seed)."""
    tag = f"h{config.hidden_size}_l{config.num_layers}"
    tag += "_shared" if config.shared_trunk else "_towers"
    tag += "_ln" if config.layer_norm else ""
    return f"bc_{tag}_s{int(seed)}.msgpack"


def save_checkpoint(path, params, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_checkpoint(path, config: ppo.Config):
    template = init_params(config)
    with open(path, "rb") as f:
        params = serialization.from_bytes(template, f.read())
    with open(path + ".json") as f:
        meta = json.load(f)
    return params, meta


def pretrain(config: ppo.Config, scfg: S.StreamConfig, streams, seed: int, bccfg: BCConfig = BCConfig(),
             verbose=False, demos=None):
    """Full BC pipeline for stream seed `seed` (one of the run's streams), starting from
    lifelong.scratch_init_params(config, seed). The demo set is the same for every seed
    and size (it depends on the run's streams, the stream config and bccfg only); pass
    demos=demo_data(...) to reuse it. Returns (params, meta, retention_tasks)."""
    import lifelong
    train_t, val_t, ret_t, dtr, dva = demo_data(scfg, streams, bccfg) if demos is None else demos
    if int(seed) not in [int(s.seed) for s in streams]:
        raise ValueError(f"seed {seed} is not one of the run's stream seeds")
    init = lifelong.scratch_init_params(config, int(seed))
    params, info = train_bc(config, dtr, dva, bccfg, verbose, init=init)
    meta = dict(info, bc_config=dataclasses.asdict(bccfg), config=dataclasses.asdict(config),
                seed=int(seed), init="lifelong.scratch_init_params(config, seed)",
                stream_config=S.config_record(scfg), stream_seeds=[int(s.seed) for s in streams],
                train_tasks_hash=tasks_hash(train_t), val_tasks_hash=tasks_hash(val_t),
                retention_tasks_hash=tasks_hash(ret_t),
                demo_success=float(np.mean(dtr["episode_return"] > 0)),
                param_count=int(sum(x.size for x in jax.tree.leaves(params))))
    return params, meta, ret_t


def demo_data(scfg: S.StreamConfig, streams, bccfg: BCConfig = BCConfig()):
    """(train_tasks, val_tasks, retention_tasks, train_demos, val_demos), seed-independent."""
    train_t, val_t, ret_t = demo_split(scfg, streams, bccfg)
    k_train, k_val = jax.random.split(jax.random.PRNGKey(bccfg.data_seed))
    return train_t, val_t, ret_t, make_demos(train_t, scfg, bccfg, k_train), make_demos(val_t, scfg, bccfg, k_val)


def main():
    import lifelong
    p = argparse.ArgumentParser(description="BC pretraining for the stage-1 pretrained arms")
    p.add_argument("--seeds", required=True, help="stream seeds of the run the checkpoint is for")
    p.add_argument("--allow-official-seeds", action="store_true")
    p.add_argument("--preset", default="lifelong_stream")
    p.add_argument("--hidden-size", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--num-new-tasks", type=int, default=S.STAGE1_STREAM.num_new_tasks)
    p.add_argument("--stream", choices=("stage1", "pilot"), default="stage1",
                   help="stream config of the run: stage1 = shared layout on (default)")
    p.add_argument("--allow-dirty", action="store_true", help="dev seeds only; recorded")
    p.add_argument("--out-dir", default="checkpoints")
    for f in dataclasses.fields(BCConfig):
        p.add_argument("--" + f.name.replace("_", "-"), type=type(f.default), default=f.default)
    a = p.parse_args()
    seeds = ppo.parse_seeds(a.seeds)
    lifelong.check_seeds(seeds, a.allow_official_seeds)
    import provenance
    prov = provenance.check_clean(seeds, a.allow_dirty)
    overrides = ppo.resolve_presets(a.preset)
    for k in ("hidden_size", "num_layers"):
        if getattr(a, k):
            overrides[k] = getattr(a, k)
    if a.layer_norm:
        overrides["layer_norm"] = True
    config = ppo.Config(**overrides)
    base = S.STAGE1_STREAM if a.stream == "stage1" else S.PILOT_STREAM
    scfg = dataclasses.replace(base, num_new_tasks=a.num_new_tasks)
    bccfg = BCConfig(**{f.name: getattr(a, f.name) for f in dataclasses.fields(BCConfig)})
    streams = [S.make_stream(scfg, s) for s in seeds]
    demos = demo_data(scfg, streams, bccfg)
    for s in seeds:
        params, meta, _ = pretrain(config, scfg, streams, s, bccfg, verbose=True, demos=demos)
        meta["git_commit"] = lifelong.git_commit()
        meta.update(prov)
        meta.update(ppo.device_info())
        path = os.path.join(a.out_dir, checkpoint_name(config, s))
        save_checkpoint(path, params, meta)
        print(f"saved {path}: best epoch {meta['best_epoch']}, val loss {meta['best_val_loss']:.4f}, "
              f"val acc {meta['best_val_acc']:.3f}, entropy {meta['best_val_entropy']:.3f}")


if __name__ == "__main__":
    main()
