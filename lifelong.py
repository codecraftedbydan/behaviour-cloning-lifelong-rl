"""Lifelong task-stream training on the procedural gridworld (milestone: plasticity loss).

One jitted program per condition. vmap is over seeds; each seed has its own task chain
and schedule (stream.py), passed in as arrays. The stream loop is a lax.scan over blocks:

  block b (task index = schedule[b], 0-based internally; written out as task number +1):
    switch:  index the task pytree, start new episodes in every env (the only switch
             event; the agent gets NO signal: heads, Adam, value-norm all carry over).
             Log policy entropy on the new task's start obs.
             Ablation: reset_adam_on_task_switch=True resets Adam moments here.
    train:   K = block_updates PPO updates (ppo.build_update_step, same code as make_train).
    eval:    greedy eval on ALL tasks of this seed's chain (unseen ones are masked
             in the analysis) -> forgetting / revisits.
    probes:  health on a fixed probe batch (probe obs from the seed's tasks): dormant
             fraction per layer, weight norms, srank / effective rank per hidden layer,
             probe entropy, random-target fit loss (trainability probe).

Fresh baselines (AUC_rel, tars-test-rules sec. 2): for every distinct task of every
seed's chain, a freshly initialised agent (same config) trains on that task alone for
the same K updates. Fresh agents use a dedicated seed block, paired one-to-one with the
stream seeds (see fresh_seed): official stream seeds 500-509 -> fresh 600-609 (+100),
stage-1 700-724 -> 800-824 and stage-1 sweep 750-759 -> 850-859 (+100);
dev stream seeds 9xx -> 19xx (+1000), disjoint from every official block. The fresh agent
for task number n (1-based) of stream seed s uses key fold_in(PRNGKey(fresh_seed(s)), n).
Revisits reuse the same fresh run (stream_metrics.py).

Task numbering: task ids are 0-based array indices internally; everything written out
(npz keys/values, logs, plots) counts from 1, so "task 1" is the first task as in the rules.
K is in PPO updates (batch_size env steps each); outputs record both.

Usage (dev seeds 900+ only until the official pilot is approved):
    python lifelong.py --seeds 900-901 --num-new-tasks 6 --block-updates 30 \\
        --out results/stream_dev/dev.npz
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import ppo
import pqn
import provenance
import stream as S

# Seed blocks live in seeds.py (single source of truth); re-exported here by their old names.
import seeds as _seeds  # noqa: E402
from seeds import (DEV_MAX_SEED, DEV_MIN_SEED, FRESH_OFFSET, OFFICIAL_FRESH_SEEDS,  # noqa: E402,F401
                   OFFICIAL_STREAM_SEEDS, PILOT_STREAM_SEEDS, RESERVED_SEEDS, SEED_BLOCKS,
                   STAGE1_STREAM_SEEDS, SWEEP_STREAM_SEEDS, fresh_seed, seed_block)

_BLOCKS_TEXT = _seeds.BLOCKS_TEXT


def under_pytest():
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def check_seeds(seeds, allow_official=False):
    """Seed guard. Dev runs: every stream seed >= 900 and no derived fresh seed in a
    reserved block. Seeds < 900 (incl. the reserved stream blocks 500-509, 700-724, 750-759
    and their fresh blocks 600-609, 800-824, 850-859) need allow_official=True, and then only
    reserved stream seeds are accepted, from ONE reserved block per call (e.g. sweep seeds
    750-759 can never share a job with test seeds 700-724), and a call with any reserved
    seed may contain nothing else (no dev seeds). Dev-only lists are still accepted.
    Second guard: under pytest (PYTEST_CURRENT_TEST set) official/reserved seeds are refused
    even with allow_official=True, so no test can ever train on them."""
    if under_pytest():
        hit = sorted({int(s) for s in seeds if int(s) in RESERVED_SEEDS or
                      (int(s) in OFFICIAL_STREAM_SEEDS and fresh_seed(int(s)) in RESERVED_SEEDS)})
        if hit:
            raise SystemExit(f"seeds {hit} are official/reserved: refused under pytest "
                             "(PYTEST_CURRENT_TEST is set), whatever the flags")
    big = [s for s in seeds if s >= DEV_MAX_SEED]
    if big:  # dev fresh seeds are s + 1000, so stream seeds >= 1900 could collide with them
        raise SystemExit(f"seeds {big}: dev stream seeds must be in [900, 1900)")
    fresh = [fresh_seed(s) if (s in OFFICIAL_STREAM_SEEDS or s >= DEV_MIN_SEED) else None for s in seeds]
    if not allow_official:
        bad = [s for s in seeds if s < DEV_MIN_SEED or s in RESERVED_SEEDS]
        bad += [f for f in fresh if f is not None and f in RESERVED_SEEDS]
        if bad:
            raise SystemExit(f"seeds {sorted(set(bad))} are reserved for official runs "
                             f"({_BLOCKS_TEXT}); dev runs use 900+")
    else:
        bad = [s for s in seeds if s not in OFFICIAL_STREAM_SEEDS and s < DEV_MIN_SEED]
        if bad:
            raise SystemExit(f"seeds {bad}: not an official stream seed and not a dev seed")
        blocks = sorted({seed_block(s) for s in seeds if s in OFFICIAL_STREAM_SEEDS})
        if len(blocks) > 1:
            raise SystemExit(f"seeds mix reserved blocks {blocks}: one reserved block per run")
        if blocks:  # a reserved run contains only that block's seeds (no dev seeds mixed in)
            other = [s for s in seeds if s not in OFFICIAL_STREAM_SEEDS]
            if other:
                raise SystemExit(f"seeds {other}: a run on reserved block {blocks[0]} cannot "
                                 "include non-reserved seeds; one reserved block per run, nothing else")
    return [fresh_seed(s) for s in seeds]


# ----------------------------------------------------------------------------- probes
def srank(features, delta=0.01):
    """Kumar et al. (2021) srank: smallest k with sum_{i<=k} s_i >= (1-delta) sum s_i."""
    s = jnp.linalg.svd(features, compute_uv=False)
    c = jnp.cumsum(s) / jnp.maximum(jnp.sum(s), 1e-12)
    return jnp.sum(c < 1.0 - delta) + 1


def effective_rank(features):
    """Roy & Vetterli (2007): exp(entropy of normalised singular values)."""
    s = jnp.linalg.svd(features, compute_uv=False)
    p = s / jnp.maximum(jnp.sum(s), 1e-12)
    return jnp.exp(-jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-12)), 0.0)))


def random_target_fit(network, params, obs, target, steps=100, lr=1e-3):
    """Trainability probe (Lyle et al. 2022 style): fit the value head of a COPY of the
    current params to a fixed random target for `steps` full-batch Adam steps; return
    the final MSE (target has unit variance, so 1.0 = learned nothing)."""
    tx = optax.adam(lr)

    def loss(p):
        _, v, _ = network.apply(p, obs)
        return jnp.mean((v - target) ** 2)

    def step(carry, _):
        p, o = carry
        g = jax.grad(loss)(p)
        u, o = tx.update(g, o, p)
        return (optax.apply_updates(p, u), o), None

    (p, _), _ = jax.lax.scan(step, (params, tx.init(params)), None, length=steps)
    return loss(p)


def probe_metrics(network, config, params, obs, rt_target):
    out = ppo.network_health(params, network.apply, obs, config.dormant_tau, config.activation,
                             ppo.lora_scale_of(config))
    logits, _, hiddens = network.apply(params, obs)
    for name, h in hiddens.items():
        out[f"srank.{name}"] = srank(h)
        out[f"erank.{name}"] = effective_rank(h)
    out["probe_entropy"] = jnp.mean(ppo.categorical_entropy(logits))
    out["random_target_mse"] = random_target_fit(network, params, obs, rt_target)
    return out


# ----------------------------------------------------------------------------- reset arms
# Stage-1 reset arms (stage1-proposal sec. 2, rules sec. 7, cells 17-18). Scratch only, and the
# only configs that get the oracle task-switch signal: a reset at EVERY block boundary from
# block 2 on (0-based b >= 1), revisits included. Default "none" = no switch signal at all.
RESET_ARMS = ("none", "heads", "heads_adam_vn")
RESET_TAG = 3_000_000   # head key at block b: fold_in(run key, RESET_TAG + b)


def reset_key(run_key, b):
    """Fresh-head key for the switch into 0-based block b, folded off the run key (the
    `key` given to train()). Not split from the training RNG stream, so the training,
    env-reset and eval keys are the same as without the arm; and not the original init
    key (that is split(run key, 3)[1]), so the new heads are a fresh draw."""
    return jax.random.fold_in(run_key, RESET_TAG + b)


def probe_reset(network, tx, params, key, obs_shape):
    """What the head-reset probe resets (run_condition, head_probe_every): actor_out and
    critic_out re-initialised from `key`, a fresh optimiser state (tx.init: Adam mu, nu and
    count all at init) and fresh value-normalisation statistics. Trunk params unchanged.
    The probe and R-HOV both call this function."""
    new = ppo.reinit_heads(network, params, key, obs_shape)
    return new, tx.init(new), ppo.ValueNorm.init()


def switch_reset(arm, network, tx, obs_shape, train_state, vnorm, key):
    """State after a task-switch reset, unconditionally (the caller selects b >= 1).
    heads (R-H): actor_out + critic_out re-initialised from `key`; trunk params, the WHOLE
      optimiser state (including the Adam moments mu/nu of the re-initialised head params
      and the step count) and the value-norm statistics carry over exactly, bit for bit.
    heads_adam_vn (R-HOV): probe_reset: heads re-initialised as above, plus a fresh
      optimiser state and fresh value-norm statistics. Trunk params carry over.
    The TrainState step (lr schedule position) is unchanged in both arms."""
    if arm == "heads":
        return train_state.replace(params=ppo.reinit_heads(network, train_state.params, key, obs_shape)), vnorm
    if arm == "heads_adam_vn":
        p, o, v = probe_reset(network, tx, train_state.params, key, obs_shape)
        return train_state.replace(params=p, opt_state=o), v
    raise ValueError(f"reset arm {arm!r} not in {RESET_ARMS}")


def scratch_init_params(config: ppo.Config, seed: int):
    """The scratch arm's initial params for stream seed `seed`: exactly the key path of
    make_stream_train (k_init = split(PRNGKey(seed), 3)[1]). BC for seed s starts here
    (review fix 1), so the pretrained and scratch arms of a seed share their init."""
    _, _, network, obs_shape = ppo.make_env_and_network(config)
    k_init = jax.random.split(jax.random.PRNGKey(seed), 3)[1]
    return network.init(k_init, jnp.zeros(obs_shape))


def check_reset_arm(arm, config, init_params=None, teacher_params=None):
    if arm not in RESET_ARMS:
        raise ValueError(f"reset arm {arm!r} not in {RESET_ARMS}")
    if arm == "none":
        return
    if init_params is not None or teacher_params is not None or config.critic_warmup_updates \
            or config.kickstart_coef or config.freeze_trunk or config.lora_rank:
        raise ValueError("reset arms are scratch-only cells (stage-1 proposal sec. 2): no "
                         "pretrained init, teacher, warm-up, kickstarting, frozen trunk or LoRA")
    if config.reset_adam_on_task_switch:
        raise ValueError("reset arms define their own optimiser handling at switches; "
                         "do not combine with reset_adam_on_task_switch")


# ----------------------------------------------------------------------------- trainer
def make_stream_train(config: ppo.Config, scfg: S.StreamConfig, num_blocks: int,
                      init_params=None, teacher_params=None, init_opt_state=None,
                      snapshot_blocks=(), retention_tasks=None, reset_arm="none"):
    """train(key, tasks, schedule, probe_obs, start_params=None, teacher=None,
    kickstart_gate=None) -> outputs.
    kickstart_gate: (num_blocks,) 0/1 per block, REQUIRED when config.kickstart_coef > 0
    (kickstart_gate_blocks): the effective coefficient in block b is kickstart_coef *
    gate[b]; outputs gain kickstart_coef_eff (B,).
    teacher: optional *traced* per-seed kickstarting teacher (e.g. that seed's BC policy);
    overrides teacher_params. With config.lora_rank > 0 and plain (adapter-free) initial
    params, the adapters are added from the LoRA network's own init at k_init (B = 0, so
    the function is unchanged) and every non-adapter leaf comes from the given params.
    tasks: stacked TaskParams (T, ...) for one seed; schedule: (num_blocks,) int32.
    start_params: optional *traced* initial params (per seed, e.g. head-reset probe);
    init_params: optional constant initial params (e.g. BC-pretrained), same for all seeds.
    snapshot_blocks: static 0-based block indices; the params at the START of each such
    block (right at the switch, before any training on it) are returned in "snapshots".
    retention_tasks: optional stacked TaskParams (BC retention: held-out demo tasks). If
    given, the agent is evaluated on them before block 0 ("retention0") and at the end of
    every block, i.e. at every task switch ("retention"), with keys folded off the eval key
    (the training RNG stream is unchanged). Off (None) by default.
    reset_arm: "none" (default: no switch signal), "heads" (R-H) or "heads_adam_vn" (R-HOV);
    see switch_reset. The reset is applied at the start of every block b >= 1 (revisits
    included), after the switch-entropy measurement and the head-probe snapshot (both
    therefore see the agent as it arrives at the switch; the probe re-inits the heads
    itself, so its start is the same either way). With an arm on, the outputs gain
    reset_applied (B,) and post_reset_entropy (B,); with "none" they are unchanged."""
    env, _, network, obs_shape = ppo.make_env_and_network(config)
    assert not config.task_switch_updates, "stream mode switches at block boundaries"
    K = scfg.block_updates
    if config.critic_warmup_updates > K:  # rules sec. 7: warm-up counts inside task 1's K
        raise ValueError(f"critic_warmup_updates={config.critic_warmup_updates} > K={K}: the "
                         "warm-up must fit inside task 1's update budget")
    RET_TAG = 7_000_001
    check_reset_arm(reset_arm, config, init_params, teacher_params)
    tx, lr_schedule = ppo.make_optimizer(config, num_blocks * K)
    build = pqn.build_update_step if config.algo == "pqn" else ppo.build_update_step
    update_step = build(env, network, config, tx, lr_schedule, teacher_params)
    take = lambda tree, i: jax.tree.map(lambda x: x[i], tree)  # noqa: E731

    # Fixed random target for the trainability probe (same for every seed/condition).
    rt_net_params = network.init(jax.random.PRNGKey(12345), jnp.zeros(obs_shape))

    snap_idx = np.full(max(num_blocks, 1), -1, np.int32)
    for i, b in enumerate(snapshot_blocks):
        snap_idx[b] = i
    snap_idx = jnp.asarray(snap_idx)

    def train(key, tasks, schedule, probe_obs, start_params=None, teacher=None, kickstart_gate=None):
        ppo.TRACE_COUNT["train"] += 1
        run_key = key
        key, k_init, k_reset = jax.random.split(key, 3)
        params = network.init(k_init, jnp.zeros(obs_shape)) if init_params is None else init_params
        if start_params is not None:
            params = start_params
        if config.lora_rank and not ppo.has_lora(params):
            params = ppo.graft_lora(network.init(k_init, jnp.zeros(obs_shape)), params)
        us = update_step if teacher is None else ppo.build_update_step(
            env, network, config, tx, lr_schedule, teacher)
        kick = config.kickstart_coef > 0
        if kick and kickstart_gate is None:
            raise ValueError("kickstart_coef > 0 in a stream run needs kickstart_gate "
                             "(lifelong.kickstart_gate_blocks: shared-layout window)")
        kteacher = teacher if teacher is not None else teacher_params
        ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        ts = ppo.handover_opt_state(config, ts, init_opt_state)
        _, rt, _ = network.apply(rt_net_params, probe_obs)
        rt = (rt - rt.mean()) / (rt.std() + 1e-8)

        env_params0 = take(tasks, schedule[0])
        obs, env_state = jax.vmap(ppo.env_reset, in_axes=(None, 0, None))(
            env, jax.random.split(k_reset, config.num_envs), env_params0)
        # Entropy over the fixed probe batch (512 states); the start-cell value is kept too.
        init_entropy = ppo.initial_entropy(network, params, probe_obs)
        init_entropy_start = ppo.initial_entropy(network, params, obs)
        key, k_eval = jax.random.split(key)
        eval0 = ppo.evaluate(env, network, config, params, tasks, k_eval, scfg.max_steps)
        probe0 = probe_metrics(network, config, params, probe_obs, rt)
        extra0 = {}
        if retention_tasks is not None:
            extra0["retention0"] = ppo.evaluate(env, network, config, params, retention_tasks,
                                                jax.random.fold_in(k_eval, RET_TAG), scfg.max_steps)
        runner = ppo.RunnerState(ts, env_state, obs, ppo.EpisodeStats.zeros(config.num_envs),
                                 key, jnp.zeros((), jnp.int32), ppo.ValueNorm.init())

        snaps0 = jax.tree.map(lambda x: jnp.zeros((len(snapshot_blocks),) + x.shape, x.dtype), params)

        def block(carry, inp):
            runner, snaps = carry
            b, tid = inp
            env_params = take(tasks, tid)
            key, k_reset = jax.random.split(runner.key)
            obs, env_state = jax.vmap(ppo.env_reset, in_axes=(None, 0, None))(
                env, jax.random.split(k_reset, config.num_envs), env_params)
            ts = runner.train_state
            switch_entropy = ppo.initial_entropy(network, ts.params, probe_obs)
            switch_entropy_start = ppo.initial_entropy(network, ts.params, obs)
            if snapshot_blocks:
                i = snap_idx[b]
                snaps = jax.tree.map(lambda S_, p: jnp.where(i >= 0, S_.at[jnp.maximum(i, 0)].set(p), S_),
                                     snaps, ts.params)
            if config.reset_adam_on_task_switch:  # ablation; never at block 0 (fresh anyway)
                reset = ppo.reset_adam_moments(ts.opt_state, tx.init(ts.params))
                ts = ts.replace(opt_state=jax.tree.map(
                    lambda r, o: jnp.where(b > 0, r, o), reset, ts.opt_state))
            vnorm = runner.value_norm
            if reset_arm != "none":  # oracle switch signal: every block boundary, b >= 1
                rts, rvn = switch_reset(reset_arm, network, tx, obs_shape, ts, vnorm,
                                        reset_key(run_key, b))
                sel = lambda r, o: jnp.where(b > 0, r, o)  # noqa: E731
                ts, vnorm = jax.tree.map(sel, rts, ts), jax.tree.map(sel, rvn, vnorm)
                reset_out = dict(reset_applied=(b > 0).astype(jnp.float32),
                                 post_reset_entropy=ppo.initial_entropy(network, ts.params, probe_obs))
            runner = runner._replace(train_state=ts, env_state=env_state, obs=obs, key=key,
                                     ep_stats=ppo.EpisodeStats.zeros(config.num_envs), value_norm=vnorm)
            # PQN (decision, 5 Nov): epsilon decays from the block's own first update, so every
            # block (new task or revisit) starts at eps_start. Not global-counter % K.
            if config.algo == "pqn":
                runner = runner._replace(update_idx=jnp.zeros((), jnp.int32))
            ub = us
            if kick:  # gated KL: coefficient kickstart_coef * gate[b] for this whole block
                ub = ppo.build_update_step(env, network, config, tx, lr_schedule, kteacher,
                                           kickstart_scale=kickstart_gate[b])
            runner, metrics = jax.lax.scan(lambda r, _: ub(r, env_params), runner,
                                           None, length=K)
            key, k_eval = jax.random.split(runner.key)
            runner = runner._replace(key=key)
            params = runner.train_state.params
            ev = ppo.evaluate(env, network, config, params, tasks, k_eval, scfg.max_steps)
            pr = probe_metrics(network, config, params, probe_obs, rt)
            bout = dict(metrics=metrics, eval=ev, probes=pr, switch_entropy=switch_entropy,
                        switch_entropy_start=switch_entropy_start)
            if reset_arm != "none":
                bout.update(reset_out)
            if kick:
                bout["kickstart_coef_eff"] = config.kickstart_coef * kickstart_gate[b].astype(jnp.float32)
            if retention_tasks is not None:
                bout["retention"] = ppo.evaluate(env, network, config, params, retention_tasks,
                                                 jax.random.fold_in(k_eval, RET_TAG), scfg.max_steps)
            return (runner, snaps), bout

        (runner, snaps), out = jax.lax.scan(block, (runner, snaps0), (jnp.arange(num_blocks), schedule))
        return dict(out, **extra0, eval0=eval0, probe0=probe0, init_entropy=init_entropy,
                    init_entropy_start=init_entropy_start,
                    final_params=runner.train_state.params, snapshots=snaps)

    return train


def kickstart_gate_blocks(stream):
    """(B,) float32 0/1 kickstarting gate (decision, 2 Nov): the KL term is on only while every
    block so far trained a task with BOTH task_shared_obs_layout and task_shared_action_perm
    true. From the first block whose task is not fully shared, the gate is 0 for the rest of
    the stream and never re-enables (also not on a later revisit of a shared task)."""
    both = np.asarray(stream.shared_obs, bool) & np.asarray(stream.shared_action, bool)
    return np.cumprod(both[np.asarray(stream.schedule)]).astype(np.float32)


def kickstart_effective_block_updates(stream, block_updates, warmup_updates):
    """(B,) int32: per block, the number of PPO updates whose effective kickstart coefficient
    is > 0 AFTER the critic warm-up (decision, 2 Nov 11:36). The update counter runs across the
    whole stream (block b holds global updates b*K .. b*K+K-1) and the warm-up covers global
    updates 0 .. W-1 (inside task 1, rules sec. 7), during which only the critic trains, so
    the KL cannot move the policy. A block with gate 0 contributes 0, so revisits after the
    first unshared task add nothing; a revisit while the gate is still on (every block so
    far fully shared) does add K, because the KL is applied there."""
    g = kickstart_gate_blocks(stream) > 0
    K, W = int(block_updates), int(warmup_updates)
    start = np.arange(len(g)) * K
    after_warm = np.clip(start + K - np.maximum(start, W), 0, K)
    return np.where(g, after_warm, 0).astype(np.int32)


class KickstartWindowError(ValueError):
    """A kickstart run with zero effective kickstart updates for some seed."""


def check_kickstart_window(seeds, block_counts):
    """Refuse a kickstart run in which any seed would have zero effective kickstart updates
    (decision, 2 Nov 11:36: such a run must never be counted quietly). block_counts: (S, B)."""
    tot = np.asarray(block_counts).sum(axis=1)
    bad = [int(s) for s, n in zip(seeds, tot) if n == 0]
    if bad:
        raise KickstartWindowError(
            f"kickstart run has 0 effective kickstart updates (coef > 0 after the critic "
            f"warm-up) for seed(s) {bad}: the shared-layout KL window is no longer than the "
            f"warm-up, so the teacher would never act. Refusing to run (decision, 2 Nov 11:36); "
            f"per-seed effective updates: {dict(zip(map(int, seeds), map(int, tot)))}")
    return tot


def kickstart_gate_tasks(stream):
    """(T,) gate of each task at its first (new-task) block."""
    g = kickstart_gate_blocks(stream)
    sched = np.asarray(stream.schedule)
    return np.array([g[int(np.flatnonzero(sched == t)[0])] for t in range(stream.num_tasks)], np.float32)


def fresh_keys(seeds, task_ids):
    """(S, T) keys; task_ids are 0-based indices, folded in as 1-based task numbers."""
    return jnp.stack([jnp.stack([jax.random.fold_in(jax.random.PRNGKey(fresh_seed(s)), int(t) + 1)
                                 for t in task_ids]) for s in seeds])


AUC_SOURCE = ("training-rollout return of the stochastic (sampling) policy: mean undiscounted "
              "return of the episodes that finished in each update's rollout, normalised by the "
              "task's noiseless BFS optimum (random = 0); NOT greedy eval")
HEAD_PROBE_TAG = 2_000_000


def scratch_config(config: ppo.Config) -> ppo.Config:
    """Fresh baselines are always scratch agents (rules sec. 2, review B-B3): random init,
    no critic warm-up, no kickstarting, no frozen trunk, whatever the arm uses."""
    return dataclasses.replace(config, critic_warmup_updates=0, kickstart_coef=0.0,
                               freeze_trunk=False, adam_reset_at_handover=True, lora_rank=0)


def head_probe_blocks(streams, every):
    """0-based block indices of new tasks number every, 2*every, ... (same for all seeds)."""
    if not every:
        return (), ()
    nums = list(range(every, streams[0].cfg.num_new_tasks + 1, every))
    blocks = []
    for n in nums:
        bs = {int(np.flatnonzero(s.is_new & (s.schedule == n - 1))[0]) for s in streams}
        assert len(bs) == 1, "block layout differs across seeds"
        blocks.append(bs.pop())
    return tuple(blocks), tuple(nums)


def run_condition(config, scfg, seeds, fresh=True, verbose=True, allow_official_seeds=False,
                  init_params=None, teacher_params=None, init_opt_state=None,
                  retention_tasks=None, reset_arm="none", seed_init_params=None,
                  seed_teacher_params=None, allow_dirty=False):
    """Stream runs for all seeds (+ fresh baselines, + head-reset probes if
    scfg.head_probe_every > 0). Returns a dict of numpy arrays.

    The seed guard runs here (not only in the CLI), so scripts/notebooks cannot bypass it;
    so does the clean-tree guard (provenance.check_clean(seeds, allow_dirty), review
    review of 0291582, fix 2), and its provenance record is stored in the result.
    init/teacher/opt-state hooks apply to the STREAM agent only; fresh agents always use
    scratch_config(config) with random init. retention_tasks (host-side task dicts, BC
    held-out demo tasks) are evaluated for the stream agent only, at every task switch
    (stream/retention, (S, B, R)) and before block 0 (stream/retention0).
    reset_arm: stage-1 reset arm for the STREAM agent only (fresh agents and head probes
    never get it); recorded as "reset_arm" when not "none".
    seed_init_params: optional list (one per seed, same order) of initial STREAM params,
    e.g. the per-seed BC checkpoints; passed as traced per-seed start params. Exclusive
    with init_params; recorded as "seed_init_params" = True.
    seed_teacher_params: optional list (one per seed) of kickstarting teachers (each seed's
    frozen BC policy, stage-1 cell 12); traced per seed. Exclusive with teacher_params;
    recorded as "seed_teacher_params" = True.
    LoRA / frozen trunk (config.lora_rank / freeze_trunk): the record gains
    trainable_param_count; the head-reset probe starts from the LoRA-merged trunk.
    Shared layout (scfg.shared_obs_layout / shared_action_perm): the record gains
    task_shared_obs_layout / task_shared_action_perm (S, T) bool and shared_layout_hash.

    Head-reset probe (plasticity vs exploration, review A-B1): at every head_probe_every-th
    new task, the stream agent's params at the switch are copied, actor_out and critic_out
    are re-initialised (entropy back to ~log 4), the trunk is frozen, and the copy trains K
    updates on that task (fresh Adam and value-norm). Key: fold_in(PRNGKey(seed),
    HEAD_PROBE_TAG + task number). If it matches the fresh agent while the stream agent
    does not, the deficit is in the heads/exploration, not in trunk plasticity.
    """
    check_seeds(seeds, allow_official_seeds)
    prov = provenance.check_clean(seeds, allow_dirty)
    if seed_init_params is not None:
        if init_params is not None:
            raise ValueError("pass init_params (same for all seeds) or seed_init_params, not both")
        if len(seed_init_params) != len(seeds):
            raise ValueError(f"seed_init_params: {len(seed_init_params)} entries for {len(seeds)} seeds")
    if seed_teacher_params is not None:
        if teacher_params is not None:
            raise ValueError("pass teacher_params (same for all seeds) or seed_teacher_params, not both")
        if len(seed_teacher_params) != len(seeds):
            raise ValueError(f"seed_teacher_params: {len(seed_teacher_params)} entries for {len(seeds)} seeds")
    check_reset_arm(reset_arm, config, init_params if seed_init_params is None else seed_init_params,
                    teacher_params if seed_teacher_params is None else seed_teacher_params)
    streams = [S.make_stream(scfg, s) for s in seeds]
    B, T = streams[0].num_blocks, streams[0].num_tasks
    tasks = S.stack_streams(streams)
    sched = jnp.asarray(np.stack([s.schedule for s in streams]))
    probe = jnp.stack([S.probe_batch(s) for s in streams])
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])

    dev = ppo.device_info()  # recorded separately for stream and fresh runs
    hp_blocks, hp_tasks = head_probe_blocks(streams, scfg.head_probe_every)
    ret = None if retention_tasks is None else S.stack_tasks(list(retention_tasks), scfg.max_steps)
    train = jax.jit(jax.vmap(make_stream_train(config, scfg, B, init_params, teacher_params,
                                               init_opt_state, snapshot_blocks=hp_blocks,
                                               retention_tasks=ret, reset_arm=reset_arm)))
    t0 = time.perf_counter()
    stack = lambda ps: None if ps is None else jax.tree.map(lambda *x: jnp.stack(x), *ps)  # noqa: E731
    gate = None
    if config.kickstart_coef > 0:
        gate = np.stack([kickstart_gate_blocks(s) for s in streams])
        # Pre-flight (before any training, so earlier than the end of task 1): refuse seeds
        # whose shared-layout window is used up by the critic warm-up.
        kick_eff_block = np.stack([kickstart_effective_block_updates(
            s, scfg.block_updates, config.critic_warmup_updates) for s in streams])
        kick_eff = check_kickstart_window(seeds, kick_eff_block)
    if seed_init_params is None and seed_teacher_params is None and gate is None:
        out = jax.block_until_ready(train(keys, tasks, sched, probe))
    else:
        out = jax.block_until_ready(train(keys, tasks, sched, probe, stack(seed_init_params),
                                          stack(seed_teacher_params),
                                          None if gate is None else jnp.asarray(gate)))
    stream_s = time.perf_counter() - t0
    if verbose:
        print(f"stream: {len(seeds)} seeds x {B} blocks x {scfg.block_updates} updates "
              f"in {stream_s:.1f}s (incl. compile)", flush=True)
    out.pop("final_params")
    snaps = out.pop("snapshots")
    res = {f"stream/{k}": v for k, v in _flatten(out).items()}
    res.update({f"stream/{k}": v for k, v in dev.items()})

    fresh_s = 0.0
    if fresh:
        fconfig = scratch_config(config)
        res["fresh_config"] = json.dumps(ppo.config_record(fconfig))
        ftrain = jax.jit(jax.vmap(make_stream_train(fconfig, scfg, 1)))
        fk = fresh_keys(seeds, range(T))
        outs = []
        t0 = time.perf_counter()
        for t in range(T):  # vmap over seeds, loop over tasks (memory); compiles once
            tk = jax.tree.map(lambda x: x[:, t:t + 1], tasks)
            o = ftrain(fk[:, t], tk, jnp.zeros((len(seeds), 1), jnp.int32), probe)
            o.pop("final_params")
            outs.append(jax.tree.map(np.asarray, o))
        fresh_s = time.perf_counter() - t0
        fo = jax.tree.map(lambda *x: np.stack(x, axis=1), *outs)  # (S, T, ...)
        res.update({f"fresh/{k}": v for k, v in _flatten(fo).items()})
        res.update({f"fresh/{k}": v for k, v in ppo.device_info().items()})
        if verbose:
            print(f"fresh: {len(seeds)} seeds x {T} tasks in {fresh_s:.1f}s", flush=True)

    hp_s = 0.0
    if hp_blocks:
        pconfig = dataclasses.replace(scratch_config(config), freeze_trunk=True)
        res["head_probe_config"] = json.dumps(ppo.config_record(pconfig))
        if config.lora_rank:  # probe network is plain: start from the merged trunk
            snaps = ppo.merge_lora(snaps, config)
        env, _, network, obs_shape = ppo.make_env_and_network(pconfig)
        ptrain = jax.jit(jax.vmap(make_stream_train(pconfig, scfg, 1)))
        # probe_reset is shared with R-HOV: heads from k_head here; the fresh Adam state and
        # value norm it also returns are exactly what ptrain builds at its start
        # (TrainState.create = tx.init, ValueNorm.init), tested in test_reset_arms.py.
        ptx, _ = ppo.make_optimizer(pconfig, scfg.block_updates)
        reinit = jax.jit(jax.vmap(lambda p, k: probe_reset(network, ptx, p, k, obs_shape)[0]))
        outs = []
        t0 = time.perf_counter()
        for i, n in enumerate(hp_tasks):
            k = jnp.stack([jax.random.fold_in(jax.random.PRNGKey(s), HEAD_PROBE_TAG + n) for s in seeds])
            k_head, k_run = jax.vmap(jax.random.split)(k).transpose(1, 0, 2)
            start = reinit(jax.tree.map(lambda x: x[:, i], snaps), k_head)
            tk = jax.tree.map(lambda x: x[:, n - 1:n], tasks)
            o = ptrain(k_run, tk, jnp.zeros((len(seeds), 1), jnp.int32), probe, start)
            o.pop("final_params"); o.pop("snapshots")
            outs.append(jax.tree.map(np.asarray, o))
        hp_s = time.perf_counter() - t0
        po = jax.tree.map(lambda *x: np.stack(x, axis=1), *outs)  # (S, P, ...)
        res.update({f"headprobe/{k}": v for k, v in _flatten(po).items()})
        res["head_probe_tasks"] = np.array(hp_tasks)                # 1-based task numbers
        res["head_probe_blocks"] = np.array(hp_blocks) + 1          # 1-based block numbers
        res.update({f"headprobe/{k}": v for k, v in ppo.device_info().items()})
        if verbose:
            print(f"head probe: {len(seeds)} seeds x {len(hp_tasks)} tasks in {hp_s:.1f}s", flush=True)
    del snaps
    if reset_arm != "none":
        res["reset_arm"] = reset_arm
    if seed_init_params is not None:
        res["seed_init_params"] = True
    if seed_teacher_params is not None:
        res["seed_teacher_params"] = True
    if gate is not None:  # per-block and per-task (first occurrence) effective coefficient
        res["kickstart_coef_block"] = (config.kickstart_coef * gate).astype(np.float32)
        res["kickstart_on_block"] = np.asarray(gate) > 0  # (S, B) bool on/off state per block
        res["kickstart_coef_task"] = np.stack([config.kickstart_coef * kickstart_gate_tasks(s)
                                               for s in streams]).astype(np.float32)
        # Logged per update in-graph (stream/metrics/kickstart_effective, 0/1) and checked
        # against the host-side count, so a mismatch cannot pass silently.
        logged = np.asarray(res["stream/metrics/kickstart_effective"]).reshape(len(seeds), B, -1)
        logged = np.rint(logged.sum(axis=2)).astype(np.int32)
        if not np.array_equal(logged, kick_eff_block):
            raise RuntimeError(f"kickstart_effective_updates: in-graph count {logged.sum(1).tolist()} "
                               f"!= expected {kick_eff.tolist()}")
        res["kickstart_effective_updates"] = kick_eff.astype(np.int32)            # (S,)
        res["kickstart_effective_updates_block"] = kick_eff_block                 # (S, B)
        res["kickstart_effective_updates_revisit"] = np.where(                    # (S,) on revisit blocks
            np.stack([s.is_new for s in streams]), 0, kick_eff_block).sum(1).astype(np.int32)
        res["kickstart_effective_rule"] = ("PPO updates with effective kickstart coefficient > 0 "
                                           f"after the {config.critic_warmup_updates}-update critic "
                                           "warm-up; a run with 0 for any seed is refused")
        if verbose:
            print("kickstart effective updates per seed:",
                  dict(zip(map(int, seeds), kick_eff.tolist())), flush=True)
        res["kickstart_gate_rule"] = ("KL on while every block so far trained a task with both "
                                      "shared obs layout and shared action perm; off for good "
                                      "from the first block that did not")
    if config.lora_rank or config.freeze_trunk:
        res["trainable_param_count"] = trainable_param_count(config)
    if scfg.shared_obs_layout or scfg.shared_action_perm:
        res["task_shared_obs_layout"] = np.stack([s.shared_obs for s in streams])
        res["task_shared_action_perm"] = np.stack([s.shared_action for s in streams])
        res["shared_layout_hash"] = _seeds.SHARED_LAYOUT_HASH
    if retention_tasks is not None:
        res["retention_optimal_return"] = np.array(
            [1.0 - 0.9 * S._bfs_np(t["walls"], tuple(t["goal"]))[tuple(t["start"])] / scfg.max_steps
             for t in retention_tasks], np.float32)
        res["retention_task_keys"] = np.array([S.task_key(t) for t in retention_tasks])

    res.update(prov)
    res.update({
        "seeds": np.array(seeds), "fresh_seeds": np.array([fresh_seed(s) for s in seeds]),
        # 1-based task number trained in each block (rules: task 1 = first task)
        "task_of_block": np.stack([s.schedule for s in streams]) + 1,
        "is_new": np.stack([s.is_new for s in streams]),
        "block_shift": np.stack([s.block_shift for s in streams]),
        "block_changed": np.stack([s.block_changed for s in streams]),
        "optimal_return": np.stack([s.optimal_return for s in streams]),
        "task_shift": np.array([[S.SHIFT_CODES[k] for k in s.task_shift] for s in streams]),
        "stream_seconds": stream_s, "fresh_seconds": fresh_s, "head_probe_seconds": hp_s,
        "auc_source": AUC_SOURCE,
        "batch_size": config.batch_size, "block_updates": scfg.block_updates,
        "block_env_steps": scfg.block_updates * config.batch_size,
        "param_count": param_count(config),
    })
    return res


def relative_metrics(res, **kw):
    """AUC_rel etc. via stream_metrics.analyze, which refuses stream/fresh runs from
    different platforms."""
    import stream_metrics
    return stream_metrics.analyze(res, **kw)


def trainable_param_count(config):
    env, _, network, obs_shape = ppo.make_env_and_network(config)
    return ppo.trainable_param_count(network.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape)), config)


def param_count(config):
    env, _, network, obs_shape = ppo.make_env_and_network(config)
    p = network.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape))
    return int(sum(x.size for x in jax.tree.leaves(p)))


def git_commit():
    import subprocess
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"]) != 0
        return h + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _flatten(tree, prefix=""):
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = np.asarray(v)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seeds", required=True)
    p.add_argument("--preset", default="lifelong_stream")
    p.add_argument("--out", required=True)
    p.add_argument("--stream-seed", type=int, default=0)
    p.add_argument("--num-new-tasks", type=int, default=StreamConfig_default("num_new_tasks"))
    p.add_argument("--block-updates", type=int, default=StreamConfig_default("block_updates"))
    p.add_argument("--revisit-every", type=int, default=StreamConfig_default("revisit_every"))
    p.add_argument("--shift-kinds", default=",".join(S.SHIFT_KINDS))
    p.add_argument("--no-fresh", action="store_true")
    p.add_argument("--head-probe-every", type=int, default=0,
                   help="head-reset probe at every n-th new task (pilot: 5); 0 = off")
    p.add_argument("--reset-adam-on-task-switch", action="store_true")
    p.add_argument("--reset-arm", choices=RESET_ARMS, default="none",
                   help="stage-1 scratch reset arms with an oracle switch signal at every block "
                        "boundary from block 2 on: heads = R-H, heads_adam_vn = R-HOV; none = off")
    p.add_argument("--shared-obs-layout", action="store_true",
                   help="task 1 uses the shared (identity) obs layout; stage-1 config: both on")
    p.add_argument("--shared-action-perm", action="store_true",
                   help="task 1 uses the shared (identity) action perm")
    p.add_argument("--allow-dirty", action="store_true",
                   help="run from a dirty tracked tree (dev seeds >= 900 only); recorded")
    p.add_argument("--arm", default="scratch",
                   choices=("scratch", "pretrained", "lora", "frozen_pretrained", "frozen_random",
                            "kickstart"),
                   help="stage-1 fine-tuning arm (arms.py); scratch = unchanged default")
    p.add_argument("--size", choices=("0.1M", "0.3M", "1M", "3M", "10M"),
                   help="size-line width (arms.size_widths, provisional); exclusive with --hidden-size")
    p.add_argument("--bc-dir", help="pretrained arms: directory of per-seed BC checkpoints")
    p.add_argument("--critic-warmup-updates", type=int,
                   help="pretrained arms: critic warm-up length (required for them)")
    p.add_argument("--kickstart-coef", type=float, help="kickstart arm: KL coefficient (from the sweep)")
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--hidden-size", type=int)
    p.add_argument("--num-layers", type=int)
    p.add_argument("--matmul-precision", choices=ppo.MATMUL_PRECISIONS, default="default",
                   help="jax_default_matmul_precision; recorded in the run record. Relative "
                        "metrics refuse to mix settings. GPU equivalence work: highest")
    p.add_argument("--require-gpu", action="store_true",
                   help="refuse to run unless jax.default_backend() is gpu (pilot), unless --allow-cpu")
    p.add_argument("--allow-cpu", action="store_true",
                   help="with --require-gpu: explicitly accept a CPU run; recorded as allow_cpu")
    p.add_argument("--allow-official-seeds", action="store_true",
                   help="required to run seeds < 900 (reserved blocks 500-509, 700-724, 750-759)")
    args = p.parse_args()

    ppo.set_matmul_precision(args.matmul_precision)
    backend = jax.default_backend()
    if args.require_gpu and backend != "gpu" and not (args.allow_cpu and backend == "cpu"):
        raise SystemExit(f"--require-gpu: jax default backend is {backend!r}; pass --allow-cpu to accept CPU")
    seeds = ppo.parse_seeds(args.seeds)
    fresh = check_seeds(seeds, args.allow_official_seeds)  # early exit; run_condition re-checks
    prov = provenance.check_clean(seeds, args.allow_dirty)
    overrides = ppo.resolve_presets(args.preset)
    if args.reset_adam_on_task_switch:
        overrides["reset_adam_on_task_switch"] = True
    if args.layer_norm:
        overrides["layer_norm"] = True
    if args.hidden_size:
        overrides["hidden_size"] = args.hidden_size
    if args.num_layers:
        overrides["num_layers"] = args.num_layers
    if args.size and args.hidden_size:
        raise SystemExit("--size and --hidden-size are exclusive")
    config = ppo.Config(**overrides)
    scfg = S.StreamConfig(stream_seed=args.stream_seed, num_new_tasks=args.num_new_tasks,
                          block_updates=args.block_updates, revisit_every=args.revisit_every,
                          shift_kinds=tuple(args.shift_kinds.split(",")),
                          head_probe_every=args.head_probe_every,
                          shared_obs_layout=args.shared_obs_layout,
                          shared_action_perm=args.shared_action_perm)
    import arms
    try:
        config, arm_kw, arm_rec = arms.prepare(
            config, scfg, seeds, args.arm, args.size, args.critic_warmup_updates,
            args.kickstart_coef, args.bc_dir, args.allow_dirty)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"arm setup: {e}")
    print(f"matmul precision: {ppo.matmul_precision()}\nconfig: {config}\nstream: {scfg}\nseeds: {seeds}\nfresh seeds: {fresh}\n"
          f"params: {param_count(config)}  K: {scfg.block_updates} updates = "
          f"{scfg.block_updates * config.batch_size} env steps", flush=True)
    t0 = time.perf_counter()
    try:
        res = run_condition(config, scfg, seeds, fresh=not args.no_fresh,
                            allow_official_seeds=args.allow_official_seeds,
                            reset_arm=args.reset_arm, allow_dirty=args.allow_dirty, **arm_kw)
    except KickstartWindowError as e:
        raise SystemExit(f"arm setup: {e}")
    res.update(arm_rec)
    res["wall_seconds"] = time.perf_counter() - t0
    res["config"] = json.dumps(ppo.config_record(config))
    res["git_commit"] = git_commit()
    res.update(ppo.device_info())
    res["default_backend"] = backend
    res["require_gpu"], res["allow_cpu"] = bool(args.require_gpu), bool(args.allow_cpu)
    res["stream_config"] = json.dumps(S.config_record(scfg))
    res.update(prov)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **res)
    print(f"saved {args.out} (wall {res['wall_seconds']:.0f}s)")


def StreamConfig_default(name):
    return S.StreamConfig.__dataclass_fields__[name].default


if __name__ == "__main__":
    main()
