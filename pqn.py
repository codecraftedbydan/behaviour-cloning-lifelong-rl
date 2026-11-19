"""PQN: Parallel Q-Networks (Gallici et al., 2024, "Simplifying Deep Temporal Difference
Learning"; reference implementation purejaxql/pqn_*.py). Stage-1 cells 15-16.

Q-learning with
  * a LayerNorm Q-network (ppo.QNetwork: the shared ppo.Trunk with layer_norm=True, plus
    a linear head q_out),
  * vectorised envs (config.num_envs) and on-policy rollouts of config.num_steps steps,
  * Q(lambda) targets computed once per update from the rollout's own Q-values
    (lambda_returns; purejaxql _get_target),
  * epsilon-greedy exploration with a linear decay schedule (epsilon),
  * no replay buffer and no target network,
  * the TD loss 0.5 * mean((Q(s,a) - target)^2) over update_epochs x num_minibatches.

build_update_step has the same signature and RunnerState as ppo.build_update_step, so the
lifelong stream runner (lifelong.make_stream_train / run_condition), its probes, greedy
evaluation, retention and provenance are shared with PPO unchanged. value_norm is carried
but never used (PQN uses raw rewards, rules sec. 7).

Spec decisions beyond purejaxql are documented in the README ("PQN (cells 15-16)")."""
from typing import NamedTuple

import jax
import jax.numpy as jnp

import ppo


# What a PQN cell changes on top of the run's base config (the lifelong_stream preset, so
# num_envs, num_steps, update_epochs, num_minibatches, lr, gamma, optimiser, activation and
# eval match the PPO cells; rules sec. 3 budget fairness). max_grad_norm 10 and lambda 0.65
# are purejaxql's; the eps schedule is purejaxql's 1.0 -> 0.05 over 10% of one task's
# K = 100 updates, reset at every block start (decision 5 Nov; README, "PQN").
PQN_OVERRIDES = dict(algo="pqn", layer_norm=True, normalize_value=False, max_grad_norm=10.0)


def pqn_config(base: ppo.Config) -> ppo.Config:
    import dataclasses
    return dataclasses.replace(base, **PQN_OVERRIDES)


def check_config(config: ppo.Config):
    """Refuse PPO-only or unsupported options for algo == "pqn"."""
    if config.algo != "pqn":
        raise ValueError(f"pqn.check_config: algo={config.algo!r}")
    bad = []
    if not config.layer_norm:
        bad.append("layer_norm=False (PQN always uses the LayerNorm Q-network)")
    if not config.shared_trunk:
        bad.append("shared_trunk=False")
    if config.normalize_value:
        bad.append("normalize_value (PQN uses raw rewards / unnormalised Q)")
    if config.kickstart_coef:
        bad.append("kickstart_coef (policy distillation is a PPO arm)")
    if config.lora_rank:
        bad.append("lora_rank")
    if config.reset_adam_on_task_switch or config.task_switch_updates:
        bad.append("task-switch hooks")
    if not config.eval_greedy:
        bad.append("eval_greedy=False (PQN is evaluated greedily, argmax Q; purejaxql EPS_TEST 0)")
    if not 0.0 <= config.pqn_lambda <= 1.0:
        bad.append(f"pqn_lambda={config.pqn_lambda} not in [0, 1]")
    if not (0.0 <= config.pqn_eps_finish <= 1.0 and 0.0 <= config.pqn_eps_start <= 1.0):
        bad.append("epsilon outside [0, 1]")
    if config.pqn_eps_decay_updates < 0:
        bad.append("pqn_eps_decay_updates < 0")
    if bad:
        raise ValueError("PQN config refuses: " + "; ".join(bad))


def epsilon(config: ppo.Config, update_idx):
    """Linear decay pqn_eps_start -> pqn_eps_finish over the first pqn_eps_decay_updates
    updates, constant afterwards (optax.linear_schedule, as purejaxql).

    update_idx is the block-local update index (decision, 5 Nov): lifelong.make_stream_train
    zeros RunnerState.update_idx at every block start when algo == "pqn", so every block
    (new task or revisit; continual and fresh alike) starts at eps_start and decays over
    that block's own first pqn_eps_decay_updates updates. Not the global counter and not
    global % K (a shorter trailing block would otherwise mis-align)."""
    n = config.pqn_eps_decay_updates
    if n <= 0:
        return jnp.asarray(config.pqn_eps_finish, jnp.float32)
    frac = jnp.clip(jnp.asarray(update_idx, jnp.float32) / n, 0.0, 1.0)
    return (config.pqn_eps_start + frac * (config.pqn_eps_finish - config.pqn_eps_start)).astype(jnp.float32)


def eps_greedy(key, q, eps):
    """purejaxql eps_greedy_exploration: with probability eps a uniform random action,
    otherwise argmax_a Q. q: (..., A)."""
    k_a, k_e = jax.random.split(key)
    greedy = jnp.argmax(q, axis=-1)
    rand = jax.random.randint(k_a, greedy.shape, 0, q.shape[-1])
    return jnp.where(jax.random.uniform(k_e, greedy.shape) < eps, rand, greedy)


def lambda_returns(rewards, q_vals, dones, last_q, gamma, lam):
    """Q(lambda) targets, purejaxql _get_target. rewards, dones: (T, N); q_vals: (T, N, A)
    = Q(s_t) from the rollout; last_q: (N,) = max_a Q(s_T). Recursion (d_t = 1: episode
    ended on transition t):
        G_{T-1} = r_{T-1} + gamma (1 - d_{T-1}) max Q(s_T)
        G_t     = r_t + gamma (1 - d_t) [(1 - lam) max Q(s_{t+1}) + lam G_{t+1}]
    lam = 0: one-step Q-learning targets; lam = 1: discounted Monte-Carlo return
    bootstrapped at T."""
    last = rewards[-1] + gamma * (1.0 - dones[-1]) * last_q

    def backward(carry, x):
        g_next, next_q = carry
        r, d, q = x
        boot = r + gamma * (1.0 - d) * next_q
        g = boot + gamma * lam * (g_next - next_q)
        g = (1.0 - d) * g + d * r
        return (g, jnp.max(q, axis=-1)), g

    # Reverse scan over t = T-2 .. 0. The carry entering step t holds (G_{t+1},
    # max_a Q(s_{t+1})); step t then passes on (G_t, max_a Q(s_t)). Deliberate deviation
    # from purejaxql: its initial carry uses max Q(s_T) (masked by d_{T-1}) as next_q, so
    # its G_{T-2} bootstraps from Q(s_T) instead of Q(s_{T-1}) (an off-by-one at one step
    # per rollout). Here the initial next_q is max Q(s_{T-1}), which makes every G_t equal
    # the recursion above (tests/test_pqn.py checks both against a transcription).
    _, g = jax.lax.scan(backward, (last, jnp.max(q_vals[-1], axis=-1)),
                        (rewards[:-1], dones[:-1], q_vals[:-1]), reverse=True)
    return jnp.concatenate([g, last[None]], axis=0)


class QTransition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    q: jax.Array          # Q(s_t, .) at rollout time, (N, A)
    reward: jax.Array     # learning reward (incl. truncation bootstrap gamma * max Q(s_final))
    done: jax.Array
    finished_return: jax.Array


def rollout(env, env_params, network, config: ppo.Config, params, env_state, obs,
            ep_stats: ppo.EpisodeStats, key, eps):
    """T steps from N envs with eps-greedy actions. Same env stepping, truncation fold and
    episode bookkeeping as ppo.rollout, with V(s_final) replaced by max_a Q(s_final)."""

    def one_step(carry, _):
        env_state, obs, ep_stats, key = carry
        key, k_act, k_env = jax.random.split(key, 3)
        q, _, _ = network.apply(params, obs)
        action = eps_greedy(k_act, q, eps)
        env_keys = jax.random.split(k_env, config.num_envs)
        out = jax.vmap(ppo.env_step, in_axes=(None, 0, 0, 0, None))(
            env, env_keys, env_state, action, env_params)
        reward = out.reward.astype(jnp.float32)
        if config.bootstrap_on_truncation:
            _, final_v, _ = network.apply(params, out.final_obs)
            reward = reward + config.gamma * jax.lax.stop_gradient(final_v) * out.truncated
        ep_stats = ep_stats.update(out.reward.astype(jnp.float32), out.done)
        tr = QTransition(obs, action, q, reward, out.done.astype(jnp.float32),
                         ep_stats.last_return * out.done)
        return (out.state, out.obs, ep_stats, key), tr

    (env_state, obs, ep_stats, _), traj = jax.lax.scan(
        one_step, (env_state, obs, ep_stats, key), None, length=config.num_steps)
    return traj, env_state, obs, ep_stats


class QBatch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    target: jax.Array


def pqn_loss(params, apply_fn, batch: QBatch):
    q, _, _ = apply_fn(params, batch.obs)
    q_sa = jnp.take_along_axis(q, batch.action[..., None], axis=-1)[..., 0]
    td = q_sa - jax.lax.stop_gradient(batch.target)
    loss = 0.5 * jnp.mean(td ** 2)
    return loss, {"loss": loss, "td_loss": loss, "q_mean": jnp.mean(q_sa),
                  "entropy": jnp.mean(ppo.categorical_entropy(q))}


def pqn_update(train_state, batch: QBatch, key, config: ppo.Config, freeze=None):
    """update_epochs x num_minibatches TD steps on shuffled minibatches (ppo.ppo_update's
    loop and freeze handling, with the TD loss)."""
    grad_fn = jax.value_and_grad(pqn_loss, has_aux=True)

    def one_epoch(carry, _):
        ts, key = carry
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, config.batch_size)
        mbs = jax.tree.map(lambda x: x[perm].reshape(
            (config.num_minibatches, config.minibatch_size) + x.shape[1:]), batch)

        def one_minibatch(ts, mb):
            (_, m), grads = grad_fn(ts.params, ts.apply_fn, mb)
            if freeze is None:
                return ts.apply_gradients(grads=grads), m
            grads = jax.tree.map(lambda g, f: jnp.where(f, 0.0, g), grads, freeze)
            new = ts.apply_gradients(grads=grads)
            return new.replace(params=jax.tree.map(lambda n, o, f: jnp.where(f, o, n),
                                                   new.params, ts.params, freeze)), m

        ts, m = jax.lax.scan(one_minibatch, ts, mbs)
        return (ts, key), m

    (train_state, _), metrics = jax.lax.scan(one_epoch, (train_state, key), None,
                                             length=config.update_epochs)
    return train_state, metrics


def build_update_step(env, network, config: ppo.Config, tx, lr_schedule, teacher_params=None,
                      kickstart_scale=None):
    """One PQN update as update_step(runner, env_params) -> (runner, metrics); drop-in for
    ppo.build_update_step (same RunnerState; vnorm passed through untouched)."""
    check_config(config)
    if teacher_params is not None or kickstart_scale is not None:
        raise ValueError("PQN has no kickstarting teacher")

    def update_step(runner: ppo.RunnerState, env_params):
        ts, env_state, obs, ep_stats, key, update_idx, vnorm = runner
        key, k_roll, k_upd = jax.random.split(key, 3)
        extra = {}
        eps = epsilon(config, update_idx)
        traj, env_state, obs, ep_stats = rollout(env, env_params, network, config, ts.params,
                                                 env_state, obs, ep_stats, k_roll, eps)
        _, last_q, _ = network.apply(ts.params, obs)
        targets = lambda_returns(traj.reward, traj.q, traj.done, last_q,
                                 config.gamma, config.pqn_lambda)
        q_sa = jnp.take_along_axis(traj.q, traj.action[..., None], axis=-1)[..., 0]
        batch = jax.tree.map(lambda x: x.reshape((config.batch_size,) + x.shape[2:]),
                             QBatch(traj.obs, traj.action, targets))
        ev = ppo.explained_variance(q_sa.reshape(-1), batch.target)
        freeze = None
        if config.critic_warmup_updates > 0:  # Q-head warm-up: trunk frozen, q_out trains
            # Count warm-up updates via the optimiser step (never reset at block starts), so
            # the per-block update_idx reset for epsilon does not re-trigger warm-up on later
            # blocks. One update = update_epochs x num_minibatches apply_gradients.
            n_mb = config.update_epochs * config.num_minibatches
            warm = (ts.step // n_mb) < config.critic_warmup_updates
            freeze = jax.tree.map(lambda m: jnp.logical_and(m, warm), ppo.critic_only_mask(ts.params))
            extra["critic_warmup"] = warm.astype(jnp.float32)
        if config.freeze_trunk:
            tm = jax.tree.map(jnp.asarray, ppo.trunk_mask(ts.params))
            freeze = tm if freeze is None else jax.tree.map(jnp.logical_or, freeze, tm)
        lr_now = lr_schedule(ts.step)
        ts, loss_m = pqn_update(ts, batch, k_upd, config, freeze)
        health = ppo.network_health(ts.params, network.apply, batch.obs,
                                    config.dormant_tau, config.activation, None)
        n_done = traj.done.sum()
        metrics = {
            "episode_return": ep_stats.last_return.mean(),
            "episode_length": ep_stats.last_length.mean(),
            "rollout_return": jnp.where(n_done > 0, traj.finished_return.sum()
                                        / jnp.maximum(n_done, 1.0), jnp.nan),
            "explained_variance": ev,
            "lr": lr_now,
            "epsilon": eps,
            "target_mean": jnp.mean(targets),
            **{k: v.mean() for k, v in loss_m.items()},
            **health,
            **extra,
        }
        return ppo.RunnerState(ts, env_state, obs, ep_stats, key, update_idx + 1, vnorm), metrics

    return update_step


def graft_trunk(pqn_params, bc_params):
    """Pretrained PQN init: the BC checkpoint's trunk (a PPO ActorCritic with the same
    Trunk) with the PQN network's own q_out. Refuses any trunk path or shape mismatch."""
    a, b = pqn_params["params"]["trunk"], bc_params["params"]["trunk"]
    sa = jax.tree.map(lambda x: x.shape, a)
    sb = jax.tree.map(lambda x: x.shape, b)
    if jax.tree.structure(sa) != jax.tree.structure(sb) or sa != sb:
        raise ValueError(f"BC trunk does not match the PQN trunk: {sb} vs {sa}")
    return {**pqn_params, "params": {**pqn_params["params"], "trunk": b}}
