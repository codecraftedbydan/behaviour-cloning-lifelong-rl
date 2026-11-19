"""PPO from scratch in JAX: milestone 1 of the TARS lifelong-RL project.

The whole training run is one jitted function. That covers env stepping, rollout
collection, GAE, the PPO update, evaluation and network-health logging. We vmap
over parallel envs inside a run and over random seeds outside it, so N seeds
train in a single XLA program.

File layout (top to bottom, roughly in the order of the build guide):
  1. Config                 hyperparameters, as a frozen dataclass
  2. Environment interface  reset/step with auto-reset and termination vs truncation
  3. Actor-critic network   separate actor and critic MLPs, orthogonal init
  4. Rollout                lax.scan over T steps
  5. GAE                    reverse lax.scan
  6. PPO loss               clipped surrogate + (clipped) value loss + entropy
  7. Update                 epochs x shuffled minibatches, also via lax.scan
  8. Network health / eval  dormant units, weight norm, per-task eval hook
  9. make_train + CLI       the jitted, seed-vmapped training function

Usage:
    python ppo.py --env CartPole-v1 --total-timesteps 500000 --num-seeds 5 \
        --out results/cartpole.npz
    python plot.py results/cartpole.npz --out results/cartpole.png
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from typing import Any, NamedTuple

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax import traverse_util
from flax.training.train_state import TrainState

import seeds as _seeds


# =============================================================================
# 1. Config
# =============================================================================
@dataclasses.dataclass(frozen=True)
class Config:
    """All hyperparameters. Frozen, so it is hashable and acts as a static
    (compile-time) constant inside jit."""

    env_name: str = "CartPole-v1"
    total_timesteps: int = 500_000  # env steps *per seed*
    num_envs: int = 4  # N parallel envs per seed
    num_steps: int = 128  # T, rollout length per env per update
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    update_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 2.5e-4
    anneal_lr: bool = True  # linear decay to 0 over training (lifelong: False)
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_value: bool = True  # PPO2-style value-loss clipping (lifelong: False)
    bootstrap_on_truncation: bool = True
    # Value normalisation (ablation, default off). The critic predicts returns in
    # normalised units, using a debiased EMA of return mean/var (MAPPO "ValueNorm").
    normalize_value: bool = False
    value_norm_beta: float = 0.99  # EMA decay per update
    # Optimiser reset hook. At each update index in task_switch_updates, Adam's
    # moments (and its bias-correction count) are reset to their initial state.
    # The LR-schedule counter is kept. The task stream will switch tasks at the
    # same indices.
    reset_adam_on_task_switch: bool = False
    task_switch_updates: tuple[int, ...] = ()
    # Network
    hidden_size: int = 64
    num_layers: int = 2
    activation: str = "tanh"  # "tanh" or "relu" (lifelong: relu)
    shared_trunk: bool = False  # one MLP trunk under params["trunk"] + two linear heads
    layer_norm: bool = False  # LayerNorm after each hidden Dense, before the activation
    # Hooks for fine-tuning a pretrained (BC) policy. All off by default.
    # critic_warmup_updates: for the first N updates only the critic trains (grads of
    #   every non-critic param are zeroed and those params are restored after the
    #   update), so the pretrained policy/trunk is frozen while V catches up.
    # kickstart_coef: weight of KL(pi_teacher || pi) added to the loss (Schmitt et al.
    #   2018, "Kickstarting"); needs teacher_params passed to make_train.
    critic_warmup_updates: int = 0
    kickstart_coef: float = 0.0
    # freeze_trunk: hidden layers never train (shared trunk: params["trunk"]; separate
    #   towers: actor_i / critic_i); only actor_out / critic_out (and LN in heads) train.
    #   Used by the head-reset probe and the frozen-trunk arm.
    freeze_trunk: bool = False
    # LoRA (Hu et al. 2021) on every trunk Dense layer (shared trunk: trunk/layer_i;
    #   separate towers: actor_i / critic_i). lora_rank > 0: each such layer computes
    #   x W + b + (lora_alpha / lora_rank) x A B, with A (in, r) and B (r, out); W and b
    #   (and trunk LayerNorm) are frozen, A, B and the heads (actor_out, critic_out) train.
    #   Stage-1 cell 9: rank 8, alpha 16 (tars-test-rules sec. 7). 0 = off (default).
    lora_rank: int = 0
    lora_alpha: float = 16.0
    # BC -> RL handover. True (default): the RL phase starts with a fresh Adam state
    # (moments reset once, at the handover). False: carry the pretraining optimiser
    # state, which must be passed to make_train as init_opt_state (same optax chain).
    adam_reset_at_handover: bool = True
    # Eval hook + network health
    eval_every: int = 16  # updates between evaluations
    eval_episodes: int = 8  # episodes per task per evaluation
    eval_max_steps: int | None = None  # None -> env's max_steps_in_episode
    eval_greedy: bool = False  # False: sample from the policy, as in training
    dormant_tau: float = 0.1  # tau for tau-dormant units (Sokar et al., 2023)
    # Algorithm (stage-1 cells 15-16). "ppo" (default) or "pqn": PQN (Gallici et al.,
    # "Simplifying Deep Temporal Difference Learning", purejaxql), implemented in pqn.py.
    # The PQN fields appear in run records only when algo == "pqn" (config_record), so
    # every PPO record is unchanged.
    algo: str = "ppo"
    pqn_lambda: float = 0.65          # lambda of the Q(lambda) return (purejaxql LAMBDA)
    pqn_eps_start: float = 1.0        # epsilon-greedy: linear from start to finish ...
    pqn_eps_finish: float = 0.05
    pqn_eps_decay_updates: int = 10   # ... over the first N updates of each block, then constant

    # ---- derived quantities --------------------------------------------------
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        assert self.batch_size % self.num_minibatches == 0
        return self.batch_size // self.num_minibatches

    @property
    def num_evals(self) -> int:
        # Training is organised as num_evals blocks of eval_every updates each,
        # so the update count is rounded down to a multiple of eval_every.
        return max(1, (self.total_timesteps // self.batch_size) // self.eval_every)

    @property
    def num_updates(self) -> int:
        return self.num_evals * self.eval_every


# =============================================================================
# 2. Environment interface
# =============================================================================
# gymnax's own `env.step` auto-resets, but it folds the time limit into `done`,
# and `info["discount"]` is 0 for time-limit endings too. It never tells us
# whether an episode was *terminated* (a real terminal state with value 0) or
# *truncated* (the time limit ran out, so the state still has value). We use
# `step_env` (no auto-reset) and work it out ourselves: we ask `is_terminal`
# again with the time limit pushed to infinity. That works for every gymnax env
# whose params have `max_steps_in_episode` (classic control, MinAtar, ...). For
# other envs every `done` counts as a termination.

_NO_TIME_LIMIT = int(np.iinfo(np.int32).max)


def _without_time_limit(env_params):
    if hasattr(env_params, "max_steps_in_episode"):
        return env_params.replace(max_steps_in_episode=_NO_TIME_LIMIT)
    return None


class StepOut(NamedTuple):
    obs: jax.Array  # next obs. This is the *reset* obs if the episode ended
    state: Any
    reward: jax.Array
    done: jax.Array  # terminated | truncated
    terminated: jax.Array
    truncated: jax.Array
    final_obs: jax.Array  # obs at the end of this transition, before any reset


def env_reset(env, key, env_params):
    return env.reset_env(key, env_params)


def env_step(env, key, state, action, env_params) -> StepOut:
    """One env step, with auto-reset and the termination/truncation split."""
    key_step, key_reset = jax.random.split(key)
    obs_next, state_next, reward, done, _ = env.step_env(
        key_step, state, action, env_params
    )
    no_limit_params = _without_time_limit(env_params)
    if no_limit_params is not None:
        terminated = env.is_terminal(state_next, no_limit_params)
    else:
        terminated = done
    truncated = jnp.logical_and(done, jnp.logical_not(terminated))

    # Auto-reset: if done, swap in a fresh episode's state and obs.
    obs_reset, state_reset = env.reset_env(key_reset, env_params)
    state = jax.tree.map(
        lambda r, s: jnp.where(done, r, s), state_reset, state_next
    )
    obs = jnp.where(done, obs_reset, obs_next)
    return StepOut(obs, state, reward, done, terminated, truncated, obs_next)


class EpisodeStats(NamedTuple):
    """Running per-env episode stats, used for logging only."""

    ep_return: jax.Array  # return so far in the current episode
    ep_length: jax.Array
    last_return: jax.Array  # return of the most recently finished episode
    last_length: jax.Array

    @staticmethod
    def zeros(num_envs: int) -> "EpisodeStats":
        z = jnp.zeros((num_envs,), jnp.float32)
        return EpisodeStats(z, z, z, z)

    def update(self, reward, done) -> "EpisodeStats":
        ret = self.ep_return + reward
        length = self.ep_length + 1
        return EpisodeStats(
            ep_return=jnp.where(done, 0.0, ret),
            ep_length=jnp.where(done, 0.0, length),
            last_return=jnp.where(done, ret, self.last_return),
            last_length=jnp.where(done, length, self.last_length),
        )


# =============================================================================
# 3. Actor-critic network
# =============================================================================
class LoRADense(nn.Module):
    """Dense layer plus a rank-r LoRA update: y = x W + b + (alpha / r) (x A) B.
    Param names kernel / bias match nn.Dense (and are created first, in the same order),
    so a plain checkpoint grafts onto it by path. A: kaiming-uniform as in PEFT
    (U(-1/sqrt(in), 1/sqrt(in))); B = 0, so at init the layer equals the plain layer."""

    features: int
    rank: int
    alpha: float
    kernel_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        d_in = x.shape[-1]
        kernel = self.param("kernel", self.kernel_init, (d_in, self.features))
        bias = self.param("bias", self.bias_init, (self.features,))
        bound = 1.0 / np.sqrt(d_in)
        a = self.param("lora_a", lambda k, sh: jax.random.uniform(k, sh, jnp.float32, -bound, bound),
                       (d_in, self.rank))
        b = self.param("lora_b", nn.initializers.zeros, (self.rank, self.features))
        return x @ kernel + bias + (self.alpha / self.rank) * ((x @ a) @ b)


def hidden_dense(features, gain, name, lora_rank=0, lora_alpha=16.0):
    """Hidden-layer Dense (orthogonal(gain), zero bias); LoRADense when lora_rank > 0."""
    if lora_rank:
        return LoRADense(features, lora_rank, lora_alpha, kernel_init=orthogonal(gain),
                         bias_init=constant(0.0), name=name)
    return nn.Dense(features, kernel_init=orthogonal(gain), bias_init=constant(0.0), name=name)


class Trunk(nn.Module):
    """Shared MLP trunk. Its params live in their own subtree, params["trunk"],
    so a later fine-tuning/LoRA/freezing experiment can address the
    "pretrained representation" as a single object."""

    hidden_size: int
    num_layers: int
    activation: str
    layer_norm: bool = False
    lora_rank: int = 0
    lora_alpha: float = 16.0

    @nn.compact
    def __call__(self, x):
        act = {"tanh": nn.tanh, "relu": nn.relu}[self.activation]
        hiddens = {}
        for i in range(self.num_layers):
            x = hidden_dense(self.hidden_size, np.sqrt(2), f"layer_{i}",
                             self.lora_rank, self.lora_alpha)(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f"layer_{i}_ln")(x)
            x = act(x)
            hiddens[f"trunk_{i}"] = x
        return x, hiddens


class ActorCritic(nn.Module):
    """Actor-critic MLP, either two separate towers (default) or a shared trunk.

    Orthogonal init as in the build guide: gain sqrt(2) on hidden layers, 0.01
    on the policy head (so initial logits are ~0, a near-uniform policy) and 1.0
    on the value head.

    Returns (logits, value, hiddens). `hiddens` is a dict {layer name:
    post-activation features} that feeds the per-layer network-health metrics.
    Separate towers use the names actor_i / critic_i, and the shared trunk uses trunk_i.

    NB: the separate-tower path keeps its original module names (actor_0, ...).
    Flax derives init RNG from module names, so renaming them would change the
    initial weights and break bit-identity with earlier runs.
    """

    num_actions: int
    hidden_size: int = 64
    num_layers: int = 2
    activation: str = "tanh"
    obs_ndim: int = 1  # number of trailing obs dims to flatten (e.g. 3 for MinAtar)
    shared_trunk: bool = False
    # LayerNorm (Ba et al.) after each hidden Dense, before the activation. Off by
    # default. The extra modules have their own names (<layer>_ln), so turning it on
    # does not change the init of any Dense kernel.
    layer_norm: bool = False
    # LoRA on every trunk Dense (hidden) layer; 0 = off. Heads stay plain Dense.
    lora_rank: int = 0
    lora_alpha: float = 16.0

    @nn.compact
    def __call__(self, obs):
        act = {"tanh": nn.tanh, "relu": nn.relu}[self.activation]
        batch_shape = obs.shape[: obs.ndim - self.obs_ndim]
        x = obs.reshape(batch_shape + (-1,)).astype(jnp.float32)

        def dense(features, gain, name):
            return nn.Dense(
                features,
                kernel_init=orthogonal(gain),
                bias_init=constant(0.0),
                name=name,
            )

        if self.shared_trunk:
            z, hiddens = Trunk(self.hidden_size, self.num_layers, self.activation,
                               self.layer_norm, self.lora_rank, self.lora_alpha, name="trunk")(x)
            logits = dense(self.num_actions, 0.01, "actor_out")(z)
            value = dense(1, 1.0, "critic_out")(z).squeeze(-1)
            return logits, value, hiddens

        def ln(h, name):
            return nn.LayerNorm(name=f"{name}_ln")(h) if self.layer_norm else h

        def hdense(name):
            return hidden_dense(self.hidden_size, np.sqrt(2), name, self.lora_rank, self.lora_alpha)

        hiddens = {}
        a = x
        for i in range(self.num_layers):
            a = act(ln(hdense(f"actor_{i}")(a), f"actor_{i}"))
            hiddens[f"actor_{i}"] = a
        logits = dense(self.num_actions, 0.01, "actor_out")(a)

        c = x
        for i in range(self.num_layers):
            c = act(ln(hdense(f"critic_{i}")(c), f"critic_{i}"))
            hiddens[f"critic_{i}"] = c
        value = dense(1, 1.0, "critic_out")(c).squeeze(-1)

        return logits, value, hiddens


# PQN Q head init gain. purejaxql uses flax's default Dense init; here the trunk keeps the
# repo's orthogonal init (so it is the BC/PPO trunk) and q_out uses gain 0.01, like the policy
# head, so Q starts near 0 instead of max over O(1) random values: with the sparse gridworld
# reward and truncation bootstrapping, gain 1 kept Q ~ 1 everywhere and nothing was learned
# on dev seeds 990-992 (README, "PQN").
Q_HEAD_GAIN = 0.01


class QNetwork(nn.Module):
    """PQN Q-network (pqn.py; Gallici et al., purejaxql): the same shared Trunk as
    ActorCritic (module name "trunk", so a seed's trunk init and a PPO BC checkpoint's
    trunk have the same paths and shapes) and one linear head q_out with num_actions
    outputs. Returns (q, max_a q, hiddens) so the policy-side helpers work unchanged:
    "logits" = Q (greedy eval = argmax Q; entropy = entropy of softmax(Q), rules sec. 7),
    "value" = max_a Q (greedy state value; used by the random-target probe)."""

    num_actions: int
    hidden_size: int = 64
    num_layers: int = 2
    activation: str = "relu"
    obs_ndim: int = 1
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs):
        batch_shape = obs.shape[: obs.ndim - self.obs_ndim]
        x = obs.reshape(batch_shape + (-1,)).astype(jnp.float32)
        z, hiddens = Trunk(self.hidden_size, self.num_layers, self.activation, self.layer_norm,
                           name="trunk")(x)
        q = nn.Dense(self.num_actions, kernel_init=orthogonal(Q_HEAD_GAIN), bias_init=constant(0.0),
                     name="q_out")(z)
        return q, jnp.max(q, axis=-1), hiddens


# Categorical distribution helpers. We always work in log space via
# log_softmax, which is numerically stable (common-bugs list).
def categorical_log_prob(logits, actions):
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp_all, actions[..., None], axis=-1).squeeze(-1)


def categorical_entropy(logits):
    logp_all = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(jnp.exp(logp_all) * logp_all, axis=-1)


# Value normalisation (optional). Debiased EMA of return mean and mean-square;
# the critic outputs normalised values v_n, and V = v_n * std + mean.
class ValueNorm(NamedTuple):
    mean_ema: jax.Array
    sq_ema: jax.Array
    debias: jax.Array  # EMA of 1.0; corrects the zero-init bias

    @staticmethod
    def init() -> "ValueNorm":
        z = jnp.zeros((), jnp.float32)
        return ValueNorm(z, z, z)

    def stats(self):
        """(mean, std). Identity (0, 1) before the first update."""
        d = jnp.maximum(self.debias, 1e-8)
        mean = jnp.where(self.debias > 0, self.mean_ema / d, 0.0)
        var = jnp.where(self.debias > 0, self.sq_ema / d - mean**2, 1.0)
        return mean, jnp.sqrt(jnp.maximum(var, 1e-2))  # std floor 0.1, as in MAPPO

    def update(self, x, beta) -> "ValueNorm":
        return ValueNorm(beta * self.mean_ema + (1 - beta) * jnp.mean(x),
                         beta * self.sq_ema + (1 - beta) * jnp.mean(x**2),
                         beta * self.debias + (1 - beta))

    def normalize(self, v):
        mean, std = self.stats()
        return (v - mean) / std

    def denormalize(self, v_n):
        mean, std = self.stats()
        return v_n * std + mean


# =============================================================================
# 4. Rollout
# =============================================================================
class Transition(NamedTuple):
    """One time step of a rollout. Every field has shape (N, ...), and after
    the scan (T, N, ...)."""

    obs: jax.Array
    action: jax.Array
    logp: jax.Array  # log pi_old(a|s), stored now and never recomputed later
    value: jax.Array  # V_old(s)
    reward: jax.Array  # reward used for learning (incl. truncation bootstrap)
    done: jax.Array  # 1.0 if the episode ended on this transition
    # Undiscounted return of the episode that ended on this transition (0 if none).
    # Logging only (per-rollout mean return of finished episodes).
    finished_return: jax.Array = None


def rollout(env, env_params, network, config: Config, params, env_state, obs,
            ep_stats: EpisodeStats, key, value_norm: ValueNorm | None = None):
    """Collect T steps from N envs with lax.scan. Returns (traj, env_state,
    obs, ep_stats). traj has leaves of shape (T, N, ...). Values in traj are
    always in raw return units (denormalised if normalize_value is on)."""

    def critic(v):
        return value_norm.denormalize(v) if config.normalize_value else v

    def one_step(carry, _):
        env_state, obs, ep_stats, key = carry
        key, k_act, k_env = jax.random.split(key, 3)

        logits, value, _ = network.apply(params, obs)
        value = critic(value)
        action = jax.random.categorical(k_act, logits)
        logp = categorical_log_prob(logits, action)

        # A separate PRNG key for every env, so envs are not correlated.
        env_keys = jax.random.split(k_env, config.num_envs)
        out = jax.vmap(env_step, in_axes=(None, 0, 0, 0, None))(
            env, env_keys, env_state, action, env_params
        )

        # Truncation bootstrap: if the episode was cut by the time limit, the
        # "true" return continues past this point. We fold gamma * V(s_final)
        # into the reward and still cut GAE at `done`. This needs V of the
        # pre-reset obs, which is why env_step returns final_obs.
        reward = out.reward.astype(jnp.float32)
        if config.bootstrap_on_truncation:
            _, final_value, _ = network.apply(params, out.final_obs)
            final_value = jax.lax.stop_gradient(critic(final_value))
            reward = reward + config.gamma * final_value * out.truncated

        ep_stats = ep_stats.update(out.reward.astype(jnp.float32), out.done)
        transition = Transition(
            obs=obs,
            action=action,
            logp=logp,
            value=value,
            reward=reward,
            done=out.done.astype(jnp.float32),
            finished_return=ep_stats.last_return * out.done,
        )
        return (out.state, out.obs, ep_stats, key), transition

    (env_state, obs, ep_stats, _), traj = jax.lax.scan(
        one_step, (env_state, obs, ep_stats, key), None, length=config.num_steps
    )
    return traj, env_state, obs, ep_stats


# =============================================================================
# 5. GAE
# =============================================================================
def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Generalised Advantage Estimation with a reverse scan.

    Inputs have shape (T, N). last_value (N,) is V(s_T), the value of the obs
    after the final rollout step. Convention: dones[t] = 1 means the episode
    ended on transition t, so s_{t+1} belongs to a new episode and we must not
    bootstrap through it:

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}
        R_t     = A_t + V(s_t)          (value-function target)
    """
    next_values = jnp.concatenate([values[1:], last_value[None]], axis=0)
    not_done = 1.0 - dones
    deltas = rewards + gamma * next_values * not_done - values

    def backward(adv_next, x):
        delta, nd = x
        adv = delta + gamma * gae_lambda * nd * adv_next
        return adv, adv

    _, advantages = jax.lax.scan(
        backward, jnp.zeros_like(last_value), (deltas, not_done), reverse=True
    )
    return advantages, advantages + values


# =============================================================================
# 6. PPO loss
# =============================================================================
class Batch(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array  # old log-prob from the rollout
    value: jax.Array  # old value from the rollout
    advantages: jax.Array
    returns: jax.Array


def ppo_loss(params, apply_fn, batch: Batch, config: Config, teacher_params=None,
             kickstart_scale=None):
    """Clipped-surrogate PPO loss on one minibatch. Returns (loss, metrics).
    With config.kickstart_coef > 0 and teacher_params given, adds
    kickstart_coef * KL(pi_teacher || pi) (teacher = same architecture).
    kickstart_scale: optional traced 0/1 gate (lifelong stream: shared-layout window). The
    effective coefficient is kickstart_coef * scale; when it is 0 the term is exactly 0
    (jnp.where, so even a non-finite KL cannot leak in). Logged as kickstart_coef_eff and
    kickstart_term. None (default): ungated, as before."""
    logits, value, _ = apply_fn(params, batch.obs)
    logp = categorical_log_prob(logits, batch.action)

    # Targets are constants with respect to the parameters being optimised.
    adv = jax.lax.stop_gradient(batch.advantages)
    returns = jax.lax.stop_gradient(batch.returns)

    # Advantage normalisation is done *per minibatch* (as in CleanRL /
    # SB3). The alternative is once per batch; see README "decisions".
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    log_ratio = logp - batch.logp
    ratio = jnp.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    if config.clip_value:
        v_clipped = batch.value + jnp.clip(
            value - batch.value, -config.clip_eps, config.clip_eps
        )
        value_loss = 0.5 * jnp.mean(
            jnp.maximum((value - returns) ** 2, (v_clipped - returns) ** 2)
        )
    else:
        value_loss = 0.5 * jnp.mean((value - returns) ** 2)

    entropy = jnp.mean(categorical_entropy(logits))
    loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
    extra = {}
    if config.kickstart_coef > 0 and teacher_params is None:
        raise ValueError("kickstart_coef > 0 needs teacher_params (the frozen BC policy)")
    if config.kickstart_coef > 0 and teacher_params is not None:
        t_logits = jax.lax.stop_gradient(apply_fn(teacher_params, batch.obs)[0])
        t_logp = jax.nn.log_softmax(t_logits, -1)
        kl = jnp.mean(jnp.sum(jnp.exp(t_logp) * (t_logp - jax.nn.log_softmax(logits, -1)), -1))
        extra["kickstart_kl"] = kl
        if kickstart_scale is None:
            loss = loss + config.kickstart_coef * kl
        else:
            coef = config.kickstart_coef * jnp.asarray(kickstart_scale, jnp.float32)
            term = jnp.where(coef > 0, coef * kl, 0.0)
            loss = loss + term
            extra["kickstart_coef_eff"], extra["kickstart_term"] = coef, term

    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        # Low-variance, always non-negative KL estimator, (r - 1) - log r
        # (Schulman's "k3"). It is exactly 0 when ratio == 1.
        "approx_kl": jnp.mean((ratio - 1.0) - log_ratio),
        "clip_frac": jnp.mean((jnp.abs(ratio - 1.0) > config.clip_eps).astype(jnp.float32)),
        "ratio_mean": jnp.mean(ratio),
        **extra,
    }
    return loss, metrics


# =============================================================================
# 7. Update: epochs x shuffled minibatches
# =============================================================================
def ppo_update(train_state: TrainState, batch: Batch, key, config: Config,
               teacher_params=None, freeze=None, kickstart_scale=None):
    """batch leaves have leading dim batch_size (= T*N, already flattened).
    Returns (train_state, metrics), where each metric has shape
    (epochs, num_minibatches).

    freeze: optional pytree of bools (same structure as params, may be traced).
    Frozen leaves get zero gradient and are restored after every step, so neither
    the params nor (for a fresh Adam state) their moments move."""
    grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    def one_epoch(carry, _):
        ts, key = carry
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, config.batch_size)
        minibatches = jax.tree.map(
            lambda x: x[perm].reshape(
                (config.num_minibatches, config.minibatch_size) + x.shape[1:]
            ),
            batch,
        )

        def one_minibatch(ts, mb):
            (_, m), grads = grad_fn(ts.params, ts.apply_fn, mb, config, teacher_params,
                                    kickstart_scale)
            if freeze is None:
                return ts.apply_gradients(grads=grads), m
            grads = jax.tree.map(lambda g, f: jnp.where(f, 0.0, g), grads, freeze)
            new = ts.apply_gradients(grads=grads)
            return new.replace(params=jax.tree.map(lambda n, o, f: jnp.where(f, o, n),
                                                   new.params, ts.params, freeze)), m

        ts, m = jax.lax.scan(one_minibatch, ts, minibatches)
        return (ts, key), m

    (train_state, _), metrics = jax.lax.scan(
        one_epoch, (train_state, key), None, length=config.update_epochs
    )
    return train_state, metrics


# =============================================================================
# 8. Network health + eval hook (these are what the lifelong task stream needs)
# =============================================================================
def network_health(params, apply_fn, obs, tau, activation="tanh", lora_scale=None):
    """Plasticity diagnostics on a batch of observations, per hidden layer and
    aggregated over all hidden units.

    * tau-dormant units (Sokar et al., 2023, "The Dormant Neuron Phenomenon in
      Deep RL"). For hidden unit i in layer l,
          s_i = E_x |h_i(x)| / ( (1/H_l) * sum_k E_x |h_k(x)| )
      and unit i is tau-dormant if s_i <= tau.
    * Saturated units (tars-test-rules.md section 2): a tanh unit whose
      |activation| > 0.99 on >= 99% of the batch. Defined for tanh only;
      NaN for ReLU.
    * Weight norms: Frobenius norm of every Dense kernel (per layer, incl.
      heads) plus the global L2 norm of all params.

    Keys: dormant_frac, saturated_frac, weight_norm (aggregates) and
    dormant_frac.<layer>, saturated_frac.<layer>, weight_norm.<path>.

    LoRA params only (lora_scale = alpha / r; keys absent otherwise, so non-LoRA records are
    unchanged): weight_norm.<path> stays the frozen BASE kernel; additionally
      lora_adapter_norm          global L2 norm of all lora_a / lora_b leaves
      lora_delta_norm.<path>     ||scale * A B||_F per adapted layer
      weight_norm_merged.<path>  ||W + scale * A B||_F per adapted layer
      weight_norm_merged.trunk   L2 norm of all merged trunk kernels
      weight_norm_merged         global L2 norm of the merged params (adapters dropped).
    """
    _, _, hiddens = apply_fn(params, obs)
    out = {}
    n_dormant, n_saturated, n_total = 0.0, 0.0, 0
    for name, h in hiddens.items():
        h = h.reshape(-1, h.shape[-1])
        mean_abs = jnp.mean(jnp.abs(h), axis=0)  # (H,)
        score = mean_abs / (jnp.mean(mean_abs) + 1e-8)
        dormant = score <= tau
        saturated = jnp.mean(jnp.abs(h) > 0.99, axis=0) >= 0.99
        out[f"dormant_frac.{name}"] = jnp.mean(dormant)
        if activation == "tanh":
            out[f"saturated_frac.{name}"] = jnp.mean(saturated)
        else:
            out[f"saturated_frac.{name}"] = jnp.float32(jnp.nan)
        n_dormant += jnp.sum(dormant)
        n_saturated += jnp.sum(saturated)
        n_total += h.shape[-1]
    out["dormant_frac"] = n_dormant / n_total
    out["saturated_frac"] = n_saturated / n_total if activation == "tanh" else jnp.float32(jnp.nan)
    flat = traverse_util.flatten_dict(params["params"] if "params" in params else params)
    for path, w in flat.items():
        if path[-1] == "kernel":
            out["weight_norm." + ".".join(path[:-1])] = jnp.linalg.norm(w)
    out["weight_norm"] = optax.tree.norm(params)
    if lora_scale is not None and any(_is_lora_leaf(p) for p in flat):
        ad, merged, trunk_sq = [], {}, 0.0
        for path, w in flat.items():
            if _is_lora_leaf(path):
                ad.append(w)
                continue
            name = ".".join(path[:-1])
            if path[-1] == "kernel" and path[:-1] + ("lora_a",) in flat:
                d = lora_scale * jnp.matmul(flat[path[:-1] + ("lora_a",)], flat[path[:-1] + ("lora_b",)])
                out["lora_delta_norm." + name] = jnp.linalg.norm(d)
                w = w + d
                out["weight_norm_merged." + name] = jnp.linalg.norm(w)
            if path[-1] == "kernel" and "trunk" in path:
                trunk_sq = trunk_sq + jnp.sum(jnp.square(w))
            merged[path] = w
        out["lora_adapter_norm"] = optax.tree.norm(ad)
        out["weight_norm_merged.trunk"] = jnp.sqrt(trunk_sq)
        out["weight_norm_merged"] = optax.tree.norm(merged)
    return out


def lora_scale_of(config) -> float | None:
    return config.lora_alpha / config.lora_rank if config.lora_rank else None


def evaluate(env, network, config: Config, params, task_params, key, max_steps):
    """Eval hook. Runs `eval_episodes` episodes on each task and returns the
    mean episode return per task, shape (num_tasks,).

    `task_params` is a *stacked* env-params pytree with leading dim num_tasks.
    For CartPole there is a single task (the default params). For the lifelong
    stream it will hold one entry per task (e.g. one per gridworld layout), and
    this function needs no changes. Episodes are cut off after `max_steps`.
    """

    def eval_one_task(env_params, key):
        k_reset, k_run = jax.random.split(key)
        reset_keys = jax.random.split(k_reset, config.eval_episodes)
        obs, state = jax.vmap(env_reset, in_axes=(None, 0, None))(
            env, reset_keys, env_params
        )

        def step(carry, _):
            obs, state, ret, alive, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            logits, _, _ = network.apply(params, obs)
            if config.eval_greedy:
                action = jnp.argmax(logits, axis=-1)
            else:
                action = jax.random.categorical(k_act, logits)
            env_keys = jax.random.split(k_env, config.eval_episodes)
            out = jax.vmap(env_step, in_axes=(None, 0, 0, 0, None))(
                env, env_keys, state, action, env_params
            )
            ret = ret + out.reward * alive  # only count each env's first episode
            alive = alive * (1.0 - out.done)
            return (out.obs, out.state, ret, alive, key), None

        init = (obs, state, jnp.zeros(config.eval_episodes),
                jnp.ones(config.eval_episodes), k_run)
        (_, _, ret, _, _), _ = jax.lax.scan(step, init, None, length=max_steps)
        return ret.mean()

    num_tasks = jax.tree.leaves(task_params)[0].shape[0]
    return jax.vmap(eval_one_task)(task_params, jax.random.split(key, num_tasks))


# =============================================================================
# 9. make_train: the whole run as one jittable function
# =============================================================================
class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: Any
    obs: jax.Array
    ep_stats: EpisodeStats
    key: jax.Array
    update_idx: jax.Array  # number of PPO updates done so far
    value_norm: ValueNorm


def reset_adam_moments(opt_state, fresh_opt_state):
    """Return opt_state with every optax ScaleByAdamState (count, mu, nu)
    replaced by its freshly initialised version. Everything else is kept,
    including the LR-schedule counter and the gradient-clipping state."""
    is_adam = lambda x: isinstance(x, optax.ScaleByAdamState)  # noqa: E731
    return jax.tree.map(lambda old, new: new if is_adam(old) else old,
                        opt_state, fresh_opt_state, is_leaf=is_adam)


# Named configs. "purejaxrl_match" reproduces PureJaxRL's MinAtar PPO config
# (examples/brax_minatar.ipynb) on our code: same network, LR, batch layout. eval_every=20
# makes 1e7 steps exactly 1220 updates, PureJaxRL's budget. Nothing here is tuned.
PRESETS = {
    "default": {},
    "purejaxrl_match": dict(num_envs=64, num_steps=128, num_minibatches=8, lr=5e-3,
                            activation="relu", hidden_size=64, eval_every=20),
    # Structural choices for the never-ending task stream. Scale settings
    # (num_envs, lr, ...) are left alone, so compose it, e.g.
    # --preset purejaxrl_match,lifelong. Eval/probe settings follow
    # tars-test-rules.md section 1-2 (greedy, 10 episodes, tau = 0.025).
    # "lifelong_stream": gridworld task-stream preset. Same as lifelong plus
    # normalize_value=True (team vote, 24 Oct: returns change scale across tasks).
    # Scale settings are sized for the 8-core CPU pilot (see lifelong.py).
    "lifelong": dict(anneal_lr=False, reset_adam_on_task_switch=False, activation="relu",
                     clip_value=False, shared_trunk=True, normalize_value=False,
                     dormant_tau=0.025, eval_greedy=True, eval_episodes=10),
}
PRESETS["lifelong_stream"] = dict(PRESETS["lifelong"], normalize_value=True, env_name="Gridworld-8",
                                  num_envs=32, num_steps=64, num_minibatches=4, update_epochs=4,
                                  lr=2.5e-4, hidden_size=128, gamma=0.99)


def resolve_presets(spec: str) -> dict:
    """"a,b" -> PRESETS[a] updated by PRESETS[b] (later wins)."""
    out = {}
    for name in spec.split(","):
        out.update(PRESETS[name.strip()])
    return out


def parse_seeds(spec: str) -> list[int]:
    """"100-109" -> [100..109]; "1,5,7" -> [1, 5, 7]."""
    out = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


# Incremented at *trace* time only. tests/ uses it to check that jit compiles once.
TRACE_COUNT = {"train": 0}


def device_info() -> dict:
    """Device + library versions, recorded in every run record. Relative metrics refuse to
    mix runs from different platforms (stream_metrics.check_same_platform)."""
    import jaxlib
    devs = jax.devices()
    return {"device_platform": devs[0].platform,
            "device_name": str(getattr(devs[0], "device_kind", devs[0])),
            "device_count": len(devs),
            "jax_version": jax.__version__,
            "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
            "matmul_precision": matmul_precision()}


MATMUL_PRECISIONS = ("default", "highest")


# Same set as lifelong.RESERVED_SEEDS: both come from seeds.py (dependency-free).
RESERVED_SEEDS_PYTEST = _seeds.RESERVED_SEEDS


def _refuse_reserved_seeds_under_pytest(seeds):
    """Tests never use the reserved stream (500-509, 700-724, 750-759) or fresh (600-609,
    800-824, 850-859) seed blocks; from seeds.py, the same set lifelong.py uses."""
    import os
    if os.environ.get("PYTEST_CURRENT_TEST"):
        hit = sorted({s for s in seeds if s in RESERVED_SEEDS_PYTEST})
        if hit:
            raise SystemExit(f"seeds {hit} are reserved: refused under pytest")


def matmul_precision() -> str:
    """Current jax_default_matmul_precision as recorded in run records ("default" = unset,
    i.e. backend default: TF32 on Ampere GPUs, full fp32 on CPU)."""
    v = jax.config.jax_default_matmul_precision
    return "default" if v is None else str(v)


def set_matmul_precision(p: str):
    assert p in MATMUL_PRECISIONS, p
    jax.config.update("jax_default_matmul_precision", None if p == "default" else p)


def explained_variance(values, returns):
    """1 - Var[R - V] / Var[R]. 1 is perfect, 0 is no better than a constant,
    and negative is worse than a constant."""
    var_r = jnp.var(returns)
    return jnp.where(var_r > 1e-8, 1.0 - jnp.var(returns - values) / var_r, jnp.nan)


def make_env_and_network(config: Config):
    if config.env_name.startswith("Gridworld"):  # "Gridworld" or "Gridworld-<size>"
        import gridworld
        size = int(config.env_name.split("-")[1]) if "-" in config.env_name else 8
        env = gridworld.Gridworld(size)
        env_params = env.default_params
    else:
        env, env_params = gymnax.make(config.env_name)
    obs_shape = env.observation_space(env_params).shape
    if config.algo == "pqn":
        if not config.shared_trunk or config.lora_rank:
            raise ValueError("PQN: shared trunk only, no LoRA")
        network = QNetwork(num_actions=env.action_space(env_params).n, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, activation=config.activation,
                           obs_ndim=len(obs_shape), layer_norm=config.layer_norm)
        return env, env_params, network, obs_shape
    if config.algo != "ppo":
        raise ValueError(f"algo {config.algo!r} not in ('ppo', 'pqn')")
    network = ActorCritic(
        num_actions=env.action_space(env_params).n,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        activation=config.activation,
        obs_ndim=len(obs_shape),
        shared_trunk=config.shared_trunk,
        layer_norm=config.layer_norm,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
    )
    return env, env_params, network, obs_shape


LORA_FIELDS = ("lora_rank", "lora_alpha")
PQN_FIELDS = ("algo", "pqn_lambda", "pqn_eps_start", "pqn_eps_finish", "pqn_eps_decay_updates")


def config_record(config: Config) -> dict:
    """asdict(config) for run records. The LoRA fields appear only when LoRA is on
    (lora_rank > 0) and the algorithm / PQN fields only when algo != "ppo", so records of
    every earlier configuration are unchanged."""
    d = dataclasses.asdict(config)
    if not config.lora_rank:
        for k in LORA_FIELDS:
            d.pop(k)
    if config.algo == "ppo":
        for k in PQN_FIELDS:
            d.pop(k)
    return d


def _is_lora_leaf(path):
    return str(path[-1]) in ("lora_a", "lora_b")


def has_lora(params) -> bool:
    return any(_is_lora_leaf(p) for p in traverse_util.flatten_dict(params))


def lora_freeze_mask(params):
    """LoRA arm: True (frozen) for every trunk param (hidden Dense kernels/biases and
    trunk LayerNorm) except the adapters lora_a / lora_b; heads are trainable."""
    flat = traverse_util.flatten_dict(params)
    tm = traverse_util.flatten_dict(trunk_mask(params))
    return traverse_util.unflatten_dict({p: bool(tm[p]) and not _is_lora_leaf(p) for p in flat})


def trainable_mask(params, config: Config):
    """True for params that can ever train under config (freeze_trunk / LoRA); ignores
    the critic warm-up, which only delays training."""
    flat = traverse_util.flatten_dict(params)
    if config.lora_rank:
        fm = traverse_util.flatten_dict(lora_freeze_mask(params))
    elif config.freeze_trunk:
        fm = traverse_util.flatten_dict(trunk_mask(params))
    else:
        fm = {p: False for p in flat}
    return traverse_util.unflatten_dict({p: not fm[p] for p in flat})


def trainable_param_count(params, config: Config) -> int:
    m = traverse_util.flatten_dict(trainable_mask(params, config))
    return int(sum(v.size for p, v in traverse_util.flatten_dict(params).items() if m[p]))


def graft_lora(lora_params, plain_params):
    """LoRA-network params with every non-adapter leaf taken from plain_params (same
    paths, e.g. a BC checkpoint of the plain network) and the adapters from lora_params."""
    fl, fp = traverse_util.flatten_dict(lora_params), traverse_util.flatten_dict(plain_params)
    missing = [p for p in fl if not _is_lora_leaf(p) and p not in fp]
    extra = [p for p in fp if p not in fl]
    if missing or extra:
        raise ValueError(f"graft_lora: param trees differ (missing {missing[:3]}, extra {extra[:3]})")
    return traverse_util.unflatten_dict({p: (v if _is_lora_leaf(p) else fp[p]) for p, v in fl.items()})


def merge_lora(params, config: Config):
    """Plain-network params equal in function to LoRA params: kernel + (alpha/r) A B,
    adapters dropped. Works on stacked (seed-batched) params (batched matmul)."""
    flat = traverse_util.flatten_dict(params)
    scale = config.lora_alpha / config.lora_rank
    out = {}
    for p, v in flat.items():
        if _is_lora_leaf(p):
            continue
        if str(p[-1]) == "kernel" and p[:-1] + ("lora_a",) in flat:
            v = v + scale * jnp.matmul(flat[p[:-1] + ("lora_a",)], flat[p[:-1] + ("lora_b",)])
        out[p] = v
    return traverse_util.unflatten_dict(out)


def make_optimizer(config: Config, num_updates: int):
    num_grad_steps = num_updates * config.update_epochs * config.num_minibatches
    if config.anneal_lr:
        lr_schedule = optax.linear_schedule(config.lr, 0.0, num_grad_steps)
    else:
        lr_schedule = optax.constant_schedule(config.lr)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(lr_schedule, eps=1e-5),
    )
    return tx, lr_schedule


HEAD_NAMES = ("actor_out", "critic_out", "q_out")  # q_out: PQN's Q head (QNetwork)


def critic_only_mask(params):
    """True for every param that is *not* part of the critic (critic_out, and the
    critic_i tower with separate towers; PQN: q_out). Used for the critic / Q warm-up hook."""
    flat = traverse_util.flatten_dict(params)
    return traverse_util.unflatten_dict(
        {p: not any(str(k).startswith("critic") or str(k) == "q_out" for k in p) for p in flat})


def trunk_mask(params):
    """True for every hidden-layer param (everything except the heads: actor_out /
    critic_out, or PQN's q_out)."""
    flat = traverse_util.flatten_dict(params)
    return traverse_util.unflatten_dict(
        {p: not any(str(k) in HEAD_NAMES for k in p) for p in flat})


def reinit_heads(network, params, key, obs_shape):
    """Copy of params with the heads freshly initialised (same init as a new network):
    actor_out and critic_out (policy head gain 0.01 -> near-uniform policy; value head
    gain 1), or PQN's q_out."""
    fresh = network.init(key, jnp.zeros(obs_shape))
    inner = dict(params["params"])
    for k in HEAD_NAMES:
        if k in fresh["params"]:
            inner[k] = fresh["params"][k]
    return {**params, "params": inner}


def initial_entropy(network, params, obs):
    """Mean policy entropy over a batch of observations (logged at the start of every run)."""
    logits, _, _ = network.apply(params, obs)
    return jnp.mean(categorical_entropy(logits))


def build_update_step(env, network, config: Config, tx, lr_schedule, teacher_params=None,
                      kickstart_scale=None):
    """One PPO update (rollout, GAE, epochs x minibatches, logging) as
    update_step(runner, env_params) -> (runner, metrics). Shared by make_train and
    the lifelong stream trainer (lifelong.py), which passes the current task's params."""
    switch_updates = jnp.asarray(config.task_switch_updates, jnp.int32)

    def update_step(runner: RunnerState, env_params):
        ts, env_state, obs, ep_stats, key, update_idx, vnorm = runner
        key, k_roll, k_upd = jax.random.split(key, 3)
        extra = {}

        # 0) Task-switch hook: optionally reset Adam's moments before this update.
        if config.reset_adam_on_task_switch and config.task_switch_updates:
            switch = jnp.any(update_idx == switch_updates)
            reset = reset_adam_moments(ts.opt_state, tx.init(ts.params))
            ts = ts.replace(opt_state=jax.tree.map(
                lambda r, o: jnp.where(switch, r, o), reset, ts.opt_state))
            extra["adam_reset"] = switch.astype(jnp.float32)

        # 1) Collect experience with the current policy.
        traj, env_state, obs, ep_stats = rollout(
            env, env_params, network, config, ts.params,
            env_state, obs, ep_stats, k_roll, vnorm,
        )
        # 2) Advantages. last_value = V(s_T) for the obs after the rollout.
        _, last_value, _ = network.apply(ts.params, obs)
        if config.normalize_value:
            last_value = vnorm.denormalize(last_value)
        advantages, returns = compute_gae(
            traj.reward, traj.value, traj.done, last_value,
            config.gamma, config.gae_lambda,
        )
        # 3) Flatten (T, N, ...) -> (T*N, ...) and run the PPO update.
        batch = Batch(traj.obs, traj.action, traj.logp, traj.value,
                      advantages, returns)
        batch = jax.tree.map(
            lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
        )
        ev = explained_variance(batch.value, batch.returns)  # raw units
        if config.normalize_value:
            # Update the return stats with this batch, then express the targets
            # and the old values (for clipping) in the new normalised units.
            vnorm = vnorm.update(batch.returns, config.value_norm_beta)
            batch = batch._replace(value=vnorm.normalize(batch.value),
                                   returns=vnorm.normalize(batch.returns))
            extra["value_norm_mean"], extra["value_norm_std"] = vnorm.stats()
        freeze = None
        if config.critic_warmup_updates > 0:
            warm = update_idx < config.critic_warmup_updates
            freeze = jax.tree.map(lambda m: jnp.logical_and(m, warm), critic_only_mask(ts.params))
            extra["critic_warmup"] = warm.astype(jnp.float32)
        if config.freeze_trunk:
            tm = jax.tree.map(jnp.asarray, trunk_mask(ts.params))
            freeze = tm if freeze is None else jax.tree.map(jnp.logical_or, freeze, tm)
        if config.lora_rank:  # base trunk frozen; adapters + heads train
            lm = jax.tree.map(jnp.asarray, lora_freeze_mask(ts.params))
            freeze = lm if freeze is None else jax.tree.map(jnp.logical_or, freeze, lm)
        if config.kickstart_coef > 0 and kickstart_scale is not None:
            # 1 iff the gated KL can act this update: coefficient > 0 and warm-up over
            # (lifelong kickstart_effective_updates). Only on the gated (stream) path.
            on = config.kickstart_coef * jnp.asarray(kickstart_scale, jnp.float32) > 0
            on = jnp.logical_and(on, update_idx >= config.critic_warmup_updates)
            extra["kickstart_effective"] = on.astype(jnp.float32)
        lr_now = lr_schedule(ts.step)  # lr at the start of this update
        ts, loss_m = ppo_update(ts, batch, k_upd, config, teacher_params, freeze, kickstart_scale)

        # 4) Logging. Losses are averaged over all epochs x minibatches.
        health = network_health(ts.params, network.apply, batch.obs,
                                config.dormant_tau, config.activation, lora_scale_of(config))
        n_done = traj.done.sum()
        metrics = {
            "episode_return": ep_stats.last_return.mean(),
            "episode_length": ep_stats.last_length.mean(),
            # Mean return of the episodes that finished during this rollout (NaN if none).
            "rollout_return": jnp.where(n_done > 0, traj.finished_return.sum()
                                        / jnp.maximum(n_done, 1.0), jnp.nan),
            "explained_variance": ev,
            "lr": lr_now,
            # Sanity signal: before any gradient step, ratio should be 1, so
            # KL on the first minibatch of epoch 0 should be ~0.
            "approx_kl_first_mb": loss_m["approx_kl"][0, 0],
            **{k: v.mean() for k, v in loss_m.items()},
            **health,
            **extra,
        }
        return RunnerState(ts, env_state, obs, ep_stats, key,
                           update_idx + 1, vnorm), metrics

    return update_step


def handover_opt_state(config: Config, train_state, init_opt_state):
    """Adam state at the start of RL: fresh unless adam_reset_at_handover=False."""
    if config.adam_reset_at_handover or init_opt_state is None:
        if not config.adam_reset_at_handover:
            raise ValueError("adam_reset_at_handover=False needs init_opt_state")
        return train_state
    return train_state.replace(opt_state=init_opt_state)


def make_train(config: Config, task_params=None, init_params=None, teacher_params=None,
               init_opt_state=None):
    """Build `train(key) -> outputs`. Wrap it as jax.jit(jax.vmap(train)) to
    train several seeds in one program.

    task_params: optional stacked env-params pytree for the eval hook. The
    default is a single task, the env's default params.
    init_params: optional params to start from (e.g. a BC-pretrained policy) instead
    of a fresh init. The init key is still drawn, so the env/rollout RNG stream is the
    same as a from-scratch run with the same seed.
    teacher_params: optional params for the kickstarting term (config.kickstart_coef).
    init_opt_state: optimiser state from pretraining; used only if
    config.adam_reset_at_handover is False.
    """
    env, env_params, network, obs_shape = make_env_and_network(config)
    if task_params is None:
        task_params = jax.tree.map(lambda x: jnp.asarray(x)[None], env_params)
    max_eval_steps = config.eval_max_steps or int(
        getattr(env_params, "max_steps_in_episode", 1000)
    )
    if config.algo != "ppo":
        raise ValueError("make_train is PPO only; PQN runs through lifelong.py / cells.py")
    tx, lr_schedule = make_optimizer(config, config.num_updates)
    update_step = build_update_step(env, network, config, tx, lr_schedule, teacher_params)

    def train(key):
        TRACE_COUNT["train"] += 1  # Python side effect, so it runs only while tracing

        key, k_init, k_reset = jax.random.split(key, 3)
        params = network.init(k_init, jnp.zeros(obs_shape))
        if init_params is not None:
            params = init_params
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        train_state = handover_opt_state(config, train_state, init_opt_state)
        obs, env_state = jax.vmap(env_reset, in_axes=(None, 0, None))(
            env, jax.random.split(k_reset, config.num_envs), env_params
        )
        init_ent = initial_entropy(network, params, obs)
        runner = RunnerState(train_state, env_state, obs,
                             EpisodeStats.zeros(config.num_envs), key,
                             jnp.zeros((), jnp.int32), ValueNorm.init())

        def train_block(runner: RunnerState, _):
            # eval_every updates, then one evaluation. Nesting the scans (rather
            # than using lax.cond on the update index) keeps eval cheap under
            # vmap, where cond would be lowered to "run both branches".
            runner, metrics = jax.lax.scan(lambda r, _: update_step(r, env_params), runner,
                                           None, length=config.eval_every)
            key, k_eval = jax.random.split(runner.key)
            runner = runner._replace(key=key)
            eval_returns = evaluate(env, network, config, runner.train_state.params,
                                    task_params, k_eval, max_eval_steps)
            return runner, (metrics, eval_returns)

        runner, (metrics, eval_returns) = jax.lax.scan(
            train_block, runner, None, length=config.num_evals
        )
        # (num_evals, eval_every) -> (num_updates,)
        metrics = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), metrics)
        return {"runner_state": runner, "metrics": metrics,
                "eval_returns": eval_returns,  # (num_evals, num_tasks)
                "init_entropy": init_ent}  # policy entropy on the initial obs batch

    return train


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--env", default="CartPole-v1")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--preset", default="default",
                   help=f"one of {sorted(PRESETS)}, or several comma-separated (later wins)")
    p.add_argument("--num-envs", type=int)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0,
                   help="seed keys = split(PRNGKey(seed), num_seeds), unless --seeds is given")
    p.add_argument("--seeds", help='explicit seed list, e.g. "100-109": one PRNGKey(s) per seed')
    p.add_argument("--out", default="results/cartpole.npz")
    # Optional overrides (None -> Config default); used e.g. to match PureJaxRL's MinAtar config.
    p.add_argument("--num-steps", type=int)
    p.add_argument("--num-minibatches", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--activation", choices=["tanh", "relu"])
    p.add_argument("--hidden-size", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--no-truncation-bootstrap", action="store_true",
                   help="treat time-limit endings as terminal (PureJaxRL behaviour)")
    B = argparse.BooleanOptionalAction
    p.add_argument("--anneal-lr", action=B)
    p.add_argument("--clip-value", action=B)
    p.add_argument("--shared-trunk", action=B)
    p.add_argument("--normalize-value", action=B)
    p.add_argument("--reset-adam-on-task-switch", action=B)
    p.add_argument("--task-switch-updates", help='update indices, e.g. "100,200"')
    p.add_argument("--dormant-tau", type=float)
    p.add_argument("--matmul-precision", choices=MATMUL_PRECISIONS, default="default",
                   help="jax_default_matmul_precision; recorded in the npz")
    p.add_argument("--allow-dirty", action="store_true",
                   help="run from a dirty tracked tree (no reserved seeds); recorded")
    args = p.parse_args()
    set_matmul_precision(args.matmul_precision)
    import provenance
    prov = provenance.check_clean(parse_seeds(args.seeds) if args.seeds else [],
                                  args.allow_dirty, require_dev=False)

    overrides = resolve_presets(args.preset)
    overrides.update({k: v for k, v in dict(
        num_envs=args.num_envs, num_steps=args.num_steps, num_minibatches=args.num_minibatches, lr=args.lr,
        activation=args.activation, hidden_size=args.hidden_size,
        eval_every=args.eval_every, anneal_lr=args.anneal_lr, clip_value=args.clip_value,
        shared_trunk=args.shared_trunk, normalize_value=args.normalize_value,
        reset_adam_on_task_switch=args.reset_adam_on_task_switch,
        dormant_tau=args.dormant_tau).items() if v is not None})
    if args.task_switch_updates:
        overrides["task_switch_updates"] = tuple(int(x) for x in args.task_switch_updates.split(","))
    if args.no_truncation_bootstrap:
        overrides["bootstrap_on_truncation"] = False
    config = Config(**{"env_name": args.env, "total_timesteps": args.total_timesteps, **overrides})
    print(f"config: {config}")

    if args.seeds:
        seed_list = parse_seeds(args.seeds)
        _refuse_reserved_seeds_under_pytest(seed_list)
        keys = jnp.stack([jax.random.PRNGKey(sd) for sd in seed_list])
        args.num_seeds = len(seed_list)
    else:
        seed_list = None
        keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_seeds)
    print(f"num_updates={config.num_updates} batch={config.batch_size} "
          f"minibatch={config.minibatch_size} seeds={seed_list or args.num_seeds}")
    train_fn = jax.jit(jax.vmap(make_train(config)))

    t0 = time.perf_counter()
    compiled = train_fn.lower(keys).compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = jax.block_until_ready(compiled(keys))
    run_s = time.perf_counter() - t0

    # Training env steps only (eval steps excluded), summed over all seeds.
    env_steps = args.num_seeds * config.num_updates * config.batch_size
    sps = env_steps / run_s
    print(f"compile {compile_s:.1f}s | run {run_s:.1f}s | {sps:,.0f} env-steps/s "
          f"(training steps, all {args.num_seeds} seeds, {jax.devices()[0].platform})")

    m = jax.tree.map(np.asarray, out["metrics"])
    final = m["episode_return"][:, -10:].mean(axis=1)
    print("final return per seed (mean of last 10 updates):", np.round(final, 1))
    print("final eval return per seed:", np.round(np.asarray(out["eval_returns"])[:, -1, 0], 1))

    steps_axis = (np.arange(config.num_updates) + 1) * config.batch_size
    eval_steps_axis = (np.arange(config.num_evals) + 1) * config.eval_every * config.batch_size
    np.savez(
        args.out,
        env_steps=steps_axis,
        eval_env_steps=eval_steps_axis,
        eval_returns=np.asarray(out["eval_returns"]),
        steps_per_sec=sps,
        init_entropy=np.asarray(out["init_entropy"]),
        compile_seconds=compile_s,
        run_seconds=run_s,
        platform=jax.devices()[0].platform,
        config=json.dumps(config_record(config)),
        seeds=np.array(seed_list if seed_list else [-1]),
        preset=args.preset,
        **prov,
        **device_info(),
        **{f"metric_{k}": v for k, v in m.items()},
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
