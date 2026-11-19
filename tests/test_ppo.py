"""Tests for the build-guide checkpoints. Run with: pytest -q"""

import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ppo  # noqa: E402

SMALL = ppo.Config(total_timesteps=4 * 128 * 4, eval_every=2, eval_episodes=2,
                   eval_max_steps=20)


# ---------------------------------------------------------------- GAE (step 5)
def gae_reference(rewards, values, dones, last_value, gamma, lam):
    """Plain-Python GAE, written straight from the definitions."""
    T = len(rewards)
    adv = np.zeros(T)
    next_adv = 0.0
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        next_adv = delta + gamma * lam * nonterminal * next_adv
        adv[t] = next_adv
    return adv, adv + values


def test_gae_matches_numpy_reference_with_mid_episode_done():
    gamma, lam = 0.99, 0.95
    rewards = np.array([1.0, 0.5, 2.0, -1.0, 0.3, 1.5])
    values = np.array([0.2, 0.4, 1.0, 0.1, -0.3, 0.7])
    dones = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # episode ends at t=2
    last_value = 0.9

    adv_ref, ret_ref = gae_reference(rewards, values, dones, last_value, gamma, lam)
    # (T,) -> (T, N=1) for the JAX version
    adv, ret = ppo.compute_gae(jnp.array(rewards)[:, None], jnp.array(values)[:, None],
                               jnp.array(dones)[:, None], jnp.array([last_value]),
                               gamma, lam)
    np.testing.assert_allclose(np.asarray(adv[:, 0]), adv_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(ret[:, 0]), ret_ref, rtol=1e-5, atol=1e-6)

    # Hand check of the boundary: at t=2 (done) there is no bootstrap, so
    # A_2 = r_2 - V_2 exactly, with nothing leaking back from t=3.
    assert adv[2, 0] == pytest.approx(rewards[2] - values[2], abs=1e-6)


def test_gae_multiple_envs_are_independent():
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    r = jax.random.normal(k1, (7, 3))
    v = jax.random.normal(k2, (7, 3))
    d = (jax.random.uniform(k3, (7, 3)) < 0.3).astype(jnp.float32)
    lv = jnp.array([0.1, -0.2, 0.5])
    adv, _ = ppo.compute_gae(r, v, d, lv, 0.99, 0.95)
    for n in range(3):
        ref, _ = gae_reference(np.asarray(r[:, n]), np.asarray(v[:, n]),
                               np.asarray(d[:, n]), float(lv[n]), 0.99, 0.95)
        np.testing.assert_allclose(np.asarray(adv[:, n]), ref, rtol=1e-5, atol=1e-5)


# ------------------------------------------------------- network (step 3)
def test_initial_entropy_is_log_n_actions():
    env, env_params, net, obs_shape = ppo.make_env_and_network(SMALL)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape))
    obs = jax.random.normal(jax.random.PRNGKey(1), (256,) + obs_shape)
    logits, value, hiddens = net.apply(params, obs)
    ent = ppo.categorical_entropy(logits)
    n_actions = env.action_space(env_params).n
    assert float(ent.mean()) == pytest.approx(math.log(n_actions), abs=1e-3)
    assert value.shape == (256,)
    assert len(hiddens) == 2 * SMALL.num_layers


# ------------------------------------------------- env interface (step 2)
def _collect(key, cfg=SMALL):
    env, env_params, net, obs_shape = ppo.make_env_and_network(cfg)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape))
    k_reset, k_roll = jax.random.split(key)
    obs, state = jax.vmap(ppo.env_reset, in_axes=(None, 0, None))(
        env, jax.random.split(k_reset, cfg.num_envs), env_params)
    fn = jax.jit(lambda p, s, o, e, k: ppo.rollout(env, env_params, net, cfg, p, s, o, e, k))
    return fn(params, state, obs, ppo.EpisodeStats.zeros(cfg.num_envs), k_roll)


def test_rollout_shapes():
    traj, _, obs, _ = _collect(jax.random.PRNGKey(0))
    T, N = SMALL.num_steps, SMALL.num_envs
    assert traj.obs.shape == (T, N, 4)
    for x in (traj.action, traj.logp, traj.value, traj.reward, traj.done):
        assert x.shape == (T, N)
    assert obs.shape == (N, 4)


def test_obs_after_done_is_reset_obs():
    # A random-ish initial policy fails CartPole within ~10-40 steps, so a
    # 128-step rollout always contains episode ends. After each one the next
    # stored obs must be a fresh reset obs (all |components| <= 0.05).
    traj, _, _, _ = _collect(jax.random.PRNGKey(3))
    done = np.asarray(traj.done).astype(bool)
    obs = np.asarray(traj.obs)
    assert done.sum() > 0
    t_idx, n_idx = np.nonzero(done[:-1])
    assert np.all(np.abs(obs[t_idx + 1, n_idx]) <= 0.05 + 1e-6)


def test_truncation_flag():
    """Force a time-limit hit, then check it is flagged as truncation, not termination."""
    env, env_params, _, _ = ppo.make_env_and_network(SMALL)
    _, state = ppo.env_reset(env, jax.random.PRNGKey(0), env_params)
    state = state.replace(time=env_params.max_steps_in_episode - 1)
    out = ppo.env_step(env, jax.random.PRNGKey(1), state, 1, env_params)
    assert bool(out.done) and bool(out.truncated) and not bool(out.terminated)
    # and a real failure is a termination
    _, state = ppo.env_reset(env, jax.random.PRNGKey(0), env_params)
    state = state.replace(theta=1.0)
    out = ppo.env_step(env, jax.random.PRNGKey(1), state, 1, env_params)
    assert bool(out.done) and bool(out.terminated) and not bool(out.truncated)


# --------------------------------------------- determinism (steps 1 / 4)
def test_same_key_identical_rollout_different_key_differs():
    a = _collect(jax.random.PRNGKey(42))[0]
    b = _collect(jax.random.PRNGKey(42))[0]
    c = _collect(jax.random.PRNGKey(43))[0]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert not np.array_equal(np.asarray(a.action), np.asarray(c.action))


def test_envs_are_not_correlated():
    traj = _collect(jax.random.PRNGKey(0))[0]
    acts = np.asarray(traj.action)
    assert not all(np.array_equal(acts[:, 0], acts[:, n]) for n in range(1, SMALL.num_envs))


# --------------------------------------------------------- PPO loss (step 6)
def _first_batch():
    cfg = SMALL
    env, env_params, net, obs_shape = ppo.make_env_and_network(cfg)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape))
    traj, _, last_obs, _ = _collect(jax.random.PRNGKey(7))
    _, last_v, _ = net.apply(params, last_obs)
    adv, ret = ppo.compute_gae(traj.reward, traj.value, traj.done, last_v,
                               cfg.gamma, cfg.gae_lambda)
    batch = ppo.Batch(traj.obs, traj.action, traj.logp, traj.value, adv, ret)
    batch = jax.tree.map(lambda x: x.reshape((cfg.batch_size,) + x.shape[2:]), batch)
    return cfg, net, params, batch


def test_ratio_is_one_and_kl_zero_before_any_update():
    cfg, net, params, batch = _first_batch()
    # Exact check: logp_old from the same network on the same batch gives ratio == 1 exactly.
    logits, value, _ = net.apply(params, batch.obs)
    exact = batch._replace(logp=ppo.categorical_log_prob(logits, batch.action), value=value)
    _, m = ppo.ppo_loss(params, net.apply, exact, cfg)
    assert float(m["ratio_mean"]) == 1.0
    assert float(m["approx_kl"]) == 0.0
    assert float(m["clip_frac"]) == 0.0
    # With logp stored at rollout time (computed on (N,) batches, not (T*N,)),
    # results agree up to float rounding.
    _, m = ppo.ppo_loss(params, net.apply, batch, cfg)
    assert abs(float(m["ratio_mean"]) - 1.0) < 1e-6
    assert abs(float(m["approx_kl"])) < 1e-7


def test_loss_gradient_does_not_flow_into_targets():
    cfg, net, params, batch = _first_batch()
    def loss_wrt_targets(adv, ret):
        return ppo.ppo_loss(params, net.apply, batch._replace(advantages=adv, returns=ret), cfg)[0]
    g_adv, g_ret = jax.grad(loss_wrt_targets, argnums=(0, 1))(batch.advantages, batch.returns)
    assert float(jnp.abs(g_adv).max()) == 0.0
    assert float(jnp.abs(g_ret).max()) == 0.0


# ------------------------------------------------- whole program (step 8)
def test_jit_compiles_once_and_first_minibatch_kl_is_zero():
    before = ppo.TRACE_COUNT["train"]
    train = jax.jit(jax.vmap(ppo.make_train(SMALL)))
    out1 = train(jax.random.split(jax.random.PRNGKey(0), 2))
    out2 = train(jax.random.split(jax.random.PRNGKey(1), 2))
    jax.block_until_ready((out1, out2))
    assert ppo.TRACE_COUNT["train"] - before == 1
    assert out1["metrics"]["episode_return"].shape == (2, SMALL.num_updates)
    assert out1["eval_returns"].shape == (2, SMALL.num_evals, 1)
    assert float(jnp.abs(out1["metrics"]["approx_kl_first_mb"]).max()) < 1e-6


def test_dormant_metric():
    # Hand-built hidden layer: unit 0 always 0 (dormant), units 1-3 active.
    class Fake:
        @staticmethod
        def apply(params, obs):
            h = jnp.stack([jnp.zeros(obs.shape[0]), jnp.ones(obs.shape[0]),
                           -jnp.ones(obs.shape[0]) * 0.5, jnp.ones(obs.shape[0]) * 0.8], -1)
            return None, None, {"layer": h}
    m = ppo.network_health({"w": jnp.ones(4)}, Fake.apply, jnp.zeros((10, 4)), 0.1)
    assert float(m["dormant_frac"]) == pytest.approx(0.25)
    assert float(m["saturated_frac"]) == pytest.approx(0.25)  # the |h|=1 unit
    assert float(m["weight_norm"]) == pytest.approx(2.0)


def test_eval_hook_supports_multiple_tasks():
    """The eval hook takes a stack of env params, one per task, ready for the task stream."""
    import gymnax
    _, p = gymnax.make("CartPole-v1")
    tasks = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]),
                         p, p.replace(length=1.0), p.replace(force_mag=5.0))
    train = jax.jit(ppo.make_train(SMALL, task_params=tasks))
    out = train(jax.random.PRNGKey(0))
    assert out["eval_returns"].shape == (SMALL.num_evals, 3)
