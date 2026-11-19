"""Tests for the lifelong-preset options. Run with: pytest -q"""

import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ppo  # noqa: E402

# 4 updates of 4 envs x 128 steps; evals after updates 2 and 4.
SMALL = ppo.Config(total_timesteps=4 * 128 * 4, eval_every=2, eval_episodes=2,
                   eval_max_steps=20)


def run(cfg, seed=0):
    out = jax.jit(ppo.make_train(cfg))(jax.random.PRNGKey(seed))
    return jax.tree.map(np.asarray, out)


def init(cfg):
    env, env_params, net, obs_shape = ppo.make_env_and_network(cfg)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape))
    return net, params, obs_shape


# ---------------------------------------------------------------- presets
def test_lifelong_preset_values_and_composition():
    life = ppo.PRESETS["lifelong"]
    assert life["anneal_lr"] is False
    assert life["reset_adam_on_task_switch"] is False
    assert life["activation"] == "relu"
    assert life["clip_value"] is False
    assert life["shared_trunk"] is True
    assert life["normalize_value"] is False
    c = ppo.resolve_presets("purejaxrl_match,lifelong")
    assert c["num_envs"] == 64 and c["lr"] == 5e-3 and c["shared_trunk"] is True
    assert ppo.resolve_presets("default") == {}
    # Defaults are unchanged (bit-identity with earlier runs depends on this).
    d = ppo.Config()
    assert (d.anneal_lr, d.clip_value, d.shared_trunk, d.normalize_value,
            d.reset_adam_on_task_switch, d.activation) == (True, True, False, False, False, "tanh")
    ppo.Config(**ppo.resolve_presets("purejaxrl_match,lifelong"))  # constructs


# ---------------------------------------------------------------- anneal_lr
def test_anneal_lr_flag():
    on = run(SMALL)["metrics"]["lr"]
    off = run(dataclasses.replace(SMALL, anneal_lr=False))["metrics"]["lr"]
    assert np.all(np.diff(on) < 0) and on[0] == pytest.approx(SMALL.lr)
    np.testing.assert_array_equal(off, np.full_like(off, SMALL.lr))


# ---------------------------------------------------------------- activation
@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_activation(act):
    cfg = dataclasses.replace(SMALL, activation=act)
    net, params, obs_shape = init(cfg)
    obs = jax.random.normal(jax.random.PRNGKey(1), (64,) + obs_shape)
    _, _, hiddens = net.apply(params, obs)
    for h in hiddens.values():
        if act == "relu":
            assert float(h.min()) >= 0.0 and float(h.max()) > 1.0
        else:
            assert float(jnp.abs(h).max()) <= 1.0 and float(h.min()) < 0.0
    m = ppo.network_health(params, net.apply, obs, 0.1, act)
    assert np.isnan(float(m["saturated_frac"])) == (act == "relu")


# ---------------------------------------------------------------- clip_value
def test_clip_value_flag():
    cfg = SMALL
    net, params, obs_shape = init(cfg)
    n = 32
    obs = jax.random.normal(jax.random.PRNGKey(1), (n,) + obs_shape)
    logits, value, _ = net.apply(params, obs)
    action = jnp.zeros(n, jnp.int32)
    logp = ppo.categorical_log_prob(logits, action)
    # Old values far from the current ones: clipping changes the loss.
    batch = ppo.Batch(obs, action, logp, value + 5.0, jnp.ones(n), value + 3.0)
    _, m_clip = ppo.ppo_loss(params, net.apply, batch, cfg)
    _, m_noclip = ppo.ppo_loss(params, net.apply, batch, dataclasses.replace(cfg, clip_value=False))
    assert float(m_noclip["value_loss"]) == pytest.approx(0.5 * 9.0, rel=1e-5)
    # The clipped prediction sits at old - 0.2 = value + 4.8, error 1.8, and max(9, 3.24) = 9.
    # Construct a case where clipping is pessimistic: returns between current and old value.
    batch2 = batch._replace(returns=value + 4.0)
    _, c2 = ppo.ppo_loss(params, net.apply, batch2, cfg)
    _, n2 = ppo.ppo_loss(params, net.apply, batch2, dataclasses.replace(cfg, clip_value=False))
    assert float(n2["value_loss"]) == pytest.approx(0.5 * 16.0, rel=1e-5)
    assert float(c2["value_loss"]) == pytest.approx(0.5 * 16.0, rel=1e-5)  # max picks unclipped
    batch3 = batch._replace(returns=value + 6.0)
    _, c3 = ppo.ppo_loss(params, net.apply, batch3, cfg)
    _, n3 = ppo.ppo_loss(params, net.apply, batch3, dataclasses.replace(cfg, clip_value=False))
    assert float(n3["value_loss"]) == pytest.approx(0.5 * 36.0, rel=1e-5)
    assert float(c3["value_loss"]) == pytest.approx(0.5 * 36.0, rel=1e-5)
    # Clipping bites when the unclipped error is SMALLER than the clipped one:
    batch4 = batch._replace(returns=value + 0.1)
    _, c4 = ppo.ppo_loss(params, net.apply, batch4, cfg)
    _, n4 = ppo.ppo_loss(params, net.apply, batch4, dataclasses.replace(cfg, clip_value=False))
    assert float(n4["value_loss"]) == pytest.approx(0.5 * 0.01, rel=1e-4)
    assert float(c4["value_loss"]) == pytest.approx(0.5 * 4.7**2, rel=1e-4)
    # And the gradient into the value head vanishes when the clipped term is active.
    g = jax.grad(lambda p: ppo.ppo_loss(p, net.apply, batch4, cfg)[1]["value_loss"])(params)
    assert float(jnp.abs(g["params"]["critic_out"]["kernel"]).max()) == 0.0


# ---------------------------------------------------------------- shared_trunk
def test_shared_trunk_param_tree_and_gradients():
    cfg = dataclasses.replace(SMALL, shared_trunk=True)
    net, params, obs_shape = init(cfg)
    p = params["params"]
    assert set(p) == {"trunk", "actor_out", "critic_out"}
    assert set(p["trunk"]) == {f"layer_{i}" for i in range(cfg.num_layers)}
    obs = jax.random.normal(jax.random.PRNGKey(1), (16,) + obs_shape)
    logits, value, hiddens = net.apply(params, obs)
    assert list(hiddens) == [f"trunk_{i}" for i in range(cfg.num_layers)]
    # Both the actor head and the critic head send gradient into the trunk.
    g_pi = jax.grad(lambda q: net.apply(q, obs)[0].sum())(params)["params"]
    g_v = jax.grad(lambda q: net.apply(q, obs)[1].sum())(params)["params"]
    assert float(jnp.abs(g_pi["trunk"]["layer_0"]["kernel"]).max()) > 0
    assert float(jnp.abs(g_v["trunk"]["layer_0"]["kernel"]).max()) > 0
    assert float(jnp.abs(g_pi["critic_out"]["kernel"]).max()) == 0
    # Separate towers keep their original names (bit-identity with old runs).
    _, sep, _ = init(SMALL)
    assert set(sep["params"]) == {"actor_0", "actor_1", "actor_out",
                                  "critic_0", "critic_1", "critic_out"}
    out = run(cfg)
    assert np.all(np.isfinite(out["metrics"]["loss"]))


def test_per_layer_health_keys():
    for shared, layers in [(False, ["actor_0", "actor_1", "critic_0", "critic_1"]),
                           (True, ["trunk_0", "trunk_1"])]:
        m = run(dataclasses.replace(SMALL, shared_trunk=shared))["metrics"]
        for name in layers:
            assert f"dormant_frac.{name}" in m and f"saturated_frac.{name}" in m
        wn = [k for k in m if k.startswith("weight_norm.")]
        assert ("weight_norm.trunk.layer_0" in wn) == shared
        assert "weight_norm.actor_out" in wn and "weight_norm.critic_out" in wn
        # The aggregate is the unit-weighted mean of the per-layer fractions.
        np.testing.assert_allclose(m["dormant_frac"],
                                   np.mean([m[f"dormant_frac.{n}"] for n in layers], 0), atol=1e-6)


def test_saturation_definition_matches_rules():
    # Unit 0 is saturated on 100% of samples; unit 1 on 98% (not saturated);
    # unit 2 has mean |h| = 0.995 but dips below 0.99 on 10% of samples (not
    # saturated under the rule, although the old mean-based metric counted it); unit 3 is 0.
    n = 100
    u0 = jnp.ones(n)
    u1 = jnp.where(jnp.arange(n) < 98, 1.0, 0.0)
    u2 = jnp.where(jnp.arange(n) < 90, 1.0, 0.95)
    h = jnp.stack([u0, u1, u2, jnp.zeros(n)], -1)
    m = ppo.network_health({"w": jnp.ones(1)}, lambda p, o: (None, None, {"l": h}),
                           jnp.zeros((n, 1)), 0.025, "tanh")
    assert float(m["saturated_frac.l"]) == pytest.approx(0.25)
    assert float(m["dormant_frac.l"]) == pytest.approx(0.25)


# ---------------------------------------------------------------- Adam reset
def test_reset_adam_moments_unit():
    params = {"w": jnp.ones(3)}
    sched = optax.linear_schedule(1e-3, 0.0, 100)
    tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(sched, eps=1e-5))
    st = tx.init(params)
    for _ in range(5):
        _, st = tx.update({"w": jnp.array([1.0, -2.0, 3.0])}, st, params)
    fresh = tx.init(params)
    reset = ppo.reset_adam_moments(st, fresh)
    adam = [s for s in jax.tree.leaves(reset, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState))
            if isinstance(s, optax.ScaleByAdamState)]
    assert len(adam) == 1
    assert int(adam[0].count) == 0
    assert float(jnp.abs(adam[0].mu["w"]).max()) == 0 and float(jnp.abs(adam[0].nu["w"]).max()) == 0
    # The LR-schedule counter is NOT reset.
    sched_counts = [s.count for s in jax.tree.leaves(reset, is_leaf=lambda x: isinstance(x, optax.ScaleByScheduleState))
                    if isinstance(s, optax.ScaleByScheduleState)]
    assert [int(c) for c in sched_counts] == [5]


def test_reset_adam_on_task_switch_hook():
    base = run(SMALL)["metrics"]
    cfg = dataclasses.replace(SMALL, reset_adam_on_task_switch=True, task_switch_updates=(2,))
    m = run(cfg)["metrics"]
    np.testing.assert_array_equal(m["adam_reset"], [0, 0, 1, 0])
    # Updates before the switch are bit-identical; the switch update differs.
    for k in ("loss", "value_loss", "policy_loss"):
        np.testing.assert_array_equal(m[k][:2], base[k][:2])
        assert m[k][2] != base[k][2]
    # The LR schedule is unaffected.
    np.testing.assert_array_equal(m["lr"], base["lr"])
    # Flag off: the indices are ignored and the run is bit-identical to base.
    off = run(dataclasses.replace(SMALL, task_switch_updates=(2,)))["metrics"]
    for k in base:
        np.testing.assert_array_equal(off[k], base[k])


def test_reset_adam_first_step_is_sign_like():
    """After a reset, the first Adam step is lr * g / (|g| + eps) element-wise
    (bias-corrected, count restarted), the same as the very first step of training."""
    params = {"w": jnp.zeros(2)}
    tx = optax.adam(1e-2, eps=1e-5)
    st = tx.init(params)
    for g in ([10.0, 0.1], [10.0, 0.1], [10.0, 0.1]):
        _, st = tx.update({"w": jnp.array(g)}, st, params)
    st = ppo.reset_adam_moments(st, tx.init(params))
    upd, _ = tx.update({"w": jnp.array([-3.0, 0.5])}, st, params)
    np.testing.assert_allclose(upd["w"], [1e-2, -1e-2], rtol=1e-4)


# ---------------------------------------------------------------- normalize_value
def test_value_norm_unit():
    vn = ppo.ValueNorm.init()
    m, s = vn.stats()
    assert float(m) == 0.0 and float(s) == 1.0  # identity before any data
    key = jax.random.PRNGKey(0)
    for _ in range(500):
        key, k = jax.random.split(key)
        vn = vn.update(50.0 + 10.0 * jax.random.normal(k, (512,)), 0.99)
    m, s = vn.stats()
    assert float(m) == pytest.approx(50.0, abs=0.5) and float(s) == pytest.approx(10.0, rel=0.05)
    x = jnp.array([-3.0, 0.0, 70.0])
    np.testing.assert_allclose(vn.denormalize(vn.normalize(x)), x, rtol=1e-5, atol=1e-4)
    # Debiasing: after ONE update the stats are those of that batch, not shrunk toward 0.
    one = ppo.ValueNorm.init().update(jnp.array([100.0, 100.0]), 0.99)
    assert float(one.stats()[0]) == pytest.approx(100.0, rel=1e-4)


def test_normalize_value_in_training():
    cfg = dataclasses.replace(SMALL, normalize_value=True)
    out = run(cfg)
    m = out["metrics"]
    assert np.all(np.isfinite(m["loss"]))
    # CartPole returns are O(10-100) and must be learned in normalised units.
    assert np.all(m["value_norm_mean"] > 1.0) and np.all(m["value_norm_std"] > 0.1)
    # The value loss is in normalised units, so it is O(1) instead of O(100s).
    base = run(SMALL)["metrics"]
    assert m["value_loss"].mean() < base["value_loss"].mean()
    # Off by default: no value-norm metrics are emitted.
    assert "value_norm_mean" not in base


def test_lifelong_preset_trains_and_compiles_once():
    cfg = dataclasses.replace(SMALL, **ppo.PRESETS["lifelong"])
    cfg = dataclasses.replace(cfg, eval_episodes=2)
    before = ppo.TRACE_COUNT["train"]
    train = jax.jit(jax.vmap(ppo.make_train(cfg)))
    o1 = train(jax.random.split(jax.random.PRNGKey(0), 2))
    o2 = train(jax.random.split(jax.random.PRNGKey(1), 2))
    jax.block_until_ready((o1, o2))
    assert ppo.TRACE_COUNT["train"] - before == 1
    assert float(jnp.abs(o1["metrics"]["approx_kl_first_mb"]).max()) < 1e-6
    np.testing.assert_array_equal(np.asarray(o1["metrics"]["lr"]), np.float32(cfg.lr))
