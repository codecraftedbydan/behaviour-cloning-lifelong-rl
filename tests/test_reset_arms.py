"""Stage-1 step 3: the scratch reset arms R-H ("heads") and R-HOV ("heads_adam_vn") with an
oracle switch signal at every block boundary from block 2 on. Dev seeds
900+ only."""
import dataclasses
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lifelong  # noqa: E402
import ppo  # noqa: E402
import stream as S  # noqa: E402

SCFG = S.StreamConfig(num_new_tasks=7, revisit_every=3, block_updates=2)
TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=64, hidden_size=16,
                         eval_episodes=2))
HEADS = ("actor_out", "critic_out")


def _flat(t):
    return traverse_util.flatten_dict(jax.tree.map(np.asarray, t)) if isinstance(t, dict) else \
        {i: np.asarray(x) for i, x in enumerate(jax.tree.leaves(t))}


def _tree_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return jax.tree.structure(a) == jax.tree.structure(b) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def _is_head(path):
    return any(str(k) in HEADS for k in path)


@pytest.fixture(scope="module")
def state():
    """A mid-training state: Adam moments (incl. the heads') non-zero, count > 0, value norm
    updated."""
    env, _, network, obs_shape = ppo.make_env_and_network(TINY)
    tx, _ = ppo.make_optimizer(TINY, 10)
    params = network.init(jax.random.PRNGKey(900), jnp.zeros(obs_shape))
    ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    for i in range(3):
        g = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(i), x.shape), ts.params)
        ts = ts.apply_gradients(grads=g)
    vn = ppo.ValueNorm.init().update(jnp.arange(10.0), TINY.value_norm_beta)
    return network, obs_shape, tx, ts, vn


# ------------------------------------------------------------------ reset functions
def test_rh_reset_changes_only_heads_and_carries_adam_and_value_norm(state):
    network, obs_shape, tx, ts, vn = state
    key = lifelong.reset_key(jax.random.PRNGKey(900), 3)
    ts2, vn2 = lifelong.switch_reset("heads", network, tx, obs_shape, ts, vn, key)
    f0, f1 = _flat(ts.params), _flat(ts2.params)
    for k in f0:
        if _is_head(k):
            assert not np.array_equal(f0[k], f1[k]) or "bias" in str(k), k  # biases init at 0
        else:
            assert np.array_equal(f0[k], f1[k]), k            # trunk bit-identical
    # heads = a fresh draw from the reset key
    fresh = network.init(key, jnp.zeros(obs_shape))["params"]
    for h in HEADS:
        assert _tree_equal(ts2.params["params"][h], fresh[h])
    # the WHOLE optimiser state carries over bit for bit, including the head moments
    assert _tree_equal(ts2.opt_state, ts.opt_state)
    mu = [s for s in jax.tree.leaves(ts.opt_state, is_leaf=lambda x: hasattr(x, "mu")) if hasattr(s, "mu")]
    assert mu and all(np.abs(np.asarray(m["params"]["actor_out"]["kernel"])).sum() > 0 for m in
                      [mu[0].mu, mu[0].nu])                   # head moments are non-trivial
    assert int(ts2.step) == int(ts.step)
    assert _tree_equal(vn2, vn) and float(vn.debias) > 0     # value norm carried over


def test_rhov_reset_heads_plus_fresh_adam_and_value_norm(state):
    network, obs_shape, tx, ts, vn = state
    key = lifelong.reset_key(jax.random.PRNGKey(900), 3)
    ts2, vn2 = lifelong.switch_reset("heads_adam_vn", network, tx, obs_shape, ts, vn, key)
    f0, f1 = _flat(ts.params), _flat(ts2.params)
    for k in f0:
        if not _is_head(k):
            assert np.array_equal(f0[k], f1[k]), k
    fresh = network.init(key, jnp.zeros(obs_shape))["params"]
    for h in HEADS:
        assert _tree_equal(ts2.params["params"][h], fresh[h])
    assert _tree_equal(ts2.opt_state, tx.init(ts2.params))   # whole optimiser at init
    assert not _tree_equal(ts2.opt_state, ts.opt_state)
    assert _tree_equal(vn2, ppo.ValueNorm.init())
    assert int(ts2.step) == int(ts.step)


def test_rhov_reset_equals_probe_reset(state, monkeypatch):
    network, obs_shape, tx, ts, vn = state
    key = jax.random.PRNGKey(12)
    ts2, vn2 = lifelong.switch_reset("heads_adam_vn", network, tx, obs_shape, ts, vn, key)
    p, o, v = lifelong.probe_reset(network, tx, ts.params, key, obs_shape)
    assert _tree_equal(ts2.params, p) and _tree_equal(ts2.opt_state, o) and _tree_equal(vn2, v)
    # the probe's training job starts from TrainState.create (+ handover) and ValueNorm.init:
    # exactly probe_reset's optimiser state and value norm for the same params
    pconfig = dataclasses.replace(lifelong.scratch_config(TINY), freeze_trunk=True)
    ptx, _ = ppo.make_optimizer(pconfig, SCFG.block_updates)
    pp, po, pv = lifelong.probe_reset(network, ptx, ts.params, key, obs_shape)
    start = ppo.handover_opt_state(pconfig, TrainState.create(apply_fn=network.apply, params=pp, tx=ptx), None)
    assert _tree_equal(start.opt_state, po) and _tree_equal(ppo.ValueNorm.init(), pv)
    # and run_condition's head probe goes through probe_reset
    calls = []
    real = lifelong.probe_reset
    monkeypatch.setattr(lifelong, "probe_reset", lambda *a: calls.append(1) or real(*a))
    lifelong.run_condition(TINY, dataclasses.replace(SCFG, head_probe_every=3), [900], fresh=False,
                           verbose=False)
    assert calls


def test_reset_key_is_fresh_per_block_and_not_the_init_key():
    run = jax.random.PRNGKey(900)
    k_init = jax.random.split(run, 3)[1]
    ks = [np.asarray(lifelong.reset_key(run, b)) for b in range(1, 6)]
    assert len({k.tobytes() for k in ks}) == 5
    assert all(not np.array_equal(k, np.asarray(k_init)) for k in ks)
    assert lifelong.RESET_TAG != lifelong.HEAD_PROBE_TAG


# ------------------------------------------------------------------ in the stream trainer
@pytest.fixture(scope="module")
def runs():
    return {arm: lifelong.run_condition(TINY, SCFG, [900, 901], fresh=True, verbose=False, reset_arm=arm)
            for arm in lifelong.RESET_ARMS}


def test_reset_at_every_block_from_block2_including_revisits(runs):
    st = S.make_stream(SCFG, 900)
    revisit = np.flatnonzero(~st.is_new.astype(bool))
    assert len(revisit) > 0
    for arm in ("heads", "heads_adam_vn"):
        r = runs[arm]
        flag = np.asarray(r["stream/reset_applied"])
        assert flag.shape == (2, st.num_blocks)
        assert (flag[:, 0] == 0).all() and (flag[:, 1:] == 1).all()
        ent = np.asarray(r["stream/post_reset_entropy"])
        assert np.allclose(ent[:, 1:], np.log(4), atol=1e-2)         # policy head re-initialised
        assert np.allclose(ent[:, revisit], np.log(4), atol=1e-2)    # revisits too
        assert r["reset_arm"] == arm


def test_block1_identical_to_off_and_switch_changes_later_blocks(runs):
    off = runs["none"]
    for arm in ("heads", "heads_adam_vn"):
        r = runs[arm]
        for k in off:
            if k.startswith("stream/metrics/"):
                np.testing.assert_array_equal(np.asarray(off[k])[:, 0], np.asarray(r[k])[:, 0], err_msg=k)
        assert not np.array_equal(np.asarray(off["stream/metrics/entropy"])[:, 1:],
                                  np.asarray(r["stream/metrics/entropy"])[:, 1:])
    # R-H and R-HOV draw the same heads at the first switch, but differ through Adam / value norm
    assert not np.array_equal(np.asarray(runs["heads"]["stream/metrics/value_loss"])[:, 1],
                              np.asarray(runs["heads_adam_vn"]["stream/metrics/value_loss"])[:, 1])


def test_no_reset_within_a_block():
    one = dataclasses.replace(SCFG, num_new_tasks=1, block_updates=4)   # a single block
    a = lifelong.run_condition(TINY, one, [900], fresh=False, verbose=False)
    for arm in ("heads", "heads_adam_vn"):
        b = lifelong.run_condition(TINY, one, [900], fresh=False, verbose=False, reset_arm=arm)
        assert np.asarray(b["stream/reset_applied"]).sum() == 0
        for k in a:
            if k.startswith("stream/") and "seconds" not in k:
                np.testing.assert_array_equal(np.asarray(a[k]), np.asarray(b[k]), err_msg=k)


def test_off_path_unchanged_and_fresh_never_gets_the_arm(runs):
    default = lifelong.run_condition(TINY, SCFG, [900, 901], fresh=True, verbose=False)
    off = runs["none"]
    assert set(default) == set(off) and not any("reset" in k for k in off)
    for k in default:
        x, y = np.asarray(default[k]), np.asarray(off[k])
        if "seconds" in k:
            continue
        if x.dtype.kind in "fc":
            assert np.array_equal(x, y, equal_nan=True), k
        else:
            assert str(x) == str(y), k
    for arm in ("heads", "heads_adam_vn"):
        for k in off:
            if k.startswith("fresh/") and "seconds" not in k and np.asarray(off[k]).dtype.kind == "f":
                assert np.array_equal(np.asarray(off[k]), np.asarray(runs[arm][k]), equal_nan=True), k


def test_reset_arms_refused_outside_scratch():
    p = ppo.make_env_and_network(TINY)[2].init(jax.random.PRNGKey(0), jnp.zeros(TINY_OBS()))
    for kw in (dict(init_params=p), dict(teacher_params=p)):
        with pytest.raises(ValueError, match="scratch"):
            lifelong.make_stream_train(TINY, SCFG, 3, reset_arm="heads", **kw)
    for c in (dataclasses.replace(TINY, critic_warmup_updates=1), dataclasses.replace(TINY, freeze_trunk=True),
              dataclasses.replace(TINY, kickstart_coef=0.1)):
        with pytest.raises(ValueError, match="scratch"):
            lifelong.make_stream_train(c, SCFG, 3, reset_arm="heads_adam_vn")
    with pytest.raises(ValueError, match="reset_adam"):
        lifelong.make_stream_train(dataclasses.replace(TINY, reset_adam_on_task_switch=True), SCFG, 3,
                                   reset_arm="heads")
    with pytest.raises(ValueError):
        lifelong.make_stream_train(TINY, SCFG, 3, reset_arm="trunk")


def TINY_OBS():
    return ppo.make_env_and_network(TINY)[3]


def test_cli_records_reset_arm(tmp_path, monkeypatch):
    out = tmp_path / "r.npz"
    monkeypatch.setattr(sys, "argv", ["lifelong.py", "--seeds", "900-901", "--num-new-tasks", "2",
                                      "--block-updates", "1", "--hidden-size", "16", "--no-fresh",
                                      "--reset-arm", "heads", "--out", str(out)])
    lifelong.main()
    d = np.load(out, allow_pickle=False)
    assert str(d["reset_arm"]) == "heads"
    assert json.loads(str(d["config"]))["hidden_size"] == 16
