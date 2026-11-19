"""Tests for BC pretraining (bc.py) and the lifelong.py BC hooks (retention, warm-up
budget). Stage-1 step 2. Only dev seeds 900+ are ever trained; stage-1 seeds are used only
to build host-side task lists (no training) and to check refusals."""
import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bc  # noqa: E402
import lifelong  # noqa: E402
import ppo  # noqa: E402
import seeds  # noqa: E402
import stream as S  # noqa: E402

SCFG = S.StreamConfig(num_new_tasks=7, revisit_every=3, block_updates=2)
TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=64, hidden_size=16,
                         eval_episodes=2))
SMALL_BC = bc.BCConfig(num_train_tasks=6, num_val_tasks=3, num_retention_tasks=5,
                       episodes_per_task=2, max_epochs=3, patience=2, batch_size=64)


def _flat(p):
    return traverse_util.flatten_dict(jax.tree.map(np.asarray, p))


def _bitwise_equal(a, b):
    fa, fb = _flat(a), _flat(b)
    return fa.keys() == fb.keys() and all(np.array_equal(fa[k], fb[k]) for k in fa)


@pytest.fixture(scope="module")
def dev_streams():
    return [S.make_stream(SCFG, s) for s in (900, 901)]


@pytest.fixture(scope="module")
def split(dev_streams):
    return bc.demo_split(SCFG, dev_streams, SMALL_BC)


# ------------------------------------------------------------------ nits folded into commit B
def test_blocks_text_derived_from_seed_blocks():
    assert seeds.BLOCKS_TEXT == "stream 500-509/700-724/750-759, fresh 600-609/800-824/850-859"
    for stream_block, fresh_block in seeds.SEED_BLOCKS.values():
        assert f"{min(stream_block)}-{max(stream_block)}" in seeds.BLOCKS_TEXT
        assert f"{min(fresh_block)}-{max(fresh_block)}" in seeds.BLOCKS_TEXT
    assert lifelong._BLOCKS_TEXT is seeds.BLOCKS_TEXT


def test_dev_reference_seeds_defined_once_in_seeds():
    assert seeds.DEV_REFERENCE_SEEDS == tuple(range(900, 940))
    assert S.DEV_REFERENCE_SEEDS is seeds.DEV_REFERENCE_SEEDS
    assert S.REFERENCE_SEEDS == tuple(sorted(seeds.PILOT_STREAM_SEEDS)) + seeds.DEV_REFERENCE_SEEDS
    assert min(seeds.DEV_REFERENCE_SEEDS) >= seeds.DEV_MIN_SEED
    assert not set(seeds.DEV_REFERENCE_SEEDS) & set(seeds.RESERVED_SEEDS)


# ------------------------------------------------------------------ demo pool needs streams (fix 3)
def test_demo_split_refuses_missing_streams_for_stage1():
    for streams in (None, []):
        with pytest.raises(ValueError, match="actual streams"):
            bc.demo_split(S.STAGE1_STREAM, streams, SMALL_BC)
        with pytest.raises(ValueError, match="actual streams"):
            bc.pretrain(TINY, S.STAGE1_STREAM, streams, 900, SMALL_BC)


def test_demo_split_refuses_streams_not_in_reference_set():
    # stage-1 seed, but a longer stream than the registered one (host-side tasks only)
    longer = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=41)
    for st in ([S.make_stream(longer, 712)], [S.make_stream(S.PILOT_STREAM, 950)],
               [S.make_stream(S.PILOT_STREAM, 712), S.make_stream(S.PILOT_STREAM, 1500)]):
        assert S.uncovered_streams(st, S.reference_streams())
        with pytest.raises(ValueError):
            bc.demo_split(S.STAGE1_STREAM, st, SMALL_BC)


def test_demo_split_accepts_registered_stage1_and_dev_streams():
    ok = [S.make_stream(S.STAGE1_STREAM, 712), S.make_stream(S.STAGE1_STREAM, 905)]
    tr, va, rt = bc.demo_split(S.STAGE1_STREAM, ok, SMALL_BC)
    assert (len(tr), len(va), len(rt)) == (6, 3, 5)


def test_demo_splits_disjoint_from_each_other_and_from_streams(split, dev_streams):
    tr, va, rt = split
    stage1 = [S.make_stream(S.STAGE1_STREAM, s) for s in (700, 724, 750)]
    stream_tasks = [t for st in [*dev_streams, *stage1] for t in st.task_list]
    for keyf in (S.walls_key, S.task_key, S.obs_perm_key):
        ks = [{keyf(t) for t in x} for x in (tr, va, rt)]
        if keyf is not S.obs_perm_key:   # splits: distinct wall layouts (so distinct tasks)
            assert not ks[0] & ks[1] and not ks[0] & ks[2] and not ks[1] & ks[2], keyf.__name__
        sk = {keyf(t) for t in stream_tasks}
        assert not (ks[0] | ks[1] | ks[2]) & sk, keyf.__name__
    assert len({S.walls_key(t) for t in tr + va + rt}) == len(tr + va + rt)


def test_demo_split_does_not_depend_on_network_or_streams(dev_streams):
    a = bc.demo_split(SCFG, dev_streams, SMALL_BC)
    b = bc.demo_split(SCFG, dev_streams[:1], SMALL_BC)
    assert [bc.tasks_hash(x) for x in a] == [bc.tasks_hash(x) for x in b]


# ------------------------------------------------------------------ BC training
def test_demos_use_rules_epsilon_and_planner_labels(split):
    assert bc.BCConfig().epsilon == 0.15 and bc.BCConfig().num_retention_tasks == 5
    d = bc.make_demos(split[0], SCFG, SMALL_BC, jax.random.PRNGKey(0))
    assert float(d["epsilon"]) == pytest.approx(0.15)
    assert d["obs"].shape[0] == d["label"].shape[0] > 0
    assert (d["label"] != d["action_taken"]).any()      # noise actions are not the labels


def test_zero_bc_epochs_returns_scratch_init(split):
    d = bc.make_demos(split[0], SCFG, SMALL_BC, jax.random.PRNGKey(0))
    z = dataclasses.replace(SMALL_BC, max_epochs=0)
    p, info = bc.train_bc(TINY, d, d, z)
    assert _bitwise_equal(p, bc.init_params(TINY, z)) and info["best_epoch"] == 0


def test_zero_bc_epochs_stream_run_identical_to_scratch(split):
    """A pretrained arm with 0 BC epochs and 0 warm-up, started from the scratch agent's own
    init, reproduces the scratch stream run bit for bit (the BC hooks add nothing)."""
    seed = 990
    _, _, network, obs_shape = ppo.make_env_and_network(TINY)
    k_init = jax.random.split(jax.random.PRNGKey(seed), 3)[1]   # make_stream_train's init key
    scratch0 = network.init(k_init, jnp.zeros(obs_shape))
    d = bc.make_demos(split[0], SCFG, SMALL_BC, jax.random.PRNGKey(0))
    p, _ = bc.train_bc(TINY, d, d, dataclasses.replace(SMALL_BC, max_epochs=0), init=scratch0)
    a = lifelong.run_condition(TINY, SCFG, [seed], fresh=False, verbose=False)
    b = lifelong.run_condition(bc.pretrained_config(TINY, 0), SCFG, [seed], fresh=False,
                               verbose=False, init_params=p)
    for k, v in a.items():
        if k.startswith("stream/metrics") or k.startswith("stream/eval"):
            np.testing.assert_array_equal(np.asarray(v), np.asarray(b[k]), err_msg=k)


def test_bc_trains_only_trunk_and_actor(split):
    d = bc.make_demos(split[0], SCFG, SMALL_BC, jax.random.PRNGKey(0))
    # validate on the train demos so the best epoch is > 0 and the params must have moved
    p, info = bc.train_bc(TINY, d, d, dataclasses.replace(SMALL_BC, patience=100))
    assert info["best_epoch"] > 0
    p0 = bc.init_params(TINY, SMALL_BC)
    f, f0 = _flat(p), _flat(p0)
    for k in f:
        if any(str(x).startswith("critic") for x in k):
            assert np.array_equal(f[k], f0[k]), k
    assert any(not np.array_equal(f[k], f0[k]) for k in f if any("actor_out" in str(x) for x in k))
    assert info["epochs_run"] == 3 and len(info["history"]["val_loss"]) == 4
    assert info["best_val_loss"] == min(info["history"]["val_loss"])
    for k in ("best_val_acc", "best_val_entropy", "best_train_acc"):
        assert np.isfinite(info[k])


def test_bc_early_stopping_keeps_best_val_epoch(split):
    d = bc.make_demos(split[0], SCFG, SMALL_BC, jax.random.PRNGKey(0))
    v = bc.make_demos(split[1], SCFG, SMALL_BC, jax.random.PRNGKey(1))
    cfg = dataclasses.replace(SMALL_BC, max_epochs=40, patience=2, lr=3e-2)
    p, info = bc.train_bc(TINY, d, v, cfg)
    h = info["history"]
    assert info["best_epoch"] == int(np.argmin(h["val_loss"]))
    assert info["epochs_run"] == 40 or info["epochs_run"] - info["best_epoch"] == 2


def test_checkpoint_roundtrip_and_name(tmp_path, split):
    p = bc.init_params(TINY, SMALL_BC)
    path = str(tmp_path / bc.checkpoint_name(TINY, 900))
    assert os.path.basename(path) == "bc_h16_l2_shared_s900.msgpack"
    bc.save_checkpoint(path, p, dict(best_epoch=0, x=np.float32(1.0)))
    q, meta = bc.load_checkpoint(path, TINY)
    assert _bitwise_equal(p, q) and meta["best_epoch"] == 0
    big = dataclasses.replace(TINY, hidden_size=950)
    assert bc.checkpoint_name(big, 900) != bc.checkpoint_name(TINY, 900)
    assert bc.checkpoint_name(TINY, 901) != bc.checkpoint_name(TINY, 900)


def test_bc_cli_refuses_reserved_seeds(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("BC reached")
    monkeypatch.setattr(bc, "pretrain", boom)
    for argv in (["bc.py", "--seeds", "700-724", "--out-dir", str(tmp_path)],
                 ["bc.py", "--seeds", "750-759", "--allow-official-seeds", "--out-dir", str(tmp_path)],
                 ["bc.py", "--seeds", "500-509", "--out-dir", str(tmp_path)]):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            bc.main()
        assert isinstance(e.value.code, str) and "reserved" in e.value.code
    assert not os.listdir(tmp_path)


# ------------------------------------------------------------------ handover / warm-up / retention
def test_pretrained_config_resets_adam_and_sets_warmup():
    c = bc.pretrained_config(TINY, 3)
    assert c.critic_warmup_updates == 3 and c.adam_reset_at_handover
    with pytest.raises(ValueError):   # no BC optimiser state can be carried without opting out
        ppo.handover_opt_state(dataclasses.replace(c, adam_reset_at_handover=False), None, None)


def test_warmup_longer_than_task1_budget_refused():
    with pytest.raises(ValueError, match="task 1"):
        lifelong.make_stream_train(bc.pretrained_config(TINY, SCFG.block_updates + 1), SCFG, 3)
    lifelong.make_stream_train(bc.pretrained_config(TINY, SCFG.block_updates), SCFG, 3)  # == K ok


def test_warmup_freezes_trunk_and_actor_inside_task1(dev_streams):
    """Warm-up W = K: at the start of block 2 (after task 1's K updates) every non-critic
    param still equals the pretrained init; the critic has moved."""
    p0 = bc.init_params(TINY, SMALL_BC)
    cfg = bc.pretrained_config(TINY, SCFG.block_updates)
    tasks, sched, probe = (S.stack_tasks(dev_streams[0].task_list, SCFG.max_steps),
                           jnp.asarray(dev_streams[0].schedule), S.probe_batch(dev_streams[0]))
    B = dev_streams[0].num_blocks
    out = jax.jit(lifelong.make_stream_train(cfg, SCFG, B, init_params=p0, snapshot_blocks=(1,)))(
        jax.random.PRNGKey(990), tasks, sched, probe)
    snap = jax.tree.map(lambda x: x[0], out["snapshots"])
    f, f0 = _flat(snap), _flat(p0)
    moved = []
    for k in f:
        if any(str(x).startswith("critic") for x in k):
            moved.append(not np.array_equal(f[k], f0[k]))
        else:
            assert np.array_equal(f[k], f0[k]), k
    assert any(moved)
    w = np.asarray(out["metrics"]["critic_warmup"])
    assert w.reshape(-1)[:SCFG.block_updates].all() and not w.reshape(-1)[SCFG.block_updates:].any()


def test_retention_reported_at_every_switch_only_when_given(split):
    rt = split[2]
    off = lifelong.run_condition(TINY, SCFG, [990], fresh=False, verbose=False)
    on = lifelong.run_condition(TINY, SCFG, [990], fresh=False, verbose=False, retention_tasks=rt)
    assert not any("retention" in k for k in off)
    B = S.make_stream(SCFG, 990).num_blocks
    assert np.asarray(on["stream/retention"]).shape == (1, B, 5)
    assert np.asarray(on["stream/retention0"]).shape == (1, 5)
    assert on["retention_optimal_return"].shape == (5,) and len(on["retention_task_keys"]) == 5
    assert list(on["retention_task_keys"]) == [S.task_key(t) for t in rt]
    for k, v in off.items():   # retention eval does not perturb training
        if k.startswith("stream/metrics") or k.startswith("stream/eval"):
            np.testing.assert_array_equal(np.asarray(v), np.asarray(on[k]), err_msg=k)


def test_cpu_dev_seed_bc_sanity(dev_streams):
    """Tiny end-to-end BC on dev streams: finite stats, demos mostly solve their tasks."""
    params, meta, rt = bc.pretrain(TINY, SCFG, dev_streams, 900, SMALL_BC)
    assert len(rt) == 5 and meta["stream_seeds"] == [900, 901]
    assert meta["demo_success"] > 0.5 and np.isfinite(meta["best_val_loss"])
    assert meta["best_val_loss"] <= meta["history"]["val_loss"][0]
