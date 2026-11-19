"""Commit D: shared obs layout / action perm (stage-1 config), demo-pool rule + cap, per-task
logging, BC per seed from the seed's scratch init, provenance + dirty-tree guard.
Only dev seeds 900+ are trained; stage-1 seeds only build host-side task lists."""
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bc  # noqa: E402
import lifelong  # noqa: E402
import ppo  # noqa: E402
import provenance  # noqa: E402
import seeds  # noqa: E402
import stream as S  # noqa: E402

TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=64, hidden_size=16,
                         eval_episodes=2))
SMALL_BC = bc.BCConfig(num_train_tasks=6, num_val_tasks=3, num_retention_tasks=5,
                       episodes_per_task=2, max_epochs=3, patience=2, batch_size=64)
SH = S.StreamConfig(num_new_tasks=7, revisit_every=3, block_updates=2,
                    shared_obs_layout=True, shared_action_perm=True)
ID_OBS, ID_ACT = np.arange(192), np.arange(4)


def _flat(p):
    return traverse_util.flatten_dict(jax.tree.map(np.asarray, p))


def _bitwise_equal(a, b):
    fa, fb = _flat(a), _flat(b)
    return fa.keys() == fb.keys() and all(np.array_equal(fa[k], fb[k]) for k in fa)


# ------------------------------------------------------------------ 1. shared constants + flags
def test_shared_constants_and_hash():
    assert seeds.SHARED_OBS_LAYOUT == tuple(range(192)) and seeds.SHARED_ACTION_PERM == (0, 1, 2, 3)
    b = (np.asarray(seeds.SHARED_OBS_LAYOUT, np.int32).tobytes()
         + np.asarray(seeds.SHARED_ACTION_PERM, np.int32).tobytes())
    assert hashlib.sha256(b).hexdigest() == seeds.SHARED_LAYOUT_HASH
    assert S.StreamConfig().shared_obs_layout is False and S.StreamConfig().shared_action_perm is False


@pytest.mark.parametrize("sd", [900, 905, 939])
def test_stage1_task1_identity_and_later_draws_unchanged(sd):
    pil, st = S.make_stream(S.PILOT_STREAM, sd), S.make_stream(S.STAGE1_STREAM, sd)
    t0p, t0 = pil.task_list[0], st.task_list[0]
    assert np.array_equal(t0["obs_perm"], ID_OBS) and np.array_equal(t0["action_perm"], ID_ACT)
    assert st.shared_obs[0] and st.shared_action[0]
    for k in ("walls", "start", "goal", "slip"):            # same draws as the pilot stream
        assert np.array_equal(t0p[k], t0[k]), k
    # random perms drawn and discarded, so the draws stay in step with the pilot stream until
    # a dynamics shift's rejection step differs (pilot rejects perm == its random prev,
    # stage 1 rejects == identity); 900 never diverges, 905 diverges at task 12.
    d = next((i for i, (a, b) in enumerate(zip(pil.task_shift, st.task_shift)) if a != b), len(st.task_shift))
    assert {900: 40, 905: 11, 939: 40}[sd] == d
    for a, b in zip(pil.task_list[:d], st.task_list[:d]):
        assert np.array_equal(a["goal"], b["goal"]) and np.array_equal(a["start"], b["start"])
    if d == 40:
        assert np.array_equal(st.schedule, pil.schedule)
    for a, b in zip(pil.task_list, st.task_list):   # walls come from task 1's draw only
        assert np.array_equal(a["walls"], b["walls"])
    # later obs / dynamics shifts permute away from the identity
    for i, (t, kind) in enumerate(zip(st.task_list, st.task_shift)):
        if kind == "obs":
            assert not np.array_equal(t["obs_perm"], ID_OBS) and not st.shared_obs[i]
        if kind == "dynamics":
            assert not np.array_equal(t["action_perm"], ID_ACT) and not st.shared_action[i]
    assert not pil.shared_obs.any()


def test_dynamics_shift_redraws_identity_only_when_flag_on():
    class R:  # rng stub: first permutation is the identity, then a non-identity one
        def __init__(self):
            self.p = [np.arange(4), np.array([1, 0, 2, 3])]

        def permutation(self, n):
            return self.p.pop(0)

        def choice(self, c):
            return c[0]
    prev = np.array([3, 2, 1, 0])
    off, _ = S._sample_dynamics(R(), S.PILOT_STREAM, prev)
    on, _ = S._sample_dynamics(R(), S.STAGE1_STREAM, prev, avoid_shared=True)
    assert np.array_equal(off, ID_ACT) and np.array_equal(on, [1, 0, 2, 3])


def test_config_record_hides_default_flags():
    assert S.config_record(S.PILOT_STREAM) == {k: v for k, v in dataclasses.asdict(S.PILOT_STREAM).items()
                                               if k not in S.SHARED_FLAGS}
    r = S.config_record(S.STAGE1_STREAM)
    assert r["shared_obs_layout"] is True and r["shared_action_perm"] is True


# ------------------------------------------------------------------ 3. demo pool
def test_stage1_demo_pool_identity_and_componentwise_disjoint():
    refs = S.reference_streams()
    st1 = [s for s in refs if s.cfg == S.STAGE1_STREAM]
    tasks, _ = S.make_demo_pool(S.STAGE1_STREAM, 300, st1)
    assert all(np.array_equal(t["obs_perm"], ID_OBS) and np.array_equal(t["action_perm"], ID_ACT)
               for t in tasks)
    stream_tasks = [t for s in refs for t in s.task_list]
    for key in (S.walls_key, S.walls_goal_key, S.layout_key, S.task_key):
        assert not ({key(t) for t in tasks} & {key(t) for t in stream_tasks}), key.__name__
    shared = {S.obs_perm_key(t) for t in tasks}   # the only shared perm is the shared layout
    assert len(shared) == 1 and shared <= {S.obs_perm_key(t) for t in stream_tasks}


def test_excluded_obs_perms_rule():
    st = S.make_stream(S.STAGE1_STREAM, 900)
    ex = S.excluded_obs_perms([st])
    assert S.obs_perm_key(st.task_list[0]) not in ex
    nonshared = [t for t in st.task_list if not np.array_equal(t["obs_perm"], ID_OBS)]
    assert nonshared and all(S.obs_perm_key(t) in ex for t in nonshared)


def test_demo_pool_rejects_nonshared_perm_and_has_hard_cap(monkeypatch):
    st = S.make_stream(S.STAGE1_STREAM, 900)
    bad = next(t for t in st.task_list if not np.array_equal(t["obs_perm"], ID_OBS))
    orig = S.random_task

    def rt(rng, cfg):  # fresh walls, but a non-shared obs perm used by the stream
        t = orig(rng, cfg)
        t["obs_perm"] = bad["obs_perm"].copy()
        return t
    monkeypatch.setattr(S, "random_task", rt)
    monkeypatch.setattr(S, "DEMO_POOL_MAX_DRAWS_PER_TASK", 1)
    with pytest.raises(RuntimeError, match="candidates drawn"):   # flags off: perm kept -> rejected
        S.make_demo_pool(S.PILOT_STREAM, 2, [st])
    tasks, _ = S.make_demo_pool(S.STAGE1_STREAM, 2, [st])           # flags on: identity -> accepted
    assert len(tasks) == 2


def test_default_demo_pool_exclusion_unchanged():
    refs = [s for s in S.reference_streams() if s.cfg == S.PILOT_STREAM]
    assert S.excluded_obs_perms(refs) == {S.obs_perm_key(t) for s in refs for t in s.task_list}


# ------------------------------------------------------------------ 4. per-task logging
def test_run_record_logs_shared_layout_per_task():
    cfg = dataclasses.replace(SH, num_new_tasks=3, block_updates=1)
    r = lifelong.run_condition(TINY, cfg, [900, 901], fresh=True, verbose=False)
    sts = [S.make_stream(cfg, s) for s in (900, 901)]
    np.testing.assert_array_equal(r["task_shared_obs_layout"], np.stack([s.shared_obs for s in sts]))
    np.testing.assert_array_equal(r["task_shared_action_perm"], np.stack([s.shared_action for s in sts]))
    assert r["task_shared_obs_layout"][:, 0].all() and str(r["shared_layout_hash"]) == seeds.SHARED_LAYOUT_HASH
    m = lifelong.relative_metrics(r)
    assert m["shared_obs_layout_share_new"].shape == (2,)
    off = lifelong.run_condition(TINY, dataclasses.replace(cfg, shared_obs_layout=False, shared_action_perm=False),
                                 [900], fresh=False, verbose=False)
    assert "task_shared_obs_layout" not in off and "shared_layout_hash" not in off


# ------------------------------------------------------------------ 5. BC per seed
def test_scratch_init_params_is_the_scratch_arm_init():
    cfg = dataclasses.replace(SH, num_new_tasks=2, block_updates=1)
    for sd in (900, 903):
        st = S.make_stream(cfg, sd)
        train = lifelong.make_stream_train(TINY, cfg, st.num_blocks, snapshot_blocks=(0,))
        out = train(jax.random.PRNGKey(sd), S.stack_tasks(st.task_list, cfg.max_steps),
                    jnp.asarray(st.schedule), S.probe_batch(st))
        snap = jax.tree.map(lambda x: x[0], out["snapshots"])
        assert _bitwise_equal(snap, lifelong.scratch_init_params(TINY, sd))
    assert not _bitwise_equal(lifelong.scratch_init_params(TINY, 900), lifelong.scratch_init_params(TINY, 901))


def test_bc_per_seed_starts_from_scratch_init_and_zero_epochs_is_scratch(monkeypatch):
    cfg = dataclasses.replace(SH, num_new_tasks=3, block_updates=2)
    seeds_ = [900, 901]
    streams = [S.make_stream(cfg, s) for s in seeds_]
    seen, orig = [], bc.train_bc

    def spy(config, tr, va, bccfg, verbose=False, init=None, **k):
        seen.append(init)
        return orig(config, tr, va, bccfg, verbose, init=init, **k)
    zero = dataclasses.replace(SMALL_BC, max_epochs=0)
    demos = bc.demo_data(cfg, streams, zero)
    monkeypatch.setattr(bc, "train_bc", spy)
    ps = []
    for s in seeds_:
        p, meta, _ = bc.pretrain(TINY, cfg, streams, s, zero, demos=demos)
        ps.append(p)
        assert meta["seed"] == s and meta["best_epoch"] == 0
    for s, init, p in zip(seeds_, seen, ps):
        assert _bitwise_equal(init, lifelong.scratch_init_params(TINY, s))
        assert _bitwise_equal(p, lifelong.scratch_init_params(TINY, s))
    # zero-epoch BC per seed as the stream agent's init == the scratch run, bit for bit
    a = lifelong.run_condition(TINY, cfg, seeds_, fresh=False, verbose=False, seed_init_params=ps)
    b = lifelong.run_condition(TINY, cfg, seeds_, fresh=False, verbose=False)
    assert a.pop("seed_init_params") is True
    for k in b:
        if k.startswith("stream/") and "seconds" not in k:
            va, vb = np.asarray(a[k]), np.asarray(b[k])
            assert np.array_equal(va, vb, equal_nan=vb.dtype.kind == "f"), k
    with pytest.raises(ValueError, match="entries"):
        lifelong.run_condition(TINY, cfg, seeds_, fresh=False, verbose=False, seed_init_params=ps[:1])
    with pytest.raises(ValueError, match="not both"):
        lifelong.run_condition(TINY, cfg, seeds_, fresh=False, verbose=False, seed_init_params=ps,
                               init_params=ps[0])
    with pytest.raises(ValueError, match="scratch-only"):
        lifelong.run_condition(TINY, cfg, seeds_, fresh=False, verbose=False, seed_init_params=ps,
                               reset_arm="heads")
    with pytest.raises(ValueError, match="stream seeds"):
        bc.pretrain(TINY, cfg, streams, 902, zero, demos=demos)


def test_until_train_acc_prefix_equals_default_run():
    cfg = dataclasses.replace(SH, num_new_tasks=2)
    streams = [S.make_stream(cfg, 900)]
    _, _, _, dtr, dva = bc.demo_data(cfg, streams, SMALL_BC)
    init = lifelong.scratch_init_params(TINY, 900)
    _, a = bc.train_bc(TINY, dtr, dva, SMALL_BC, init=init)
    _, b = bc.train_bc(TINY, dtr, dva, SMALL_BC, init=init, until_train_acc=2.0)
    n = len(a["history"]["train_acc"])
    assert b["history"]["train_acc"][:n] == a["history"]["train_acc"]
    assert b["epochs_run"] == SMALL_BC.max_epochs


# ------------------------------------------------------------------ 6. provenance / dirty tree
def test_source_hashes_cover_tracked_py_and_scripts():
    h = provenance.source_hashes()
    assert {"ppo.py", "lifelong.py", "bc.py", "stream.py", "seeds.py", "scripts/run_pilot.sh",
            "scripts/bc_dev_checks.py"} <= set(h)
    assert not any(p.startswith("results/") for p in h)
    with open(os.path.join(ROOT, "seeds.py"), "rb") as f:
        assert h["seeds.py"] == hashlib.sha256(f.read()).hexdigest()
    r = provenance.record()
    assert json.loads(r["source_hashes"]) == h and r["source_tree_hash"] == provenance.tree_hash(h)
    assert len(r["git_commit_full"]) == 40


def test_check_clean_rules(monkeypatch):
    monkeypatch.setattr(provenance, "dirty_files", lambda: ["lifelong.py"])
    with pytest.raises(SystemExit, match="dirty"):
        provenance.check_clean([900], False, force=True)
    for bad in ([899], [700], [], [900, 500]):
        with pytest.raises(SystemExit, match="dev seeds"):
            provenance.check_clean(bad, True, force=True)
    r = provenance.check_clean([900, 901], True, force=True)
    assert r["git_dirty"] is True and r["allow_dirty"] is True
    with pytest.raises(SystemExit, match="reserved"):            # ppo.py: no reserved seed
        provenance.check_clean([500], True, require_dev=False, force=True)
    assert provenance.check_clean([], True, require_dev=False, force=True)["allow_dirty"]
    assert provenance.check_clean([900], False)["git_dirty"] is True   # under pytest: record only
    monkeypatch.setattr(provenance, "dirty_files", lambda: [])
    r = provenance.check_clean([900], False, force=True)
    assert r["git_dirty"] is False and r["allow_dirty"] is False


def _load_dev_checks():
    spec = importlib.util.spec_from_file_location("bc_dev_checks", os.path.join(ROOT, "scripts", "bc_dev_checks.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cli_paths_refuse_dirty_tree(monkeypatch, tmp_path):
    """Outside pytest (simulated) with a dirty tracked tree, every CLI path exits before
    any training or output; --allow-dirty with a dev seed runs and is recorded."""
    monkeypatch.setattr(provenance, "under_pytest", lambda: False)
    monkeypatch.setattr(provenance, "dirty_files", lambda: ["README.md"])

    def boom(*a, **k):
        raise AssertionError("training reached")
    monkeypatch.setattr(lifelong, "run_condition", boom)
    monkeypatch.setattr(bc, "demo_data", boom)
    monkeypatch.setattr(ppo, "make_train", boom)
    dc = _load_dev_checks()
    out = tmp_path / "o"
    cases = [(lifelong.main, ["lifelong.py", "--seeds", "990", "--out", str(out / "x.npz")]),
             (bc.main, ["bc.py", "--seeds", "990", "--out-dir", str(out)]),
             (ppo.main, ["ppo.py", "--num-seeds", "1", "--out", str(out / "c.npz")]),
             (dc.main, ["bc_dev_checks.py", "--seeds", "990", "--out-dir", str(out)])]
    for fn, argv in cases:
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            fn()
        assert "tracked tree is dirty" in str(e.value.code), argv
    assert not out.exists()
    monkeypatch.undo()
    monkeypatch.setattr(provenance, "under_pytest", lambda: False)
    monkeypatch.setattr(provenance, "dirty_files", lambda: ["README.md"])
    monkeypatch.setattr(sys, "argv", ["lifelong.py", "--seeds", "990", "--out", str(tmp_path / "x.npz"),
                                      "--num-new-tasks", "1", "--block-updates", "1", "--hidden-size", "8",
                                      "--no-fresh", "--allow-dirty"])
    lifelong.main()
    d = np.load(tmp_path / "x.npz")
    assert bool(d["git_dirty"]) and bool(d["allow_dirty"]) and len(str(d["source_tree_hash"])) == 64
    assert "shared_obs_layout" not in json.loads(str(d["stream_config"]))
    monkeypatch.setattr(sys, "argv", ["lifelong.py", "--seeds", "899", "--allow-official-seeds",
                                      "--out", str(tmp_path / "y.npz"), "--allow-dirty"])
    with pytest.raises(SystemExit):
        lifelong.main()
    assert not (tmp_path / "y.npz").exists()
