"""Tests for the gridworld, the task stream, the BFS expert, the stream trainer, the
relative-to-fresh metrics and the new ppo.py hooks. Run with: pytest -q"""
import dataclasses
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import expert  # noqa: E402
import gridworld as gw  # noqa: E402
import lifelong  # noqa: E402
import ppo  # noqa: E402
import stream as S  # noqa: E402
import stream_metrics as SM  # noqa: E402

SCFG = S.StreamConfig(num_new_tasks=7, revisit_every=3, block_updates=2)
ENV = gw.Gridworld(8)


def task_params(t, max_steps=64):
    return S.stack_tasks([t], max_steps).__class__(**{
        k: jnp.asarray(v)[0] for k, v in dataclasses.asdict(S.stack_tasks([t], max_steps)).items()})


def open_task(**kw):
    p = ENV.default_params
    return p.replace(**kw)


# ------------------------------------------------------------------ env
def test_reset_puts_agent_at_start_and_obs_is_permuted_render():
    s = S.make_stream(SCFG, 900)
    t = task_params(s.task_list[0])
    obs, st = ENV.reset_env(jax.random.PRNGKey(0), t)
    assert np.array_equal(np.asarray(st.pos), s.task_list[0]["start"]) and int(st.time) == 0
    flat = np.asarray(gw.render_flat(t.walls, st.pos, t.goal))
    assert np.array_equal(np.asarray(obs), flat[s.task_list[0]["obs_perm"]])
    assert obs.shape == ENV.observation_space(t).shape


def test_dynamics_walls_permutation_goal_and_truncation():
    p = open_task()  # open 6x6 room, start (1,1), goal (6,6), identity perm
    k = jax.random.PRNGKey(0)
    _, st = ENV.reset_env(k, p)
    _, st2, r, d, _ = ENV.step_env(k, st, 0, p)  # up into the border wall -> stays
    assert np.array_equal(st2.pos, [1, 1]) and float(r) == 0 and not bool(d)
    _, st2, *_ = ENV.step_env(k, st, 1, p)  # right
    assert np.array_equal(st2.pos, [1, 2])
    pp = p.replace(action_perm=jnp.array([2, 3, 0, 1], jnp.int32))  # action 0 -> down
    _, st3, *_ = ENV.step_env(k, st, 0, pp)
    assert np.array_equal(st3.pos, [2, 1])
    # One step from the goal: reaching it terminates with 1 - 0.9 t/T.
    near = gw.EnvState(pos=jnp.array([6, 5], jnp.int32), time=jnp.array(9, jnp.int32))
    _, st4, r, d, _ = ENV.step_env(k, near, 1, p)
    assert bool(d) and np.isclose(float(r), 1 - 0.9 * 10 / 64) and bool(ENV.is_terminal(st4, p))
    # Truncation at max_steps without reaching the goal.
    late = gw.EnvState(pos=jnp.array([1, 1], jnp.int32), time=jnp.array(63, jnp.int32))
    _, st5, r, d, _ = ENV.step_env(k, late, 0, p)
    assert bool(d) and float(r) == 0
    # ppo.env_step sees it as truncated, not terminated.
    out = ppo.env_step(ENV, k, late, jnp.int32(0), p)
    assert bool(out.truncated) and not bool(out.terminated)


def test_slip_one_gives_random_directions():
    p = open_task(slip=jnp.float32(1.0))
    st = gw.EnvState(pos=jnp.array([3, 3], jnp.int32), time=jnp.array(0, jnp.int32))
    keys = jax.random.split(jax.random.PRNGKey(1), 400)
    pos = jax.vmap(lambda k: ENV.step_env(k, st, 1, p)[1].pos)(keys)
    moves = {tuple(np.asarray(x) - 3) for x in pos}
    assert moves == {(-1, 0), (1, 0), (0, 1), (0, -1)}


def test_obs_permutation_is_bijection_and_no_task_id_in_obs():
    s = S.make_stream(SCFG, 901)
    F = 3 * 8 * 8
    for t in s.task_list:
        assert np.array_equal(np.sort(t["obs_perm"]), np.arange(F))
    t = s.task_list[0]
    flat = np.asarray(gw.render_flat(jnp.asarray(t["walls"]), jnp.asarray(t["start"]), jnp.asarray(t["goal"])))
    obs = flat[t["obs_perm"]]
    inv = np.argsort(t["obs_perm"])
    assert np.array_equal(obs[inv], flat)  # invertible -> no information lost
    # Two tasks that differ only in dynamics give identical observations: nothing in
    # the obs identifies the task beyond the grid itself.
    a = task_params(t)
    b = a.replace(action_perm=a.action_perm[::-1], slip=jnp.float32(0.1))
    assert np.array_equal(ENV.reset_env(jax.random.PRNGKey(0), a)[0], ENV.reset_env(jax.random.PRNGKey(5), b)[0])


# ------------------------------------------------------------------ stream
def test_revisit_schedule():
    s = S.make_stream(SCFG, 902)
    new = s.schedule[s.is_new]
    assert np.array_equal(new, np.arange(SCFG.num_new_tasks))  # new tasks in order, once each
    b_new = np.flatnonzero(s.is_new)
    for b in np.flatnonzero(~s.is_new):
        i = int(s.schedule[b - 1])  # the new task just trained
        assert (i + 1) % SCFG.revisit_every == 0  # revisit after every 3rd new task
        assert s.schedule[b] < i  # an earlier task, not the one just trained
        assert s.is_new[b - 1]
    n_rev = sum(1 for i in range(SCFG.num_new_tasks) if (i + 1) % 3 == 0 and i >= 1)
    assert (~s.is_new).sum() == n_rev and len(b_new) == SCFG.num_new_tasks
    assert all(s.block_shift[b] == S.SHIFT_CODES["revisit"] for b in np.flatnonzero(~s.is_new))


def test_task_order_randomised_per_seed_and_deterministic():
    a, a2, b = S.make_stream(SCFG, 903), S.make_stream(SCFG, 903), S.make_stream(SCFG, 904)
    assert all(S.task_key(x) == S.task_key(y) for x, y in zip(a.task_list, a2.task_list))
    assert np.array_equal(a.schedule, a2.schedule)
    assert any(S.task_key(x) != S.task_key(y) for x, y in zip(a.task_list, b.task_list))


def test_each_new_task_applies_exactly_its_shift():
    s = S.make_stream(S.StreamConfig(num_new_tasks=30), 905)
    expect = {"goal": {"goal"}, "dynamics": {"dynamics"}, "obs": {"obs"}}
    for i in range(1, s.num_tasks):
        c = S.changed_components(s.task_list[i - 1], s.task_list[i])
        assert {k for k, v in c.items() if v} == expect[s.task_shift[i]]
    assert set(s.task_shift[1:]) == {"goal", "dynamics", "obs"}
    keys = [S.task_key(t) for t in s.task_list]
    assert len(set(keys)) == len(keys)  # distinct tasks


def test_demo_pool_disjoint_from_streams():
    streams = [S.make_stream(SCFG, sd) for sd in (906, 907, 908)]
    tasks, stacked = S.make_demo_pool(SCFG, 40, streams)
    stream_l = {S.layout_key(t) for s in streams for t in s.task_list}
    stream_t = {S.task_key(t) for s in streams for t in s.task_list}
    assert not ({S.layout_key(t) for t in tasks} & stream_l)
    assert not ({S.task_key(t) for t in tasks} & stream_t)
    assert stacked.walls.shape[0] == 40


# ------------------------------------------------------------------ expert
def test_bfs_matches_numpy_bfs():
    s = S.make_stream(S.StreamConfig(num_new_tasks=10, shift_kinds=("goal",)), 909)
    for t in s.task_list:
        dj = np.asarray(expert.bfs_distance(jnp.asarray(t["walls"]), jnp.asarray(t["goal"])))
        dn = S._bfs_np(t["walls"], tuple(t["goal"]))
        reach = dn >= 0
        assert np.array_equal(dj[reach], dn[reach]) and (dj[~reach] >= expert.UNREACHABLE).all()


def test_expert_eps0_gives_optimal_path_lengths_under_action_permutations():
    s = S.make_stream(S.StreamConfig(num_new_tasks=12), 910)  # slip = 0, permuted actions
    assert any(not np.array_equal(t["action_perm"], np.arange(4)) for t in s.task_list)
    demos = expert.generate_demos(s.tasks, 3, 0.0, jax.random.PRNGKey(0))
    dist = np.array([S._bfs_np(t["walls"], tuple(t["goal"]))[tuple(t["start"])] for t in s.task_list])
    assert np.array_equal(demos["episode_length"], np.repeat(dist, 3))
    assert np.allclose(demos["episode_return"], np.repeat(s.optimal_return, 3), atol=1e-6)
    assert np.array_equal(demos["label"], demos["action_taken"])  # no noise


def test_expert_eps015_still_reaches_goal_and_records_both_actions():
    s = S.make_stream(S.StreamConfig(num_new_tasks=12), 911)
    tasks = s.tasks.replace(max_steps_in_episode=jnp.full_like(s.tasks.max_steps_in_episode, 200))
    demos = expert.generate_demos(tasks, 8, 0.15, jax.random.PRNGKey(1), max_steps=200)
    assert (demos["episode_return"] > 0).all()  # every episode reached the goal
    dist = np.repeat([S._bfs_np(t["walls"], tuple(t["goal"]))[tuple(t["start"])] for t in s.task_list], 8)
    assert (demos["episode_length"] >= dist).all()
    frac_noisy = np.mean(demos["label"] != demos["action_taken"])
    assert 0.05 < frac_noisy < 0.2  # ~0.15 * 3/4 of the steps take a non-planner action


def test_expert_label_is_optimal_at_every_state():
    t = task_params(S.make_stream(SCFG, 912).task_list[0])
    dist = expert.bfs_distance(t.walls, t.goal)
    free = np.argwhere(~np.asarray(t.walls))
    for pos in free:
        if (pos == np.asarray(t.goal)).all() or int(dist[pos[0], pos[1]]) >= expert.UNREACHABLE:
            continue  # goal itself, or a free cell walled off from the goal
        a = expert.expert_action(t, jnp.asarray(pos), dist)
        nxt = gw.move(t.walls, jnp.asarray(pos), t.action_perm[a])
        assert int(dist[nxt[0], nxt[1]]) == int(dist[pos[0], pos[1]]) - 1


# ------------------------------------------------------------------ stream trainer
TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=64, hidden_size=16,
                         eval_episodes=2))


def _stream_inputs(scfg, seeds):
    streams = [S.make_stream(scfg, s) for s in seeds]
    return (streams, S.stack_streams(streams), jnp.asarray(np.stack([s.schedule for s in streams])),
            jnp.stack([S.probe_batch(s, n=64) for s in streams]))


def test_task_switching_inside_jit_compiles_once():
    scfg = S.StreamConfig(num_new_tasks=4, revisit_every=2, block_updates=2)
    streams, tasks, sched, probe = _stream_inputs(scfg, [913, 914])
    train = jax.jit(jax.vmap(lifelong.make_stream_train(TINY, scfg, streams[0].num_blocks)))
    keys = jnp.stack([jax.random.PRNGKey(s) for s in (913, 914)])
    n0 = ppo.TRACE_COUNT["train"]
    out = train(keys, tasks, sched, probe)
    # Different tasks + schedule (same shapes) -> no retrace.
    streams2, tasks2, sched2, probe2 = _stream_inputs(scfg, [915, 916])
    out2 = train(keys, tasks2, sched2, probe2)
    assert ppo.TRACE_COUNT["train"] - n0 == 1
    B, T, K = streams[0].num_blocks, streams[0].num_tasks, scfg.block_updates
    assert out["metrics"]["rollout_return"].shape == (2, B, K)
    assert out["eval"].shape == (2, B, T) and out["switch_entropy"].shape == (2, B)
    np.testing.assert_allclose(out["init_entropy"], np.log(4), atol=1e-3)
    for k in ("srank.trunk_0", "erank.trunk_1", "dormant_frac.trunk_0", "random_target_mse",
              "weight_norm", "probe_entropy"):
        assert out["probes"][k].shape == (2, B)
    assert not np.array_equal(out["metrics"]["loss"], out2["metrics"]["loss"])  # tasks really differ


def test_stream_carries_adam_by_default_and_reset_ablation_differs():
    scfg = S.StreamConfig(num_new_tasks=2, revisit_every=0, block_updates=2)
    streams, tasks, sched, probe = _stream_inputs(scfg, [917])
    one = lambda c: jax.jit(jax.vmap(lifelong.make_stream_train(c, scfg, 2)))(  # noqa: E731
        jnp.stack([jax.random.PRNGKey(917)]), tasks, sched, probe)
    a = one(TINY)
    b = one(dataclasses.replace(TINY, reset_adam_on_task_switch=True))
    ra, rb = a["metrics"]["loss"], b["metrics"]["loss"]
    np.testing.assert_array_equal(ra[:, 0], rb[:, 0])  # block 0 identical
    assert not np.array_equal(ra[:, 1], rb[:, 1])      # reset at the switch changes block 1


# ------------------------------------------------------------------ metrics
def test_steps_to_threshold_and_auc():
    c = np.array([[0.1, 0.5, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4]])
    assert list(SM.steps_to_threshold(c, 0.8)) == [3, 4]  # never reached -> capped at K (rules)
    assert list(SM.steps_to_threshold(c, 0.8, cap_at_k=False)) == [3, 5]
    np.testing.assert_allclose(SM.auc(c), [0.625, 0.25])
    np.testing.assert_allclose(SM.auc(c, M=2), [0.3, 0.15])


def test_relative_metric_reuses_fresh_runs_on_revisits():
    sched = np.array([[0, 1, 0, 2]])
    fresh = np.array([[0.5, 0.6, 0.7]])
    stream = np.array([[0.5, 0.4, 0.9, 0.3]])
    np.testing.assert_allclose(SM.relative(stream, fresh, sched), [[0.0, -0.2, 0.4, -0.4]])


def test_slope_and_summary_recover_known_trend():
    rng = np.random.default_rng(0)
    S_, B = 10, 12
    x = np.arange(B)
    y = -0.02 * x[None] + 0.01 * rng.standard_normal((S_, B))
    sl = SM.ols_slope(y)
    np.testing.assert_allclose(sl, -0.02, atol=0.005)
    summ = SM.summarize(sl)
    assert summ["ci_excludes_zero"] and summ["ci95"][1] < 0
    flat = SM.summarize(SM.ols_slope(0.01 * rng.standard_normal((S_, B))))
    assert not flat["ci_excludes_zero"] or abs(flat["iqm"]) < 0.002
    # mask / custom x
    mask = np.zeros((1, 4), bool); mask[0, [0, 2, 3]] = True
    np.testing.assert_allclose(SM.ols_slope(np.array([[0.0, 9.0, 2.0, 3.0]]), mask=mask), [1.0])


def test_forgetting_definition():
    # 3 blocks: task 0, task 1, task 0 again. Eval on task 0: 0.9 after block 0,
    # 0.3 after block 1 -> forgetting = 0.9 - 0.3 = 0.6.
    ev = np.zeros((1, 3, 2)); ev[0, 0, 0] = 0.9; ev[0, 1, 0] = 0.3; ev[0, 2, 0] = 1.0
    f = SM.forgetting(ev, np.zeros((1, 2)), np.array([[0, 1, 0]]))
    # (block number, task number) are 1-based: 3rd block, task 1.
    assert len(f[0]) == 1 and f[0][0][:2] == (3, 1) and np.isclose(f[0][0][2], 0.6)


def _synthetic(drops, S_=6, K=10):
    """Blocks: tasks 1,2,3, revisit 1, task 4, task 5 (1-based). Stream curve of block b
    peaks at 1 - drops[b]; fresh curves all peak at 1."""
    tob = np.array([[1, 2, 3, 1, 4, 5]] * S_)
    is_new = np.array([[1, 1, 1, 0, 1, 1]] * S_, bool)
    rr = np.stack([np.stack([np.linspace(0, 1 - dr, K) for dr in drops]) for _ in range(S_)])
    fr = np.stack([np.stack([np.linspace(0, 1, K)[None] for _ in range(5)]) for _ in range(S_)])
    return {"task_of_block": tob, "is_new": is_new, "optimal_return": np.ones((S_, 5)), "batch_size": 128,
            "stream/metrics/rollout_return": rr, "fresh/metrics/rollout_return": fr,
            "block_shift": np.array([[0, 1, 2, 5, 3, 1]] * S_),
            "stream/eval": np.ones((S_, 6, 5)), "stream/eval0": np.zeros((S_, 5))}


def test_analyze_progressive_decline_is_detected_by_primary():
    d = _synthetic([0.0, 0.1, 0.2, 0.0, 0.3, 0.4])
    r = SM.analyze(d, thr=0.8)
    assert r["primary"] == "slope_auc_rel_new_excl_task1"
    assert r["slope_auc_rel_new_excl_task1_summary"]["iqm"] < 0
    assert r["slope_steps_rel_new_excl_task1_summary"]["iqm"] > 0
    np.testing.assert_allclose(r["auc_rel"][:, 0], 0.0, atol=1e-9)  # task 1 equals fresh
    assert set(r["by_shift"]) == {"goal", "dynamics", "obs", "revisit"}  # task 1 ("init") excluded
    assert r["steps_rel"].max() % 128 == 0  # steps in env steps
    assert np.isfinite(r["late_auc_rel"]).all() and r["late_auc_rel_summary"]["iqm"] < 0
    assert r["task_of_block"].min() == 1


def test_one_off_first_switch_drop_is_primacy_bias_not_primary_slope():
    # Drop only at the first switch (task 1 -> task 2), flat afterwards.
    d = _synthetic([0.0, 0.5, 0.5, 0.0, 0.5, 0.5])
    r = SM.analyze(d, thr=0.8)
    np.testing.assert_allclose(r["slope_auc_rel_new_excl_task1"], 0.0, atol=1e-9)  # primary: no slope
    assert r["slope_auc_rel_new_incl_task1_summary"]["iqm"] < 0  # secondary sees the drop
    assert r["primacy_bias_summary"]["iqm"] < -0.2                 # reported as primacy bias
    assert not r["slope_auc_rel_new_excl_task1_summary"]["ci_excludes_zero"]


# ------------------------------------------------------------------ seeds
def test_fresh_seed_mapping_paired_and_disjoint():
    assert [lifelong.fresh_seed(s) for s in range(500, 510)] == list(range(600, 610))
    dev = [lifelong.fresh_seed(s) for s in range(900, 1000)]
    assert dev == list(range(1900, 2000))
    official = set(range(500, 510)) | set(range(600, 610))
    assert not (set(dev) & official) and not (set(dev) & set(range(900, 1000)))
    with pytest.raises(ValueError):
        lifelong.fresh_seed(42)
    # Fresh keys differ from stream keys and between tasks; task numbers are 1-based.
    fk = np.asarray(lifelong.fresh_keys([900], range(3)))
    assert fk.shape[:2] == (1, 3) and len({tuple(k) for k in fk[0]}) == 3
    assert np.array_equal(fk[0, 0], np.asarray(jax.random.fold_in(jax.random.PRNGKey(1900), 1)))
    assert not np.array_equal(fk[0, 0], np.asarray(jax.random.PRNGKey(900)))


@pytest.mark.parametrize("seeds", [[500], [505, 900], [600], [609], [42], [899], [1900], [2500]])
def test_seed_guard_blocks_official_and_fresh_blocks(seeds):
    with pytest.raises(SystemExit):
        lifelong.check_seeds(seeds)


def test_seed_guard_allows_dev_and_flagged_official(monkeypatch):
    assert lifelong.check_seeds([900, 919]) == [1900, 1919]
    # Second guard: under pytest, official seeds are refused even with the flag.
    assert lifelong.under_pytest()
    for seeds in (list(range(500, 510)), [505], [600]):
        with pytest.raises(SystemExit, match="pytest"):
            lifelong.check_seeds(seeds, allow_official=True)
    # The flag logic itself is checked with the pytest marker removed. This is a pure
    # function of the seed list (no stream, no training is reachable from here).
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert lifelong.check_seeds(list(range(500, 510)), allow_official=True) == list(range(600, 610))
    with pytest.raises(SystemExit):  # 600-609 are fresh seeds, never stream seeds
        lifelong.check_seeds([600], allow_official=True)


def test_cli_refuses_official_seeds(monkeypatch, tmp_path):
    for spec in ("500-509", "600-609"):
        monkeypatch.setattr(sys, "argv", ["lifelong.py", "--seeds", spec, "--out", str(tmp_path / "x.npz")])
        with pytest.raises(SystemExit):
            lifelong.main()
    assert not (tmp_path / "x.npz").exists()


def test_run_condition_outputs_one_based_tasks_and_same_k_for_fresh():
    scfg = S.StreamConfig(num_new_tasks=3, revisit_every=3, block_updates=2)
    res = lifelong.run_condition(TINY, scfg, [918, 919], verbose=False)
    assert res["task_of_block"].min() == 1 and res["task_of_block"][:, 0].tolist() == [1, 1]
    assert res["fresh_seeds"].tolist() == [1918, 1919]
    K = scfg.block_updates
    assert res["stream/metrics/rollout_return"].shape[-1] == K
    assert res["fresh/metrics/rollout_return"].shape[-1] == K  # same K updates per task
    assert res["block_env_steps"] == K * TINY.batch_size and res["param_count"] > 0
    r = SM.analyze(res)
    assert np.isfinite(r["slope_auc_rel_new_excl_task1"]).all()


# ------------------------------------------------------------------ ppo hooks
def _net(cfg):
    env, ep, net, shp = ppo.make_env_and_network(cfg)
    return net, net.init(jax.random.PRNGKey(0), jnp.zeros(shp)), shp


@pytest.mark.parametrize("shared", [False, True])
def test_layer_norm_flag(shared):
    base = ppo.Config(shared_trunk=shared)
    assert base.layer_norm is False
    net0, p0, shp = _net(base)
    net1, p1, _ = _net(dataclasses.replace(base, layer_norm=True))
    from flax import traverse_util
    f0, f1 = traverse_util.flatten_dict(p0), traverse_util.flatten_dict(p1)
    ln = [k for k in f1 if any("_ln" in str(x) for x in k)]
    assert ln and not any(any("_ln" in str(x) for x in k) for k in f0)
    for k in f0:  # Dense inits unchanged by adding LayerNorm modules
        np.testing.assert_array_equal(f0[k], f1[k])
    x = jax.random.normal(jax.random.PRNGKey(1), (32,) + shp)
    _, _, h = net1.apply(p1, x)
    pre = list(h.values())[0]
    assert pre.shape[-1] == base.hidden_size


def test_critic_warmup_freezes_everything_but_the_critic():
    cfg = ppo.Config(total_timesteps=4 * 128 * 2, eval_every=1, eval_episodes=1, eval_max_steps=5,
                     critic_warmup_updates=1, shared_trunk=True)
    net, p0, _ = _net(cfg)
    out = jax.jit(ppo.make_train(cfg))(jax.random.PRNGKey(0))
    m = jax.tree.map(np.asarray, out["metrics"])
    assert list(m["critic_warmup"]) == [1.0, 0.0]
    # Run exactly one update (warm-up only) and compare with init.
    cfg1 = dataclasses.replace(cfg, total_timesteps=4 * 128)
    out1 = jax.jit(ppo.make_train(cfg1))(jax.random.PRNGKey(0))
    p1 = out1["runner_state"].train_state.params["params"]
    pi = net.init(jax.random.split(jax.random.PRNGKey(0), 3)[1], jnp.zeros((4,)))["params"]
    np.testing.assert_array_equal(p1["trunk"]["layer_0"]["kernel"], pi["trunk"]["layer_0"]["kernel"])
    np.testing.assert_array_equal(p1["actor_out"]["kernel"], pi["actor_out"]["kernel"])
    assert not np.array_equal(p1["critic_out"]["kernel"], pi["critic_out"]["kernel"])


def test_kickstart_term():
    cfg = ppo.Config(kickstart_coef=0.5)
    net, p, shp = _net(cfg)
    n = 16
    obs = jax.random.normal(jax.random.PRNGKey(2), (n,) + shp)
    batch = ppo.Batch(obs, jnp.zeros(n, jnp.int32), jnp.full(n, -np.log(2.0)), jnp.zeros(n),
                      jax.random.normal(jax.random.PRNGKey(3), (n,)), jnp.zeros(n))
    l0, m0 = ppo.ppo_loss(p, net.apply, batch, ppo.Config())
    l_self, m_self = ppo.ppo_loss(p, net.apply, batch, cfg, teacher_params=p)
    assert np.isclose(float(m_self["kickstart_kl"]), 0.0, atol=1e-7) and np.isclose(l_self, l0)
    other = net.init(jax.random.PRNGKey(9), jnp.zeros(shp))
    other = jax.tree.map(lambda x: x * 50.0, other)  # very different policy
    l_t, m_t = ppo.ppo_loss(p, net.apply, batch, cfg, teacher_params=other)
    assert float(m_t["kickstart_kl"]) > 0 and np.isclose(float(l_t), float(l0) + 0.5 * float(m_t["kickstart_kl"]), rtol=1e-5)
    assert "kickstart_kl" not in m0  # off by default


def test_adam_handover_hook_and_init_params():
    cfg = ppo.Config(total_timesteps=4 * 128 * 2, eval_every=1, eval_episodes=1, eval_max_steps=5)
    assert cfg.adam_reset_at_handover is True
    net, p, _ = _net(cfg)
    tx, _ = ppo.make_optimizer(cfg, cfg.num_updates)
    # A "pretraining" opt state with non-zero moments.
    st = tx.init(p)
    g = jax.tree.map(jnp.ones_like, p)
    _, st = tx.update(g, st, p)
    base = jax.tree.map(np.asarray, jax.jit(ppo.make_train(cfg, init_params=p))(jax.random.PRNGKey(0))["metrics"])
    reset = jax.tree.map(np.asarray, jax.jit(ppo.make_train(cfg, init_params=p, init_opt_state=st))(
        jax.random.PRNGKey(0))["metrics"])
    carry_cfg = dataclasses.replace(cfg, adam_reset_at_handover=False)
    carry = jax.tree.map(np.asarray, jax.jit(ppo.make_train(carry_cfg, init_params=p, init_opt_state=st))(
        jax.random.PRNGKey(0))["metrics"])
    np.testing.assert_array_equal(base["loss"], reset["loss"])  # default: state ignored (reset once)
    assert not np.array_equal(base["loss"], carry["loss"])  # carried moments change the update
    with pytest.raises(ValueError):
        ppo.make_train(carry_cfg)(jax.random.PRNGKey(0))


def test_initial_entropy_logged_by_make_train():
    cfg = ppo.Config(total_timesteps=4 * 128, eval_every=1, eval_episodes=1, eval_max_steps=5)
    out = jax.jit(ppo.make_train(cfg))(jax.random.PRNGKey(0))
    assert np.isclose(float(out["init_entropy"]), np.log(2), atol=1e-3)


# ------------------------------------------------------------------ device logging
def test_device_info_fields():
    info = ppo.device_info()
    assert info["device_platform"] == jax.devices()[0].platform
    assert info["jax_version"] == jax.__version__ and info["jaxlib_version"]
    assert info["device_name"] and info["device_count"] >= 1


def test_run_condition_records_device_for_stream_and_fresh():
    scfg = S.StreamConfig(num_new_tasks=2, revisit_every=0, block_updates=1)
    res = lifelong.run_condition(TINY, scfg, [919], verbose=False)
    for side in ("stream", "fresh"):
        assert res[f"{side}/device_platform"] == jax.devices()[0].platform
        assert res[f"{side}/jax_version"] == jax.__version__
    assert SM.check_same_platform(res, require=True) == jax.devices()[0].platform
    r = lifelong.relative_metrics(res)
    assert r["platform"] == jax.devices()[0].platform


def test_relative_metrics_refuse_mixed_platforms():
    d = _synthetic([0.0, 0.1, 0.2, 0.0, 0.3, 0.4])
    SM.analyze(d)  # no device info (older files): accepted
    with pytest.raises(SM.PlatformMismatch):
        SM.analyze(d, require_device_info=True)
    d["stream/device_platform"], d["fresh/device_platform"] = "cpu", "gpu"
    with pytest.raises(SM.PlatformMismatch):
        SM.analyze(d)
    with pytest.raises(SM.PlatformMismatch):
        lifelong.relative_metrics(d)
    d["fresh/device_platform"] = "cpu"
    assert SM.analyze(d)["platform"] == "cpu"


def test_ppo_cli_records_device(tmp_path, monkeypatch):
    out = tmp_path / "r.npz"
    monkeypatch.setattr(sys, "argv", ["ppo.py", "--total-timesteps", "1024", "--num-seeds", "1",
                                      "--eval-every", "1", "--out", str(out)])
    ppo.main()
    d = np.load(out)
    assert str(d["device_platform"]) == jax.devices()[0].platform
    assert str(d["jax_version"]) == jax.__version__ and "jaxlib_version" in d.files


# ------------------------------------------------------------------ review 2026-10-24 fixes
def test_demo_pool_componentwise_disjoint_from_official_and_dev_streams():
    """Every reference stream (official 500-509 + dev 900-939, pilot length, generated
    without training) vs the demo pool: walls alone, (walls, goal), full layout, obs
    permutation and full task are each disjoint."""
    refs = S.reference_streams()
    official, dev = list(range(500, 510)), list(range(900, 940))
    stage1, sweep = list(range(700, 725)), list(range(750, 760))
    assert [(s.seed, s.cfg.num_new_tasks) for s in refs] == (
        [(sd, 40) for sd in official + dev] + [(sd, 200) for sd in official]
        + [(sd, 40) for sd in stage1 + sweep]            # pilot-config registrations (kept)
        + [(sd, 40) for sd in stage1 + sweep + dev])     # STAGE1_STREAM (shared layout)
    assert [s.cfg for s in refs[-75:]] == [S.STAGE1_STREAM] * 75
    # the 200-task streams use the pilot settings apart from num_new_tasks
    long_ = [s for s in refs if s.cfg.num_new_tasks == 200]
    assert all(dataclasses.replace(s.cfg, num_new_tasks=40) == S.PILOT_STREAM for s in long_)
    assert S.NULL200_STREAM.revisit_every == 3 and S.NULL200_STREAM.head_probe_every == 5
    assert S.NULL200_STREAM.block_updates == 100 and S.NULL200_STREAM.slip_choices == (0.0,)
    assert all(s.num_blocks == 266 for s in long_)
    tasks, stacked = S.make_demo_pool(SCFG, 300)  # default exclusion = reference streams
    assert len(refs) == 170  # 95 + 75 STAGE1_STREAM registrations (commit D)
    for key in (S.walls_key, S.walls_goal_key, S.layout_key, S.obs_perm_key, S.task_key):
        ref = {key(t) for s in refs for t in s.task_list}
        assert not ({key(t) for t in tasks} & ref), key.__name__
    # goals are compared WITHIN a wall layout (a goal cell alone is just a coordinate)
    ref_wg = {(S.walls_key(t), tuple(t["goal"])) for s in refs for t in s.task_list}
    assert not ({(S.walls_key(t), tuple(t["goal"])) for t in tasks} & ref_wg)
    assert stacked.walls.shape[0] == 300


def test_demo_pool_rejects_walls_only_and_perm_only_clashes(monkeypatch):
    refs = [S.make_stream(SCFG, 906)]
    t0 = refs[0].task_list[0]
    rng = np.random.default_rng(0)
    fresh = S.random_task(rng, SCFG)
    walls_clash = dict(fresh, walls=t0["walls"].copy())              # same walls only
    perm_clash = dict(S.random_task(rng, SCFG), obs_perm=t0["obs_perm"].copy())  # same perm only
    ok = S.random_task(rng, SCFG)
    seq = iter([walls_clash, perm_clash, ok])
    monkeypatch.setattr(S, "reference_streams", lambda specs=None: refs)
    monkeypatch.setattr(S, "random_task", lambda rng, cfg: next(seq))
    tasks, _ = S.make_demo_pool(SCFG, 1, refs)
    assert S.task_key(tasks[0]) == S.task_key(ok)


def test_run_condition_guards_seeds_itself(monkeypatch):
    for seeds in ([500], [609], [1900], [899]):
        with pytest.raises(SystemExit):
            lifelong.run_condition(TINY, SCFG, seeds, verbose=False)

    class Reached(Exception):
        pass

    def stop(*a, **k):  # guard passed; stop before anything is generated or trained
        raise Reached
    monkeypatch.setattr(S, "make_stream", stop)
    monkeypatch.setattr(lifelong, "make_stream_train", stop)  # belt and braces: nothing can train
    with pytest.raises(SystemExit, match="pytest"):  # second guard, flag or not
        lifelong.run_condition(TINY, SCFG, [500], verbose=False, allow_official_seeds=True)
    monkeypatch.delenv("PYTEST_CURRENT_TEST")  # only to show the guard runs before any work
    monkeypatch.setattr(lifelong.provenance, "dirty_files", lambda: [])  # isolate the seed guard
    with pytest.raises(Reached):
        lifelong.run_condition(TINY, SCFG, [500], verbose=False, allow_official_seeds=True)


def test_fresh_baselines_use_scratch_config_not_arm_hooks():
    # shared-layout stream: task 1 is in the kickstart window (1 effective update after the
    # 1-update warm-up), so the kickstart condition is not refused for a zero window
    scfg = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=2, revisit_every=0, block_updates=2,
                               head_probe_every=0)
    net, p_init, _ = _net(TINY)
    arm = dataclasses.replace(TINY, critic_warmup_updates=1, kickstart_coef=0.3, freeze_trunk=True)
    teacher = jax.tree.map(lambda x: x * 0.5, p_init)
    a = lifelong.run_condition(arm, scfg, [916], verbose=False, init_params=p_init, teacher_params=teacher)
    b = lifelong.run_condition(TINY, scfg, [916], verbose=False)
    assert "stream/metrics/critic_warmup" in a and "fresh/metrics/critic_warmup" not in a
    assert "stream/metrics/kickstart_kl" in a and "fresh/metrics/kickstart_kl" not in a
    import json
    fc = json.loads(str(a["fresh_config"]))
    assert fc["critic_warmup_updates"] == 0 and fc["kickstart_coef"] == 0.0 and not fc["freeze_trunk"]
    for k in b:  # fresh runs identical to the plain scratch condition, bit for bit
        if k.startswith("fresh/") and not k.startswith("fresh/device"):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert not np.array_equal(a["stream/metrics/loss"], b["stream/metrics/loss"])


def test_freeze_trunk_and_reinit_heads():
    cfg = ppo.Config(total_timesteps=4 * 128 * 2, eval_every=1, eval_episodes=1, eval_max_steps=5,
                     shared_trunk=True, freeze_trunk=True)
    net, _, shp = _net(cfg)
    out = jax.jit(ppo.make_train(cfg))(jax.random.PRNGKey(0))
    p1 = out["runner_state"].train_state.params["params"]
    pi = net.init(jax.random.split(jax.random.PRNGKey(0), 3)[1], jnp.zeros(shp))["params"]
    for k in pi["trunk"]:
        np.testing.assert_array_equal(p1["trunk"][k]["kernel"], pi["trunk"][k]["kernel"])
    assert not np.array_equal(p1["actor_out"]["kernel"], pi["actor_out"]["kernel"])
    assert not np.array_equal(p1["critic_out"]["kernel"], pi["critic_out"]["kernel"])
    full = {"params": p1}
    r = ppo.reinit_heads(net, full, jax.random.PRNGKey(7), shp)
    np.testing.assert_array_equal(r["params"]["trunk"]["layer_0"]["kernel"], p1["trunk"]["layer_0"]["kernel"])
    assert not np.array_equal(r["params"]["actor_out"]["kernel"], p1["actor_out"]["kernel"])
    fresh = net.init(jax.random.PRNGKey(7), jnp.zeros(shp))["params"]
    np.testing.assert_array_equal(r["params"]["critic_out"]["kernel"], fresh["critic_out"]["kernel"])


def test_head_probe_runs_on_every_nth_new_task_and_is_analysed():
    scfg = S.StreamConfig(num_new_tasks=4, revisit_every=3, block_updates=2, head_probe_every=2)
    res = lifelong.run_condition(TINY, scfg, [917], verbose=False)
    assert res["head_probe_tasks"].tolist() == [2, 4]
    tob = res["task_of_block"][0]
    for t, b in zip(res["head_probe_tasks"], res["head_probe_blocks"]):
        assert tob[b - 1] == t and res["is_new"][0, b - 1]
    K = scfg.block_updates
    assert res["headprobe/metrics/rollout_return"].shape == (1, 2, 1, K)
    # re-initialised heads -> near-uniform policy at the probe start, whatever the stream did
    np.testing.assert_allclose(res["headprobe/init_entropy"], np.log(4), atol=1e-2)
    import json
    assert json.loads(str(res["head_probe_config"]))["freeze_trunk"] is True
    r = SM.analyze(res)
    hp = r["head_probe"]
    assert hp["auc_rel_headprobe"].shape == (1, 2) and np.isfinite(hp["auc_rel_headprobe"]).all()
    np.testing.assert_allclose(hp["auc_rel_stream"], r["auc_rel"][:, res["head_probe_blocks"] - 1])
    assert "stochastic" in r["auc_source"] and "stochastic" in str(res["auc_source"])
    off = lifelong.run_condition(TINY, dataclasses.replace(scfg, head_probe_every=0), [917], verbose=False)
    assert "head_probe_tasks" not in off
    np.testing.assert_array_equal(off["stream/metrics/loss"], res["stream/metrics/loss"])  # probe is passive


def test_snapshot_is_params_at_the_switch():
    scfg = S.StreamConfig(num_new_tasks=3, revisit_every=0, block_updates=2)
    streams, tasks, sched, probe = _stream_inputs(scfg, [912])
    one = lambda x: x[0]  # noqa: E731
    key = jax.random.PRNGKey(912)
    full = jax.jit(lifelong.make_stream_train(TINY, scfg, 3, snapshot_blocks=(2,)))(
        key, jax.tree.map(one, tasks), sched[0], probe[0])
    two = jax.jit(lifelong.make_stream_train(TINY, scfg, 2))(
        key, jax.tree.map(one, tasks), sched[0, :2], probe[0])
    jax.tree.map(lambda s, p: np.testing.assert_allclose(s[0], p, atol=1e-6),
                 full["snapshots"], two["final_params"])


def test_switch_entropy_is_probe_batch_mean():
    scfg = S.StreamConfig(num_new_tasks=2, revisit_every=0, block_updates=2)
    streams, tasks, sched, probe = _stream_inputs(scfg, [911])
    key = jax.random.PRNGKey(911)
    out = jax.jit(lifelong.make_stream_train(TINY, scfg, 2))(
        key, jax.tree.map(lambda x: x[0], tasks), sched[0], probe[0])
    _, _, net, shp = ppo.make_env_and_network(TINY)
    p0 = net.init(jax.random.split(key, 3)[1], jnp.zeros(shp))
    np.testing.assert_allclose(out["switch_entropy"][0], ppo.initial_entropy(net, p0, probe[0]), rtol=1e-6)
    np.testing.assert_allclose(out["init_entropy"], out["switch_entropy"][0], rtol=1e-6)
    assert out["switch_entropy_start"].shape == (2,)


def _stub_pilot(tmp_path):
    """Copy of run_pilot.sh in a tmp tree whose run/plot commands are replaced by echo. The
    tmp tree has no lifelong.py / plot_stream.py at all, only a .venv symlink for the backend
    check, so it can never train on any platform. The real script is never executed."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    txt = open(os.path.join(root, "scripts/run_pilot.sh")).read()
    stub = txt.replace(".venv/bin/python lifelong.py", "echo STUB lifelong.py").replace(
        ".venv/bin/python plot_stream.py", "echo STUB plot_stream.py")
    assert stub.count("echo STUB") == 2 and "lifelong.py \\" not in stub.replace("STUB lifelong.py", "")
    (tmp_path / "scripts").mkdir()
    sh = tmp_path / "scripts" / "run_pilot.sh"
    sh.write_text(stub)
    # the 200-task wrapper only execs run_pilot.sh next to it, i.e. the stub
    wrapper = open(os.path.join(root, "scripts/run_null200.sh")).read()
    assert "python" not in wrapper and 'exec bash "$(dirname "$0")/run_pilot.sh"' in wrapper
    (tmp_path / "scripts" / "run_null200.sh").write_text(wrapper)
    os.symlink(os.path.join(root, ".venv"), tmp_path / ".venv")
    return root, txt, str(sh)


def _results_snapshot(root):
    """(path, size, mtime) of everything under the repo's results/: the stubbed scripts must
    not create, modify or remove anything there (incl. the finished pilot's outputs)."""
    snap = []
    for dp, dns, fns in os.walk(os.path.join(root, "results")):
        for n in dns + fns:
            st = os.stat(os.path.join(dp, n))
            snap.append((os.path.join(dp, n), st.st_size if n in fns else 0, st.st_mtime_ns))
    return sorted(snap)


def test_pilot_script_pinned_and_gated(tmp_path):
    import subprocess
    root, txt, sh = _stub_pilot(tmp_path)
    before = _results_snapshot(root)
    for frag in ("--num-new-tasks 40", "--block-updates 100", "--revisit-every 3",
                 "--hidden-size 950", "--seeds 500-509 --allow-official-seeds",
                 "--head-probe-every 5", "--preset lifelong_stream", '--matmul-precision "$PREC"',
                 "--require-gpu", "TARS_PILOT_DRY_RUN", "§5", "--require-device-info"):
        assert frag in txt, frag
    assert "sec. 7" not in txt and "box CPU" not in txt
    out = tmp_path / "out_pilot_500_509"
    env = dict(os.environ, OUT=str(out), TARS_PILOT_DRY_RUN="1")
    env.pop("PYTEST_CURRENT_TEST", None)  # irrelevant here: nothing can run, but be explicit

    def run(*args):
        return subprocess.run(["bash", sh, *args], env=env, capture_output=True, text=True, timeout=120)
    for args, msg in ((["--confirmed"], "must be given explicitly"),
                      (["--confirmed", "--matmul-precision", "default"], "only 'highest'"),
                      (["--confirmed", "--matmul-precision", "fast"], "only 'highest'"),
                      (["--matmul-precision", "highest"], "not authorised"),
                      (["--confirmed", "--matmul-precision", "highest", "--bogus"], "unknown argument")):
        r = run(*args)
        assert r.returncode == 2 and msg in r.stderr, (args, r.stderr)
    backend = jax.default_backend()
    if backend != "gpu":  # CPU refused unless --allow-cpu
        r = run("--confirmed", "--matmul-precision", "highest")
        assert r.returncode == 2 and "not 'gpu'" in r.stderr and "--allow-cpu" in r.stderr, r.stderr
    args = ["--confirmed", "--matmul-precision", "highest"] + (["--allow-cpu"] if backend == "cpu" else [])
    r = run(*args)
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN:")]
    cmd = next(ln for ln in lines if "STUB lifelong.py" in ln)
    for frag in ("--seeds 500-509", "--matmul-precision highest", "--require-gpu", "--head-probe-every 5"):
        assert frag in cmd, frag
    assert ("--allow-cpu" in cmd) == (backend == "cpu")
    assert any("STUB plot_stream.py" in ln and "--require-device-info" in ln for ln in lines)
    # nothing created: no output dir for the 500-509 pilot, here or in the repo
    assert not out.exists() and sorted(p.name for p in tmp_path.iterdir()) == [".venv", "scripts"]
    assert _results_snapshot(root) == before  # repo results/ untouched
    assert lifelong.param_count(dataclasses.replace(ppo.Config(**ppo.PRESETS["lifelong_stream"]),
                                                    hidden_size=950)) == 1_091_555


# ------------------------------------------------------------------ entropy-adjusted slope, matmul precision
def _entropy_case(explained, S_=8, K=10, seed=0):
    """Blocks: tasks 1,2,3, revisit 1, task 4..7. Stream drop d per new task trends with the
    task number; switch entropy = 1.4 - 2 d (+ noise unrelated to d when not explained)."""
    rng = np.random.default_rng(seed)
    tob = np.array([[1, 2, 3, 1, 4, 5, 6, 7]] * S_)
    is_new = np.array([[1, 1, 1, 0, 1, 1, 1, 1]] * S_, bool)
    rr, ent = [], []
    for _ in range(S_):
        d = np.array([0, *(0.06 * np.arange(1, 4) + rng.uniform(0, 0.15, 3)), 0,
                      *(0.06 * np.arange(4, 7) + rng.uniform(0, 0.15, 3))])
        rr.append(np.stack([np.linspace(0, 1 - x, K) for x in d]))
        ent.append(1.4 - 2 * d if explained else 1.0 + rng.uniform(0, 0.3, len(d)))
    return {"task_of_block": tob, "is_new": is_new, "optimal_return": np.ones((S_, 7)), "batch_size": 128,
            "stream/metrics/rollout_return": np.stack(rr),
            "fresh/metrics/rollout_return": np.stack([np.stack([np.linspace(0, 1, K)[None]] * 7)] * S_),
            "block_shift": np.array([[0, 1, 2, 5, 3, 1, 2, 3]] * S_),
            "stream/eval": np.ones((S_, 8, 7)), "stream/eval0": np.zeros((S_, 7)),
            "stream/switch_entropy": np.stack(ent)}


def test_entropy_adjusted_slope_is_zero_when_entropy_explains_the_decline():
    key = SM.PRIMARY + "_entropy_adj"
    r = SM.analyze(_entropy_case(explained=True))
    assert r["primary"] == "slope_auc_rel_new_excl_task1"                # primary unchanged
    assert r["slope_auc_rel_new_excl_task1_summary"]["iqm"] < -0.02       # raw decline present
    assert r["slope_auc_rel_new_excl_task1_summary"]["ci_excludes_zero"]
    np.testing.assert_allclose(r[key], 0.0, atol=1e-9)                  # fully explained
    sm = r[key + "_summary"]
    assert sm["n"] == 8 and abs(sm["iqm"]) < 1e-9 and not sm["ci_excludes_zero"]
    # Control: entropy unrelated to the drop -> adjusted slope keeps the decline.
    r2 = SM.analyze(_entropy_case(explained=False))
    assert r2[key + "_summary"]["iqm"] < -0.02 and r2[key + "_summary"]["ci_excludes_zero"]
    # Same seeds-bootstrap machinery as the primary (IQM + 10k percentile bootstrap).
    assert sm["ci95"] == SM.summarize(r[key])["ci95"]


def test_entropy_adjusted_slope_nan_when_underdetermined():
    d = _synthetic([0.0, 0.1, 0.2, 0.0, 0.3, 0.4])  # 3 new tasks after task 1: < 4 points
    d["stream/switch_entropy"] = np.ones((6, 6))
    r = SM.analyze(d)
    assert np.isnan(r[SM.PRIMARY + "_entropy_adj"]).all()
    assert r[SM.PRIMARY + "_entropy_adj_summary"]["n"] == 0


def test_matmul_precision_recorded_and_mixing_refused():
    try:
        assert ppo.device_info()["matmul_precision"] == "default"
        ppo.set_matmul_precision("highest")
        assert ppo.device_info()["matmul_precision"] == "highest"
    finally:
        ppo.set_matmul_precision("default")
    assert jax.config.jax_default_matmul_precision is None
    d = _synthetic([0.0, 0.1, 0.2, 0.0, 0.3, 0.4])
    d["stream/device_platform"] = d["fresh/device_platform"] = "cpu"
    with pytest.raises(SM.PlatformMismatch):  # precision not recorded, strict mode
        SM.analyze(d, require_device_info=True)
    d["stream/matmul_precision"], d["fresh/matmul_precision"] = "highest", "default"
    with pytest.raises(SM.PlatformMismatch, match="matmul_precision"):
        SM.analyze(d)
    with pytest.raises(SM.PlatformMismatch):
        lifelong.relative_metrics(d)
    d["fresh/matmul_precision"] = "highest"
    assert SM.analyze(d, require_device_info=True)["matmul_precision"] == "highest"


def test_lifelong_cli_matmul_precision_flag(tmp_path, monkeypatch):
    out = tmp_path / "r.npz"
    monkeypatch.setattr(sys, "argv", ["lifelong.py", "--seeds", "919", "--out", str(out),
                                      "--num-new-tasks", "2", "--revisit-every", "0", "--block-updates", "1",
                                      "--hidden-size", "8", "--matmul-precision", "highest"])
    try:
        lifelong.main()
    finally:
        ppo.set_matmul_precision("default")
    d = np.load(out)
    for k in ("matmul_precision", "stream/matmul_precision", "fresh/matmul_precision"):
        assert str(d[k]) == "highest", k
    assert SM.analyze(dict(d), require_device_info=True)["matmul_precision"] == "highest"
    monkeypatch.setattr(sys, "argv", ["lifelong.py", "--seeds", "919", "--out", str(out),
                                      "--matmul-precision", "fast"])
    with pytest.raises(SystemExit):
        lifelong.main()


def test_demo_pool_refuses_streams_beyond_reference_set(monkeypatch):
    monkeypatch.setattr(S, "_REFERENCE_SPECS", list(S._REFERENCE_SPECS))  # restored after the test
    long_cfg = dataclasses.replace(S.PILOT_STREAM, num_new_tasks=200)
    long_stream = S.make_stream(long_cfg, 900)
    with pytest.raises(ValueError, match="extend_reference_set"):
        S.make_demo_pool(SCFG, 5, [long_stream])
    with pytest.raises(ValueError):                     # unregistered seed, any length
        S.make_demo_pool(SCFG, 5, [S.make_stream(SCFG, 1500)])
    S.make_demo_pool(SCFG, 5, [S.make_stream(SCFG, 905)])  # covered prefix: fine
    S.extend_reference_set(long_cfg, [900])
    assert (long_cfg, (900,)) in S.reference_specs()
    tasks, _ = S.make_demo_pool(SCFG, 20, [long_stream])
    ref_w = {S.walls_key(t) for t in long_stream.task_list}
    ref_p = {S.obs_perm_key(t) for t in long_stream.task_list}
    assert not ({S.walls_key(t) for t in tasks} & ref_w) and not ({S.obs_perm_key(t) for t in tasks} & ref_p)


def test_plot_stream_require_device_info(tmp_path, monkeypatch):
    import plot_stream
    d = _synthetic([0.0, 0.1, 0.2, 0.0, 0.3, 0.4])
    np.savez(tmp_path / "x.npz", **d)
    monkeypatch.setattr(sys, "argv", ["plot_stream.py", str(tmp_path / "x.npz"), "--out-dir",
                                      str(tmp_path / "o"), "--require-device-info"])
    with pytest.raises(SM.PlatformMismatch):
        plot_stream.main()


# ------------------------------------------------------------------ second seed guard, CPU gate, precision
def test_every_cli_refuses_official_seeds_under_pytest(monkeypatch, tmp_path):
    """lifelong.py (even with --allow-official-seeds), time_stream.py and ppo.py all refuse
    500-509 / 600-609 under pytest, before any compile or training; no output is written."""
    import time_stream

    def boom(*a, **k):
        raise AssertionError("training reached")
    monkeypatch.setattr(lifelong, "make_stream_train", boom)
    monkeypatch.setattr(ppo, "make_train", boom)
    out = tmp_path / "x.npz"
    for argv, mod in ((["lifelong.py", "--seeds", "500-509", "--allow-official-seeds", "--out", str(out)], lifelong),
                      (["lifelong.py", "--seeds", "505", "--allow-official-seeds", "--out", str(out)], lifelong),
                      (["time_stream.py", "--seeds", "500-509"], time_stream),
                      (["ppo.py", "--seeds", "500-509", "--out", str(out)], ppo),
                      (["ppo.py", "--seeds", "600", "--out", str(out)], ppo)):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            mod.main()
        # the seed guard fired (a message), not argparse (exit code 2)
        assert isinstance(e.value.code, str) and "reserved" in e.value.code, (argv, e.value.code)
    assert not out.exists()


def test_lifelong_require_gpu_and_allow_cpu(monkeypatch, tmp_path):
    out = tmp_path / "r.npz"
    base = ["lifelong.py", "--seeds", "919", "--out", str(out), "--num-new-tasks", "2",
            "--revisit-every", "0", "--block-updates", "1", "--hidden-size", "8"]
    if jax.default_backend() == "gpu":
        pytest.skip("CPU gate only testable on a non-GPU backend")
    monkeypatch.setattr(sys, "argv", base + ["--require-gpu"])
    with pytest.raises(SystemExit, match="allow-cpu"):
        lifelong.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", base + ["--require-gpu", "--allow-cpu", "--head-probe-every", "2"])
    lifelong.main()
    d = np.load(out)
    assert bool(d["allow_cpu"]) and bool(d["require_gpu"]) and str(d["default_backend"]) == "cpu"
    assert str(d["device_platform"]) == "cpu"
    for pre in ("stream/", "fresh/", "headprobe/"):
        assert str(d[pre + "device_platform"]) == "cpu" and str(d[pre + "matmul_precision"]) == "default"
    assert SM.analyze(dict(d), require_device_info=True)["platform"] == "cpu"


def test_head_probe_platform_or_precision_mismatch_refused():
    d = _synthetic([0.0, 0.1, 0.2, 0.0, 0.3, 0.4])
    d["head_probe_tasks"], d["head_probe_blocks"] = np.array([2]), np.array([2])
    d["headprobe/metrics/rollout_return"] = d["fresh/metrics/rollout_return"][:, 1:2]
    for k in ("device_platform", "matmul_precision"):
        d[f"stream/{k}"] = d[f"fresh/{k}"] = {"device_platform": "cpu", "matmul_precision": "highest"}[k]
    with pytest.raises(SM.PlatformMismatch, match="headprobe"):  # not recorded, strict
        SM.analyze(d, require_device_info=True)
    d["headprobe/device_platform"], d["headprobe/matmul_precision"] = "gpu", "highest"
    with pytest.raises(SM.PlatformMismatch, match="head probe"):
        SM.analyze(d)
    d["headprobe/device_platform"], d["headprobe/matmul_precision"] = "cpu", "default"
    with pytest.raises(SM.PlatformMismatch, match="head probe"):
        SM.analyze(d)
    d["headprobe/matmul_precision"] = "highest"
    assert "head_probe" in SM.analyze(d, require_device_info=True)


def test_ppo_cli_matmul_precision_recorded(tmp_path, monkeypatch):
    out = tmp_path / "r.npz"
    monkeypatch.setattr(sys, "argv", ["ppo.py", "--total-timesteps", "1024", "--num-seeds", "1",
                                      "--eval-every", "1", "--out", str(out), "--matmul-precision", "highest"])
    try:
        ppo.main()
    finally:
        ppo.set_matmul_precision("default")
    assert str(np.load(out)["matmul_precision"]) == "highest"


def test_null200_streams_covered_and_40_task_streams_still_covered():
    """The demo pool accepts the 200-task official streams and every 40-task / dev stream
    (layouts only, no agent), and still refuses longer or unregistered ones."""
    for sd in range(500, 510):
        S.make_demo_pool(SCFG, 3, [S.make_stream(S.NULL200_STREAM, sd)])
        S.make_demo_pool(SCFG, 3, [S.make_stream(S.PILOT_STREAM, sd)])
    S.make_demo_pool(SCFG, 3, [S.make_stream(S.PILOT_STREAM, sd) for sd in (900, 939)])
    with pytest.raises(ValueError, match="extend_reference_set"):
        S.make_demo_pool(SCFG, 3, [S.make_stream(dataclasses.replace(S.NULL200_STREAM, num_new_tasks=201), 500)])
    with pytest.raises(ValueError):  # dev seeds are only covered at 40 tasks
        S.make_demo_pool(SCFG, 3, [S.make_stream(S.NULL200_STREAM, 900)])
    # the 40-task official streams are exact prefixes of the 200-task ones
    for sd in (500, 509):
        a, b = S.make_stream(S.PILOT_STREAM, sd), S.make_stream(S.NULL200_STREAM, sd)
        assert [S.task_key(t) for t in b.task_list[:40]] == [S.task_key(t) for t in a.task_list]


def test_null200_script_stubbed_dry_run(tmp_path):
    import subprocess
    root, txt, sh = _stub_pilot(tmp_path)
    before = _results_snapshot(root)
    wrapper = str(tmp_path / "scripts" / "run_null200.sh")
    out = tmp_path / "out_null_200_500_509"
    env = dict(os.environ, OUT=str(out), TARS_PILOT_DRY_RUN="1")
    env.pop("PYTEST_CURRENT_TEST", None)

    def run(*args):
        return subprocess.run(["bash", wrapper, *args], env=env, capture_output=True, text=True, timeout=120)
    for args, msg in ((["--confirmed"], "must be given explicitly"),
                      (["--confirmed", "--matmul-precision", "default"], "only 'highest'"),
                      (["--matmul-precision", "highest"], "not authorised"),
                      (["--confirmed", "--matmul-precision", "highest", "--num-new-tasks", "40"], "fixed at 200")):
        r = run(*args)
        assert r.returncode == 2 and msg in r.stderr, (args, r.stderr)
    r = subprocess.run(["bash", sh, "--confirmed", "--matmul-precision", "highest", "--num-new-tasks", "100"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "40 (pilot) or 200" in r.stderr
    backend = jax.default_backend()
    if backend != "gpu":
        r = run("--confirmed", "--matmul-precision", "highest")
        assert r.returncode == 2 and "--allow-cpu" in r.stderr, r.stderr
    r = run("--confirmed", "--matmul-precision", "highest", *(["--allow-cpu"] if backend == "cpu" else []))
    assert r.returncode == 0, r.stderr
    cmd = next(ln for ln in r.stdout.splitlines() if "STUB lifelong.py" in ln)
    for frag in ("--seeds 500-509 --allow-official-seeds", "--num-new-tasks 200", "--block-updates 100",
                 "--revisit-every 3", "--head-probe-every 5", "--hidden-size 950",
                 "--matmul-precision highest", "--require-gpu", f"--out {out}/pilot.npz"):
        assert frag in cmd, frag
    assert ("--allow-cpu" in cmd) == (backend == "cpu")
    assert not out.exists() and sorted(p.name for p in tmp_path.iterdir()) == [".venv", "scripts"]
    # default output dir for the 200-task run (without OUT): results/null_200, still not created
    env.pop("OUT")
    r = run("--confirmed", "--matmul-precision", "highest", *(["--allow-cpu"] if backend == "cpu" else []))
    assert r.returncode == 0 and "--out results/null_200/pilot.npz" in r.stdout
    assert not (tmp_path / "results").exists()
    # (results/null_200 exists once the real run has finished; the dry run must not create it)
    assert os.path.exists(os.path.join(root, "results", "null_200")) == any(
        p == os.path.join(root, "results", "null_200") for p, _, _ in before)
    assert _results_snapshot(root) == before  # repo results/ untouched


# ------------------------------------------------------------------ stage-1 seed blocks + reference set (30 Oct)
STAGE1_GUARD_SEEDS = [700, 724, 750, 759, 800, 824, 850, 859]


def test_stage1_fresh_seed_mapping():
    assert [lifelong.fresh_seed(s) for s in range(700, 725)] == list(range(800, 825))
    assert [lifelong.fresh_seed(s) for s in range(750, 760)] == list(range(850, 860))
    assert [lifelong.fresh_seed(s) for s in range(500, 510)] == list(range(600, 610))  # unchanged
    for s in (510, 699, 725, 749, 760, 799, 800, 824, 850, 859, 899):  # no mapping outside blocks
        with pytest.raises(ValueError):
            lifelong.fresh_seed(s)
    assert lifelong.seed_block(700) == lifelong.seed_block(724) == "stage1"
    assert lifelong.seed_block(750) == lifelong.seed_block(759) == "sweep"
    assert lifelong.seed_block(505) == "pilot" and lifelong.seed_block(800) is None


def test_all_seed_blocks_pairwise_disjoint():
    blocks = {"pilot": set(range(500, 510)), "pilot_fresh": set(range(600, 610)),
              "stage1": set(range(700, 725)), "stage1_fresh": set(range(800, 825)),
              "sweep": set(range(750, 760)), "sweep_fresh": set(range(850, 860)),
              "dev": set(range(lifelong.DEV_MIN_SEED, lifelong.DEV_MAX_SEED)),
              "dev_fresh": {lifelong.fresh_seed(s) for s in range(lifelong.DEV_MIN_SEED, lifelong.DEV_MAX_SEED)}}
    assert set(range(1930, 1940)) <= blocks["dev_fresh"]  # dev fresh seeds used so far
    names = sorted(blocks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (blocks[a] & blocks[b]), (a, b)
    # the module's blocks are exactly these, fresh = stream + 100
    for name, (st, fr) in lifelong.SEED_BLOCKS.items():
        assert st == blocks[name] and fr == blocks[name + "_fresh"] == {s + 100 for s in st}
    assert lifelong.RESERVED_SEEDS == set().union(*(blocks[n] for n in names if not n.startswith("dev")))
    import seeds
    # one source of truth: ppo.py and lifelong.py both use seeds.py's objects
    assert ppo.RESERVED_SEEDS_PYTEST is seeds.RESERVED_SEEDS and lifelong.RESERVED_SEEDS is seeds.RESERVED_SEEDS
    assert lifelong.SEED_BLOCKS is seeds.SEED_BLOCKS and lifelong.fresh_seed is seeds.fresh_seed
    assert S.STAGE1_SEEDS == tuple(sorted(seeds.STAGE1_STREAM_SEEDS))
    assert S.SWEEP_SEEDS == tuple(sorted(seeds.SWEEP_STREAM_SEEDS))
    assert S.OFFICIAL_SEEDS == tuple(sorted(seeds.PILOT_STREAM_SEEDS))


def test_seeds_module_is_dependency_free():
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_mods = {f[:-3] for f in os.listdir(root) if f.endswith(".py")}
    tree = ast.parse(open(os.path.join(root, "seeds.py")).read())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported, imported  # no imports at all, so certainly none from the repo
    assert "seeds" in repo_mods
    # the hand-copied ranges are gone from ppo.py
    src = open(os.path.join(root, "ppo.py")).read()
    assert "range(500, 510)" not in src and "500 <= s <= 509" not in src


@pytest.mark.parametrize("seed", STAGE1_GUARD_SEEDS)
def test_pytest_guard_refuses_stage1_and_sweep_blocks(seed):
    assert lifelong.under_pytest()
    for flag in (False, True):
        with pytest.raises(SystemExit, match="pytest"):
            lifelong.check_seeds([seed], allow_official=flag)
        with pytest.raises(SystemExit, match="pytest"):
            lifelong.check_seeds([900, seed], allow_official=flag)
    with pytest.raises(SystemExit, match="reserved"):
        ppo._refuse_reserved_seeds_under_pytest([seed])
    with pytest.raises(SystemExit):
        lifelong.run_condition(TINY, SCFG, [seed], verbose=False, allow_official_seeds=True)


def test_stage1_blocks_gated_like_pilot_block(monkeypatch):
    """Flag logic with the pytest marker removed (pure function of the seed list; nothing
    downstream of check_seeds is reachable from here)."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    for seeds in ([700], [724], [750], [759], [800], [859], [725], [749], [760]):
        with pytest.raises(SystemExit):  # without the flag: refused
            lifelong.check_seeds(seeds)
    assert lifelong.check_seeds(list(range(700, 725)), allow_official=True) == list(range(800, 825))
    assert lifelong.check_seeds(list(range(750, 760)), allow_official=True) == list(range(850, 860))
    assert lifelong.check_seeds(list(range(500, 510)), allow_official=True) == list(range(600, 610))
    for seeds in ([800], [824], [850], [859], [725], [749], [760], [799]):
        with pytest.raises(SystemExit):  # fresh seeds and gaps are never stream seeds
            lifelong.check_seeds(seeds, allow_official=True)
    for seeds in ([700, 750], [505, 700], [759, 500]):  # one reserved block per run
        with pytest.raises(SystemExit, match="one reserved block"):
            lifelong.check_seeds(seeds, allow_official=True)
    # a reserved run contains nothing else: no dev (or other non-reserved) seeds mixed in
    for seeds in ([700, 900], [750, 951], [505, 900], [900, 724], [759, 1899], list(range(500, 510)) + [939]):
        with pytest.raises(SystemExit, match="non-reserved"):
            lifelong.check_seeds(seeds, allow_official=True)
    assert lifelong.check_seeds([900, 919], allow_official=True) == [1900, 1919]  # dev-only still fine


def test_cli_refuses_stage1_seeds(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("training reached")
    monkeypatch.setattr(lifelong, "make_stream_train", boom)
    monkeypatch.setattr(ppo, "make_train", boom)
    out = tmp_path / "x.npz"
    for argv, mod in ((["lifelong.py", "--seeds", "700-724", "--out", str(out)], lifelong),
                      (["lifelong.py", "--seeds", "750-759", "--allow-official-seeds", "--out", str(out)], lifelong),
                      (["ppo.py", "--seeds", "700-701", "--out", str(out)], ppo),
                      (["ppo.py", "--seeds", "855", "--out", str(out)], ppo)):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as e:
            mod.main()
        assert isinstance(e.value.code, str) and "reserved" in e.value.code, (argv, e.value.code)
    assert not out.exists()


def test_reference_set_covers_stage1_and_sweep_streams():
    specs = S.reference_specs()
    assert (S.PILOT_STREAM, tuple(range(700, 725))) in specs
    assert (S.PILOT_STREAM, tuple(range(750, 760))) in specs
    assert (S.STAGE1_STREAM, tuple(range(700, 725))) in specs
    assert (S.STAGE1_STREAM, tuple(range(750, 760))) in specs
    assert S.STAGE1_STREAM.num_new_tasks == 40 and S.STAGE1_STREAM.block_updates == 100
    # stage 1 = pilot geometry + shared layout; PILOT_STREAM itself is untouched
    assert S.STAGE1_STREAM != S.PILOT_STREAM
    assert dataclasses.replace(S.STAGE1_STREAM, shared_obs_layout=False, shared_action_perm=False) == S.PILOT_STREAM
    assert not S.PILOT_STREAM.shared_obs_layout and not S.PILOT_STREAM.shared_action_perm
    refs = S.reference_streams()
    st1 = [S.make_stream(S.STAGE1_STREAM, sd) for sd in [*range(700, 725), *range(750, 760)]]
    assert not S.uncovered_streams(st1, refs)
    new = [S.make_stream(S.PILOT_STREAM, sd) for sd in [*range(700, 725), *range(750, 760)]]
    assert not S.uncovered_streams(new, refs)
    assert all(s.num_tasks == 40 and s.num_blocks == 53 for s in new)
    # shorter prefixes are covered too; a longer stage-1 stream is not
    assert not S.uncovered_streams([S.make_stream(dataclasses.replace(S.PILOT_STREAM, num_new_tasks=10), 712)], refs)
    assert S.uncovered_streams([S.make_stream(dataclasses.replace(S.PILOT_STREAM, num_new_tasks=41), 712)], refs)
    # a demo pool for the new streams is allowed and disjoint from them component-wise
    tasks, _ = S.make_demo_pool(S.PILOT_STREAM, 300, new)
    for key in (S.walls_key, S.walls_goal_key, S.layout_key, S.obs_perm_key, S.task_key):
        assert not ({key(t) for t in tasks} & {key(t) for s in new for t in s.task_list}), key.__name__
