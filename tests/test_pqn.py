"""Step 5: PQN (pqn.py, cells 15-16). LayerNorm Q-network and Q(lambda) targets against
references (incl. a transcription of purejaxql), epsilon-greedy, tiny-task learning sanity
on dev seeds, PQN cell refusals, same-layout chunk bit-identity. Dev seeds 990-992 only."""
import dataclasses
import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import arms
import bc
import cells
import lifelong
import ppo
import pqn
import provenance
import stream as S

TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=16, eval_episodes=1))
SCFG = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=2, revisit_every=0, block_updates=2,
                           head_probe_every=0)
NONDET = set(cells.TIMING_KEYS) | {"chunk_" + k for k in cells.TIMING_KEYS} | {"cell_manifest_sha256"}


def _same(a, b, skip=()):
    assert set(a) - set(skip) == set(b) - set(skip), sorted(set(a) ^ set(b))
    for k in a:
        if k in skip or k in NONDET:
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype.kind in "OUS":
            assert str(x) == str(y), k
        else:
            assert x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), k


# ----------------------------------------------------------------------------- network
class PurejaxqlMLP(nn.Module):
    """purejaxql pqn_gymnax.QNetwork (norm_type layer_norm, norm_input False), minus its
    unused dummy BatchNorm: Dense -> LayerNorm -> relu per layer, then Dense."""
    action_dim: int
    hidden_size: int
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.relu(nn.LayerNorm()(nn.Dense(self.hidden_size)(x)))
        return nn.Dense(self.action_dim)(x)


def test_qnetwork_layernorm_matches_purejaxql_and_numpy():
    c = pqn.pqn_config(dataclasses.replace(TINY, hidden_size=16))
    _, _, net, obs_shape = ppo.make_env_and_network(c)
    assert isinstance(net, ppo.QNetwork) and net.layer_norm
    params = net.init(jax.random.PRNGKey(990), jnp.zeros(obs_shape))
    p = params["params"]
    assert set(p) == {"trunk", "q_out"}
    assert set(p["trunk"]) == {"layer_0", "layer_0_ln", "layer_1", "layer_1_ln"}
    obs = jax.random.uniform(jax.random.PRNGKey(1), (5,) + obs_shape)
    q, v, _ = net.apply(params, obs)
    assert q.shape == (5, 4) and np.allclose(v, q.max(-1))
    # purejaxql's module with our weights
    ref = PurejaxqlMLP(4, 16)
    t = p["trunk"]
    rp = {"params": {"Dense_0": t["layer_0"], "LayerNorm_0": t["layer_0_ln"], "Dense_1": t["layer_1"],
                     "LayerNorm_1": t["layer_1_ln"], "Dense_2": p["q_out"]}}
    x = obs.reshape(5, -1)
    np.testing.assert_allclose(ref.apply(rp, x), q, rtol=1e-6, atol=1e-6)
    # plain numpy
    h = np.asarray(x, np.float64)
    for i in range(2):
        h = h @ np.asarray(t[f"layer_{i}"]["kernel"]) + np.asarray(t[f"layer_{i}"]["bias"])
        h = (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + 1e-6)
        h = np.maximum(h * np.asarray(t[f"layer_{i}_ln"]["scale"]) + np.asarray(t[f"layer_{i}_ln"]["bias"]), 0)
    np.testing.assert_allclose(h @ np.asarray(p["q_out"]["kernel"]) + np.asarray(p["q_out"]["bias"]), q,
                               rtol=1e-5, atol=1e-5)
    # trunk has the PPO+LN trunk's paths and shapes (BC graft); masks treat q_out as the head
    _, _, ac, _ = ppo.make_env_and_network(dataclasses.replace(c, algo="ppo"))
    pa = ac.init(jax.random.PRNGKey(0), jnp.zeros(obs_shape))
    assert jax.tree.map(jnp.shape, pa["params"]["trunk"]) == jax.tree.map(jnp.shape, t)
    cm, tm = ppo.critic_only_mask(params)["params"], ppo.trunk_mask(params)["params"]
    assert all(jax.tree.leaves(cm["trunk"])) and not any(jax.tree.leaves(cm["q_out"]))
    assert all(jax.tree.leaves(tm["trunk"])) and not any(jax.tree.leaves(tm["q_out"]))
    re = ppo.reinit_heads(net, params, jax.random.PRNGKey(5), obs_shape)["params"]
    assert not np.array_equal(re["q_out"]["kernel"], p["q_out"]["kernel"])
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(re["trunk"]), jax.tree.leaves(t)))


# ----------------------------------------------------------------------------- lambda returns
def _ref_lambda(r, q, d, last_q, gamma, lam):
    T = r.shape[0]
    g = np.zeros_like(r, dtype=np.float64)
    g[T - 1] = r[T - 1] + gamma * (1 - d[T - 1]) * last_q
    for t in range(T - 2, -1, -1):
        nq = q[t + 1].max(-1)
        g[t] = r[t] + gamma * (1 - d[t]) * ((1 - lam) * nq + lam * g[t + 1])
    return g


def _purejaxql(r, q, d, last_q, gamma, lam):
    """Transcription of purejaxql pqn_gymnax/_minatar target code."""
    def get_target(carry, tr):
        lr_, next_q = carry
        rew, done, qv = tr
        boot = rew + gamma * (1 - done) * next_q
        lr_ = boot + gamma * lam * (lr_ - next_q)
        lr_ = (1 - done) * lr_ + done * rew
        return (lr_, jnp.max(qv, -1)), lr_
    lq = last_q * (1 - d[-1])
    lr_ = r[-1] + gamma * lq
    _, t = jax.lax.scan(get_target, (lr_, lq), (r[:-1], d[:-1], q[:-1]), reverse=True)
    return jnp.concatenate([t, lr_[None]])


@pytest.mark.parametrize("lam", [0.0, 0.65, 0.95, 1.0])
def test_lambda_returns_match_reference(lam):
    rng = np.random.default_rng(990)
    T, N, A = 9, 3, 4
    r = rng.uniform(-1, 1, (T, N)).astype(np.float32)
    q = rng.normal(size=(T, N, A)).astype(np.float32)
    d = (rng.uniform(size=(T, N)) < 0.25).astype(np.float32)
    last_q = rng.normal(size=N).astype(np.float32)
    got = np.asarray(pqn.lambda_returns(r, q, d, last_q, 0.99, lam))
    np.testing.assert_allclose(got, _ref_lambda(r, q, d, last_q, 0.99, lam), rtol=1e-5, atol=1e-5)
    if lam == 0.0:  # one-step Q-learning targets
        one = r[:-1] + 0.99 * (1 - d[:-1]) * q[1:].max(-1)
        np.testing.assert_allclose(got[:-1], one, rtol=1e-6, atol=1e-6)
    # purejaxql: its t = T-2 step bootstraps from max Q(s_T) (masked by d_{T-1}) instead of
    # max Q(s_{T-1}). At lambda = 0 only that entry differs (for lambda > 0 the difference
    # also flows back into earlier G_t); identical everywhere once the two coincide.
    pj = np.asarray(_purejaxql(r, q, d, last_q, 0.99, lam))
    if lam == 0.0:
        np.testing.assert_allclose(np.delete(got, T - 2, 0), np.delete(pj, T - 2, 0), rtol=1e-5, atol=1e-6)
    d2 = d.copy(); d2[-1] = 0
    lq2 = q[-1].max(-1)
    np.testing.assert_allclose(np.asarray(pqn.lambda_returns(r, q, d2, lq2, 0.99, lam)),
                               np.asarray(_purejaxql(r, q, d2, lq2, 0.99, lam)), rtol=1e-5, atol=1e-6)
    # a perturbed lambda is caught
    if 0 < lam < 1:
        bad = np.asarray(pqn.lambda_returns(r, q, d, last_q, 0.99, lam + 0.01))
        assert not np.allclose(bad, got, rtol=1e-5, atol=1e-5)


def test_epsilon_schedule_and_eps_greedy():
    """Block-local schedule: epsilon(i) is the value at update i of a block (decision 5 Nov)."""
    c = pqn.pqn_config(TINY)
    assert (c.pqn_eps_start, c.pqn_eps_finish, c.pqn_eps_decay_updates, c.pqn_lambda) == (1.0, 0.05, 10, 0.65)
    e = [float(pqn.epsilon(c, i)) for i in (0, 5, 10, 11, 1000)]
    np.testing.assert_allclose(e, [1.0, 0.525, 0.05, 0.05, 0.05], rtol=1e-6)
    assert float(pqn.epsilon(dataclasses.replace(c, pqn_eps_decay_updates=0), 0)) == pytest.approx(0.05)
    q = jnp.tile(jnp.array([0.0, 2.0, 1.0, -1.0]), (4000, 1))
    assert np.all(np.asarray(pqn.eps_greedy(jax.random.PRNGKey(0), q, 0.0)) == 1)
    a = np.asarray(pqn.eps_greedy(jax.random.PRNGKey(0), q, 1.0))
    assert set(np.unique(a)) == {0, 1, 2, 3} and abs((a == 1).mean() - 0.25) < 0.03
    a = np.asarray(pqn.eps_greedy(jax.random.PRNGKey(0), q, 0.2))
    assert abs((a != 1).mean() - 0.15) < 0.03


def test_epsilon_resets_at_every_block_continual_and_fresh():
    """decision 5 Nov: every block (new task and revisit) starts at ε 1.0 and is at 0.05 from
    update 10 of that block onward, for both the continual stream and the fresh agents.
    Uses a multi-task stream with a revisit; logs are per-update stream/fresh metrics/epsilon."""
    c = pqn.pqn_config(TINY)
    assert c.pqn_eps_decay_updates == 10 and c.pqn_eps_start == 1.0 and c.pqn_eps_finish == 0.05
    # 2 new tasks, revisit every 1 -> schedule has a revisit of task 1 after task 2
    scfg = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=2, revisit_every=1, block_updates=12,
                               head_probe_every=0)
    stream = S.make_stream(scfg, 990)
    assert len(stream.schedule) >= 3 and not stream.is_new[2], (
        stream.schedule, stream.is_new)  # block 2 is a revisit
    r = lifelong.run_condition(c, scfg, [990], fresh=True, verbose=False)
    B, K = len(stream.schedule), scfg.block_updates
    # stream: (S, B, K); fresh: (S, T_new, 1, K) — one fresh agent per new task, each its own run
    checks = [("stream", np.asarray(r["stream/metrics/epsilon"]).reshape(B, K)),
              ("fresh", np.asarray(r["fresh/metrics/epsilon"]).reshape(-1, K))]
    for tag, eps in checks:
        for b in range(eps.shape[0]):
            assert eps[b, 0] == pytest.approx(1.0), (tag, b, eps[b, 0])
            assert eps[b, 10] == pytest.approx(0.05), (tag, b, eps[b, 10])
            assert np.all(eps[b, 10:] == pytest.approx(0.05)), (tag, b, eps[b, 10:])
            # mid-decay is between start and finish (not stuck at finish from a global counter)
            assert 0.05 < float(eps[b, 5]) < 1.0, (tag, b, eps[b, 5])
    # warm-up still task-1-only under the per-block update_idx reset
    c_w = dataclasses.replace(c, critic_warmup_updates=2)
    r_w = lifelong.run_condition(c_w, scfg, [990], fresh=False, verbose=False)
    warm = np.asarray(r_w["stream/metrics/critic_warmup"]).reshape(B, K)
    assert np.all(warm[0, :2] == 1.0) and np.all(warm[0, 2:] == 0.0)
    assert np.all(warm[1:] == 0.0)


def test_check_config_refusals():
    c = pqn.pqn_config(TINY)
    pqn.check_config(c)
    for kw, msg in ((dict(layer_norm=False), "layer_norm"), (dict(normalize_value=True), "normalize_value"),
                    (dict(eval_greedy=False), "eval_greedy"), (dict(lora_rank=8), "lora_rank"),
                    (dict(kickstart_coef=0.1), "kickstart"), (dict(pqn_lambda=1.5), "pqn_lambda"),
                    (dict(shared_trunk=False), "shared_trunk"), (dict(pqn_eps_finish=-0.1), "epsilon"),
                    (dict(reset_adam_on_task_switch=True, task_switch_updates=(5,)), "task-switch")):
        with pytest.raises(ValueError, match=msg):
            pqn.check_config(dataclasses.replace(c, **kw))
    with pytest.raises(ValueError, match="PPO only"):
        ppo.make_train(c)
    # PPO records are unchanged: no algo / PQN fields
    assert not set(ppo.PQN_FIELDS) & set(ppo.config_record(TINY))
    assert set(ppo.PQN_FIELDS) <= set(ppo.config_record(c))


# ----------------------------------------------------------------------------- learning sanity
def test_tiny_task_learning_sanity_dev_seeds():
    """One gridworld task, K = 100 updates at the lifelong_stream scale, dev seeds 990-992:
    greedy return rises from 0 to near optimal on at least 2 of 3 seeds; Q stays bounded."""
    c = pqn.pqn_config(ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], eval_episodes=4, eval_every=10)))
    scfg = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=1, revisit_every=0, block_updates=100,
                               head_probe_every=0)
    r = lifelong.run_condition(c, scfg, [990, 991, 992], fresh=False, verbose=False)
    final = np.asarray(r["stream/eval"]).reshape(3, -1)[:, -1]
    opt = np.asarray(r["optimal_return"]).reshape(3, -1)[:, 0]
    assert np.all(np.asarray(r["stream/eval0"]).ravel() < 0.5 * opt)
    assert (final >= 0.8 * opt).sum() >= 2, (final, opt)
    qm = np.asarray(r["stream/metrics/q_mean"])
    assert np.all(np.isfinite(qm)) and np.all(np.abs(qm) <= 1.0 / (1 - c.gamma))
    assert np.all(np.asarray(r["stream/metrics/epsilon"])[:, :10] > 0.05)


# ----------------------------------------------------------------------------- cells
def _run(name, seeds, chunk, d, **kw):
    return cells.run_cell(name, seeds, chunk, str(d), base_config=TINY, scfg=SCFG, verbose=False, **kw)


def _save_bc(d, s, **meta):
    pc = arms.plain_config(cells.cell_config(cells.CELLS["ln_scratch_1M"], TINY))
    m = dict(config=ppo.config_record(pc), seed=s, stream_config=S.config_record(SCFG), git_dirty=False,
             source_tree_hash=provenance.record()["source_tree_hash"])
    m.update(meta)
    params = lifelong.scratch_init_params(pc, s + 7)  # any PPO+LN params, distinct from the PQN init
    bc.save_checkpoint(str(d / bc.checkpoint_name(pc, s)), params, m)
    return params


def test_pqn_cells_and_refusals(tmp_path):
    assert cells.CELLS["pqn_scratch_1M"].algo == cells.CELLS["pqn_pretrained_1M"].algo == "pqn"
    c = cells.cell_config(cells.CELLS["pqn_pretrained_1M"], TINY, warmup_updates=1)
    assert (c.algo, c.layer_norm, c.hidden_size, c.normalize_value, c.critic_warmup_updates) == \
        ("pqn", True, 950, False, 1)
    assert "algo" not in cells.cell_spec(cells.CELLS["ln_scratch_1M"])
    with pytest.raises(ValueError, match="no kickstart coefficient"):
        _run("pqn_scratch_1M", [990], 1, tmp_path / "a", kickstart_coef=0.1)
    with pytest.raises(ValueError, match="needs bc_dir"):
        _run("pqn_pretrained_1M", [990], 1, tmp_path / "a", warmup_updates=1)
    bcd = tmp_path / "bc"
    with pytest.raises(ValueError, match="scratch cell"):
        _run("pqn_scratch_1M", [990], 1, tmp_path / "a", bc_dir=str(bcd))
    _save_bc(bcd, 991, seed=990)
    with pytest.raises(ValueError, match="checkpoint seed 990 != 991"):
        _run("pqn_pretrained_1M", [991], 1, tmp_path / "a", bc_dir=str(bcd), warmup_updates=1)
    _save_bc(bcd, 992, source_tree_hash="0" * 64)
    with pytest.raises(ValueError, match="BC source_tree_hash"):
        _run("pqn_pretrained_1M", [992], 1, tmp_path / "a", bc_dir=str(bcd), warmup_updates=1)
    pc = arms.plain_config(cells.cell_config(cells.CELLS["ln_scratch_1M"], TINY))
    _save_bc(bcd, 993, config=dict(ppo.config_record(pc), algo="pqn"))
    with pytest.raises(ValueError, match="algo 'pqn' != 'ppo'"):
        _run("pqn_pretrained_1M", [993], 1, tmp_path / "a", bc_dir=str(bcd), warmup_updates=1)
    with pytest.raises(ValueError, match="--bc-tree-hash refused"):
        _run("pqn_pretrained_1M", [700], 1, tmp_path / "a", bc_dir=str(bcd), bc_tree_hash="0" * 64)
    assert not (tmp_path / "a").exists()
    # pretrained init = BC trunk + this seed's scratch q_out
    bcp = _save_bc(bcd, 990)
    ps, rec = arms.load_bc_checkpoints(c, SCFG, [990], str(bcd),
                                       expected_tree_hash=provenance.record()["source_tree_hash"])
    scratch = lifelong.scratch_init_params(arms.plain_config(c), 990)
    for a, b in zip(jax.tree.leaves(ps[0]["params"]["trunk"]), jax.tree.leaves(bcp["params"]["trunk"])):
        assert np.array_equal(a, b)
    assert np.array_equal(ps[0]["params"]["q_out"]["kernel"], scratch["params"]["q_out"]["kernel"])
    assert set(ps[0]["params"]) == {"trunk", "q_out"} and json.loads(rec["bc_checkpoint_sha256"]).keys() == {"990"}
    man = _run("pqn_pretrained_1M", [990], 1, tmp_path / "p", bc_dir=str(bcd), warmup_updates=1)
    d = np.load(tmp_path / "p" / man["chunks"][0]["file"])
    assert json.loads(str(d["config"]))["algo"] == "pqn" and json.loads(str(d["fresh_config"]))["algo"] == "pqn"
    assert json.loads(str(d["cell_spec"]))["algo"] == "pqn" and not bool(d["ranking_binding"])
    assert np.asarray(d["stream/metrics/critic_warmup"]).ravel()[0] == 1.0
    # Review note: warm-up on the BC-graft path, counted via optimiser step (not block-local update_idx)
    scfg_w = dataclasses.replace(SCFG, num_new_tasks=2, revisit_every=1, block_updates=4, head_probe_every=0)
    stream_w = S.make_stream(scfg_w, 990)
    B_w, K_w = len(stream_w.schedule), scfg_w.block_updates
    man_w = cells.run_cell("pqn_pretrained_1M", [990], 1, str(tmp_path / "pw"), base_config=TINY,
                           scfg=scfg_w, bc_dir=str(bcd), warmup_updates=2, verbose=False)
    warm = np.asarray(np.load(tmp_path / "pw" / man_w["chunks"][0]["file"])["stream/metrics/critic_warmup"])
    warm = warm.reshape(B_w, K_w)
    assert np.all(warm[0, :2] == 1.0) and np.all(warm[0, 2:] == 0.0), warm[0]
    assert np.all(warm[1:] == 0.0), warm


def test_pqn_cell_same_layout_bit_identical(tmp_path):
    """PQN same-layout bit-identity (bitid.sh covers PPO only vs 9ce4217; review fix)."""
    seeds = [990, 991]
    m1 = _run("pqn_scratch_1M", seeds, 1, tmp_path / "r1")
    m2 = _run("pqn_scratch_1M", seeds, 1, tmp_path / "r2")
    cells.check_same_layout(m1, m2)
    for e1, e2 in zip(m1["chunks"], m2["chunks"]):
        _same(dict(np.load(tmp_path / "r1" / e1["file"])), dict(np.load(tmp_path / "r2" / e2["file"])))
    a, b = cells.merge_cell(str(tmp_path / "r1")), cells.merge_cell(str(tmp_path / "r2"))
    _same(a, b)
    # each chunk equals a direct run_condition call on its seed
    cfg = cells.cell_config(cells.CELLS["pqn_scratch_1M"], TINY)
    direct = lifelong.run_condition(cfg, SCFG, [991], fresh=True, verbose=False)
    ch = dict(np.load(tmp_path / "r1" / m1["chunks"][1]["file"]))
    for k in direct:
        if k.startswith(("stream/metrics", "fresh/metrics", "stream/eval", "fresh/eval")):
            assert np.array_equal(np.asarray(direct[k]), ch[k], equal_nan=True), k
    # PQN and PPO+LN cells pair (same identity fields); a different layout does not
    m3 = _run("ln_scratch_1M", seeds, 1, tmp_path / "ln")
    cells.check_same_layout(m1, m3)
    m4 = _run("pqn_scratch_1M", seeds, 2, tmp_path / "r4")
    with pytest.raises(ValueError, match="chunk layouts differ"):
        cells.check_same_layout(m1, m4)
