"""Step 4: size line, LoRA (cell 9), frozen pretrained / random trunk (cells 10-11),
kickstarting with a per-seed BC teacher (cell 12) and its sweep hook (arms.py)."""
import dataclasses
import json
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import traverse_util

import arms
import bc
import lifelong
import ppo
import stream as S

TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=16, hidden_size=16,
                         eval_episodes=1))
SCFG = S.StreamConfig(num_new_tasks=2, revisit_every=0, block_updates=2)
SHARED = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=2, revisit_every=0, block_updates=2)
FLAT = traverse_util.flatten_dict


def _init(cfg, seed=0):
    _, _, net, shp = ppo.make_env_and_network(cfg)
    return net, net.init(jax.random.PRNGKey(seed), jnp.zeros(shp)), shp


# ----------------------------------------------------------------------------- defaults
def test_defaults_unchanged():
    c = ppo.Config()
    assert c.lora_rank == 0 and c.lora_alpha == 16.0
    assert "lora_rank" not in ppo.config_record(c) and "lora_alpha" not in ppo.config_record(c)
    rec = ppo.config_record(dataclasses.replace(c, lora_rank=8))
    assert rec["lora_rank"] == 8 and rec["lora_alpha"] == 16.0
    cfg, kw, rec = arms.prepare(TINY, SCFG, [990])  # default arm, no size: config untouched
    assert cfg == TINY and kw == {} and rec == {}
    assert lifelong.scratch_config(dataclasses.replace(TINY, lora_rank=8)).lora_rank == 0


@pytest.mark.parametrize("shared", [True, False])
def test_lora_base_init_matches_plain_and_starts_equal(shared):
    plain = dataclasses.replace(TINY, shared_trunk=shared)
    lora = dataclasses.replace(plain, lora_rank=8)
    net_p, pp, shp = _init(plain, 3)
    net_l, pl, _ = _init(lora, 3)
    fp, fl = FLAT(pp), FLAT(pl)
    adapters = [k for k in fl if k[-1] in ("lora_a", "lora_b")]
    assert set(fl) - set(adapters) == set(fp)
    for k in fp:  # same names, same creation order -> same init for kernel / bias / heads
        np.testing.assert_array_equal(fl[k], fp[k])
    n_hidden = plain.num_layers * (1 if shared else 2)
    assert len(adapters) == 2 * n_hidden  # every trunk Dense, heads excluded
    for k in adapters:
        assert k[-2] not in ("actor_out", "critic_out")
        if k[-1] == "lora_b":
            assert not np.any(fl[k])
        else:
            assert fl[k].shape[1] == 8 and np.abs(fl[k]).max() <= 1 / np.sqrt(fl[k].shape[0])
    obs = jax.random.normal(jax.random.PRNGKey(1), (5,) + shp)
    np.testing.assert_array_equal(net_l.apply(pl, obs)[0], net_p.apply(pp, obs)[0])
    g = ppo.graft_lora(pl, jax.tree.map(lambda x: x + 1.0, pp))
    np.testing.assert_array_equal(FLAT(g)[adapters[0]], fl[adapters[0]])
    with pytest.raises(ValueError):
        ppo.graft_lora(pl, {"params": {"other": {"kernel": jnp.zeros(1)}}})


def test_lora_forward_and_merge():
    lora = dataclasses.replace(TINY, lora_rank=8, lora_alpha=16.0)
    net_l, pl, shp = _init(lora)
    pl = jax.tree_util.tree_map_with_path(
        lambda p, x: x + 0.1 * jax.random.normal(jax.random.PRNGKey(len(p)), x.shape)
        if p[-1].key == "lora_b" else x, pl)
    obs = jax.random.normal(jax.random.PRNGKey(1), (7,) + shp)
    x = obs.reshape(7, -1)
    l0 = FLAT(pl)[("params", "trunk", "layer_0", "kernel")]
    a, b = (FLAT(pl)[("params", "trunk", "layer_0", k)] for k in ("lora_a", "lora_b"))
    y = nn.relu(x @ l0 + FLAT(pl)[("params", "trunk", "layer_0", "bias")] + 2.0 * (x @ a) @ b)
    _, _, h = net_l.apply(pl, obs)
    np.testing.assert_allclose(h["trunk_0"], y, rtol=1e-5, atol=1e-6)  # alpha / r = 2
    net_p, _, _ = _init(TINY)
    merged = ppo.merge_lora(pl, lora)
    assert not ppo.has_lora(merged)
    np.testing.assert_allclose(net_p.apply(merged, obs)[0], net_l.apply(pl, obs)[0], rtol=1e-4, atol=1e-5)
    stacked = jax.tree.map(lambda v: jnp.stack([v, v]), pl)  # seed-batched merge
    np.testing.assert_allclose(ppo.merge_lora(stacked, lora)["params"]["trunk"]["layer_1"]["kernel"][1],
                               merged["params"]["trunk"]["layer_1"]["kernel"], rtol=1e-6)


def _train_final(cfg, start=None, teacher=None, seed=916, scfg=SCFG, gate=None):
    streams = [S.make_stream(scfg, seed)]
    tasks, sched = S.stack_streams(streams), jnp.asarray(np.stack([s.schedule for s in streams]))
    probe = jnp.stack([S.probe_batch(s) for s in streams])
    tr = jax.vmap(lifelong.make_stream_train(cfg, scfg, streams[0].num_blocks))
    st = None if start is None else jax.tree.map(lambda x: x[None], start)
    te = None if teacher is None else jax.tree.map(lambda x: x[None], teacher)
    g = None if gate is None else jnp.asarray(gate, jnp.float32)[None]
    out = jax.jit(tr)(jnp.stack([jax.random.PRNGKey(seed)]), tasks, sched, probe, st, te, g)
    return jax.tree.map(lambda x: x[0], out)


def test_lora_arm_trains_adapters_and_heads_only():
    base = TINY
    pre = jax.tree.map(lambda x: x * 0.9, lifelong.scratch_init_params(base, 916))  # "BC" params
    cfg = arms.arm_config(base, "lora", warmup_updates=1)
    assert cfg.lora_rank == 8 and cfg.lora_alpha == 16.0 and not cfg.freeze_trunk
    out = _train_final(cfg, start=pre)
    f, f0 = FLAT(out["final_params"]), FLAT(pre)
    for k, v in f.items():
        if k[1] == "trunk" and k[-1] not in ("lora_a", "lora_b"):
            np.testing.assert_array_equal(v, f0[k])  # frozen base
    assert np.any(f[("params", "trunk", "layer_1", "lora_b")] != 0)
    assert not np.array_equal(f[("params", "actor_out", "kernel")], f0[("params", "actor_out", "kernel")])
    # adapters: layer_0 (192x8 + 8x16), layer_1 (16x8 + 8x16); heads: 17x4 + 17
    assert lifelong.trainable_param_count(dataclasses.replace(base, lora_rank=8)) == \
        (192 * 8 + 8 * 16) + (16 * 8 + 8 * 16) + 17 * 4 + 17


@pytest.mark.parametrize("arm", ["frozen_pretrained", "frozen_random"])
def test_frozen_trunk_arms(arm):
    cfg = arms.arm_config(TINY, arm, warmup_updates=1 if arm == "frozen_pretrained" else None)
    assert cfg.freeze_trunk and not cfg.lora_rank
    pre = jax.tree.map(lambda x: x * 0.9, lifelong.scratch_init_params(TINY, 916))
    start = pre if arm == "frozen_pretrained" else None
    out = _train_final(cfg, start=start)
    ref = pre if start is not None else lifelong.scratch_init_params(TINY, 916)
    f, f0 = FLAT(out["final_params"]), FLAT(ref)
    for k in f:
        if k[1] == "trunk":
            np.testing.assert_array_equal(f[k], f0[k])
    for h in ("actor_out", "critic_out"):
        assert not np.array_equal(f[("params", h, "kernel")], f0[("params", h, "kernel")])
    assert lifelong.trainable_param_count(cfg) == 17 * 4 + 17


def test_kickstart_needs_teacher_and_per_seed_teacher_equals_constant():
    cfg = arms.arm_config(TINY, "kickstart", warmup_updates=0, kickstart_coef=0.5)
    with pytest.raises(ValueError, match="gate"):
        _train_final(cfg)
    with pytest.raises(ValueError, match="teacher"):
        _train_final(cfg, gate=np.ones(2))
    t = jax.tree.map(lambda x: x * 3.0, lifelong.scratch_init_params(TINY, 5))
    # SHARED (stage-1 shared layout): task 1 is in the KL window, so the run is not refused
    # for zero effective kickstart updates (the unshared SCFG stream would be).
    a = lifelong.run_condition(cfg, SHARED, [916], fresh=False, verbose=False, seed_teacher_params=[t])
    b = lifelong.run_condition(cfg, SHARED, [916], fresh=False, verbose=False, teacher_params=t)
    assert bool(a["seed_teacher_params"]) and "seed_teacher_params" not in b
    for k in b:
        if k.startswith("stream/") and "device" not in k:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    assert np.all(a["stream/metrics/kickstart_kl"] > 0)
    # each seed gets its own teacher: swapping teachers between seeds changes the result
    t2 = lifelong.scratch_init_params(TINY, 6)
    c = lifelong.run_condition(cfg, SHARED, [916, 917], fresh=False, verbose=False, seed_teacher_params=[t, t2])
    d = lifelong.run_condition(cfg, SHARED, [916, 917], fresh=False, verbose=False, seed_teacher_params=[t2, t])
    assert not np.array_equal(c["stream/metrics/kickstart_kl"][0], d["stream/metrics/kickstart_kl"][0])
    with pytest.raises(ValueError, match="not both"):
        lifelong.run_condition(cfg, SHARED, [916], verbose=False, teacher_params=t, seed_teacher_params=[t])
    with pytest.raises(ValueError, match="entries"):
        lifelong.run_condition(cfg, SHARED, [916, 917], verbose=False, seed_teacher_params=[t])


def test_kickstart_teacher_equal_to_init_gives_zero_kl_at_start():
    cfg = arms.arm_config(TINY, "kickstart", warmup_updates=0, kickstart_coef=1.0)
    p = lifelong.scratch_init_params(TINY, 916)
    out = _train_final(cfg, start=p, teacher=p, gate=np.ones(2))
    kl = np.asarray(out["metrics"]["kickstart_kl"]).reshape(-1)
    assert kl[0] < 1e-3 and np.all(kl >= -1e-6)


def test_lora_head_probe_and_records():
    scfg = dataclasses.replace(SCFG, num_new_tasks=2, head_probe_every=2)
    cfg = arms.arm_config(TINY, "lora", warmup_updates=1)
    pre = lifelong.scratch_init_params(TINY, 916)
    res = lifelong.run_condition(cfg, scfg, [916], verbose=False, seed_init_params=[pre])
    assert res["trainable_param_count"] < res["param_count"]
    np.testing.assert_allclose(res["headprobe/init_entropy"], np.log(4), atol=1e-2)
    assert "lora_rank" not in json.loads(str(res["fresh_config"]))
    assert "lora_rank" not in json.loads(str(res["head_probe_config"]))
    with pytest.raises(ValueError, match="LoRA"):
        lifelong.check_reset_arm("heads", cfg)
    for k in ("lora_adapter_norm", "weight_norm_merged", "weight_norm_merged.trunk",
              "lora_delta_norm.trunk.layer_0", "weight_norm_merged.trunk.layer_1"):
        assert f"stream/metrics/{k}" in res and f"stream/probes/{k}" in res, k
    assert np.all(res["stream/metrics/lora_adapter_norm"] > 0)


def test_lora_norm_metrics():
    lora = dataclasses.replace(TINY, lora_rank=8, lora_alpha=16.0)
    net_l, pl, shp = _init(lora)
    pl = jax.tree_util.tree_map_with_path(
        lambda p, x: x + 0.1 * jax.random.normal(jax.random.PRNGKey(len(p)), x.shape)
        if p[-1].key == "lora_b" else x, pl)
    obs = jax.random.normal(jax.random.PRNGKey(1), (7,) + shp)
    h = ppo.network_health(pl, net_l.apply, obs, 0.025, lora.activation, ppo.lora_scale_of(lora))
    f, m = FLAT(pl), FLAT(ppo.merge_lora(pl, lora))
    ad = [v for k, v in f.items() if k[-1] in ("lora_a", "lora_b")]
    np.testing.assert_allclose(h["lora_adapter_norm"], np.sqrt(sum(np.sum(np.square(v)) for v in ad)), rtol=1e-5)
    for layer in ("layer_0", "layer_1"):
        k = ("params", "trunk", layer)
        np.testing.assert_allclose(h[f"weight_norm_merged.trunk.{layer}"], np.linalg.norm(m[k + ("kernel",)]), rtol=1e-5)
        np.testing.assert_allclose(h[f"lora_delta_norm.trunk.{layer}"],
                                   np.linalg.norm(2.0 * f[k + ("lora_a",)] @ f[k + ("lora_b",)]), rtol=1e-5)
        np.testing.assert_allclose(h[f"weight_norm.trunk.{layer}"], np.linalg.norm(f[k + ("kernel",)]), rtol=1e-6)
    tk = [v for k, v in m.items() if k[1] == "trunk" and k[-1] == "kernel"]
    np.testing.assert_allclose(h["weight_norm_merged.trunk"], np.sqrt(sum(np.sum(np.square(v)) for v in tk)), rtol=1e-5)
    np.testing.assert_allclose(h["weight_norm_merged"], np.sqrt(sum(np.sum(np.square(v)) for v in m.values())), rtol=1e-5)
    net_p, pp, _ = _init(TINY)  # no LoRA: keys absent (records unchanged)
    hp = ppo.network_health(pp, net_p.apply, obs, 0.025, TINY.activation, ppo.lora_scale_of(TINY))
    assert not any(k.startswith(("lora_", "weight_norm_merged")) for k in hp)


# ----------------------------------------------------------------------------- arm config
def test_arm_config_validation():
    for arm in ("pretrained", "lora", "frozen_pretrained", "kickstart"):
        with pytest.raises(ValueError, match="warm-up"):
            arms.arm_config(TINY, arm, kickstart_coef=0.1 if arm == "kickstart" else None)
    for arm in ("scratch", "frozen_random"):
        with pytest.raises(ValueError, match="scratch arm"):
            arms.arm_config(TINY, arm, warmup_updates=3)
    with pytest.raises(ValueError, match="kickstart_coef"):
        arms.arm_config(TINY, "kickstart", warmup_updates=3)
    with pytest.raises(ValueError, match="no kickstarting"):
        arms.arm_config(TINY, "pretrained", warmup_updates=3, kickstart_coef=0.1)
    with pytest.raises(ValueError, match="already"):
        arms.arm_config(dataclasses.replace(TINY, freeze_trunk=True), "scratch")
    p = arms.arm_config(TINY, "pretrained", warmup_updates=10)
    assert p.critic_warmup_updates == 10 and p.adam_reset_at_handover and not p.freeze_trunk
    k = arms.arm_config(TINY, "kickstart", warmup_updates=10, kickstart_coef=0.3)
    assert k.kickstart_coef == 0.3 and k.lora_rank == 0 and not k.freeze_trunk


def test_sizes():
    c = ppo.Config(**ppo.PRESETS["lifelong_stream"])
    od, na = arms.dims(c)
    assert (od, na) == (192, 4)
    w = arms.size_widths(c)
    assert w["1M"] == 950
    for name, width in w.items():
        n = lifelong.param_count(dataclasses.replace(c, hidden_size=width))
        assert n == arms.formula_param_count(width, 2, od, na)
        if name != "1M":
            assert abs(n - arms.SIZE_TARGETS[name]) / arms.SIZE_TARGETS[name] < 0.01
    assert lifelong.param_count(dataclasses.replace(c, hidden_size=950)) == 1_091_555
    for shared, ln in ((False, False), (True, True), (False, True)):
        cc = dataclasses.replace(c, hidden_size=40, shared_trunk=shared, layer_norm=ln)
        assert lifelong.param_count(cc) == arms.formula_param_count(40, 2, od, na, shared, ln)
    assert arms.apply_size(c, "0.3M").hidden_size == w["0.3M"]
    with pytest.raises(ValueError):
        arms.apply_size(c, "2M")


def test_kickstart_sweep_hook():
    assert arms.check_kickstart_sweep([0.01, 0.03, 0.1, 0.3], range(750, 760))[0] == [0.01, 0.03, 0.1, 0.3]
    arms.check_kickstart_sweep([1, 2, 3, 4], [990, 991])
    for bad in ([0.1, 0.2, 0.3], [0.1, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3], [0.1] * 5):
        with pytest.raises(ValueError, match="exactly 4"):
            arms.check_kickstart_sweep(bad, [990])
    for seeds in ([700], [500], [750, 990], [850]):
        with pytest.raises(ValueError, match="sweep seeds"):
            arms.check_kickstart_sweep([1, 2, 3, 4], seeds)


# ----------------------------------------------------------------------------- BC loading
def _save(tmp_path, cfg, scfg, s, **meta):
    p = lifelong.scratch_init_params(cfg, s)
    m = dict(config=dataclasses.asdict(cfg), seed=s, stream_config=S.config_record(scfg), git_dirty=False)
    m.update(meta)
    bc.save_checkpoint(str(tmp_path / bc.checkpoint_name(cfg, s)), p, m)
    return p


def test_load_bc_checkpoints(tmp_path):
    cfg = arms.arm_config(TINY, "lora", warmup_updates=1)
    p0 = _save(tmp_path, TINY, SHARED, 990)
    _save(tmp_path, TINY, SHARED, 991)
    ps, rec = arms.load_bc_checkpoints(cfg, SHARED, [990, 991], str(tmp_path))
    np.testing.assert_array_equal(FLAT(ps[0])[("params", "actor_out", "kernel")],
                                  FLAT(p0)[("params", "actor_out", "kernel")])
    assert set(json.loads(rec["bc_checkpoint_sha256"])) == {"990", "991"}
    with pytest.raises(FileNotFoundError):
        arms.load_bc_checkpoints(cfg, SHARED, [992], str(tmp_path))
    with pytest.raises(ValueError, match="flags"):  # BC on the shared layout, run on pilot config
        arms.load_bc_checkpoints(cfg, SCFG, [990], str(tmp_path))
    _save(tmp_path, TINY, SHARED, 993, seed=994)
    with pytest.raises(ValueError, match="seed"):
        arms.load_bc_checkpoints(cfg, SHARED, [993], str(tmp_path))
    _save(tmp_path, TINY, SHARED, 995, git_dirty=True)
    with pytest.raises(ValueError, match="dirty"):
        arms.load_bc_checkpoints(cfg, SHARED, [995], str(tmp_path))
    arms.load_bc_checkpoints(cfg, SHARED, [995], str(tmp_path), allow_dirty=True)
    _save(tmp_path, TINY, SHARED, 996, config=dict(dataclasses.asdict(TINY), activation="tanh"))
    with pytest.raises(ValueError, match="architecture"):
        arms.load_bc_checkpoints(cfg, SHARED, [996], str(tmp_path))
    # prepare(): pretrained arms need bc_dir; kickstart teacher = the seed's BC params
    with pytest.raises(ValueError, match="bc_dir"):
        arms.prepare(TINY, SHARED, [990], "pretrained", warmup_updates=1)
    with pytest.raises(ValueError, match="scratch arm"):
        arms.prepare(TINY, SHARED, [990], "frozen_random", bc_dir=str(tmp_path))
    c, kw, rec = arms.prepare(TINY, SHARED, [990, 991], "kickstart", warmup_updates=1,
                              kickstart_coef=0.2, bc_dir=str(tmp_path))
    assert kw["seed_teacher_params"] is kw["seed_init_params"] and rec["arm"] == "kickstart"


def test_cli_arm_end_to_end(tmp_path, monkeypatch):
    _save(tmp_path, dataclasses.replace(TINY, hidden_size=8), SHARED, 990)
    out = tmp_path / "x.npz"
    base = ["lifelong.py", "--seeds", "990", "--num-new-tasks", "2", "--block-updates", "2",
            "--revisit-every", "0", "--hidden-size", "8", "--no-fresh", "--shared-obs-layout",
            "--shared-action-perm"]
    monkeypatch.setattr(sys, "argv", base + ["--out", str(out), "--arm", "lora"])
    with pytest.raises(SystemExit, match="warm-up"):
        lifelong.main()
    monkeypatch.setattr(sys, "argv", base + ["--out", str(out), "--arm", "lora", "--bc-dir", str(tmp_path),
                                             "--critic-warmup-updates", "1"])
    lifelong.main()
    d = np.load(out)
    c = json.loads(str(d["config"]))
    assert c["lora_rank"] == 8 and c["lora_alpha"] == 16.0 and c["critic_warmup_updates"] == 1
    assert str(d["arm"]) == "lora" and bool(d["seed_init_params"]) and "bc_checkpoint_sha256" in d.files
    monkeypatch.setattr(sys, "argv", base + ["--out", str(out), "--size", "1M"])
    with pytest.raises(SystemExit, match="exclusive"):
        lifelong.main()


def _load_sweep():
    import importlib.util
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location("kickstart_sweep", os.path.join(root, "scripts", "kickstart_sweep.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kickstart_sweep_script(tmp_path, monkeypatch):
    ks = _load_sweep()
    cfg = dataclasses.replace(ppo.Config(**ppo.PRESETS["lifelong_stream"]), hidden_size=arms.size_widths(TINY)["0.1M"])
    _save(tmp_path, cfg, SHARED, 990)
    args = ["--seeds", "990", "--bc-dir", str(tmp_path), "--critic-warmup-updates", "1", "--size", "0.1M",
            "--num-new-tasks", "2", "--block-updates", "2", "--revisit-every", "0", "--out-dir", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="exactly 4"):
        ks.main(args + ["--coefs", "0.1,0.2,0.3"])
    with pytest.raises(SystemExit):  # reserved test seeds: refused (seed guard / pytest guard)
        ks.main([a if a != "990" else "700" for a in args] + ["--coefs", "1,2,3,4", "--allow-official-seeds"])
    assert not (tmp_path / "o").exists()
    with pytest.raises(SystemExit, match="0 effective kickstart updates"):  # warm-up eats the window
        ks.main([a if a != "1" else "2" for a in args] + ["--coefs", "0.01,0.1,1,10"])
    out = ks.main(args + ["--coefs", "0.01,0.1,1,10"])
    assert all(t["kickstart_effective_updates"] == [1] for t in out["trials"])  # K=2 minus 1 warm-up
    assert [t["coef"] for t in out["trials"]] == [0.01, 0.1, 1.0, 10.0]
    assert out["ranking_binding"] is False and "mechanics-only" in out["ranking_binding_reason"]
    assert "ranked_best" not in out  # dev seeds: no ranking written at all
    with open(tmp_path / "o" / "sweep.json") as f:
        js = json.load(f)
    assert js["ranking_binding"] is False and "ranked_best" not in js
    assert "set in advance" in js["selection_metric"] and "21-40" in js["selection_metric"]
    d = np.load(tmp_path / "o" / "kickstart_c3.npz")
    assert json.loads(str(d["config"]))["kickstart_coef"] == 10.0 and bool(d["seed_teacher_params"])
    assert not bool(d["ranking_binding"]) and "kickstart_coef_task" in d.files
    assert "ranked_best" not in d.files
    # the binding path (only reachable on 750-759 / 850-859, never run here) writes ranked_best:
    # exercised by stubbing the binding check on dev seed 990
    monkeypatch.setattr(ks.arms, "sweep_binding", lambda seeds, fresh: (True, "stub"))
    out = ks.main(args[:-1] + [str(tmp_path / "o2"), "--coefs", "0.01,0.1,1,10"])
    best = max(out["trials"], key=lambda t: t["iqm"])["coef"]
    assert out["ranking_binding"] is True and out["ranked_best"] == best


# ----------------------------------------------------------------------------- KL gating
GATE_SCFG = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=3, revisit_every=3, block_updates=2,
                                head_probe_every=0)


def test_kickstart_loss_gate_exact_zero():
    cfg = ppo.Config(kickstart_coef=0.5)
    _, _, net, shp = ppo.make_env_and_network(cfg)
    p = net.init(jax.random.PRNGKey(0), jnp.zeros(shp))
    t = jax.tree.map(lambda x: x * 50.0, net.init(jax.random.PRNGKey(9), jnp.zeros(shp)))
    n = 16
    obs = jax.random.normal(jax.random.PRNGKey(2), (n,) + shp)
    b = ppo.Batch(obs, jnp.zeros(n, jnp.int32), jnp.full(n, -np.log(2.0)), jnp.zeros(n),
                  jax.random.normal(jax.random.PRNGKey(3), (n,)), jnp.zeros(n))
    l0, _ = ppo.ppo_loss(p, net.apply, b, ppo.Config())
    l_on, m_on = ppo.ppo_loss(p, net.apply, b, cfg, t, kickstart_scale=1.0)
    l_off, m_off = ppo.ppo_loss(p, net.apply, b, cfg, t, kickstart_scale=0.0)
    assert float(m_on["kickstart_term"]) > 0 and float(m_on["kickstart_coef_eff"]) == 0.5
    assert float(m_off["kickstart_term"]) == 0.0 and float(m_off["kickstart_coef_eff"]) == 0.0
    assert float(l_off) == float(l0) and float(m_off["kickstart_kl"]) > 0
    l_un, _ = ppo.ppo_loss(p, net.apply, b, cfg, t)  # ungated (CartPole path) unchanged
    np.testing.assert_allclose(float(l_on), float(l_un), rtol=1e-6)


def test_kickstart_gate_blocks_never_reenables():
    st = S.make_stream(GATE_SCFG, 916)
    both = st.shared_obs & st.shared_action
    assert st.schedule.tolist() == [0, 1, 2, 0] and both.tolist() == [True, True, False]
    assert lifelong.kickstart_gate_blocks(st).tolist() == [1, 1, 0, 0]  # block 4 revisits shared task 1
    assert lifelong.kickstart_gate_tasks(st).tolist() == [1, 1, 0]
    pil = S.make_stream(dataclasses.replace(GATE_SCFG, shared_obs_layout=False, shared_action_perm=False), 916)
    assert not lifelong.kickstart_gate_blocks(pil).any()  # no shared layout -> never on


def test_kickstart_kl_gated_in_stream_run():
    """Non-zero KL contribution on the shared tasks; exactly 0 from the first unshared task
    on, including the later revisit of shared task 1."""
    c = 0.5
    cfg = arms.arm_config(TINY, "kickstart", warmup_updates=0, kickstart_coef=c)
    t = jax.tree.map(lambda x: x * 3.0, lifelong.scratch_init_params(TINY, 5))
    res = lifelong.run_condition(cfg, GATE_SCFG, [916], fresh=False, verbose=False, seed_teacher_params=[t])
    assert res["task_of_block"][0].tolist() == [1, 2, 3, 1]
    np.testing.assert_array_equal(res["kickstart_coef_block"][0], [c, c, 0, 0])
    # on/off state per block: block 4 is a revisit of task 1 AFTER the first unshared task (3)
    assert res["kickstart_on_block"][0].tolist() == [True, True, False, False]
    np.testing.assert_array_equal(res["kickstart_coef_task"][0], [c, c, 0])
    np.testing.assert_array_equal(res["stream/kickstart_coef_eff"][0], [c, c, 0, 0])
    term = res["stream/metrics/kickstart_term"][0]          # (B, K), mean over minibatches
    ceff = res["stream/metrics/kickstart_coef_eff"][0]
    assert np.all(term[:2] > 0) and np.all(ceff[:2] == c)  # constant inside the window
    assert np.all(term[2:] == 0.0) and np.all(ceff[2:] == 0.0)
    assert np.all(res["stream/metrics/kickstart_kl"][0][2:] > 0)  # KL itself is non-zero there
    # no gate leaks into non-kickstart runs
    r0 = lifelong.run_condition(TINY, GATE_SCFG, [916], fresh=False, verbose=False)
    assert "kickstart_coef_block" not in r0 and "stream/metrics/kickstart_term" not in r0
    assert "kickstart_on_block" not in r0


def test_kickstart_effective_updates_count_on_gated_seed():
    """decision 2 Nov 11:36: kickstart_effective_updates = updates with coef > 0 after the critic
    warm-up, per seed, in the run record; checked against the in-graph per-update flag."""
    c, K, W = 0.5, 3, 2
    scfg = dataclasses.replace(GATE_SCFG, block_updates=K)
    cfg = arms.arm_config(TINY, "kickstart", warmup_updates=W, kickstart_coef=c)
    st916, st900 = S.make_stream(scfg, 916), S.make_stream(scfg, 900)
    # 916: gate on for tasks 1-2 -> (K - W) + K = 4; 900: task 1 only -> K - W = 1.
    # Both revisit task 1 in block 4 after the first unshared task: it adds nothing.
    assert lifelong.kickstart_effective_block_updates(st916, K, W).tolist() == [1, 3, 0, 0]
    assert lifelong.kickstart_effective_block_updates(st900, K, W).tolist() == [1, 0, 0, 0]
    t = jax.tree.map(lambda x: x * 3.0, lifelong.scratch_init_params(TINY, 5))
    res = lifelong.run_condition(cfg, scfg, [916, 900], fresh=False, verbose=False,
                                 seed_teacher_params=[t, t])
    assert res["task_of_block"].tolist() == [[1, 2, 3, 1], [1, 2, 3, 1]]
    assert res["kickstart_effective_updates"].tolist() == [4, 1]
    assert res["kickstart_effective_updates_block"].tolist() == [[1, 3, 0, 0], [1, 0, 0, 0]]
    assert res["kickstart_effective_updates_revisit"].tolist() == [0, 0]  # task-1 revisit adds 0
    flag = res["stream/metrics/kickstart_effective"]                     # (S, B, K) 0/1
    np.testing.assert_array_equal(flag[0], [[0, 0, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(flag[1], [[0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert "after the 2-update critic warm-up" in str(res["kickstart_effective_rule"])
    # the warm-up itself: only the critic trains there (critic_warmup flag on updates 0..W-1)
    np.testing.assert_array_equal(res["stream/metrics/critic_warmup"][0].reshape(-1)[:W + 1], [1, 1, 0])
    # non-kickstart arms: no new keys, no per-update flag
    r0 = lifelong.run_condition(arms.arm_config(TINY, "pretrained", warmup_updates=W), scfg, [916],
                                fresh=False, verbose=False, seed_init_params=[t])
    assert not any(k.startswith("kickstart") for k in r0)
    assert "stream/metrics/kickstart_effective" not in r0


def test_kickstart_effective_updates_revisit_while_gate_on():
    """A revisit of task 1 BEFORE any unshared task keeps the gate on (gate rule, f464906),
    so its K updates count: the count is the updates the KL actually acts on. Reported
    separately as kickstart_effective_updates_revisit. Dev seed 908 (full stage-1 stream)
    trains tasks 1, 2, 3, 1, 4 with tasks 1-3 shared and task 4 not."""
    st = S.make_stream(dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=6), 908)
    assert st.schedule[:5].tolist() == [0, 1, 2, 0, 3] and not st.is_new[3]
    blk = lifelong.kickstart_effective_block_updates(st, 100, 10)
    assert blk.tolist()[:5] == [90, 100, 100, 100, 0] and not blk[4:].any()
    assert int(np.where(st.is_new, 0, blk).sum()) == 100  # what ..._revisit reports


def test_kickstart_zero_window_refused():
    """4 updates per block with a 4-update warm-up (the f464906 smoke) on seeds whose window
    is task 1 only: 0 effective updates -> refused before any training, never counted."""
    scfg = dataclasses.replace(GATE_SCFG, block_updates=4)
    cfg = arms.arm_config(TINY, "kickstart", warmup_updates=4, kickstart_coef=0.1)
    t = lifelong.scratch_init_params(TINY, 5)
    for s in (900, 901):
        assert lifelong.kickstart_effective_block_updates(S.make_stream(scfg, s), 4, 4).sum() == 0
    n = ppo.TRACE_COUNT["train"]
    with pytest.raises(lifelong.KickstartWindowError, match=r"0 effective kickstart updates.*\[900, 901\]"):
        lifelong.run_condition(cfg, scfg, [900, 901], fresh=False, verbose=False, seed_teacher_params=[t, t])
    assert ppo.TRACE_COUNT["train"] == n  # refused before the trainer was ever traced
    # one bad seed is enough, even next to a seed with a window (916: task 2 is shared too)
    with pytest.raises(lifelong.KickstartWindowError, match=r"seed\(s\) \[900\].*916: 4"):
        lifelong.run_condition(cfg, scfg, [916, 900], fresh=False, verbose=False, seed_teacher_params=[t, t])
    # an unshared (pilot) stream never opens the window, whatever the warm-up
    pil = dataclasses.replace(scfg, shared_obs_layout=False, shared_action_perm=False)
    with pytest.raises(lifelong.KickstartWindowError):
        lifelong.run_condition(arms.arm_config(TINY, "kickstart", warmup_updates=0, kickstart_coef=0.1),
                               pil, [916], fresh=False, verbose=False, seed_teacher_params=[t])
    with pytest.raises(lifelong.KickstartWindowError):  # warm-up > window but <= K
        lifelong.check_kickstart_window([900], [[0, 0]])
    assert lifelong.check_kickstart_window([916], [[1, 0]]).tolist() == [1]


def test_sweep_binding():
    assert arms.sweep_binding(range(750, 760), range(850, 860))[0] is True
    assert arms.sweep_binding([900, 901], [1900, 1901]) == (
        False, "mechanics-only: stream seeds [900, 901] are not exactly 750-759")
    assert arms.sweep_binding(range(750, 755), range(850, 855))[0] is False
    assert arms.sweep_binding(range(750, 760), range(1750, 1760))[0] is False
