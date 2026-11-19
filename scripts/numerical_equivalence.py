"""Numerical equivalence: our GAE / PPO loss / gradients vs PureJaxRL's, on
identical params and an identical rollout batch.

PureJaxRL's `_calculate_gae` and `_loss_fn` are closures nested inside
`make_train`, so they cannot be imported. We pull their source text verbatim out
of the UNMODIFIED purejaxrl/ppo.py with `ast` and exec it with the two closure
variables it needs bound: `config` (their dict) and `network` (their ActorCritic).
Nothing in their file is edited or re-typed by hand.

Procedure (per env):
  1. Build our network and theirs. Map our params onto their tree
     (actor_0, actor_1, actor_out, critic_0, critic_1, critic_out ->
     Dense_0 ... Dense_5). Check the forward passes agree.
  2. Collect one real rollout batch with OUR rollout() under params p_old.
  3. GAE: our compute_gae vs their _calculate_gae on the same (reward, value,
     done, last_value).
  4. Loss and grads: evaluate both losses at p_new = p_old + noise (so the
     ratio != 1 and the clipping branches are active), per minibatch, on the
     same minibatches. Compare total / value / actor / entropy and every
     gradient leaf (mapped back to our names).

Run with PureJaxRL's venv (it has distrax):
  ../purejaxrl-ref/.venv/bin/python scripts/numerical_equivalence.py
"""
import argparse
import ast
import json
import os
import sys
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.environ.get("PUREJAXRL_DIR", "../purejaxrl-ref")

# purejaxrl/wrappers.py imports brax/navix at module level; stub them (unused).
for name in ["brax", "brax.envs", "brax.envs.wrappers", "brax.envs.wrappers.training", "navix"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["brax"].envs = sys.modules["brax.envs"]
sys.modules["brax.envs.wrappers.training"].EpisodeWrapper = object
sys.modules["brax.envs.wrappers.training"].AutoResetWrapper = object
sys.path.insert(0, os.path.join(REF, "purejaxrl"))  # for their `from wrappers import ...`
sys.path.append(ROOT)  # last, for ppo.py's dependency-free `import seeds` (no name clash with REF)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

# Both files are called ppo.py, so load each explicitly by path.
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("tars_ppo", os.path.join(ROOT, "ppo.py"))
ours = importlib.util.module_from_spec(_s)
sys.modules["tars_ppo"] = ours
_s.loader.exec_module(ours)
_spec = importlib.util.spec_from_file_location("pjr_ppo", os.path.join(REF, "purejaxrl", "ppo.py"))
pjr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pjr)

NAME_MAP = {"actor_0": "Dense_0", "actor_1": "Dense_1", "actor_out": "Dense_2",
            "critic_0": "Dense_3", "critic_1": "Dense_4", "critic_out": "Dense_5"}


def extract_ref_fns(config, network):
    """exec PureJaxRL's nested _calculate_gae and _loss_fn, verbatim."""
    path = os.path.join(REF, "purejaxrl", "ppo.py")
    src = open(path).read()
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_calculate_gae", "_loss_fn"):
            found[node.name] = textwrap.dedent(ast.get_source_segment(src, node))
    assert set(found) == {"_calculate_gae", "_loss_fn"}, found.keys()
    ns = {"jax": jax, "jnp": jnp, "config": config, "network": network}
    for code in found.values():
        exec(compile(code, path, "exec"), ns)
    return ns["_calculate_gae"], ns["_loss_fn"], found


def to_ref_params(p):
    return {"params": {NAME_MAP[k]: v for k, v in p["params"].items()}}


def from_ref_grads(g):
    inv = {v: k for k, v in NAME_MAP.items()}
    return {"params": {inv[k]: v for k, v in g["params"].items()}}


def maxdiff(a, b):
    return float(max(jnp.max(jnp.abs(x - y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))))


def maxabs(a):
    return float(max(jnp.max(jnp.abs(x)) for x in jax.tree.leaves(a)))


def run_env(env_name, preset, seed=0, noise=0.05, num_envs=None, num_steps=None, ref_overrides=None):
    over = dict(ours.resolve_presets(preset))
    if num_envs: over["num_envs"] = num_envs
    if num_steps: over["num_steps"] = num_steps
    cfg = ours.Config(env_name=env_name, **over)
    assert not cfg.shared_trunk and not cfg.normalize_value, "equivalence is for the match config"
    env, env_params, net, obs_shape = ours.make_env_and_network(cfg)
    n_actions = env.action_space(env_params).n
    ref_net = pjr.ActorCritic(n_actions, activation=cfg.activation)
    ref_cfg = {"GAMMA": cfg.gamma, "GAE_LAMBDA": cfg.gae_lambda, "CLIP_EPS": cfg.clip_eps,
               "VF_COEF": cfg.vf_coef, "ENT_COEF": cfg.ent_coef}
    ref_cfg.update(ref_overrides or {})  # only for the negative control
    calc_gae_ref, loss_ref, _ = extract_ref_fns(ref_cfg, ref_net)
    flat = int(np.prod(obs_shape))

    k_init, k_noise, k_reset, k_roll, k_perm = jax.random.split(jax.random.PRNGKey(seed), 5)
    p_old = net.init(k_init, jnp.zeros(obs_shape))
    # Their tree must have exactly the same shapes after renaming.
    ref_shapes = jax.tree.map(lambda x: x.shape, ref_net.init(k_init, jnp.zeros((flat,))))
    assert jax.tree.map(lambda x: x.shape, to_ref_params(p_old)) == ref_shapes
    leaves, tdef = jax.tree.flatten(p_old)
    ks = jax.random.split(k_noise, len(leaves))
    p_new = jax.tree.unflatten(tdef, [x + noise * jax.random.normal(k, x.shape) for x, k in zip(leaves, ks)])

    # 1) forward pass
    obs_probe = jax.random.uniform(k_perm, (256,) + obs_shape)
    lo, vo, _ = net.apply(p_new, obs_probe)
    pi, vr = ref_net.apply(to_ref_params(p_new), obs_probe.reshape(256, flat))
    fwd = {"logits": float(jnp.max(jnp.abs(jax.nn.log_softmax(lo) - pi.logits))),
           "value": float(jnp.max(jnp.abs(vo - vr)))}

    # 2) a real rollout with our code under p_old
    keys = jax.random.split(k_reset, cfg.num_envs)
    obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys, env_params)
    traj, env_state, obs, _ = ours.rollout(env, env_params, net, cfg, p_old, env_state, obs,
                                           ours.EpisodeStats.zeros(cfg.num_envs), k_roll)
    _, last_value, _ = net.apply(p_old, obs)

    # 3) GAE
    adv_o, ret_o = ours.compute_gae(traj.reward, traj.value, traj.done, last_value,
                                    cfg.gamma, cfg.gae_lambda)
    RefT = types.SimpleNamespace  # their scan only reads .done/.value/.reward
    ref_traj = pjr.Transition(done=traj.done.astype(jnp.float32) if traj.done.dtype != jnp.bool_ else traj.done,
                              action=traj.action, value=traj.value, reward=traj.reward,
                              log_prob=traj.logp, obs=traj.obs.reshape(traj.obs.shape[:2] + (flat,)),
                              info=jnp.zeros(traj.reward.shape))
    adv_r, tgt_r = calc_gae_ref(ref_traj, last_value)
    gae = {"advantages": float(jnp.max(jnp.abs(adv_o - adv_r))),
           "targets": float(jnp.max(jnp.abs(ret_o - tgt_r))),
           "advantages_scale": float(jnp.max(jnp.abs(adv_o))),
           "n_done": int(jnp.sum(traj.done))}

    # 4) loss + grads per minibatch (same minibatches for both)
    B = cfg.batch_size
    flat_b = lambda x: x.reshape((B,) + x.shape[2:])  # noqa: E731
    batch = ours.Batch(flat_b(traj.obs), flat_b(traj.action), flat_b(traj.logp),
                       flat_b(traj.value), flat_b(adv_o), flat_b(ret_o))
    perm = jax.random.permutation(k_perm, B)
    mbs = jax.tree.map(lambda x: x[perm].reshape((cfg.num_minibatches, -1) + x.shape[1:]), batch)

    def ours_fn(p, mb):
        return ours.ppo_loss(p, net.apply, mb, cfg)

    def ref_fn(p, mb):
        rt = pjr.Transition(done=None, action=mb.action, value=mb.value, reward=None,
                            log_prob=mb.logp, obs=mb.obs.reshape(mb.obs.shape[0], flat), info=None)
        return loss_ref(p, rt, mb.advantages, mb.returns)

    res = {k: [] for k in ("total", "value_loss", "actor_loss", "entropy", "grad",
                           "grad_scale", "clip_frac")}
    for i in range(cfg.num_minibatches):
        mb = jax.tree.map(lambda x: x[i], mbs)
        (lo_, m), g_o = jax.value_and_grad(ours_fn, has_aux=True)(p_new, mb)
        (lr_, (vl, al, ent)), g_r = jax.value_and_grad(ref_fn, has_aux=True)(to_ref_params(p_new), mb)
        g_r = from_ref_grads(g_r)
        res["total"].append(abs(float(lo_ - lr_)))
        res["value_loss"].append(abs(float(m["value_loss"] - vl)))
        res["actor_loss"].append(abs(float(m["policy_loss"] - al)))
        res["entropy"].append(abs(float(m["entropy"] - ent)))
        res["grad"].append(maxdiff(g_o, g_r))
        res["grad_scale"].append(maxabs(g_o))
        res["clip_frac"].append(float(m["clip_frac"]))
    loss = {k: max(v) for k, v in res.items() if k not in ("clip_frac",)}
    loss["clip_frac_mean"] = float(np.mean(res["clip_frac"]))
    # Per-leaf grad diffs on the last minibatch, for the report.
    per_leaf = {"/".join(str(getattr(k, "key", k)) for k in path): float(jnp.max(jnp.abs(a - b)))
                for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(g_o)[0], jax.tree.leaves(g_r))}
    return {"env": env_name, "preset": preset, "batch": B, "minibatches": cfg.num_minibatches,
            "forward": fwd, "gae": gae, "loss": loss, "grad_per_leaf_last_mb": per_leaf}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(ROOT, "results", "numerical_equivalence.json"))
    p.add_argument("--tol", type=float, default=1e-5,
                   help="max abs diff allowed per quantity (default 1e-5; use 1e-5 on GPU too)")
    p.add_argument("--matmul-precision", choices=ours.MATMUL_PRECISIONS, default="default",
                   help="jax_default_matmul_precision, as in lifelong.py (not forced); recorded in "
                        "the JSON. On Ampere GPUs 'default' = TF32: pass 'highest' for the 1e-5 check")
    a = p.parse_args()
    ours.set_matmul_precision(a.matmul_precision)
    # noise is the std of the Gaussian perturbation p_new - p_old. It is chosen so
    # that a few % of samples are clipped (exercises both min/max branches).
    runs = [run_env("CartPole-v1", "default", noise=0.12),
            run_env("CartPole-v1", "purejaxrl_match", noise=0.16),
            run_env("Breakout-MinAtar", "purejaxrl_match", noise=0.05)]
    # Negative control: the comparison must detect a small real difference.
    ctrl = run_env("Breakout-MinAtar", "purejaxrl_match", noise=0.05,
                   ref_overrides={"GAE_LAMBDA": 0.94, "CLIP_EPS": 0.19})
    ctrl_detects = ctrl["gae"]["advantages"] > a.tol and max(
        ctrl["loss"][k] for k in ("total", "value_loss", "actor_loss", "grad")) > a.tol
    print(f"negative control (ref lambda 0.94, clip 0.19): adv diff {ctrl['gae']['advantages']:.2e}, "
          f"loss diff {ctrl['loss']['total']:.2e}, grad diff {ctrl['loss']['grad']:.2e} -> "
          f"{'detected' if ctrl_detects else 'NOT DETECTED'}")
    ok = ctrl_detects
    for r in runs:
        diffs = {**{f"fwd.{k}": v for k, v in r["forward"].items()},
                 "gae.advantages": r["gae"]["advantages"], "gae.targets": r["gae"]["targets"],
                 **{f"loss.{k}": v for k, v in r["loss"].items() if k in ("total", "value_loss", "actor_loss", "entropy", "grad")}}
        r["max_abs_diffs"] = diffs
        r["pass"] = all(v <= a.tol for v in diffs.values())
        ok &= r["pass"]
        print(f"{r['env']:18s} {r['preset']:16s} pass={r['pass']} clip_frac={r['loss']['clip_frac_mean']:.3f} "
              f"dones={r['gae']['n_done']}")
        for k, v in diffs.items():
            print(f"    {k:22s} {v:.3e}")
        print(f"    (scale: |adv|max={r['gae']['advantages_scale']:.3g}, |grad|max={r['loss']['grad_scale']:.3g})")
    import jaxlib
    dev = {"device_platform": jax.devices()[0].platform,
           "device_name": str(getattr(jax.devices()[0], "device_kind", jax.devices()[0])),
           "jax_version": jax.__version__, "jaxlib_version": getattr(jaxlib, "__version__", "unknown"),
           "matmul_precision": ours.matmul_precision()}
    print(f"device: {dev}")
    json.dump({"tol": a.tol, **dev, "runs": runs, "negative_control": ctrl,
               "negative_control_detected": bool(ctrl_detects)}, open(a.out, "w"), indent=1)
    print("ALL PASS" if ok else "SOME FAIL", f"(tol {a.tol:g}); saved {a.out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
