"""Run PureJaxRL's unmodified purejaxrl/ppo.py make_train, vmapped over seeds.

brax / navix are imported at module level by purejaxrl/wrappers.py but only used
by Brax/Navix wrapper classes we never touch, so we stub them instead of installing.
"""
import argparse, sys, time, types, json
for name in ["brax", "brax.envs", "brax.envs.wrappers", "brax.envs.wrappers.training", "navix"]:
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["brax"].envs = sys.modules["brax.envs"]
sys.modules["brax.envs.wrappers.training"].EpisodeWrapper = object
sys.modules["brax.envs.wrappers.training"].AutoResetWrapper = object
sys.path.insert(0, "purejaxrl")
import jax, jax.numpy as jnp, numpy as np
from ppo import make_train  # PureJaxRL's ppo.py, unmodified

CARTPOLE = {  # purejaxrl/ppo.py __main__ defaults (DEBUG dropped)
    "LR": 2.5e-4, "NUM_ENVS": 4, "NUM_STEPS": 128, "TOTAL_TIMESTEPS": 5e5,
    "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 4, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "tanh", "ENV_NAME": "CartPole-v1", "ANNEAL_LR": True,
}
MINATAR = {  # examples/brax_minatar.ipynb "PPO Gymnax MinAtar" config, env swapped to Breakout
    "LR": 5e-3, "NUM_ENVS": 64, "NUM_STEPS": 128, "TOTAL_TIMESTEPS": 1e7,
    "UPDATE_EPOCHS": 4, "NUM_MINIBATCHES": 8, "GAMMA": 0.99, "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2, "ENT_COEF": 0.01, "VF_COEF": 0.5, "MAX_GRAD_NORM": 0.5,
    "ACTIVATION": "relu", "ENV_NAME": "Breakout-MinAtar", "ANNEAL_LR": True,
}

p = argparse.ArgumentParser()
p.add_argument("--preset", choices=["cartpole", "minatar"], required=True)
p.add_argument("--total-timesteps", type=float)
p.add_argument("--num-seeds", type=int, default=5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--seeds", help='explicit list, e.g. "100-104": one PRNGKey(s) per seed')
p.add_argument("--out", required=True)
p.add_argument("--env", help="override ENV_NAME, e.g. Asterix-MinAtar (config otherwise unchanged)")
a = p.parse_args()
config = dict(CARTPOLE if a.preset == "cartpole" else MINATAR)
if a.env: config["ENV_NAME"] = a.env
if a.total_timesteps: config["TOTAL_TIMESTEPS"] = a.total_timesteps

train = make_train(config)

def run(keys):
    info = jax.vmap(train)(keys)["metrics"]  # leaves (S, U, T, N)
    r = info["returned_episode_returns"]
    return {
        # Same definition as ours: last finished-episode return, at the end of each
        # rollout, mean over envs.
        "end_of_rollout": r[:, :, -1, :].mean(-1),
        # PureJaxRL's usual plotting definition: mean over all T steps and envs.
        "all_steps": r.mean((-2, -1)),
    }

if a.seeds:
    lo_hi = [x for x in a.seeds.split(",")]
    seed_list = []
    for part in lo_hi:
        if "-" in part:
            lo, hi = map(int, part.split("-")); seed_list += list(range(lo, hi + 1))
        else:
            seed_list.append(int(part))
    keys = jnp.stack([jax.random.PRNGKey(s) for s in seed_list]); a.num_seeds = len(seed_list)
else:
    seed_list = [-1]
    keys = jax.random.split(jax.random.PRNGKey(a.seed), a.num_seeds)
fn = jax.jit(run)
t0 = time.perf_counter(); c = fn.lower(keys).compile(); comp = time.perf_counter() - t0
t0 = time.perf_counter(); out = jax.block_until_ready(c(keys)); run_s = time.perf_counter() - t0
steps = a.num_seeds * config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
print(f"compile {comp:.1f}s run {run_s:.1f}s {steps/run_s:,.0f} env-steps/s")
out = {k: np.asarray(v) for k, v in out.items()}
print("final (last 10 updates, end-of-rollout def):", np.round(out["end_of_rollout"][:, -10:].mean(1), 2))
np.savez(a.out, env_steps=(np.arange(config["NUM_UPDATES"]) + 1) * config["NUM_STEPS"] * config["NUM_ENVS"],
         ret_end_of_rollout=out["end_of_rollout"], ret_all_steps=out["all_steps"],
         run_seconds=run_s, compile_seconds=comp, steps_per_sec=steps / run_s,
         config=json.dumps(config), seeds=np.array(seed_list),
         device_platform=jax.devices()[0].platform,
         device_name=str(getattr(jax.devices()[0], "device_kind", jax.devices()[0])),
         jax_version=jax.__version__, jaxlib_version=__import__("jaxlib").__version__)
print("saved", a.out)
