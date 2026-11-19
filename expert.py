"""BFS shortest-path expert and demo-dataset generation (for later BC pretraining).

* `bfs_distance(walls, goal)`: shortest-path distance (in moves) from every cell to the
  goal; walls and unreachable cells get a large value. Pure JAX (H*W rounds of
  relaxation), jit/vmap-friendly.
* `expert_action(task, pos, dist)`: an optimal action for the agent. It handles the
  dynamics permutation: agent action a moves in direction action_perm[a], so we pick
  the a whose resulting cell has the smallest distance (ties -> lowest a). With
  slip > 0 the greedy-on-distance policy is the natural expert (every non-slipped
  move is optimal), but the expected return is no longer exactly the noiseless optimum.
* `generate_demos(tasks, ...)`: roll out the expert with epsilon-noise. At every step,
  with prob epsilon the executed action is uniformly random; otherwise it is the planner's
  action. The planner then recovers from wherever the agent ends up, so the data covers
  off-optimal states (DART-style noise injection). **We store both**: `label` = the
  planner's action at the visited state (the BC target), and `action_taken` = the
  action actually executed. epsilon = 0 is the noiseless ablation.
"""
import jax
import jax.numpy as jnp
import numpy as np

import gridworld as gw

UNREACHABLE = 10_000


def bfs_distance(walls, goal):
    H, W = walls.shape
    dist = jnp.full((H, W), UNREACHABLE, jnp.int32).at[goal[0], goal[1]].set(0)

    def relax(_, d):
        pad = jnp.pad(d, 1, constant_values=UNREACHABLE)
        nbr = jnp.minimum(jnp.minimum(pad[:-2, 1:-1], pad[2:, 1:-1]),
                          jnp.minimum(pad[1:-1, :-2], pad[1:-1, 2:]))
        return jnp.where(walls, UNREACHABLE, jnp.minimum(d, nbr + 1))

    return jax.lax.fori_loop(0, H * W, relax, dist)


def optimal_return(task, dist=None):
    """Return of the noiseless optimal policy from the start cell."""
    if dist is None:
        dist = bfs_distance(task.walls, task.goal)
    d = dist[task.start[0], task.start[1]]
    return 1.0 - 0.9 * d / task.max_steps_in_episode


def expert_action(task, pos, dist):
    nxt = jax.vmap(lambda a: gw.move(task.walls, pos, task.action_perm[a]))(jnp.arange(gw.NUM_ACTIONS))
    d = dist[nxt[:, 0], nxt[:, 1]]
    return jnp.argmin(d).astype(jnp.int32)  # argmin returns the first min -> lowest a


def expert_episode(env, task, key, epsilon, max_steps):
    """One epsilon-noisy expert episode. Returns per-step arrays (length max_steps) and a
    `valid` mask (steps before the episode ended)."""
    dist = bfs_distance(task.walls, task.goal)
    k_reset, key = jax.random.split(key)
    obs, state = env.reset_env(k_reset, task)

    def step(carry, _):
        obs, state, alive, key = carry
        key, k_eps, k_rand, k_env = jax.random.split(key, 4)
        label = expert_action(task, state.pos, dist)
        noisy = jax.random.uniform(k_eps) < epsilon
        taken = jnp.where(noisy, jax.random.randint(k_rand, (), 0, gw.NUM_ACTIONS), label)
        obs2, state2, reward, done, _ = env.step_env(k_env, state, taken, task)
        out = dict(obs=obs, label=label, action_taken=taken, reward=reward * alive,
                   valid=alive, pos=state.pos)
        return (obs2, state2, alive * (1.0 - done), key), out

    _, traj = jax.lax.scan(step, (obs, state, jnp.float32(1.0), key), None, length=max_steps)
    return traj


def generate_demos(tasks, episodes_per_task, epsilon, key, max_steps=None, size=None):
    """Demo dataset from a stacked pool of tasks. Returns a dict of numpy arrays with
    one row per valid step: obs (N, F), label (N,), action_taken (N,), task_id (N,),
    episode_id (N,), plus per-episode `episode_return` and `episode_length`
    (episodes_total,) and `task_of_episode`."""
    size = size or int(tasks.walls.shape[-1])
    env = gw.Gridworld(size)
    T = int(tasks.walls.shape[0])
    max_steps = max_steps or int(np.max(np.asarray(tasks.max_steps_in_episode)))
    keys = jax.random.split(key, T * episodes_per_task).reshape(T, episodes_per_task, -1)
    run = jax.jit(jax.vmap(jax.vmap(expert_episode, in_axes=(None, None, 0, None, None)),
                           in_axes=(None, 0, 0, None, None)), static_argnums=(0, 4))
    traj = jax.tree.map(np.asarray, run(env, tasks, keys, epsilon, max_steps))  # (T, E, S, ...)
    valid = traj["valid"] > 0
    task_id = np.broadcast_to(np.arange(T)[:, None, None], valid.shape)
    ep_id = np.broadcast_to(np.arange(T * episodes_per_task).reshape(T, episodes_per_task, 1), valid.shape)
    return dict(
        obs=traj["obs"][valid], label=traj["label"][valid], action_taken=traj["action_taken"][valid],
        task_id=task_id[valid], episode_id=ep_id[valid],
        episode_return=traj["reward"].sum(-1).reshape(-1), episode_length=valid.sum(-1).reshape(-1),
        task_of_episode=np.repeat(np.arange(T), episodes_per_task), epsilon=np.float32(epsilon),
    )
