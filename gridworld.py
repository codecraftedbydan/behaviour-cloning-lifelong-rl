"""Procedural gridworld for the lifelong task stream. Pure JAX, gymnax-style API.

A *task* is everything in `TaskParams`:
  * layout: interior walls, a start cell and a goal cell (the outer border is always wall);
  * dynamics: an action permutation (agent action a moves in direction action_perm[a])
    and a slip probability (with prob `slip` the move is a uniformly random direction);
  * observation permutation: a fixed permutation of the flat observation features.

All task fields are arrays, so a task is a pytree. A stack of tasks has a leading
task axis, and switching tasks inside a jitted loop is just indexing.

Observation (full observability, chosen so the task is Markov and the BFS expert
is a true optimum): three binary channels over the H x W grid (wall, agent, goal),
flattened to F = 3*H*W features and then permuted, obs = flat[obs_perm]. With the
identity permutation the layout is (H, W, 3) row-major.

Reward: on reaching the goal, 1 - 0.9 * t / max_steps (as in MiniGrid), else 0.
The episode ends at the goal (termination) or after max_steps (truncation). The
optimal return of a noiseless task is 1 - 0.9 * d / max_steps, where d is the BFS
distance from start to goal (see expert.py). We use that to normalise returns.
"""
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces

# Directions: 0 up, 1 right, 2 down, 3 left (row, col deltas).
DIRS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)
NUM_ACTIONS = 4


@struct.dataclass
class TaskParams:
    walls: jax.Array        # (H, W) bool, border included
    start: jax.Array        # (2,) int32
    goal: jax.Array         # (2,) int32
    action_perm: jax.Array  # (4,) int32: agent action -> direction
    slip: jax.Array         # () float32
    obs_perm: jax.Array     # (F,) int32
    max_steps_in_episode: Any = 64  # name/semantics shared with gymnax (truncation)


@struct.dataclass
class EnvState:
    pos: jax.Array   # (2,) int32
    time: jax.Array  # () int32


def render_flat(walls, pos, goal):
    """Unpermuted (H*W*3,) float32 observation."""
    H, W = walls.shape
    agent = jnp.zeros((H, W), jnp.float32).at[pos[0], pos[1]].set(1.0)
    goal_c = jnp.zeros((H, W), jnp.float32).at[goal[0], goal[1]].set(1.0)
    return jnp.stack([walls.astype(jnp.float32), agent, goal_c], -1).reshape(-1)


def move(walls, pos, direction):
    """Move one cell in `direction` unless a wall blocks it."""
    new = pos + DIRS[direction]
    return jnp.where(walls[new[0], new[1]], pos, new)


class Gridworld(environment.Environment):
    def __init__(self, size: int = 8):
        super().__init__()
        self.size = size
        self.num_features = 3 * size * size

    @property
    def name(self) -> str:
        return "Gridworld"

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def default_params(self) -> TaskParams:
        """Open room, start top-left, goal bottom-right, identity dynamics/obs."""
        n = self.size
        walls = jnp.zeros((n, n), bool).at[0].set(True).at[-1].set(True).at[:, 0].set(True).at[:, -1].set(True)
        return TaskParams(walls=walls, start=jnp.array([1, 1], jnp.int32),
                          goal=jnp.array([n - 2, n - 2], jnp.int32),
                          action_perm=jnp.arange(4, dtype=jnp.int32), slip=jnp.float32(0.0),
                          obs_perm=jnp.arange(self.num_features, dtype=jnp.int32),
                          max_steps_in_episode=64)

    def get_obs(self, state: EnvState, params: TaskParams, key=None) -> jax.Array:
        return render_flat(params.walls, state.pos, params.goal)[params.obs_perm]

    def is_terminal(self, state: EnvState, params: TaskParams) -> jax.Array:
        reached = jnp.all(state.pos == params.goal)
        return jnp.logical_or(reached, state.time >= params.max_steps_in_episode)

    def reset_env(self, key, params: TaskParams):
        state = EnvState(pos=params.start.astype(jnp.int32), time=jnp.zeros((), jnp.int32))
        return self.get_obs(state, params), state

    def step_env(self, key, state: EnvState, action, params: TaskParams):
        k_slip, k_dir = jax.random.split(key)
        slipped = jax.random.uniform(k_slip) < params.slip
        direction = jnp.where(slipped, jax.random.randint(k_dir, (), 0, 4),
                              params.action_perm[action])
        pos = move(params.walls, state.pos, direction)
        time = state.time + 1
        reached = jnp.all(pos == params.goal)
        reward = jnp.where(reached, 1.0 - 0.9 * time / params.max_steps_in_episode, 0.0)
        state = EnvState(pos=pos, time=time)
        done = self.is_terminal(state, params)
        return (jax.lax.stop_gradient(self.get_obs(state, params)), jax.lax.stop_gradient(state),
                reward.astype(jnp.float32), done, {"discount": 1.0 - done.astype(jnp.float32)})

    def action_space(self, params=None):
        return spaces.Discrete(NUM_ACTIONS)

    def observation_space(self, params=None):
        return spaces.Box(0.0, 1.0, (self.num_features,), jnp.float32)
