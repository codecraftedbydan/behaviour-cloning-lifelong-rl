"""Task stream for the lifelong experiments: generation, schedule, demo pool.

Decided defaults (team, 24 Oct):
  * No task-switch signal to the agent. Heads, Adam state and value-norm stats carry
    over across switches; there is no task ID in the observation (the goal is visible
    in the grid, nothing else identifies the task). The only thing that happens at a
    switch is that the envs start new episodes, because the old agent position may be
    invalid in the new layout. To the agent that looks like an ordinary episode start.
  * Task order is randomised per seed. Each seed gets its own task chain and revisit
    schedule from `np.random.default_rng([stream_seed, seed])`, so it is a deterministic
    function of (stream_seed, seed). Record both.

Per-seed chain (StreamConfig):
  * Task 0 is fully random. Every new task after it applies ONE shift to the previous
    new task, drawn uniformly from `shift_kinds`:
      "goal"     new start + goal on the same walls (goal moves; BFS dist >= min_dist)
      "dynamics" new action permutation (always different) + slip from slip_choices
      "obs"      new observation permutation
      "layout"   (optional, not in the default) new walls + start + goal
    The shift type of every switch is logged, so the relative metric can be split by type.
  * Scheduled revisits: after every `revisit_every` new tasks, one earlier task (uniform
    over those already seen, excluding the one just trained) is replayed. Its shift type
    is "revisit" (several components may change at once).
  * Each schedule entry is one block of `block_updates` PPO updates (the switch period K).

Demo pool: `make_demo_pool` draws fully random tasks from the same family with its own RNG
and rejects any that shares a wall layout, an observation permutation or a full task with
ANY reference stream (official 500-509 at 40 and 200 tasks, stage-1 700-724 and sweep
750-759 at 40 tasks, and dev 900-939 at pilot length). BC pretraining
therefore never sees a stream layout (tested component-wise).
"""
import dataclasses
import hashlib
from collections import deque

import jax.numpy as jnp
import numpy as np

import gridworld as gw
import seeds as _seeds

SHIFT_KINDS = ("goal", "dynamics", "obs")
SHIFT_CODES = {"init": 0, "goal": 1, "dynamics": 2, "obs": 3, "layout": 4, "revisit": 5}


@dataclasses.dataclass(frozen=True)
class StreamConfig:
    stream_seed: int = 0
    size: int = 8                 # grid incl. border (6x6 interior)
    wall_density: float = 0.25    # fraction of interior cells that are walls
    min_dist: int = 4             # min BFS distance start -> goal
    max_steps: int = 64
    num_new_tasks: int = 20
    revisit_every: int = 3        # 0 = no revisits
    block_updates: int = 50       # K, PPO updates per block
    shift_kinds: tuple = SHIFT_KINDS
    slip_choices: tuple = (0.0,)  # e.g. (0.0, 0.1) for stochastic dynamics
    head_probe_every: int = 0     # lifelong.py head-reset probe period (new tasks); 0 = off
    # Shared layout (stage 1): task 1 and every demo task use the identity obs layout /
    # identity action perm (seeds.SHARED_*). Task 1's random perms are still drawn and
    # then discarded; later obs / dynamics shifts draw fresh random perms, redrawn if they
    # hit the identity (flag on only). Walls always match the pilot twin, but later
    # dynamics permutations can drift from it: the "perm != previous perm" rejection
    # compares against the (now identity) carried perm, which changes the RNG path (README).
    shared_obs_layout: bool = False
    shared_action_perm: bool = False


SHARED_FLAGS = ("shared_obs_layout", "shared_action_perm")


def config_record(cfg: StreamConfig) -> dict:
    """asdict(cfg) for run records; the shared-layout flags appear only when on, so records
    of pre-existing configs (pilot, null_200) are byte-identical to older ones."""
    d = dataclasses.asdict(cfg)
    for k in SHARED_FLAGS:
        if not d[k]:
            del d[k]
    return d


def shared_obs_layout(cfg):
    p = np.asarray(_seeds.SHARED_OBS_LAYOUT, np.int32)
    if len(p) != 3 * cfg.size * cfg.size:
        raise ValueError(f"seeds.SHARED_OBS_LAYOUT is for an 8x8 grid, not size {cfg.size}")
    return p


SHARED_ACTION_PERM = np.asarray(_seeds.SHARED_ACTION_PERM, np.int32)


def apply_shared(t, cfg):
    """Task 1 / demo tasks: replace the (already drawn) perms by the shared ones per flags."""
    if cfg.shared_obs_layout:
        t["obs_perm"] = shared_obs_layout(cfg).copy()
    if cfg.shared_action_perm:
        t["action_perm"] = SHARED_ACTION_PERM.copy()
    return t


def is_shared_obs(t, cfg):
    return len(t["obs_perm"]) == len(_seeds.SHARED_OBS_LAYOUT) and \
        np.array_equal(t["obs_perm"], np.asarray(_seeds.SHARED_OBS_LAYOUT, np.int32))


def is_shared_action(t):
    return np.array_equal(t["action_perm"], SHARED_ACTION_PERM)


# ----------------------------------------------------------------- host-side generation
def _bfs_np(walls, goal):
    H, W = walls.shape
    dist = np.full((H, W), -1, np.int64)
    dist[goal] = 0
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
            rr, cc = r + dr, c + dc
            if not walls[rr, cc] and dist[rr, cc] < 0:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _sample_walls(rng, cfg):
    n = cfg.size
    walls = np.zeros((n, n), bool)
    walls[0] = walls[-1] = walls[:, 0] = walls[:, -1] = True
    interior = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
    k = int(round(cfg.wall_density * len(interior)))
    for i in rng.choice(len(interior), k, replace=False):
        walls[interior[i]] = True
    return walls


def _sample_start_goal(rng, walls, cfg):
    free = np.argwhere(~walls)
    for _ in range(1000):
        goal = tuple(free[rng.integers(len(free))])
        dist = _bfs_np(walls, goal)
        cands = np.argwhere(dist >= cfg.min_dist)
        if len(cands):
            return tuple(cands[rng.integers(len(cands))]), goal
    return None


def _sample_layout(rng, cfg):
    while True:
        walls = _sample_walls(rng, cfg)
        sg = _sample_start_goal(rng, walls, cfg)
        if sg is not None:
            return walls, sg[0], sg[1]


def _sample_dynamics(rng, cfg, prev=None, avoid_shared=False):
    while True:
        perm = rng.permutation(4).astype(np.int32)
        if avoid_shared and np.array_equal(perm, SHARED_ACTION_PERM):
            continue  # shared_action_perm streams: dynamics shifts permute AWAY from identity
        if prev is None or not np.array_equal(perm, prev):
            return perm, float(rng.choice(cfg.slip_choices))


def _sample_obs_perm(rng, cfg, avoid_shared=False):
    while True:
        p = rng.permutation(3 * cfg.size * cfg.size).astype(np.int32)
        if not (avoid_shared and np.array_equal(p, np.arange(len(p)))):
            return p


def _task_dict(walls, start, goal, perm, slip, obs_perm):
    return dict(walls=walls, start=np.array(start, np.int32), goal=np.array(goal, np.int32),
                action_perm=np.asarray(perm, np.int32), slip=np.float32(slip),
                obs_perm=np.asarray(obs_perm, np.int32))


def random_task(rng, cfg):
    walls, start, goal = _sample_layout(rng, cfg)
    perm, slip = _sample_dynamics(rng, cfg)
    return _task_dict(walls, start, goal, perm, slip, _sample_obs_perm(rng, cfg))


def shift_task(prev, kind, rng, cfg):
    t = {k: v.copy() for k, v in prev.items()}
    if kind == "goal":
        while True:
            sg = _sample_start_goal(rng, t["walls"], cfg)
            if sg is None:  # no start/goal pair at min_dist on these walls (never seen in practice)
                raise RuntimeError("goal shift impossible on this wall layout")
            start, goal = sg
            if tuple(goal) != tuple(prev["goal"]):
                break
        t["start"], t["goal"] = np.array(start, np.int32), np.array(goal, np.int32)
    elif kind == "dynamics":
        perm, slip = _sample_dynamics(rng, cfg, prev["action_perm"], avoid_shared=cfg.shared_action_perm)
        t["action_perm"], t["slip"] = perm, np.float32(slip)
    elif kind == "obs":
        t["obs_perm"] = _sample_obs_perm(rng, cfg, avoid_shared=cfg.shared_obs_layout)
    elif kind == "layout":
        walls, start, goal = _sample_layout(rng, cfg)
        t["walls"], t["start"], t["goal"] = walls, np.array(start, np.int32), np.array(goal, np.int32)
    else:
        raise ValueError(kind)
    return t


def layout_key(t):
    return hashlib.sha1(t["walls"].tobytes() + t["start"].tobytes() + t["goal"].tobytes()).hexdigest()


def task_key(t):
    return hashlib.sha1(b"".join(np.ascontiguousarray(t[k]).tobytes() for k in
                                 ("walls", "start", "goal", "action_perm", "slip", "obs_perm"))).hexdigest()


def stack_tasks(tasks, max_steps):
    out = {k: np.stack([t[k] for t in tasks]) for k in tasks[0]}
    return gw.TaskParams(walls=jnp.asarray(out["walls"]), start=jnp.asarray(out["start"]),
                         goal=jnp.asarray(out["goal"]), action_perm=jnp.asarray(out["action_perm"]),
                         slip=jnp.asarray(out["slip"]), obs_perm=jnp.asarray(out["obs_perm"]),
                         max_steps_in_episode=jnp.full((len(tasks),), max_steps, jnp.int32))


def make_schedule(cfg: StreamConfig, rng):
    """Block -> task id, plus is_new flags. New tasks appear in order 0..N-1 (the
    randomisation is in the per-seed chain itself)."""
    sched, is_new = [], []
    for i in range(cfg.num_new_tasks):
        sched.append(i); is_new.append(True)
        if cfg.revisit_every and (i + 1) % cfg.revisit_every == 0 and i >= 1:
            sched.append(int(rng.integers(0, i)))  # an earlier task, not task i itself
            is_new.append(False)
    return np.array(sched, np.int32), np.array(is_new, bool)


def changed_components(a, b):
    """Which components differ between consecutive tasks a -> b."""
    return dict(goal=bool((a["goal"] != b["goal"]).any() or (a["start"] != b["start"]).any()),
                walls=bool((a["walls"] != b["walls"]).any()),
                dynamics=bool((a["action_perm"] != b["action_perm"]).any() or a["slip"] != b["slip"]),
                obs=bool((a["obs_perm"] != b["obs_perm"]).any()))


@dataclasses.dataclass
class Stream:
    cfg: StreamConfig
    seed: int
    tasks: gw.TaskParams      # stacked, leading dim = num_new_tasks
    task_list: list           # host-side dicts
    task_shift: list          # shift kind that created each new task ("init" for task 0)
    schedule: np.ndarray      # (num_blocks,) task id per block
    is_new: np.ndarray        # (num_blocks,) first visit?
    block_shift: np.ndarray   # (num_blocks,) SHIFT_CODES: creating shift kind (new) or revisit
    block_changed: np.ndarray  # (num_blocks, 4) bool: goal, walls, dynamics, obs changed
    optimal_return: np.ndarray  # (num_tasks,) noiseless optimum, for normalisation
    shared_obs: np.ndarray = None     # (num_tasks,) bool: task uses seeds.SHARED_OBS_LAYOUT
    shared_action: np.ndarray = None  # (num_tasks,) bool: task uses seeds.SHARED_ACTION_PERM

    @property
    def num_blocks(self):
        return len(self.schedule)

    @property
    def num_tasks(self):
        return len(self.task_list)


def make_stream(cfg: StreamConfig, seed: int) -> Stream:
    """The task chain + schedule for one agent seed (deterministic in (stream_seed, seed))."""
    rng = np.random.default_rng([cfg.stream_seed, seed, 0])
    tasks, kinds, seen = [apply_shared(random_task(rng, cfg), cfg)], ["init"], set()
    seen.add(task_key(tasks[0]))
    while len(tasks) < cfg.num_new_tasks:
        kind = cfg.shift_kinds[rng.integers(len(cfg.shift_kinds))]
        t = shift_task(tasks[-1], kind, rng, cfg)
        if task_key(t) in seen:
            continue
        seen.add(task_key(t)); tasks.append(t); kinds.append(kind)
    sched, is_new = make_schedule(cfg, np.random.default_rng([cfg.stream_seed, seed, 1]))
    block_shift, changed = [], []
    for b, tid in enumerate(sched):
        if b == 0:
            block_shift.append(SHIFT_CODES["init"]); changed.append([True] * 4); continue
        # New task: the kind of shift that created it from the previous NEW task (after a
        # revisit block the actual change vs the block before is larger; see block_changed).
        block_shift.append(SHIFT_CODES[kinds[tid]] if is_new[b] else SHIFT_CODES["revisit"])
        c = changed_components(tasks[sched[b - 1]], tasks[tid])
        changed.append([c["goal"], c["walls"], c["dynamics"], c["obs"]])
    opt = np.array([1.0 - 0.9 * _bfs_np(t["walls"], tuple(t["goal"]))[tuple(t["start"])] / cfg.max_steps
                    for t in tasks], np.float32)
    return Stream(cfg, seed, stack_tasks(tasks, cfg.max_steps), tasks, kinds, sched, is_new,
                  np.array(block_shift, np.int32), np.array(changed, bool), opt,
                  np.array([is_shared_obs(t, cfg) for t in tasks]), np.array([is_shared_action(t) for t in tasks]))


def stack_streams(streams):
    """Stack per-seed streams (same cfg) along a leading seed axis, for vmap."""
    import jax
    return jax.tree.map(lambda *x: jnp.stack(x), *[s.tasks for s in streams])


PILOT_STREAM = StreamConfig(num_new_tasks=40, block_updates=100, revisit_every=3, head_probe_every=5)
REFERENCE_SEEDS = tuple(sorted(_seeds.PILOT_STREAM_SEEDS)) + _seeds.DEV_REFERENCE_SEEDS  # official pilot + all dev seeds


# Registry of (StreamConfig, seeds) the demo pool must avoid. Extend with
# extend_reference_set(cfg, seeds) BEFORE building a demo pool for any longer stream or new
# seed block (e.g. the 200-task null plan); make_demo_pool refuses otherwise.
# 200-task null-rule run: identical generation settings, num_new_tasks=200, official seeds.
NULL200_STREAM = dataclasses.replace(PILOT_STREAM, num_new_tasks=200)
OFFICIAL_SEEDS = tuple(sorted(_seeds.PILOT_STREAM_SEEDS))
DEV_REFERENCE_SEEDS = _seeds.DEV_REFERENCE_SEEDS
# Stage 1 (Dan approved 30 Oct 2026): 40-task streams with the pilot geometry on the test
# seeds 700-724 and on the sweep / held-out-stream seeds 750-759 (tuning only), with the
# shared layout on (31 Oct): task 1 uses the identity obs layout and action perm.
# PILOT_STREAM itself is untouched (pilot_40 / null_200 replay bit-identically).
STAGE1_STREAM = dataclasses.replace(PILOT_STREAM, shared_obs_layout=True, shared_action_perm=True)
STAGE1_SEEDS = tuple(sorted(_seeds.STAGE1_STREAM_SEEDS))
SWEEP_SEEDS = tuple(sorted(_seeds.SWEEP_STREAM_SEEDS))
_REFERENCE_SPECS = [(PILOT_STREAM, REFERENCE_SEEDS),          # 40-task pilot 500-509 + dev 900-939
                    (NULL200_STREAM, OFFICIAL_SEEDS),         # 200-task null run 500-509
                    # pilot-config registrations of the stage-1 seeds are KEPT (conservative:
                    # the demo pool also stays disjoint from those random-perm variants)
                    (PILOT_STREAM, STAGE1_SEEDS),
                    (PILOT_STREAM, SWEEP_SEEDS),
                    (STAGE1_STREAM, STAGE1_SEEDS),            # stage-1 test streams 700-724
                    (STAGE1_STREAM, SWEEP_SEEDS),             # stage-1 sweep streams 750-759
                    (STAGE1_STREAM, DEV_REFERENCE_SEEDS)]     # stage-1 config, dev 900-939


def extend_reference_set(cfg: StreamConfig, seeds):
    """Register more streams for the demo-pool overlap check (and its test)."""
    _REFERENCE_SPECS.append((cfg, tuple(int(x) for x in seeds)))


def reference_specs():
    return list(_REFERENCE_SPECS)


def reference_streams(specs=None):
    """Every stream the demo pool must avoid, generated without training: by default the
    official seeds 500-509 at 40 new tasks (pilot) and at 200 new tasks (null-rule run),
    the dev seeds 900-939 at 40 new tasks, the stage-1 test seeds 700-724 and sweep seeds
    750-759 at 40 new tasks, plus anything added with extend_reference_set. Shorter streams with the same
    generation settings are exact prefixes of these chains (separate task-chain RNG)."""
    specs = _REFERENCE_SPECS if specs is None else specs
    return [make_stream(cfg, sd) for cfg, seeds in specs for sd in seeds]


def uncovered_streams(streams, refs):
    """Streams whose task chain is not a prefix of a reference stream with the same seed."""
    by_seed = {}
    for r in refs:
        by_seed.setdefault(r.seed, []).append([task_key(t) for t in r.task_list])
    bad = []
    for st in streams:
        keys = [task_key(t) for t in st.task_list]
        if not any(len(rk) >= len(keys) and rk[:len(keys)] == keys for rk in by_seed.get(st.seed, [])):
            bad.append(st)
    return bad


def walls_key(t):
    return hashlib.sha1(t["walls"].tobytes()).hexdigest()


def walls_goal_key(t):
    return hashlib.sha1(t["walls"].tobytes() + t["goal"].tobytes()).hexdigest()


def obs_perm_key(t):
    return hashlib.sha1(t["obs_perm"].tobytes()).hexdigest()


def excluded_obs_perms(streams):
    """Obs-perm keys a demo task must not use: every obs perm of every given stream task,
    EXCEPT the shared layout (seeds.SHARED_OBS_LAYOUT), which demo tasks and stage-1 task 1
    are allowed to share by design (31 Oct). Any non-shared perm stays excluded."""
    shared = hashlib.sha1(np.asarray(_seeds.SHARED_OBS_LAYOUT, np.int32).tobytes()).hexdigest()
    return {obs_perm_key(t) for s in streams for t in s.task_list} - {shared}


DEMO_POOL_MAX_DRAWS_PER_TASK = 200


def make_demo_pool(cfg: StreamConfig, num: int, streams=None, demo_seed: int = 1):
    """Fully random demo tasks from the same family, disjoint from every task of every
    reference stream (and of the given streams, which must be covered by the reference set:
    longer/unregistered streams are refused), component-wise:
      * no shared wall layout (walls alone),
      * hence no shared (walls, goal) or (walls, start, goal),
      * no shared observation permutation, except the shared layout (excluded_obs_perms),
      * no shared full task.
    Action permutations (24 possible) and slip values cannot be disjoint and are not.
    With cfg.shared_obs_layout / shared_action_perm the demo tasks use the shared perms
    (drawn, then replaced: apply_shared). Sampling is capped at
    DEMO_POOL_MAX_DRAWS_PER_TASK * num + 10_000 candidates (RuntimeError)."""
    refs = reference_streams()
    if streams is not None:
        bad = uncovered_streams(streams, refs)
        if bad:
            raise ValueError(
                f"demo pool requested for streams not in the overlap-checked reference set "
                f"(seeds {[x.seed for x in bad]}, up to {max(x.num_tasks for x in bad)} tasks). Register "
                f"them with stream.extend_reference_set(cfg, seeds) and rerun the overlap test first.")
    streams = refs + list(streams or [])
    rng = np.random.default_rng([cfg.stream_seed, 10_000 + demo_seed])
    ex_w = {walls_key(t) for s in streams for t in s.task_list}
    ex_p = excluded_obs_perms(streams)
    ex_t = {task_key(t) for s in streams for t in s.task_list}
    tasks, seen = [], set()
    cap, draws = DEMO_POOL_MAX_DRAWS_PER_TASK * num + 10_000, 0
    while len(tasks) < num:
        if draws >= cap:
            raise RuntimeError(f"make_demo_pool: {draws} candidates drawn, only {len(tasks)} of {num} "
                               f"accepted; the exclusion rules leave too few tasks")
        draws += 1
        t = apply_shared(random_task(rng, cfg), cfg)
        if (walls_key(t) in ex_w or obs_perm_key(t) in ex_p or task_key(t) in ex_t
                or task_key(t) in seen):
            continue
        seen.add(task_key(t)); tasks.append(t)
    return tasks, stack_tasks(tasks, cfg.max_steps)


def probe_batch(stream: Stream, n: int = 512, seed: int = 0):
    """A fixed batch of observations for the health probes: random free cells of random
    stream tasks, rendered with each task's own obs permutation. Same for every
    condition and every point in training, so values are comparable across switches."""
    rng = np.random.default_rng([stream.cfg.stream_seed, stream.seed, 2000 + seed])
    obs = []
    for _ in range(n):
        t = stream.task_list[rng.integers(stream.num_tasks)]
        free = np.argwhere(~t["walls"])
        pos = free[rng.integers(len(free))]
        grid = np.zeros(t["walls"].shape + (3,), np.float32)
        grid[..., 0] = t["walls"]; grid[pos[0], pos[1], 1] = 1; grid[t["goal"][0], t["goal"][1], 2] = 1
        obs.append(grid.reshape(-1)[t["obs_perm"]])
    return jnp.asarray(np.stack(obs))
