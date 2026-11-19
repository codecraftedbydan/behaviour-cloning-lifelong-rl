"""Stage-1 model sizes and fine-tuning arms (stage1-proposal sec. 2 cells 1-12;
tars-test-rules sec. 7). Everything here is opt-in: the default arm "scratch" with no
--size leaves every existing configuration unchanged.

Sizes (cells 1-8). PPO, shared trunk, full FT, ~0.1M / ~0.3M / ~1M / ~3-10M params.
  The rules set the exact widths "from pilot / GPU throughput", which has not been
  measured, so SIZE_WIDTHS are PROVISIONAL: the centre is pinned to the pilot width (950,
  1,091,555 params, the width the BC dev checks used) and the others are the widths whose
  param count is closest to the nominal target (width_for_params). The ~3-10M end is
  offered at both ends ("3M", "10M"); choosing one is left to the throughput measurement.

Arms (all PPO; pretrained arms start from that seed's BC checkpoint, bc.checkpoint_name):
  scratch            random init, full FT                         (cells 1-8 scratch, centre)
  pretrained         BC init, full FT, critic warm-up              (cells 1-8 pre)
  lora               BC init, LoRA r=8 a=16 on every trunk Dense,  (cell 9)
                     base trunk frozen, adapters + heads train, critic warm-up
  frozen_pretrained  BC init, trunk frozen, heads train, warm-up   (cell 10)
  frozen_random      random init, trunk frozen, heads train        (cell 11, H6)
  kickstart          BC init, full FT, warm-up, + kickstart_coef * (cell 12)
                     KL(pi_BC || pi) to that seed's frozen BC policy
  "trunk" = every hidden layer: trunk/layer_i (+ layer_i_ln) for the shared trunk,
  actor_i / critic_i (+ LN) for separate towers; "heads" = actor_out and critic_out.
  The critic warm-up length is not fixed here (rules sec. 7 measures it); pretrained arms
  must be given it explicitly, scratch arms refuse it.

Kickstarting coefficient (cell 12): constant inside the shared-layout window and exactly 0
from the first block whose task does not have BOTH the shared obs layout and the shared
action perm, for the rest of the stream (never re-enabled; decision, 2 Nov; implemented in
lifelong.kickstart_gate_blocks, logged as kickstart_coef_block / kickstart_coef_task). Per
seed, kickstart_effective_updates counts the updates with coefficient > 0 after the critic
warm-up; a run where any seed has 0 is refused before training (decision, 2 Nov 11:36). The
value is chosen by an equal-trial (KICKSTART_SWEEP_TRIALS = 4) sweep on the held-out
stream (sweep stream seeds 750-759, fresh 850-859, tuning only), see
scripts/kickstart_sweep.py. The candidate values are NOT specified by the rules and must
be passed explicitly.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import ppo

LORA_RANK = 8      # tars-test-rules sec. 7 / stage1-proposal cell 9
LORA_ALPHA = 16.0
KICKSTART_SWEEP_TRIALS = 4

SIZE_TARGETS = {"0.1M": 1e5, "0.3M": 3e5, "1M": 1e6, "3M": 3e6, "10M": 1e7}
CENTRE_WIDTH = 950  # pilot / BC-dev-check width, 1,091,555 params (pinned, not re-derived)

ARMS = {
    #                    init    warmup  freeze_trunk  lora   kickstart
    "scratch":           dict(init="scratch", warmup=False, freeze_trunk=False, lora=False, kickstart=False),
    "pretrained":        dict(init="bc", warmup=True, freeze_trunk=False, lora=False, kickstart=False),
    "lora":              dict(init="bc", warmup=True, freeze_trunk=False, lora=True, kickstart=False),
    "frozen_pretrained": dict(init="bc", warmup=True, freeze_trunk=True, lora=False, kickstart=False),
    "frozen_random":     dict(init="scratch", warmup=False, freeze_trunk=True, lora=False, kickstart=False),
    "kickstart":         dict(init="bc", warmup=True, freeze_trunk=False, lora=False, kickstart=True),
}


# ----------------------------------------------------------------------------- sizes
def dims(config: ppo.Config):
    """(obs_dim, num_actions) of the config's env (flattened obs)."""
    import numpy as np
    env, env_params, _, obs_shape = ppo.make_env_and_network(config)
    return int(np.prod(obs_shape)), int(env.action_space(env_params).n)


def formula_param_count(width, num_layers, obs_dim, num_actions, shared_trunk=True,
                        layer_norm=False):
    """Analytic param count of the plain ActorCritic (tested against network.init)."""
    ln = 2 * width if layer_norm else 0
    tower = obs_dim * width + width + ln + (num_layers - 1) * (width * width + width + ln)
    if shared_trunk:
        return tower + (width + 1) * num_actions + (width + 1)
    return 2 * tower + (width + 1) * num_actions + (width + 1)


def width_for_params(target, num_layers, obs_dim, num_actions, shared_trunk=True, layer_norm=False):
    """Width whose param count is closest to target (ties: smaller width)."""
    best, w = None, 1
    while True:
        n = formula_param_count(w, num_layers, obs_dim, num_actions, shared_trunk, layer_norm)
        d = abs(n - target)
        if best is None or d < best[0]:
            best = (d, w)
        if n > target:
            return best[1]
        w += 1


def size_widths(config: ppo.Config):
    """{size name: width} for config's env / depth / architecture. PROVISIONAL (module doc)."""
    obs_dim, na = dims(config)
    out = {k: width_for_params(t, config.num_layers, obs_dim, na, config.shared_trunk, config.layer_norm)
           for k, t in SIZE_TARGETS.items()}
    out["1M"] = CENTRE_WIDTH
    return out


def apply_size(config: ppo.Config, size):
    if size is None:
        return config
    widths = size_widths(config)
    if size not in widths:
        raise ValueError(f"size {size!r} not in {sorted(widths)}")
    return dataclasses.replace(config, hidden_size=widths[size])


# ----------------------------------------------------------------------------- arms
def arm_config(config: ppo.Config, arm: str, warmup_updates=None, kickstart_coef=None) -> ppo.Config:
    """Config of a stage-1 arm on top of config (which must have no arm hooks set)."""
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {sorted(ARMS)}")
    if config.critic_warmup_updates or config.kickstart_coef or config.freeze_trunk or config.lora_rank:
        raise ValueError("arm_config: base config already has arm hooks set")
    a = ARMS[arm]
    if a["warmup"]:
        if warmup_updates is None:
            raise ValueError(f"arm {arm!r} is pretrained: pass the critic warm-up length explicitly "
                             "(rules sec. 7 measures it; 0 is allowed but must be stated)")
        if int(warmup_updates) < 0:
            raise ValueError("warm-up must be >= 0")
    elif warmup_updates:
        raise ValueError(f"arm {arm!r} is a scratch arm: no critic warm-up")
    if a["kickstart"]:
        if kickstart_coef is None or not float(kickstart_coef) > 0:
            raise ValueError("kickstart arm needs kickstart_coef > 0 (from the 4-trial sweep)")
    elif kickstart_coef:
        raise ValueError(f"arm {arm!r} has no kickstarting term")
    return dataclasses.replace(
        config,
        critic_warmup_updates=int(warmup_updates or 0),
        adam_reset_at_handover=True,
        freeze_trunk=a["freeze_trunk"],
        lora_rank=LORA_RANK if a["lora"] else 0,
        lora_alpha=LORA_ALPHA,
        kickstart_coef=float(kickstart_coef) if a["kickstart"] else 0.0,
    )


def plain_config(config: ppo.Config) -> ppo.Config:
    """The network the BC checkpoint was trained with (no adapters, no arm hooks)."""
    import lifelong
    return lifelong.scratch_config(config)


def load_bc_checkpoints(config: ppo.Config, scfg, seeds, bc_dir, allow_dirty=False,
                        expected_tree_hash=None):
    """Per-seed BC params for config's (plain) network, checked against the run: seed,
    architecture, shared-layout flags, and clean provenance unless allow_dirty.
    expected_tree_hash (step 6, cells.py): if given, every checkpoint's recorded
    source_tree_hash (provenance.record at BC time) must equal it; a checkpoint without one
    is refused too. None (lifelong.py / kickstart_sweep.py): not checked, as before.
    Returns (list of params, record dict)."""
    import bc
    import stream as S
    pc = plain_config(config)
    pqn_cfg = None
    if config.algo == "pqn":
        # PQN (cells 15-16): the BC checkpoint is the PPO+LN network of the same trunk
        # (cells 13-14's). Load it as that network, then graft its trunk onto this seed's
        # scratch PQN init (lifelong.scratch_init_params), so q_out is the scratch q_out.
        pqn_cfg, pc = pc, dataclasses.replace(pc, algo="ppo")
    params, files = [], {}
    want_arch = {k: getattr(pc, k) for k in ("hidden_size", "num_layers", "shared_trunk",
                                             "layer_norm", "activation", "env_name")}
    flags = {k: v for k, v in S.config_record(scfg).items() if k in S.SHARED_FLAGS}
    for s in seeds:
        path = os.path.join(bc_dir, bc.checkpoint_name(pc, s))
        if not os.path.exists(path):
            raise FileNotFoundError(f"no BC checkpoint for seed {s}: {path}")
        p, meta = bc.load_checkpoint(path, pc)
        if int(meta.get("seed", -1)) != int(s):
            raise ValueError(f"{path}: checkpoint seed {meta.get('seed')} != {s}")
        got = {k: meta["config"][k] for k in want_arch}
        if got != want_arch:
            raise ValueError(f"{path}: architecture {got} != run {want_arch}")
        got_flags = {k: v for k, v in meta["stream_config"].items() if k in S.SHARED_FLAGS}
        if got_flags != flags:
            raise ValueError(f"{path}: BC shared-layout flags {got_flags} != run {flags}")
        if meta.get("git_dirty") and not allow_dirty:
            raise ValueError(f"{path}: BC checkpoint came from a dirty tree (pass allow_dirty)")
        if expected_tree_hash is not None:
            got_h = meta.get("source_tree_hash")
            if not got_h:
                raise ValueError(f"{path}: BC checkpoint has no source_tree_hash (no provenance); "
                                 f"expected {expected_tree_hash}")
            if got_h != expected_tree_hash:
                raise ValueError(f"{path}: BC source_tree_hash {got_h} != expected {expected_tree_hash} "
                                 f"(checkpoint from commit {meta.get('git_commit_full', meta.get('git_commit'))}); "
                                 "re-run bc.py at this commit or pin the hash explicitly")
        if meta["config"].get("algo", "ppo") != "ppo":
            raise ValueError(f"{path}: BC checkpoint algo {meta['config'].get('algo')!r} != 'ppo'")
        with open(path, "rb") as f:
            files[str(s)] = hashlib.sha256(f.read()).hexdigest()
        if pqn_cfg is not None:
            import lifelong
            import pqn
            p = pqn.graft_trunk(lifelong.scratch_init_params(pqn_cfg, s), p)
        params.append(p)
    rec = {"bc_dir": os.path.abspath(bc_dir), "bc_checkpoint_sha256": json.dumps(files)}
    if expected_tree_hash is not None:
        rec["bc_expected_tree_hash"] = str(expected_tree_hash)
    return params, rec


def prepare(config: ppo.Config, scfg, seeds, arm="scratch", size=None, warmup_updates=None,
            kickstart_coef=None, bc_dir=None, allow_dirty=False):
    """(config, run_condition kwargs, record) for an arm. Pretrained arms load bc_dir;
    the kickstart teacher is the same seed's BC checkpoint (frozen)."""
    config = arm_config(apply_size(config, size), arm, warmup_updates, kickstart_coef)
    kw, rec = {}, {}
    if arm != "scratch" or size is not None:
        rec = {"arm": arm}
        if size is not None:
            rec["size"] = size
    if ARMS[arm]["init"] == "bc":
        if bc_dir is None:
            raise ValueError(f"arm {arm!r} needs bc_dir (per-seed BC checkpoints)")
        ps, r = load_bc_checkpoints(config, scfg, seeds, bc_dir, allow_dirty)
        kw["seed_init_params"] = ps
        if ARMS[arm]["kickstart"]:
            kw["seed_teacher_params"] = ps
        rec.update(r)
    elif bc_dir is not None:
        raise ValueError(f"arm {arm!r} is a scratch arm: bc_dir must not be given")
    return config, kw, rec


# ----------------------------------------------------------------------------- sweep hook
def check_kickstart_sweep(coefs, seeds):
    """Equal-trial sweep: exactly KICKSTART_SWEEP_TRIALS distinct coefficients > 0, run on
    the held-out sweep stream (750-759) or, for smoke tests only, dev seeds >= 900."""
    from seeds import DEV_MIN_SEED, SWEEP_STREAM_SEEDS
    coefs = [float(c) for c in coefs]
    if len(coefs) != KICKSTART_SWEEP_TRIALS or len(set(coefs)) != len(coefs) or min(coefs) <= 0:
        raise ValueError(f"kickstart sweep needs exactly {KICKSTART_SWEEP_TRIALS} distinct "
                         f"coefficients > 0, got {coefs}")
    seeds = [int(s) for s in seeds]
    sweep = set(SWEEP_STREAM_SEEDS)
    if not (all(s in sweep for s in seeds) or all(s >= DEV_MIN_SEED for s in seeds)):
        raise ValueError(f"kickstart sweep seeds must all be sweep seeds {min(sweep)}-{max(sweep)} "
                         f"(held-out stream) or all dev seeds >= {DEV_MIN_SEED}, got {seeds}")
    return coefs, seeds


def sweep_binding(seeds, fresh_seeds):
    """(binding, reason). A kickstart-sweep ranking is binding only for stream seeds exactly
    750-759 with fresh agents exactly 850-859; dev-seed (900+) runs are mechanics-only."""
    from seeds import SWEEP_STREAM_SEEDS
    seeds, fresh = sorted(int(s) for s in seeds), sorted(int(s) for s in fresh_seeds)
    if seeds != sorted(SWEEP_STREAM_SEEDS):
        return False, f"mechanics-only: stream seeds {seeds} are not exactly 750-759"
    if fresh != [s + 100 for s in sorted(SWEEP_STREAM_SEEDS)]:
        return False, f"mechanics-only: fresh seeds {fresh} are not exactly 850-859"
    return True, "binding: stream seeds 750-759, fresh agents 850-859"
