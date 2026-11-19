"""Stage-1 cell runner (step 6): one named cell = one arm x size x stream, run over a
fixed seed list in fixed vmap chunks, with every pre-run check done before any training.

  PYTHONPATH=. python cells.py run --cell scratch_1M --seeds 900-903 --chunk-size 2 \
      --out-dir <dir> [--num-new-tasks N --block-updates K ...]          (dev smoke)
  PYTHONPATH=. python cells.py run --cell reset_heads_1M --seeds ... --chunk-size 2 \
      --fresh-from <dir of the scratch_1M cell> --out-dir <dir>
  PYTHONPATH=. python cells.py merge <cell dir> --out <file.npz>

Checks (all refuse with a clear message, before the first chunk trains):
  * seed guard (lifelong.check_seeds) and clean-tree guard (provenance.check_clean);
  * BC checkpoints (pretrained, lora, frozen_pretrained, kickstart, ln_pretrained): each
    seed's meta["seed"] must equal the stream seed, and its recorded source_tree_hash must
    equal the run's own source_tree_hash (BC made at this exact code), unless
    --bc-tree-hash pins a different expected hash, which is then recorded
    (bc_tree_hash_pinned); a checkpoint without a hash is refused; pinning is refused when any
    stream or fresh seed is reserved (counted runs cannot pin; decision 2 Nov) and allowed only on
    dev seeds 900-1899, whose records carry ranking_binding false;
  * kickstart: zero effective kickstart updates for any seed (lifelong.check_kickstart_window);
  * reset cells R-H / R-HOV (reset_heads_1M, reset_heads_adam_vn_1M) always run with
    fresh=False (--no-fresh) and take the fresh-agent runs of the centre scratch cell
    (scratch_1M) given by --fresh-from. Refused if that cell is missing or incomplete, or
    differs in commit, source tree hash, config, fresh config, stream config, seeds, fresh
    seeds, chunk layout, matmul precision or device platform. No other cell accepts
    --fresh-from.

Chunking (tars-test-rules sec. 7, decision 31 Oct 22:36): results reproduce bit for bit only at
the same vmap batch composition, so --chunk-size is required, fixed per cell, and the
layout (contiguous chunks of the seed list in the given order; the last one may be
shorter) is recorded in the manifest and in every chunk record (chunk_size, chunk_layout).
merge_cell() refuses chunk records whose layout, commit, tree hash or configs differ from
the manifest, or whose file hash changed; check_same_layout() refuses to pair cells with
different layouts. Use --chunk-size >= number of seeds for an unchunked cell.

Output: <out-dir>/manifest.json + chunk_<i>.npz (each a normal lifelong run record for its
seeds). The out-dir must not already hold a manifest (nothing is overwritten).
PQN cells (15-16) are not here yet (step 5).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time

import numpy as np

import arms
import lifelong
import ppo
import pqn
import provenance
import seeds as SD
import stream as S

CENTRE_SCRATCH = "scratch_1M"


@dataclasses.dataclass(frozen=True)
class Cell:
    arm: str
    size: str
    layer_norm: bool = False
    reset_arm: str = "none"
    fresh_from: str | None = None   # name of the cell whose fresh runs this cell reuses
    number: str = ""
    algo: str = "ppo"               # "pqn": cells 15-16 (pqn.py); recorded only when not "ppo"


def cell_spec(cell: Cell) -> dict:
    """asdict(cell) for manifests and chunks; "algo" appears only for non-PPO cells, so every
    PPO cell's recorded spec is unchanged."""
    d = dataclasses.asdict(cell)
    if cell.algo == "ppo":
        d.pop("algo")
    return d


def _cells():
    c = {}
    for i, size in enumerate(("0.1M", "0.3M", "1M", "3M", "10M")):
        c[f"scratch_{size}"] = Cell("scratch", size, number="1-8")
        c[f"pretrained_{size}"] = Cell("pretrained", size, number="1-8")
    c["lora_1M"] = Cell("lora", "1M", number="9")
    c["frozen_pretrained_1M"] = Cell("frozen_pretrained", "1M", number="10")
    c["frozen_random_1M"] = Cell("frozen_random", "1M", number="11")
    c["kickstart_1M"] = Cell("kickstart", "1M", number="12")
    c["ln_scratch_1M"] = Cell("scratch", "1M", layer_norm=True, number="13-14")
    c["ln_pretrained_1M"] = Cell("pretrained", "1M", layer_norm=True, number="13-14")
    # PQN ~1M (LayerNorm), full FT, pre + scratch (H5, vs cells 13-14). Same trunk as cells
    # 13-14 (width 950, LayerNorm); pretrained = the ln_pretrained_1M BC checkpoint's trunk.
    c["pqn_scratch_1M"] = Cell("scratch", "1M", layer_norm=True, number="15-16", algo="pqn")
    c["pqn_pretrained_1M"] = Cell("pretrained", "1M", layer_norm=True, number="15-16", algo="pqn")
    c["reset_heads_1M"] = Cell("scratch", "1M", reset_arm="heads", fresh_from=CENTRE_SCRATCH, number="17")
    c["reset_heads_adam_vn_1M"] = Cell("scratch", "1M", reset_arm="heads_adam_vn",
                                       fresh_from=CENTRE_SCRATCH, number="18")
    return c


CELLS = _cells()
TIMING_KEYS = ("wall_seconds", "stream_seconds", "fresh_seconds", "head_probe_seconds")
# Top-level (unprefixed) per-seed arrays of a lifelong run record: concatenated on merge.
PER_SEED_KEYS = ("seeds", "fresh_seeds", "task_of_block", "is_new", "block_shift", "block_changed",
                 "optimal_return", "task_shift", "kickstart_coef_block", "kickstart_on_block",
                 "kickstart_coef_task", "kickstart_effective_updates", "kickstart_effective_updates_block",
                 "kickstart_effective_updates_revisit", "task_shared_obs_layout", "task_shared_action_perm")
PER_SEED_PREFIXES = ("stream/", "fresh/", "headprobe/")
CHUNK_ONLY_KEYS = ("chunk_index", "chunk_seeds", "bc_checkpoint_sha256")
# Fields that must match between a reset cell and the centre scratch cell it takes fresh runs from.
FRESH_IDENTITY = ("git_commit_full", "source_tree_hash", "config", "fresh_config", "stream_config",
                  "seeds", "fresh_seeds", "chunk_size", "chunk_layout", "matmul_precision",
                  "device_platform")


def chunk_layout(seeds, chunk_size):
    seeds = [int(s) for s in seeds]
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1 (use >= len(seeds) for an unchunked cell)")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError(f"seed list must be non-empty with no repeats: {seeds}")
    k = int(chunk_size)
    return [seeds[i:i + k] for i in range(0, len(seeds), k)]


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cell_config(cell: Cell, base: ppo.Config, warmup_updates=None, kickstart_coef=None):
    base = dataclasses.replace(base, layer_norm=True) if cell.layer_norm else base
    if cell.algo == "pqn":
        if kickstart_coef is not None:
            raise ValueError("PQN cells take no kickstart coefficient")
        base = pqn.pqn_config(base)
    config = arms.arm_config(arms.apply_size(base, cell.size), cell.arm, warmup_updates, kickstart_coef)
    if config.algo == "pqn":
        pqn.check_config(config)
    return config


def read_manifest(cell_dir):
    path = os.path.join(cell_dir, "manifest.json")
    if not os.path.exists(path):
        raise ValueError(f"no cell manifest at {path}")
    with open(path) as f:
        return json.load(f)


# Fields two paired cells (e.g. pretrained vs scratch for H1') must share: FRESH_IDENTITY
# without the arm-specific config / fresh_config.
PAIR_IDENTITY = tuple(k for k in FRESH_IDENTITY if k not in ("config", "fresh_config"))


def check_same_layout(a: dict, b: dict):
    """Refuse to pair two cell manifests whose seeds or chunk layouts differ, or that differ in
    any other PAIR_IDENTITY field (commit, source tree hash, stream config, fresh seeds, chunk
    size, matmul precision, device platform)."""
    if a["seeds"] != b["seeds"] or a["chunk_layout"] != b["chunk_layout"]:
        raise ValueError(f"chunk layouts differ: {a['cell']} {a['chunk_layout']} vs "
                         f"{b['cell']} {b['chunk_layout']}")
    for k in PAIR_IDENTITY:
        if a.get(k) != b.get(k):
            raise ValueError(f"cells {a['cell']!r} and {b['cell']!r} cannot be paired: different {k}: "
                             f"{a.get(k)!r} != {b.get(k)!r}")


def check_fresh_source(man: dict, src: dict, src_dir: str):
    """The reset cell (man, not yet run) may reuse src's fresh runs only if src is the
    centre scratch cell, complete, and identical in every FRESH_IDENTITY field."""
    if src.get("cell") != man["fresh_from_cell"]:
        raise ValueError(f"{src_dir}: fresh runs must come from cell {man['fresh_from_cell']!r}, "
                         f"not {src.get('cell')!r}")
    if src.get("status") != "complete":
        raise ValueError(f"{src_dir}: cell {src.get('cell')!r} is not complete ({src.get('status')})")
    if not src.get("fresh"):
        raise ValueError(f"{src_dir}: cell {src.get('cell')!r} has no fresh runs of its own")
    for k in FRESH_IDENTITY:
        if src.get(k) != man.get(k):
            raise ValueError(f"{src_dir}: fresh runs came from a different {k}: "
                             f"{src.get(k)!r} != {man.get(k)!r}")


def _load_chunk(cell_dir, man, i):
    ent = man["chunks"][i]
    path = os.path.join(cell_dir, ent["file"])
    if not os.path.exists(path):
        raise ValueError(f"{path}: chunk file missing")
    if _sha256(path) != ent["sha256"]:
        raise ValueError(f"{path}: file changed since it was written (sha256 mismatch)")
    d = np.load(path, allow_pickle=False)
    rec = {k: d[k] for k in d.files}
    for k, want in (("chunk_layout", json.dumps(man["chunk_layout"])), ("chunk_size", man["chunk_size"]),
                    ("git_commit_full", man["git_commit_full"]), ("source_tree_hash", man["source_tree_hash"]),
                    ("config", man["config"]), ("stream_config", man["stream_config"]),
                    ("cell", man["cell"])):
        got = rec[k].item() if k in rec else None
        if got != want:
            raise ValueError(f"{path}: {k} {got!r} != manifest {want!r}")
    if [int(s) for s in rec["seeds"]] != man["chunk_layout"][i] or int(rec["chunk_index"]) != i:
        raise ValueError(f"{path}: seeds {rec['seeds'].tolist()} / index {int(rec['chunk_index'])} "
                         f"do not match chunk {i} of the layout {man['chunk_layout']}")
    return rec


def merge_cell(cell_dir):
    """Merged run record of a complete cell (same keys and shapes as one unchunked
    lifelong run record over all its seeds; timings become chunk_<key> arrays plus the
    total; bc_checkpoint_sha256 is merged). Refuses anything inconsistent."""
    man = read_manifest(cell_dir)
    if man.get("status") != "complete" or len(man["chunks"]) != len(man["chunk_layout"]):
        raise ValueError(f"{cell_dir}: cell is not complete ({man.get('status')}, "
                         f"{len(man['chunks'])}/{len(man['chunk_layout'])} chunks)")
    check_pin_record(man, man["seeds"])
    recs = [_load_chunk(cell_dir, man, i) for i in range(len(man["chunk_layout"]))]
    for r in recs:
        check_pin_record(r, r["seeds"].tolist())
    keys = set(recs[0])
    for i, r in enumerate(recs[1:], 1):
        if set(r) != keys:
            raise ValueError(f"chunk {i} has different keys: {sorted(set(r) ^ keys)}")
    out = {}
    for k in sorted(keys):
        vals = [r[k] for r in recs]
        if k in TIMING_KEYS:
            out["chunk_" + k] = np.array([float(v) for v in vals])
            out[k] = float(sum(float(v) for v in vals))
        elif k == "bc_checkpoint_sha256":
            m = {}
            for v in vals:
                m.update(json.loads(str(v)))
            out[k] = json.dumps(m)
        elif k in ("chunk_index",):
            continue
        elif k in PER_SEED_KEYS or (k.startswith(PER_SEED_PREFIXES) and vals[0].ndim >= 1):
            out[k] = np.concatenate(vals, axis=0)
        else:
            a = vals[0]
            for i, v in enumerate(vals[1:], 1):
                same = (str(a) == str(v)) if a.dtype.kind in "OUS" else (
                    a.shape == v.shape and np.array_equal(a, v, equal_nan=a.dtype.kind == "f"))
                if not same:
                    raise ValueError(f"key {k!r} differs between chunk 0 and chunk {i}: refusing to merge")
            out[k] = a
    out["cell_manifest_sha256"] = _sha256(os.path.join(cell_dir, "manifest.json"))
    return out


def _write_manifest(out_dir, man):
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))


def _reserved_hits(seeds):
    """Reserved seeds (seeds.RESERVED_SEEDS) among the stream seeds and their fresh seeds."""
    hits = set()
    for s in (int(x) for x in seeds):
        if s in SD.RESERVED_SEEDS:
            hits.add(s)
        try:
            f = SD.fresh_seed(s)
        except ValueError:
            continue
        if f in SD.RESERVED_SEEDS:
            hits.add(f)
    return sorted(hits)


def check_pin_allowed(seeds, bc_tree_hash):
    """Counted runs cannot pin the BC hash (decision, 2 Nov 2026): --bc-tree-hash is refused if any
    stream or fresh seed is reserved, and allowed only when every stream seed is a dev seed
    (900-1899; mechanics only, ranking_binding false). Runs before the seed guard and any work."""
    if bc_tree_hash is None:
        return
    hits = _reserved_hits(seeds)
    if hits:
        raise ValueError(f"--bc-tree-hash refused: seeds {hits} are reserved ({SD.BLOCKS_TEXT}); "
                         "counted runs cannot pin the BC hash: regenerate BC at this commit")
    non_dev = sorted(int(s) for s in seeds if not SD.DEV_MIN_SEED <= int(s) < SD.DEV_MAX_SEED)
    if non_dev:
        raise ValueError(f"--bc-tree-hash refused: seeds {non_dev} are not dev seeds "
                         f"({SD.DEV_MIN_SEED}-{SD.DEV_MAX_SEED - 1}); pinning is for dev mechanics only")


def cell_binding(seeds, pinned):
    """(ranking_binding, reason) of a cell run. Binding only with every stream seed and every
    fresh seed reserved (one block, enforced by the seed guard) and an unpinned BC hash."""
    seeds = [int(s) for s in seeds]
    if pinned:
        return False, "mechanics-only: BC source hash pinned (--bc-tree-hash)"
    if seeds and all(s in SD.OFFICIAL_STREAM_SEEDS for s in seeds) and \
            all(SD.fresh_seed(s) in SD.OFFICIAL_FRESH_SEEDS for s in seeds):
        return True, f"binding: reserved {SD.seed_block(seeds[0])} block seeds"
    return False, f"mechanics-only: stream seeds {seeds} are not reserved seeds"


def check_pin_record(rec, seeds):
    """Defensive check at record time (manifest, every chunk, merge): a pinned BC hash never
    sits in a record with reserved seeds or ranking_binding true. Raises AssertionError
    (explicit raise, so python -O cannot strip it)."""
    pinned = bool(np.asarray(rec.get("bc_tree_hash_pinned", False)).item())
    binding = bool(np.asarray(rec.get("ranking_binding", False)).item())
    hits = _reserved_hits(seeds)
    if pinned and (hits or binding):
        raise AssertionError(f"bc_tree_hash_pinned=true with reserved seeds {hits} / ranking_binding="
                             f"{binding}: counted runs cannot pin the BC hash")


def run_cell(cell_name, seeds, chunk_size, out_dir, base_config=None, scfg=None, bc_dir=None,
             bc_tree_hash=None, warmup_updates=None, kickstart_coef=None, fresh_from=None,
             allow_official_seeds=False, allow_dirty=False, verbose=True):
    """Run one cell; returns its manifest. See the module docstring for the checks."""
    if cell_name not in CELLS:
        raise ValueError(f"unknown cell {cell_name!r}; known: {sorted(CELLS)}")
    cell = CELLS[cell_name]
    seeds = [int(s) for s in seeds]
    if cell.fresh_from and not fresh_from:
        raise ValueError(f"cell {cell_name!r} runs with --no-fresh and needs --fresh-from "
                         f"(the {cell.fresh_from} cell directory)")
    if fresh_from and not cell.fresh_from:
        raise ValueError(f"cell {cell_name!r} runs its own fresh agents; --fresh-from is only for "
                         f"the reset cells")
    check_pin_allowed(seeds, bc_tree_hash)   # before the seed guard: no reserved seed gets further
    lifelong.check_seeds(seeds, allow_official_seeds)
    prov = provenance.check_clean(seeds, allow_dirty)
    layout = chunk_layout(seeds, chunk_size)
    if os.path.exists(os.path.join(out_dir, "manifest.json")):
        raise ValueError(f"{out_dir} already holds a cell manifest: refusing to overwrite")
    base = base_config or ppo.Config(**ppo.resolve_presets("lifelong_stream"))
    scfg = scfg or S.STAGE1_STREAM
    config = cell_config(cell, base, warmup_updates, kickstart_coef)
    streams = [S.make_stream(scfg, s) for s in seeds]
    # BC checkpoints: all seeds, before any training
    kw_all, arm_rec = {}, {"arm": cell.arm, "size": cell.size}
    if arms.ARMS[cell.arm]["init"] == "bc":
        if bc_dir is None:
            raise ValueError(f"cell {cell_name!r} needs bc_dir (per-seed BC checkpoints)")
        want = bc_tree_hash if bc_tree_hash is not None else prov["source_tree_hash"]
        ps, r = arms.load_bc_checkpoints(config, scfg, seeds, bc_dir, allow_dirty, expected_tree_hash=want)
        arm_rec.update(r, bc_tree_hash_pinned=bc_tree_hash is not None)
        kw_all["seed_init_params"] = ps
        if arms.ARMS[cell.arm]["kickstart"]:
            kw_all["seed_teacher_params"] = ps
    elif bc_dir is not None or bc_tree_hash is not None:
        raise ValueError(f"cell {cell_name!r} is a scratch cell: no BC checkpoints")
    if config.kickstart_coef > 0:
        lifelong.check_kickstart_window(seeds, np.stack([lifelong.kickstart_effective_block_updates(
            st, scfg.block_updates, config.critic_warmup_updates) for st in streams]))
    binding, why = cell_binding(seeds, arm_rec.get("bc_tree_hash_pinned", False))
    arm_rec.update(ranking_binding=binding, ranking_binding_reason=why)
    check_pin_record(arm_rec, seeds)
    dev = ppo.device_info()
    man = dict(cell=cell_name, cell_spec=cell_spec(cell), seeds=seeds,
               fresh_seeds=[lifelong.fresh_seed(s) for s in seeds],
               chunk_size=int(chunk_size), chunk_layout=layout,
               config=json.dumps(ppo.config_record(config)),
               fresh_config=json.dumps(ppo.config_record(lifelong.scratch_config(config))),
               stream_config=json.dumps(S.config_record(scfg)),
               matmul_precision=dev["matmul_precision"], device_platform=dev["device_platform"],
               fresh=not cell.fresh_from, fresh_from_cell=cell.fresh_from, fresh_from_dir=None,
               reset_arm=cell.reset_arm, status="running", chunks=[],
               bc_tree_hash_pinned=bool(arm_rec.get("bc_tree_hash_pinned", False)),
               ranking_binding=binding, ranking_binding_reason=why,
               **{k: v for k, v in prov.items() if k != "source_hashes"})
    src_man = None
    if cell.fresh_from:
        src_man = read_manifest(fresh_from)
        check_fresh_source(man, src_man, fresh_from)
        man["fresh_from_dir"] = os.path.abspath(fresh_from)
        man["fresh_from_manifest_sha256"] = _sha256(os.path.join(fresh_from, "manifest.json"))
    os.makedirs(out_dir, exist_ok=True)
    _write_manifest(out_dir, man)
    for i, cs in enumerate(layout):
        idx = [seeds.index(s) for s in cs]
        kw = {k: [v[j] for j in idx] for k, v in kw_all.items()}
        t0 = time.perf_counter()
        res = lifelong.run_condition(config, scfg, cs, fresh=not cell.fresh_from, verbose=verbose,
                                     allow_official_seeds=allow_official_seeds,
                                     reset_arm=cell.reset_arm, allow_dirty=allow_dirty, **kw)
        res.update(arm_rec)
        if arm_rec.get("bc_checkpoint_sha256"):
            full = json.loads(arm_rec["bc_checkpoint_sha256"])
            res["bc_checkpoint_sha256"] = json.dumps({str(s): full[str(s)] for s in cs})
        if cell.fresh_from:
            src = _load_chunk(fresh_from, src_man, i)
            if src["fresh_config"].item() != man["fresh_config"] or src["fresh_seeds"].tolist() != \
                    np.asarray(res["fresh_seeds"]).tolist():
                raise ValueError(f"{fresh_from}: chunk {i} fresh runs do not match this cell")
            res.update({k: src[k] for k in src if k.startswith("fresh/")})
            res["fresh_config"] = src["fresh_config"].item()
            res["fresh_seconds"] = 0.0
            res["fresh_shared_from"] = man["fresh_from_dir"]
            res["fresh_shared_manifest_sha256"] = man["fresh_from_manifest_sha256"]
        res["wall_seconds"] = time.perf_counter() - t0
        res["config"] = man["config"]
        res["stream_config"] = man["stream_config"]
        res["git_commit"] = lifelong.git_commit()
        res.update(ppo.device_info())
        res["default_backend"] = __import__("jax").default_backend()
        res.update(prov)
        res.update(cell=cell_name, chunk_index=i, chunk_size=int(chunk_size),
                   chunk_layout=json.dumps(layout), cell_spec=json.dumps(cell_spec(cell)))
        check_pin_record(res, cs)
        fn = f"chunk_{i:03d}.npz"
        path = os.path.join(out_dir, fn)
        np.savez_compressed(path, **res)
        man["chunks"].append(dict(file=fn, seeds=cs, sha256=_sha256(path)))
        _write_manifest(out_dir, man)
        if verbose:
            print(f"cell {cell_name}: chunk {i + 1}/{len(layout)} seeds {cs} saved", flush=True)
    man["status"] = "complete"
    _write_manifest(out_dir, man)
    return man


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cell", required=True, choices=sorted(CELLS))
    r.add_argument("--seeds", required=True)
    r.add_argument("--chunk-size", type=int, required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--bc-dir")
    r.add_argument("--bc-tree-hash", help="expected BC source_tree_hash if not this tree's (recorded)")
    r.add_argument("--critic-warmup-updates", type=int)
    r.add_argument("--kickstart-coef", type=float)
    r.add_argument("--fresh-from", help="reset cells: directory of the scratch_1M cell")
    r.add_argument("--preset", default="lifelong_stream")
    r.add_argument("--num-new-tasks", type=int, default=S.STAGE1_STREAM.num_new_tasks)
    r.add_argument("--block-updates", type=int, default=S.STAGE1_STREAM.block_updates)
    r.add_argument("--revisit-every", type=int, default=S.STAGE1_STREAM.revisit_every)
    r.add_argument("--head-probe-every", type=int, default=S.STAGE1_STREAM.head_probe_every)
    r.add_argument("--matmul-precision", choices=ppo.MATMUL_PRECISIONS, default="default")
    r.add_argument("--allow-official-seeds", action="store_true")
    r.add_argument("--allow-dirty", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("cell_dir")
    m.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "merge":
        try:
            res = merge_cell(a.cell_dir)
        except ValueError as e:
            raise SystemExit(f"merge: {e}")
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        np.savez_compressed(a.out, **res)
        print(f"saved {a.out}")
        return res
    seeds = ppo.parse_seeds(a.seeds)
    try:
        check_pin_allowed(seeds, a.bc_tree_hash)   # before the seed guard, output or work
    except ValueError as e:
        raise SystemExit(f"cell {a.cell}: {e}")
    lifelong.check_seeds(seeds, a.allow_official_seeds)   # before any output or work
    provenance.check_clean(seeds, a.allow_dirty)
    ppo.set_matmul_precision(a.matmul_precision)
    scfg = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=a.num_new_tasks, block_updates=a.block_updates,
                               revisit_every=a.revisit_every, head_probe_every=a.head_probe_every)
    try:
        return run_cell(a.cell, seeds, a.chunk_size, a.out_dir,
                        base_config=ppo.Config(**ppo.resolve_presets(a.preset)), scfg=scfg,
                        bc_dir=a.bc_dir, bc_tree_hash=a.bc_tree_hash,
                        warmup_updates=a.critic_warmup_updates, kickstart_coef=a.kickstart_coef,
                        fresh_from=a.fresh_from, allow_official_seeds=a.allow_official_seeds,
                        allow_dirty=a.allow_dirty)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"cell {a.cell}: {e}")


if __name__ == "__main__":
    main()
