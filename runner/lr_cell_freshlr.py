"""Out-of-repo LR runner for every stage-1 sweep and counted cell (one path, incl. lr 2.5e-4).

Runs the FROZEN `cells.py run` CLI unchanged (all of its seed, provenance, BC, chunk and
kickstart checks), with two differences: the base config's `lr` is set to --lr, and (optional, 10 Nov amended
rule, decision 22:04) the FRESH baselines' `lr` is set to --fresh-lr (default: --lr, which is
byte-for-byte the ec4061bc behaviour).
Nothing inside the repo is edited, imported-and-modified on disk, or written.

  PYTHONPATH=<repo> <repo>/.venv/bin/python lr_cell.py --repo <repo> --lr 5e-4 -- \
      --cell pretrained_1M --seeds 750-759 --chunk-size 5 --out-dir <dir> \
      --bc-dir <ckpts> --critic-warmup-updates 10 --matmul-precision highest --allow-official-seeds

Guards (before any import of the repo or any output):
  * --out-dir and --repo are resolved once with os.path.realpath BEFORE the chdir; the
    absolute out-dir is substituted into the forwarded args; out-dir must not lie inside
    realpath(repo) and must not already hold manifest.json or runner_identity.json;
  * forwarded args must contain exactly one `--chunk-size 5` and one `--matmul-precision
    highest` (the fixed stage-1 settings); `=` forms and abbreviations of these flags and of
    --out-dir are refused so argparse cannot pick up a different value.
  * jax.default_backend() must be 'gpu' (no silent CPU fallback); checked before any output.
Writes <out-dir>/runner_identity.json (runner path + sha256, LR, forwarded argv, repo, Python
executable/version, jax/jaxlib versions, backend, jax.devices() and device names) BEFORE training, and after the run
checks: manifest config.lr == --lr and fresh_config.lr == --fresh-lr; in every chunk
config.lr == stream/metrics/lr == --lr, fresh_config.lr == fresh/metrics/lr == --fresh-lr and
(if present) head_probe_config.lr == headprobe/metrics/lr == --lr; status complete; and
device_platform 'gpu'.

--fresh-lr mechanism: lifelong.scratch_config is wrapped. The wrapper sets lr=--fresh-lr ONLY
when called from one of the two fresh-baseline call sites at 33660f6, identified by caller
file, function and exact source line:
  cells.py    run_cell       `fresh_config=json.dumps(ppo.config_record(lifelong.scratch_config(config))),`
  lifelong.py run_condition  `fconfig = scratch_config(config)`
Every other caller (head-reset probe pconfig, arms.plain_config) gets the unwrapped result,
so the head probe keeps the stream LR. Each fresh-site call is counted; the run is refused
afterwards unless both sites were hit (cells that run their own fresh agents).
Pre-training: the first manifest write (before any training) is intercepted and refused unless
config.lr == --lr and fresh_config.lr == --fresh-lr, each checked separately. With --fresh-from
the frozen cells.py check still requires the source cell's fresh_config (lr included) to match.
"""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import linecache
import os
import sys

FIXED = {"--chunk-size": "5", "--matmul-precision": "highest"}
GUARDED = ("--out-dir",) + tuple(FIXED)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def flag_value(rest, flag):
    """Value of a forwarded flag given exactly once as `flag VALUE`; refuses `flag=V`,
    abbreviations (argparse prefix matching) and repeats."""
    for t in rest:
        name = t.split("=", 1)[0]
        if t.startswith("--") and name != flag and flag.startswith(name) and len(name) > 2:
            raise SystemExit(f"refusing abbreviated flag {t!r} (write {flag} in full)")
        if name == flag and "=" in t:
            raise SystemExit(f"refusing {t!r}: write `{flag} VALUE`")
    idx = [j for j, t in enumerate(rest) if t == flag]
    if len(idx) != 1 or idx[0] + 1 >= len(rest):
        raise SystemExit(f"forwarded args need {flag} exactly once, with a value")
    return idx[0]


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: lr_cell.py --repo REPO --lr LR -- <cells.py run args>")
    i = argv.index("--")
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--repo", required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--fresh-lr", type=float, default=None,
                   help="LR of the fresh baselines only (default: --lr)")
    a = p.parse_args(argv[:i])
    rest = list(argv[i + 1:])
    if not a.lr > 0:
        raise SystemExit("--lr must be > 0")
    fresh_lr = a.lr if a.fresh_lr is None else a.fresh_lr
    if not fresh_lr > 0:
        raise SystemExit("--fresh-lr must be > 0")
    for flag in GUARDED:
        flag_value(rest, flag)
    for flag, want in FIXED.items():
        got = rest[flag_value(rest, flag) + 1]
        if got != want:
            raise SystemExit(f"{flag} {got!r} refused: stage-1 sweep/counted cells use {flag} {want}")

    # Resolve everything BEFORE the chdir (review fix 1, 9 Nov).
    me = os.path.realpath(__file__)
    repo = os.path.realpath(a.repo)
    j = flag_value(rest, "--out-dir")
    out_dir = os.path.realpath(rest[j + 1])
    rest[j + 1] = out_dir
    if os.path.commonpath([repo, out_dir]) == repo:
        raise SystemExit(f"--out-dir {out_dir} must be outside the repo {repo}")
    if os.path.exists(os.path.join(out_dir, "manifest.json")) or \
            os.path.exists(os.path.join(out_dir, "runner_identity.json")):
        raise SystemExit(f"{out_dir} already holds a manifest or runner identity: refusing")

    sys.path.insert(0, repo)
    os.chdir(repo)  # provenance/git checks run against the repo, as with `cells.py run`
    import cells
    import jax

    # Refuse a silent CPU (or other) fallback before any output (decision, 9 Nov 11:57).
    backend = jax.default_backend()
    if backend != "gpu":
        raise SystemExit(f"jax.default_backend() is {backend!r}, not 'gpu': refusing before any output")
    devices = jax.devices()

    frozen_run_cell = cells.run_cell

    def run_cell_with_lr(*args, base_config=None, **kw):
        if base_config is None:
            raise SystemExit("internal: cells.main passed no base_config")
        return frozen_run_cell(*args, base_config=dataclasses.replace(base_config, lr=a.lr), **kw)

    cells.run_cell = run_cell_with_lr

    import lifelong
    frozen_scratch_config = lifelong.scratch_config
    fresh_sites = {
        ("cells.py", "run_cell"):
            "fresh_config=json.dumps(ppo.config_record(lifelong.scratch_config(config))),",
        ("lifelong.py", "run_condition"): "fconfig = scratch_config(config)",
    }
    site_hits = {k: 0 for k in fresh_sites}

    def scratch_config_fresh_lr(config):
        out = frozen_scratch_config(config)
        f = sys._getframe(1)
        fn = os.path.realpath(f.f_code.co_filename)
        key = (os.path.basename(fn), f.f_code.co_name)
        line = linecache.getline(f.f_code.co_filename, f.f_lineno).strip()
        if key in fresh_sites and os.path.dirname(fn) == repo and line == fresh_sites[key]:
            site_hits[key] += 1
            return dataclasses.replace(out, lr=fresh_lr)
        return out

    lifelong.scratch_config = scratch_config_fresh_lr

    # Pre-training check (decision 22:04: check config.lr and fresh_config.lr separately): the
    # first manifest write happens before any training; refuse there if either LR is wrong.
    frozen_write_manifest = cells._write_manifest
    pre = {"checked": False}

    def write_manifest_checked(d, man):
        if not pre["checked"]:
            c_lr = json.loads(man["config"])["lr"]
            f_lr = json.loads(man["fresh_config"])["lr"]
            if c_lr != a.lr:
                raise SystemExit(f"pre-check failed: config.lr {c_lr} != --lr {a.lr}; nothing trained")
            if f_lr != fresh_lr:
                raise SystemExit(f"pre-check failed: fresh_config.lr {f_lr} != --fresh-lr {fresh_lr}; "
                                 f"nothing trained")
            pre["checked"] = True
        return frozen_write_manifest(d, man)

    cells._write_manifest = write_manifest_checked
    try:
        jaxlib_version = importlib.metadata.version("jaxlib")
    except importlib.metadata.PackageNotFoundError:
        jaxlib_version = None
    runner_sha = sha256(me)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "runner_identity.json"), "w") as f:
        json.dump({"runner": me, "runner_sha256": runner_sha, "lr": a.lr, "fresh_lr": fresh_lr,
                   "fresh_lr_explicit": a.fresh_lr is not None,
                   "forwarded_argv": ["run"] + rest, "repo": repo,
                   "python_executable": sys.executable, "python_version": sys.version,
                   "jax_version": jax.__version__, "jaxlib_version": jaxlib_version,
                   "jax_backend": backend, "jax_devices": [str(d) for d in devices],
                   "device_kinds": [d.device_kind for d in devices]}, f, indent=1)
    cells.main(["run"] + rest)
    import numpy as np
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)
    problems = [] if pre["checked"] else ["pre-training manifest check never ran"]
    got = (json.loads(man["config"])["lr"], json.loads(man["fresh_config"])["lr"])
    if got != (a.lr, fresh_lr):
        problems.append(f"manifest config/fresh lr {got} != {(a.lr, fresh_lr)}")
    if man.get("status") != "complete" or man.get("device_platform") != "gpu":
        problems.append(f"status {man.get('status')}, device_platform {man.get('device_platform')!r}")
    own_fresh = bool(man.get("fresh"))
    if own_fresh and not all(site_hits.values()):
        problems.append(f"fresh-lr call sites not all hit: {site_hits}")
    for c in man["chunks"]:
        z = np.load(os.path.join(out_dir, c["file"]), allow_pickle=False)
        checks = [("config", "stream/metrics/lr", a.lr), ("fresh_config", "fresh/metrics/lr", fresh_lr)]
        if "head_probe_config" in z.files:
            checks.append(("head_probe_config", "headprobe/metrics/lr", a.lr))
        # metric keys are REQUIRED (decision 11 Nov 08:10): stream and head probe always, fresh
        # whenever fresh runs are present (own or shared via --fresh-from)
        for ck, mk, want in checks:
            if ck not in z.files or mk not in z.files:
                problems.append(f"{c['file']}: missing {ck if ck not in z.files else mk}")
                continue
            if json.loads(str(z[ck]))["lr"] != want:
                problems.append(f"{c['file']}: {ck}.lr != {want}")
            if not np.all(np.asarray(z[mk]) == np.float32(want)):
                problems.append(f"{c['file']}: {mk} values {np.unique(z[mk])[:4]} != {want}")
    if problems:
        raise SystemExit("post-check failed: " + "; ".join(problems))
    print(f"lr_cell: OK lr={a.lr} fresh_lr={fresh_lr} (manifest + every chunk); "
          f"fresh sites {site_hits}; runner sha256 {runner_sha}")

if __name__ == "__main__":
    main()
