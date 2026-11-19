"""Run provenance (step-2 review, review fix 2): source-file hashes, git commit and a
clean-tree guard for the CLI paths and run scripts.

  * source_hashes(): sha256 of the working-tree content of every tracked *.py file and
    every tracked file under scripts/ (git ls-files), plus every UNTRACKED *.py file and
    untracked file under scripts/ (review of 0291582, fix 1), plus a combined
    tree hash. Untracked files under results/, checkpoints/ and .venv/ are never counted
    (UNTRACKED_EXCLUDE); other untracked files (e.g. *.out logs) are not sources.
  * check_clean(seeds, allow_dirty): refuses (SystemExit) to run if the tree is dirty:
    tracked changes (git status --porcelain --untracked-files=no) or any untracked source
    file as above. --allow-dirty is accepted only for
    dev seeds (all >= seeds.DEV_MIN_SEED; ppo.py CartPole runs: no reserved seed), and is
    recorded. Under pytest the guard only records (tests run on a working tree), unless
    force=True (used by the guard's own tests).
  * Unhashable trees (review notes on 6c10b27): a source that cannot be read -- a broken
    *.py symlink, or a tracked file deleted from the working tree -- and any symlink to a
    directory inside the repo (its *.py files would be importable but are invisible to git
    and to the hashes) are listed by dirty_files() ("!! missing <p>", "!! broken-symlink <p>",
    "!! symlinked-dir <d>") and check_clean REFUSES them with a SystemExit even under
    --allow-dirty, never a FileNotFoundError. source_hashes() records such files as
    "missing" / "broken-symlink" instead of raising.
  * record(...) -> dict of scalars/strings stored in every run record and checkpoint meta.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

import seeds as _seeds

ROOT = os.path.dirname(os.path.abspath(__file__))


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


UNTRACKED_EXCLUDE = ("results/", "checkpoints/", ".venv/")
SOURCE_PATHSPEC = ("*.py", "scripts/*")


def tracked_sources():
    out = _git("ls-files", "-z", "--", *SOURCE_PATHSPEC)
    return sorted(p for p in out.split("\0") if p)


def untracked_sources():
    """Untracked *.py / scripts/* files (gitignored ones included: a .py file is a source
    wherever it sits), except under results/, checkpoints/ and .venv/."""
    excl = [f":(exclude){d.rstrip('/')}" for d in UNTRACKED_EXCLUDE]
    out = _git("ls-files", "-z", "--others", "--", *SOURCE_PATHSPEC, *excl)
    return sorted(p for p in out.split("\0")
                  if p and not p.startswith(UNTRACKED_EXCLUDE) and "/__pycache__/" not in f"/{p}")


UNHASHABLE = ("missing", "broken-symlink")


def _digest(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.lexists(path):
        return "missing"
    if os.path.islink(path) and not os.path.exists(path):
        return "broken-symlink"
    if os.path.isdir(path):
        return "symlinked-dir" if os.path.islink(path) else "missing"
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def source_hashes():
    return {p: _digest(p) for p in sorted(set(tracked_sources()) | set(untracked_sources()))}


def symlinked_dirs():
    """Symlinks to directories anywhere in the tree (outside .git and UNTRACKED_EXCLUDE)."""
    out = []
    for d, dirs, _ in os.walk(ROOT, followlinks=False):
        rel_d = os.path.relpath(d, ROOT)
        keep = []
        for n in sorted(dirs):
            rel = n if rel_d == "." else f"{rel_d}/{n}"
            if n in (".git", "__pycache__") or (rel + "/").startswith(UNTRACKED_EXCLUDE):
                continue
            if os.path.islink(os.path.join(d, n)):
                out.append(rel)
            else:
                keep.append(n)
        dirs[:] = keep
    return out


def unhashable():
    """'!! <reason> <path>' for every source that cannot be hashed, plus symlinked dirs."""
    bad = [f"!! {v} {p}" for p, v in source_hashes().items() if v in UNHASHABLE]
    return bad + [f"!! symlinked-dir {d}" for d in symlinked_dirs()]


def tree_hash(hashes):
    return hashlib.sha256("".join(f"{p}:{v}\n" for p, v in sorted(hashes.items())).encode()).hexdigest()


def git_commit():
    try:
        return _git("rev-parse", "HEAD").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dirty_files():
    """Tracked files with uncommitted changes (staged or not), plus untracked source files
    (untracked_sources, listed as "?? path")."""
    out = _git("status", "--porcelain", "--untracked-files=no")
    return [line[3:] for line in out.splitlines() if line.strip()] + \
        [f"?? {p}" for p in untracked_sources()] + unhashable()


def under_pytest():
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def check_clean(seeds, allow_dirty=False, require_dev=True, force=False):
    """Refuse a dirty tracked tree unless allow_dirty with dev seeds only. Returns record()."""
    dirty = dirty_files()
    if dirty and (force or not under_pytest()):
        shown = ", ".join(dirty[:8]) + (" ..." if len(dirty) > 8 else "")
        bad_src = [d for d in dirty if d.startswith("!! ")]
        if bad_src:
            raise SystemExit(f"tracked tree is dirty and cannot be hashed ({', '.join(bad_src[:8])}): "
                             "restore or remove these (refused even with --allow-dirty)")
        if not allow_dirty:
            raise SystemExit(f"tracked tree is dirty ({shown}): commit first (run records must "
                             "name a commit), or pass --allow-dirty (dev seeds only)")
        seeds = list(seeds) if seeds is not None else []
        if require_dev:
            bad = [s for s in seeds if s < _seeds.DEV_MIN_SEED]
            if bad or not seeds:
                raise SystemExit(f"--allow-dirty is for dev seeds (>= {_seeds.DEV_MIN_SEED}) only; got {bad or seeds}")
        else:
            bad = [s for s in seeds if s in _seeds.RESERVED_SEEDS]
            if bad:
                raise SystemExit(f"--allow-dirty is refused with reserved seeds {bad}")
    return record(dirty=bool(dirty), allow_dirty=allow_dirty)


def record(dirty=None, allow_dirty=False):
    h = source_hashes()
    if dirty is None:
        dirty = bool(dirty_files())
    return dict(git_commit_full=git_commit(), git_dirty=bool(dirty), allow_dirty=bool(allow_dirty),
                source_tree_hash=tree_hash(h), source_hashes=json.dumps(h, sort_keys=True))
