"""Step 6 / 6c10b27 TODO (a): the dirty-tree refusal checked for real, outside the
in-process guard. The committed HEAD is cloned to a temp dir, the clone is dirtied, and
every CLI runs there as a subprocess with PYTEST_CURRENT_TEST unset (so the guard is
live, not record-only). Each must exit 1 with the refusal on stderr, before printing
anything or writing any output. The clone is a plain `git clone` (no worktree is added to
this repo)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLIS = {  # every CLI that writes run records or checkpoints; dev seed 990 only
    "lifelong": ["lifelong.py", "--seeds", "990", "--out", "OUT/x.npz"],
    "ppo": ["ppo.py", "--seeds", "990", "--total-timesteps", "1000", "--out", "OUT/x.npz"],
    "bc": ["bc.py", "--seeds", "990", "--out-dir", "OUT"],
    "bc_dev_checks": ["scripts/bc_dev_checks.py", "--seeds", "990", "--out-dir", "OUT"],
    "kickstart_sweep": ["scripts/kickstart_sweep.py", "--seeds", "990", "--coefs", "1,2,3,4",
                        "--bc-dir", "OUT/bc", "--critic-warmup-updates", "1", "--out-dir", "OUT"],
    "cells": ["cells.py", "run", "--cell", "scratch_1M", "--seeds", "990", "--chunk-size", "1",
              "--out-dir", "OUT"],
}


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    d = tmp_path_factory.mktemp("dirty") / "repo"
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", ROOT, str(d)], check=True)
    return d


def _env(repo):
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.update(JAX_PLATFORMS="cpu", PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE="1")
    return env


def _run_all(repo, out_root, extra=()):
    procs = {}
    for name, argv in CLIS.items():
        out = out_root / name
        args = [a.replace("OUT", str(out)) for a in argv] + list(extra)
        procs[name] = (out, subprocess.Popen([sys.executable] + args, cwd=repo, env=_env(repo),
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    res = {}
    for name, (out, p) in procs.items():
        so, se = p.communicate(timeout=600)
        res[name] = (p.returncode, so, se, out)
    return res


def _dirty_check(repo):
    code = "import provenance, json; print(json.dumps(provenance.dirty_files()))"
    return subprocess.run([sys.executable, "-c", code], cwd=repo, env=_env(repo), capture_output=True,
                          text=True, check=True).stdout.strip()


def test_clone_is_clean_and_live_guard(clone):
    assert _dirty_check(clone) == "[]"  # control: the refusals below come from the dirt
    assert "PYTEST_CURRENT_TEST" not in _env(clone)


@pytest.mark.parametrize("kind", ["tracked", "untracked", "unhashable"])
def test_every_cli_refuses_dirty_tree_in_subprocess(clone, tmp_path, kind):
    restore = None
    if kind == "tracked":
        p = clone / "stream.py"
        orig = p.read_text()
        p.write_text(orig + "\n# dirty\n")
        restore = lambda: p.write_text(orig)  # noqa: E731
        want, extra = "tracked tree is dirty (stream.py)", ()
    elif kind == "untracked":
        p = clone / "extra_source.py"
        p.write_text("X = 1\n")
        restore = p.unlink
        want, extra = "tracked tree is dirty (?? extra_source.py)", ()
    else:  # broken *.py symlink: refused even with --allow-dirty
        p = clone / "gone.py"
        os.symlink(clone / "does_not_exist.py", p)
        restore = p.unlink
        want, extra = "cannot be hashed (!! broken-symlink gone.py)", ("--allow-dirty",)
    try:
        assert _dirty_check(clone) != "[]"
        res = _run_all(clone, tmp_path, extra)
    finally:
        restore()
    for name, (rc, so, se, out) in res.items():
        assert rc == 1, (name, rc, se[-2000:])
        assert want in se, (name, se[-2000:])
        assert so == "", (name, so)               # refused before printing anything
        assert not out.exists(), name             # and before writing anything
    assert _dirty_check(clone) == "[]"
