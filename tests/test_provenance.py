"""Provenance fixes from the review of 0291582: untracked sources count as dirty
and are hashed (fix 1); run_condition runs the clean-tree guard itself (fix 2)."""
import os
import subprocess

import pytest

import lifelong
import ppo
import provenance
import stream as S

TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=8, hidden_size=8,
                         eval_episodes=1))
SCFG = S.StreamConfig(num_new_tasks=1, revisit_every=0, block_updates=1)


def _write(root, rel, text="x = 1\n"):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo (never the real tree) with one committed source file."""
    root = str(tmp_path / "r")
    os.makedirs(root)
    g = ["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.check_call(g + ["init", "-q"])
    _write(root, "a.py")
    _write(root, ".gitignore", ".venv/\ncheckpoints/\n")
    subprocess.check_call(g + ["add", "a.py", ".gitignore"])
    subprocess.check_call(g + ["commit", "-q", "-m", "init"])
    monkeypatch.setattr(provenance, "ROOT", root)
    return root


def test_clean_repo_and_non_source_untracked(repo):
    _write(repo, "launch.out", "log\n")          # untracked non-source: not dirty
    _write(repo, "notes.md", "n\n")
    assert provenance.dirty_files() == []
    assert set(provenance.source_hashes()) == {"a.py"}


@pytest.mark.parametrize("rel", ["b.py", "pkg/c.py", "scripts/run.sh", "scripts/x.py"])
def test_untracked_source_is_dirty_and_hashed(repo, rel):
    before = provenance.tree_hash(provenance.source_hashes())
    _write(repo, rel)
    assert provenance.dirty_files() == [f"?? {rel}"]
    h = provenance.source_hashes()
    assert set(h) == {"a.py", rel}
    assert provenance.tree_hash(h) != before
    with pytest.raises(SystemExit, match="tracked tree is dirty"):
        provenance.check_clean([990], False, force=True)
    r = provenance.check_clean([990], True, force=True)
    assert r["git_dirty"] is True and rel in r["source_hashes"]


@pytest.mark.parametrize("rel", ["results/r.py", "results/sub/s.py", "checkpoints/c.py",
                                 ".venv/lib/v.py", "__pycache__/m.py"])
def test_excluded_dirs_never_count(repo, rel):
    _write(repo, rel)
    assert provenance.dirty_files() == []
    assert set(provenance.source_hashes()) == {"a.py"}
    assert provenance.check_clean([990], False, force=True)["git_dirty"] is False


def test_gitignored_source_outside_excluded_dirs_counts(repo):
    _write(repo, ".gitignore", ".venv/\ncheckpoints/\nscratch_*.py\n")
    subprocess.check_call(["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t",
                           "commit", "-qam", "ignore"])
    _write(repo, "scratch_1.py")
    assert provenance.dirty_files() == ["?? scratch_1.py"]


def test_run_condition_refuses_dirty_tree_itself(monkeypatch):
    """Scripts / notebooks calling run_condition directly hit the guard (not only the CLI)."""
    monkeypatch.setattr(provenance, "under_pytest", lambda: False)
    monkeypatch.setattr(provenance, "dirty_files", lambda: ["?? stray.py"])

    def boom(*a, **k):
        raise AssertionError("training reached")
    monkeypatch.setattr(lifelong, "make_stream_train", boom)
    monkeypatch.setattr(S, "make_stream", boom)
    with pytest.raises(SystemExit, match="tracked tree is dirty"):
        lifelong.run_condition(TINY, SCFG, [990], verbose=False)
    # seed guard first: reserved seeds never get as far as the dirty-tree message
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    with pytest.raises(SystemExit, match="reserved|official"):
        lifelong.run_condition(TINY, SCFG, [899], verbose=False, allow_dirty=True)


def test_run_condition_records_provenance(monkeypatch):
    monkeypatch.setattr(provenance, "under_pytest", lambda: False)
    monkeypatch.setattr(provenance, "dirty_files", lambda: ["README.md"])
    res = lifelong.run_condition(TINY, SCFG, [990], fresh=False, verbose=False, allow_dirty=True)
    assert bool(res["git_dirty"]) and bool(res["allow_dirty"])
    assert len(str(res["source_tree_hash"])) == 64 and "lifelong.py" in str(res["source_hashes"])
    monkeypatch.undo()
    res = lifelong.run_condition(TINY, SCFG, [990], fresh=False, verbose=False)  # under pytest: records only
    for k in ("git_commit_full", "git_dirty", "allow_dirty", "source_tree_hash", "source_hashes"):
        assert k in res


# ---- review notes on 6c10b27: unhashable trees refuse cleanly, never FileNotFoundError
def _refused(match="cannot be hashed"):
    for allow in (False, True):
        with pytest.raises(SystemExit, match=match):
            provenance.check_clean([990], allow, force=True)


def test_broken_py_symlink_refused_cleanly(repo):
    os.symlink(os.path.join(repo, "nowhere.py"), os.path.join(repo, "b.py"))
    h = provenance.source_hashes()  # no FileNotFoundError
    assert h["b.py"] == "broken-symlink"
    assert "!! broken-symlink b.py" in provenance.dirty_files()
    _refused()
    assert provenance.record()["git_dirty"] is True  # recording-only path never raises


def test_deleted_tracked_file_refused_cleanly(repo):
    os.remove(os.path.join(repo, "a.py"))
    assert provenance.source_hashes() == {"a.py": "missing"}
    d = provenance.dirty_files()
    assert "a.py" in d and "!! missing a.py" in d
    _refused()


def test_symlinked_dir_of_py_files_detected_and_refused(repo, tmp_path):
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "m.py").write_text("y = 2\n")
    os.symlink(str(ext), os.path.join(repo, "pkg"))
    assert provenance.symlinked_dirs() == ["pkg"]
    assert "!! symlinked-dir pkg" in provenance.dirty_files()
    _refused()
    # a committed symlinked dir is still refused (contents invisible to git and the hashes)
    g = ["git", "-C", repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.check_call(g + ["add", "pkg"])
    subprocess.check_call(g + ["commit", "-q", "-m", "link"])
    assert provenance.dirty_files() == ["!! symlinked-dir pkg"]
    _refused()


def test_symlinked_dirs_in_excluded_paths_ignored(repo, tmp_path):
    os.makedirs(os.path.join(repo, ".venv"))
    os.symlink(str(tmp_path), os.path.join(repo, ".venv", "lib64"))
    os.makedirs(os.path.join(repo, "results"))
    os.symlink(str(tmp_path), os.path.join(repo, "results", "link"))
    assert provenance.symlinked_dirs() == []
    assert provenance.dirty_files() == []
