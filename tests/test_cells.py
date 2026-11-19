"""Step 6: cell runner (cells.py): BC checkpoint seed / source-hash checks, --no-fresh reset
cells sharing the centre scratch cell's fresh runs, fixed recorded chunk layouts."""
import dataclasses
import json
import os
import shutil

import numpy as np
import pytest

import arms
import bc
import cells
import lifelong
import ppo
import provenance
import seeds as SD
import stream as S

TINY = ppo.Config(**dict(ppo.PRESETS["lifelong_stream"], num_envs=4, num_steps=16, eval_episodes=1))
SCFG = dataclasses.replace(S.STAGE1_STREAM, num_new_tasks=2, revisit_every=0, block_updates=2,
                           head_probe_every=0)
NONDET = set(cells.TIMING_KEYS) | {"chunk_" + k for k in cells.TIMING_KEYS} | {"cell_manifest_sha256"}


def _run(name, seeds, chunk, d, **kw):
    return cells.run_cell(name, seeds, chunk, str(d), base_config=TINY, scfg=SCFG, verbose=False, **kw)


def _same(a, b, skip=()):
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    for k in a:
        if k in skip or k in NONDET:
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype.kind in "OUS":
            assert str(x) == str(y), k
        else:
            assert x.shape == y.shape and np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), k


def _save_bc(d, cell, s, **meta):
    c = cells.CELLS[cell]  # the BC network = the cell's plain (scratch) network
    pc = arms.plain_config(cells.cell_config(cells.Cell("scratch", c.size, c.layer_norm), TINY))
    m = dict(config=dataclasses.asdict(pc), seed=s, stream_config=S.config_record(SCFG), git_dirty=False,
             source_tree_hash=provenance.record()["source_tree_hash"])
    m.update(meta)
    bc.save_checkpoint(str(d / bc.checkpoint_name(pc, s)), lifelong.scratch_init_params(pc, s), m)


# ----------------------------------------------------------------------------- layout
def test_chunk_layout_and_cells():
    assert cells.chunk_layout([990, 991, 992], 2) == [[990, 991], [992]]
    assert cells.chunk_layout([990, 991, 992], 3) == [[990, 991, 992]]
    assert cells.chunk_layout([992, 990], 5) == [[992, 990]]  # order as given
    for bad in (([990], 0), ([990, 990], 1), ([], 1)):
        with pytest.raises(ValueError):
            cells.chunk_layout(*bad)
    assert cells.CELLS["reset_heads_1M"].fresh_from == cells.CENTRE_SCRATCH == "scratch_1M"
    assert cells.CELLS["reset_heads_adam_vn_1M"].reset_arm == "heads_adam_vn"
    assert {c.fresh_from for n, c in cells.CELLS.items() if not n.startswith("reset_")} == {None}
    a = dict(cell="x", seeds=[1, 2, 3], chunk_layout=[[1, 2], [3]])
    cells.check_same_layout(a, dict(a))
    with pytest.raises(ValueError, match="chunk layouts differ"):
        cells.check_same_layout(a, dict(a, cell="y", chunk_layout=[[1, 2, 3]]))


@pytest.mark.parametrize("field", ["git_commit_full", "source_tree_hash", "stream_config",
                                   "matmul_precision", "device_platform", "fresh_seeds", "chunk_size"])
def test_check_same_layout_refuses_identity_mismatch(field):
    """review fix 1: paired cells (e.g. pretrained vs scratch for H1') must share every
    PAIR_IDENTITY field (FRESH_IDENTITY minus the arm-specific configs)."""
    assert set(cells.PAIR_IDENTITY) == set(cells.FRESH_IDENTITY) - {"config", "fresh_config"}
    assert field in cells.PAIR_IDENTITY
    a = dict(cell="pretrained_1M", seeds=[990, 991], fresh_seeds=[1990, 1991], chunk_size=1,
             chunk_layout=[[990], [991]], git_commit_full="a" * 40, source_tree_hash="b" * 64,
             stream_config=json.dumps(S.config_record(SCFG)), matmul_precision="default",
             device_platform="cpu", config="{arm: pretrained}", fresh_config="{}")
    b = dict(a, cell="scratch_1M", config="{arm: scratch}", fresh_config="{x}")  # configs may differ
    cells.check_same_layout(a, b)
    other = {"stream_config": json.dumps(S.config_record(dataclasses.replace(SCFG, block_updates=3))),
             "fresh_seeds": [1990, 1992], "chunk_size": 2, "matmul_precision": "highest",
             "device_platform": "gpu"}.get(field, "c" * 8)
    with pytest.raises(ValueError, match=f"cannot be paired: different {field}"):
        cells.check_same_layout(a, dict(b, **{field: other}))
    with pytest.raises(ValueError, match=f"different {field}"):
        cells.check_same_layout(a, {k: v for k, v in b.items() if k != field})  # missing field


# ----------------------------------------------------------------------------- BC checks
def test_bc_checkpoint_seed_and_hash_refused(tmp_path):
    bcd = tmp_path / "bc"
    _save_bc(bcd, "pretrained_0.1M", 990)
    kw = dict(bc_dir=str(bcd), warmup_updates=1)
    # wrong seed inside the file named for 991
    _save_bc(bcd, "pretrained_0.1M", 991, seed=990)
    with pytest.raises(ValueError, match=r"checkpoint seed 990 != 991"):
        _run("pretrained_0.1M", [990, 991], 1, tmp_path / "a", **kw)
    # source hash of other code
    _save_bc(bcd, "pretrained_0.1M", 992, source_tree_hash="0" * 64, git_commit_full="abc")
    with pytest.raises(ValueError, match=r"BC source_tree_hash 0{64} != expected .*commit abc"):
        _run("pretrained_0.1M", [992], 1, tmp_path / "b", **kw)
    # no provenance at all
    _save_bc(bcd, "pretrained_0.1M", 993, source_tree_hash=None)
    with pytest.raises(ValueError, match="no source_tree_hash"):
        _run("pretrained_0.1M", [993], 1, tmp_path / "c", **kw)
    for sub in "abc":  # refused before anything was written
        assert not (tmp_path / sub).exists()
    # every BC cell is checked (kickstart, lora, frozen, LN); scratch cells refuse a bc_dir
    for cell in ("lora_1M", "frozen_pretrained_1M", "kickstart_1M", "ln_pretrained_1M"):
        _save_bc(bcd, cell, 994, source_tree_hash="f" * 64)
        with pytest.raises(ValueError, match="BC source_tree_hash"):
            _run(cell, [994], 1, tmp_path / "d", kickstart_coef=0.1 if cell == "kickstart_1M" else None, **kw)
    with pytest.raises(ValueError, match="scratch cell"):
        _run("scratch_0.1M", [990], 1, tmp_path / "e", bc_dir=str(bcd))
    with pytest.raises(ValueError, match="needs bc_dir"):
        _run("pretrained_0.1M", [990], 1, tmp_path / "e", warmup_updates=1)
    # an explicitly pinned hash is accepted and recorded as pinned
    man = _run("pretrained_0.1M", [992], 1, tmp_path / "p", bc_tree_hash="0" * 64, **kw)
    d = np.load(tmp_path / "p" / man["chunks"][0]["file"])
    assert str(d["bc_expected_tree_hash"]) == "0" * 64 and bool(d["bc_tree_hash_pinned"])
    assert json.loads(str(d["bc_checkpoint_sha256"])).keys() == {"992"}
    assert not bool(d["ranking_binding"]) and "pinned" in str(d["ranking_binding_reason"])
    assert man["bc_tree_hash_pinned"] is True and man["ranking_binding"] is False
    # and the unpinned default is this tree's hash
    man = _run("pretrained_0.1M", [990], 1, tmp_path / "q", **kw)
    d = np.load(tmp_path / "q" / man["chunks"][0]["file"])
    assert str(d["bc_expected_tree_hash"]) == man["source_tree_hash"] and not bool(d["bc_tree_hash_pinned"])
    assert bool(d["seed_init_params"]) and str(d["arm"]) == "pretrained"
    assert not bool(d["ranking_binding"]) and man["ranking_binding"] is False  # dev seeds: mechanics only


# ----------------------------------------------------------------------------- BC hash pin (decision 2 Nov)
def _no_work(*a, **k):
    raise AssertionError("work started")


def test_bc_hash_pin_refused_on_reserved_seeds(tmp_path, monkeypatch):
    """Reserved stream / fresh seeds + --bc-tree-hash refuse before the seed guard, BC loading,
    training or any output; mixed reserved/dev sets too. Only the refusal path runs."""
    monkeypatch.setattr(lifelong, "run_condition", _no_work)
    monkeypatch.setattr(arms, "load_bc_checkpoints", _no_work)
    monkeypatch.setattr(provenance, "check_clean", _no_work)
    bcd, pin = tmp_path / "bc", "0" * 64
    reserved = sorted(SD.RESERVED_SEEDS)
    assert {700, 724, 750, 759, 800, 824, 850, 859} <= set(reserved)
    cases = [[s] for s in reserved] + [[990, 700], [700, 990], [991, 855], [990, 991, 759]]
    for i, seeds in enumerate(cases):
        out = tmp_path / f"o{i}"
        with pytest.raises(ValueError, match=r"--bc-tree-hash refused: seeds \[.*\] are reserved"):
            _run("pretrained_1M", seeds, 1, out, bc_dir=str(bcd), bc_tree_hash=pin)
        with pytest.raises(SystemExit, match="--bc-tree-hash refused"):  # CLI, before any output
            cells.main(["run", "--cell", "kickstart_1M", "--seeds", ",".join(map(str, seeds)),
                        "--chunk-size", "1", "--out-dir", str(out), "--bc-dir", str(bcd),
                        "--bc-tree-hash", pin, "--kickstart-coef", "0.1", "--allow-official-seeds"])
        assert not out.exists()
    with pytest.raises(ValueError, match=r"seeds \[123\] are not dev seeds"):
        _run("pretrained_1M", [123], 1, tmp_path / "n", bc_dir=str(bcd), bc_tree_hash=pin)
    # the pin check does not change the unpinned path: reserved seeds still meet the seed guard
    with pytest.raises(SystemExit, match="refused under pytest"):
        _run("pretrained_1M", [700], 1, tmp_path / "g", bc_dir=str(bcd))
    assert not any((tmp_path / d).exists() for d in ("n", "g"))
    assert cells.cell_binding([990], False)[0] is False and cells.cell_binding([700], True)[0] is False
    assert cells.cell_binding(sorted(SD.STAGE1_STREAM_SEEDS), False)[0] is True


def test_bc_hash_pin_record_time_assert(tmp_path, monkeypatch):
    for rec, seeds in ((dict(bc_tree_hash_pinned=True), [700]), (dict(bc_tree_hash_pinned=True), [990, 850]),
                       (dict(bc_tree_hash_pinned=np.array(True), ranking_binding=np.array(True)), [990])):
        with pytest.raises(AssertionError, match="counted runs cannot pin"):
            cells.check_pin_record(rec, seeds)
    cells.check_pin_record(dict(bc_tree_hash_pinned=True, ranking_binding=False), [990])
    cells.check_pin_record(dict(bc_tree_hash_pinned=False, ranking_binding=True), [700])
    # a pinned dev run that would be recorded binding fails before training or output
    bcd = tmp_path / "bc"
    _save_bc(bcd, "pretrained_0.1M", 990, source_tree_hash="0" * 64)
    with monkeypatch.context() as m:
        m.setattr(cells, "cell_binding", lambda seeds, pinned: (True, "forced"))
        m.setattr(lifelong, "run_condition", _no_work)
        with pytest.raises(AssertionError, match="ranking_binding=True"):
            _run("pretrained_0.1M", [990], 1, tmp_path / "x", bc_dir=str(bcd), bc_tree_hash="0" * 64,
                 warmup_updates=1)
    assert not (tmp_path / "x").exists()
    # merge re-checks: a pinned cell whose manifest claims reserved seeds is refused
    man = _run("pretrained_0.1M", [990], 1, tmp_path / "p", bc_dir=str(bcd), bc_tree_hash="0" * 64,
               warmup_updates=1)
    assert man["bc_tree_hash_pinned"] is True and man["ranking_binding"] is False
    assert not bool(cells.merge_cell(str(tmp_path / "p"))["ranking_binding"])
    with open(tmp_path / "p" / "manifest.json") as f:
        js = json.load(f)
    with open(tmp_path / "p" / "manifest.json", "w") as f:
        json.dump(dict(js, seeds=[700]), f)
    with pytest.raises(AssertionError, match=r"reserved seeds \[700, 800\]"):
        cells.merge_cell(str(tmp_path / "p"))


# ----------------------------------------------------------------------------- reset cells
def test_reset_cells_share_centre_fresh_runs(tmp_path):
    seeds = [990, 991]
    centre = tmp_path / "centre"
    cm = _run("scratch_1M", seeds, 1, centre)
    assert cm["fresh"] and cm["status"] == "complete"
    with pytest.raises(ValueError, match="needs --fresh-from"):
        _run("reset_heads_1M", seeds, 1, tmp_path / "x")
    with pytest.raises(ValueError, match="only for the reset cells"):
        _run("scratch_1M", seeds, 1, tmp_path / "x", fresh_from=str(centre))
    with pytest.raises(ValueError, match="no cell manifest"):
        _run("reset_heads_1M", seeds, 1, tmp_path / "x", fresh_from=str(tmp_path / "missing"))

    def tampered(name, **edit):
        d = tmp_path / name
        shutil.copytree(centre, d)
        m = cells.read_manifest(d)
        m.update(edit)
        cells._write_manifest(str(d), m)
        return str(d)

    for edit, msg in ((dict(git_commit_full="0" * 40), "different git_commit_full"),
                      (dict(source_tree_hash="1" * 64), "different source_tree_hash"),
                      (dict(config=cm["config"].replace('"lr": ', '"lr": 9')), "different config"),
                      (dict(stream_config="{}"), "different stream_config"),
                      (dict(chunk_size=2, chunk_layout=[seeds]), "different chunk_size"),
                      (dict(matmul_precision="highest"), "different matmul_precision"),
                      (dict(status="running"), "not complete"),
                      (dict(cell="pretrained_1M"), "must come from cell 'scratch_1M'"),
                      (dict(fresh=False), "no fresh runs")):
        with pytest.raises(ValueError, match=msg):
            _run("reset_heads_1M", seeds, 1, tmp_path / "x", fresh_from=tampered(f"t{len(os.listdir(tmp_path))}", **edit))
    # different seeds / layout of the reset cell itself
    with pytest.raises(ValueError, match="different chunk_size|different seeds"):
        _run("reset_heads_1M", seeds, 2, tmp_path / "x", fresh_from=str(centre))
    with pytest.raises(ValueError, match="different seeds"):
        _run("reset_heads_1M", [990], 1, tmp_path / "x", fresh_from=str(centre))
    # a different real config (other stream geometry) is refused too
    with pytest.raises(ValueError, match="different stream_config"):
        cells.run_cell("reset_heads_1M", seeds, 1, str(tmp_path / "x"), base_config=TINY,
                       scfg=dataclasses.replace(SCFG, block_updates=3), fresh_from=str(centre), verbose=False)
    assert not (tmp_path / "x").exists()

    rh = tmp_path / "rh"
    rm = _run("reset_heads_1M", seeds, 1, rh, fresh_from=str(centre))
    assert not rm["fresh"] and rm["fresh_from_dir"] == str(centre)
    for i, cs in enumerate(cm["chunk_layout"]):
        a, b = np.load(rh / f"chunk_{i:03d}.npz"), np.load(centre / f"chunk_{i:03d}.npz")
        fk = [k for k in b.files if k.startswith("fresh/")]
        assert fk and set(fk) == {k for k in a.files if k.startswith("fresh/")}
        for k in fk:
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        assert str(a["fresh_config"]) == str(b["fresh_config"]) and str(a["reset_arm"]) == "heads"
        assert str(a["fresh_shared_from"]) == str(centre)
        # the shared fresh runs are exactly what the reset cell would have run itself
        own = lifelong.run_condition(cells.cell_config(cells.CELLS["reset_heads_1M"], TINY), SCFG, cs,
                                     verbose=False, reset_arm="heads")
        for k in fk:
            np.testing.assert_array_equal(own[k], a[k], err_msg=k)
        np.testing.assert_array_equal(own["stream/metrics/loss"], a["stream/metrics/loss"])
    merged = cells.merge_cell(str(rh))
    assert merged["fresh/metrics/loss"].shape[0] == 2 and str(merged["reset_arm"]) == "heads"
    # a changed centre chunk file is caught
    with open(centre / "chunk_001.npz", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _run("reset_heads_adam_vn_1M", seeds, 1, tmp_path / "y", fresh_from=str(centre))


# ----------------------------------------------------------------------------- chunking
def test_same_chunk_layout_is_bit_identical_and_merge_checks(tmp_path):
    seeds = [990, 991, 992]
    m1 = _run("scratch_0.1M", seeds, 2, tmp_path / "r1")
    m2 = _run("scratch_0.1M", seeds, 2, tmp_path / "r2")
    assert m1["chunk_layout"] == [[990, 991], [992]] == m2["chunk_layout"]
    a, b = cells.merge_cell(str(tmp_path / "r1")), cells.merge_cell(str(tmp_path / "r2"))
    _same(a, b)  # same layout -> bit-identical (CPU, dev seeds)
    assert a["seeds"].tolist() == seeds and a["stream/metrics/loss"].shape[0] == 3
    assert json.loads(str(a["chunk_layout"])) == [[990, 991], [992]] and int(a["chunk_size"]) == 2
    assert a["chunk_wall_seconds"].shape == (2,)
    # each chunk is exactly a plain run_condition call on that chunk's seeds
    cfg = cells.cell_config(cells.CELLS["scratch_0.1M"], TINY)
    for i, cs in enumerate(m1["chunk_layout"]):
        direct = lifelong.run_condition(cfg, SCFG, cs, verbose=False)
        ch = np.load(tmp_path / "r1" / f"chunk_{i:03d}.npz")
        for k in direct:
            if k.startswith(("stream/metrics", "fresh/metrics", "stream/eval")):
                np.testing.assert_array_equal(direct[k], ch[k], err_msg=k)
    # unchunked cell: same keys and shapes as the merged chunked one
    _run("scratch_0.1M", seeds, 3, tmp_path / "u")
    u = cells.merge_cell(str(tmp_path / "u"))
    assert set(u) == set(a)
    for k in a:
        assert np.shape(a[k]) == np.shape(u[k]) or k.startswith("chunk_"), k
    with pytest.raises(ValueError, match="chunk layouts differ"):
        cells.check_same_layout(cells.read_manifest(tmp_path / "r1"), cells.read_manifest(tmp_path / "u"))
    # refusals
    with pytest.raises(ValueError, match="refusing to overwrite"):
        _run("scratch_0.1M", seeds, 2, tmp_path / "r1")
    bad = tmp_path / "bad"
    shutil.copytree(tmp_path / "r1", bad)
    m = cells.read_manifest(bad)
    cells._write_manifest(str(bad), dict(m, chunk_layout=[[990], [991, 992]]))
    with pytest.raises(ValueError, match="chunk_layout"):
        cells.merge_cell(str(bad))
    cells._write_manifest(str(bad), dict(m, status="running"))
    with pytest.raises(ValueError, match="not complete"):
        cells.merge_cell(str(bad))
    cells._write_manifest(str(bad), dict(m, chunks=m["chunks"][:1]))
    with pytest.raises(ValueError, match="not complete"):
        cells.merge_cell(str(bad))
    cells._write_manifest(str(bad), m)
    shutil.copy(tmp_path / "r2" / "chunk_001.npz", bad / "chunk_001.npz")  # a chunk from another run
    with pytest.raises(ValueError, match="sha256 mismatch"):
        cells.merge_cell(str(bad))
    os.remove(bad / "chunk_001.npz")
    with pytest.raises(ValueError, match="chunk file missing"):
        cells.merge_cell(str(bad))
    # CLI merge
    out = tmp_path / "merged.npz"
    cells.main(["merge", str(tmp_path / "r1"), "--out", str(out)])
    _same(dict(np.load(out)), a, skip=("cell_manifest_sha256",))
    with pytest.raises(SystemExit, match="merge: "):
        cells.main(["merge", str(bad), "--out", str(out)])


def test_kickstart_cell_refuses_zero_window_before_training(tmp_path):
    bcd = tmp_path / "bc"
    _save_bc(bcd, "kickstart_1M", 900)
    sc = dataclasses.replace(SCFG, block_updates=4)
    with pytest.raises(lifelong.KickstartWindowError):
        cells.run_cell("kickstart_1M", [900], 1, str(tmp_path / "k"), base_config=TINY, scfg=sc,
                       bc_dir=str(bcd), warmup_updates=4, kickstart_coef=0.1, verbose=False)
    assert not (tmp_path / "k").exists()
