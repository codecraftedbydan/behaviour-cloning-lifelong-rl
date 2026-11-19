"""Scrub machine-specific paths from the counted-cell bundles and from release copies of text files.

The only identifying string in the counted artifacts is the absolute run directory on the training machine,
which is replaced with the placeholder <BASE>. Chunk files (*.npz) are never modified. For every file whose
bytes change, the old and new SHA-256 are recorded, and JSON files get a parsed comparison showing that every
difference is a replaced path string and that every number is unchanged.

Usage: python tools/scrub_release.py <dir of original *_review_*.tgz> <dir holding the two sweep dirs> <out dir> <run dir to replace> <dir holding pilot_40/ and null_200/>
"""
import gzip, hashlib, io, json, os, sys, tarfile

OLD = None  # the absolute run directory to replace; passed on the command line, never stored in the repo
NEW = "<BASE>"
sha = lambda b: hashlib.sha256(b).hexdigest()


def json_diff(a, b, path="$"):
    """Yield (path, old, new) for every leaf that differs; raise if a non-string leaf differs."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), path
        for k in a:
            yield from json_diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            yield from json_diff(x, y, f"{path}[{i}]")
    elif a != b:
        assert isinstance(a, str) and isinstance(b, str) and a.replace(OLD, NEW) == b, (path, a, b)
        yield path, a, b


def scrub_bytes(name, data, record):
    if name.endswith(".npz") or OLD.encode() not in data:
        return data
    new = data.replace(OLD.encode(), NEW.encode())
    entry = {"file": name, "old_sha256": sha(data), "new_sha256": sha(new),
             "replaced_occurrences": data.count(OLD.encode())}
    if name.endswith(".json"):
        diffs = list(json_diff(json.loads(data), json.loads(new)))
        entry["json_check"] = {"changed_leaves": len(diffs), "all_changed_leaves_are_path_strings": True,
                               "numbers_unchanged": True, "changed_paths": [d[0] for d in diffs]}
    record.append(entry)
    return new


def pack(files):
    """Deterministic .tgz: sorted members, zero mtimes, fixed modes, gzip header mtime 0."""
    tbuf = io.BytesIO()
    with tarfile.open(fileobj=tbuf, mode="w", format=tarfile.PAX_FORMAT) as t:
        for n in sorted(files):
            ti = tarfile.TarInfo(f"./{n}"); ti.size = len(files[n]); ti.mtime = 0; ti.mode = 0o644
            t.addfile(ti, io.BytesIO(files[n]))
    gbuf = io.BytesIO()
    with gzip.GzipFile(fileobj=gbuf, mode="wb", mtime=0, filename="") as g:
        g.write(tbuf.getvalue())
    return gbuf.getvalue()


def sweeps(src, out):
    """Pack the per-trial directories of the LR and kickstart sweeps (no queue logs, no inner archives)."""
    record, files = [], {}
    for top in ("sweeps_20261110", "kickstart_20261112"):
        for d in sorted(os.listdir(os.path.join(src, top))):
            for r, _, fs in os.walk(os.path.join(src, top, d)):
                for f in fs:
                    rel = os.path.relpath(os.path.join(r, f), src)
                    files[rel] = scrub_bytes(rel, open(os.path.join(r, f), "rb").read(), record)
    data = pack(files)
    open(os.path.join(out, "lr_sweeps_scrubbed.tgz"), "wb").write(data)
    json.dump({"asset": "lr_sweeps_scrubbed.tgz", "sha256": sha(data), "files": len(files), "changed_files": record},
              open(os.path.join(out, "SCRUB_MANIFEST_lr_sweeps.json"), "w"), indent=1)
    with open(os.path.join(out, "SHA256SUMS"), "a") as f:
        f.write(f"{sha(data)}  lr_sweeps_scrubbed.tgz\n")


def pilots(src, out):
    """Pack the pilot_40 and null_200 run arrays (pilot.npz only; no launch logs or plots). Nothing to scrub."""
    files = {f"{r}/pilot.npz": open(os.path.join(src, r, "pilot.npz"), "rb").read() for r in ("pilot_40", "null_200")}
    data = pack(files)
    open(os.path.join(out, "pilot_runs.tgz"), "wb").write(data)
    with open(os.path.join(out, "SHA256SUMS"), "a") as f:
        f.write(f"{sha(data)}  pilot_runs.tgz\n")


def main(src, out):
    os.makedirs(out, exist_ok=True)
    record, bundles = [], []
    for fn in sorted(f for f in os.listdir(src) if f.endswith(".tgz") and "_review_" in f):
        raw = open(os.path.join(src, fn), "rb").read()
        files = {}
        with tarfile.open(fileobj=io.BytesIO(raw)) as t:
            for m in t.getmembers():
                if m.isfile():
                    files[m.name.lstrip("./")] = t.extractfile(m).read()
        cell = fn.split("_review_")[0]
        for n in list(files):
            files[n] = scrub_bytes(f"{cell}/{n}", files[n], record)
        # regenerate each sha256sums.txt from the (possibly scrubbed) files it lists; chunk lines stay identical
        for n in [n for n in files if n.endswith("sha256sums.txt")]:
            d = os.path.dirname(n)
            lines = []
            for line in files[n].decode().splitlines():
                h, f = line.split(None, 1)
                key = f"{d}/{f}" if d else f
                lines.append(f"{sha(files[key]) if (f != 'sha256sums.txt' and key in files) else h}  {f}")
            new = ("\n".join(lines) + "\n").encode()
            if new != files[n]:
                record.append({"file": f"{cell}/{n}", "old_sha256": sha(files[n]), "new_sha256": sha(new),
                               "reason": "regenerated after scrubbing the files it lists; chunk lines unchanged"})
            files[n] = new
        data = pack(files)
        name = fn.replace(".tgz", "_scrubbed.tgz")
        open(os.path.join(out, name), "wb").write(data)
        bundles.append({"cell": cell, "original": fn, "original_sha256": sha(raw),
                        "scrubbed": name, "scrubbed_sha256": sha(data)})
    json.dump({"placeholder": {"replaced": "absolute run directory on the training machine", "with": NEW},
               "bundles": bundles, "changed_files": record},
              open(os.path.join(out, "SCRUB_MANIFEST.json"), "w"), indent=1)
    with open(os.path.join(out, "SHA256SUMS"), "w") as f:
        for b in bundles:
            f.write(f"{b['scrubbed_sha256']}  {b['scrubbed']}\n")


if __name__ == "__main__":
    OLD = sys.argv[4]
    main(sys.argv[1], sys.argv[3])
    sweeps(sys.argv[2], sys.argv[3])
    pilots(sys.argv[5], sys.argv[3])
