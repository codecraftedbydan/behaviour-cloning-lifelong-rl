"""Numerical equivalence with PureJaxRL (GAE, loss terms, gradients) on identical
params + rollout batch. Needs the PureJaxRL clone and its venv (distrax); skipped
otherwise. The real work is in scripts/numerical_equivalence.py."""
import os
import subprocess

import pytest

REF = os.environ.get("PUREJAXRL_DIR", "../purejaxrl-ref")
PY = os.path.join(REF, ".venv", "bin", "python")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(not os.path.exists(PY), reason="PureJaxRL clone/venv not available")
def test_numerical_equivalence_with_purejaxrl(tmp_path):
    out = tmp_path / "eq.json"
    r = subprocess.run([PY, "-W", "ignore", os.path.join(ROOT, "scripts", "numerical_equivalence.py"),
                        "--out", str(out), "--tol", "1e-5"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "ALL PASS" in r.stdout and "detected" in r.stdout
    import json
    assert json.load(open(out))["matmul_precision"] == "default"  # not forced any more


@pytest.mark.skipif(not os.path.exists(PY), reason="PureJaxRL clone/venv not available")
def test_numerical_equivalence_tolerance_argument():
    """--tol exists with default 1e-5 (unchanged), and a too-tight tol makes it fail."""
    r = subprocess.run([PY, "-W", "ignore", os.path.join(ROOT, "scripts", "numerical_equivalence.py"), "--help"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "--tol" in r.stdout and "1e-5" in r.stdout
    assert "--matmul-precision" in r.stdout and "default,highest" in r.stdout


@pytest.mark.skipif(not os.path.exists(PY), reason="PureJaxRL clone/venv not available")
def test_numerical_equivalence_records_tol_and_device(tmp_path):
    import json
    out = tmp_path / "eq.json"
    r = subprocess.run([PY, "-W", "ignore", os.path.join(ROOT, "scripts", "numerical_equivalence.py"),
                        "--out", str(out), "--tol", "0", "--matmul-precision", "highest"],
                       capture_output=True, text=True, timeout=900)
    d = json.load(open(out))
    assert d["tol"] == 0 and d["device_platform"] and d["jax_version"] and d["jaxlib_version"]
    assert d["matmul_precision"] == "highest"  # explicit setting is applied and recorded
    assert r.returncode == 1 and "SOME FAIL" in r.stdout  # tol 0 is unreachable in float32
