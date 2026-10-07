"""WASM notebook packaging tests; synthetic wheels are not runtime evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("rustred_wasm_export", ROOT / "scripts/export_rustred_wasm.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def wheel_and_report(tmp_path):
    wheel = tmp_path / "symbolica-3.0.1-cp314-abi3-pyemscripten_2026_0_wasm32.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("symbolica/community/hepkit/rustred/__init__.py", "")
    report = {
        "schema": "rustred-wasm-runtime-v1", "wheel": wheel.name,
        "scope": "rustred-only",
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "execution_mode": "synchronous", "generated_and_certified_k6": True,
        "graph_ibp": True, "exact_homogeneity": True, "pinch_and_numerator": True,
    }
    return wheel, report


def test_namespace_alone_does_not_admit_browser_export(tmp_path, monkeypatch):
    wheel, _ = wheel_and_report(tmp_path)
    monkeypatch.setattr(EXPORT.subprocess, "run", lambda *a, **kw: pytest.fail("Unvalidated export started"))
    with pytest.raises(ValueError, match="test_pyodide.mjs"):
        EXPORT.export(wheel, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field,value", [
    ("wheel_sha256", "different"), ("generated_and_certified_k6", False),
    ("graph_ibp", None), ("execution_mode", "background-coordinator"),
    ("scope", None), ("scope", "unverified"),
])
def test_validation_must_cover_this_wheel_and_mathematics(tmp_path, monkeypatch, field, value):
    wheel, report = wheel_and_report(tmp_path)
    report[field] = value
    (tmp_path / "rustred-wasm-validation.json").write_text(json.dumps(report))
    monkeypatch.setattr(EXPORT.subprocess, "run", lambda *a, **kw: pytest.fail("Unvalidated export started"))
    with pytest.raises(ValueError, match="exact wheel"):
        EXPORT.export(wheel, tmp_path / "output")


@pytest.mark.parametrize("notebook", tuple(EXPORT.INPUTS))
def test_export_includes_checked_inputs_but_no_precomputed_artifacts(tmp_path, monkeypatch, notebook):
    wheel, report = wheel_and_report(tmp_path)
    (tmp_path / "rustred-wasm-validation.json").write_text(json.dumps(report))
    output = tmp_path / "output"

    def export_template(command, *, check):
        assert check and command[-1] == str(output)
        assert "--show-code" in command and "--execute" not in command
        output.mkdir()

    monkeypatch.setattr(EXPORT.subprocess, "run", export_template)
    EXPORT.export(wheel, output, notebook=notebook)
    manifest = json.loads((output / "rustred-assets.json").read_text())
    assert manifest["notebook"] == notebook
    assert set(manifest["files"]) == set(EXPORT.INPUTS[notebook])
    assert manifest["wheel_sha256"] == hashlib.sha256((output / wheel.name).read_bytes()).hexdigest()
    assert all(Path(name).suffix in {".py", ".dot", ".toml"} for name in manifest["files"])
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
