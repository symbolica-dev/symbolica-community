"""Export admission checks; synthetic wheel metadata is not runtime evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("higgs_jet_export", ROOT / "scripts/export_gg_hg_wasm.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def wheel_and_report(tmp_path):
    wheel = tmp_path / "symbolica-3.0.1-cp314-abi3-pyemscripten_2026_0_wasm32.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("symbolica/community/hep/integration/__init__.py", "")
    report = {
        "schema": "supplied-loop-transport-runtime-v1",
        "wheel": wheel.name,
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "supplied_loop_transport": True,
    }
    return wheel, report


def test_namespace_without_runtime_validation_does_not_start_export(tmp_path, monkeypatch):
    wheel, _ = wheel_and_report(tmp_path)
    monkeypatch.setattr(EXPORT.subprocess, "run", lambda *a, **kw: pytest.fail("Unvalidated export started"))
    with pytest.raises(ValueError, match="test_pyodide.mjs"):
        EXPORT.export(wheel, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field,value", [
    ("schema", "old"), ("wheel", "different.whl"),
    ("wheel_sha256", "incorrect"), ("supplied_loop_transport", False),
])
def test_runtime_validation_must_cover_this_wheel(tmp_path, monkeypatch, field, value):
    wheel, report = wheel_and_report(tmp_path)
    report[field] = value
    (tmp_path / "loop-transport-validation.json").write_text(json.dumps(report))
    monkeypatch.setattr(EXPORT.subprocess, "run", lambda *a, **kw: pytest.fail("Unvalidated export started"))
    with pytest.raises(ValueError, match="exact wheel"):
        EXPORT.export(wheel, tmp_path / "output")


def test_export_binds_inputs_and_wheel_to_manifest(tmp_path, monkeypatch):
    wheel, report = wheel_and_report(tmp_path)
    (tmp_path / "loop-transport-validation.json").write_text(json.dumps(report))
    output = tmp_path / "output"

    def export_template(command, *, check):
        assert check and command[-1] == str(output)
        assert "--show-code" in command
        output.mkdir()

    monkeypatch.setattr(EXPORT.subprocess, "run", export_template)
    EXPORT.export(wheel, output)
    manifest = json.loads((output / "gg-hg-assets.json").read_text())
    assert manifest["wheel_sha256"] == hashlib.sha256((output / wheel.name).read_bytes()).hexdigest()
    assert json.loads((output / "loop-transport-validation.json").read_text()) == report
    assert set(manifest["files"]) == set(EXPORT.INPUTS)
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
