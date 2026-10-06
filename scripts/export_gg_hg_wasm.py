#!/usr/bin/env python3
"""Export the Higgs-jet notebook with a transport-enabled Pyodide wheel and inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "gg_hg_boundaries.py",
    "data/gg_hg/amplitude-validation.json",
    "data/gg_hg/boundaries-manifest.json",
    "data/gg_hg/boundaries.json.gz",
)


def validated_wheel(wheel):
    """Require a successful Pyodide smoke run bound to these executable bytes."""
    with wheel.open("rb") as stream:
        hasher = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
        digest = hasher.hexdigest()
    report_path = wheel.parent / "loop-transport-validation.json"
    if not report_path.is_file():
        raise ValueError("Run .github/scripts/test_pyodide.mjs on this wheel before exporting")
    report = json.loads(report_path.read_text())
    if (report.get("schema") != "supplied-loop-transport-runtime-v1"
            or report.get("wheel") != wheel.name
            or report.get("wheel_sha256") != digest
            or report.get("supplied_loop_transport") is not True
            or report.get("higgs_standard_model") is not True):
        raise ValueError("Successful Pyodide transport and Standard Model validation must match this exact wheel")
    return digest, report


def export(wheel, output):
    wheel, output = Path(wheel).resolve(), Path(output).resolve()
    if not wheel.name.endswith("-pyemscripten_2026_0_wasm32.whl"):
        raise ValueError("Supply a PyEmscripten wheel, not a native Python wheel")
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    with ZipFile(wheel) as archive:
        if "symbolica/community/hep/integration/__init__.py" not in archive.namelist():
            raise ValueError("The wheel must include the numerical integration namespace")
    wheel_digest, validation = validated_wheel(wheel)
    subprocess.run([
        sys.executable, "-m", "marimo", "export", "html-wasm",
        str(ROOT / "examples/hep/gg_hg.py"), "--show-code", "-o", str(output),
    ], check=True)
    files = {}
    for relative in INPUTS:
        payload = (ROOT / "examples/hep" / relative).read_bytes()
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        files[relative] = hashlib.sha256(payload).hexdigest()
    shutil.copyfile(wheel, output / wheel.name)
    (output / "loop-transport-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = {
        "schema": "higgs-jet-browser-assets-v1",
        "notebook_sha256": hashlib.sha256((ROOT / "examples/hep/gg_hg.py").read_bytes()).hexdigest(),
        "wheel": wheel.name,
        "wheel_sha256": wheel_digest,
        "files": files,
    }
    (output / "gg-hg-assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {output}; serve this directory over HTTP.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.wheel, args.output)
