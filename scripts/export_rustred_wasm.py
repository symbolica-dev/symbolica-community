#!/usr/bin/env python3
"""Export an explicit RustRed notebook with a validated Pyodide wheel and inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "three_loop_reduction": (
        "three_loop_reduction_support.py", "rustred_campaign_support.py",
        "data/rustred_three_loop/k6.dot", "data/rustred_three_loop/k6.toml",
    ),
    "four_loop_reduction": (
        "rustred_campaign_support.py",
        "data/rustred_four_loop/h.dot", "data/rustred_four_loop/x.dot",
        "data/rustred_four_loop/bmw.dot", "data/rustred_four_loop/fg.dot",
    ),
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def validated_wheel(wheel):
    report_path = wheel.parent / "rustred-wasm-validation.json"
    if not report_path.is_file():
        raise ValueError("Run .github/scripts/test_pyodide.mjs on this wheel before exporting")
    report = json.loads(report_path.read_text())
    wheel_digest = digest(wheel)
    if (report.get("schema") != "rustred-wasm-runtime-v1"
            or report.get("scope") not in {"rustred-only", "full-community"}
            or report.get("wheel") != wheel.name
            or report.get("wheel_sha256") != wheel_digest
            or report.get("execution_mode") != "synchronous"
            or any(report.get(field) is not True for field in (
                "generated_and_certified_k6", "graph_ibp", "exact_homogeneity",
                "pinch_and_numerator",
            ))):
        raise ValueError("Successful Pyodide RustRed validation must match this exact wheel")
    return wheel_digest, report


def export(wheel, output, *, notebook="three_loop_reduction"):
    wheel, output = Path(wheel).resolve(), Path(output).resolve()
    if notebook not in INPUTS:
        raise ValueError("Unknown RustRed notebook")
    if not wheel.name.endswith("-pyemscripten_2026_0_wasm32.whl"):
        raise ValueError("Supply a PyEmscripten wheel, not a native Python wheel")
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    with ZipFile(wheel) as archive:
        if "symbolica/community/hepkit/rustred/__init__.py" not in archive.namelist():
            raise ValueError("The wheel must include HEPKit's RustRed namespace")
    wheel_digest, validation = validated_wheel(wheel)
    source = ROOT / "examples/hep" / f"{notebook}.py"
    subprocess.run([
        sys.executable, "-m", "marimo", "export", "html-wasm",
        str(source), "--show-code", "-o", str(output),
    ], check=True)
    files = {}
    for relative in INPUTS[notebook]:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "examples/hep" / relative, destination)
        files[relative] = digest(destination)
    shutil.copyfile(wheel, output / wheel.name)
    (output / "rustred-wasm-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest = {
        "schema": "rustred-browser-assets-v1", "notebook": notebook,
        "notebook_sha256": digest(source), "wheel": wheel.name,
        "wheel_sha256": wheel_digest, "files": files,
        "validation_scope": "K6 generation, closure and exact reduction; no four-loop completion claim",
    }
    (output / "rustred-assets.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {output}; serve this directory over HTTP. Generation requires an explicit click.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--notebook", choices=tuple(INPUTS), default="three_loop_reduction")
    args = parser.parse_args()
    export(args.wheel, args.output, notebook=args.notebook)
