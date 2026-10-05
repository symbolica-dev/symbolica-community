#!/usr/bin/env python3
"""Export the notebook with explicit, hash-bound browser assets and wheel.

This packages an already built Pyodide wheel. It does not build dependencies,
execute the integral, publish the site, or assert browser feasibility.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True, help="Community cp314 pyemscripten_2026_0 wasm32 wheel with FastSecDec")
    parser.add_argument("--output", type=Path, required=True, help="New export directory, served over HTTP")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    if not wheel.is_file() or not wheel.name.endswith("-cp314-abi3-pyemscripten_2026_0_wasm32.whl"):
        parser.error("Expected a real cp314-abi3-pyemscripten_2026_0_wasm32.whl wheel")
    output = args.output.resolve()
    if output.exists():
        parser.error("Use a new output directory; existing exports are preserved")
    here = Path(__file__).resolve().parent
    notebook = here.parent / "fastsecdec_showcase.py"
    files = [here / "fastsecdec_inputs.py", here / "fastsecdec_views.py"]
    files += [here / "fixtures/fastsecdec" / name for name in ("README.md", "scalar.json", "triangle.dot", "box.dot", "box_rank2_numerator.dot", "sunset_2loop_numerator.dot")]
    if not all(path.is_file() for path in files):
        parser.error("Missing required local example assets")
    with tempfile.TemporaryDirectory(prefix="fastsecdec-showcase-export-") as temporary:
        stage = Path(temporary)
        shutil.copy2(notebook, stage / notebook.name)
        public = stage / "public/fastsecdec"
        public.mkdir(parents=True)
        shutil.copy2(wheel, public / wheel.name)
        archive = public / "example-assets.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(path, path.relative_to(here).as_posix())
        manifest = {
            "format": "fastsecdec-showcase-assets-v1",
            "marimo": "0.24.2", "python": "3.14", "pyodide": "314.0.0",
            "notebook_sha256": digest(notebook),
            "wheel": {"filename": wheel.name, "sha256": digest(wheel)},
            "assets": {"filename": archive.name, "sha256": digest(archive),
                       "files": {path.relative_to(here).as_posix(): digest(path) for path in files}},
            "validation": "Packaging only; browser runtime and scientific gates are separate.",
        }
        (public / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        subprocess.run([sys.executable, "-m", "marimo", "export", "html-wasm", str(stage / notebook.name), "--mode", "run", "--no-execute", "-o", str(output)], check=True)
        # Explicit copy also makes the mount layout independent of changes to
        # marimo's automatic public-directory discovery.
        shutil.copytree(stage / "public", output / "public", dirs_exist_ok=True)
    print(f"Exported {output}; serve this directory over HTTP.")
    print("No notebook cells were executed. Browser feasibility is not yet certified.")


if __name__ == "__main__":
    main()
