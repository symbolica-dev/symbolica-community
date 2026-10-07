"""Exercise the physical-point bindings after marimo compiles the notebook."""

from pathlib import Path
import subprocess
import sys

import pytest


def test_gghh_runtime_point_from_compiled_cells():
    pytest.importorskip("marimo", minversion="0.24.0")
    root = Path(__file__).parents[1]
    script = """
import math
import runpy
import sys

app = runpy.run_path(sys.argv[1], run_name="notebook_check")["app"]
cells = [cell for _, cell in app._cell_manager.valid_cells()]
base_cell = next(cell for cell in cells if "ShowcaseInput" in cell.defs)
_, base = base_cell.run()
input_cell = next(cell for cell in cells if "gghh_prepare" in cell.defs)
_, definitions = input_cell.run(ShowcaseInput=base["ShowcaseInput"])
source = definitions["gghh_catalogue"](progress=None)
prepared = definitions["gghh_prepare"](source=source)
first = prepared.runtime_point({"sqrt_s": 300, "higgs_mass": 125, "cos_theta": 0.8})
second = prepared.runtime_point({"sqrt_s": 400, "higgs_mass": 125, "cos_theta": 0.4})
assert first
assert set(first) == set(second) == {symbol for _, _, symbol in prepared.gram_symbols}
assert all(math.isfinite(value) for point in (first, second) for value in point.values())
assert first != second, "Changing the physical point must update the Gram matrix"

# A clean launch leaves the calculation idle. Exercise the explicit build and
# timer actions too: these resolve helpers after their defining cells have run.
_, notebook = app.run()
study, mo = notebook["study"], notebook["mo"]
built = study.build(1, mo)
assert built.default_diagram.id in study.choices().values()
study.last_publication = -float("inf")
revision = study.dispatch(study.seen.copy(), "tick", built.default_diagram.id, {}, mo)
assert revision is not None
assert study.run.error is None
assert study.run.prepared is None
assert not study.run.work_active
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root / "examples/hep/gghh_complete.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gghh_notebook_types(tmp_path):
    pytest.importorskip("marimo", minversion="0.24.0")
    ty = Path(sys.executable).with_name("ty")
    if not ty.is_file():
        pytest.skip("ty is required for the notebook type check")
    import runpy

    root = Path(__file__).parents[1]
    app = runpy.run_path(str(root / "examples/hep/gghh_complete.py"))["app"]
    # Like marimo's editor document, place the cell bodies in one module so
    # the checker can follow dependencies between cells.
    source = "\n\n".join(cell._cell.code for _, cell in app._cell_manager.valid_cells())
    source += "\nfrom typing_extensions import assert_type\n"
    source += "assert_type(study, Study)\nassert_type(study.run, RunState)\n"
    source += "assert_type(study.catalogue, GGHHCatalogue | None)\n"
    path = tmp_path / "gghh_types.py"
    path.write_text(source)
    result = subprocess.run(
        [str(ty), "check", "--python", sys.executable, str(path)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
