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
_, notebook = app.run()
inputs = notebook["gghh_inputs"]
source = inputs.catalogue(progress=None)
# The unfiltered catalogue intentionally includes exact-zero diagrams. Select
# a nonzero box for this changing-runtime-point regression.
box = next(diagram for diagram in source.diagrams
           if diagram.loop_count == 1 and len(diagram.internal_edges) == 4
           and all(abs(edge.particle.pdg_code) == 6 for edge in diagram.internal_edges))
prepared = inputs.prepare(source=source, selected=box.id)
first = prepared.runtime_point({"sqrt_s": 300, "higgs_mass": 125, "cos_theta": 0.8})
second = prepared.runtime_point({"sqrt_s": 400, "higgs_mass": 125, "cos_theta": 0.4})
assert first
assert set(first) == set(second) == {symbol for _, _, symbol in prepared.gram_symbols}
assert all(math.isfinite(value) for point in (first, second) for value in point.values())
assert first != second, "Changing the physical point must update the Gram matrix"

# A clean launch leaves the calculation idle. Exercise the explicit build and
# timer actions too: these resolve helpers after their defining cells have run.
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


def test_gghh_notebook_dataflow_and_native_types(tmp_path):
    pytest.importorskip("marimo", minversion="0.24.0")
    root = Path(__file__).parents[1]
    checked = subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict",
         str(root / "examples/hep/gghh_complete.py")],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

    ty = Path(sys.executable).with_name("ty")
    if not ty.is_file():
        pytest.skip("ty is required for the native API type check")
    # Marimo owns cell scoping. Concatenating cell bodies bypasses that compiler
    # and exposes notebook-private dynamic UI wrappers as a synthetic module.
    # Check the installed native objects that users actually call instead.
    source = """
from typing_extensions import assert_type
from symbolica.community.hepkit import sector_decomposition as sd

def generation_types(integral: sd.Integral) -> None:
    session = integral.generation_session(
        mode="numerical_dual", subtraction="taylor",
        compilation_settings=sd.CompilationSettings(backend="eager"))
    assert_type(session, sd.GenerationSession)
    assert_type(session.mode, str)
    assert_type(session.subtraction, str)
    assert_type(session.generated, sd.GeneratedIntegral | None)
    assert_type(session.kernels, sd.Kernels | None)
    snapshot = session.snapshot()
    assert_type(snapshot, sd.GenerationSnapshot)
    assert_type(snapshot.formula_preparation, sd.FormulaPreparationSnapshot | None)
    assert_type(snapshot.timings.formula_preparation_seconds, float | None)
"""
    path = tmp_path / "gghh_native_types.py"
    path.write_text(source)
    checked = subprocess.run(
        [str(ty), "check", "--python", sys.executable, str(path)],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
