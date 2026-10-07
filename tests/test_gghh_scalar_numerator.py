"""Triple-gluon diagrams must reach scalar parametrization in the complete notebook."""

from pathlib import Path
import subprocess
import sys

import pytest


def test_triple_gluon_notebook_preparation_reaches_nonempty_geometry():
    pytest.importorskip("marimo", minversion="0.24.0")
    root = Path(__file__).parents[1]
    script = """
import runpy
import sys

from symbolica import E

app = runpy.run_path(sys.argv[1], run_name="notebook_check")["app"]
cells = [cell for _, cell in app._cell_manager.valid_cells()]
_, base = next(cell for cell in cells if "ShowcaseInput" in cell.defs).run()
_, inputs = next(cell for cell in cells if "gghh_prepare" in cell.defs).run(
    ShowcaseInput=base["ShowcaseInput"])
_, science = next(cell for cell in cells if "science_generation" in cell.defs).run()
catalogue = inputs["gghh_catalogue"](progress=None)
# The exact reported graph has two triple-gluon vertices. Minimal contraction
# left a closed tensor network: is_scalar passed, but parametrization failed.
identity = "6586fc41a2a00087ef7be79f59b61224"
diagram = catalogue.selected(identity)
assert diagram.loop_count == 2
assert sum(catalogue.model.vertex_rule(vertex.interaction).particles == ["g", "g", "g"]
           for vertex in diagram.vertices) == 2
prepared = inputs["gghh_prepare"](selected=identity, source=catalogue)
assert prepared.simplified_numerator != E("0")
legend = prepared.gram_legend()
assert {row["Runtime symbol"] for row in legend} == {
    str(symbol.formatted(show_namespaces=True))
    for symbol in prepared.integral_arguments()["runtime_parameters"]
}
assert all("leg " in row["Left vector"] and "leg " in row["Right vector"] for row in legend)
assert any("eps1" in row["Left vector"] and "eps2" in row["Right vector"] for row in legend)
session = science["science_generation"](prepared, {"max_order": 0})
snapshot = session.step(max_units=1)
assert session.failed is None and snapshot.stage == "parametrization"
assert snapshot.timings.parametrization_seconds > 0
for _ in range(3):
    snapshot = session.step(max_units=1)
    if snapshot.stage == "geometry":
        break
assert session.failed is None
assert snapshot.stage == "geometry" and snapshot.completed > 0
print("GGHH_SCALAR_REGRESSION", len(catalogue.diagrams), identity, snapshot.completed)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root / "examples/hep/gghh_complete.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
