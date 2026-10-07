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
_, notebook = app.run()
inputs, science = notebook["gghh_inputs"], notebook["science"]
catalogue = inputs.catalogue(progress=None)
# The exact reported graph has two triple-gluon vertices. Minimal contraction
# left a closed tensor network: is_scalar passed, but parametrization failed.
identity = "6586fc41a2a00087ef7be79f59b61224"
diagram = catalogue.selected(identity)
assert diagram.loop_count == 2
assert sum(catalogue.model.vertex_rule(vertex.interaction).particles == ["g", "g", "g"]
           for vertex in diagram.vertices) == 2
prepared = inputs.prepare(selected=identity, source=catalogue)
assert prepared.simplified_numerator != E("0")
legend = prepared.gram_legend()
assert {row["Runtime symbol"] for row in legend} == {
    str(symbol.formatted(show_namespaces=True))
    for symbol in prepared.integral_arguments()["runtime_parameters"]
}
assert all("leg " in row["Left vector"] and "leg " in row["Right vector"] for row in legend)
assert any("eps1" in row["Left vector"] and "eps2" in row["Right vector"] for row in legend)
session = science.generation(prepared, {"max_order": 0})
assert session.mode == "symbolic" and session.subtraction == "taylor"
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


def test_generation_observer_can_render_real_run_state():
    pytest.importorskip("marimo", minversion="0.24.0")
    root = Path(__file__).parents[1]
    script = """
import runpy
import sys

app = runpy.run_path(sys.argv[1], run_name="notebook_check")["app"]
_, notebook = app.run()
inputs, science = notebook["gghh_inputs"], notebook["science"]
catalogue = inputs.catalogue(progress=None)
# Keep the callback active through real sector and evaluator construction.
box = next(diagram for diagram in catalogue.diagrams
           if diagram.loop_count == 1 and len(diagram.internal_edges) == 4
           and all(abs(edge.particle.pdg_code) == 6 for edge in diagram.internal_edges))
state = notebook["RunState"]()
rendered = []

def display():
    # This executes synchronously inside the actual native step's observer.
    # Reading generation_session.complete here re-borrows the mutably borrowed
    # PyO3 owner and used to pause generation with "Already mutably borrowed".
    view = notebook["generation_views"].generation_view(notebook["mo"], state)
    rendered.append((state.events[-1].stage if state.events else "preparing", view.text))

state.start_generation(
    lambda: inputs.prepare(source=catalogue, selected=box.id),
    {"max_order": 0}, science.generation, display=display)
assert state.error is None, state.error
for _ in range(100):
    if not state.generation_active:
        break
    state.generation_last_display = -float("inf")
    state.advance_generation()
    assert state.error is None, state.error
assert state.phase == "ready", (state.phase, state.message)
assert state.generation_session.complete
assert state.generated is not None and state.kernels is not None
assert state.kernels.sector_count > 0
assert state.generation_session.mode == "symbolic"
assert state.generation_session.subtraction == "taylor"
stages = {stage for stage, _ in rendered}
assert "parametrization" in stages and "complete" in stages, stages
assert all("Generation failed" not in html for _, html in rendered)
assert all("not a zero integral" not in html for _, html in rendered)
print("GGHH_CALLBACK_REGRESSION", len(rendered), sorted(stages))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root / "examples/hep/gghh_complete.py")],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
