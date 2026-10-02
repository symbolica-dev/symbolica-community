"""Check notebook graph weights against one-loop QCD statistics factors."""

from importlib import import_module
from pathlib import Path

import pytest
from symbolica import E
from symbolica.community.hep import Model, SnailFilterOptions


@pytest.fixture(
    scope="module", params=["examples.hep_showcase", "examples.hep_showcase_uv"]
)
def diagram_weight(request):
    pytest.importorskip("marimo", minversion="0.24.0")
    app = import_module(request.param).app
    cell = next(
        cell
        for cell in app._cell_manager.cells()
        if cell is not None and "diagram_weight" in cell.defs
    )
    _, definitions = cell.run()
    return definitions["diagram_weight"]


@pytest.fixture(scope="module", params=["1", "7/3"])
def bubbles(request):
    model = Model.standard_model()
    prefactor = E(request.param)
    diagrams = model.process(
        ["g"], ["g"], particle_veto=["c", "t", "s", "u", "d"]
    ).generate_diagrams(
        loops=1,
        coupling_orders={"QCD": 2, "QED": 0},
        zero_snails=SnailFilterOptions(),
        numerator_prefactor=prefactor,
        threads=1,
        progress=None,
    )
    diagrams = diagrams.diagrams
    by_particles = {
        tuple(sorted(edge.particle_name for edge in diagram.internal_edges)): diagram
        for diagram in diagrams
    }
    assert len(diagrams) == len(by_particles) == 3
    return prefactor, by_particles


@pytest.mark.parametrize(
    ("particles", "factor"),
    [("ghG", "-1"), ("g", "1/2"), ("b", "-1")],
)
def test_qcd_loop_weight(diagram_weight, bubbles, particles, factor):
    prefactor, diagrams = bubbles
    # Each closed Grassmann loop contributes -1; the gluon bubble has
    # a symmetry factor of 1/2. The requested prefactor applies exactly once.
    assert diagram_weight(diagrams[(particles, particles)]) == E(factor) * prefactor
