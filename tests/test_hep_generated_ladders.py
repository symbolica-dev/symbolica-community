"""Check the generated ladders' topology, graph weights and quark color algebra."""

from collections import Counter
from importlib import import_module
from types import SimpleNamespace

import pytest
from symbolica import E, S
from symbolica.community.graph import Graph, node, edge, source, sink, Orientation
from symbolica.community import hepkit as hep
from symbolica.community import tensor as sp

from examples.hep.generated_ladder_helpers import (
    project_and_split,
    reduce_color,
)


@pytest.fixture(scope="module")
def notebook_helpers():
    pytest.importorskip("marimo", minversion="0.24.0")
    app = import_module("examples.hep.three_gluon_rung_ladder").app
    helpers = {}
    for name in ("ladder_filter",):
        cell = next(cell for cell in app._cell_manager.cells() if name in cell.defs)
        _, definitions = cell.run()
        helpers[name] = definitions[name]
    generation = next(
        cell for cell in app._cell_manager.cells() if "ladders" in cell.defs
    )
    _, definitions = generation.run(ladder_filter=helpers["ladder_filter"])
    helpers.update(definitions)
    return helpers


@pytest.fixture(params=[False, True], ids=["gluon", "outer_quark"])
def ladder(request, notebook_helpers):
    quark = request.param
    app = import_module("examples.hep.three_gluon_rung_ladder").app
    select = next(cell for cell in app._cell_manager.cells() if "diagram" in cell.defs)
    _, values = select.run(
        ladder_choice=SimpleNamespace(value="Fermionic" if quark else "Gluonic"),
        ladders=notebook_helpers["ladders"],
    )
    return quark, values["diagram"]


def test_generated_particles_weights_and_color(ladder):
    quark, diagram = ladder
    assert diagram.loop_count == 4
    counts = Counter(edge.particle_name for edge in diagram.internal_edges)
    assert counts == (Counter({"b": 8, "g": 3}) if quark else Counter({"g": 11}))
    assert diagram.overall_factor_expression(evaluate=True) == E(
        "-1" if quark else "1/2"
    )
    _, color, _ = project_and_split(diagram.numerator_expression(), 4)
    coefficient = reduce_color(color)
    assert coefficient == E("-1/54" if quark else "81")
    if quark:
        # Independent explicit Fierz contractions, with no native color identities.
        # Sum_a T^a_ij T^a_kl = (delta_il delta_kj - delta_ij delta_kl/3)/2.
        a, i, j, k, l = S(*(f"ladder_test::{n}_" for n in ("a", "i", "j", "k", "l")))
        t = sp.TensorExpression.color_t(8, 3)
        fundamental = sp.Representation.cof(3)
        delta = sp.TensorExpression.g(fundamental, fundamental.dual())
        lhs = (t(a, i, j) * t(a, k, l)).to_expression()
        rhs = (
            delta(i, l) * delta(k, j) / 2 - delta(i, j) * delta(k, l) / 6
        ).to_expression()
        color = color.contract(collect_chains=False, collect_traces=False)
        for _ in range(4):
            color = (
                sp.TensorExpression(color.to_expression().replace(lhs, rhs))
                .expand()
                .contract(collect_chains=False, collect_traces=False)
            )
        assert color.to_expression().expand() == coefficient


@pytest.mark.parametrize("quark", [False, True])
def test_filter_is_label_independent_and_rejects_crossed_rungs(quark, notebook_helpers):
    # A nontrivial relabeling must not affect the topology match.
    order = [7, 4, 9, 2, 8, 1, 5, 0, 6, 3]
    graph = Graph()
    for _ in order:
        graph.add_node(node(data=0))
    graph.node(order[8]).data = E("-1")
    graph.node(order[9]).data = E("2")
    for i in range(8):
        graph.add_edge(
            edge(
                source(order[i]),
                second=sink(order[(i + 1) % 8]),
                orientation=Orientation.Default if quark else Orientation.Undirected,
                data=5 if quark else 21,
            )
        )
    for i, j in ((0, 8), (4, 9), (1, 7), (2, 6), (3, 5)):
        graph.add_edge(
            edge(
                source(order[i]),
                second=sink(order[j]),
                data=21,
                orientation=Orientation.Undirected,
            )
        )
    select_ladder = notebook_helpers["ladder_filter"]
    accept = select_ladder(quark)
    assert accept(graph, len(graph))
    # Cross two rungs while preserving every vertex degree and particle count.
    crossed = Graph()
    for vertex in graph.nodes():
        crossed.add_node(node(data=vertex.data))
    for connection in graph.edges():
        start, end = connection.source.node.index, connection.sink.node.index
        if {start, end} == {order[1], order[7]}:
            start, end = order[1], order[6]
        elif {start, end} == {order[2], order[6]}:
            start, end = order[2], order[7]
        crossed.add_edge(
            edge(
                source(start),
                second=sink(end),
                orientation=connection.orientation,
                data=connection.data,
            )
        )
    assert not accept(crossed, len(crossed))
    # Pruning an unfinished snapshot on a full-graph mismatch would lose ladders.
    partial = Graph()
    partial.add_node(node(data=-1))
    assert accept(partial, 0)


def test_notebook_projects_and_reduces_selected_ladder_without_splitting(ladder):
    quark, diagram = ladder
    app = import_module("examples.hep.three_gluon_rung_ladder").app
    cells = list(app._cell_manager.cells())
    assert not any(
        name in cell.defs
        for cell in cells
        for name in ("project_and_split", "reduce_color")
    )
    model = hep.Model.standard_model()
    build = next(cell for cell in cells if "numerator" in cell.defs)
    _, values = build.run(diagram=diagram, model=model)
    numerator = values["numerator"]
    dimension = values["D"]
    assert not numerator.to_expression().contains(model.particle("b").mass)
    project = next(cell for cell in cells if "projector" in cell.defs)
    _, values = project.run(D=dimension, numerator=numerator)
    projector = values["projector"]
    assert (projector * numerator).rank == 0

    # The previous representation-filtered construction provides an independent
    # projection oracle; the production notebook keeps the tensor together.
    old_projector, color, spacetime = project_and_split(numerator, dimension)
    assert projector.to_expression() == old_projector.to_expression()
    reduce = next(cell for cell in cells if "reduced" in cell.defs)
    _, values = reduce.run(numerator=numerator, projector=projector)
    reduced = values["reduced"]
    assert reduced.reduction_status == sp.ReductionStatus.Complete
    assert reduced.rank == 0
    color_factor = E("-1/54" if quark else "81")
    expected = (
        color_factor
        * spacetime.simplify_algebra(
            gamma=True, color=False, contract="dots"
        ).to_expression()
    )
    assert (reduced.to_expression() - expected).expand(via_poly=True) == 0
    assert reduce_color(color) == color_factor
    assert not reduced.to_expression().contains(sp.Representation.coad(8).casimir())
