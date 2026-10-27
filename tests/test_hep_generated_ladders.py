"""Check the generated ladders' topology, graph weights and color algebra."""

from collections import Counter
from importlib import import_module
from itertools import combinations, permutations
from types import SimpleNamespace

import pytest
from symbolica import E, Graph, S
from symbolica.community import hepkit as hep
from symbolica.community import tensor as sp

from examples.hep.generated_ladder_helpers import (
    project_and_split,
    reduce_color,
)


def explicit_gluon_color(tensor):
    """Contract the actual port ordering with exact Gell-Mann SU(3) f values."""
    import numpy as np

    assert tensor.is_scalar
    # Store 2*f as A+B*sqrt(3). The table follows [lambda_a, lambda_b]
    # = 2*i*f_abc*lambda_c, with tr(lambda_a*lambda_b) = 2*delta_ab.
    rational_f = np.zeros((8, 8, 8), dtype=np.int64)
    radical_f = np.zeros_like(rational_f)
    entries = [
        ((1, 2, 3), 2, 0),
        ((1, 4, 7), 1, 0),
        ((1, 5, 6), -1, 0),
        ((2, 4, 6), 1, 0),
        ((2, 5, 7), 1, 0),
        ((3, 4, 5), 1, 0),
        ((3, 6, 7), -1, 0),
        ((4, 5, 8), 0, 1),
        ((6, 7, 8), 0, 1),
    ]
    for labels, rational, radical in entries:
        for order in permutations(range(3)):
            sign = (-1) ** sum(
                order[i] > order[j] for i in range(3) for j in range(i + 1, 3)
            )
            index = tuple(labels[i] - 1 for i in order)
            rational_f[index], radical_f[index] = sign * rational, sign * radical

    # Each adjoint metric identifies two index labels without a sign.
    parent = {}

    def find(index):
        parent.setdefault(index, index)
        if parent[index] != index:
            parent[index] = find(parent[index])
        return parent[index]

    f_head = sp.TensorName.color_f().to_expression().get_name()
    g_head = sp.TensorName.g().to_expression().get_name()
    adjoint = sp.Representation.coad(8)
    factors, metrics = [], 0
    for factor in tensor.to_expression():
        head, indices = factor.get_name(), tuple(factor)
        assert head in (f_head, g_head)
        assert all(index == adjoint(index[1]).to_expression() for index in indices)
        if head == g_head:
            assert len(indices) == 2
            parent[find(indices[0])] = find(indices[1])
            metrics += 1
        else:
            factors.append(indices)
    assert len(factors) == 8 and metrics == 12
    labels = [tuple(find(index) for index in indices) for indices in factors]
    assert all(len(set(indices)) == 3 for indices in labels)
    assert set(Counter(index for indices in labels for index in indices).values()) == {2}
    tensors = [(indices, rational_f, radical_f) for indices in labels]
    while len(tensors) > 1:
        i, j = min(
            combinations(range(len(tensors)), 2),
            key=lambda pair: len(set(tensors[pair[0]][0]) ^ set(tensors[pair[1]][0])),
        )
        left, a, b = tensors[i]
        right, c, d = tensors[j]
        shared = set(left) & set(right)
        axes = (
            [left.index(index) for index in shared],
            [right.index(index) for index in shared],
        )
        remaining = tuple(index for index in left + right if index not in shared)
        assert len(remaining) <= 4
        rational = np.tensordot(a, c, axes) + 3 * np.tensordot(b, d, axes)
        radical = np.tensordot(a, d, axes) + np.tensordot(b, c, axes)
        tensors.pop(j)
        tensors.pop(i)
        tensors.append((remaining, rational, radical))
    indices, rational, radical = tensors[0]
    assert not indices and int(radical) == 0
    # Eight factors and twelve summed indices bound every intermediate by
    # 2**8 * 8**12 = 2**44, safely within the exact int64 arithmetic above.
    scale = 2 ** len(factors)
    assert int(rational) % scale == 0
    return int(rational) // scale


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
    if not quark:
        # Antisymmetric port canonicalization can move a minus sign from color
        # into spacetime. Check the actual signed contraction independently.
        explicit_coefficient = explicit_gluon_color(color)
        assert explicit_coefficient**2 == 81**2
        assert coefficient == E(str(explicit_coefficient))
    else:
        assert coefficient == E("-1/54")
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
        graph.add_node(0)
    graph.set_node_data(order[8], E("-1"))
    graph.set_node_data(order[9], E("2"))
    for i in range(8):
        graph.add_edge(
            order[i], order[(i + 1) % 8], directed=quark, data=5 if quark else 21
        )
    for i, j in ((0, 8), (4, 9), (1, 7), (2, 6), (3, 5)):
        graph.add_edge(order[i], order[j], data=21)
    select_ladder = notebook_helpers["ladder_filter"]
    accept = select_ladder(quark)
    assert accept(graph, len(graph))
    # Cross two rungs while preserving every vertex degree and particle count.
    crossed = Graph()
    for _, data in graph.nodes():
        crossed.add_node(data)
    for source, target, directed, data in graph.edges():
        if {source, target} == {order[1], order[7]}:
            source, target = order[1], order[6]
        elif {source, target} == {order[2], order[6]}:
            source, target = order[2], order[7]
        crossed.add_edge(source, target, directed=directed, data=data)
    assert not accept(crossed, len(crossed))
    # Pruning an unfinished snapshot on a full-graph mismatch would lose ladders.
    partial = Graph()
    partial.add_node(-1)
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
    dimension = S("direct_ladder_test::D")
    build = next(cell for cell in cells if "numerator" in cell.defs)
    _, values = build.run(
        D=dimension, diagram=diagram, model=model, TensorExpression=sp.TensorExpression
    )
    numerator = values["numerator"]
    assert not numerator.to_expression().contains(model.particle("b").mass)
    project = next(cell for cell in cells if "projector" in cell.defs)
    _, values = project.run(
        D=dimension,
        numerator=numerator,
        Representation=sp.Representation,
        TensorExpression=sp.TensorExpression,
    )
    projector = values["projector"]
    assert (projector * numerator).rank == 0

    # The previous representation-filtered construction provides an independent
    # projection oracle; the production notebook keeps the tensor together.
    old_projector, color, spacetime = project_and_split(numerator, dimension)
    assert projector.to_expression() == old_projector.to_expression()
    reduce = next(cell for cell in cells if "reduced" in cell.defs)
    _, values = reduce.run(numerator=numerator, projector=projector)
    reduced = values["reduced"]
    dot_numerator = values["dot_numerator"]
    assert reduced.reduction_status == sp.ReductionStatus.Complete
    assert reduced.rank == 0
    color_factor = E("-1/54") if quark else E(str(explicit_gluon_color(color)))
    expected = (
        color_factor
        * spacetime.simplify_algebra(
            gamma=True, color=False, contract="dots"
        ).to_expression()
    )
    assert (reduced.to_expression() - expected).expand(via_poly=True) == 0
    assert reduce_color(color) == color_factor
    assert not reduced.to_expression().contains(sp.Representation.coad(8).casimir())

    coordinates = next(cell for cell in cells if "dot_coordinates" in cell.defs)
    _, helpers = coordinates.run()
    polynomial_cell = next(cell for cell in cells if "polynomial" in cell.defs)
    _, result = polynomial_cell.run(
        D=dimension,
        S=S,
        diagram=diagram,
        dot_coordinates=helpers["dot_coordinates"],
        dot_numerator=dot_numerator,
        graph_weight=diagram.overall_factor_expression(evaluate=True),
        model=model,
        Symbols=hep.Symbols,
    )
    weight = E("-1" if quark else "1/2")
    assert result["weighted_numerator"] == weight * result["polynomial"]
    assert not result["denominators"].contains(model.particle("b").mass)
