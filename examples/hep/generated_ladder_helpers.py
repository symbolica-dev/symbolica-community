"""Topology selection and scalar notation shared by the generated ladder notebooks."""

from math import prod

from symbolica import E, Graph, Replacement, S
from symbolica.community import hepkit as hep
from symbolica.community import tensor as sp


def ladder_filter(outer_quark=False):
    """Select a ring of eight vertices with three uncrossed gluon rungs.

    This graph is only an isomorphism target. Feynman diagrams, weights, rules,
    directed fermion flow and momentum routing all come from the generator.
    """
    target = Graph()
    for _ in range(8):
        target.add_node(0)
    # Generator labels external legs by their signed, one-based global index.
    target.add_node(-1)
    target.add_node(2)
    for i in range(8):
        target.add_edge(i, (i + 1) % 8, data=5 if outer_quark else 21)
    for i, j in ((1, 7), (2, 6), (3, 5), (0, 8), (4, 9)):
        target.add_edge(i, j, data=21)
    target = target.canonize()[0]

    def accept(graph, completed):
        # An incomplete branch can still grow into the requested ladder.
        if completed < len(graph):
            return True
        if len(graph) != 10 or graph.num_edges() != 13:
            return False
        # Ignore orientation only in the callback snapshot. The generator keeps
        # the physical quark arrows and their signs in the returned diagram.
        for edge in range(graph.num_edges()):
            graph.set_directed(edge, False)
        return graph.canonize()[0] == target

    return accept


def project_and_split(numerator, dimension):
    """Apply g_mu_nu delta_ab/8 and separate color from spacetime factors."""
    projector = sp.TensorExpression(E("1/8"))
    for representation in (
        sp.Representation.mink(dimension),
        sp.Representation.coad(8),
    ):
        slots = [
            slot.dual()
            for slot in numerator.structure.slots()
            if slot.representation == representation
        ]
        assert len(slots) == 2
        projector *= sp.TensorExpression.g(representation)(*slots)
    projected = projector * numerator
    assert not projected.structure.slots()
    color_reps = (
        sp.Representation.cof(3),
        sp.Representation.cof(3).dual(),
        sp.Representation.coad(8),
    )
    color, spacetime = [], []
    for factor in projected.to_expression():
        slots = sp.TensorExpression(factor).structure.slots()
        if slots and all(slot.representation in color_reps for slot in slots):
            color.append(factor)
        else:
            spacetime.append(factor)
    color_tensor = sp.TensorExpression(prod(color))
    spacetime_tensor = sp.TensorExpression(prod(spacetime))
    assert (
        color_tensor.to_expression() * spacetime_tensor.to_expression()
    ) == projected.to_expression()
    return projector, color_tensor, spacetime_tensor


def reduce_color(tensor):
    """Reduce the color tensor with exact SU(3), T_R=1/2 algebra."""
    tensor = tensor.simplify_algebra(
        gamma=False,
        color=True,
        color_substitute_cof_dimension_invariants=True,
    )
    assert tensor.reduction_status == sp.ReductionStatus.Complete
    result = tensor.to_expression().expand()
    # A remaining color trace is not a scalar numerical color coefficient.
    assert result.get_all_symbols() == []
    return result


def dot_coordinates(expression, dimension, namespace):
    """Name the 15 independent products of four loop vectors and external p."""
    loop_momentum = hep.Kinematics.loop_momentum()
    external_momentum = hep.Kinematics.external_momentum()
    vectors = [loop_momentum(i) for i in range(4)]
    vectors.append(external_momentum(0))
    kinematics = hep.Kinematics(dimension, momenta=vectors)
    labels = ["k0", "k1", "k2", "k3", "p"]
    replacements, rows, symbols = [], [], []
    for i in range(5):
        for j in range(i, 5):
            symbol = S(f"{namespace}::s{i}{j}")
            product = kinematics.scalar_product(vectors[i], vectors[j])
            replacements.append(Replacement(product, symbol))
            symbols.append(symbol)
            rows.append(
                {"symbol": f"s{i}{j}", "dot product": f"{labels[i]} · {labels[j]}"}
            )
    polynomial = expression.replace_multiple(replacements).expand(via_poly=True)
    return polynomial, symbols, rows
