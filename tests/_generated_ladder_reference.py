"""Authored, oriented color reference for the generated gluon ladder tests."""

import json

from symbolica.community import hepkit as hep
from symbolica.community import tensor as sp

from examples.hep.generated_ladder_helpers import project_and_split


def authored_gluon_color_reference(diagram, model):
    """Keep the model's f(1,2,3) order and every resulting scalar phase."""
    graph = json.loads(diagram.to_json())
    model_data = json.loads(model.to_json())
    rule_id, rule = next(
        (i, rule) for i, rule in enumerate(model_data["vertex_rules"])
        if rule["name"] == "V_36"
    )
    assert rule == {
        "name": "V_36", "particles": ["g", "g", "g"],
        "color_structures": ["f(1,2,3)"], "lorentz_structures": ["VVV1"],
        "couplings": [["GC_10"]],
    }
    assert len(graph["vertices"]) == 8
    assert all(vertex["interaction"] == rule_id for vertex in graph["vertices"])
    endpoints = {
        (half["edge"], half["endpoint"]): hedge
        for hedge, half in enumerate(graph["half_edge_order"])
    }
    legs = {vertex: {} for vertex in range(8)}
    for edge, (ends, data) in enumerate(graph["edges"]):
        for endpoint, vertex in ends.items():
            if vertex is not None:
                slot = data[endpoint + "_slot"]
                assert slot not in legs[vertex]
                legs[vertex][slot] = endpoints[edge, endpoint]

    adjoint = sp.Representation.coad(8)
    head = hep.Symbols.half_edge()
    structure_constant = sp.TensorExpression.color_f(8)
    vertices = {vertex.id: vertex for vertex in diagram.vertices}
    reference = sp.TensorExpression(1)
    slots = {}
    for vertex in range(8):
        assert set(legs[vertex]) == {0, 1, 2}
        hedges = [legs[vertex][slot] for slot in range(3)]
        factor = structure_constant(*(head(hedge, 1) for hedge in hedges), intern="indices")
        # .axes is model-leg order; .structure.slots() is canonical order.
        ordered = factor.axes
        actual = {
            slot for slot in vertices[vertex].numerator_expression().axes
            if slot.representation == adjoint
        }
        assert len(ordered) == len(actual) == 3 and set(ordered) == actual
        assert not set(hedges).intersection(slots)
        slots.update(zip(hedges, ordered))
        reference *= factor

    assert len(slots) == len(set(slots.values())) == 24
    external = []
    internal_count = 0
    for edge, (ends, data) in enumerate(graph["edges"]):
        if data["external"] is None:
            reference *= sp.TensorExpression.g(
                slots[endpoints[edge, "source"]], slots[endpoints[edge, "target"]]
            )
            internal_count += 1
        else:
            endpoint = "source" if ends["source"] is not None else "target"
            external.append(slots[endpoints[edge, endpoint]])
    assert internal_count == 11 and len(external) == 2
    reference *= sp.TensorExpression.g(*external)
    assert reference.rank == 0
    return reference, slots


def oriented_gluon_split(numerator, dimension, reference):
    """Transport a proven symbolic phase; do not infer it from a color value."""
    projector, color, spacetime = project_and_split(numerator, dimension)
    authored, raw = reference.to_expression(), color.to_expression()
    if authored == raw:
        phase = 1
    else:
        assert authored == -raw
        phase = -1
    oriented_spacetime = phase * spacetime
    whole = (projector * numerator).to_expression()
    assert raw * spacetime.to_expression() == whole
    assert authored * oriented_spacetime.to_expression() == whole
    return projector, reference, oriented_spacetime
