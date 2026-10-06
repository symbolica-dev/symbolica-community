"""Installed owner graph identity, physical selection and rendering smoke."""

import argparse
import gc
from pathlib import Path
import weakref

import linnet
from symbolica.community import hepkit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("owner", type=Path)
    args = parser.parse_args()
    model = hepkit.Model(args.owner / "crates/feynkit-py/tests/fixtures/scalars_2p_3p.json")
    diagram = (
        model.process(["scalar_0"], ["scalar_0", "scalar_0"], vertex_allow=["V_3_SCALAR_000"])
        .generate_diagrams(loops=1, max_vertices=3, allow_self_loops=True)
        .diagrams[0]
    )
    graph = diagram.to_linnet()
    assert type(graph) is linnet.Graph
    assert graph is diagram.to_linnet()
    assert graph.n_nodes == len(diagram.vertices)
    assert graph.n_edges == len(diagram.edges)
    selection = graph.full_subgraph()
    assert type(selection) is linnet.Subgraph
    physical = diagram.subgraph(selection)
    assert physical.numerator_expression() == diagram.numerator_expression()
    assert physical.denominator_expression() == diagram.denominator_expression()
    assert physical.loop_count == diagram.loop_count
    restored = hepkit.FeynmanDiagram.from_json(model, diagram.to_json())
    assert restored.to_json() == diagram.to_json()
    try:
        diagram.subgraph(restored.to_linnet().full_subgraph())
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("foreign graph selections must be rejected")
    assert "<svg" in linnet.PreparedRender.from_sources(
        {"main.typ": b"Native graph renderer"}, config={}
    ).to_svg()

    class Payload:
        pass

    payload = Payload()
    payload.diagram = diagram
    graph.edge(0).data = payload
    reference = weakref.ref(payload)
    del payload, diagram, graph, selection, physical
    gc.collect()
    assert reference() is None
    print("Native Graph/Subgraph identity, physics, rendering and cycle collection passed")


if __name__ == "__main__":
    main()
