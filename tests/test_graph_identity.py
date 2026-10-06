"""Graph ownership is independent of Community import order."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "order",
    [
        ("graph", "hepkit", "tensor"),
        ("hepkit", "tensor", "graph"),
        ("tensor", "graph", "hepkit"),
    ],
)
def test_common_graph_and_render_classes(order):
    script = "\n".join(f"from symbolica.community import {name}" for name in order)
    script += """
from symbolica import core, E
assert not hasattr(core, "Graph")
assert not hasattr(core, "HalfEdge")
settings = graph.RenderSettings(layouts=graph.LayoutSettings(impred_steps=1))
network = tensor.TensorNetwork(E("x + 2"))
assert type(network.render(config=settings)) is graph.DiagramRender
seen = []
def select(value, completed):
    assert type(value) is graph.Graph
    seen.append(completed)
    return True
model = hepkit.Model.phi3()
diagram = model.process(["phi"], ["phi", "phi"]).generate_diagrams(
    loops=0, filter=select, progress=None,
).diagrams[0]
assert seen
assert type(diagram.to_graph()) is graph.Graph
assert type(diagram.to_graph().full_subgraph()) is graph.Subgraph
assert type(diagram.render(config=settings)) is graph.DiagramRender
assert type(hepkit.Amplitude([diagram]).render(config=settings)) is graph.DiagramRender
assert not hasattr(hepkit, "AmplitudeRender")
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
