"""Generate a scalar one-loop diagram and its CFF representation."""

from pathlib import Path

from symbolica.community.hepkit import FeynmanDiagram, Model

model = Model(str(Path(__file__).with_name("scalar_phi3.json")))
process = model.process(["scalar_0"], ["scalar_0", "scalar_0"])
result = process.generate_diagrams(loops=1, max_vertices=3, allow_self_loops=False)

diagram: FeynmanDiagram = result[0]
diagram.validate()
print("Diagram:", diagram.name)
print("Loops:", diagram.loop_count)
print("DOT graph:")
print(diagram.to_dot())
print("CFF:", diagram.build_cff().to_expression())
