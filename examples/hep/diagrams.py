"""Generate a scalar one-loop diagram and its CFF representation."""

from pathlib import Path

from symbolica.community import hep

model = hep.Model(str(Path(__file__).with_name("scalar_phi3.json")))
process = hep.Process.amplitude(["scalar_0"], ["scalar_0", "scalar_0"]).with_loop_count(1, 1)
options = hep.GenerationOptions(max_vertices=3, allow_self_loops=False)
result = hep.Generator(model).generate(process, options)

diagram: hep.FeynmanDiagram = result[0]
diagram.validate()
print("Diagram:", diagram.name)
print("Loops:", diagram.loop_count)
print("DOT graph:")
print(diagram.to_dot())
print("CFF:", diagram.build_cff().to_expression())
