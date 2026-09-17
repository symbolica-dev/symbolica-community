"""Exercise the flat HEP API against the host's Symbolica kernel."""

import ast
from pathlib import Path

import pytest

from symbolica import E, S, Expression
from symbolica.community import hep


MODEL_PATH = Path(__file__).parents[1] / "examples/hep/scalar_phi3.json"


@pytest.fixture
def model():
    return hep.Model(str(MODEL_PATH))


def test_flat_namespace_and_stubs():
    for name in (
        "FeynmanDiagram", "DiagramEdge", "DiagramVertex", "LoopMomentumBasis",
        "Model", "Particle", "Generator", "GenerationOptions", "Process",
        "CffGenerator", "CffResult", "TensorReducer", "FourMomentum",
        "ThreeMomentum", "JetDefinition", "UfoLoader", "FeynkitError",
    ):
        assert getattr(hep, name).__module__ == "symbolica.community.hep"
    assert not hasattr(hep, "initialize_module")
    classes = {name: value for name, value in vars(hep).items() if isinstance(value, type)}
    assert all(value.__module__ == "symbolica.community.hep" for value in classes.values())
    stub = Path(hep.__file__).with_name("__init__.pyi").read_text()
    declarations = {node.name for node in ast.parse(stub).body if isinstance(node, ast.ClassDef)}
    assert classes.keys() <= declarations
    assert "symbolica.community.feynkit" not in stub


def test_generate_diagram_and_cff(model):
    options = hep.GenerationOptions(max_vertices=3, allow_self_loops=False)
    process = hep.Process.amplitude(["scalar_0"], ["scalar_0", "scalar_0"]).with_loop_count(1, 1)
    generated = hep.Generator(model).generate(process, options)
    assert generated.report.completed
    assert len(generated) > 0
    diagram = generated[0]
    assert isinstance(diagram, hep.FeynmanDiagram)
    assert diagram.loop_count == 1
    diagram.validate()
    restored = hep.FeynmanDiagram.from_json(model, diagram.to_json())
    from_dot = hep.FeynmanDiagram.from_dot(model, diagram.to_dot())
    assert restored.loop_count == from_dot.loop_count == 1
    assert restored.name == diagram.name
    assert len(restored.loop_momentum_bases()[0].loop_edges) == 1
    cff = hep.CffGenerator().generate(restored)
    assert isinstance(cff, hep.CffResult)
    assert len(cff) > 0
    assert isinstance(cff.to_expression(), Expression)
    assert diagram.build_cff().to_expression() == cff.to_expression()


def test_tensor_reduction_uses_host_expressions():
    dimension, mu, nu = S("hep_test::D", "hep_test::mu", "hep_test::nu")
    k, p = S("hep_test::k", "hep_test::p")
    mink, dot = S("spenso::mink", "spenso::dot")
    k_vector, p_vector = k(mink(dimension)), p(mink(dimension))
    numerator = k(mink(dimension, mu)) * k(mink(dimension, nu)) * p(mink(dimension, mu)) * p(mink(dimension, nu))
    reducer = hep.TensorReducer(dimension).with_integrated_vector(k_vector)
    reduced = reducer.reduce(numerator)
    assert isinstance(reduced, Expression)
    assert reduced == dot(k_vector, k_vector) * dot(p_vector, p_vector) / dimension
    assert hep.TensorReducer.feynkit(E("4")).reduce(E("3")) == E("3")


def test_kinematics_and_error_types():
    assert hep.ThreeMomentum(3.0, 4.0, 0.0).on_shell().components() == (5.0, 3.0, 4.0, 0.0)
    jets = hep.JetDefinition.anti_kt(0.4).cluster([
        hep.FourMomentum(10.0, 10.0, 0.0, 0.0),
        hep.FourMomentum(5.0, 5.0, 0.0, 0.0),
    ])
    assert len(jets) == 1
    assert jets[0].constituent_indices == [0, 1]
    assert jets[0].momentum.components() == (15.0, 15.0, 0.0, 0.0)
    with pytest.raises(hep.ModelError) as caught:
        hep.Model.from_json("{}")
    assert isinstance(caught.value, hep.FeynkitError)
