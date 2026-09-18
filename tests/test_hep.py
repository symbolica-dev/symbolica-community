"""Exercise the flat HEP API against the host's Symbolica kernel."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from symbolica import E, S, Expression
from symbolica.community import hep
from symbolica.community.spenso import TensorExpression


MODEL_PATH = Path(__file__).parents[1] / "examples/hep/scalar_phi3.json"


@pytest.fixture
def model():
    return hep.Model(str(MODEL_PATH))


def test_flat_namespace_and_stubs():
    for name in (
        "FeynmanDiagram", "DiagramEdge", "DiagramVertex", "LoopMomentumBasis",
        "Model", "Particle", "Generator", "Process", "NumeratorGrouping",
        "SelfEnergyFilterOptions", "TadpoleFilterOptions", "SnailFilterOptions", "GenerationProgress",
        "CffGenerator", "CffResult", "TensorReducer", "FourMomentum",
        "ThreeMomentum", "JetDefinition", "UfoLoader", "FeynkitError",
    ):
        assert getattr(hep, name).__module__ == "symbolica.community.hep"
    assert not hasattr(hep, "initialize_module")
    assert not hasattr(hep, "GenerationOptions")
    classes = {name: value for name, value in vars(hep).items() if isinstance(value, type)}
    assert all(value.__module__ == "symbolica.community.hep" for value in classes.values())
    stub = Path(hep.__file__).with_name("__init__.pyi").read_text()
    declarations = {node.name for node in ast.parse(stub).body if isinstance(node, ast.ClassDef)}
    assert classes.keys() <= declarations
    assert "symbolica.community.feynkit" not in stub


def test_generate_diagram_and_cff(model):
    process = hep.Process.amplitude(["scalar_0"], ["scalar_0", "scalar_0"]).with_loop_count(1, 1)
    generated = hep.Generator(model).generate(process, max_vertices=3, allow_self_loops=False)
    assert generated.report.completed
    assert len(generated) > 0
    diagram = generated[0]
    assert isinstance(diagram, hep.FeynmanDiagram)
    assert diagram.loop_count == 1
    denominator = diagram.denominator_expression()
    assert isinstance(denominator, TensorExpression)
    assert denominator.is_scalar
    assert diagram.superficial_degree_of_divergence() == -2
    assert diagram.superficial_degree_of_divergence(dimension=6) == 0
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


def test_generation_keywords_and_empty_results(model):
    incoming, outgoing = ["scalar_0"], ["scalar_0", "scalar_0"]
    process = hep.Process.amplitude(incoming, outgoing)
    generator = hep.Generator(model)
    kwargs = dict(max_vertices=3, coupling_orders={"QCD": 1})
    generated = generator.generate(process, **kwargs)
    ranged = model.generate_diagrams(
        incoming, outgoing, max_vertices=3, coupling_orders={"QCD": (1, 1)},
    )
    assert len(generated) > 0
    assert [diagram.id for diagram in generated] == [diagram.id for diagram in ranged]
    assert kwargs["coupling_orders"] == {"QCD": 1}
    for result in (
        generator.generate(process, particle_veto=["scalar_0"], **kwargs),
        model.generate_diagrams(incoming, outgoing, max_vertices=3, coupling_orders={"QCD": 0}),
    ):
        assert result.report.completed
        assert len(result) == 0
        assert list(result) == []


@pytest.mark.parametrize("via_model", [True, False])
def test_generation_progress_on_calling_thread(model, via_model):
    events = []
    caller = threading.get_ident()
    process = hep.Process.amplitude(["scalar_0"], ["scalar_0", "scalar_0"]).with_loop_count(1, 1)

    def report(progress):
        assert threading.get_ident() == caller
        assert isinstance(progress, hep.GenerationProgress)
        assert progress.completed >= 0
        assert progress.total is None or progress.total >= progress.completed
        events.append(progress)

    def generate(callback):
        if via_model:
            return model.generate_diagrams(
                ["scalar_0"], ["scalar_0", "scalar_0"], loops=1,
                threads=2, max_vertices=3, progress=callback,
            )
        return hep.Generator(model).generate(process, threads=2, max_vertices=3, progress=callback)

    result = generate(report)
    assert result.report.completed and len(result) > 0
    assert events and events[-1].stage == "complete"

    def fail(progress):
        raise RuntimeError("progress callback failed")

    with pytest.raises(RuntimeError, match="progress callback failed"):
        generate(fail)
    assert generate(report).report.completed


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX SIGINT delivery")
@pytest.mark.parametrize("via_model", [True, False])
def test_generation_keyboard_interrupt_and_recovery(via_model):
    # Isolate the native call so a missing signal hook cannot hang pytest.
    script = r'''
import os
import signal
import sys
import threading
from symbolica.community import hep

model = hep.Model(sys.argv[1])
process = hep.Process.amplitude(["scalar_0"], ["scalar_0", "scalar_0"]).with_loop_count(6, 6)
timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT))
timer.daemon = True
timer.start()
try:
    if sys.argv[2] == "True":
        model.generate_diagrams(["scalar_0"], ["scalar_0", "scalar_0"], loops=6, threads=2, max_vertices=13)
    else:
        hep.Generator(model).generate(process, threads=2, max_vertices=13)
except KeyboardInterrupt:
    print("interrupted", flush=True)
else:
    raise AssertionError("Slow generation did not raise KeyboardInterrupt")
finally:
    timer.cancel()
    timer.join()

result = model.generate_diagrams(["scalar_0"], ["scalar_0", "scalar_0"], loops=0, threads=2)
assert result.report.completed and len(result) == 1
print("recovered", flush=True)
'''
    result = subprocess.run(
        [sys.executable, "-c", script, str(MODEL_PATH), str(via_model)],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "interrupted\nrecovered" in result.stdout


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
