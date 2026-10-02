"""Exercise the flat HEP API against the host's Symbolica kernel."""

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from symbolica import E, S, Expression
from symbolica.community import hepkit as hep
from symbolica.community.tensor import Representation, TensorExpression, TensorName, dot


MODEL_PATH = Path(__file__).parents[1] / "examples/hep/scalar_phi3.json"


@pytest.fixture
def model():
    return hep.Model(str(MODEL_PATH))


def test_flat_namespace_and_stubs():
    assert hep.__name__ == "symbolica.community.hepkit"
    assert importlib.import_module("symbolica.community.hepkit") is hep
    for name in (
        "FeynmanDiagram",
        "DiagramEdge",
        "DiagramVertex",
        "LoopMomentumBasis",
        "Model",
        "Particle",
        "Process",
        "NumeratorGrouping",
        "SelfEnergyFilterOptions",
        "TadpoleFilterOptions",
        "SnailFilterOptions",
        "GenerationProgress",
        "CffGenerator",
        "CffResult",
        "TensorReducer",
        "FourMomentum",
        "ThreeMomentum",
        "JetDefinition",
        "UfoLoader",
        "FeynkitError",
    ):
        assert getattr(hep, name).__module__ == "symbolica.community.hepkit"
    assert not hasattr(hep, "initialize_module")
    assert not hasattr(hep, "GenerationOptions")
    classes = {
        name: value for name, value in vars(hep).items() if isinstance(value, type)
    }
    assert all(
        value.__module__ == "symbolica.community.hepkit" for value in classes.values()
    )
    stub = Path(hep.__file__).with_name("__init__.pyi").read_text()
    declarations = {
        node.name for node in ast.parse(stub).body if isinstance(node, ast.ClassDef)
    }
    assert classes.keys() <= declarations
    assert "symbolica.community.feynkit" not in stub


@pytest.mark.parametrize(
    "module_name", ["symbolica.hep", "symbolica.community.hep", "symbolica.hepkit"]
)
def test_superseded_hep_import_is_removed(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.skipif(not hasattr(hep, "IBPFamily"), reason="IBP is native-only")
def test_ibp_package_reuses_flat_class_identities():
    ibp = importlib.import_module("symbolica.community.hepkit.ibp")
    assert ibp.__name__ == "symbolica.community.hepkit.ibp"
    assert ibp.__path__
    assert ibp.__spec__.submodule_search_locations is not None
    for name in ("IBPFamily", "IBPRule", "IBPSolution"):
        assert getattr(ibp, name) is getattr(hep, name)
        assert getattr(ibp, name).__module__ == "symbolica.community.hepkit"


def test_generate_diagram_and_cff(model):
    process = model.process(["scalar_0"], ["scalar_0", "scalar_0"])
    generated = process.generate_diagrams(
        loops=1, max_vertices=3, allow_self_loops=False
    )
    assert generated.report.completed
    assert len(generated.diagrams) > 0
    diagram = generated.diagrams[0]
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
    process = model.process(incoming, outgoing)
    kwargs = dict(max_vertices=3, coupling_orders={"QCD": 1})
    generated = process.generate_diagrams(**kwargs)
    ranged = model.process(incoming, outgoing).generate_diagrams(
        max_vertices=3,
        coupling_orders={"QCD": (1, 1)},
    )
    assert len(generated.diagrams) > 0
    assert [diagram.id for diagram in generated.diagrams] == [
        diagram.id for diagram in ranged.diagrams
    ]
    assert kwargs["coupling_orders"] == {"QCD": 1}
    for result in (
        model.process(incoming, outgoing, particle_veto=["scalar_0"]).generate_diagrams(
            **kwargs
        ),
        process.generate_diagrams(max_vertices=3, coupling_orders={"QCD": 0}),
    ):
        assert result.report.completed
        assert len(result.diagrams) == 0
        assert result.diagrams == []


@pytest.mark.parametrize("threads", [1, 2])
def test_generation_progress_on_calling_thread(model, threads):
    events = []
    caller = threading.get_ident()
    process = model.process(["scalar_0"], ["scalar_0", "scalar_0"])

    def report(progress):
        assert threading.get_ident() == caller
        assert isinstance(progress, hep.GenerationProgress)
        assert progress.completed >= 0
        assert progress.total is None or progress.total >= progress.completed
        events.append(progress)

    def generate(callback):
        return process.generate_diagrams(
            loops=1, threads=threads, max_vertices=3, progress=callback
        )

    result = generate(report)
    assert result.report.completed and len(result.diagrams) > 0
    assert events and events[-1].stage == "complete"

    def fail(progress):
        raise RuntimeError("progress callback failed")

    with pytest.raises(RuntimeError, match="progress callback failed"):
        generate(fail)
    assert generate(report).report.completed


@pytest.mark.skipif(os.name != "posix", reason="Uses POSIX SIGINT delivery")
@pytest.mark.parametrize("threads", [1, 2])
def test_generation_keyboard_interrupt_and_recovery(threads):
    # Isolate the native call so a missing signal hook cannot hang pytest.
    script = r"""
import os
import signal
import sys
import threading
from symbolica.community import hepkit as hep

model = hep.Model(sys.argv[1])
process = model.process(["scalar_0"], ["scalar_0", "scalar_0"])
timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT))
timer.daemon = True
timer.start()
try:
    process.generate_diagrams(loops=6, threads=int(sys.argv[2]), max_vertices=13)
except KeyboardInterrupt:
    print("interrupted", flush=True)
else:
    raise AssertionError("Slow generation did not raise KeyboardInterrupt")
finally:
    timer.cancel()
    timer.join()

result = process.generate_diagrams(loops=0, threads=int(sys.argv[2]))
assert result.report.completed and len(result.diagrams) == 1
print("recovered", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(MODEL_PATH), str(threads)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "interrupted\nrecovered" in result.stdout


@pytest.mark.parametrize("compose_tensors", [False, True])
def test_tensor_reduction_uses_host_expressions(compose_tensors):
    dimension, mu, nu = S("hep_test::D", "hep_test::mu", "hep_test::nu")
    space = Representation.mink(dimension)
    k, p = TensorName.vector("hep_test::k"), TensorName.vector("hep_test::p")
    k_vector, p_vector = k(space), p(space)
    numerator = (
        k(space(mu)).to_expression()
        * k(space(nu)).to_expression()
        * p(space(mu)).to_expression()
        * p(space(nu)).to_expression()
    )
    if compose_tensors:
        # Incremental tensor composition can encode the first scalar product
        # inside a weighted operand of the second dot. It must have the same
        # isotropic moment as the explicit indexed components.
        numerator = (
            k(space(mu)) * k(space(nu)) * p(space(mu)) * p(space(nu))
        ).to_expression()
    reducer = hep.TensorReducer(dimension, integrated=[k_vector.to_expression()])
    reduced = reducer.reduce(numerator)
    assert isinstance(reduced, Expression)
    expected = (
        dot(k_vector, k_vector) * dot(p_vector, p_vector) / dimension
    ).to_expression()
    assert (reduced - expected).expand() == E("0")
    assert hep.TensorReducer.feynkit(E("4")).reduce(E("3")) == E("3")


def test_kinematics_and_error_types():
    assert hep.ThreeMomentum(3.0, 4.0, 0.0).on_shell().components() == (
        5.0,
        3.0,
        4.0,
        0.0,
    )
    jets = hep.JetDefinition.anti_kt(0.4).cluster(
        [
            hep.FourMomentum(10.0, 10.0, 0.0, 0.0),
            hep.FourMomentum(5.0, 5.0, 0.0, 0.0),
        ]
    )
    assert len(jets) == 1
    assert jets[0].constituent_indices == [0, 1]
    assert jets[0].momentum.components() == (15.0, 15.0, 0.0, 0.0)
    with pytest.raises(hep.ModelError) as caught:
        hep.Model.from_json("{}")
    assert isinstance(caught.value, hep.FeynkitError)
