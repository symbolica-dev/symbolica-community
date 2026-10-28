"""Public community packages must ship their canonical stubs in wheels."""

import ast
import importlib
import inspect
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "classes"),
    [
        (
            "symbolica.community.graph",
            {"DiagramRender", "LayoutSettings", "Stroke", "RenderSettings"},
        ),
        (
            "symbolica.community.hep.integration",
            {
                "IntegralEvaluator",
                "PreparedIntegralFamily",
                "KinematicTransport",
                "BoundaryCache",
                "EvaluationOptions",
                "DifferentialSystem",
                "BoundaryData",
                "LaurentExpansion",
                "TransportResult",
            },
        ),
        (
            "symbolica.community.tensor",
            {"Tensor", "TensorExpression", "Representation"},
        ),
        ("symbolica.community.hepkit.oneloop", {"Reduction", "MasterIntegral"}),
        (
            "symbolica.community.hepkit.sector_decomposition",
            {
                "Integral",
                "GeneratedIntegral",
                "GenerationSession",
                "Kernels",
                "CompilationSettings",
                "FormulaPreparationSnapshot",
            },
        ),
        (
            "symbolica.community.hepkit.vakint",
            {
                "CacheFileIdentity",
                "CacheFileOptions",
                "Vakint",
                "VakintExpression",
                "VakintEvaluationMethod",
                "VakintNumericalResult",
            },
        ),
    ],
)
def test_public_package_has_matching_stub(module_name, classes):
    module = importlib.import_module(module_name)
    if module_name.startswith("symbolica.community.hepkit."):
        assert module.__path__
        assert module.__spec__.submodule_search_locations is not None
    stub = Path(module.__file__).with_suffix(".pyi")
    assert stub.is_file(), f"Missing packaged stub: {stub}"
    # Check the minimum supported grammar even when tests run on newer Python.
    declarations = ast.parse(
        stub.read_text(encoding="utf-8"), filename=str(stub), feature_version=9
    )
    declared_classes = {
        node.name for node in declarations.body if isinstance(node, ast.ClassDef)
    }
    assert classes <= declared_classes
    for name in classes:
        assert getattr(module, name).__module__ == module_name


def test_ibp_package_stub_reexports_canonical_types():
    hepkit = importlib.import_module("symbolica.community.hepkit")
    if not hasattr(hepkit, "IBPFamily"):
        pytest.skip("IBP is native-only")
    ibp = importlib.import_module("symbolica.community.hepkit.ibp")
    stub = Path(ibp.__file__).with_suffix(".pyi")
    declarations = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    classes = {"IBPFamily", "IBPRule", "IBPSolution"}
    assert not any(isinstance(node, ast.ClassDef) for node in declarations.body)
    reexports = {
        alias.name
        for node in declarations.body
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module is None
        for alias in node.names
        if alias.name == alias.asname
    }
    assert classes <= reexports
    for name in classes:
        assert getattr(ibp, name) is getattr(hepkit, name)
        assert getattr(ibp, name).__module__ == "symbolica.community.hepkit"


def test_tensor_evaluator_stub_matches_loaded_signature():
    tensor = importlib.import_module("symbolica.community.tensor")
    stub = Path(tensor.__file__).with_suffix(".pyi")
    declarations = ast.parse(stub.read_text(), feature_version=9)
    tensor_class = next(
        node
        for node in declarations.body
        if isinstance(node, ast.ClassDef) and node.name == "Tensor"
    )
    evaluator = next(
        node
        for node in tensor_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluator"
    )
    declared = [arg.arg for arg in evaluator.args.posonlyargs + evaluator.args.args]
    assert declared == list(inspect.signature(tensor.Tensor.evaluator).parameters)


@pytest.mark.parametrize("name", ["evaluator", "evaluator_multiple"])
def test_kernel_evaluator_stub_matches_loaded_signature(name):
    core = importlib.import_module("symbolica.core")
    stub = Path(core.__file__).parent / "core.pyi"
    declarations = ast.parse(stub.read_text(), feature_version=9)
    expression = next(
        node
        for node in declarations.body
        if isinstance(node, ast.ClassDef) and node.name == "Expression"
    )
    method = next(
        node
        for node in expression.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    declared = [arg.arg for arg in method.args.posonlyargs + method.args.args]
    if declared[0] == "_cls":
        declared.pop(0)
    signature = inspect.signature(getattr(core.Expression, name))
    assert declared == list(signature.parameters)
    assert declared[-1] == "horner_use_cpe"
    assert signature.parameters["horner_use_cpe"].default is False
    assert isinstance(method.args.defaults[-1], ast.Constant)
    assert method.args.defaults[-1].value is False


def test_graph_package_exposes_generic_graph_and_rendering_api():
    from symbolica.community import graph

    generic_api = {
        "CutPartition",
        "Cycle",
        "DirectionBasis",
        "DotCodec",
        "DotEdgeData",
        "DotHalfEdgeData",
        "DotVertexData",
        "Edge",
        "EdgeSignature",
        "EdgeSpec",
        "EdgeValue",
        "Flow",
        "GlobalData",
        "Graph",
        "HalfEdge",
        "HalfEdgeSpec",
        "HalfEdgeValue",
        "LayoutFrame",
        "LayoutStream",
        "Node",
        "NodeSpec",
        "NodeStore",
        "NodeValue",
        "Orientation",
        "OrientedCut",
        "Subgraph",
        "TraversalTree",
        "build",
        "edge",
        "node",
        "sink",
        "source",
    }
    rendering_api = {"DiagramRender", "LayoutSettings", "Stroke", "RenderSettings"}
    expected_exports = generic_api | rendering_api
    assert expected_exports <= vars(graph).keys()
    native = importlib.import_module("symbolica.community.graph_native")
    for name in expected_exports:
        exported = getattr(graph, name)
        assert exported is getattr(native, name)
        if inspect.isclass(exported):
            assert exported.__module__ == "symbolica.community.graph"
    declarations = ast.parse(Path(graph.__file__).with_suffix(".pyi").read_text())
    stub_exports = {
        node.name
        for node in declarations.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert expected_exports <= stub_exports


def test_graph_presentation_types_are_shared_by_both_renderers():
    from symbolica import E
    from symbolica.community import graph, hepkit, tensor

    for name in ("DiagramRender", "LayoutSettings", "Stroke", "RenderSettings"):
        assert getattr(graph, name).__module__ == "symbolica.community.graph"

    layout = graph.LayoutSettings(impred_steps=1)
    stroke = graph.Stroke(thickness=1)
    settings = graph.RenderSettings(
        layouts=layout, drawing=graph.DrawOptions(edge_stroke=stroke)
    )
    drawing = tensor.TensorNetwork(E("1")).render(config=settings)
    assert isinstance(drawing, graph.DiagramRender)
    diagram = (
        hepkit.Model.phi4()
        .process(["phi", "phi"], ["phi", "phi"])
        .generate_diagrams(progress=None)
        .diagrams[0]
    )
    assert isinstance(diagram.render(config=settings), graph.DiagramRender)


def test_ibp_campaign_api_survives_stub_regeneration():
    hepkit = importlib.import_module("symbolica.community.hepkit")
    if not hasattr(hepkit, "IBPFamily"):
        pytest.skip("IBP is native-only")
    maintained = Path(__file__).parents[1] / "stubs" / "ibp.pyi"
    generated = Path(hepkit.__file__).with_suffix(".pyi")
    source_ast = ast.parse(maintained.read_text(), feature_version=9)
    generated_ast = ast.parse(generated.read_text(), feature_version=9)

    def family_methods(tree):
        family = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "IBPFamily"
        )
        return {
            node.name: node for node in family.body if isinstance(node, ast.FunctionDef)
        }

    source_methods, generated_methods = (
        family_methods(source_ast),
        family_methods(generated_ast),
    )
    for name in (
        "parameter_bindings",
        "start_generation",
        "normalize_candidate_terminals",
    ):
        assert name in source_methods and name in generated_methods
        assert ast.dump(source_methods[name]) == ast.dump(generated_methods[name])
        assert hasattr(hepkit.IBPFamily, name)
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module is None
        and any(alias.name == alias.asname == "rustred" for alias in node.names)
        for node in generated_ast.body
    )
