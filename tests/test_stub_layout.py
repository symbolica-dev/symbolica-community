"""Public community packages must ship their canonical stubs in wheels."""

import ast
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "classes"),
    [
        (
            "symbolica.community.hep.integration",
            {
                "IntegralEvaluator", "PreparedIntegralFamily", "KinematicTransport",
                "BoundaryCache", "EvaluationOptions", "DifferentialSystem",
                "BoundaryData", "LaurentExpansion", "TransportResult",
            },
        ),
        (
            "symbolica.community.tensor",
            {"Tensor", "TensorExpression", "Representation"},
        ),
        ("symbolica.community.hepkit.oneloop", {"Reduction", "MasterIntegral"}),
        (
            "symbolica.community.hepkit.vakint",
            {
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


def test_ibp_campaign_api_survives_stub_regeneration():
    hepkit = importlib.import_module("symbolica.community.hepkit")
    if not hasattr(hepkit, "IBPFamily"):
        pytest.skip("IBP is native-only")
    maintained = Path(__file__).parents[1] / "stubs" / "ibp.pyi"
    generated = Path(hepkit.__file__).with_suffix(".pyi")
    source_ast = ast.parse(maintained.read_text(), feature_version=9)
    generated_ast = ast.parse(generated.read_text(), feature_version=9)

    def family_methods(tree):
        family = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "IBPFamily")
        return {node.name: node for node in family.body if isinstance(node, ast.FunctionDef)}

    source_methods, generated_methods = family_methods(source_ast), family_methods(generated_ast)
    for name in ("parameter_bindings", "start_generation", "normalize_candidate_terminals"):
        assert name in source_methods and name in generated_methods
        assert ast.dump(source_methods[name]) == ast.dump(generated_methods[name])
        assert hasattr(hepkit.IBPFamily, name)
    assert any(
        isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
        and any(alias.name == alias.asname == "rustred" for alias in node.names)
        for node in generated_ast.body
    )
