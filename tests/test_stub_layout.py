"""Public community packages must ship their canonical stubs in wheels."""

import ast
import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "classes"),
    [
        ("symbolica.community.tensor", {"Tensor", "TensorExpression", "Representation"}),
        (
            "symbolica.community.hep.vakint",
            {"Vakint", "VakintExpression", "VakintEvaluationMethod", "VakintNumericalResult"},
        ),
    ],
)
def test_public_package_has_matching_stub(module_name, classes):
    module = importlib.import_module(module_name)
    stub = Path(module.__file__).with_suffix(".pyi")
    assert stub.is_file(), f"Missing packaged stub: {stub}"
    declarations = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    declared_classes = {
        node.name for node in declarations.body if isinstance(node, ast.ClassDef)
    }
    assert classes <= declared_classes
    for name in classes:
        assert getattr(module, name).__module__ == module_name
