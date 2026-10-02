"""Tests for the bundled community extensions."""

from symbolica import E


def test_example_extension_is_not_bundled():
    import importlib.util
    import sys

    from symbolica import core

    assert importlib.util.find_spec("symbolica.community.example_extension") is None
    assert "symbolica.community.example_extension_native" not in sys.modules
    assert not hasattr(core, "example_extension_native")


def test_tensor_metric_trace():
    from symbolica.community.tensor import TensorExpression

    metric = TensorExpression(E("g(bis(4,1),bis(4,1))", default_namespace="spenso"))
    assert metric.contract().to_expression() == E("4")


def test_spenso_import():
    from symbolica.community.spenso import Representation, Tensor, TensorExpression

    assert Representation is not None
    assert Tensor is not None
    assert TensorExpression is not None


def test_vakint_import():
    from symbolica.community.hep.vakint import Vakint, VakintEvaluationMethod

    assert Vakint is not None
    assert VakintEvaluationMethod is not None


def test_vakint_import_paths_and_citations():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from symbolica import E, get_citations
before = {citation.id for citation in get_citations()}
from symbolica.community.hep import vakint
from symbolica.community import vakint as legacy
for name in ("Vakint", "VakintEvaluationMethod", "VakintExpression", "VakintNumericalResult"):
    cls = getattr(vakint, name)
    assert cls is getattr(legacy, name)
    assert cls.__module__ == "symbolica.community.hep.vakint"
assert {citation.id for citation in get_citations()} == before
assert not hasattr(vakint, "get_citations")
vakint.VakintExpression(E("0"))
assert "https://github.com/alphal00p/vakint#vakint" in {citation.id for citation in get_citations()}
""",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
