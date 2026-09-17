"""Tests for the bundled community extensions."""
from symbolica import E


def test_example_extension():
    from symbolica.community.example_extension import add_two

    assert add_two(E("x")) == E("x+2")


def test_idenso_metric_trace():
    from symbolica.community.idenso import simplify_metrics

    assert simplify_metrics(E("g(bis(4,1),bis(4,1))", default_namespace="spenso")) == E("4")


def test_spenso_import():
    from symbolica.community.spenso import Representation, Tensor, TensorExpression

    assert Representation is not None
    assert Tensor is not None
    assert TensorExpression is not None


def test_vakint_import():
    from symbolica.community.vakint import Vakint, VakintEvaluationMethod

    assert Vakint is not None
    assert VakintEvaluationMethod is not None
