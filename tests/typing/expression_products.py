"""Mixed expression products retain the extension's declared return type."""

from typing_extensions import assert_type

from symbolica import Expression, S
from symbolica.community.hepkit import FeynmanDiagram, Model
from symbolica.community.tensor import TensorExpression, TensorStructure


def products(
    scalar: Expression, tensor: TensorExpression
) -> tuple[TensorExpression, TensorExpression, Expression, TensorExpression]:
    return scalar * tensor, tensor * scalar, scalar * scalar, scalar.__mul__(tensor)


class TaggedProduct:
    def __symbolica_rmul__(self, left: Expression) -> tuple[Expression, str]:
        return left, "product"


def tagged_products(
    scalar: Expression, tensor: TensorExpression
) -> tuple[tuple[Expression, str], tuple[Expression, str]]:
    return scalar * TaggedProduct(), tensor * TaggedProduct()


def diagram_numerator(model: Model, diagram: FeynmanDiagram) -> None:
    graph_weight = diagram.overall_factor_expression(evaluate=True)
    raw = diagram.numerator_expression(in_lmb=True)
    assert_type(raw * graph_weight, TensorExpression)
    assert_type(graph_weight * raw, TensorExpression)
    massless = model.expand_couplings(raw * graph_weight)
    reflected = model.expand_couplings(graph_weight * raw)
    assert_type(massless, TensorExpression)
    assert_type(reflected, TensorExpression)
    numerator = massless.with_lorentz_dimension(S("D")).collect_factors()
    assert_type(numerator, TensorExpression)
    assert_type(numerator.structure, TensorStructure)
    assert_type(model.expand_couplings(graph_weight), Expression)
