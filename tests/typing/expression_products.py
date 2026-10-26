"""Check normal Python multiplication with explicit tensor-result narrowing."""

from typing_extensions import assert_type

from symbolica import Expression, S
from symbolica.community.hepkit import FeynmanDiagram, Model
from symbolica.community.tensor import TensorExpression, TensorStructure


def products(
    scalar: Expression, tensor: TensorExpression
) -> tuple[TensorExpression, TensorExpression, Expression, Expression]:
    product = scalar * tensor
    # ty conservatively admits Expression subclasses with different dispatch:
    # https://github.com/astral-sh/ruff/pull/26474
    assert isinstance(product, TensorExpression)
    # A direct method call bypasses Python's reflected-operator dispatch.
    return product, tensor * scalar, scalar * scalar, scalar.__mul__(tensor)


class ReflectedProduct:
    def __rmul__(self, left: Expression) -> tuple[Expression, str]:
        return left, "product"


def reflected_product(scalar: Expression) -> tuple[Expression, str]:
    return scalar * ReflectedProduct()


def diagram_numerator(model: Model, diagram: FeynmanDiagram) -> None:
    graph_weight = diagram.overall_factor_expression(evaluate=True)
    raw = diagram.numerator_expression(in_lmb=True)
    assert_type(raw * graph_weight, TensorExpression)
    weighted = graph_weight * raw
    assert isinstance(weighted, TensorExpression)
    assert_type(weighted, TensorExpression)
    massless = model.expand_couplings(raw * graph_weight)
    reflected = model.expand_couplings(weighted)
    assert_type(massless, TensorExpression)
    assert_type(reflected, TensorExpression)
    numerator = massless.with_lorentz_dimension(S("D")).collect_factors()
    assert_type(numerator, TensorExpression)
    assert_type(numerator.structure, TensorStructure)
    assert_type(model.expand_couplings(graph_weight), Expression)
