"""Check reflected arithmetic without casts or runtime narrowing."""

from symbolica import Expression, HeldExpression, Matrix, S, Series
from symbolica.community.hepkit import FeynmanDiagram, Model
from symbolica.community.tensor import (
    Tensor,
    TensorExpression,
    TensorNetwork,
    TensorStructure,
)
from typing_extensions import assert_type


def products(
    scalar: Expression, tensor: TensorExpression
) -> tuple[TensorExpression, TensorExpression, Expression, Expression]:
    product = scalar * tensor
    assert_type(product, TensorExpression)
    # Direct dunder calls are outside the reflected-operator typing workaround.
    return product, tensor * scalar, scalar * scalar, scalar.__mul__(tensor)


def arithmetic(scalar: Expression, tensor: TensorExpression) -> None:
    assert_type(scalar + tensor, TensorExpression)
    assert_type(scalar - tensor, TensorExpression)
    assert_type(scalar * tensor, TensorExpression)
    assert_type(scalar / tensor, TensorExpression)
    assert_type(scalar**tensor, TensorExpression)
    assert_type(tensor + scalar, TensorExpression)
    assert_type(tensor - scalar, TensorExpression)
    assert_type(tensor * scalar, TensorExpression)
    assert_type(tensor / scalar, TensorExpression)
    assert_type(tensor**scalar, TensorExpression)
    assert_type(scalar + scalar, Expression)
    assert_type(scalar - scalar, Expression)
    assert_type(scalar * scalar, Expression)
    assert_type(scalar / scalar, Expression)
    assert_type(scalar**scalar, Expression)
    assert_type(scalar + 2, Expression)
    assert_type(scalar - 2, Expression)
    assert_type(scalar * 2, Expression)
    assert_type(scalar / 2, Expression)
    assert_type(scalar**2, Expression)


def other_reflected_operands(
    scalar: Expression,
    tensor: Tensor,
    network: TensorNetwork,
    held: HeldExpression,
    series: Series,
    matrix: Matrix,
) -> None:
    assert_type(scalar + tensor, TensorNetwork)
    assert_type(scalar - tensor, TensorNetwork)
    assert_type(scalar * tensor, TensorNetwork)
    assert_type(scalar / tensor, TensorNetwork)
    assert_type(scalar + network, TensorNetwork)
    assert_type(scalar - network, TensorNetwork)
    assert_type(scalar * network, TensorNetwork)
    assert_type(scalar / network, TensorNetwork)
    assert_type(scalar + held, HeldExpression)
    assert_type(scalar - held, HeldExpression)
    assert_type(scalar * held, HeldExpression)
    assert_type(scalar / held, HeldExpression)
    assert_type(scalar**held, HeldExpression)
    assert_type(scalar + series, Series)
    assert_type(scalar - series, Series)
    assert_type(scalar * series, Series)
    assert_type(scalar / series, Series)
    assert_type(scalar * matrix, Matrix)
    assert_type(scalar @ matrix, Matrix)


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
    assert_type(weighted, TensorExpression)
    massless = model.expand_couplings(raw * graph_weight)
    reflected = model.expand_couplings(weighted)
    assert_type(massless, TensorExpression)
    assert_type(reflected, TensorExpression)
    numerator = massless.with_lorentz_dimension(S("D")).collect_factors()
    assert_type(numerator, TensorExpression)
    assert_type(numerator.structure, TensorStructure)
    assert_type(model.expand_couplings(graph_weight), Expression)
