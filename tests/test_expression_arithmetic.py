"""Runtime behavior represented by Community's reflected-arithmetic overloads."""

import operator

import pytest
from symbolica import Expression, S
from symbolica.community.tensor import Representation, TensorExpression, TensorName


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "truediv", "pow"])
def test_scalar_tensor_arithmetic(operation):
    scalar, value = S("stub_arithmetic::x", "stub_arithmetic::y")
    tensor = TensorExpression(value)
    apply = getattr(operator, operation)

    result = apply(scalar, tensor)
    assert type(result) is TensorExpression
    assert result.rank == 0
    assert result.to_expression() == apply(scalar, value)

    reverse = apply(tensor, scalar)
    assert type(reverse) is TensorExpression
    assert reverse.rank == 0
    assert reverse.to_expression() == apply(value, scalar)

    # The stubs intentionally describe operator syntax; explicit calls bypass
    # Python's reflected dispatch and continue to produce a plain Expression.
    direct = getattr(scalar, f"__{operation}__")(tensor)
    assert type(direct) is Expression
    assert direct == apply(scalar, value)


@pytest.mark.parametrize("zero", [False, True])
def test_reflected_multiplication_preserves_open_axes(zero):
    scalar = S("stub_arithmetic::x")
    tensor = TensorName("stub_arithmetic::v")(Representation.euc(2))
    if zero:
        tensor = 0 * tensor

    result = scalar * tensor
    assert type(result) is TensorExpression
    assert result.axes == tensor.axes
    assert result.to_expression() == scalar * tensor.to_expression()
