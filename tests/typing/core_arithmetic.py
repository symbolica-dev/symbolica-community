"""Arithmetic typing without any optional Community imports."""

from decimal import Decimal

from symbolica import ComplexFloat, Expression, Float, S
from typing_extensions import assert_type


def scalar_arithmetic(left: Expression, right: Expression) -> None:
    assert_type(left + right, Expression)
    assert_type(left - right, Expression)
    assert_type(left * right, Expression)
    assert_type(left / right, Expression)
    assert_type(left**right, Expression)
    assert_type(left + 2, Expression)
    assert_type(left - 2.0, Expression)
    assert_type(left * (2 + 1j), Expression)
    assert_type(left / Decimal(2), Expression)
    assert_type(left ** Float(2), Expression)
    assert_type(left * ComplexFloat(2), Expression)
    assert_type(S("x") * S("y"), Expression)


class Reflected:
    def __radd__(self, left: Expression, /) -> str:
        return "sum"

    def __rsub__(self, left: Expression, /) -> int:
        return 1

    def __rmul__(self, left: Expression, /) -> tuple[Expression, str]:
        return left, "product"

    def __rtruediv__(self, left: Expression, /) -> float:
        return 0.5

    def __rpow__(self, left: Expression, /) -> bool:
        return True


def unrelated_reflected_results(scalar: Expression, operand: Reflected) -> None:
    assert_type(scalar + operand, str)
    assert_type(scalar - operand, int)
    assert_type(scalar * operand, tuple[Expression, str])
    assert_type(scalar / operand, float)
    assert_type(scalar**operand, bool)
