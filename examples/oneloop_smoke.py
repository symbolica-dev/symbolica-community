"""Run with .venv-feynkit/bin/python examples/oneloop_smoke.py (requires NumPy)."""

from decimal import Decimal
from math import isclose, log

from symbolica import Expression, S
from symbolica.community.hep import FourMomentum, TensorReducer, oneloop


def main():
    assert not oneloop.is_initialized()  # Import leaves numerical backends lazy.
    assert oneloop.EXPRESSION_INTEROP

    # FeynKit supplies Minkowski kinematics; OneLOop takes squared invariants.
    momentum = FourMomentum(3.0, 1.0, 0.0, 0.0)
    bubble = oneloop.b0(momentum.mass_squared, 4.0, 4.0)
    bubble_jit = oneloop.b0(momentum.mass_squared, 4.0, 4.0, backend="symjit")
    assert all(abs(a - b) < 1e-12 for a, b in zip(bubble, bubble_jit))
    print("B0 (finite, 1/eps, 1/eps^2):", bubble)

    # These are the host's Expression objects, usable by every HEP extension.
    x = S("oneloop_smoke::x")
    coefficients = oneloop.master_coefficients(oneloop.A0(x, 1))
    assert all(isinstance(c, Expression) for c in coefficients)
    reducer = TensorReducer.feynkit(S("oneloop_smoke::D"))
    expression = reducer.reduce(coefficients[0] + coefficients[1])
    evaluator = oneloop.compile_native([expression], [x])
    value = complex(evaluator.evaluate_complex([2 + 0j]).reshape(-1)[0])
    assert isclose(value.real, 2 * (1 - log(2)) + 2, rel_tol=1e-12)
    assert abs(value.imag) < 1e-12
    print("Shared Symbolica/FeynKit expression:", expression)
    print("Evaluation at x=2:", value)

    high_precision = oneloop.a0(Decimal("2"), prec=50)
    assert isinstance(high_precision[0], oneloop.DecimalComplex)
    print("A0 at 50 digits:", high_precision)
    print("Symbolica revision:", oneloop.SYMBOLICA_REVISION)


if __name__ == "__main__":
    main()
