"""Reduce a shared Feynkit family and evaluate a raised tadpole's coefficients.

Run with the community Symbolica build. Tagged primitive master symbols use
OneLoopMaster's registered native Rust numerical hooks directly.
"""

from math import isclose, log

from symbolica import E, S
from symbolica.community.hep import IntegralFamily, Kinematics, oneloop


def main():
    mass_squared = S("oneloop_example::mass_squared")
    dimension, loop = S("oneloop_example::D", "oneloop_example::k")
    kinematics = Kinematics(dimension, momenta=[loop])
    family = IntegralFamily(
        [loop],
        [],
        [kinematics.scalar_product(loop, loop) - mass_squared],
        kinematics=kinematics,
    )
    reduction = oneloop.reduce(family, [2]).simplify()
    print("Reduction:", reduction.to_expression())

    # Keep the O(eps) part of (d-2)/(2*m²): it multiplies the A0 pole
    # and contributes -1 to the finite part.
    coefficients = oneloop.reduction_coefficients(reduction, mu_squared=E("1"))
    print("Tagged OneLoopMaster coefficients:", coefficients)
    values = [
        coefficient.evaluate({mass_squared: 2 + 0j}) for coefficient in coefficients
    ]
    print("At m²=2 (finite, 1/eps, 1/eps²):", values)
    assert isclose(values[0].real, -log(2), rel_tol=1e-12)
    assert abs(values[0].imag) < 1e-12
    assert abs(values[1] - 1) < 1e-12
    assert abs(values[2]) < 1e-12


if __name__ == "__main__":
    main()
