import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="One-loop reduction and native master symbols",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop reduction and native master symbols

    [Browse notebooks](/) · [Integral families](/?file=hep/integral_families.py) ·
    [Unequal-mass bubble IBP](/?file=hep/ibp_bubble.py)

    Build the shared `IntegralFamily` with Feynkit kinematics and pass it to
    `oneloop.reduce`, reducing a loop-momentum numerator directly
    to the primitive **`oneloopmaster::A0`, `B0`, `C0`, and `D0` symbols**.
    Their registered numerical hooks run OneLoopMaster's generated native Rust
    arithmetic. Calling `Expression.evaluate` on the resulting coefficients
    uses those hooks directly, without constructing an evaluator or function map.

    We reduce a triangle numerator, inspect the exact symbolic expression of a
    second triangle on a chosen analytic branch, and check a squared tadpole
    whose finite term depends on keeping $d=4-2\epsilon$ during reduction.
    Results use OneLoopMaster's normalization and coefficient order
    **finite, $1/\epsilon$, $1/\epsilon^2$**.

    The conventions and inspection API follow the
    [OneLoopMaster README](https://github.com/alphal00p/oneloopmaster#compact-master-symbols).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Imports and supporting routines are folded below. Expand a cell’s code to
    inspect or edit it; the calculation that follows shows the HEP operations.
    """)
    return


@app.cell(hide_code=True)
def _():
    import math

    import marimo as mo
    import numpy as np
    from symbolica import E, N, Replacement, S, Float, ComplexFloat
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import IBPFamily, IntegralFamily, Kinematics, oneloop

    _set_namespace("olr_example")
    return (
        ComplexFloat,
        E,
        Float,
        IBPFamily,
        IntegralFamily,
        Kinematics,
        N,
        Replacement,
        S,
        math,
        mo,
        np,
        oneloop,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(S, oneloop):
    # Ordinary momentum names and an arbitrary symbolic dimension belong to
    # the shared Feynkit frontend; only scalar masters use its backend namespace.
    k, q1, q2, d = S("k", "q1", "q2", "D")
    A0, B0, C0 = oneloop.A0, oneloop.B0, oneloop.C0
    mass2, mu2, mass2B = S("mass2", "mu2", "mass2B", is_positive=True)
    p1sq, p2sq, invariant = S(
        "p1sq",
        "p2sq",
        "s",
        is_real=True,
    )
    return A0, B0, C0, d, invariant, k, mass2, mu2, p1sq, p2sq, q1, q2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A triangle with a loop-momentum numerator

    Let $r_0=0$, $r_1=q_1$, and $r_2=q_1+q_2$. The denominators are
    $D_i=(k+r_i)^2-m^2+i0$, and the numerator is $k\cdot q_1$.

    Declare $q_1^2=p_1^2$, $q_2^2=p_2^2$, and
    $q_1\cdot q_2=(s-p_1^2-p_2^2)/2$ in `Kinematics`.
    Build the three denominator expressions with `scalar_product`; the reducer
    derives the masses and master invariants from this shared family.
    The output master is
    `C0(p1sq, p2sq, s, mass2, mass2, mass2, mu2)`.
    The **last argument is always the squared renormalization scale**.

    Keep the dimension symbolic: `Kinematics(D, ...)` retains the dependence
    needed for $D=4-2\epsilon$. The explicit powers `[1, 1, 1]` specify this
    integral; zero powers pinch propagators, and negative powers put them in
    the numerator. The same family can also be passed to `IBPFamily`.
    """)
    return


@app.cell
def _(
    IBPFamily,
    IntegralFamily,
    Kinematics,
    d,
    invariant,
    k,
    mass2,
    mo,
    mu2,
    oneloop,
    p1sq,
    p2sq,
    q1,
    q2,
):
    triangle_kinematics = (
        Kinematics(d, momenta=[k, q1, q2])
        .with_scalar_product(q1, q1, p1sq)
        .with_scalar_product(q2, q2, p2sq)
        .with_scalar_product(q1, q2, (invariant - p1sq - p2sq) / 2)
    )
    triangle = IntegralFamily(
        [k],
        [q1, q2],
        [
            triangle_kinematics.scalar_product(_momentum, _momentum) - mass2
            for _momentum in [k, k + q1, k + q1 + q2]
        ],
        kinematics=triangle_kinematics,
    )
    triangle_ibp = IBPFamily(triangle, name="oneloop_triangle")
    triangle_reduction = oneloop.reduce(
        triangle, [1, 1, 1], numerator=triangle_kinematics.scalar_product(k, q1)
    ).simplify()
    mo.vstack(
        [
            mo.md("**Shared Feynkit family, also accepted by `IBPFamily`**"),
            triangle,
            mo.md("**Reduced expression in primitive OneLoopMaster symbols**"),
            triangle_reduction.to_expression(mu_squared=mu2),
            mo.md("**Coefficient × scalar master**"),
            *[
                mo.hstack(
                    [
                        _coefficient,
                        _master.to_expression(mu_squared=mu2),
                    ],
                    justify="start",
                )
                for _coefficient, _master in triangle_reduction.terms
            ],
        ]
    )
    return (triangle_reduction,)


@app.cell
def _(B0, C0, E, invariant, mass2, mu2, p1sq, p2sq, triangle_reduction):
    # Independent algebraic check: 2 k.q1 = D1 - D0 - q1².
    triangle_expected = (
        B0(invariant, mass2, mass2, mu2)
        - B0(p2sq, mass2, mass2, mu2)
        - p1sq * C0(p1sq, p2sq, invariant, mass2, mass2, mass2, mu2)
    ) / 2
    assert (
        triangle_reduction.to_expression(mu_squared=mu2) - triangle_expected
    ).expand() == E("0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Laurent coefficients and native numerical hooks

    An untagged call describes the master integral for symbolic inspection.
    A **leading Laurent-power tag** selects a numerical coefficient:
    `B0(0, s, m0sq, m1sq, mu2)` is the finite term;
    tags `-1` and `-2` select the simple and double poles.

    `reduction_coefficients` expands the rational prefactors at $d=4-2\epsilon$
    and combines them with these tagged primitive calls. Evaluate each resulting
    expression with a map of parameter values. OneLoopMaster's symbol hooks
    handle the master calls and return complex numbers, including at real inputs.

    The exported symbols also evaluate a tagged call during construction when
    all its arguments are numeric and at least one is inexact (`Float` or
    `ComplexFloat`). The native Rust hook preserves the supplied working precision.
    Untagged calls and tagged calls with only exact arguments stay symbolic.
    Here the leading `0` requests the finite term; the complex squared mass has
    the nonpositive imaginary part required by OneLoopMaster.
    """)
    return


@app.cell
def _(ComplexFloat, Float, mo, oneloop):
    _exact_master = oneloop.A0(2, 1)
    _exact_finite = oneloop.A0(0, 2, 1)
    _float_finite = oneloop.A0(0, Float("2", decimal_digits=50), 1)
    _complex_finite = oneloop.A0(0, ComplexFloat("2", "-0.1", decimal_digits=50), 1)
    mo.vstack(
        [
            mo.hstack(
                [mo.md("Untagged master for inspection"), _exact_master],
                justify="start",
            ),
            mo.hstack([mo.md("Exact tagged call"), _exact_finite], justify="start"),
            mo.hstack(
                [mo.md("Native result from a 50-digit Float input"), _float_finite],
                justify="start",
            ),
            mo.hstack(
                [
                    mo.md("Native result from a 50-digit ComplexFloat input"),
                    _complex_finite,
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(mo, mu2, oneloop, triangle_reduction):
    triangle_coefficients = oneloop.reduction_coefficients(
        triangle_reduction,
        mu_squared=mu2,
    )
    mo.vstack(
        [
            mo.hstack([mo.md(_label), _expression], justify="start")
            for _label, _expression in zip(
                ["Finite", "$1/\\epsilon$", "$1/\\epsilon^2$"],
                triangle_coefficients,
            )
        ]
    )
    return (triangle_coefficients,)


@app.cell(hide_code=True)
def _(mo):
    mass_input = mo.ui.slider(
        start=0.5, stop=5, step=0.5, value=2, label="Mass squared m²"
    )
    scale_input = mo.ui.slider(
        start=0.5, stop=5, step=0.5, value=1, label="Scale squared μ²"
    )
    momentum_input = mo.ui.slider(
        start=0.5, stop=4, step=0.5, value=1, label="Spacelike scale Q²"
    )
    mo.vstack(
        [
            mo.md(
                r"Choose positive squared masses and scales. For the reduced triangle, "
                r"$p_1^2=-Q^2$, $p_2^2=-2Q^2$, and $s=-3Q^2$."
            ),
            mass_input,
            scale_input,
            momentum_input,
        ]
    )
    return mass_input, momentum_input, scale_input


@app.cell
def _(
    B0,
    C0,
    invariant,
    mass2,
    mass_input,
    momentum_input,
    mu2,
    np,
    p1sq,
    p2sq,
    scale_input,
    triangle_coefficients,
):
    _m, _mu, _q = mass_input.value, scale_input.value, momentum_input.value
    triangle_point = {mass2: _m, mu2: _mu, p1sq: -_q, p2sq: -2 * _q, invariant: -3 * _q}
    triangle_values = np.asarray(
        [
            _coefficient.evaluate(triangle_point)
            for _coefficient in triangle_coefficients
        ]
    )
    # Independently construct the tagged masters in the numerator identity.
    triangle_reference = np.asarray(
        [
            (
                B0(_tag, -3 * _q, _m, _m, _mu)
                - B0(_tag, -2 * _q, _m, _m, _mu)
                + _q * C0(_tag, -_q, -2 * _q, -3 * _q, _m, _m, _m, _mu)
            ).evaluate({})
            / 2
            for _tag in [0, -1, -2]
        ]
    )
    np.testing.assert_allclose(
        triangle_values, triangle_reference, rtol=1e-10, atol=1e-12
    )
    return triangle_reference, triangle_values


@app.cell(hide_code=True)
def _(mo, triangle_reference, triangle_values):
    mo.vstack(
        [
            mo.md(
                "**Triangle: reduced coefficients evaluated through native symbol hooks**"
            ),
            mo.ui.table(
                [
                    {
                        "Coefficient": _label,
                        "Reduction": str(complex(_got)),
                        "Master identity": str(complex(_expected)),
                        "Absolute difference": float(abs(_got - _expected)),
                    }
                    for _label, _got, _expected in zip(
                        ["Finite", "1/ε", "1/ε²"],
                        triangle_values,
                        triangle_reference,
                    )
                ],
                selection=None,
            ),
            mo.md(
                "The exact numerator identity and all three numerical coefficients agree."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect an exact triangle expression and select its analytic branch

    Take the nontrivial one-mass triangle
    $C_0(0,0,s;0,m^2,0;\mu^2=1)$, which contains a dilogarithm.
    `get_expression` takes an **untagged primitive master call** and returns
    its three exact Laurent coefficients with every OneLoopMaster helper expanded.
    The invariant $s$ stays real and symbolic; $m^2$ stays positive and symbolic.

    `select_branch` uses numerical **probe replacements only in branch conditions**.
    It selects a formula for the probe's analytic region without substituting
    those numbers into the returned formula. Choose a spacelike or timelike
    probe below; the timelike expression includes the prescribed continuation.
    A selected formula must be selected again when crossing a branch cut.
    """)
    return


@app.cell
def _(C0, invariant, mass2, mo, oneloop):

    inspection_master = C0(0, 0, invariant, 0, mass2, 0, 1)
    all_c0_branches = oneloop.get_expression(inspection_master)
    mo.vstack(
        [
            inspection_master,
            mo.accordion(
                {
                    "Complete exact coefficients before selecting a branch": mo.vstack(
                        list(all_c0_branches)
                    ),
                }
            ),
        ]
    )
    return (all_c0_branches,)


@app.cell(hide_code=True)
def _(mo):
    branch_input = mo.ui.dropdown(
        options={"Spacelike: s < 0": -1, "Timelike: s > 0": 1},
        value="Spacelike: s < 0",
        label="Analytic region",
    )
    branch_input
    return (branch_input,)


@app.cell
def _(
    N,
    Replacement,
    all_c0_branches,
    branch_input,
    invariant,
    mass2,
    mo,
    oneloop,
):
    branch_probe = [
        Replacement(invariant, N(2 * branch_input.value)),
        Replacement(mass2, N(1)),
    ]
    selected_c0 = oneloop.select_branch(all_c0_branches, branch_probe)
    assert invariant in selected_c0[0].get_all_symbols(False)
    assert mass2 in selected_c0[0].get_all_symbols(False)
    assert "if(" not in str(selected_c0)
    mo.vstack(
        [
            mo.md(
                f"**Branch probe:** $s={2 * branch_input.value}$, $m^2=1$. "
                "The exact selected coefficients below retain both symbols."
            ),
            *[
                mo.hstack([mo.md(_label), _expression], justify="start")
                for _label, _expression in zip(
                    ["Finite", "$1/\\epsilon$", "$1/\\epsilon^2$"],
                    selected_c0,
                )
            ],
        ]
    )
    return (selected_c0,)


@app.cell
def _(
    C0,
    branch_input,
    invariant,
    mass2,
    mass_input,
    mo,
    momentum_input,
    np,
    selected_c0,
):
    # Evaluate at the slider values, independently of the fixed branch probe.
    inspection_point = {
        invariant: 2 * branch_input.value * momentum_input.value,
        mass2: mass_input.value,
    }
    selected_c0_values = np.asarray(
        [_expr.evaluate(inspection_point) for _expr in selected_c0]
    )
    native_c0_values = np.asarray(
        [
            C0(_tag, 0, 0, invariant, 0, mass2, 0, 1).evaluate(inspection_point)
            for _tag in [0, -1, -2]
        ]
    )
    np.testing.assert_allclose(
        selected_c0_values, native_c0_values, rtol=1e-11, atol=1e-12
    )
    mo.vstack(
        [
            mo.md(
                f"**Selected exact formula versus primitive native hook**, at "
                f"$s={inspection_point[invariant]}$, $m^2={inspection_point[mass2]}$, $\\mu^2=1$."
            ),
            mo.ui.table(
                [
                    {
                        "Coefficient": _label,
                        "Selected formula": str(complex(_exact)),
                        "Native C0 hook": str(complex(_native)),
                        "Absolute difference": float(abs(_exact - _native)),
                    }
                    for _label, _exact, _native in zip(
                        ["Finite", "1/ε", "1/ε²"],
                        selected_c0_values,
                        native_c0_values,
                    )
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A squared tadpole: epsilon terms change the finite answer

    For $I_2=\int_k (k^2-m^2+i0)^{-2}$, reduction gives
    $I_2=(d-2)A_0(m^2;\mu^2)/(2m^2)$.
    Since $(d-2)/(2m^2)=(1-\epsilon)/m^2$, the tadpole's pole contributes
    an additional $-1$ to the finite term. Setting $d=4$ prematurely would
    miss it. The correct coefficients are
    $[-\log(m^2/\mu^2),\,1,\,0]$.
    """)
    return


@app.cell
def _(A0, E, IntegralFamily, Kinematics, d, k, mass2, mo, mu2, oneloop):
    _vacuum = Kinematics(d, momenta=[k])
    tadpole = IntegralFamily(
        [k], [], [_vacuum.scalar_product(k, k) - mass2], kinematics=_vacuum
    )
    tadpole_reduction = oneloop.reduce(tadpole, [2]).simplify()
    assert (
        tadpole_reduction.to_expression(mu_squared=mu2)
        - (d - 2) / (2 * mass2) * A0(mass2, mu2)
    ).expand() == E("0")
    tadpole_coefficients = oneloop.reduction_coefficients(
        tadpole_reduction, mu_squared=mu2
    )
    mo.vstack([tadpole_reduction.to_expression(mu_squared=mu2), *tadpole_coefficients])
    return (tadpole_coefficients,)


@app.cell
def _(mass2, mass_input, math, mo, mu2, np, scale_input, tadpole_coefficients):
    tadpole_point = {mass2: mass_input.value, mu2: scale_input.value}
    tadpole_values = np.asarray(
        [_expr.evaluate(tadpole_point) for _expr in tadpole_coefficients]
    )
    tadpole_reference = [-math.log(mass_input.value / scale_input.value), 1, 0]
    np.testing.assert_allclose(
        tadpole_values, tadpole_reference, rtol=1e-12, atol=1e-12
    )
    mo.vstack(
        [
            mo.md(
                "**Squared tadpole: native master hooks versus the analytic answer**"
            ),
            mo.ui.table(
                [
                    {
                        "Coefficient": _label,
                        "Reduction": str(complex(_got)),
                        "Analytic answer": _expected,
                        "Absolute difference": float(abs(_got - _expected)),
                    }
                    for _label, _got, _expected in zip(
                        ["Finite", "1/ε", "1/ε²"],
                        tadpole_values,
                        tadpole_reference,
                    )
                ],
                selection=None,
            ),
            mo.md(
                "Changing the mass or scale above updates the result, including the finite "
                "contribution from the epsilon-dependent coefficient."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
