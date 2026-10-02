import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="IBP: massless form-factor triangle",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Massless form-factor triangle
    [Browse all notebooks](/) · [Unequal-mass bubble](/?file=hep/ibp_bubble.py) ·
    [Differential equations](/?file=hep/ibp_differential_equations.py)

    Vertex form factors with two on-shell massless legs contain
    $D_1=k^2$, $D_2=(k-p)^2$, $D_3=(k-p-q)^2$, where
    $p^2=q^2=0$ and $s=2p\cdot q\ne0$. Mass insertions and expansions raise
    propagator powers. RustRed reduces these integrals at symbolic dimension
    $D$, before the Laurent expansion at $D=4-2\epsilon$.

    The ordinary triangle reduces to a bubble with a $1/(D-4)$ coefficient.
    Its double infrared pole is therefore retained even though the bubble has
    only a single ultraviolet pole. Setting $D=4$ during reduction would lose
    this information.
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
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import IBPFamily, IntegralFamily, Kinematics, oneloop

    _set_namespace("formfactor")
    return E, IBPFamily, IntegralFamily, Kinematics, S, math, mo, oneloop


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(E, IBPFamily, IntegralFamily, Kinematics, S):
    D, k, p, q, s, eps, I = S(
        "D",
        "k",
        "p",
        "q",
        "s",
        "eps",
        "I",
    )
    zero, one = E("0"), E("1")
    kinematics = (
        Kinematics(D, momenta=[k, p, q])
        .with_scalar_product(p, p, zero)
        .with_scalar_product(q, q, zero)
        .with_scalar_product(p, q, s / 2)
    )
    triangle_family = IntegralFamily(
        [k],
        [p, q],
        [
            kinematics.scalar_product(k, k),
            kinematics.scalar_product(k - p, k - p),
            kinematics.scalar_product(k - p - q, k - p - q),
        ],
        kinematics=kinematics,
    )
    triangle_ibp = IBPFamily(triangle_family, name="triangle")
    targets = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 1, 2), (2, 0, 1)]
    solution = triangle_ibp.reduce_laporta([list(t) for t in targets], max_depth=2)
    assert solution.residuals == [[1, 0, 1]]
    coefficients = {
        t: (solution.reduce(list(t), integral=I) / I(1, 0, 1)).together()
        for t in targets
    }
    return (
        D,
        I,
        coefficients,
        eps,
        one,
        s,
        solution,
        targets,
        triangle_family,
        zero,
    )


@app.cell(hide_code=True)
def _(I, coefficients, mo, solution, triangle_family):
    mo.vstack(
        [
            triangle_family,
            mo.md("**Exact reduction to $I(1,0,1)$**"),
            *[
                mo.hstack([I(*t), mo.md(r"$\longrightarrow$"), c * I(1, 0, 1)])
                for t, c in coefficients.items()
            ],
            mo.ui.table([solution.stats], selection=None),
        ]
    )
    return


@app.cell
def _(D, coefficients, math, mo, one, s, targets, zero):
    # Independent parameter-integral result; no IBP identities enter this check.
    # Gamma(z+n)/Gamma(z) becomes a finite product at integer shifts n.
    exact_checks = []
    for _a, _b, _c in targets:
        _N = _a + _b + _c
        _reference = one * (-1) ** _N * (-s) ** (2 - _N)
        for _j in range(_N - 2):
            _reference *= (2 - D / 2 + _j) * (D - _N + _j)
        for _j in range(_b + _c - 1):
            _reference /= D / 2 - _b - _c + _j
        for _j in range(_a + _b - 1):
            _reference /= D / 2 - _a - _b + _j
        _reference /= math.factorial(_a - 1) * math.factorial(_c - 1)
        assert (coefficients[_a, _b, _c] - _reference).together() == zero
        exact_checks.append((_a, _b, _c))
    mo.md(r"""
    **Independent check: all six rational identities agree exactly.**

    Feynman parameters give, for positive integer $a,c$ and $b\ge0$,
    $$I(a,b,c)=(-1)^{N}(-s-i0)^{D/2-N}
      \frac{\Gamma(N-D/2)\Gamma(D/2-b-c)\Gamma(D/2-a-b)}
      {\Gamma(a)\Gamma(c)\Gamma(D-N)},\qquad N=a+b+c.$$
    This formula uses the measure $d^Dk/(i\pi^{D/2})$; scale and common
    normalization factors cancel in ratios. For $b=0$ it is the two-point
    integral. Dividing by $I(1,0,1)$ and applying the Gamma recurrence gives
    the rational expressions checked above, independently of RustRed.
    Dimensional regularization supplies analytic continuation outside the
    convergence domain of the parameter integral.
    """)
    return


@app.cell
def _(D, E, S, coefficients, eps, mo, s, zero):
    Bfinite = S("Bfinite")
    laurent = {}
    for _target, _coefficient in coefficients.items():
        _order = -1 if _target == (1, 1, 1) else 0
        laurent[_target] = (
            (_coefficient.replace(D, 4 - 2 * eps) * (1 / eps + Bfinite))
            .series(eps, 0, _order)
            .to_expression()
            .expand()
        )
    assert (laurent[1, 1, 1].coefficient(eps**-2) - 1 / s).together() == zero
    assert (
        laurent[1, 1, 1].coefficient(eps**-1) - (Bfinite - 2) / s
    ).together() == zero
    finite_parts = {
        t: dict(value.coefficient_list(eps)).get(E("1"), zero).together()
        for t, value in laurent.items()
        if t != (1, 1, 1)
    }
    mo.vstack(
        [
            mo.md("**Triangle poles, in terms of the finite bubble coefficient**"),
            laurent[1, 1, 1],
            mo.md(r"""
        $B_{\rm finite}=2-\log[(-s-i0)/\mu^2]$. OneLOop checks the double and
        single poles below, including the timelike imaginary part. Computing
        the ordinary triangle's finite term from this reduction additionally
        requires the bubble through $O(\epsilon)$; it is deliberately not
        inferred from the pole and finite bubble coefficients alone.

        All five other targets have coefficients regular at $D=4$, so their
        finite terms follow from the bubble pole and finite term, retaining
        the $O(\epsilon)$ reduction coefficients.
        """),
        ]
    )
    return Bfinite, finite_parts, laurent


@app.cell
def _(Bfinite, eps, laurent, mo, oneloop, s):
    numeric_checks = []
    for _sv, _scale in [(-1.0, 1.0), (-5.0, 2.0), (1.0, 1.0), (9.0, 3.0)]:
        _bubble = oneloop.b0(_sv, 0.0, 0.0, _scale)
        _triangle = oneloop.c0(0.0, 0.0, _sv, 0.0, 0.0, 0.0, _scale)
        _values = {s: _sv, Bfinite: complex(_bubble[0])}
        _double = complex(laurent[1, 1, 1].coefficient(eps**-2).evaluate(_values))
        _single = complex(laurent[1, 1, 1].coefficient(eps**-1).evaluate(_values))
        assert abs(_double - _triangle[2]) < 1e-12
        assert abs(_single - _triangle[1]) < 1e-12
        numeric_checks.append(
            {
                "s": _sv,
                "μ²": _scale,
                "double pole": str(_double),
                "single pole": str(_single),
                "OneLOop error": max(
                    abs(_double - _triangle[2]), abs(_single - _triangle[1])
                ),
            }
        )
    mo.vstack(
        [
            mo.md("**Pole checks against OneLOop**"),
            mo.ui.table(numeric_checks, selection=None),
        ]
    )
    return


@app.cell
def _(mo, targets):
    target_choice = mo.ui.dropdown(
        {str(t): t for t in targets if t != (1, 1, 1)},
        value="(2, 1, 1)",
        label="Raised powers",
    )
    point_choice = mo.ui.dropdown(
        {
            "Spacelike, s=-1": (-1.0, 1.0),
            "Spacelike, s=-5": (-5.0, 2.0),
            "Timelike, s=1": (1.0, 1.0),
            "Timelike, s=9": (9.0, 3.0),
        },
        value="Spacelike, s=-1",
        label="Kinematics",
    )
    mo.hstack([target_choice, point_choice])
    return point_choice, target_choice


@app.cell
def _(
    Bfinite,
    finite_parts,
    laurent,
    mo,
    oneloop,
    point_choice,
    s,
    target_choice,
):
    _sv, _scale = point_choice.value
    selected_powers = target_choice.value
    selected_finite = complex(
        finite_parts[selected_powers].evaluate(
            {
                s: _sv,
                Bfinite: complex(oneloop.b0(_sv, 0.0, 0.0, _scale)[0]),
            }
        )
    )
    mo.vstack(
        [
            mo.md(f"**I{selected_powers}: pole and finite coefficients**"),
            laurent[selected_powers],
            mo.md(
                f"Finite value at s={_sv:g}, μ²={_scale:g}: **`{selected_finite:.12g}`**"
            ),
            mo.md(r"""
        These are dimensionally regulated Laurent coefficients, not ordinary
        convergent four-dimensional integrals. The exact reductions assume
        generic $D$ and $s\ne0$; the $s=0$ family is scaleless and must be
        constructed and reduced separately.
        """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
