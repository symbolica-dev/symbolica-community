import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Tadpole mass insertions")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tadpole mass insertions

    [Browse notebooks](/) · [Higgs to gluons](/?file=hep/higgs_gluons.py) ·
    [Odd photons](/?file=hep/odd_photons.py) ·
    [Unequal-mass bubble](/?file=hep/ibp_bubble.py)

    Generate the massive top-quark contribution to the Higgs one-point
    function, then differentiate its integrand with respect to the quark mass.
    Raised propagator powers are reduced by native IBP and evaluated with
    OneLOop. A numerical derivative checks the complete finite answer.

    The Yukawa coupling $y=y_t/\sqrt2$ and the scale $\mu^2$ are held fixed
    during differentiation. Although the selected vertex comes from the SM
    model, this derivative treats the Yukawa coupling and propagator mass as
    independent inputs: it does not impose $y=m/v$.

    Write $\Delta=k^2-m^2+i0$ and $I_n=\int_k\Delta^{-n}$ in the OneLOop
    scalar-integral convention, with $I_1=A_0(m^2;\mu^2)$.
    All displayed amplitudes have the common loop measure $i/(16\pi^2)$
    stripped off. The finite Laurent coefficient shown here is **not a
    renormalized tadpole**; no counterterm or vacuum renormalization condition
    has been imposed.
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
    from symbolica.community import hep
    from symbolica.community import tensor as sp
    import math

    import marimo as mo
    from symbolica import E, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import IBPFamily, Kinematics, Model, oneloop
    from symbolica.community.tensor import TensorExpression

    _set_namespace("tadinsert")
    return (
        E,
        IBPFamily,
        Kinematics,
        Model,
        S,
        Symbol,
        TensorExpression,
        hep,
        math,
        mo,
        oneloop,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the Yukawa interaction
    """)
    return


@app.cell
def _(Model, S, hep, sp):
    model = Model.standard_model()
    D, eps = S("D", "eps")
    m = model.particle("t").mass
    Nc, y, L, scale2 = S("Nc", "y", "L", "mu2")
    I, Af, coordinate = S("I", "Af", "d0")
    K, mink, cof, metric = (
        hep.Kinematics.loop_momentum,
        sp.Representation.mink,
        sp.Representation.cof,
        sp.TensorName.g().to_expression(),
    )
    index, left, right = S("index_", "left_", "right_")
    top, higgs = (model.particle(_name) for _name in ("t", "H"))
    vertices = [
        _vertex
        for _vertex in model.vertex_rules
        if sorted(_vertex.particles) == sorted([top.antiname, top.name, higgs.name])
    ]
    assert len(vertices) == 1

    # Establish the Yukawa phase from the same typed model rule used in the loop.
    return (
        Af,
        D,
        I,
        K,
        L,
        Nc,
        coordinate,
        eps,
        higgs,
        index,
        left,
        m,
        metric,
        model,
        right,
        scale2,
        top,
        vertices,
        y,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fix its tree-level normalization
    """)
    return


@app.cell
def _(E, Symbol, higgs, left, metric, model, right, top, vertices, y):
    _tree_result = model.process(
        [higgs], [top, top.antiparticle], vertex_allow=vertices
    ).generate_diagrams(max_vertices=1, numerator_grouping=None, progress=None)
    assert len(_tree_result.diagrams) == 1
    yukawa_tree = _tree_result.diagrams[0]
    tree_coupling = (
        model.expand_couplings(yukawa_tree.numerator_expression().to_expression())
        .replace(model.parameter("yt").symbol * E("1/2").sqrt(), y)
        .replace(metric(left, right), E("1"))
    )
    assert (tree_coupling + Symbol.I * y).expand() == E("0")
    assert yukawa_tree.overall_factor_expression(evaluate=True) == E("1")
    assert yukawa_tree.numerator_prefactor_expression() == E("1")
    return tree_coupling, yukawa_tree


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the fermion tadpole
    """)
    return


@app.cell
def _(E, higgs, model, vertices):
    _generated = model.process([higgs], [], vertex_allow=vertices).generate_diagrams(
        loops=1,
        max_vertices=1,
        allow_self_loops=True,
        allow_zero_flow_edges=True,
        maximum_bridges=None,
        tadpoles=None,
        zero_snails=None,
        self_energy=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(_generated.diagrams) == 1
    diagram = _generated.diagrams[0]
    native_weight = diagram.overall_factor_expression(evaluate=True)
    # One psi-psibar contraction and one closed-fermion-loop minus sign.
    # The oriented edge cannot exchange its two endpoints.
    wick_weight = E("-1")
    assert native_weight == wick_weight
    assert diagram.symmetry_factor == 1
    assert diagram.numerator_prefactor_expression() == E("1")
    return diagram, native_weight


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate its Dirac and color traces
    """)
    return


@app.cell
def _(
    D,
    E,
    Nc,
    TensorExpression,
    diagram,
    index,
    m,
    model,
    native_weight,
    sp,
    y,
):
    _numerator = (
        model.expand_couplings(
            diagram.numerator_expression(in_lmb=True).to_expression()
        )
        .replace(model.parameter("yt").symbol * E("1/2").sqrt(), y)
        .replace(
            sp.PortPattern.exact(sp.Representation.cof(3), index),
            sp.PortPattern.exact(sp.Representation.cof(Nc), index),
        )
        .replace(
            sp.PortPattern.exact(sp.Representation.mink(4), index),
            sp.PortPattern.exact(sp.Representation.mink(D), index),
        )
    )
    trace = (
        TensorExpression(_numerator.expand())
        .simplify_algebra(
            contract="dots",
            gamma=True,
            epsilon=True,
            color=True,
            color_substitute_cof_dimension_invariants=True,
        )
        .expand()
        .to_expression()
    )
    assert (trace - 4 * Nc * y * m).expand() == E("0")
    coefficient = (
        trace * native_weight * diagram.numerator_prefactor_expression()
    ).expand()
    assert (coefficient + 4 * Nc * y * m).expand() == E("0")
    return coefficient, trace


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Read the scalar tadpole family
    """)
    return


@app.cell
def _(D, E, K, Kinematics, coefficient, coordinate, diagram, m):
    kinematics = Kinematics(D, momenta=[K(0)])
    family = diagram.integral_family(kinematics=kinematics)
    assert family.is_complete and family.is_independent
    assert len(family.denominators) == 1
    denominator = family.denominators[0]
    assert (denominator - kinematics.scalar_product(K(0), K(0)) + m**2).expand() == E(
        "0"
    )
    assert family.rewrite_numerator(coefficient, [coordinate]) == coefficient
    return denominator, family


@app.cell(hide_code=True)
def _(
    coefficient,
    diagram,
    family,
    mo,
    native_weight,
    trace,
    tree_coupling,
    yukawa_tree,
):
    mo.vstack(
        [
            mo.md("**Generated Yukawa tree and quark tadpole**"),
            mo.hstack([yukawa_tree, diagram]),
            mo.hstack([mo.md("Tree coupling:"), tree_coupling]),
            mo.hstack([mo.md("Dirac and color trace:"), trace]),
            mo.hstack([mo.md("Native graph weight:"), native_weight]),
            mo.md(r"""
            There is one contraction of the quark and antiquark at the Yukawa
            vertex. Its closed loop supplies a minus sign; there is no extra
            factor $1/2$. The Dirac trace uses $\operatorname{tr}1=4$ and the
            color trace supplies $N_c$.

            **Generated one-propagator family and coefficient of $I_1$**
            """),
            family,
            coefficient,
        ]
    )
    return


@app.cell
def _(
    D,
    E,
    I,
    IBPFamily,
    Nc,
    coefficient,
    coordinate,
    denominator,
    family,
    m,
    y,
):
    solution = IBPFamily(family, name="massive_tadpole_insertions").reduce_laporta(
        [[1], [2], [3]], max_depth=2
    )
    assert solution.stats["rows"] > 0
    assert {tuple(_powers) for _powers in solution.residuals} == {(1,)}
    reductions = {
        power: solution.reduce([power], integral=I).together() for power in (1, 2, 3)
    }
    assert (reductions[1] - I(1)).together() == E("0")
    assert (reductions[2] - (D - 2) * I(1) / (2 * m**2)).together() == E("0")
    assert (reductions[3] - (D - 4) * (D - 2) * I(1) / (8 * m**4)).together() == E("0")

    # Differentiate the generated rational integrand before invoking IBP.
    integrand = coefficient / denominator
    inserted_integrand = integrand.derivative(m).together()
    assert (
        inserted_integrand + 4 * Nc * y * (1 / denominator + 2 * m**2 / denominator**2)
    ).together() == E("0")
    # Express the differentiated numerator over Delta^2 in the native family
    # coordinate; polynomial degrees determine the remaining denominator powers.
    _insertion_numerator = (inserted_integrand * denominator**2).cancel().expand()
    _polynomial = family.rewrite_numerator(_insertion_numerator, [coordinate]).expand()
    insertion_terms = []
    for _monomial, _coefficient in _polynomial.coefficient_list(coordinate):
        _degree = _monomial.to_polynomial(vars=[coordinate]).degree(coordinate)
        power = 2 - _degree
        assert power in reductions
        assert _coefficient.derivative(coordinate).expand() == E("0")
        insertion_terms.append((power, _coefficient))
    assert {p for p, _ in insertion_terms} == {1, 2}
    inserted_integrals = sum((_c * I(_p) for _p, _c in insertion_terms), E("0"))
    derivative = sum(
        (_c * reductions[_p] for _p, _c in insertion_terms), E("0")
    ).together()
    assert (derivative + 4 * Nc * y * (D - 1) * I(1)).together() == E("0")
    tadpole = coefficient * reductions[1]
    return derivative, inserted_integrals, reductions, solution, tadpole


@app.cell(hide_code=True)
def _(I, derivative, inserted_integrals, mo, reductions, solution):
    mo.vstack(
        [
            mo.md("**Native IBP reductions**"),
            *[
                mo.hstack([I(power), mo.md(r"$\longrightarrow$"), _value])
                for power, _value in reductions.items()
            ],
            mo.ui.table([solution.stats], selection=None),
            mo.md(r"**Mass derivative, before and after IBP**"),
            inserted_integrals,
            derivative,
            mo.md(r"""
            Differentiation acts on both the numerator mass and the propagator:
            $$\frac{\partial}{\partial m}\frac{m}{\Delta}
              =\frac{1}{\Delta}+\frac{2m^2}{\Delta^2}.$$
            The coefficients above come from that differentiated integrand.
            The solver reduces the doubled and tripled propagators to $I_1$.
            """),
        ]
    )
    return


@app.cell
def _(Af, D, E, I, L, Nc, derivative, eps, m, reductions, tadpole, y):
    laurent, finite = {}, {}
    for _label, _expression in [
        ("I1", reductions[1]),
        ("I2", reductions[2]),
        ("I3", reductions[3]),
        ("T", tadpole),
        ("derivative", derivative),
    ]:
        laurent[_label] = (
            _expression.replace(I(1), m**2 / eps + Af)
            .replace(D, 4 - 2 * eps)
            .series(eps, 0, 0)
            .to_expression()
            .expand()
        )
        finite[_label] = (
            dict(laurent[_label].coefficient_list(eps)).get(E("1"), E("0")).together()
        )
        assert laurent[_label].coefficient(eps**-2).expand() == E("0")
    assert (laurent["I1"].coefficient(eps**-1) - m**2).expand() == E("0")
    assert (laurent["I2"].coefficient(eps**-1) - 1).expand() == E("0")
    assert laurent["I3"].coefficient(eps**-1).expand() == E("0")
    assert (laurent["T"].coefficient(eps**-1) + 4 * Nc * y * m**3).expand() == E("0")
    assert (
        laurent["derivative"].coefficient(eps**-1) + 12 * Nc * y * m**2
    ).expand() == E("0")
    analytic_finite = {
        _label: _expression.replace(Af, m**2 * (1 - L)).expand()
        for _label, _expression in finite.items()
    }
    assert (analytic_finite["I2"] + L).together() == E("0")
    assert (analytic_finite["I3"] + 1 / (2 * m**2)).together() == E("0")
    assert (
        analytic_finite["derivative"] + 4 * Nc * y * m**2 * (1 - 3 * L)
    ).expand() == E("0")
    _premature = derivative.replace(D, E("4")).replace(I(1), m**2 * (1 - L))
    rational_correction = (analytic_finite["derivative"] - _premature).expand()
    assert (rational_correction - 8 * Nc * y * m**2).expand() == E("0")
    return analytic_finite, finite, rational_correction


@app.cell(hide_code=True)
def _(analytic_finite, mo, rational_correction):
    mo.vstack(
        [
            mo.md(r"""
            **Laurent expansion and its finite rational term**

            OneLOop supplies
            $$A_0=m^2\left[\frac1\epsilon+1-L\right]+O(\epsilon),
              \qquad L=\log\frac{m^2}{\mu^2},\quad D=4-2\epsilon.$$
            Recombine the exact $D$-dependent coefficients with this pole before
            expanding. This gives
            $$I_2=\frac1\epsilon-L+O(\epsilon),\qquad
              I_3=-\frac1{2m^2}+O(\epsilon),$$
            $$\frac{\partial T}{\partial m}
              =-4N_cym^2\left[\frac3\epsilon+1-3L\right]+O(\epsilon).$$
            Setting $D=4$ early would miss the $-2$ from
            $(3-2\epsilon)/\epsilon$. The resulting finite correction is:
            """),
            rational_correction,
            mo.md(r"**Finite coefficient of $\partial T/\partial m$**"),
            analytic_finite["derivative"],
        ]
    )
    return


@app.cell(hide_code=True)
def _(Af, finite, m, oneloop, scale2):
    master_coefficients = oneloop.master_coefficients(oneloop.A0(m**2, scale2))
    numeric_expressions = {
        _label: _expression.replace(Af, master_coefficients[0])
        for _label, _expression in finite.items()
    }
    return master_coefficients, numeric_expressions


@app.cell
def _(mo):
    mass_control = mo.ui.number(0.05, 500.0, step=0.05, value=2.0, label="Quark mass m")
    scale_control = mo.ui.number(
        0.01, 250000.0, step=0.05, value=4.0, label="Scale squared μ²"
    )
    mo.vstack(
        [
            mo.md(
                "**Numerical check:** vary positive mass and scale; $N_c=3$, $y=0.7$."
            ),
            mo.hstack([mass_control, scale_control]),
        ]
    )
    return mass_control, scale_control


@app.cell
def _(
    Nc,
    m,
    mass_control,
    master_coefficients,
    math,
    numeric_expressions,
    oneloop,
    scale2,
    scale_control,
    y,
):
    mass, selected_scale2 = mass_control.value, scale_control.value
    _point = {m: mass, scale2: selected_scale2, Nc: 3.0, y: 0.7}
    _master = oneloop.a0(mass**2, selected_scale2)
    for _order in (0, 1, 2):
        assert abs(
            complex(master_coefficients[_order].evaluate(_point))
            - complex(_master[_order])
        ) < 2e-12 * max(1.0, abs(_master[_order]))
    assert abs(complex(_master[1]) - mass**2) < 2e-12 * max(1.0, mass**2)
    assert abs(complex(_master[2])) < 1e-14
    values = {
        _label: complex(_expression.evaluate(_point))
        for _label, _expression in numeric_expressions.items()
    }
    _logarithm = math.log(mass**2 / selected_scale2)
    _references = {
        "I2": -_logarithm,
        "I3": -1 / (2 * mass**2),
        "T": -4 * 3 * 0.7 * mass**3 * (1 - _logarithm),
        "derivative": -4 * 3 * 0.7 * mass**2 * (1 - 3 * _logarithm),
    }
    _characteristic_scales = {
        "T": 4 * 3 * 0.7 * mass**3,
        "derivative": 4 * 3 * 0.7 * mass**2,
    }
    for _label, _reference in _references.items():
        assert abs(values[_label] - _reference) < 3e-11 * max(
            1.0, abs(_reference), _characteristic_scales.get(_label, 0.0)
        )
    # Re-evaluate the complete finite tadpole, varying m alone. The shared
    # OneLOop callback sees the shifted mass while y and mu^2 remain fixed.
    _h = mass * 1e-3
    _samples = {
        _j: complex(numeric_expressions["T"].evaluate({**_point, m: mass + _j * _h}))
        for _j in (-2, -1, 1, 2)
    }
    finite_difference = (
        _samples[-2] - 8 * _samples[-1] + 8 * _samples[1] - _samples[2]
    ) / (12 * _h)
    derivative_error = abs(finite_difference - values["derivative"]) / max(
        1.0, abs(values["derivative"]), _characteristic_scales["derivative"]
    )
    assert math.isfinite(derivative_error) and derivative_error < 2e-9
    return derivative_error, finite_difference, mass, selected_scale2, values


@app.cell(hide_code=True)
def _(derivative_error, finite_difference, mass, mo, selected_scale2, values):
    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "m": mass,
                        "μ²": selected_scale2,
                        "Finite T": values["T"].real,
                        "Finite I₂": values["I2"].real,
                        "Finite I₃": values["I3"].real,
                        "Finite ∂T/∂m (IBP + OneLOop)": values["derivative"].real,
                        "Five-point numerical derivative": finite_difference.real,
                        "Scaled derivative error": derivative_error,
                    }
                ],
                selection=None,
            ),
            mo.md(r"""
            The five-point difference varies only $m$, with step $10^{-3}m$.
            Its samples all use the full finite tadpole evaluated by OneLOop.
            The error is divided by $\max(1,|\partial T/\partial m|,4N_cym^2)$,
            which remains meaningful where the derivative crosses zero.
            Both derivatives include the numerator mass and the mass dependence
            of the scalar master. Changing the scale control changes the finite
            coefficients, while the mass derivative keeps that chosen scale fixed.
            """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
