import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Generated phi3 renormalization")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop $\phi^3$ renormalization
    [Browse notebooks](/) · [Quartic scalar renormalization](/?file=hep/phi4_renormalization.py) · [Tadpole mass insertions](/?file=hep/tadpole_mass_insertions.py)

    Generate the bubble, triangle and local counterterms from `Model.phi3()`. The [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/Phi3/OneLoop/Renormalization) determines the field, mass-squared and cubic-coupling constants in four dimensions.

    Use $\mathcal L=(\partial\phi)^2/2-m^2\phi^2/2-g\phi^3/3!$, with nonzero $m^2=M$ and dimensionful coupling $g$. This calculation concerns the one-particle-irreducible two- and three-point functions. A one-point tadpole or a choice of vacuum is separate.

    All displayed loop amplitudes have the common $i/(16\pi^2)$ removed. Keeping $M>0$ makes the zero-momentum Taylor coefficients infrared safe.
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
    from symbolica.community import hepkit as hep
    import json
    import math

    import marimo as mo
    import numpy as np
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import (
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        oneloop,
    )
    from symbolica.community.tensor import TensorExpression

    _set_namespace("phi3")
    return (
        E,
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Matrix,
        Model,
        Replacement,
        S,
        Symbol,
        TensorExpression,
        hep,
        json,
        math,
        mo,
        np,
        oneloop,
    )


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the cubic model and regulator
    """)
    return


@app.cell
def _(E, Model, S, Symbol, Symbols, hep):
    model = Model.phi3()
    particle = model.particle("phi")
    mass = model.parameter("mass").symbol
    coupling = model.parameter("g").symbol
    d, M, p2, mu2, eps, k, coordinate, integral = S(
        "d",
        "M",
        "p2",
        "mu2",
        "eps",
        "k",
        "x",
        "I",
    )
    Q, K, P = (
        Symbols.edge_momentum,
        hep.Kinematics.loop_momentum,
        hep.Kinematics.external_momentum,
    )
    index_pattern, external_pattern = S("index_", "external_")
    zero, one, pi = (E("0"), E("1"), Symbol.PI)
    return (
        K,
        M,
        P,
        Q,
        coordinate,
        coupling,
        d,
        eps,
        external_pattern,
        index_pattern,
        integral,
        k,
        mass,
        model,
        mu2,
        one,
        p2,
        particle,
        pi,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the vacuum recurrence
    """)
    return


@app.cell
def _(IBPFamily, IntegralFamily, Kinematics, M, d, k):
    _kinematics = Kinematics(d, momenta=[k])
    family = IntegralFamily(
        [k], [], [_kinematics.scalar_product(k, k) - M], kinematics=_kinematics
    )
    ibp = IBPFamily(family, name="phi3_tadpole")
    recurrence = ibp.solve_parametric([True], max_depth=1)
    assert recurrence.rules
    return family, ibp, recurrence


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select one-loop diagrams
    """)
    return


@app.cell
def _():
    options = {
        "maximum_bridges": 0,
        "self_energy": None,
        "tadpoles": None,
        "zero_snails": None,
        "numerator_grouping": None,
        "progress": None,
    }
    return (options,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and normalize the amplitudes
    """)
    return


@app.cell
def _(
    K,
    Kinematics,
    M,
    P,
    Symbol,
    coordinate,
    coupling,
    d,
    external_pattern,
    family,
    index_pattern,
    integral,
    k,
    mass,
    model,
    options,
    particle,
    zero,
):
    diagrams, inputs, coefficients = ({}, {}, {})
    for _label, _incoming, _outgoing, _loops in [
        ("self_energy", 1, 1, 1),
        ("vertex", 2, 1, 1),
        ("tree", 2, 1, 0),
    ]:
        _generated = model.process(
            [particle] * _incoming, [particle] * _outgoing
        ).generate_diagrams(
            loops=_loops, max_vertices=_incoming + _outgoing if _loops else 1, **options
        )
        assert len(_generated.diagrams) == 1
        _diagram = _generated.diagrams[0]
        diagrams[_label] = _diagram
        _coefficient = (
            model.expand_couplings(_diagram.numerator_expression().to_expression())
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        coefficients[_label] = _coefficient
        if not _loops:
            continue
        _generated_family = _diagram.propagator_family(kinematics=Kinematics(d))
        _denominators = [
            _den.replace(P(external_pattern, index_pattern), zero)
            .replace(K(0, index_pattern), k(index_pattern))
            .replace(mass**2, M)
            for _den in _generated_family.denominators
        ]
        _powers = len(_denominators)
        assert _powers == _incoming + _outgoing
        assert all(
            family.rewrite_numerator(_den, [coordinate]) == coordinate
            for _den in _denominators
        )
        inputs[_label] = _coefficient * integral(_powers)
    assert coefficients == {
        "self_energy": coupling**2 / 2,
        "vertex": coupling**3,
        "tree": -Symbol.I * coupling,
    }
    return coefficients, diagrams, inputs


@app.cell(hide_code=True)
def _(diagrams, family, inputs, mo):
    mo.vstack(
        [
            mo.md("## Generated graphs and their zero-momentum integrals"),
            mo.hstack([diagrams["self_energy"], diagrams["vertex"]]),
            family,
            mo.hstack([inputs["self_energy"], inputs["vertex"]]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Native IBP and finite terms
    The parametric recurrence and bounded Laporta reduction agree for $I_2$ and $I_3$, where $I_n=\int_k(k^2-M+i0)^{-n}$ and $I_1=A_0(M)$.

    Expand the dimension-dependent coefficients before removing the master pole. In particular, the factor $D-4$ in $I_3$ multiplies the tadpole's $1/\epsilon$ pole and produces a nonzero finite triangle. The generic-momentum bubble separately checks that the UV pole has no $p^2$ dependence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the two IBP routes

    Parametric recurrence and finite Laporta reduction must give the same raised tadpole powers.
    """)
    return


@app.cell
def _(M, d, ibp, integral, recurrence, zero):
    reductions = {1: integral(1)}
    for _power in (2, 3):
        _terms = recurrence.reduce([_power], integral=integral)
        reductions[_power] = _terms.replace(
            integral(_power - 1), reductions[_power - 1]
        ).together()
    assert (reductions[2] / integral(1) - (d - 2) / (2 * M)).together() == zero
    assert (
        reductions[3] / integral(1) - (d - 2) * (d - 4) / (8 * M**2)
    ).together() == zero
    laporta = ibp.reduce_laporta([[2], [3]], max_depth=2)
    for _power in (2, 3):
        assert (
            laporta.reduce([_power], integral=integral) - reductions[_power]
        ).together() == zero
    return (reductions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Insert the master coefficients
    """)
    return


@app.cell
def _(
    M,
    S,
    coupling,
    d,
    eps,
    family,
    inputs,
    integral,
    mu2,
    oneloop,
    reductions,
    zero,
):
    _master_reduction = oneloop.reduce(family, [1])
    _master = _master_reduction.terms[0][1].to_expression(mu2)
    _master_pole = oneloop.get_expression(_master, coefficient=-1)
    assert _master_pole == M
    _master_finite = oneloop.reduction_coefficients(_master_reduction, mu2)[0]
    _finite_symbol = S("Afinite")
    _reduced, poles, finite = ({}, {}, {})
    for _label, _power in [("self_energy", 2), ("vertex", 3)]:
        _reduced[_label] = inputs[_label].replace(integral(_power), reductions[_power])
        _expanded = (
            _reduced[_label]
            .replace(d, 4 - 2 * eps)
            .replace(integral(1), M / eps + _finite_symbol)
            .series(eps, 0, 0)
            .to_expression()
            .expand()
        )
        poles[_label] = _expanded.coefficient(eps ** (-1))
        finite[_label] = (
            (_expanded - poles[_label] / eps)
            .together()
            .replace(_finite_symbol, _master_finite)
        )
    assert poles == {"self_energy": coupling**2 / 2, "vertex": zero}
    assert (finite["vertex"] + coupling**3 / (2 * M)).together() == zero
    return finite, poles


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the momentum-dependent bubble

    Read its invariant from the graph routing and compare its pole with the vacuum calculation.
    """)
    return


@app.cell
def _(
    IntegralFamily,
    K,
    Kinematics,
    M,
    P,
    Q,
    Replacement,
    coefficients,
    d,
    diagrams,
    mu2,
    one,
    oneloop,
    p2,
    poles,
    zero,
):
    _self_diagram = diagrams["self_energy"]
    _basis = _self_diagram.momentum_basis()
    _routed_kinematics = Kinematics(d, momenta=[K(0), P(0), P(1)]).with_scalar_product(
        P(0), P(0), p2
    )
    _shifts = []
    for _edge in _self_diagram.internal_edges:
        _sign = _basis.edge_signatures[_edge.id].loops[0]
        assert abs(_sign) == 1
        _shift = (_basis.route_expression(Q(_edge.id)) / _sign - K(0)).expand()
        _shifts.append(_shift)
    _invariant = _routed_kinematics.scalar_product(
        _shifts[0] - _shifts[1], _shifts[0] - _shifts[1]
    )
    assert _invariant == p2
    _bubble_family = IntegralFamily(
        [K(0)],
        [P(0)],
        [
            _routed_kinematics.scalar_product(_momentum, _momentum) - M
            for _momentum in [K(0), K(0) + P(0)]
        ],
        kinematics=_routed_kinematics,
    )
    _bubble_reduction = oneloop.reduce(_bubble_family, [1, 1])
    _bubble_master = _bubble_reduction.terms[0][1].to_expression(mu2)
    _bubble_pole = oneloop.select_branch(
        oneloop.get_expression(_bubble_master, coefficient=-1),
        [Replacement(M, one), Replacement(p2, one)],
    )
    assert _bubble_pole * coefficients["self_energy"] == poles["self_energy"]
    assert poles["self_energy"].derivative(p2) == zero
    bubble_finite = (
        coefficients["self_energy"]
        * oneloop.reduction_coefficients(_bubble_reduction, mu2)[0]
    )
    return (bubble_finite,)


@app.cell(hide_code=True)
def _(finite, mo, poles, reductions):
    mo.vstack(
        [
            mo.md("**Doubled and tripled tadpoles**"),
            mo.hstack([reductions[2], reductions[3]]),
            mo.md("**Self-energy and vertex UV residues**"),
            mo.hstack([poles["self_energy"], poles["vertex"]]),
            mo.md("**Finite zero-momentum triangle**"),
            finite["vertex"],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated counterterms
    Expand the bare factors $Z_\phi$, $Z_\phi Z_m$ and $Z_g Z_\phi^{3/2}$. The generated local vertices supply the three-by-three matching system; no renormalization constants are provided as inputs.

    MS subtracts $1/\epsilon$. MS̄ subtracts $\Delta=1/\epsilon+\log(4\pi)-\gamma_E$. The exact checks retain the finite difference between schemes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the local counterterm vertices
    """)
    return


@app.cell
def _(E, Model, S, Symbol, coupling, json, mass, model, particle):
    h, field, mass_ct, vertex = S("h", "field", "mass_ct", "vertex")
    _Zfield, _Zmass, _Zvertex = (1 + h * _x for _x in (field, mass_ct, vertex))
    _specification = json.loads(model.to_json())
    _specification["orders"].append(
        {"name": "CT", "expansion_order": 1, "hierarchy": 1}
    )
    for _label, _valence, _lorentz, _factor in [
        ("kinetic", 2, "P(dummy(1),1)*P(dummy(1),1)", Symbol.I * (_Zfield - 1)),
        ("mass", 2, "1", -Symbol.I * mass**2 * (_Zfield * _Zmass - 1)),
        ("cubic", 3, "1", -Symbol.I * coupling * (_Zvertex * _Zfield ** E("3/2") - 1)),
    ]:
        _name = "CT_" + _label
        _coefficient = _factor.series(h, 0, 1).to_expression().expand().coefficient(h)
        _specification["lorentz_structures"].append(
            {"name": _name, "spins": [1] * _valence, "structure": _lorentz}
        )
        _specification["couplings"].append(
            {
                "name": _name,
                "expression": repr(h * _coefficient),
                "orders": [["SCALAR", int(_valence == 3)], ["CT", 1]],
                "value": None,
            }
        )
        _specification["vertex_rules"].append(
            {
                "name": _name,
                "particles": [particle.name] * _valence,
                "color_structures": ["1"],
                "lorentz_structures": [_name],
                "couplings": [[_name]],
            }
        )
    ct_model = Model.from_json(json.dumps(_specification))
    return ct_model, field, h, mass_ct, vertex


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate their amplitudes
    """)
    return


@app.cell
def _(
    Kinematics,
    M,
    P,
    Symbol,
    ct_model,
    h,
    mass,
    options,
    p2,
    particle,
    zero,
):
    ct_diagrams, counterterms = ({}, {})
    _self_kinematics = Kinematics().with_scalar_product(P(0), P(0), p2)
    for _label, _incoming, _count in [("self_energy", 1, 2), ("vertex", 2, 1)]:
        _generated = ct_model.process(
            [particle.name] * _incoming, [particle.name]
        ).generate_diagrams(
            loops=0, max_vertices=1, coupling_orders={"CT": 1}, **options
        )
        ct_diagrams[_label] = _generated.diagrams
        assert len(_generated.diagrams) == _count
        _amplitude = zero
        for _diagram in _generated.diagrams:
            _numerator = ct_model.expand_couplings(_diagram.numerator_expression())
            _numerator = _numerator.contract().to_dots().expand().to_expression()
            _numerator = _diagram.momentum_basis().route_expression(_numerator)
            _amplitude += (
                _self_kinematics.apply(_numerator)
                * _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
            )
        counterterms[_label] = (
            (_amplitude.replace(mass**2, M) / (Symbol.I * h)).together().expand()
        )
    return counterterms, ct_diagrams


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the counterterm equations
    """)
    return


@app.cell
def _(
    M,
    Matrix,
    counterterms,
    coupling,
    field,
    mass_ct,
    p2,
    poles,
    vertex,
    zero,
):
    unknowns = [field, mass_ct, vertex]
    _rows = [
        counterterms["self_energy"].coefficient(p2),
        counterterms["self_energy"].replace(p2, zero),
        counterterms["vertex"],
    ]
    ct_matrix = Matrix.from_linear(
        3, 3, [_row.coefficient(_x) for _row in _rows for _x in unknowns]
    )
    _solution = ct_matrix.solve(
        Matrix.vec([zero, -poles["self_energy"], -poles["vertex"]])
    )
    residues = [_solution[_row, 0].to_expression() for _row in range(3)]
    assert residues == [zero, coupling**2 / (2 * M), zero]
    return ct_matrix, residues, unknowns


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare subtraction schemes
    """)
    return


@app.cell
def _(Replacement, S, counterterms, eps, pi, poles, residues, unknowns, zero):
    _log4pi, _gamma_e = S("log4pi", "gamma_E")
    constants = {}
    for _scheme, _delta in [("MS", 1 / eps), ("MSbar", 1 / eps + _log4pi - _gamma_e)]:
        constants[_scheme] = [
            1 + _residue * _delta / (16 * pi**2) for _residue in residues
        ]
        _replacements = [
            Replacement(_x, _residue * _delta)
            for _x, _residue in zip(unknowns, residues, strict=True)
        ]
        for _label in poles:
            _remainder = (
                poles[_label] * (1 / eps + _log4pi - _gamma_e)
                + counterterms[_label].replace_multiple(_replacements)
            ).together()
            _expected = (
                zero if _scheme == "MSbar" else poles[_label] * (_log4pi - _gamma_e)
            )
            assert (_remainder - _expected).together() == zero
    return (constants,)


@app.cell(hide_code=True)
def _(ct_diagrams, ct_matrix, mo, residues):
    mo.vstack(
        [
            mo.hstack(ct_diagrams["self_energy"] + ct_diagrams["vertex"]),
            mo.md("**Matching matrix**"),
            ct_matrix,
            mo.md("**Solved residues, with 1/(16π²) removed**"),
            mo.hstack(residues),
        ]
    )
    return


@app.cell
def _(M, bubble_finite, coupling, finite, math, mu2, np, p2):
    _coefficients = [finite["self_energy"], finite["vertex"], bubble_finite]
    _nodes, _weights = np.polynomial.legendre.leggauss(128)
    _nodes, _weights = ((_nodes + 1) / 2, _weights / 2)
    _points = [
        (1.0, 0.5, 1.0, 0.0),
        (2.0, 1.0, 3.0, -4.0),
        (4.0, 2.0, 0.5, 3.0),
        (0.25, 0.1, 2.0, 0.5),
    ]
    numeric_checks = []
    for _mv, _gv, _scale, _pv in _points:
        _point = {M: _mv, coupling: _gv, mu2: _scale, p2: _pv}
        _values = [_coefficient.evaluate(_point) for _coefficient in _coefficients]
        _reference = [
            -(_gv**2) * math.log(_mv / _scale) / 2,
            -(_gv**3) / (2 * _mv),
            -(_gv**2)
            * sum(
                (
                    float(_w)
                    * math.log((_mv - _pv * float(_x) * (1 - float(_x))) / _scale)
                    for _x, _w in zip(_nodes, _weights, strict=True)
                )
            )
            / 2,
        ]
        assert all(
            (
                abs(_actual - _expected) < 2e-11
                for _actual, _expected in zip(_values, _reference, strict=True)
            )
        )
        numeric_checks.append((_mv, _gv, _scale, _pv, _values, _reference))
    return (numeric_checks,)


@app.cell
def _(mo):
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    kinematic_point = mo.ui.dropdown(
        {
            "Zero external momentum": 0,
            "Spacelike momentum": 1,
            "Timelike below threshold": 2,
            "Lighter scalar": 3,
        },
        value="Zero external momentum",
        label="Finite evaluation",
    )
    mo.vstack([subtraction_scheme, kinematic_point])
    return kinematic_point, subtraction_scheme


@app.cell
def _(constants, kinematic_point, math, numeric_checks, subtraction_scheme):
    selected_constants = constants[subtraction_scheme.value]
    _selected_point = numeric_checks[kinematic_point.value]
    M_value, g_value, mu_value, p_value, _native, _reference = _selected_point
    _scheme_shift = (
        0
        if subtraction_scheme.value == "MSbar"
        else g_value**2 * (math.log(4 * math.pi) - 0.5772156649015329) / 2
    )
    _normalization = 1j / (16 * math.pi**2)
    selected_amplitudes = [
        _normalization * (value + (_scheme_shift if _i != 1 else 0))
        for _i, value in enumerate(_native)
    ]
    selected_reference = [
        _normalization * (value + (_scheme_shift if _i != 1 else 0))
        for _i, value in enumerate(_reference)
    ]
    finite_error = max(
        (
            abs(value - expected)
            for value, expected in zip(
                selected_amplitudes, selected_reference, strict=True
            )
        )
    )
    assert finite_error < 1e-11
    return (
        M_value,
        finite_error,
        g_value,
        mu_value,
        p_value,
        selected_amplitudes,
        selected_constants,
        selected_reference,
    )


@app.cell(hide_code=True)
def _(
    M_value,
    finite_error,
    g_value,
    mo,
    mu_value,
    p_value,
    selected_amplitudes,
    selected_constants,
    selected_reference,
):
    mo.vstack(
        [
            mo.md("## Renormalization constants"),
            *[
                mo.hstack([mo.md(_label), value])
                for _label, value in zip(
                    ["Zφ", "Zm", "Zg"], selected_constants, strict=True
                )
            ],
            mo.md("## Finite renormalized amplitudes"),
            mo.ui.table(
                [{"M": M_value, "g": g_value, "μ²": mu_value, "p²": p_value}],
                selection=None,
            ),
            mo.ui.table(
                [
                    {
                        "Amplitude": _label,
                        "Generated + IBP/OneLOop": str(value),
                        "Independent reference": str(expected),
                    }
                    for _label, value, expected in zip(
                        [
                            "Self-energy at p²=0",
                            "Triangle at zero external momenta",
                            "Self-energy at selected p²",
                        ],
                        selected_amplitudes,
                        selected_reference,
                        strict=True,
                    )
                ],
                selection=None,
            ),
            mo.md(
                f"Maximum absolute difference: `{finite_error:.3g}`. The self-energy reference is a direct Feynman-parameter integral. The triangle reference is the zero-momentum simplex integral −g³/(2M). The displayed amplitudes include i/(16π²)."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
