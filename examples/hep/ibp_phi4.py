import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Generated two-loop phi4 renormalization",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Generated two-loop $\phi^4$ renormalization

    [Browse all notebooks](/) · [Unequal-mass bubble IBP](/?file=hep/ibp_bubble.py) · [Two-loop coupling renormalization](/?file=hep/phi4_two_loop_vertex.py)

    Start from `Model.phi4()` and reproduce the [FeynCalc two-loop self-energy example](https://feyncalc.github.io/FeynCalcExamples/Phi4/TwoLoops/Renormalization-SS).
    The generator supplies two bare diagrams, three one-loop counterterm diagrams, and two local two-loop counterterms. Their native factors and numerators enter the shared UV expansion, tensor reduction, and IBP solvers.

    The reference's scalar integrands provide independent checks, with $ig^2$ suppressed:
    $1/(4D_1^2D_2)$ and $1/[6D_1D_2((k_1+k_2+p)^2-M)]$, where $M=m^2>0$.
    Here $D_1=k_1^2-M$, $D_2=k_2^2-M$, $D_3=(k_1+k_2)^2-M$ and $p_2=p^2$.
    Taylor expansion through $p^2$ preserves the UV poles without an infrared singularity.

    Analytic master coefficients and the one-loop renormalization constants are explicit reference inputs, as in FeynCalc. Finite-depth Laporta residuals are not certified as a minimal master basis.
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
    from symbolica.community import tensor as sp
    import json
    from math import prod

    import marimo as mo
    from symbolica import E, Expression, Matrix, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import (
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        TensorReducer,
    )

    _set_namespace("ibp_phi4")
    return (
        E,
        Expression,
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Matrix,
        Model,
        Replacement,
        S,
        Symbol,
        TensorReducer,
        hep,
        json,
        mo,
        prod,
        sp,
    )


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    K,
    Symbol,
    ct_coordinate,
    ct_integral,
    ct_model,
    d,
    den,
    dimension,
    edge_,
    external_kinematics,
    h,
    index,
    k1,
    mass_,
    mass_squared,
    model_coupling,
    model_mass,
    mom_,
    quad_,
    sp,
    tadpole_family,
    zero,
):
    def project_counterterm_diagram(_diagram, _loops):
        """Project one generated insertion, separating loop and local tree contributions."""
        ct_input, tree_ct = zero, zero
        _numerator = ct_model.expand_couplings(_diagram.numerator_expression())
        _numerator = _numerator.contract().to_dots().expand().to_expression()
        if _loops:
            _numerator = _diagram.uv_expansion(
                model_mass, numerator=_numerator
            ).to_expression()
        _numerator = _diagram.momentum_basis().route_expression(_numerator)
        _numerator = (
            _numerator.replace(
                sp.PortPattern.exact(sp.Representation.mink(dimension)),
                sp.PortPattern.exact(sp.Representation.mink(d)),
            )
            .replace(K(0, index), k1(index))
            .replace(model_mass**2, mass_squared)
        )
        for _match in list(_numerator.match(den(edge_, mom_, mass_, quad_))):
            _values = dict(_match)
            _numerator = _numerator.replace(
                den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                tadpole_family.rewrite_numerator(_values[quad_], [ct_coordinate]),
            )
        _factor = (
            _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        _numerator = external_kinematics.apply(_numerator) * _factor
        if _loops:
            _scalar = (
                (
                    tadpole_family.rewrite_numerator(_numerator, [ct_coordinate])
                    / (h * model_coupling)
                )
                .together()
                .expand()
            )
            for _monomial, _coefficient in _scalar.coefficient_list(ct_coordinate):
                _power = -int(
                    (
                        _monomial.derivative(ct_coordinate) * ct_coordinate / _monomial
                    ).together()
                )
                assert _monomial == ct_coordinate ** (-_power)
                ct_input += _coefficient * ct_integral(_power)
        else:
            tree_ct += _numerator / (Symbol.I * h**2)
        return ct_input, tree_ct

    return (project_counterterm_diagram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare dimensions and scalar invariants
    """)
    return


@app.cell
def _(E, Expression, S, sp):
    d, mass_squared, p_squared, eps, coupling = S(
        "d",
        "M",
        "p2",
        "eps",
        "g",
    )
    k1, k2 = (sp.TensorName.vector(name).to_expression() for name in ("k1", "k2"))
    tadpole_squared, vacuum_integral, log_mass, log_4pi = S("T2", "V", "Lm", "L4pi")
    integral = S("I")
    zero = E("0")
    pi = Expression.PI
    return (
        coupling,
        d,
        eps,
        integral,
        k1,
        k2,
        log_4pi,
        log_mass,
        mass_squared,
        p_squared,
        pi,
        tadpole_squared,
        vacuum_integral,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the angular average of a shifted propagator
    """)
    return


@app.cell
def _(E, S, TensorReducer, d, mass_squared, p_squared, sp, zero):
    _q, _p, _scaling, formal_denominator = (
        sp.TensorName.vector("q").to_expression(),
        sp.TensorName.vector("p").to_expression(),
        S("scaling"),
        S("D3"),
    )
    mink, _dot = (sp.Representation.mink, sp.TensorPattern.dot)
    _qc, _pc = (
        _q(sp.PortPattern.exact(sp.Representation.mink(d))),
        _p(sp.PortPattern.exact(sp.Representation.mink(d))),
    )
    _shifted_line = 1 / (
        formal_denominator
        + 2 * _scaling * _dot(_qc, _pc)
        + _scaling**2 * _dot(_pc, _pc)
    )
    averaged_taylor = (
        _shifted_line.series(_scaling, 0, 2).to_expression().replace(_scaling, E("1"))
    )
    averaged_taylor = TensorReducer(d, integrated=[_qc]).reduce(averaged_taylor)
    averaged_taylor = (
        averaged_taylor.replace(_dot(_qc, _qc), formal_denominator + mass_squared)
        .replace(_dot(_pc, _pc), p_squared)
        .expand()
    )
    _expected_taylor = (
        1 / formal_denominator
        + p_squared * (4 / d - 1) / formal_denominator**2
        + 4 * mass_squared * p_squared / (d * formal_denominator**3)
    )
    assert (averaged_taylor - _expected_taylor).together() == zero
    return averaged_taylor, formal_denominator


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## State the independent bare-integrand reference
    """)
    return


@app.cell
def _(Replacement, averaged_taylor, formal_denominator, integral):
    reference_bare_input = (
        integral(2, 1, 0) / 4
        + averaged_taylor.replace_multiple(
            [
                Replacement(formal_denominator ** (-1), integral(1, 1, 1)),
                Replacement(formal_denominator ** (-2), integral(1, 1, 2)),
                Replacement(formal_denominator ** (-3), integral(1, 1, 3)),
            ]
        )
        / 6
    )
    return (reference_bare_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the two-loop vacuum family
    """)
    return


@app.cell
def _(IntegralFamily, Kinematics, d, k1, k2, mass_squared):
    vacuum_kinematics = Kinematics(d, momenta=[k1, k2])
    family = IntegralFamily(
        [k1, k2],
        [],
        [
            vacuum_kinematics.scalar_product(_k, _k) - mass_squared
            for _k in (k1, k2, k1 + k2)
        ],
        kinematics=vacuum_kinematics,
    )
    assert family.is_complete and family.is_independent
    return family, vacuum_kinematics


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the self-energy diagrams
    """)
    return


@app.cell
def _(Model):
    model = Model.phi4()
    particle = model.particle("phi")
    generated = model.process([particle], [particle]).generate_diagrams(
        loops=2,
        max_vertices=2,
        maximum_bridges=0,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(generated.diagrams) == 2
    return generated, model, particle


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Prepare routing and tensor projection
    """)
    return


@app.cell
def _(
    S,
    Symbols,
    TensorReducer,
    d,
    hep,
    k1,
    k2,
    model,
    p_squared,
    sp,
    vacuum_kinematics,
):
    K, P, dim = (
        hep.Kinematics.loop_momentum,
        hep.Kinematics.external_momentum,
        Symbols.dimension,
    )
    model_mass = model.parameter("mass").symbol
    model_coupling = model.parameter("lam").symbol
    index, dimension = S("index_", "dimension_")
    den, edge_, mom_, mass_, quad_ = (
        Symbols.denominator,
        S("edge_"),
        S("mom_"),
        S("mass_"),
        S("quad_"),
    )
    coordinates = S("x1", "x2", "x3")
    external_kinematics = vacuum_kinematics.with_scalar_product(P(0), P(0), p_squared)
    vacuum_reducer = TensorReducer(
        d,
        integrated=[
            k1(sp.PortPattern.exact(sp.Representation.mink(d))),
            k2(sp.PortPattern.exact(sp.Representation.mink(d))),
        ],
    )
    return (
        K,
        P,
        coordinates,
        den,
        dim,
        dimension,
        edge_,
        external_kinematics,
        index,
        mass_,
        model_coupling,
        model_mass,
        mom_,
        quad_,
        vacuum_reducer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Project each bare diagram
    """)
    return


@app.cell
def _(
    K,
    Kinematics,
    S,
    Symbol,
    coordinates,
    d,
    den,
    dim,
    edge_,
    external_kinematics,
    family,
    generated,
    index,
    integral,
    k1,
    k2,
    mass_,
    mass_squared,
    model,
    model_coupling,
    model_mass,
    mom_,
    prod,
    quad_,
    vacuum_reducer,
    zero,
):
    bare_input = zero
    bare_terms = []
    raw_checks = []
    for _diagram in generated.diagrams:
        _numerator = model.expand_couplings(
            _diagram.numerator_expression().to_expression()
        )
        _factor = (
            _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        _raw_family = _diagram.propagator_family(kinematics=Kinematics(d))
        _raw = _factor * _numerator / (Symbol.I * model_coupling**2)
        for _propagator in _raw_family.denominators:
            _raw /= _propagator
        _raw = (
            _raw.replace(K(0, index), k1(index))
            .replace(K(1, index), k2(index))
            .replace(model_mass**2, mass_squared)
        )
        raw_checks.append(_raw)
        _expanded = _diagram.momentum_basis().route_expression(
            _diagram.uv_expansion(model_mass, numerator=_numerator).to_expression()
        )
        _expanded = (
            _expanded.replace(dim, d)
            .replace(K(0, index), k1(index))
            .replace(K(1, index), k2(index))
            .replace(model_mass**2, mass_squared)
        )
        for _match in list(_expanded.match(den(edge_, mom_, mass_, quad_))):
            _values = dict(_match)
            _expanded = _expanded.replace(
                den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                family.rewrite_numerator(_values[quad_], coordinates),
            )
        _scalar = family.rewrite_numerator(
            external_kinematics.apply(vacuum_reducer.reduce(_expanded)), coordinates
        )
        _scalar = (
            (_scalar * _factor / (Symbol.I * model_coupling**2)).together().expand()
        )
        for _monomial, _coefficient in _scalar.coefficient_list(*coordinates):
            _powers = [
                -int((_monomial.derivative(_x) * _x / _monomial).together())
                for _x in coordinates
            ]
            assert _monomial == prod(
                (
                    _x ** (-_power)
                    for _x, _power in zip(coordinates, _powers, strict=True)
                )
            )
            assert not _coefficient.matches(K(S("args___")))
            bare_terms.append((_powers, _coefficient))
            bare_input += _coefficient * integral(*_powers)
    return bare_input, bare_terms, raw_checks


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify each projected integrand
    """)
    return


@app.cell
def _(
    P,
    bare_input,
    external_kinematics,
    family,
    k1,
    k2,
    mass_squared,
    raw_checks,
    reference_bare_input,
    zero,
):
    _raw_reference = [
        1 / (4 * family.denominators[0] ** 2 * family.denominators[1]),
        1
        / (
            6
            * family.denominators[0]
            * family.denominators[1]
            * (
                external_kinematics.scalar_product(k1 + k2 + P(0), k1 + k2 + P(0))
                - mass_squared
            )
        ),
    ]
    assert all(
        sum(
            external_kinematics.apply(_raw - _reference).together() == zero
            for _raw in raw_checks
        )
        == 1
        for _reference in _raw_reference
    )
    assert (bare_input - reference_bare_input).together() == zero
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the IBP identities and exact family symmetries
    """)
    return


@app.cell
def _(IBPFamily, family, k1, k2):
    ibp = IBPFamily(family, name="phi4_vacuum")
    _identities = ibp.ibp_identities()
    assert len(_identities) == 4
    symmetries = [
        family.mapping_to(family, _images)
        for _images in (
            [k1, k2],
            [k2, k1],
            [-k1 - k2, k2],
            [k1, -k1 - k2],
            [k2, -k1 - k2],
            [-k1 - k2, k1],
        )
    ]
    assert all(_mapping is not None for _mapping in symmetries)
    assert len({tuple(_mapping.denominator_map) for _mapping in symmetries}) == 6
    return ibp, symmetries


@app.cell(hide_code=True)
def _(bare_input, family, generated, mo):
    mo.vstack(
        [
            mo.md("## Generated bare diagrams"),
            mo.hstack(generated.diagrams),
            mo.md("**Vacuum integral family**"),
            family,
            mo.md("**Generated UV-expanded integral input**"),
            bare_input,
            mo.md(
                "The unexpanded integrands and Taylor coefficients match the reference exactly. Six verified momentum maps identify equivalent vacuum integrals."
            ),
        ]
    )
    return


@app.cell
def _(
    Replacement,
    bare_input,
    bare_terms,
    d,
    ibp,
    integral,
    mass_squared,
    symmetries,
    tadpole_squared,
    vacuum_integral,
    zero,
):
    _targets = sorted({tuple(_powers) for _powers, _coefficient in bare_terms})
    assert set(_targets) == {(2, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3)}
    laporta = ibp.reduce_laporta(_targets, max_depth=2)
    assert laporta.stats["rows"] > 0
    _canonical = {
        tuple(_powers): min(
            tuple(_mapping.map_powers(_powers)) for _mapping in symmetries
        )
        for _powers in laporta.residuals
    }
    reference_basis = {(0, 1, 1): tadpole_squared, (1, 1, 1): vacuum_integral}
    _raw_reductions = {tuple(target): laporta.reduce(target) for target in _targets}
    reduced = {}
    for target, _terms in _raw_reductions.items():
        assert all((tuple(_powers) in _canonical for _powers, _coefficient in _terms))
        assert all(
            (
                _canonical[tuple(_powers)] in reference_basis
                for _powers, _coefficient in _terms
            )
        )
        reduced[target] = sum(
            (
                _coefficient * reference_basis[_canonical[tuple(_powers)]]
                for _powers, _coefficient in _terms
            ),
            zero,
        ).together()
    assert (
        reduced[2, 1, 0] - (d - 2) * tadpole_squared / (2 * mass_squared)
    ).together() == zero
    assert (
        reduced[1, 1, 2] - (d - 3) * vacuum_integral / (3 * mass_squared)
    ).together() == zero
    assert (
        reduced[1, 1, 3]
        - (d - 8) * (d - 3) * vacuum_integral / (18 * mass_squared**2)
        - (d - 2) ** 2 * tadpole_squared / (12 * mass_squared**3)
    ).together() == zero
    bare_reduced = bare_input.replace_multiple(
        [Replacement(integral(*target), value) for target, value in reduced.items()]
    ).together()
    return bare_reduced, laporta, reduced


@app.cell(hide_code=True)
def _(bare_reduced, integral, laporta, mo, reduced):
    mo.vstack(
        [
            mo.md("## Native IBP reductions"),
            mo.md(
                f"The finite Laporta search used {laporta.stats['rows']} equations from {laporta.stats['seeds']} seeds."
            ),
            *[
                mo.hstack([integral(*target), mo.md("$\\longrightarrow$"), value])
                for target, value in reduced.items()
            ],
            mo.md("**Reduced bare self-energy**"),
            bare_reduced,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Analytic reference inputs

    Use $d=4-2\epsilon$, $L_m=\log M$ and $L_{4\pi}=\log(4\pi)$.
    The [one-loop tadpole](https://raw.githubusercontent.com/FeynCalc/feyncalc/master/FeynCalc/Examples/MasterIntegrals/Tadpoles/tad1LxFx1x1xxEp999x.m)
    and [equal-mass two-loop vacuum](https://raw.githubusercontent.com/FeynCalc/feyncalc/master/FeynCalc/Examples/MasterIntegrals/Tadpoles/tad2LxFx111x111xxEp1x.m)
    give the needed coefficients:
    $$T=M[1/\epsilon+1-L_m+O(\epsilon)],\qquad
      V=M[3/(2\epsilon^2)+(9/2-3L_m)/\epsilon+O(1)].$$

    These analytic values are supplied, not evaluated by the reducer. Below,
    amplitudes are divided by $ig^2/(16\pi^2)^2$. Restoring the two-loop measure
    contributes $-(1+2\epsilon L_{4\pi})$ through the required order.
    """)
    return


@app.cell
def _(
    E,
    bare_reduced,
    d,
    eps,
    log_4pi,
    log_mass,
    mass_squared,
    mo,
    p_squared,
    tadpole_squared,
    vacuum_integral,
    zero,
):
    tadpole_poles = mass_squared * (1 / eps + 1 - log_mass)
    tadpole_squared_poles = mass_squared**2 * (1 / eps**2 + 2 * (1 - log_mass) / eps)
    vacuum_poles = mass_squared * (E("3/2") / eps**2 + (E("9/2") - 3 * log_mass) / eps)
    bare_uv = (
        (
            -(1 + 2 * eps * log_4pi)
            * bare_reduced.replace(d, 4 - 2 * eps)
            .replace(tadpole_squared, tadpole_squared_poles)
            .replace(vacuum_integral, vacuum_poles)
        )
        .series(eps, 0, -1)
        .to_expression()
        .expand()
    )
    _expected_bare = (
        -mass_squared / (2 * eps**2)
        + (mass_squared * (log_mass - 1 - log_4pi) + p_squared / 24) / eps
    )
    assert (bare_uv - _expected_bare).together() == zero
    mo.vstack([mo.md("**Bare two-loop UV poles**"), bare_uv])
    return bare_uv, tadpole_poles


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generated counterterms

    Expand the bare factors $Z_\phi$, $Z_\phi Z_m$, and $Z_g Z_\phi^2$ in the kinetic, mass, and quartic operators. Keep the one-loop field constant symbolic while generating the three one-loop insertions. The two-point rules at second order supply the final linear system.

    Parametric IBP derives the doubled-tadpole recurrence. Then insert $z_\phi^{(1)}=0$, $z_g^{(1)}=3/(2\epsilon)$ and $z_m^{(1)}=1/(2\epsilon)$, in units of $g/(16\pi^2)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Derive the doubled-tadpole recurrence
    """)
    return


@app.cell
def _(IBPFamily, IntegralFamily, Kinematics, d, k1, mass_squared, zero):
    _tadpole_kin = Kinematics(d, momenta=[k1])
    tadpole_family = IntegralFamily(
        [k1],
        [],
        [_tadpole_kin.scalar_product(k1, k1) - mass_squared],
        kinematics=_tadpole_kin,
    )
    parametric = IBPFamily(tadpole_family, name="phi4_tadpole").solve_parametric(
        [True], max_depth=1
    )
    assert parametric.rules
    _tadpole_terms = parametric.reduce([2])
    assert len(_tadpole_terms) == 1 and _tadpole_terms[0][0] == [1]
    tadpole_coefficient = _tadpole_terms[0][1]
    assert (tadpole_coefficient - (d - 2) / (2 * mass_squared)).together() == zero
    return parametric, tadpole_coefficient, tadpole_family


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the local counterterm operators
    """)
    return


@app.cell
def _(S, eps):
    zg1, zm1 = (3 / (2 * eps), 1 / (2 * eps))
    h, field1, mass1, vertex1, field2, mass2, vertex2 = S(
        "h",
        "field1",
        "mass1",
        "vertex1",
        "field2",
        "mass2",
        "vertex2",
    )
    Zfield = 1 + h * field1 + h**2 * field2
    Zmass = 1 + h * mass1 + h**2 * mass2
    Zvertex = 1 + h * vertex1 + h**2 * vertex2
    return (
        Zfield,
        Zmass,
        Zvertex,
        field1,
        field2,
        h,
        mass1,
        mass2,
        vertex1,
        zg1,
        zm1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Add the counterterms to the model
    """)
    return


@app.cell
def _(
    Model,
    S,
    Symbol,
    Zfield,
    Zmass,
    Zvertex,
    h,
    json,
    model,
    model_coupling,
    model_mass,
    particle,
):
    _specification = json.loads(model.to_json())
    _specification["orders"].append(
        {"name": "CT", "expansion_order": 2, "hierarchy": 1}
    )
    for _label, _valence, _lorentz, _bare_factor in [
        ("kinetic", 2, "P(dummy(1),1)*P(dummy(1),1)", Symbol.I * (Zfield - 1)),
        ("mass", 2, "1", -Symbol.I * model_mass**2 * (Zfield * Zmass - 1)),
        ("quartic", 4, "1", -Symbol.I * model_coupling * (Zvertex * Zfield**2 - 1)),
    ]:
        _specification["lorentz_structures"].append(
            {"name": "CT_L_" + _label, "spins": [1] * _valence, "structure": _lorentz}
        )
        for _order in [1, 2]:
            _name = f"CT_{_label}_{_order}"
            _specification["couplings"].append(
                {
                    "name": _name,
                    "expression": repr(
                        _bare_factor.expand().coefficient(h**_order) * h**_order
                    ),
                    "orders": [["SCALAR", 1 if _valence == 4 else 0], ["CT", _order]],
                    "value": None,
                }
            )
            _specification["vertex_rules"].append(
                {
                    "name": _name,
                    "particles": [particle.name] * _valence,
                    "color_structures": ["1"],
                    "lorentz_structures": ["CT_L_" + _label],
                    "couplings": [[_name]],
                }
            )
    ct_model = Model.from_json(json.dumps(_specification))
    ct_integral, ct_coordinate = S("J", "y")
    return ct_coordinate, ct_integral, ct_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and project the counterterm graphs
    """)
    return


@app.cell
def _(ct_model, particle, project_counterterm_diagram, zero):
    ct_diagrams = {}
    ct_input, tree_ct = (zero, zero)
    for _loops, _ct_order in [(1, 1), (0, 2)]:
        _result = ct_model.process([particle.name], [particle.name]).generate_diagrams(
            loops=_loops,
            max_vertices=2 if _loops else 1,
            coupling_orders={"CT": _ct_order},
            maximum_bridges=0,
            self_energy=None,
            tadpoles=None,
            zero_snails=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_result.diagrams) == (3 if _loops else 2)
        ct_diagrams[_loops] = _result.diagrams
        for _diagram in _result.diagrams:
            _loop_term, _tree_term = project_counterterm_diagram(_diagram, _loops)
            ct_input += _loop_term
            tree_ct += _tree_term
    return ct_diagrams, ct_input, tree_ct


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the unintegrated counterterms
    """)
    return


@app.cell
def _(
    ct_input,
    ct_integral,
    field1,
    field2,
    mass1,
    mass2,
    mass_squared,
    p_squared,
    tree_ct,
    vertex1,
    zero,
):
    assert (
        ct_input
        - (vertex1 + field1) * ct_integral(1) / 2
        - mass_squared * mass1 * ct_integral(2) / 2
    ).together() == zero
    assert (
        tree_ct - p_squared * field2 + mass_squared * (mass2 + field2 + field1 * mass1)
    ).together() == zero
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce the tadpole insertions
    """)
    return


@app.cell
def _(
    ct_input,
    ct_integral,
    field1,
    mass1,
    tadpole_coefficient,
    vertex1,
    zero,
    zg1,
    zm1,
):
    ct_reduced = ct_input.replace(ct_integral(2), tadpole_coefficient * ct_integral(1))
    ct_reduced = (
        ct_reduced.replace(field1, zero).replace(vertex1, zg1).replace(mass1, zm1)
    )
    return (ct_reduced,)


@app.cell(hide_code=True)
def _(ct_diagrams, ct_input, mo, parametric, tadpole_coefficient, tree_ct):
    mo.vstack(
        [
            mo.md("**One-loop counterterm diagrams**"),
            mo.hstack(ct_diagrams[1]),
            mo.md("**Generated counterterm integral combination**"),
            ct_input,
            mo.md("**Parametric doubled-tadpole coefficient**"),
            tadpole_coefficient,
            mo.accordion(
                {
                    "Recurrence conditions": mo.vstack(
                        [
                            mo.md(
                                f"Nonzero conditions: {_rule.nonzero_conditions}; exceptional index loci: {_rule.exceptions}"
                            )
                            for _rule in parametric.rules
                        ]
                    )
                }
            ),
            mo.md("**Local two-loop counterterms**"),
            mo.hstack(ct_diagrams[0]),
            tree_ct,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Combine the bare and insertion poles
    """)
    return


@app.cell
def _(
    E,
    bare_uv,
    ct_integral,
    ct_reduced,
    d,
    eps,
    log_4pi,
    log_mass,
    mass_squared,
    p_squared,
    tadpole_poles,
    zero,
):
    counterterm_uv = (
        (
            (1 + eps * log_4pi)
            * ct_reduced.replace(d, 4 - 2 * eps).replace(ct_integral(1), tadpole_poles)
        )
        .series(eps, 0, -1)
        .to_expression()
        .expand()
    )
    _expected_counterterm = (
        mass_squared / eps**2 + mass_squared * (E("3/4") - log_mass + log_4pi) / eps
    )
    assert (counterterm_uv - _expected_counterterm).together() == zero
    loop_uv = (bare_uv + counterterm_uv).expand()
    assert (
        loop_uv
        - mass_squared / (2 * eps**2)
        + mass_squared / (4 * eps)
        - p_squared / (24 * eps)
    ).together() == zero
    return counterterm_uv, loop_uv


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the local pole-cancellation equations
    """)
    return


@app.cell
def _(
    Matrix,
    field1,
    field2,
    loop_uv,
    mass2,
    mass_squared,
    p_squared,
    tree_ct,
    zero,
):
    _normalized_tree_ct = tree_ct.replace(field1, zero).expand()
    _ct_rows = [
        _normalized_tree_ct.coefficient(p_squared),
        (_normalized_tree_ct.replace(p_squared, zero) / mass_squared)
        .together()
        .expand(),
    ]
    ct_matrix = Matrix.from_linear(
        2,
        2,
        [
            _row.coefficient(_unknown)
            for _row in _ct_rows
            for _unknown in (field2, mass2)
        ],
    )
    _rhs = Matrix.vec(
        [
            -loop_uv.coefficient(p_squared),
            -loop_uv.replace(p_squared, zero) / mass_squared,
        ]
    )
    solution = ct_matrix.solve(_rhs)
    zphi2, zm2 = (solution[_row, 0].to_expression().expand() for _row in range(2))
    assert (
        loop_uv + _normalized_tree_ct.replace(field2, zphi2).replace(mass2, zm2)
    ).together() == zero
    assert (
        loop_uv + p_squared * zphi2 - mass_squared * (zm2 + zphi2)
    ).together() == zero
    return ct_matrix, zm2, zphi2


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Restore the coupling normalization
    """)
    return


@app.cell
def _(coupling, eps, pi, zero, zm1, zm2, zphi2):
    loop_coupling = coupling / (16 * pi**2)
    zphi = (1 + loop_coupling**2 * zphi2).expand()
    zm = (1 + loop_coupling * zm1 + loop_coupling**2 * zm2).expand()
    assert (zphi - 1 + coupling**2 / (6144 * pi**4 * eps)).together() == zero
    assert (
        zm
        - 1
        - coupling / (32 * pi**2 * eps)
        - coupling**2 / (512 * pi**4 * eps**2)
        + 5 * coupling**2 / (6144 * pi**4 * eps)
    ).together() == zero
    return zm, zphi


@app.cell(hide_code=True)
def _(counterterm_uv, ct_matrix, loop_uv, mo, zm, zphi):
    mo.vstack(
        [
            mo.md("## Solve the generated counterterm system"),
            ct_matrix,
            mo.md("**One-loop counterterm poles**"),
            counterterm_uv,
            mo.md("**Sum before local subtraction**"),
            loop_uv,
            mo.md("**Field renormalization**"),
            zphi,
            mo.md("**Mass-squared renormalization**"),
            zm,
            mo.md(
                "With $a=g/(16\\pi^2)$ these are $Z_\\phi=1-a^2/(24\\epsilon)$ and $Z_m=1+a/(2\\epsilon)+a^2[1/(2\\epsilon^2)-5/(24\\epsilon)]$. The generated local counterterms cancel every pole exactly."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
