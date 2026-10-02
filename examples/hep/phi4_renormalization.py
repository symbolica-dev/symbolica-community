import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Generated one-loop phi4 renormalization",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop $\phi^4$ renormalization
    [Browse all notebooks](/) · [Cubic scalar](/?file=hep/phi3_renormalization.py) · [Yukawa theories](/?file=hep/yukawa_renormalization.py) · [Finite scattering](/?file=hep/phi4_scattering.py) · [Two-loop self-energy](/?file=hep/ibp_phi4.py)

    Generate the tadpole, three four-point channels, and local counterterms from `Model.phi4()`. Native numerators, graph factors and momentum routing determine the coefficients of the scalar masters. Their UV poles and the generated counterterm matrix reproduce the [FeynCalc MS/MS̄ example](https://feyncalc.github.io/FeynCalcExamples/Phi4/OneLoop/Renormalization).

    The Lagrangian is $\mathcal L=(\partial\phi)^2/2-m^2\phi^2/2-g\phi^4/4!$, and $Z_m$ renormalizes $m^2$. Write $Z_j=1+[g/(16\pi^2)]\delta_j$. The ordinary amplitude contains the factor $i$ from loop integration. OneLOop supplies $A_0$ and $B_0$ in a common master normalization.
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
    import json

    import marimo as mo
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import IntegralFamily, Kinematics, Model, oneloop
    from symbolica.community.tensor import TensorExpression

    _set_namespace("phi4_one")
    return (
        E,
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
        mo,
        oneloop,
    )


@app.cell(hide_code=True)
def _():
    from symbolica.community.hep import Symbols

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
    ## Declare the quartic model and kinematics
    """)
    return


@app.cell
def _(E, Kinematics, Model, S, Symbol, Symbols, hep):
    model = Model.phi4()
    particle = model.particle("phi")
    mass = model.parameter("mass").symbol
    coupling = model.parameter("lam").symbol
    M, s, t, u, mu2, p2, eps = S(
        "M",
        "s",
        "t",
        "u",
        "mu2",
        "p2",
        "eps",
    )
    Q, K, P = (
        Symbols.edge_momentum,
        hep.Kinematics.loop_momentum,
        hep.Kinematics.external_momentum,
    )
    dimension = S("D")
    zero, one, pi = (E("0"), E("1"), Symbol.PI)
    kinematics = Kinematics.mandelstam([P(_i) for _i in range(4)], [M] * 4, [s, t, u])
    routing_kinematics = Kinematics(momenta=[K(0), *[P(_i) for _i in range(4)]])
    return (
        K,
        M,
        P,
        Q,
        coupling,
        dimension,
        eps,
        kinematics,
        mass,
        model,
        mu2,
        one,
        p2,
        particle,
        pi,
        routing_kinematics,
        s,
        t,
        u,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate tree and one-loop diagrams
    """)
    return


@app.cell
def _(Symbol, coupling, model, particle):
    options = {
        "maximum_bridges": 0,
        "self_energy": None,
        "tadpoles": None,
        "zero_snails": None,
        "numerator_grouping": None,
        "progress": None,
    }
    diagrams = {}
    for _label, _legs, _loops, _vertices in [
        ("self_energy", 1, 1, 1),
        ("vertex", 2, 1, 2),
        ("tree", 2, 0, 1),
    ]:
        diagrams[_label] = (
            model.process([particle] * _legs, [particle] * _legs)
            .generate_diagrams(loops=_loops, max_vertices=_vertices, **options)
            .diagrams
        )
    assert [len(diagrams[_k]) for _k in ("self_energy", "vertex", "tree")] == [1, 3, 1]
    _tree = diagrams["tree"][0]
    tree_amplitude = (
        model.expand_couplings(_tree.numerator_expression().to_expression())
        * _tree.overall_factor_expression(evaluate=True)
        * _tree.numerator_prefactor_expression()
    )
    assert tree_amplitude == -Symbol.I * coupling
    return diagrams, options


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Identify the three bubble channels
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
    coupling,
    diagrams,
    dimension,
    kinematics,
    mass,
    model,
    oneloop,
    routing_kinematics,
    s,
    t,
    u,
    zero,
):
    channel_coefficients, channel_reductions = ({}, {})
    for _diagram in diagrams["vertex"]:
        _basis = _diagram.momentum_basis()
        _family = _diagram.propagator_family(kinematics=kinematics)
        _shifts = []
        for _edge, _denominator in zip(
            _diagram.internal_edges, _family.denominators, strict=True
        ):
            _sign = _basis.edge_signatures[_edge.id].loops[0]
            assert abs(_sign) == 1
            _shift = (_basis.route_expression(Q(_edge.id)) / _sign - K(0)).expand()
            _reconstructed = (
                kinematics.apply(
                    routing_kinematics.scalar_product(K(0) + _shift, K(0) + _shift)
                )
                - M
            )
            assert (_denominator.replace(mass**2, M) - _reconstructed).expand() == zero
            _shifts.append(_shift)
        _invariant = kinematics.scalar_product(
            _shifts[0] - _shifts[1], _shifts[0] - _shifts[1]
        )
        assert _invariant in (s, t, u) and _invariant not in channel_coefficients
        _coefficient = (
            model.expand_couplings(_diagram.numerator_expression().to_expression())
            * _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        assert _coefficient == coupling**2 / 2
        _master_kinematics = Kinematics(
            dimension, momenta=[K(0), P(0)]
        ).with_scalar_product(P(0), P(0), _invariant)
        _master_family = IntegralFamily(
            [K(0)],
            [P(0)],
            [
                _master_kinematics.scalar_product(_momentum, _momentum) - M
                for _momentum in [K(0), K(0) + P(0)]
            ],
            kinematics=_master_kinematics,
        )
        _reduction = oneloop.reduce(_master_family, [1, 1])
        assert len(_reduction.terms) == 1
        channel_coefficients[_invariant] = _coefficient
        channel_reductions[_invariant] = _reduction
    assert set(channel_coefficients) == {s, t, u}
    return channel_coefficients, channel_reductions


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate their poles and finite parts
    """)
    return


@app.cell
def _(
    M,
    Replacement,
    channel_coefficients,
    channel_reductions,
    coupling,
    mu2,
    one,
    oneloop,
    zero,
):
    vertex_finite, vertex_pole = (zero, zero)
    for _invariant, _reduction in channel_reductions.items():
        _coefficients = oneloop.reduction_coefficients(_reduction, mu2)
        _master = _reduction.terms[0][1].to_expression(mu2)
        _pole_expression = oneloop.get_expression(_master, coefficient=-1)
        _pole = oneloop.select_branch(
            _pole_expression, [Replacement(_invariant, one), Replacement(M, one)]
        )
        assert _pole == one
        _weight = channel_coefficients[_invariant] / coupling**2
        vertex_pole += _weight * _pole
        vertex_finite += _weight * _coefficients[0]
    return vertex_finite, vertex_pole


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate the tadpole
    """)
    return


@app.cell
def _(
    E,
    IntegralFamily,
    K,
    Kinematics,
    M,
    coupling,
    diagrams,
    dimension,
    model,
    mu2,
    oneloop,
    vertex_finite,
    vertex_pole,
    zero,
):
    _self_diagram = diagrams["self_energy"][0]
    _self_numerator = (
        model.expand_couplings(_self_diagram.numerator_expression().to_expression())
        * _self_diagram.overall_factor_expression(evaluate=True)
        * _self_diagram.numerator_prefactor_expression()
    )
    assert _self_numerator == coupling / 2
    _self_family = _self_diagram.propagator_family()
    assert len(_self_family.denominators) == 1
    _vacuum = Kinematics(dimension, momenta=[K(0)])
    _tadpole_family = IntegralFamily(
        [K(0)], [], [_vacuum.scalar_product(K(0), K(0)) - M], kinematics=_vacuum
    )
    _self_reduction = oneloop.reduce(_tadpole_family, [1])
    _self_master = _self_reduction.terms[0][1].to_expression(mu2)
    _self_pole = (
        _self_numerator
        / coupling
        * oneloop.get_expression(_self_master, coefficient=-1)
    )
    _self_finite = (
        _self_numerator
        / coupling
        * oneloop.reduction_coefficients(_self_reduction, mu2)[0]
    )
    assert (_self_pole - M / 2).together() == zero
    assert (vertex_pole - E("3/2")).together() == zero
    loop_poles = {"self_energy": _self_pole, "vertex": vertex_pole}
    finite_parts = {"self_energy": _self_finite, "vertex": vertex_finite}
    return (loop_poles,)


@app.cell(hide_code=True)
def _(
    channel_coefficients,
    channel_reductions,
    diagrams,
    loop_poles,
    mo,
    s,
    t,
    u,
    zero,
):
    mo.vstack(
        [
            mo.md("## Generated one-loop diagrams"),
            mo.hstack(diagrams["self_energy"]),
            mo.hstack(diagrams["vertex"]),
            mo.md("**Scalar four-point master combination, before the loop measure**"),
            sum(
                (
                    channel_coefficients[_q] * channel_reductions[_q].to_expression()
                    for _q in (s, t, u)
                ),
                zero,
            ),
            mo.md("**UV coefficients**"),
            mo.hstack([loop_poles["self_energy"], loop_poles["vertex"]]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Match generated local counterterms
    Expand the bare kinetic, mass and quartic factors $Z_\phi$, $Z_\phi Z_m$ and $Z_gZ_\phi^2$. The three generated local diagrams supply a linear system for $\delta_\phi,\delta_m,\delta_g$. No renormalization constants are supplied as inputs.

    The common loop measure has $\Delta=1/\epsilon+\log(4\pi)-\gamma_E$. MS subtracts its pole; MS̄ subtracts the complete $\Delta$. The calculation checks both schemes and their finite difference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the local counterterm vertices
    """)
    return


@app.cell
def _(Model, S, Symbol, coupling, json, mass, model, particle):
    h, delta_field, delta_mass, delta_vertex = S(
        "h",
        "delta_field",
        "delta_mass",
        "delta_vertex",
    )
    _Zfield, _Zmass, _Zvertex = (
        1 + h * _x for _x in (delta_field, delta_mass, delta_vertex)
    )
    _specification = json.loads(model.to_json())
    _specification["orders"].append(
        {"name": "CT", "expansion_order": 1, "hierarchy": 1}
    )
    for _label, _valence, _lorentz, _factor in [
        ("kinetic", 2, "P(dummy(1),1)*P(dummy(1),1)", Symbol.I * (_Zfield - 1)),
        ("mass", 2, "1", -Symbol.I * mass**2 * (_Zfield * _Zmass - 1)),
        ("quartic", 4, "1", -Symbol.I * coupling * (_Zvertex * _Zfield**2 - 1)),
    ]:
        _name = "CT_" + _label
        _specification["lorentz_structures"].append(
            {"name": _name, "spins": [1] * _valence, "structure": _lorentz}
        )
        _specification["couplings"].append(
            {
                "name": _name,
                "expression": repr(_factor.expand().coefficient(h) * h),
                "orders": [["SCALAR", int(_valence == 4)], ["CT", 1]],
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
    return ct_model, delta_field, delta_mass, delta_vertex, h


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the counterterm amplitudes
    """)
    return


@app.cell
def _(
    Kinematics,
    M,
    P,
    Symbol,
    coupling,
    ct_model,
    h,
    mass,
    one,
    options,
    p2,
    particle,
    zero,
):
    ct_diagrams, ct_amplitudes = ({}, {})
    _self_kinematics = Kinematics().with_scalar_product(P(0), P(0), p2)
    for _label, _legs, _count in [("self_energy", 1, 2), ("vertex", 2, 1)]:
        _result = ct_model.process(
            [particle.name] * _legs, [particle.name] * _legs
        ).generate_diagrams(
            loops=0, max_vertices=1, coupling_orders={"CT": 1}, **options
        )
        assert len(_result.diagrams) == _count
        ct_diagrams[_label] = _result.diagrams
        _amplitude = zero
        for _diagram in _result.diagrams:
            _numerator = ct_model.expand_couplings(_diagram.numerator_expression())
            _numerator = _numerator.contract().to_dots().expand().to_expression()
            _numerator = _diagram.momentum_basis().route_expression(_numerator)
            _amplitude += (
                _self_kinematics.apply(_numerator)
                * _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
            )
        ct_amplitudes[_label] = (
            (
                _amplitude.replace(mass**2, M)
                / (Symbol.I * h * (coupling if _legs == 2 else one))
            )
            .together()
            .expand()
        )
    return ct_amplitudes, ct_diagrams


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the three renormalization conditions
    """)
    return


@app.cell
def _(
    E,
    M,
    Matrix,
    ct_amplitudes,
    delta_field,
    delta_mass,
    delta_vertex,
    loop_poles,
    p2,
    zero,
):
    _rows = [
        ct_amplitudes["self_energy"].coefficient(p2),
        (ct_amplitudes["self_energy"].replace(p2, zero) / M).together().expand(),
        ct_amplitudes["vertex"],
    ]
    unknowns = [delta_field, delta_mass, delta_vertex]
    ct_matrix = Matrix.from_linear(
        3, 3, [_row.coefficient(_x) for _row in _rows for _x in unknowns]
    )
    _rhs = Matrix.vec([zero, -loop_poles["self_energy"] / M, -loop_poles["vertex"]])
    _solved = ct_matrix.solve(_rhs)
    residues = [_solved[_i, 0].to_expression() for _i in range(3)]
    assert residues == [zero, E("1/2"), E("3/2")]
    return ct_matrix, residues, unknowns


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify MS and MSbar cancellation
    """)
    return


@app.cell
def _(
    Replacement,
    S,
    ct_amplitudes,
    eps,
    loop_poles,
    residues,
    unknowns,
    zero,
):
    L, gamma_e = S("log4pi", "gamma_E")
    renormalization_constants = {}
    for _scheme, _subtraction in [("MS", 1 / eps), ("MSbar", 1 / eps + L - gamma_e)]:
        _constants = [_x * _subtraction for _x in residues]
        renormalization_constants[_scheme] = _constants
        _replacement = [
            Replacement(_x, _y) for _x, _y in zip(unknowns, _constants, strict=True)
        ]
        for _label in loop_poles:
            _total = (
                loop_poles[_label] * (1 / eps + L - gamma_e)
                + ct_amplitudes[_label].replace_multiple(_replacement)
            ).together()
            _expected = (
                zero if _scheme == "MSbar" else loop_poles[_label] * (L - gamma_e)
            )
            assert (_total - _expected).together() == zero
    return (renormalization_constants,)


@app.cell(hide_code=True)
def _(ct_diagrams, ct_matrix, mo, residues):
    mo.vstack(
        [
            mo.hstack(ct_diagrams["self_energy"]),
            mo.hstack(ct_diagrams["vertex"]),
            mo.md("**Generated matching matrix**"),
            ct_matrix,
            mo.md("**Solved pole residues**"),
            mo.hstack(residues),
        ]
    )
    return


@app.cell
def _(mo):
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    mo.vstack([subtraction_scheme])
    return (subtraction_scheme,)


@app.cell
def _(coupling, mo, pi, renormalization_constants, subtraction_scheme):
    selected_constants = renormalization_constants[subtraction_scheme.value]
    mo.vstack(
        [
            mo.md("**Field, mass-squared and coupling renormalization constants**"),
            *[
                mo.hstack([mo.md(_label), 1 + coupling * _value / (16 * pi**2)])
                for _label, _value in zip(
                    ["Zφ", "Zm", "Zg"], selected_constants, strict=True
                )
            ],
            mo.md(
                "All UV poles cancel against the generated counterterms. In MS the finite amplitude retains the loop residue times log(4π)−γE."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
