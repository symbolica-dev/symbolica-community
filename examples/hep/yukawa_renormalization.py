import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Scalar and pseudoscalar Yukawa renormalization",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Scalar and pseudoscalar Yukawa renormalization
    [Browse notebooks](/) · [Quartic scalar](/?file=hep/phi4_renormalization.py) · [Cubic scalar](/?file=hep/phi3_renormalization.py)

    Reproduce both [scalar](https://feyncalc.github.io/FeynCalcExamples/YukawaS/OneLoop/Renormalization) and [pseudoscalar](https://feyncalc.github.io/FeynCalcExamples/YukawaPS/OneLoop/Renormalization) references using one calculation. The interactions are $-g\phi\bar\psi\psi$ or $-ig\phi\bar\psi\gamma_5\psi$, together with $-\lambda\phi^4/4!$.

    Reuse `Model.phi4()` and the Standard Model's massive Dirac propagators through the existing model-definition API. For each interaction, the generator produces 13 bare diagrams and six local counterterms. Shared graph UV expansion, Spenso Dirac traces, covariant tensor reduction and native IBP determine all poles.

    Take $D=4-2\epsilon$. The scalar mass renormalizes as $m_\phi^2 Z_M$ and the fermion mass as $m_\psi Z_m$. These are the two-, three- and four-point functions in the references; vacuum and odd-scalar operators are separate. The pseudoscalar checks concern these leading UV poles, with paired gamma-five factors in the projected traces, and do not address anomalous axial traces.
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
    import copy
    import json

    import marimo as mo
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import (
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        TensorReducer,
        oneloop,
    )
    from symbolica.community.tensor import TensorExpression

    _set_namespace("yukawa")
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
        TensorReducer,
        copy,
        hep,
        json,
        mo,
        oneloop,
        sp,
    )


@app.cell(hide_code=True)
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    Model,
    Symbol,
    ZM,
    Zg,
    Zlam,
    Zm,
    Zphi,
    Zpsi,
    copy,
    h,
    json,
    lam,
    mass,
    scalar_mass,
    specification,
):
    def build_yukawa_model(_variant, _lorentz, _local_coupling):
        """Build the chosen interaction and its local counterterm operators."""
        _definition = copy.deepcopy(specification)
        _definition["name"] = "yukawa_" + _variant.lower()
        _definition["lorentz_structures"].append(
            {"name": "YUKAWA", "spins": [2, 2, 1], "structure": _lorentz}
        )
        _definition["couplings"].append(
            {
                "name": "YUKAWA",
                "expression": repr(_local_coupling),
                "orders": [["YUKAWA", 1]],
                "value": None,
            }
        )
        _definition["vertex_rules"].append(
            {
                "name": "YUKAWA",
                "particles": ["e+", "e-", "phi"],
                "color_structures": ["1"],
                "lorentz_structures": ["YUKAWA"],
                "couplings": [["YUKAWA"]],
            }
        )
        _model = Model.from_json(json.dumps(_definition))
        _fermion, _scalar = (_model.particle("e-"), _model.particle("phi"))
        _ct_definition = json.loads(_model.to_json())
        _ct_definition["orders"].append(
            {"name": "CT", "expansion_order": 1, "hierarchy": 1}
        )
        for _label, _particles, _structure, _factor in [
            (
                "fermion_kinetic",
                ["e+", "e-"],
                "Gamma(dummy(1),idx(1,1),idx(1,2))*P(dummy(1),2)",
                Symbol.I * (Zpsi - 1),
            ),
            (
                "fermion_mass",
                ["e+", "e-"],
                "Identity(idx(1,1),idx(1,2))",
                -Symbol.I * mass * (Zpsi * Zm - 1),
            ),
            (
                "scalar_kinetic",
                ["phi"] * 2,
                "P(dummy(1),1)*P(dummy(1),1)",
                Symbol.I * (Zphi - 1),
            ),
            (
                "scalar_mass",
                ["phi"] * 2,
                "1",
                -Symbol.I * scalar_mass**2 * (Zphi * ZM - 1),
            ),
            (
                "vertex",
                ["e+", "e-", "phi"],
                _lorentz,
                _local_coupling * (Zpsi * Zg * Zphi.sqrt() - 1),
            ),
            ("quartic", ["phi"] * 4, "1", -Symbol.I * lam * (Zlam * Zphi**2 - 1)),
        ]:
            _name = "CT_" + _label
            _ct_definition["lorentz_structures"].append(
                {
                    "name": _name,
                    "spins": [_model.particle(_p).spin for _p in _particles],
                    "structure": _structure,
                }
            )
            _ct_definition["couplings"].append(
                {
                    "name": _name,
                    "expression": repr(_factor.series(h, 0, 1).to_expression()),
                    "orders": [["CT", 1]],
                    "value": None,
                }
            )
            _ct_definition["vertex_rules"].append(
                {
                    "name": _name,
                    "particles": _particles,
                    "color_structures": ["1"],
                    "lorentz_structures": [_name],
                    "couplings": [[_name]],
                }
            )
        _ct_model = Model.from_json(json.dumps(_ct_definition))
        return _model, _ct_model

    return (build_yukawa_model,)


@app.cell(hide_code=True)
def _(
    D,
    K,
    M,
    P,
    S,
    Symbol,
    TensorExpression,
    coordinate,
    den,
    dim,
    edge_,
    family,
    g,
    gamma,
    h,
    index,
    kinematics,
    mUV,
    mass,
    mass_,
    metric,
    mom_,
    mu,
    one,
    options,
    quad_,
    reducer,
    s,
    sp,
    wave,
):
    def project_yukawa_channels(_variant, _model, _ct_model, _local_coupling):
        """Generate and tensor-project the bare and local diagrams in the common family."""
        _fermion, _scalar = _model.particle("e-"), _model.particle("phi")
        _stage_parts, _stage_diagrams = ({}, {})
        for _stage, _stage_model, _loops in [("bare", _model, 1), ("ct", _ct_model, 0)]:
            _parts, _diagrams = ({}, {})
            for _kind, _incoming, _outgoing, _bare_count, _ct_count in [
                ("fermion", [_fermion], [_fermion], 1, 2),
                ("scalar", [_scalar], [_scalar], 2, 2),
                ("vertex", [_fermion], [_scalar, _fermion], 1, 1),
                ("quartic", [_scalar] * 2, [_scalar] * 2, 9, 1),
            ]:
                _generated = _stage_model.process(
                    _incoming, _outgoing
                ).generate_diagrams(
                    loops=_loops,
                    max_vertices=len(_incoming) + len(_outgoing) if _loops else 1,
                    coupling_orders={"CT": 1} if not _loops else None,
                    **options,
                )
                _diagrams[_kind] = _generated.diagrams
                assert len(_generated.diagrams) == (
                    _bare_count if _loops else _ct_count
                ), (_variant, _kind, len(_generated.diagrams))
                for _diagram in _generated.diagrams:
                    _ports = {}
                    if _kind in ("fermion", "vertex"):
                        for _edge in _diagram.external_edges:
                            if _kind == "vertex" and _edge.external_index == 1:
                                continue
                            _ports[_edge.external_index] = dict(
                                next(
                                    _diagram.projector_expression().match(
                                        wave(
                                            _edge.id,
                                            sp.PortPattern.exact(
                                                sp.Representation.bis(4), index
                                            ),
                                        ),
                                        max_level=0,
                                    )
                                )
                            )[index]
                    _numerator = _stage_model.expand_couplings(
                        _diagram.numerator_expression().to_expression()
                    )
                    if _loops:
                        _numerator = _diagram.uv_expansion(
                            mUV, numerator=_numerator
                        ).to_expression()
                    _numerator = _diagram.momentum_basis().route_expression(_numerator)
                    _numerator = (
                        _numerator.replace(
                            sp.PortPattern.exact(sp.Representation.mink(dim), index),
                            sp.PortPattern.exact(sp.Representation.mink(D), index),
                        )
                        .replace(
                            sp.PortPattern.exact(sp.Representation.mink(dim)),
                            sp.PortPattern.exact(sp.Representation.mink(D)),
                        )
                        .replace(mUV**2, M)
                    )
                    for _match in list(
                        _numerator.match(den(edge_, mom_, mass_, quad_))
                    ):
                        _values = dict(_match)
                        _formal = family.rewrite_numerator(_values[quad_], [coordinate])
                        assert _formal == coordinate
                        _numerator = _numerator.replace(
                            den(
                                _values[edge_],
                                _values[mom_],
                                _values[mass_],
                                _values[quad_],
                            ),
                            _formal,
                        )
                    _factor = (
                        _diagram.overall_factor_expression(evaluate=True)
                        * _diagram.numerator_prefactor_expression()
                    )
                    if not _loops:
                        _factor /= Symbol.I * h
                    if _kind == "fermion":
                        _probes = [
                            (
                                "fermion_p",
                                gamma(
                                    sp.PortPattern.exact(
                                        sp.Representation.bis(4), _ports[0]
                                    ),
                                    sp.PortPattern.exact(
                                        sp.Representation.bis(4), _ports[1]
                                    ),
                                    sp.PortPattern.exact(sp.Representation.mink(D), mu),
                                )
                                * P(
                                    0,
                                    sp.PortPattern.exact(sp.Representation.mink(D), mu),
                                )
                                / (4 * s),
                            ),
                            (
                                "fermion_m",
                                metric(
                                    sp.PortPattern.exact(
                                        sp.Representation.bis(4), _ports[0]
                                    ),
                                    sp.PortPattern.exact(
                                        sp.Representation.bis(4), _ports[1]
                                    ),
                                )
                                / (4 * mass),
                            ),
                        ]
                    elif _kind == "vertex":
                        _probe = (
                            metric(
                                sp.PortPattern.exact(
                                    sp.Representation.bis(4), _ports[0]
                                ),
                                sp.PortPattern.exact(
                                    sp.Representation.bis(4), _ports[2]
                                ),
                            )
                            if _variant == "Scalar"
                            else TensorExpression.gamma5(4)(
                                _ports[0], _ports[2]
                            ).to_expression()
                        )
                        _vertex_phase = _local_coupling / (-Symbol.I * g)
                        _probes = [(_kind, _probe / (4 * _vertex_phase))]
                    else:
                        _probes = [(_kind, one)]
                    for _label, _projector in _probes:
                        _trace = (
                            TensorExpression((_numerator * _projector).expand())
                            .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                            .expand()
                            .to_expression()
                        )
                        _expression = family.rewrite_numerator(
                            kinematics.apply(reducer.reduce(_trace)), [coordinate]
                        )
                        _expression = (_expression * _factor).together().expand()
                        _terms = _parts.setdefault(_label, [])
                        for _monomial, _coefficient in _expression.coefficient_list(
                            coordinate
                        ):
                            _power = -int(
                                (
                                    _monomial.derivative(coordinate)
                                    * coordinate
                                    / _monomial
                                ).together()
                            )
                            assert _monomial == coordinate ** (-_power)
                            assert not _coefficient.matches(K(S("args___")))
                            _terms.append(([_power], _coefficient))
            _stage_parts[_stage], _stage_diagrams[_stage] = (_parts, _diagrams)
        return _stage_parts, _stage_diagrams

    return (project_yukawa_channels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extend the scalar model with a fermion
    """)
    return


@app.cell
def _(Model, copy, json):
    specification = json.loads(Model.phi4().to_json())
    _standard = json.loads(Model.standard_model().to_json())
    specification["particles"] += [
        _p for _p in _standard["particles"] if _p["name"] in ("e-", "e+")
    ]
    _fermion_names = {
        _p["name"] for _p in specification["particles"] if _p["spin"] == 2
    }
    specification["propagators"] += [
        _p for _p in _standard["propagators"] if _p["particle"] in _fermion_names
    ]
    specification["parameters"] += [
        _p for _p in _standard["parameters"] if _p["name"] == "Me"
    ]
    _parameter = copy.deepcopy(specification["parameters"][2])
    _parameter.update(name="g", lhacode=[3])
    specification["parameters"].append(_parameter)
    specification["orders"].append(
        {"name": "YUKAWA", "expansion_order": 99, "hierarchy": 1}
    )
    return (specification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare couplings, dimensions and tensor patterns
    """)
    return


@app.cell
def _(E, Model, S, Symbols, hep, json, sp, specification):
    D, eps, M, mUV, coordinate, integral, s = S(
        "D",
        "eps",
        "M",
        "mUV",
        "x",
        "I",
        "s",
    )
    _parameter_model = Model.from_json(json.dumps(specification))
    K, P = hep.Kinematics.loop_momentum, hep.Kinematics.external_momentum
    mass = _parameter_model.particle("e-").mass
    scalar_mass = _parameter_model.parameter("mass").symbol
    g = _parameter_model.parameter("g").symbol
    lam = _parameter_model.parameter("lam").symbol
    mink, bis, gamma, metric = (
        sp.Representation.mink,
        sp.Representation.bis,
        sp.TensorName.dirac_gamma().to_expression(),
        sp.TensorName.g().to_expression(),
    )
    index, dim, wave, mu = S("index_", "dim_", "wave_", "mu")
    den, edge_, mom_, mass_, quad_ = (
        Symbols.denominator,
        S("edge_"),
        S("mom_"),
        S("mass_"),
        S("quad_"),
    )
    zero, one = (E("0"), E("1"))
    return (
        D,
        K,
        M,
        P,
        coordinate,
        den,
        dim,
        edge_,
        eps,
        g,
        gamma,
        index,
        integral,
        lam,
        mUV,
        mass,
        mass_,
        metric,
        mom_,
        mu,
        one,
        quad_,
        s,
        scalar_mass,
        wave,
        zero,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the vacuum family and its master pole
    """)
    return


@app.cell
def _(
    D,
    IntegralFamily,
    K,
    Kinematics,
    M,
    P,
    TensorReducer,
    one,
    oneloop,
    s,
    sp,
):
    kinematics = Kinematics(D, momenta=[K(0), P(0)]).with_scalar_product(P(0), P(0), s)
    _vacuum = Kinematics(D, momenta=[K(0)])
    family = IntegralFamily(
        [K(0)], [], [_vacuum.scalar_product(K(0), K(0)) - M], kinematics=_vacuum
    )
    _master_reduction = oneloop.reduce(family, [1])
    _master = _master_reduction.terms[0][1].to_expression(one)
    master_pole = oneloop.get_expression(_master, coefficient=-1)
    assert master_pole == M
    reducer = TensorReducer(
        D, integrated=[K(0, sp.PortPattern.exact(sp.Representation.mink(D)))]
    )
    options = {
        "maximum_bridges": 0,
        "self_energy": None,
        "tadpoles": None,
        "zero_snails": None,
        "numerator_grouping": None,
        "progress": None,
    }
    return family, kinematics, master_pole, options, reducer


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parameterize the local renormalization constants
    """)
    return


@app.cell
def _(S):
    h, _dpsi, _dm, _dphi, _dM, _dg, _dlam = S(
        "h",
        "dpsi",
        "dm",
        "dphi",
        "dM",
        "dg",
        "dlam",
    )
    unknowns = [_dpsi, _dm, _dphi, _dM, _dg, _dlam]
    Zpsi, Zm, Zphi, ZM, Zg, Zlam = (1 + h * _x for _x in unknowns)
    return ZM, Zg, Zlam, Zm, Zphi, Zpsi, h, unknowns


@app.cell
def _(Symbol, build_yukawa_model, g):
    models, ct_models, yukawa_couplings = {}, {}, {}
    for _variant, _lorentz, _local_coupling in [
        ("Scalar", "Identity(idx(1,1),idx(1,2))", -Symbol.I * g),
        ("Pseudoscalar", "Gamma5(idx(1,1),idx(1,2))", g),
    ]:
        models[_variant], ct_models[_variant] = build_yukawa_model(
            _variant, _lorentz, _local_coupling
        )
        yukawa_couplings[_variant] = _local_coupling
    return ct_models, models, yukawa_couplings


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and project both theories

    The folded channel routine uses `generate_diagrams`, `uv_expansion`, `simplify_algebra` and `TensorReducer` with the same family for every process.
    """)
    return


@app.cell
def _(ct_models, models, project_yukawa_channels, yukawa_couplings):
    all_stages, all_diagrams = {}, {}
    for _variant, _model in models.items():
        all_stages[_variant], all_diagrams[_variant] = project_yukawa_channels(
            _variant, _model, ct_models[_variant], yukawa_couplings[_variant]
        )
    return all_diagrams, all_stages


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce the integrals and check the UV poles
    """)
    return


@app.cell
def _(
    D,
    IBPFamily,
    M,
    all_stages,
    eps,
    family,
    g,
    integral,
    lam,
    mass,
    master_pole,
    s,
    scalar_mass,
    zero,
):
    all_poles, all_reductions = {}, {}
    for _variant, _stage_parts in all_stages.items():
        _parts = _stage_parts["bare"]
        _ibp = IBPFamily(family, name="yukawa_one_loop")
        _solution = _ibp.reduce_laporta(
            sorted({tuple(_p) for _terms in _parts.values() for _p, _c in _terms}),
            max_depth=2,
        )
        assert _solution.residuals == [[1]]
        _reduced, _poles = ({}, {})
        for _label, _terms in _parts.items():
            _expression = sum(
                (_c * _solution.reduce(_p, integral=integral) for _p, _c in _terms),
                zero,
            ).together()
            _reduced[_label] = _expression
            _poles[_label] = (
                _expression.replace(integral(1), master_pole / eps)
                .replace(D, 4 - 2 * eps)
                .series(eps, 0, -1)
                .to_expression()
                .expand()
            )
            assert _poles[_label].derivative(M).expand() == zero
        all_reductions[_variant] = _reduced
        all_poles[_variant] = _poles
        _reference_poles = {
            "fermion_p": -(g**2) / (2 * eps),
            "fermion_m": (-1 if _variant == "Scalar" else 1) * g**2 / eps,
            "scalar": (
                lam * scalar_mass**2 / 2
                + 2 * g**2 * s
                - (12 if _variant == "Scalar" else 4) * g**2 * mass**2
            )
            / eps,
            "vertex": -(g**3) / eps,
            "quartic": (3 * lam**2 / 2 - 24 * g**4) / eps,
        }
        for _label, _reference in _reference_poles.items():
            assert (_poles[_label] - _reference).together() == zero
    return (all_poles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the local counterterms and check pole cancellation
    """)
    return


@app.cell
def _(Matrix, all_poles, all_stages, s, unknowns, zero):
    all_counterterms, all_matrices, all_constants = {}, {}, {}
    for _variant, _stage_parts in all_stages.items():
        _poles = all_poles[_variant]
        _counterterms = {}
        for _label, _terms in _stage_parts["ct"].items():
            assert all((_powers == [0] for _powers, _coefficient in _terms))
            _counterterms[_label] = sum(
                (_coefficient for _powers, _coefficient in _terms), zero
            ).expand()
        _ct_rows = [
            _counterterms["fermion_p"],
            _counterterms["fermion_m"],
            _counterterms["scalar"].coefficient(s),
            _counterterms["scalar"].replace(s, zero),
            _counterterms["vertex"],
            _counterterms["quartic"],
        ]
        _loop_rows = [
            _poles["fermion_p"],
            _poles["fermion_m"],
            _poles["scalar"].coefficient(s),
            _poles["scalar"].replace(s, zero),
            _poles["vertex"],
            _poles["quartic"],
        ]
        _entries = [_row.coefficient(_x) for _row in _ct_rows for _x in unknowns]
        for _row_index, _row in enumerate(_ct_rows):
            assert (
                _row
                - sum(
                    (
                        _entries[_row_index * 6 + _i] * _x
                        for _i, _x in enumerate(unknowns)
                    ),
                    zero,
                )
            ).expand() == zero
        _matrix = Matrix.from_linear(6, 6, _entries)
        _solved = _matrix.solve(Matrix.vec([-_row for _row in _loop_rows]))
        _constants = [_solved[_i, 0].to_expression().expand() for _i in range(6)]
        all_counterterms[_variant] = _counterterms
        all_matrices[_variant] = _matrix
        all_constants[_variant] = _constants
    return all_constants, all_counterterms, all_matrices


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check all six constants and reconstructed pole cancellations
    """)
    return


@app.cell
def _(
    Replacement,
    all_constants,
    all_counterterms,
    all_poles,
    eps,
    g,
    lam,
    mass,
    scalar_mass,
    unknowns,
    zero,
):
    for _variant, _constants in all_constants.items():
        _poles = all_poles[_variant]
        _counterterms = all_counterterms[_variant]
        _expected = [
            -(g**2) / (2 * eps),
            (3 if _variant == "Scalar" else -1) * g**2 / (2 * eps),
            -2 * g**2 / eps,
            (
                2 * g**2
                + lam / 2
                - (12 if _variant == "Scalar" else 4) * g**2 * mass**2 / scalar_mass**2
            )
            / eps,
            5 * g**2 / (2 * eps),
            (4 * g**2 + 3 * lam / 2 - 24 * g**4 / lam) / eps,
        ]
        assert all(
            (
                (_actual - _reference).together() == zero
                for _actual, _reference in zip(_constants, _expected, strict=True)
            )
        ), (_variant, _constants)
        _replacements = [
            Replacement(_x, _value)
            for _x, _value in zip(unknowns, _constants, strict=True)
        ]
        for _label, _pole in _poles.items():
            assert (
                _pole + _counterterms[_label].replace_multiple(_replacements)
            ).together() == zero
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Both interactions reproduce all six reference constants; all projected UV poles cancel exactly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coupling running and the zero-quartic limit
    The local operators use $Z_\psi$, $Z_\psi Z_m$, $Z_\phi$, $Z_\phi Z_M$, $Z_g Z_\psi\sqrt{Z_\phi}$ and $Z_\lambda Z_\phi^2$. Their generated coefficients form a six-by-six system.

    The calculation derives $\beta_g$ and $\beta_\lambda$ by differentiating the bare couplings $\mu^\epsilon gZ_g$ and $\mu^{2\epsilon}\lambda Z_\lambda$. The two theories have equal coupling beta functions but different mass constants. Although $Z_\lambda$ contains $1/\lambda$, the additive shift $\lambda(Z_\lambda-1)$ stays finite at $\lambda=0$: Yukawa loops generate a quartic interaction.
    """)
    return


@app.cell
def _(
    Matrix,
    Replacement,
    S,
    Symbol,
    all_constants,
    all_counterterms,
    all_poles,
    eps,
    g,
    h,
    lam,
    unknowns,
    zero,
):
    _log4pi, _gamma_e = S("log4pi", "gamma_E")
    scheme_constants, additive_shifts, beta_functions = ({}, {}, {})
    for _variant, _constants in all_constants.items():
        for _scheme, _delta in [
            ("MS", 1 / eps),
            ("MSbar", 1 / eps + _log4pi - _gamma_e),
        ]:
            _shifts = [(_constant * eps * _delta).expand() for _constant in _constants]
            scheme_constants[_variant, _scheme] = [
                1 + _shift / (16 * Symbol.PI**2) for _shift in _shifts
            ]
            _replacements = [
                Replacement(_x, _value)
                for _x, _value in zip(unknowns, _shifts, strict=True)
            ]
            for _label, _pole in all_poles[_variant].items():
                _remainder = (
                    _pole * (1 + eps * (_log4pi - _gamma_e))
                    + all_counterterms[_variant][_label].replace_multiple(_replacements)
                ).together()
                _expected = (
                    zero if _scheme == "MSbar" else _pole * eps * (_log4pi - _gamma_e)
                )
                assert (_remainder - _expected).together() == zero
        _shifts = [(g * _constants[4]).expand(), (lam * _constants[5]).expand()]
        additive_shifts[_variant] = _shifts
        assert (_shifts[1].replace(lam, zero) + 24 * g**4 / eps).together() == zero
        _bare_couplings = [g + h * _shifts[0], lam + h * _shifts[1]]
        _jacobian = Matrix.from_linear(
            2,
            2,
            [_value.derivative(_x) for _value in _bare_couplings for _x in (g, lam)],
        )
        _flow = _jacobian.solve(
            Matrix.vec([-eps * _bare_couplings[0], -2 * eps * _bare_couplings[1]])
        )
        _beta = [
            _flow[_i, 0]
            .to_expression()
            .series(h, 0, 1)
            .to_expression()
            .series(eps, 0, 0)
            .to_expression()
            .expand()
            .coefficient(h)
            for _i in range(2)
        ]
        assert (_beta[0] - 5 * g**3).together() == zero
        assert (_beta[1] - 3 * lam**2 - 8 * g**2 * lam + 48 * g**4).together() == zero
        beta_functions[_variant] = [_value / (16 * Symbol.PI**2) for _value in _beta]
    assert beta_functions["Scalar"] == beta_functions["Pseudoscalar"]
    return additive_shifts, beta_functions, scheme_constants


@app.cell
def _(mo):
    interaction = mo.ui.dropdown(
        ["Scalar", "Pseudoscalar"], value="Scalar", label="Yukawa interaction"
    )
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    process = mo.ui.dropdown(
        {
            "Fermion self-energy": "fermion",
            "Scalar self-energy": "scalar",
            "Yukawa vertex": "vertex",
            "Four-scalar vertex": "quartic",
        },
        value="Scalar self-energy",
        label="Amplitude",
    )
    diagram_order = mo.ui.dropdown(
        {"Bare one-loop": "bare", "Local counterterms": "ct"},
        value="Bare one-loop",
        label="Diagrams",
    )
    mo.vstack([interaction, subtraction_scheme, process, diagram_order])
    return diagram_order, interaction, process, subtraction_scheme


@app.cell
def _(
    Symbol,
    additive_shifts,
    all_matrices,
    beta_functions,
    interaction,
    lam,
    mo,
    scheme_constants,
    subtraction_scheme,
    zero,
):
    selected_constants = scheme_constants[interaction.value, subtraction_scheme.value]
    selected_beta = beta_functions[interaction.value]
    _selected_additive = additive_shifts[interaction.value][1] / (16 * Symbol.PI**2)
    zero_quartic_shift = _selected_additive.replace(lam, zero)
    mo.vstack(
        [
            mo.md("## Generated matching matrix"),
            all_matrices[interaction.value],
            mo.md("## Renormalization constants"),
            *[
                mo.hstack([mo.md(_label), _value])
                for _label, _value in zip(
                    ["Zψ", "Zmψ", "Zφ", "Zmφ²", "Zg", "Zλ"],
                    selected_constants,
                    strict=True,
                )
            ],
            mo.md("## Derived beta functions"),
            mo.hstack([mo.md("βg"), selected_beta[0]]),
            mo.hstack([mo.md("βλ"), selected_beta[1]]),
            mo.md("**Additive quartic MS counterterm at λ=0**"),
            zero_quartic_shift,
        ]
    )
    return


@app.cell
def _(all_diagrams, all_poles, diagram_order, interaction, mo, process):
    selected_diagrams = all_diagrams[interaction.value][diagram_order.value][
        process.value
    ]
    _keys = (
        ["fermion_p", "fermion_m"] if process.value == "fermion" else [process.value]
    )
    mo.vstack(
        [
            mo.md(f"## {len(selected_diagrams)} generated diagrams"),
            *[
                mo.hstack(selected_diagrams[_i : _i + 3])
                for _i in range(0, len(selected_diagrams), 3)
            ],
            mo.md("**Projected bare UV poles, with i/(16π²) removed**"),
            *[
                mo.hstack([mo.md(_key), all_poles[interaction.value][_key]])
                for _key in _keys
            ],
            mo.md(
                "The displayed projections retain the generator's external-fermion ordering. The pseudoscalar vertex is projected onto the scalar phase convention before matching; the same projection is applied to its counterterm."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
