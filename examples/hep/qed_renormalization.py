import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="One-loop QED renormalization")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop QED renormalization

    [Browse all notebooks](/) ·
    [Massive electron self-energy](/?file=hep/electron_self_energy.py) ·
    [Electron g−2](/?file=hep/gminus2.py) ·
    [Two-loop photon](/?file=hep/photon_two_loop.py) ·
    [Two-loop electron](/?file=hep/electron_two_loop.py)

    Generate the massive electron self-energy, photon self-energy and
    electron-photon vertex with a **symbolic covariant gauge parameter** $\xi$.
    Generate the counterterm diagrams from local model rules as well. Their
    ultraviolet poles determine all six constants in the
    [FeynCalc reference](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/Renormalization).

    The model's photon numerator is
    $-i[g^{\mu\nu}-(1-\xi)q^\mu q^\nu/q^2]$.
    Feynkit expands each graph at large loop momentum, keeping the mass
    corrections through its degree of divergence. Shared Dirac traces and
    vacuum tensor reduction feed the native IBP solver.

    We use $D=4-2\epsilon$, $a_4=e^2/(16\pi^2)$ and
    $M=m_{\rm UV}^2>0$. The supplied analytic input is the tadpole pole
    $I(1)=M/\epsilon+O(1)$ in the $i/(16\pi^2)$ loop measure.
    The auxiliary mass cancels from every pole in this full massive UV expansion.
    Below, compare it with the massless reference's direct massification prescription
    and choose MS or MSbar subtraction. The displayed finite subtraction constants
    are scheme conversions; this UV calculation does not give finite off-shell amplitudes.
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

    import marimo as mo
    from symbolica import E, Matrix, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import (
        IBPFamily,
        IntegralFamily,
        Kinematics,
        Model,
        TensorReducer,
        oneloop,
    )
    from symbolica.community.tensor import TensorExpression

    _set_namespace("qed_ren")
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
        hep,
        json,
        mo,
        oneloop,
        sp,
    )


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    D,
    K,
    M,
    P,
    S,
    Symbols,
    TensorExpression,
    bis,
    coordinate,
    den,
    edge_,
    family,
    gamma,
    index,
    kinematics,
    mUV,
    mass,
    mass_,
    metric,
    mink,
    model,
    mom_,
    mu,
    nu,
    one,
    ordering_pattern,
    ordering_value,
    quad_,
    reducer,
    s,
    sp,
    wave,
):
    def project_qed_diagram(_kind, _loops, _diagram):
        """Project the generated open tensor onto the electron, photon or vertex basis."""
        parts = {}
        _ports = {}
        for _edge in _diagram.external_edges:
            _rep = (
                mink
                if _kind == "photon"
                or (_kind in ("tree", "vertex") and _edge.external_index == 1)
                else bis
            )
            _ports[_edge.external_index] = dict(
                next(
                    _diagram.projector_expression().match(
                        wave(_edge.id, sp.PortPattern.exact(_rep(4), index)),
                        max_level=0,
                    )
                )
            )[index]
        _numerator = model.expand_couplings(
            _diagram.numerator_expression().to_expression()
        )
        if _loops:
            _numerator = _diagram.momentum_basis().route_expression(
                _diagram.uv_expansion(mUV, numerator=_numerator).to_expression()
            )
        _numerator = (
            TensorExpression(_numerator)
            .with_lorentz_dimension(D)
            .to_expression()
            .replace(Symbols.dimension, D)
            .replace(mUV**2, M)
        )
        if _kind == "photon":
            _numerator = (
                sp.TensorExpression(_numerator)
                .rename_indices({_ports[0]: mu, _ports[1]: nu})
                .to_expression()
            )
        _pattern = den(edge_, mom_, mass_, quad_)
        for _match in list(_numerator.match(_pattern)):
            _values = dict(_match)
            _formal = family.rewrite_numerator(_values[quad_], [coordinate])
            assert _formal == coordinate
            _numerator = _numerator.replace(
                den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                _formal,
            )
        _factor = (
            _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        if _kind == "electron":
            _raw = _diagram.overall_factor_expression()
            _removed = (
                _raw / _raw.replace(ordering_pattern(ordering_value), one)
            ).replace(ordering_pattern(ordering_value), ordering_value)
            assert _removed == -one
            _factor /= _removed
            _probes = [
                (
                    "electron_p",
                    gamma(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                        sp.PortPattern.exact(sp.Representation.mink(D), mu),
                    )
                    * P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu))
                    / (4 * s),
                ),
                (
                    "electron_m",
                    metric(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                    )
                    / (4 * mass),
                ),
            ]
        elif _kind in ("vertex", "tree"):
            _probes = [
                (
                    _kind,
                    gamma(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[2]),
                        sp.PortPattern.exact(sp.Representation.mink(D), _ports[1]),
                    )
                    / (4 * D),
                )
            ]
        else:
            _probes = [(_kind, one)]
        for _label, _projector in _probes:
            _trace = (
                TensorExpression((_numerator * _projector).expand())
                .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                .expand()
                .to_expression()
            )
            _scalar = family.rewrite_numerator(
                kinematics.apply(reducer.reduce(_trace)), [coordinate]
            )
            _scalar = (_scalar * _factor).together().expand()
            _terms = []
            for _monomial, _coefficient in _scalar.coefficient_list(coordinate):
                _power = -int(
                    (
                        _monomial.derivative(coordinate) * coordinate / _monomial
                    ).together()
                )
                assert _monomial == coordinate ** (-_power)
                assert not _coefficient.matches(K(S("args___")))
                _terms.append(([_power], _coefficient))
            parts[_label] = _terms

        return parts

    return (project_qed_diagram,)


@app.cell(hide_code=True)
def _(
    D,
    P,
    Symbol,
    TensorExpression,
    a4,
    bis,
    ct_model,
    external_ordering,
    gamma,
    index,
    kinematics,
    mass,
    metric,
    mink,
    mu,
    nu,
    one,
    s,
    sp,
    tree,
    wave,
    zero,
):
    def project_qed_counterterm(_kind, _diagram):
        """Project a local generated insertion onto the same open tensor basis."""
        coefficients = {}
        assert len(_diagram.internal_edges) == 0
        assert _diagram.symmetry_factor == one
        assert _diagram.numerator_prefactor_expression() == one
        _factor = _diagram.overall_factor_expression(evaluate=True)
        assert _factor == (one if _kind == "photon" else external_ordering)
        _ports = {}
        for _edge in _diagram.external_edges:
            _rep = (
                mink
                if _kind == "photon"
                or (_kind == "vertex" and _edge.external_index == 1)
                else bis
            )
            _ports[_edge.external_index] = dict(
                next(
                    _diagram.projector_expression().match(
                        wave(_edge.id, sp.PortPattern.exact(_rep(4), index)),
                        max_level=0,
                    )
                )
            )[index]
        _numerator = ct_model.expand_couplings(
            _diagram.numerator_expression(in_lmb=True).to_expression()
        ).replace(
            sp.PortPattern.exact(sp.Representation.mink(4), index),
            sp.PortPattern.exact(sp.Representation.mink(D), index),
        )
        if _kind == "electron":
            _probes = [
                (
                    "electron_p",
                    gamma(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                        sp.PortPattern.exact(sp.Representation.mink(D), mu),
                    )
                    * P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu))
                    / (4 * s),
                ),
                (
                    "electron_m",
                    metric(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                    )
                    / (4 * mass),
                ),
            ]
            # Convert only the named external-state convention, matching the
            # bare electron insertion above; retain the native generated factor.
            _normalization = Symbol.I * a4 * external_ordering
        elif _kind == "photon":
            _numerator = _numerator.replace(
                sp.PortPattern.exact(sp.Representation.mink(D), _ports[0]),
                sp.PortPattern.exact(sp.Representation.mink(D), mu),
            ).replace(
                sp.PortPattern.exact(sp.Representation.mink(D), _ports[1]),
                sp.PortPattern.exact(sp.Representation.mink(D), nu),
            )
            _probes = [(_kind, one)]
            _normalization = Symbol.I * a4
        else:
            _probes = [
                (
                    _kind,
                    gamma(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[2]),
                        sp.PortPattern.exact(sp.Representation.mink(D), _ports[1]),
                    )
                    / (4 * D),
                )
            ]
            _normalization = a4 * tree
        for _label, _projector in _probes:
            _trace = (
                TensorExpression((_numerator * _projector).expand())
                .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                .expand()
                .to_expression()
            )
            _coefficient = (
                (kinematics.apply(_trace) * _factor / _normalization)
                .together()
                .expand()
            )
            coefficients[_label] = coefficients.get(_label, zero) + _coefficient

        return coefficients

    return (project_qed_counterterm,)


@app.cell(hide_code=True)
def _(
    D,
    K,
    M,
    Nf,
    P,
    Replacement,
    Symbol,
    Symbols,
    TensorExpression,
    bis,
    charge,
    coordinate,
    den,
    dim,
    dot_pattern,
    edge_,
    edge_momentum,
    external_ordering,
    family,
    gamma,
    gmunu,
    index,
    mUV,
    mass,
    mass_,
    massless_Qg,
    massless_Qpp,
    massless_a,
    massless_args,
    massless_b,
    massless_kinematics,
    massless_loop_square,
    massless_model,
    massless_photon,
    massless_photon_coordinate,
    massless_scale,
    mink,
    mom_,
    mu,
    nu,
    one,
    ppmunu,
    quad_,
    reducer,
    s,
    sp,
    tree,
    wave,
    xi,
    zero,
):
    def project_massless_qed(_kind, _diagram):
        """Compare full UV expansion, selective massification and direct IR rearrangement for one diagram."""
        massless_parts, massless_scalars = {}, {}
        massless_components, massless_component_scalars = {}, {}
        massless_targets = set()
        _ports = {}
        for _edge in _diagram.external_edges:
            _rep = (
                mink
                if _kind == "photon"
                or (_kind == "vertex" and _edge.external_index == 1)
                else bis
            )
            _ports[_edge.external_index] = dict(
                next(
                    _diagram.projector_expression().match(
                        wave(_edge.id, sp.PortPattern.exact(_rep(4), index)),
                        max_level=0,
                    )
                )
            )[index]
        _numerator = massless_model.expand_couplings(
            _diagram.numerator_expression().to_expression()
        ).replace(mass, zero)
        # Preserve the primitive denominators used by the n=0 prescription.
        # Combining the Feynman term over a squared photon denominator first
        # would turn 1/q² into q²/(q²-M)² after massification and shift finite
        # terms. Only the longitudinal remainder has a genuine extra 1/q².
        _photon_edges = [
            _edge
            for _edge in _diagram.internal_edges
            if _edge.particle_name == massless_photon.name
        ]
        assert len(_photon_edges) <= 1
        _components = [("fermion_loop", _numerator, {})]
        if _photon_edges:
            _photon_edge = _photon_edges[0]
            _photon_square = dot_pattern(
                edge_momentum(
                    _photon_edge.id, sp.PortPattern.exact(sp.Representation.mink(4))
                ),
                edge_momentum(
                    _photon_edge.id, sp.PortPattern.exact(sp.Representation.mink(4))
                ),
            )
            _feynman_numerator = _numerator.replace(xi, one)
            _longitudinal_numerator = (
                ((_numerator - _feynman_numerator) * _photon_square).together().expand()
            )
            assert _longitudinal_numerator.replace(xi, one).expand() == zero
            assert (
                _feynman_numerator
                + _longitudinal_numerator / _photon_square
                - _numerator
            ).together() == zero
            _components = [
                ("feynman", _feynman_numerator, {}),
                ("longitudinal", _longitudinal_numerator, {_photon_edge.id: 2}),
            ]
            # At xi=1 the primitive numerator is unchanged and its photon has
            # the default single propagator; no q² has been multiplied into it.
            assert _components[0][1] == _numerator.replace(xi, one)
            assert _components[0][2] == {}
        massless_components[_kind] = _components
        _factor = (
            _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        if _kind == "electron_p":
            assert _factor == external_ordering
            _factor /= external_ordering
            _projector = (
                gamma(
                    sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                    sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                    sp.PortPattern.exact(sp.Representation.mink(D), mu),
                )
                * P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu))
                / (4 * s)
            )
        elif _kind == "vertex":
            _projector = gamma(
                sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                sp.PortPattern.exact(sp.Representation.bis(4), _ports[2]),
                sp.PortPattern.exact(sp.Representation.mink(D), _ports[1]),
            ) / (4 * D)
        else:
            _projector = one
        for _component, _numerator, _powers in _components:
            _numerator = _numerator.replace(
                sp.PortPattern.exact(sp.Representation.mink(dim), index),
                sp.PortPattern.exact(sp.Representation.mink(D), index),
            ).replace(
                sp.PortPattern.exact(sp.Representation.mink(dim)),
                sp.PortPattern.exact(sp.Representation.mink(D)),
            )
            if _kind == "photon":
                _numerator = _numerator.replace(
                    sp.PortPattern.exact(sp.Representation.mink(D), _ports[0]),
                    sp.PortPattern.exact(sp.Representation.mink(D), mu),
                ).replace(
                    sp.PortPattern.exact(sp.Representation.mink(D), _ports[1]),
                    sp.PortPattern.exact(sp.Representation.mink(D), nu),
                )

            # Retain the shared full UV result as an independent contrast. Freeze its
            # tagged massive denominators before extracting the zeroth explicit mUV
            # coefficient; M in the family is held fixed, never replaced by zero.
            _expanded = (
                TensorExpression(
                    _diagram.momentum_basis().route_expression(
                        _diagram.uv_expansion(
                            mUV, numerator=_numerator, edge_powers=_powers
                        ).to_expression()
                    )
                )
                .with_lorentz_dimension(D)
                .to_expression()
                .replace(Symbols.dimension, D)
            )
            _pattern = den(edge_, mom_, mass_, quad_)
            for _match in list(_expanded.match(_pattern)):
                _values = dict(_match)
                _formal = family.rewrite_numerator(
                    _values[quad_].replace(mUV**2, M), [coordinate]
                )
                assert _formal == coordinate
                _expanded = _expanded.replace(
                    den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                    coordinate,
                )
            assert not _expanded.matches(_pattern)
            _variants = {
                "full_uv": _expanded.replace(mUV**2, M),
                "selective_uv": _expanded.replace(mUV, zero),
            }

            # Implement the primary prescription literally and independently: replace
            # massless graph denominators by q²-M, then Taylor-expand external momenta.
            _routed = _diagram.momentum_basis().route_expression(_numerator)
            _traced = (
                TensorExpression((_routed * _projector).expand())
                .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                .expand()
                .to_expression()
            )
            _denominator = (
                _diagram.denominator_expression(in_lmb=True, edge_powers=_powers)
                .to_expression()
                .replace(
                    sp.PortPattern.exact(sp.Representation.mink(dim), index),
                    sp.PortPattern.exact(sp.Representation.mink(D), index),
                )
                .replace(
                    sp.PortPattern.exact(sp.Representation.mink(dim)),
                    sp.PortPattern.exact(sp.Representation.mink(D)),
                )
            )
            for _match in list(_denominator.match(_pattern)):
                _values = dict(_match)
                assert _values[mass_] == zero
                if _photon_edges and _values[edge_] == _photon_edge.id:
                    # Verify the actual graph denominator power independently of
                    # the override map, before any auxiliary-mass replacement.
                    _photon_tag = den(
                        _values[edge_], _values[mom_], _values[mass_], _values[quad_]
                    )
                    _checked_denominator = _denominator.replace(
                        _photon_tag, massless_photon_coordinate
                    )
                    _actual_power = (
                        _checked_denominator.derivative(massless_photon_coordinate)
                        * massless_photon_coordinate
                        / _checked_denominator
                    ).together()
                    assert _actual_power == (1 if _component == "feynman" else 2)
                _denominator = _denominator.replace(
                    den(_values[edge_], _values[mom_], _values[mass_], _values[quad_]),
                    _values[quad_] - M,
                )
            _direct = massless_kinematics.apply(_traced / _denominator).replace(
                massless_loop_square, coordinate + M
            )
            _direct = _direct.replace_multiple(
                [
                    Replacement(
                        dot_pattern(
                            K(0, sp.PortPattern.exact(sp.Representation.mink(D))),
                            P(
                                massless_a,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                        ),
                        massless_scale
                        * dot_pattern(
                            K(0, sp.PortPattern.exact(sp.Representation.mink(D))),
                            P(
                                massless_a,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                        ),
                    ),
                    Replacement(
                        dot_pattern(
                            P(
                                massless_a,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                            K(0, sp.PortPattern.exact(sp.Representation.mink(D))),
                        ),
                        massless_scale
                        * dot_pattern(
                            P(
                                massless_a,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                            K(0, sp.PortPattern.exact(sp.Representation.mink(D))),
                        ),
                    ),
                    Replacement(
                        dot_pattern(
                            P(
                                massless_a,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                            P(
                                massless_b,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                        ),
                        massless_scale**2
                        * dot_pattern(
                            P(
                                massless_a,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                            P(
                                massless_b,
                                sp.PortPattern.exact(sp.Representation.mink(D)),
                            ),
                        ),
                    ),
                    Replacement(
                        P(
                            massless_a,
                            sp.PortPattern.exact(sp.Representation.mink(D), index),
                        ),
                        massless_scale
                        * P(
                            massless_a,
                            sp.PortPattern.exact(sp.Representation.mink(D), index),
                        ),
                    ),
                    Replacement(s, massless_scale**2 * s),
                ]
            )
            # The electron's pslash/(4p²) projector lowers the external degree by one,
            # so projected order0 retains its degree1 self-energy. Photon needs order2;
            # the logarithmically divergent vertex needs only order0.
            _variants["direct_irr"] = (
                _direct.series(massless_scale, 0, 2 if _kind == "photon" else 0)
                .to_expression()
                .replace(massless_scale, one)
            )
            for _scheme, _expression in _variants.items():
                _trace = (
                    _expression
                    if _scheme == "direct_irr"
                    else TensorExpression((_expression * _projector).expand())
                    .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                    .expand()
                    .to_expression()
                )
                _scalar = family.rewrite_numerator(
                    massless_kinematics.apply(reducer.reduce(_trace)), [coordinate]
                )
                _normalization = (
                    _factor
                    / charge**2
                    * (Symbol.I / tree if _kind == "vertex" else one)
                    * (Nf if _kind == "photon" else one)
                )
                _scalar = (_normalization * _scalar).together().expand()
                massless_component_scalars[_scheme, _kind, _component] = _scalar
                massless_scalars[_scheme, _kind] = (
                    massless_scalars.get((_scheme, _kind), zero) + _scalar
                ).expand()
            assert (
                massless_component_scalars["direct_irr", _kind, _component]
                - massless_component_scalars["selective_uv", _kind, _component]
            ).together() == zero
        for _scheme in ("full_uv", "selective_uv", "direct_irr"):
            _scalar = massless_scalars[_scheme, _kind]
            _terms = []
            for _monomial, _coefficient in _scalar.coefficient_list(coordinate):
                _power = -int(
                    (
                        _monomial.derivative(coordinate) * coordinate / _monomial
                    ).together()
                )
                assert _monomial == coordinate ** (-_power)
                assert not _coefficient.matches(K(massless_args))
                _scalar_coefficient = _coefficient.replace(gmunu, massless_Qg).replace(
                    ppmunu, massless_Qpp
                )
                assert not _scalar_coefficient.matches(P(massless_args))
                assert _coefficient.derivative(coordinate) == zero
                massless_targets.add((_power,))
                _terms.append(([_power], _coefficient))
            massless_parts[_scheme, _kind] = _terms
        # This is an exact integrand-level check after vacuum tensor reduction,
        # not merely agreement of the final UV pole with a supplied formula.
        assert (
            massless_scalars["direct_irr", _kind]
            - massless_scalars["selective_uv", _kind]
        ).together() == zero

        return (
            massless_parts,
            massless_targets,
            massless_scalars,
            massless_components,
            massless_component_scalars,
        )

    return (project_massless_qed,)


@app.cell(hide_code=True)
def _(
    M,
    Symbol,
    ZA,
    ZAm,
    Ze,
    Zm,
    Zpsi,
    Zxi,
    a4,
    electron,
    json,
    local_tree,
    mass,
    model,
    one,
    photon,
    vertices,
    xi,
):
    def qed_counterterm_definition():
        """Encode the local QED operators and their bare-field factors in model data."""
        specification = json.loads(model.to_json())
        specification["orders"].append(
            {"name": "CT", "expansion_order": 1, "hierarchy": 1}
        )
        _ffv = next(
            structure
            for structure in specification["lorentz_structures"]
            if structure["name"] == vertices[0].lorentz_structures[0]
        )
        for _label, _particles, _spins, _lorentz, _coupling, _qed_order in [
            (
                "ee_kinetic",
                [electron.antiname, electron.name],
                [2, 2],
                # Normalized JSON uses the spinor order of the imported SM FFV rule.
                # UFO momenta are incoming: leg 2 carries the electron's momentum.
                "Gamma(dummy(1),idx(1,1),idx(1,2))*P(dummy(1),2)",
                Symbol.I * (Zpsi - one),
                2,
            ),
            (
                "ee_mass",
                [electron.antiname, electron.name],
                [2, 2],
                "Identity(idx(1,1),idx(1,2))",
                -Symbol.I * mass * (Zpsi * Zm - one),
                2,
            ),
            (
                "aa_kinetic",
                [photon.name] * 2,
                [3, 3],
                (
                    "Metric(idx(1,1),idx(1,2))*P(dummy(1),1)*P(dummy(1),1)"
                    "-P(idx(1,1),1)*P(idx(1,2),1)"
                ),
                -Symbol.I * (ZA - one),
                2,
            ),
            (
                "aa_gauge",
                [photon.name] * 2,
                [3, 3],
                "P(idx(1,1),1)*P(idx(1,2),1)",
                -Symbol.I * (ZA / Zxi - one) / xi,
                2,
            ),
            (
                "aa_auxmass",
                [photon.name] * 2,
                [3, 3],
                "Metric(idx(1,1),idx(1,2))",
                Symbol.I * M * (ZAm**2 - one),
                2,
            ),
            (
                "eea",
                list(vertices[0].particles),
                list(_ffv["spins"]),
                _ffv["structure"],
                local_tree * (Zpsi * Ze * ZA.sqrt() - one),
                3,
            ),
        ]:
            specification["lorentz_structures"].append(
                {"name": "CT_L_" + _label, "spins": _spins, "structure": _lorentz}
            )
            specification["couplings"].append(
                {
                    "name": "CT_GC_" + _label,
                    "expression": repr(_coupling.series(a4, 0, 1).to_expression()),
                    "orders": [["QED", _qed_order], ["CT", 1]],
                    "value": None,
                }
            )
            specification["vertex_rules"].append(
                {
                    "name": "CT_" + _label,
                    "particles": _particles,
                    "color_structures": ["1"],
                    "lorentz_structures": ["CT_L_" + _label],
                    "couplings": [["CT_GC_" + _label]],
                }
            )
        return specification

    return (qed_counterterm_definition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Choose the covariant-gauge photon propagator
    """)
    return


@app.cell
def _(E, Model, S, Symbols, json, sp):
    model = Model.standard_model()
    D, epsilon, M, mUV, coordinate, integral, xi, s, Nf = S(
        "D",
        "eps",
        "M",
        "mUV",
        "d0",
        "I",
        "xi",
        "s",
        "Nf",
    )
    _specification = json.loads(model.to_json())
    for _propagator in _specification["propagators"]:
        if _propagator["particle"] == "a":
            _propagator["numerator"] = (
                E("-1𝑖")
                * (
                    Symbols.ufo_metric(Symbols.ufo_index(1, 1), Symbols.ufo_index(1, 2))
                    - (1 - S("xi"))
                    * Symbols.ufo_momentum(Symbols.ufo_index(1, 1))
                    * Symbols.ufo_momentum(Symbols.ufo_index(1, 2))
                    / sp.TensorName.g().to_expression()(
                        Symbols.ufo_momentum(
                            sp.PortPattern.exact(sp.Representation.mink(4))
                        ),
                        Symbols.ufo_momentum(
                            sp.PortPattern.exact(sp.Representation.mink(4))
                        ),
                    )
                )
            ).format_plain()
    model = Model.from_json(json.dumps(_specification))
    return D, M, Nf, coordinate, epsilon, integral, mUV, model, s, xi


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select QED and declare the tensor patterns
    """)
    return


@app.cell
def _(S, Symbols, hep, model, sp):
    electron, photon = (model.particle(_name) for _name in ("e-", "a"))
    vertices = [
        vertex
        for vertex in model.vertex_rules
        if sorted(vertex.particles)
        == sorted([electron.antiname, electron.name, photon.name])
    ]
    assert len(vertices) == 1
    K, P = hep.Kinematics.loop_momentum, hep.Kinematics.external_momentum
    _electron = model.particle("e-")
    mass, charge = _electron.mass, -_electron.electric_charge
    mink, bis, gamma, metric = (
        sp.Representation.mink,
        sp.Representation.bis,
        sp.TensorName.dirac_gamma().to_expression(),
        sp.TensorName.g().to_expression(),
    )
    index, dim, wave, mu, nu = S(
        "index_",
        "dim_",
        "wave_",
        "mu",
        "nu",
    )
    den, edge_, mom_, mass_, quad_ = (
        Symbols.denominator,
        S("edge_"),
        S("mom_"),
        S("mass_"),
        S("quad_"),
    )
    ordering_pattern, ordering_value = S(
        "feynkit_generator_factor::ExternalFermionOrderingSign", "value_"
    )
    return (
        K,
        P,
        bis,
        charge,
        den,
        dim,
        edge_,
        electron,
        gamma,
        index,
        mass,
        mass_,
        metric,
        mink,
        mom_,
        mu,
        nu,
        ordering_pattern,
        ordering_value,
        photon,
        quad_,
        vertices,
        wave,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the vacuum family and open photon basis
    """)
    return


@app.cell
def _(
    D,
    E,
    IntegralFamily,
    K,
    Kinematics,
    M,
    P,
    TensorReducer,
    metric,
    mu,
    nu,
    s,
    sp,
):
    kinematics = Kinematics(D, momenta=[K(0), P(0)]).with_scalar_product(P(0), P(0), s)
    vacuum = Kinematics(D, momenta=[K(0)])
    family = IntegralFamily(
        [K(0)], [], [vacuum.scalar_product(K(0), K(0)) - M], kinematics=vacuum
    )
    reducer = TensorReducer(
        D, integrated=[K(0, sp.PortPattern.exact(sp.Representation.mink(D)))]
    )
    gmunu = metric(
        sp.PortPattern.exact(sp.Representation.mink(D), mu),
        sp.PortPattern.exact(sp.Representation.mink(D), nu),
    )
    ppmunu = P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu)) * P(
        0, sp.PortPattern.exact(sp.Representation.mink(D), nu)
    )
    zero, one = E("0"), E("1")
    return family, gmunu, kinematics, one, ppmunu, reducer, vacuum, zero


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and project the one-loop diagrams
    """)
    return


@app.cell
def _(electron, model, photon, project_qed_diagram, vertices):
    bare_parts, diagrams = {}, {}
    for _kind, _incoming, _outgoing, _loops in [
        ("tree", [electron], [photon, electron], 0),
        ("electron", [electron], [electron], 1),
        ("photon", [photon], [photon], 1),
        ("vertex", [electron], [photon, electron], 1),
    ]:
        _generated = model.process(
            _incoming, _outgoing, vertex_allow=vertices
        ).generate_diagrams(
            loops=_loops,
            max_vertices=len(_incoming) + len(_outgoing) - 2 + 2 * _loops,
            maximum_bridges=0,
            self_energy=None,
            tadpoles=None,
            zero_snails=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == 1
        _diagram = _generated.diagrams[0]
        diagrams[_kind] = _diagram
        bare_parts.update(project_qed_diagram(_kind, _loops, _diagram))
    return bare_parts, diagrams


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fix the tree convention and reduce the integrals
    """)
    return


@app.cell
def _(IBPFamily, Symbol, bare_parts, charge, family, oneloop):
    tree = bare_parts["tree"][0][1]
    assert tree == Symbol.I * charge
    loop_parts = {name: terms for name, terms in bare_parts.items() if name != "tree"}
    solution = IBPFamily(family, name="qed_one_loop").reduce_laporta(
        sorted({tuple(p) for _terms in loop_parts.values() for p, c in _terms}),
        max_depth=2,
    )
    assert solution.residuals == [[1]]
    assert abs(complex(oneloop.a0(1.0, 1.0)[1]) - 1) < 1e-12
    return loop_parts, solution, tree


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate the master pole
    """)
    return


@app.cell
def _(
    D,
    M,
    Nf,
    Symbol,
    charge,
    epsilon,
    index,
    integral,
    loop_parts,
    mass,
    one,
    solution,
    sp,
    tree,
    zero,
):
    _reduced, uv_poles = {}, {}
    for _label, _terms in loop_parts.items():
        _expression = sum(
            (c * solution.reduce(p, integral=integral) for p, c in _terms), zero
        ).together()
        _reduced[_label] = _expression
        _normalized = (
            _expression * (Symbol.I / tree if _label == "vertex" else one) / charge**2
        )
        if _label == "photon":
            _normalized *= Nf
        _pole = (
            _normalized.replace(integral(1), M / epsilon)
            .replace(D, 4 - 2 * epsilon)
            .series(epsilon, 0, -1)
            .to_expression()
            .expand()
        )
        _pole = _pole.replace(
            sp.PortPattern.exact(sp.Representation.mink(4), index),
            sp.PortPattern.exact(sp.Representation.mink(D), index),
        )
        uv_poles[_label] = _pole
        assert _pole.derivative(M).expand() == zero
        assert _pole.derivative(mass).expand() == zero
        assert _pole.coefficient(epsilon**-2) == zero
    # Compare the four UV structures with the published symbolic-gauge result.
    return (uv_poles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the massive UV references
    """)
    return


@app.cell
def _(Nf, epsilon, gmunu, ppmunu, s, uv_poles, xi, zero):
    _expected = [
        xi / epsilon,
        -(xi + 3) / epsilon,
        -4 * Nf * (s * gmunu - ppmunu) / (3 * epsilon),
        xi / epsilon,
    ]
    for _label, _reference in zip(
        ("electron_p", "electron_m", "photon", "vertex"), _expected, strict=True
    ):
        assert (uv_poles[_label] - _reference).together() == zero
    return


@app.cell(hide_code=True)
def _(diagrams, family, mo, solution, uv_poles):
    mo.vstack(
        [
            mo.md("## Generated diagrams and UV poles"),
            mo.hstack([diagrams[_name] for _name in ("electron", "photon", "vertex")]),
            mo.md(
                r"The electron and photon poles below multiply $i a_4$; the vertex entry multiplies its generated tree vertex times $a_4$. Electron entries are the coefficients of $\not p$ and $m$, respectively."
            ),
            mo.vstack(
                [
                    mo.hstack([mo.md("**" + _name + "**"), _pole])
                    for _name, _pole in uv_poles.items()
                ]
            ),
            mo.md("## Shared vacuum family and IBP reduction"),
            family,
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "The powers I(1) through I(4) reduce to the tadpole I(1). Its analytic value is a supplied input; a finite-depth residual is not a general proof of a minimal master basis."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Parameterize local renormalization factors
    """)
    return


@app.cell
def _(S, Symbol, charge, diagrams, one, tree):
    # Counterterm structures follow the local kinetic, mass and vertex operators.
    # Solve for coefficients of Z=1+a4*deltaZ. Their coupling combinations come from
    # expanding the bare factors, while actual generated diagrams supply the matrix.
    # M=mUV². The reference auxiliary-mass operator is M*(ZAm²-1)*A²/2:
    # its linear coefficient is 2*deltaZAm, unlike an additive mass-squared shift.
    a4, _delta_psi, delta_m, _delta_A, _delta_xi, _delta_e, _delta_Am = S(
        "a4",
        "deltaZpsi",
        "deltaZm",
        "deltaZA",
        "deltaZxi",
        "deltaZe",
        "deltaZAm",
    )
    unknowns = [_delta_psi, delta_m, _delta_A, _delta_xi, _delta_e, _delta_Am]
    Zpsi, Zm, ZA, Zxi, Ze, ZAm = (one + a4 * delta for delta in unknowns)
    external_ordering = diagrams["tree"].overall_factor_expression(evaluate=True)
    assert external_ordering == -one
    assert (
        diagrams["electron"].overall_factor_expression(evaluate=True)
        == external_ordering
    )
    local_tree = tree / external_ordering
    assert local_tree == -Symbol.I * charge
    return (
        ZA,
        ZAm,
        Ze,
        Zm,
        Zpsi,
        Zxi,
        a4,
        delta_m,
        external_ordering,
        local_tree,
        unknowns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Add their local operators to the model
    """)
    return


@app.cell
def _(qed_counterterm_definition):
    specification = qed_counterterm_definition()
    return (specification,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select the counterterm vertices
    """)
    return


@app.cell
def _(Model, json, specification):
    ct_model = Model.from_json(json.dumps(specification))
    ct_electron, ct_photon = (ct_model.particle(_name) for _name in ("e-", "a"))
    ct_vertices = [
        vertex for vertex in ct_model.vertex_rules if vertex.name.startswith("CT_")
    ]
    assert len(ct_vertices) == 6
    return ct_electron, ct_model, ct_photon, ct_vertices


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and project local insertions
    """)
    return


@app.cell
def _(
    ct_electron,
    ct_model,
    ct_photon,
    ct_vertices,
    project_qed_counterterm,
    zero,
):
    ct_diagrams, ct_coefficients = {}, {}
    for _kind, _incoming, _outgoing, _count, _qed_order in [
        ("electron", [ct_electron], [ct_electron], 2, 2),
        ("photon", [ct_photon], [ct_photon], 3, 2),
        ("vertex", [ct_electron], [ct_photon, ct_electron], 1, 3),
    ]:
        # These are tree topologies at perturbative CT order one. Bound vertices
        # explicitly because arbitrarily many two-point insertions add no loops.
        _options = {
            "loops": 0,
            "max_vertices": 1,
            "maximum_bridges": None,
            "self_energy": None,
            "tadpoles": None,
            "zero_snails": None,
            "numerator_grouping": None,
            "progress": None,
        }
        _generated = ct_model.process(
            _incoming, _outgoing, vertex_allow=ct_vertices
        ).generate_diagrams(coupling_orders={"QED": _qed_order, "CT": 1}, **_options)
        assert len(_generated.diagrams) == _count
        assert (
            not ct_model.process(_incoming, _outgoing, vertex_allow=ct_vertices)
            .generate_diagrams(coupling_orders={"CT": 0}, **_options)
            .diagrams
        )
        ct_diagrams[_kind] = _generated.diagrams
        for _diagram in _generated.diagrams:
            for _label, _coefficient in project_qed_counterterm(
                _kind, _diagram
            ).items():
                ct_coefficients[_label] = (
                    ct_coefficients.get(_label, zero) + _coefficient
                )
    return ct_coefficients, ct_diagrams


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the linear counterterm matrix
    """)
    return


@app.cell
def _(M, Matrix, ct_coefficients, gmunu, ppmunu, s, unknowns, xi, zero):
    _ct_photon_g = ct_coefficients["photon"].coefficient(gmunu)
    _ct_photon_pp = ct_coefficients["photon"].coefficient(ppmunu)
    assert (
        ct_coefficients["photon"] - _ct_photon_g * gmunu - _ct_photon_pp * ppmunu
    ).together() == zero
    assert _ct_photon_g.derivative(s).derivative(s) == zero
    ct_rows = [
        ct_coefficients["electron_p"],
        ct_coefficients["electron_m"],
        _ct_photon_g.coefficient(s),
        xi * _ct_photon_pp,
        ct_coefficients["vertex"],
        _ct_photon_g.replace(s, zero) / M,
    ]
    # Unknown order: deltaZpsi, deltaZm, deltaZA, deltaZxi, deltaZe, deltaZAm.
    # Extract every matrix entry from generated amplitudes; check linearity so no
    # constant or nonlinear term can be silently dropped by differentiation.
    _ct_entries = [
        _row.derivative(delta).together() for _row in ct_rows for delta in unknowns
    ]
    assert all(
        entry.derivative(delta) == zero for entry in _ct_entries for delta in unknowns
    )
    for _position, _row in enumerate(ct_rows):
        assert (
            _row
            - sum(
                (
                    _ct_entries[6 * _position + j] * delta
                    for j, delta in enumerate(unknowns)
                ),
                zero,
            )
        ).together() == zero
    ct_matrix = Matrix.from_linear(6, 6, _ct_entries)
    return ct_matrix, ct_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve and verify exact pole cancellation
    """)
    return


@app.cell
def _(
    M,
    Matrix,
    Nf,
    ct_matrix,
    epsilon,
    gmunu,
    ppmunu,
    s,
    solution,
    uv_poles,
    xi,
    zero,
):
    _photon_g = uv_poles["photon"].coefficient(gmunu)
    _photon_pp = uv_poles["photon"].coefficient(ppmunu)
    _rhs = Matrix.vec(
        [
            -uv_poles["electron_p"],
            -uv_poles["electron_m"],
            -_photon_g.coefficient(s),
            -xi * _photon_pp,
            -uv_poles["vertex"],
            -_photon_g.replace(s, zero) / M,
        ]
    )
    deltas = ct_matrix.solve(_rhs)
    for _row, _reference in enumerate(
        [
            -xi / epsilon,
            -3 / epsilon,
            -4 * Nf / (3 * epsilon),
            -4 * Nf / (3 * epsilon),
            2 * Nf / (3 * epsilon),
            zero,
        ]
    ):
        assert (deltas[_row, 0].to_expression() - _reference).together() == zero
    assert (deltas[0, 0].to_expression() + uv_poles["vertex"]).together() == zero
    assert (
        deltas[4, 0].to_expression() + deltas[2, 0].to_expression() / 2
    ).together() == zero
    _residual = ct_matrix * deltas - _rhs
    assert all(
        _residual[_row, 0].to_expression().together() == zero for _row in range(6)
    )
    print(
        "Generated QED one-loop: symbolic gauge, four IBP targets, six generated CT diagrams and Ward identities passed",
        solution.stats,
    )
    return (deltas,)


@app.cell(hide_code=True)
def _(ct_coefficients, ct_diagrams, ct_matrix, deltas, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Generated counterterm diagrams

    Write $Z=1+a_4\delta Z$, with $\psi_0=\sqrt{Z_\psi}\psi$,
    $m_0=Z_m m$, $A_0=\sqrt{Z_A}A$, $\xi_0=Z_\xi\xi$ and $e_0=Z_e e$.
    The local operator basis is a model input. Symbolica expands the bare factors
    $Z_\psi$, $Z_\psi Z_m$, $Z_A$, $Z_A/Z_\xi$ and
    $Z_\psi Z_e\sqrt{Z_A}$ to obtain the first-order couplings.

    The ordinary generator then supplies two electron insertions, three photon
    insertions and one vertex counterterm. They have **zero topological loops**
    and **CT order one**. An explicit vertex bound prevents arbitrarily long
    chains of two-point insertions. Changing the CT order to zero excludes them.
    """),
            *[
                mo.vstack(
                    [mo.md("**" + _name + " counterterms**"), mo.hstack(_diagrams)]
                )
                for _name, _diagrams in ct_diagrams.items()
            ],
            mo.md(r"**Coefficients from the actual generated numerators**"),
            *[
                mo.hstack([mo.md("**" + _name + "**"), _coefficient])
                for _name, _coefficient in ct_coefficients.items()
            ],
            mo.md(r"""
    The same native graph weights and external ordering convention are used for
    loops and counterterms. Differentiating the generated coefficients with
    respect to the unknown $\delta Z$ values builds the matrix below. Linearity
    and the complete tensor cancellation are checked separately.

    The column order is
    $(\delta Z_\psi,\delta Z_m,\delta Z_A,\delta Z_\xi,\delta Z_e,\delta Z_{Am})$.
    The auxiliary operator is $M(Z_{Am}^2-1)A_\mu A^\mu/2$, so its two-point rule
    contains **$2\delta Z_{Am}$**, visible in the last matrix entry.
    """),
            ct_matrix,
            deltas,
            mo.md(r"""
    The Ward relations are $Z_\xi=Z_A$ and $\delta Z_e+\delta Z_A/2=0$.
    Landau gauge is obtained after solving symbolically, so the intermediate
    $1/\xi$ gauge-fixing rule is never evaluated in isolation at zero.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the massless theory and the IR rearrangement prescription
    """)
    return


@app.cell
def _(Model, json, model):
    # The massless gallery uses a different, explicit IRR prescription: first set
    # the physical electron mass to zero, then FCLoopAddAuxiliaryMass[..., -M, 0]
    # directly replaces every massless denominator q² by q²-M. FourSeries then
    # Taylor-expands external momenta at fixed M through the UV degree. The n=0
    # choice omits the compensating mass terms of a UV-preserving rearrangement.
    # Reference: https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/RenormalizationMassless
    _massless_specification = json.loads(model.to_json())
    for _particle in _massless_specification["particles"]:
        if _particle["name"] in ("e-", "e+"):
            _particle["mass"] = "ZERO"
    massless_model = Model.from_json(json.dumps(_massless_specification))
    massless_electron, massless_photon = (
        massless_model.particle(_name) for _name in ("e-", "a")
    )
    massless_vertices = [
        vertex
        for vertex in massless_model.vertex_rules
        if sorted(vertex.particles)
        == sorted(
            [
                massless_electron.antiname,
                massless_electron.name,
                massless_photon.name,
            ]
        )
    ]
    assert len(massless_vertices) == 1
    return (
        massless_electron,
        massless_model,
        massless_photon,
        massless_vertices,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the massless projection kinematics
    """)
    return


@app.cell
def _(D, K, Kinematics, P, S, Symbols, s, sp, vacuum):
    (
        edge_momentum,
        dot_pattern,
        massless_scale,
        massless_a,
        massless_b,
        massless_args,
    ) = (
        Symbols.edge_momentum,
        sp.TensorPattern.dot,
        S("massless_scale"),
        S("massless_a_"),
        S("massless_b_"),
        S("massless_args___"),
    )
    massless_kinematics = Kinematics(D, momenta=[K(0), P(0), P(1)]).with_scalar_product(
        P(0), P(0), s
    )
    massless_loop_square = vacuum.scalar_product(K(0), K(0))
    massless_Qg, massless_Qpp = S("massless_Qg", "massless_Qpp")
    massless_photon_coordinate = S("massless_photon_coordinate")
    return (
        dot_pattern,
        edge_momentum,
        massless_Qg,
        massless_Qpp,
        massless_a,
        massless_args,
        massless_b,
        massless_kinematics,
        massless_loop_square,
        massless_photon_coordinate,
        massless_scale,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the three massless channels

    The folded projection routine compares all three prescriptions at integrand level before any pole is integrated.
    """)
    return


@app.cell
def _(
    massless_electron,
    massless_model,
    massless_photon,
    massless_vertices,
    project_massless_qed,
):
    massless_parts, massless_targets, massless_scalars, massless_diagrams = (
        {},
        set(),
        {},
        {},
    )
    massless_components, massless_component_scalars = {}, {}
    for _kind, _incoming, _outgoing in [
        ("electron_p", [massless_electron], [massless_electron]),
        ("photon", [massless_photon], [massless_photon]),
        ("vertex", [massless_electron], [massless_photon, massless_electron]),
    ]:
        _generated = massless_model.process(
            _incoming, _outgoing, vertex_allow=massless_vertices
        ).generate_diagrams(
            loops=1,
            max_vertices=len(_incoming) + len(_outgoing),
            maximum_bridges=0,
            self_energy=None,
            tadpoles=None,
            zero_snails=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == 1
        _diagram = _generated.diagrams[0]
        massless_diagrams[_kind] = _diagram
        _parts, _targets, _scalars, _components, _component_scalars = (
            project_massless_qed(_kind, _diagram)
        )
        massless_parts.update(_parts)
        massless_targets.update(_targets)
        massless_scalars.update(_scalars)
        massless_components.update(_components)
        massless_component_scalars.update(_component_scalars)
    return massless_diagrams, massless_parts, massless_targets


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reduce their vacuum integrals
    """)
    return


@app.cell
def _(IBPFamily, family, massless_targets):
    massless_solution = IBPFamily(family, name="qed_massless_irr").reduce_laporta(
        [list(target) for target in sorted(massless_targets)], max_depth=2
    )
    assert massless_solution.residuals == [[1]]
    return (massless_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check each prescription’s ultraviolet poles
    """)
    return


@app.cell
def _(
    D,
    M,
    Nf,
    epsilon,
    gmunu,
    integral,
    massless_Qg,
    massless_Qpp,
    massless_parts,
    massless_solution,
    ppmunu,
    s,
    xi,
    zero,
):
    massless_integrated, massless_uv_poles = {}, {}
    for _label, _terms in massless_parts.items():
        _expression = sum(
            (
                _coefficient * massless_solution.reduce(_power, integral=integral)
                for _power, _coefficient in _terms
            ),
            zero,
        ).together()
        massless_integrated[_label] = _expression
        massless_uv_poles[_label] = (
            _expression.replace(gmunu, massless_Qg)
            .replace(ppmunu, massless_Qpp)
            .replace(integral(1), M / epsilon)
            .replace(D, 4 - 2 * epsilon)
            .series(epsilon, 0, -1)
            .to_expression()
            .expand()
            .replace(massless_Qg, gmunu)
            .replace(massless_Qpp, ppmunu)
        )
        assert massless_uv_poles[_label].coefficient(epsilon**-2) == zero
    for _scheme in ("direct_irr", "selective_uv", "full_uv"):
        assert (
            massless_uv_poles[_scheme, "electron_p"] - xi / epsilon
        ).together() == zero
        assert (massless_uv_poles[_scheme, "vertex"] - xi / epsilon).together() == zero
        _reference = (
            Nf
            * (
                -4 * (s * gmunu - ppmunu) / 3
                + (4 * M * gmunu if _scheme != "full_uv" else zero)
            )
            / epsilon
        )
        assert (massless_uv_poles[_scheme, "photon"] - _reference).together() == zero
    # Each full-UV propagator retains -M/(K²-M)² at this order. The sum of the
    # two photon-bubble compensation terms reduces to -(D-2)²*Nf*A0(M)*g:
    # its -4*M*Nf*g/epsilon pole cancels the direct IRR auxiliary photon mass.
    assert (
        massless_integrated["full_uv", "photon"]
        - massless_integrated["direct_irr", "photon"]
        + (D - 2) ** 2 * Nf * integral(1) * gmunu
    ).together() == zero

    # Reuse the already generated local CT amplitudes and matching matrix. There
    # is no physical mass operator in the massless calculation: remove the
    # electron_m row and deltaZm column, preserving all other generated entries.
    return (massless_uv_poles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Restrict the local counterterm system
    """)
    return


@app.cell
def _(Matrix, ct_matrix, ct_rows, delta_m, unknowns, zero):
    _massless_indices = [0, 2, 3, 4, 5]
    massless_unknowns = [unknowns[position] for position in _massless_indices]
    massless_ct_matrix = Matrix.from_linear(
        5,
        5,
        [
            ct_matrix[_row, column].to_expression()
            for _row in _massless_indices
            for column in _massless_indices
        ],
    )
    assert massless_ct_matrix[4, 4].to_expression() == 2
    assert all(
        ct_rows[position].derivative(delta_m) == zero for position in _massless_indices
    )
    return massless_ct_matrix, massless_unknowns


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the massless pole-cancellation equations
    """)
    return


@app.cell
def _(
    M,
    Matrix,
    Nf,
    epsilon,
    gmunu,
    massless_ct_matrix,
    massless_uv_poles,
    ppmunu,
    s,
    xi,
    zero,
):
    _massless_photon_g = massless_uv_poles["direct_irr", "photon"].coefficient(gmunu)
    _massless_photon_pp = massless_uv_poles["direct_irr", "photon"].coefficient(ppmunu)
    _massless_rhs = Matrix.vec(
        [
            -massless_uv_poles["direct_irr", "electron_p"],
            -_massless_photon_g.coefficient(s),
            -xi * _massless_photon_pp,
            -massless_uv_poles["direct_irr", "vertex"],
            -_massless_photon_g.replace(s, zero) / M,
        ]
    )
    massless_deltas = massless_ct_matrix.solve(_massless_rhs)
    for _row, _reference in enumerate(
        [
            -xi / epsilon,
            -4 * Nf / (3 * epsilon),
            -4 * Nf / (3 * epsilon),
            2 * Nf / (3 * epsilon),
            -2 * Nf / epsilon,
        ]
    ):
        assert (
            massless_deltas[_row, 0].to_expression() - _reference
        ).together() == zero
    assert (
        massless_deltas[0, 0].to_expression()
        + massless_uv_poles["direct_irr", "vertex"]
    ).together() == zero
    assert (
        massless_deltas[3, 0].to_expression()
        + massless_deltas[1, 0].to_expression() / 2
    ).together() == zero
    _massless_residual = massless_ct_matrix * massless_deltas - _massless_rhs
    assert all(
        _massless_residual[_row, 0].to_expression().together() == zero
        for _row in range(5)
    )
    # Cancel the complete generated tensor/Dirac structures, including the
    # auxiliary-mass term; the matrix solution is not the only validation.
    return (massless_deltas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify all reconstructed counterterms
    """)
    return


@app.cell
def _(
    Replacement,
    ct_coefficients,
    massless_deltas,
    massless_solution,
    massless_unknowns,
    massless_uv_poles,
    zero,
):
    _massless_ct_rules = [
        Replacement(delta, massless_deltas[position, 0].to_expression())
        for position, delta in enumerate(massless_unknowns)
    ]
    for _kind in ("electron_p", "photon", "vertex"):
        assert (
            massless_uv_poles["direct_irr", _kind]
            + ct_coefficients[_kind].replace_multiple(_massless_ct_rules)
        ).together() == zero
    print(
        "Massless QED: literal n=0 IRR, full-UV contrast, five generated CT coefficients and Ward identities passed",
        massless_solution.stats,
    )
    return


@app.cell(hide_code=True)
def _(
    massless_ct_matrix,
    massless_deltas,
    massless_diagrams,
    massless_solution,
    massless_uv_poles,
    mo,
):
    mo.vstack(
        [
            mo.md(r"""
    ## Massless infrared rearrangement

    The [massless reference](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/RenormalizationMassless)
    sets the electron mass to zero and replaces every massless denominator
    $q^2$ by $q^2-M$. It then Taylor-expands external momenta at fixed $M$.
    First split the Feynman and longitudinal photon terms. Keep the Feynman
    term's single denominator and promote only the longitudinal remainder to
    power two, so both receive the specified auxiliary-mass replacement.

    The existing denominator builder, Symbolica series, tensor reducer and native
    IBP solver implement this prescription directly. A second construction
    freezes the full UV expansion's denominators before removing only its explicit
    auxiliary-mass compensation terms. The two agree **before IBP**.
    """),
            mo.hstack(list(massless_diagrams.values())),
            mo.ui.table([massless_solution.stats], selection=None),
            mo.md(r"**Massless photon pole: direct massification**"),
            massless_uv_poles["direct_irr", "photon"],
            mo.md(r"**Massless photon pole: full UV expansion**"),
            massless_uv_poles["full_uv", "photon"],
            mo.md(r"""
    Direct massification leaves $4N_f M g^{\mu\nu}/\epsilon$ in the photon pole.
    The full UV expansion retains compensation terms whose exact-$D$ contribution
    is $-N_f(D-2)^2A_0(M)g^{\mu\nu}$; their pole cancels that auxiliary term.
    The kinetic and vertex poles agree in both prescriptions.

    The same generated counterterm rules now give a five-by-five system: the
    fermion mass operator vanishes, so there is no equation for $Z_m$.
    The auxiliary pole requires $\delta Z_{Am}=-2N_f/\epsilon$.
    """),
            massless_ct_matrix,
            massless_deltas,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare MS and MSbar conventions
    """)
    return


@app.cell
def _(
    D,
    S,
    ct_coefficients,
    deltas,
    epsilon,
    massless_deltas,
    massless_unknowns,
    massless_uv_poles,
    one,
    unknowns,
    uv_poles,
    zero,
):
    # Compare the reference's MS and MSbar conventions at the same scale. OneLOop
    # divides by rGamma; (4*pi)^epsilon*rGamma=1+cDelta*epsilon+O(epsilon^2),
    # with cDelta=log(4*pi)-EulerGamma. Keep that real constant symbolic through
    # exact simplification. No finite part is inferred from a UV-expanded graph.
    scheme_shift = S("cDelta")
    assert (-2 / (D - 4)).replace(D, 4 - 2 * epsilon) == 1 / epsilon
    counterterms_by_scheme = {
        "MS": [deltas[row, 0].to_expression().together() for row in range(6)],
        "MSbar": [
            (
                epsilon * deltas[row, 0].to_expression() * (1 / epsilon + scheme_shift)
            ).together()
            for row in range(6)
        ],
    }
    _massless_counterterms_by_scheme = {
        "MS": [massless_deltas[row, 0].to_expression().together() for row in range(5)],
        "MSbar": [
            (
                epsilon
                * massless_deltas[row, 0].to_expression()
                * (1 / epsilon + scheme_shift)
            ).together()
            for row in range(5)
        ],
    }
    for _pole_set, _delta_names, _schemes in [
        (uv_poles, unknowns, counterterms_by_scheme),
        (
            {
                kind: massless_uv_poles["direct_irr", kind]
                for kind in ("electron_p", "photon", "vertex")
            },
            massless_unknowns,
            _massless_counterterms_by_scheme,
        ),
    ]:
        for _scheme, _constants in _schemes.items():
            for _label, _pole in _pole_set.items():
                _generated_ct = ct_coefficients[_label]
                for _delta, _constant in zip(_delta_names, _constants, strict=True):
                    _generated_ct = _generated_ct.replace(_delta, _constant)
                # These are the complete generated scalar/tensor CT coefficients,
                # retaining the external convention of the corresponding loop sector.
                _converted_uv = _pole * (
                    1 + epsilon * scheme_shift if _scheme == "MSbar" else one
                )
                assert (_generated_ct + _converted_uv).together() == zero

    # Actual OneLOop values check the measure conversion independently of the
    # counterterm solve. The remaining cDelta*P in MS is a finite scheme shift.
    return counterterms_by_scheme, scheme_shift


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the scalar-master normalization
    """)
    return


@app.cell
def _(epsilon, one, oneloop, scheme_shift, zero):
    for _master in [
        oneloop.A0(4, 7),
        oneloop.B0(-3, 4, 0, 7),
    ]:
        _finite_master, _pole_master, _double_master = oneloop.get_expression(_master)
        assert _double_master == zero
        _physical_master = (
            ((one + epsilon * scheme_shift) * (_pole_master / epsilon + _finite_master))
            .series(epsilon, 0, 0)
            .to_expression()
        )
        assert (
            _physical_master
            - _pole_master / epsilon
            - _finite_master
            - scheme_shift * _pole_master
        ).together().expand() == zero
        assert (
            _physical_master
            - _pole_master * (1 / epsilon + scheme_shift)
            - _finite_master
        ).together().expand() == zero

    print(
        "Generated QED CT amplitudes: MS/MSbar conversion and OneLOop measure checks passed"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Collect the derived renormalization constants
    """)
    return


@app.cell
def _(deltas, epsilon, massless_deltas):
    counterterm_names = ["Zψ", "Zm", "ZA", "Zξ", "Ze", "ZAm"]
    residues = [
        (epsilon * deltas[row, 0].to_expression()).together() for row in range(6)
    ]
    massless_names = ["Zψ", "ZA", "Zξ", "Ze", "ZAm"]
    massless_residues = [
        (epsilon * massless_deltas[row, 0].to_expression()).together()
        for row in range(5)
    ]
    return counterterm_names, massless_names, massless_residues, residues


@app.cell(hide_code=True)
def _(counterterm_names, counterterms_by_scheme, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## MS and MSbar at the same scale

    The [second QED reference](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/Renormalization2)
    compares $\Delta=1/\epsilon+c_\Delta$ with $1/\epsilon$, where
    $c_\Delta=\log(4\pi)-\gamma_E$. With $D=4-2\epsilon$,
    $-2/(D-4)=1/\epsilon$.

    [OneLOop, Eq. (2)](https://arxiv.org/abs/1007.4716) divides its masters by
    $r_\Gamma=\Gamma(1-\epsilon)^2\Gamma(1+\epsilon)/\Gamma(1-2\epsilon)$.
    Converting to $\mu^{2\epsilon}\int d^D\ell/(2\pi)^D$, with the common
    $i/(16\pi^2)$ stripped off, multiplies the result by
    $$(4\pi)^\epsilon r_\Gamma=1+c_\Delta\epsilon+O(\epsilon^2).$$
    Thus a normalized master $P/\epsilon+F$ becomes
    $P/\epsilon+F+c_\Delta P$. MS subtracts its pole; MSbar subtracts its pole
    and the $c_\Delta P$ term. Actual OneLOop tadpole and bubble values check this
    conversion independently, and the generated CT amplitudes cancel the
    corresponding loop terms in both schemes.

    The finite column below is the counterterm's **subtraction constant** at a
    common scale. It is not a computed finite loop amplitude.
    """),
            mo.ui.table(
                [
                    {"Constant": _name, "MS": str(_ms), "MSbar": str(_msbar)}
                    for _name, _ms, _msbar in zip(
                        counterterm_names,
                        counterterms_by_scheme["MS"],
                        counterterms_by_scheme["MSbar"],
                        strict=True,
                    )
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    flavor_count = mo.ui.slider(
        1, 6, step=1, value=1, label="Identical charged flavors Nf"
    )
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    mo.hstack([gauge_parameter, flavor_count, subtraction_scheme])
    return flavor_count, gauge_parameter, subtraction_scheme


@app.cell
def _(
    Nf,
    Symbol,
    flavor_count,
    gauge_parameter,
    massless_residues,
    residues,
    subtraction_scheme,
    xi,
):
    _point = {xi: gauge_parameter.value, Nf: flavor_count.value}
    selected_residues = [complex(value.evaluate(_point)).real for value in residues]
    selected_massless_residues = [
        complex(value.evaluate(_point)).real for value in massless_residues
    ]
    _c_delta = complex(((4 * Symbol.PI).log() - Symbol.EULER_GAMMA).evaluate({})).real
    finite_factor = _c_delta if subtraction_scheme.value == "MSbar" else 0.0
    ward_error = abs(selected_residues[4] + selected_residues[2] / 2)
    massless_ward_error = abs(
        selected_massless_residues[3] + selected_massless_residues[1] / 2
    )
    assert max(ward_error, massless_ward_error) < 1e-12
    assert selected_residues[2] == selected_residues[3]
    assert selected_massless_residues[1] == selected_massless_residues[2]
    assert selected_massless_residues[-1] == -2 * flavor_count.value
    return (
        finite_factor,
        massless_ward_error,
        selected_massless_residues,
        selected_residues,
        ward_error,
    )


@app.cell(hide_code=True)
def _(
    counterterm_names,
    finite_factor,
    massless_names,
    massless_ward_error,
    mo,
    selected_massless_residues,
    selected_residues,
    ward_error,
):
    mo.vstack(
        [
            mo.md(
                r"**Counterterms** $Z=1+a_4\delta Z$ — vary gauge, flavor count and scheme."
            ),
            *[
                mo.vstack(
                    [
                        mo.md("**" + _prescription + "**"),
                        mo.ui.table(
                            [
                                {
                                    "Constant": _name,
                                    "1/ε coefficient": value,
                                    "Finite subtraction coefficient": finite_factor
                                    * value,
                                }
                                for _name, value in zip(_names, _values, strict=True)
                            ],
                            selection=None,
                        ),
                    ]
                )
                for _prescription, _names, _values in (
                    ("Massive / full UV", counterterm_names, selected_residues),
                    (
                        "Massless / direct massification",
                        massless_names,
                        selected_massless_residues,
                    ),
                )
            ],
            mo.md(
                f"Ward residuals: **{ward_error:.1e}** (massive), **{massless_ward_error:.1e}** (massless)."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
