import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="One-loop QCD renormalization")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # One-loop QCD renormalization

    [Browse all notebooks](/) · [QED renormalization](/?file=hep/qed_renormalization.py) ·
    [Ghost–gluon vertex](/?file=hep/qcd_ghost_vertex.py) ·
    [Two-loop electron](/?file=hep/electron_two_loop.py)

    Generate the quark, ghost and gluon self-energies and the quark–gluon
    vertex in a **symbolic covariant gauge**. Shared color contractions,
    Dirac traces, UV expansion and native IBP give the six physical
    renormalization constants in the
    [FeynCalc MS/MSbar reference](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/Renormalization2).
    The quark mass remains symbolic throughout.

    We use $D=4-2\epsilon$, $a_4=g_s^2/(16\pi^2)$,
    $T_R=1/2$ and the gluon numerator
    $-i[g^{\mu\nu}-(1-\xi)q^\mu q^\nu/q^2]$.
    The auxiliary squared mass is $M=m_{\rm UV}^2>0$.
    The analytic tadpole pole $I(1)=M/\epsilon+O(1)$ is a supplied input,
    checked with OneLOop. IBP determines its rational coefficients.

    The full UV expansion retains the mass correction terms. They cancel
    auxiliary mass poles, so our two auxiliary mass counterterms vanish.
    A second calculation follows the [selective infrared rearrangement reference](https://feyncalc.github.io/FeynCalcExamples/QCD/OneLoop/Renormalization)
    directly, retaining the quark mass and adding an auxiliary mass only to
    massless propagators. It reproduces the nonzero gluon auxiliary counterterm;
    the six physical constants agree between prescriptions. Here **eight generated counterterm diagrams** supply the
    matching equations, and the controls compare MS and MSbar subtraction.
    The finite column contains subtraction constants, not finite loop amplitudes.
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
    from symbolica.community.tensor import (
        TensorExpression,
    )

    _set_namespace("qcd_ren")
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
def _(mo):
    mo.md("""
    ## Choose the covariant-gauge gluon propagator
    """)
    return


@app.cell(hide_code=True)
def _(
    CA,
    CF,
    D,
    K,
    M,
    Nc,
    P,
    Replacement,
    Symbols,
    TensorExpression,
    arguments,
    bis,
    coad,
    cof,
    coordinate,
    dA,
    den,
    edge_,
    family,
    gamma,
    gluon,
    index,
    kinematics,
    left,
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
    ordering,
    quad_,
    reducer,
    right,
    s,
    sp,
    value,
    wave,
):
    def project_qcd_diagram(_kind, _number, _diagram, _incoming, _loops):
        """Retain graph factors and project one color/Lorentz numerator into the vacuum family."""
        parts, irr_inputs = {}, {}
        _ports = {}
        if _kind != "ghost":
            for _edge in _diagram.external_edges:
                _is_gluon = _incoming == [gluon] or (
                    _kind in ("tree", "vertex") and _edge.external_index == 1
                )
                _rep = mink if _is_gluon else bis
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
        if _kind in ("tree", "vertex"):
            # Conjugate the tree's color tensor; the generated tree fixes the norm.
            _color_projector = (
                TensorExpression.color_t(dA, Nc)(_ports[1], _ports[0], _ports[2])
                .dirac_adjoint()
                .to_expression()
            )
            _numerator = (
                _numerator.replace(
                    sp.PortPattern.exact(sp.Representation.cof(3), index),
                    sp.PortPattern.exact(sp.Representation.cof(Nc), index),
                ).replace(
                    sp.PortPattern.exact(sp.Representation.coad(8), index),
                    sp.PortPattern.exact(sp.Representation.coad(dA), index),
                )
                * _color_projector
            )
            _fundamental_dimension = Nc
        else:
            _particle = _incoming[0]
            _rep, _numeric_dim, _symbolic_dim = (
                (cof, 3, Nc) if _kind == "quark" else (coad, 8, dA)
            )
            _color_slots = [
                slot.dual().to_expression()
                for slot in TensorExpression(_numerator).structure.slots
                if slot.to_expression().matches(
                    sp.PortPattern.exact(_rep(_numeric_dim), index)
                )
            ]
            assert len(_color_slots) == 2
            _color_indices = dict(
                next(
                    metric(*_color_slots).match(
                        _particle.color_sum(left, right), max_level=0
                    )
                )
            )
            _color_projector = _particle.color_sum(
                _color_indices[left], _color_indices[right]
            )
            _numerator = (
                (_numerator * _color_projector / _symbolic_dim)
                .replace(
                    sp.PortPattern.exact(sp.Representation.cof(3), index),
                    sp.PortPattern.exact(sp.Representation.cof(Nc), index),
                )
                .replace(
                    sp.PortPattern.exact(sp.Representation.coad(8), index),
                    sp.PortPattern.exact(sp.Representation.coad(dA), index),
                )
            )
            _fundamental_dimension = CA
        _numerator = (
            TensorExpression(_numerator)
            .simplify_algebra(
                contract="selected",
                representations=[sp.Representation.cof(Nc), sp.Representation.coad(dA)],
                gamma=False,
                color=True,
            )
            .to_dots()
            .to_expression()
        )
        # CF and CA are the shared representation-aware invariants;
        # the quark-loop trace uses the conventional fundamental index TR=1/2.
        _numerator = _numerator.replace_multiple(
            [
                # Keep invariant representation labels fixed while rewriting
                # scalar dimension factors. Earlier matches protect their subtrees.
                Replacement(CF, CF),
                Replacement(CA, CA),
                Replacement(sp.Representation.cof(Nc).dynkin_index(), one / 2),
                Replacement(dA, 2 * _fundamental_dimension * CF),
                Replacement(Nc, _fundamental_dimension),
            ]
        )
        # Keep the unexpanded graph numerator for the independent direct IRR path.
        _raw_numerator = _numerator
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
        if _incoming == [gluon]:
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
        if _kind in ("quark", "ghost"):
            # Amputated Grassmann two-point kernels omit external ordering.
            # Closed-loop signs and all other graph factors remain included.
            _raw = _diagram.overall_factor_expression()
            _removed = (_raw / _raw.replace(ordering(value), one)).replace(
                ordering(value), value
            )
            assert _removed == -one
            _factor /= _removed
        if _kind == "quark":
            _probes = [
                (
                    "quark_p",
                    gamma(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                        sp.PortPattern.exact(sp.Representation.mink(D), mu),
                    )
                    * P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu))
                    / (4 * s),
                ),
                (
                    "quark_m",
                    metric(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                    )
                    / (4 * mass),
                ),
            ]
        elif _kind in ("tree", "vertex"):
            _probes = [
                (
                    f"{_kind}_{_number}",
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
        if _loops:
            irr_inputs[_kind, _number] = (
                _diagram,
                _raw_numerator,
                _ports.copy(),
                _probes,
                _factor,
            )
        for _label, _projector in _probes:
            _traced = (
                TensorExpression((_numerator * _projector).expand())
                .simplify_algebra(
                    contract="dots", color=False, gamma=True, epsilon=True
                )
                .expand()
                .to_expression()
            )
            _scalar = family.rewrite_numerator(
                kinematics.apply(reducer.reduce(_traced)), [coordinate]
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
                assert not _coefficient.matches(K(arguments))
                _terms.append(([_power], _coefficient))
            parts[_label] = _terms

        return parts, irr_inputs

    return (project_qcd_diagram,)


@app.cell(hide_code=True)
def _(
    CA,
    CF,
    D,
    Nc,
    P,
    Replacement,
    Symbol,
    TensorExpression,
    a4,
    bis,
    coad,
    cof,
    ct_model,
    dA,
    dim,
    external_ordering,
    gamma,
    index,
    kinematics,
    left,
    mass,
    metric,
    mink,
    mu,
    nu,
    one,
    right,
    s,
    sp,
    tree,
    wave,
    zero,
):
    def project_qcd_counterterm(_kind, _diagram, _incoming):
        """Project a generated local insertion with its external-state convention."""
        coefficients = {}
        assert len(_diagram.internal_edges) == 0
        assert _diagram.symmetry_factor == one
        _factor = (
            _diagram.overall_factor_expression(evaluate=True)
            * _diagram.numerator_prefactor_expression()
        )
        assert _factor == (
            external_ordering if _kind in ("quark", "ghost", "vertex") else one
        )
        _ports = {}
        if _kind != "ghost":
            for _edge in _diagram.external_edges:
                _rep = (
                    mink
                    if _kind == "gluon"
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
        )
        if _kind == "vertex":
            _color_projector = (
                TensorExpression.color_t(dA, Nc)(_ports[1], _ports[0], _ports[2])
                .dirac_adjoint()
                .to_expression()
            )
            _numerator = (
                _numerator.replace(
                    sp.PortPattern.exact(sp.Representation.cof(3), index),
                    sp.PortPattern.exact(sp.Representation.cof(Nc), index),
                ).replace(
                    sp.PortPattern.exact(sp.Representation.coad(8), index),
                    sp.PortPattern.exact(sp.Representation.coad(dA), index),
                )
                * _color_projector
            )
            _fundamental_dimension = Nc
        else:
            _particle = _incoming[0]
            _rep, _numeric_dim, _symbolic_dim = (
                (cof, 3, Nc) if _kind == "quark" else (coad, 8, dA)
            )
            _color_slots = [
                slot.dual().to_expression()
                for slot in TensorExpression(_numerator).structure.slots
                if slot.to_expression().matches(
                    sp.PortPattern.exact(_rep(_numeric_dim), index)
                )
            ]
            assert len(_color_slots) == 2, (_kind, _numerator)
            _color_indices = dict(
                next(
                    metric(*_color_slots).match(
                        _particle.color_sum(left, right), max_level=0
                    )
                )
            )
            _color_projector = _particle.color_sum(
                _color_indices[left], _color_indices[right]
            )
            _numerator = (
                (_numerator * _color_projector / _symbolic_dim)
                .replace(
                    sp.PortPattern.exact(sp.Representation.cof(3), index),
                    sp.PortPattern.exact(sp.Representation.cof(Nc), index),
                )
                .replace(
                    sp.PortPattern.exact(sp.Representation.coad(8), index),
                    sp.PortPattern.exact(sp.Representation.coad(dA), index),
                )
            )
            _fundamental_dimension = CA
        _numerator = (
            TensorExpression(_numerator)
            .simplify_algebra(
                contract="selected",
                representations=[sp.Representation.cof(Nc), sp.Representation.coad(dA)],
                gamma=False,
                color=True,
            )
            .to_dots()
            .to_expression()
        )
        _numerator = (
            _numerator.replace_multiple(
                [
                    # Keep invariant representation labels fixed while rewriting
                    # scalar dimension factors. Earlier matches protect their subtrees.
                    Replacement(CF, CF),
                    Replacement(CA, CA),
                    Replacement(sp.Representation.cof(Nc).dynkin_index(), one / 2),
                    Replacement(dA, 2 * _fundamental_dimension * CF),
                    Replacement(Nc, _fundamental_dimension),
                ]
            )
            .replace(
                sp.PortPattern.exact(sp.Representation.mink(dim), index),
                sp.PortPattern.exact(sp.Representation.mink(D), index),
            )
            .replace(
                sp.PortPattern.exact(sp.Representation.mink(dim)),
                sp.PortPattern.exact(sp.Representation.mink(D)),
            )
        )
        if _kind == "quark":
            _probes = [
                (
                    "quark_p",
                    gamma(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                        sp.PortPattern.exact(sp.Representation.mink(D), mu),
                    )
                    * P(0, sp.PortPattern.exact(sp.Representation.mink(D), mu))
                    / (4 * s),
                ),
                (
                    "quark_m",
                    metric(
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[0]),
                        sp.PortPattern.exact(sp.Representation.bis(4), _ports[1]),
                    )
                    / (4 * mass),
                ),
            ]
            _normalization = Symbol.I * a4 * external_ordering
        elif _kind == "vertex":
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
            _normalization = tree * a4
        else:
            if _kind == "gluon":
                _numerator = _numerator.replace(
                    sp.PortPattern.exact(sp.Representation.mink(D), _ports[0]),
                    sp.PortPattern.exact(sp.Representation.mink(D), mu),
                ).replace(
                    sp.PortPattern.exact(sp.Representation.mink(D), _ports[1]),
                    sp.PortPattern.exact(sp.Representation.mink(D), nu),
                )
            _probes = [(_kind, one)]
            _normalization = Symbol.I * a4
            if _kind == "ghost":
                _normalization *= external_ordering
        for _label, _projector in _probes:
            _trace = (
                TensorExpression((_numerator * _projector).expand())
                .simplify_algebra(
                    contract="dots", color=False, gamma=True, epsilon=True
                )
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

    return (project_qcd_counterterm,)


@app.cell(hide_code=True)
def _(
    D,
    K,
    M,
    Nf,
    P,
    Replacement,
    S,
    Symbol,
    TensorExpression,
    beyond_uv,
    den,
    dim,
    edge_,
    gluon,
    gs,
    index,
    irr_a,
    irr_b,
    irr_coordinate,
    irr_dot,
    irr_edge_momentum,
    irr_kinematics,
    irr_scaling,
    mass,
    mass_,
    mom_,
    mu,
    nu,
    one,
    quad_,
    reducer,
    s,
    sp,
    tree,
    vacuum,
    xi,
    zero,
):
    def rearrange_qcd_diagram(_kind, _number, _record, _massless):
        """Massify primitive propagators, Taylor-expand, and preserve the longitudinal components independently."""
        _diagram, _raw, _ports, _probes, _factor = _record
        _irr_scalars = {}
        if _massless:
            _raw = _raw.replace(mass, zero)
        # Separate every primitive longitudinal 1/q² before changing propagators.
        # A common squared denominator would spuriously massify the Feynman term.
        _tags = {
            edge.id: S(f"qcd_irr::gluon_{edge.id}")
            for edge in _diagram.internal_edges
            if edge.particle_name == gluon.name
        }
        _tagged = _raw
        for _edge_id, _tag in _tags.items():
            _tagged = _tagged.replace(
                irr_dot(
                    irr_edge_momentum(
                        _edge_id, sp.PortPattern.exact(sp.Representation.mink(4))
                    ),
                    irr_edge_momentum(
                        _edge_id, sp.PortPattern.exact(sp.Representation.mink(4))
                    ),
                ),
                _tag,
            )
        _components = (
            _tagged.expand().coefficient_list(*_tags.values())
            if _tags
            else [(one, _tagged)]
        )
        _reconstructed = zero
        for _monomial, _numerator in _components:
            _powers = {}
            for _edge_id, _tag in _tags.items():
                _exponent = int(
                    (_monomial.derivative(_tag) * _tag / _monomial).together()
                )
                assert _exponent in (0, -1), (_kind, _exponent, _monomial)
                _powers[_edge_id] = 1 - _exponent
                _monomial = _monomial.replace(
                    _tag,
                    irr_dot(
                        irr_edge_momentum(
                            _edge_id,
                            sp.PortPattern.exact(sp.Representation.mink(4)),
                        ),
                        irr_edge_momentum(
                            _edge_id,
                            sp.PortPattern.exact(sp.Representation.mink(4)),
                        ),
                    ),
                )
            _reconstructed += _monomial * _numerator
            if any(_power == 2 for _power in _powers.values()):
                assert _numerator.replace(xi, one).expand() == zero
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
            # Keep massive quark denominators intact; only massless ones get M.
            for _match in list(_denominator.match(den(edge_, mom_, mass_, quad_))):
                _vals = dict(_match)
                _propagator_mass = (
                    _vals[mass_].replace(mass, zero) if _massless else _vals[mass_]
                )
                _quadratic = (
                    _vals[quad_].replace(mass, zero) if _massless else _vals[quad_]
                )
                _denominator = _denominator.replace(
                    den(_vals[edge_], _vals[mom_], _vals[mass_], _vals[quad_]),
                    _quadratic - (M if _propagator_mass == zero else zero),
                )
            _numerator = (
                _diagram.momentum_basis()
                .route_expression(_numerator)
                .replace(
                    sp.PortPattern.exact(sp.Representation.mink(dim), index),
                    sp.PortPattern.exact(sp.Representation.mink(D), index),
                )
                .replace(
                    sp.PortPattern.exact(sp.Representation.mink(dim)),
                    sp.PortPattern.exact(sp.Representation.mink(D)),
                )
            )
            if _kind in ("gluon_loop", "ghost_loop", "quark_loop", "tadpole"):
                _numerator = _numerator.replace(
                    sp.PortPattern.exact(sp.Representation.mink(D), _ports[0]),
                    sp.PortPattern.exact(sp.Representation.mink(D), mu),
                ).replace(
                    sp.PortPattern.exact(sp.Representation.mink(D), _ports[1]),
                    sp.PortPattern.exact(sp.Representation.mink(D), nu),
                )
            for _label, _projector in _probes:
                if _massless and _label == "quark_m":
                    continue
                _traced = (
                    TensorExpression((_numerator * _projector).expand())
                    .simplify_algebra(
                        contract="dots", color=False, gamma=True, epsilon=True
                    )
                    .expand()
                    .to_expression()
                )
                _direct = irr_kinematics.apply(_traced / _denominator).replace(
                    vacuum.scalar_product(K(0), K(0)), irr_coordinate
                )
                _direct = _direct.replace_multiple(
                    [
                        Replacement(
                            irr_dot(
                                K(
                                    0,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                                P(
                                    irr_a,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                            ),
                            irr_scaling
                            * irr_dot(
                                K(
                                    0,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                                P(
                                    irr_a,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                            ),
                        ),
                        Replacement(
                            irr_dot(
                                P(
                                    irr_a,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                                K(
                                    0,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                            ),
                            irr_scaling
                            * irr_dot(
                                P(
                                    irr_a,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                                K(
                                    0,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                            ),
                        ),
                        Replacement(
                            irr_dot(
                                P(
                                    irr_a,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                                P(
                                    irr_b,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                            ),
                            irr_scaling**2
                            * irr_dot(
                                P(
                                    irr_a,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                                P(
                                    irr_b,
                                    sp.PortPattern.exact(sp.Representation.mink(D)),
                                ),
                            ),
                        ),
                        Replacement(
                            P(
                                irr_a,
                                sp.PortPattern.exact(sp.Representation.mink(D), index),
                            ),
                            irr_scaling
                            * P(
                                irr_a,
                                sp.PortPattern.exact(sp.Representation.mink(D), index),
                            ),
                        ),
                        Replacement(s, irr_scaling**2 * s),
                    ]
                )
                # The pslash projector lowers the quark external degree by one.
                # Two-point gluon/ghost tensors need degree two; the vertex needs zero.
                _degree = (
                    2
                    if _kind
                    in (
                        "gluon_loop",
                        "ghost_loop",
                        "quark_loop",
                        "tadpole",
                        "ghost",
                    )
                    else 0
                )
                _taylor = (
                    _direct.series(irr_scaling, 0, _degree + int(_massless))
                    .to_expression()
                    .expand()
                )
                _direct = _taylor.replace(irr_scaling, one)
                if _massless:
                    # The reference tags the first term beyond the divergence
                    # degree; prove that it contributes no UV pole after IBP.
                    _direct += (beyond_uv - one) * _taylor.coefficient(
                        irr_scaling ** (_degree + 1)
                    )
                _scalar = irr_kinematics.apply(reducer.reduce(_direct)).replace(
                    vacuum.scalar_product(K(0), K(0)), irr_coordinate
                )
                _scalar = (
                    (
                        _scalar
                        * _factor
                        / gs**2
                        * (Symbol.I / tree if _kind == "vertex" else one)
                        * (Nf if _kind == "quark_loop" else one)
                    )
                    .together()
                    .expand()
                )
                _irr_scalars[_label] = (
                    _irr_scalars.get(_label, zero) + _scalar
                ).together()
        assert (_reconstructed - _raw).together() == zero

        return _irr_scalars

    return (rearrange_qcd_diagram,)


@app.cell(hide_code=True)
def _(M, coordinate, irr_coordinate, mass, zero):
    def rearranged_tadpole_terms(_scalar, _massless):
        """Separate physical and auxiliary tadpole poles and prove exact reconstruction."""
        _irr_targets = set()
        _apart = _scalar.apart(irr_coordinate)
        assert (_apart - _scalar).together() == zero
        _reconstructed = zero
        _terms = []
        for _squared_mass in (M,) if _massless else (M, mass**2):
            _principal = (
                _apart.replace(irr_coordinate, coordinate + _squared_mass)
                .series(coordinate, 0, -1)
                .to_expression()
                .expand()
            )
            _reconstructed += _principal.replace(
                coordinate, irr_coordinate - _squared_mass
            )
            for _monomial, _coefficient in _principal.coefficient_list(coordinate):
                if _coefficient == zero:
                    continue
                _power = -int(
                    (
                        _monomial.derivative(coordinate) * coordinate / _monomial
                    ).together()
                )
                assert _monomial == coordinate ** (-_power) and _power > 0
                _terms.append((_squared_mass, _power, _coefficient))
                _irr_targets.add((_power,))
        assert (_scalar - _reconstructed).together() == zero

        return _terms, _irr_targets

    return (rearranged_tadpole_terms,)


@app.cell(hide_code=True)
def _(
    M,
    Symbol,
    ZA,
    ZAm,
    Zc,
    Zcm,
    Zm,
    Zq,
    Zxi,
    a4,
    ghost,
    gluon,
    json,
    mass,
    one,
    quark,
    xi,
):
    def qcd_two_point_definition(source):
        """Encode the local quark, gluon and ghost two-point operators."""
        spec = json.loads(json.dumps(source))
        spec["orders"].append({"name": "CT", "expansion_order": 1, "hierarchy": 1})
        for _label, _particles, _spins, _color, _lorentz, _coupling in [
            (
                "qq_kinetic",
                [quark.antiname, quark.name],
                [2, 2],
                "Identity(1,2)",
                "Gamma(dummy(1),idx(1,1),idx(1,2))*P(dummy(1),2)",
                Symbol.I * (Zq - one),
            ),
            (
                "qq_mass",
                [quark.antiname, quark.name],
                [2, 2],
                "Identity(1,2)",
                "Identity(idx(1,1),idx(1,2))",
                -Symbol.I * mass * (Zq * Zm - one),
            ),
            (
                "gg_kinetic",
                [gluon.name] * 2,
                [3, 3],
                "Identity(1,2)",
                "Metric(idx(1,1),idx(1,2))*P(dummy(1),1)*P(dummy(1),1)-P(idx(1,1),1)*P(idx(1,2),1)",
                -Symbol.I * (ZA - one),
            ),
            (
                "gg_gauge",
                [gluon.name] * 2,
                [3, 3],
                "Identity(1,2)",
                "P(idx(1,1),1)*P(idx(1,2),1)",
                -Symbol.I * (ZA / Zxi - one) / xi,
            ),
            (
                "gg_auxmass",
                [gluon.name] * 2,
                [3, 3],
                "Identity(1,2)",
                "Metric(idx(1,1),idx(1,2))",
                Symbol.I * M * (ZAm**2 - one),
            ),
            (
                "ghost_kinetic",
                [ghost.antiname, ghost.name],
                [-1, -1],
                "Identity(1,2)",
                "P(dummy(1),2)*P(dummy(1),2)",
                Symbol.I * (Zc - one),
            ),
            (
                "ghost_auxmass",
                [ghost.antiname, ghost.name],
                [-1, -1],
                "Identity(1,2)",
                "1",
                Symbol.I * M * (Zcm**2 - one) / 2,
            ),
        ]:
            spec["lorentz_structures"].append(
                {"name": "CT_L_" + _label, "spins": _spins, "structure": _lorentz}
            )
            spec["couplings"].append(
                {
                    "name": "CT_GC_" + _label,
                    "expression": repr(_coupling.series(a4, 0, 1).to_expression()),
                    "orders": [["QCD", 2], ["CT", 1]],
                    "value": None,
                }
            )
            spec["vertex_rules"].append(
                {
                    "name": "CT_" + _label,
                    "particles": _particles,
                    "color_structures": [_color],
                    "lorentz_structures": ["CT_L_" + _label],
                    "couplings": [["CT_GC_" + _label]],
                }
            )
        # Copy every color/Lorentz slot of the actual model rule and dress its coupling.
        return spec

    return (qcd_two_point_definition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
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
    Nc, dA = S("Nc", "dA")
    CF = sp.Representation.cof(Nc).casimir()
    CA = sp.Representation.coad(dA).casimir()
    _specification = json.loads(model.to_json())
    for _propagator in _specification["propagators"]:
        if _propagator["particle"] == "g":
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
    return (
        CA,
        CF,
        D,
        M,
        Nc,
        Nf,
        coordinate,
        dA,
        epsilon,
        integral,
        mUV,
        model,
        s,
        xi,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the color and Lorentz patterns
    """)
    return


@app.cell
def _(S, Symbols, hep, model, sp):
    K, P = hep.Kinematics.loop_momentum, hep.Kinematics.external_momentum
    mass = model.particle("b").mass
    gs = model.parameter("G").symbol
    mink, bis, gamma, metric = (
        sp.Representation.mink,
        sp.Representation.bis,
        sp.TensorName.dirac_gamma().to_expression(),
        sp.TensorName.g().to_expression(),
    )
    cof, coad = sp.Representation.cof, sp.Representation.coad
    index, dim, wave, mu, nu, left, right = S(
        "index_",
        "dim_",
        "wave_",
        "mu",
        "nu",
        "left_",
        "right_",
    )
    den, edge_, mom_, mass_, quad_ = (
        Symbols.denominator,
        S("edge_"),
        S("mom_"),
        S("mass_"),
        S("quad_"),
    )
    ordering, value, arguments = S(
        "feynkit_generator_factor::ExternalFermionOrderingSign",
        "value_",
        "arguments___",
    )
    return (
        K,
        P,
        arguments,
        bis,
        coad,
        cof,
        den,
        dim,
        edge_,
        gamma,
        gs,
        index,
        left,
        mass,
        mass_,
        metric,
        mink,
        mom_,
        mu,
        nu,
        ordering,
        quad_,
        right,
        value,
        wave,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the vacuum family and open tensor basis
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
    ## Select the QCD interactions
    """)
    return


@app.cell
def _(model):
    quark, gluon, ghost = (model.particle(_name) for _name in ("b", "g", "ghG"))
    interaction_rules = {}
    for _label, _particles in [
        ("qqg", [quark.antiname, quark.name, gluon.name]),
        ("ccg", [ghost.antiname, ghost.name, gluon.name]),
        ("ggg", [gluon.name] * 3),
        ("gggg", [gluon.name] * 4),
    ]:
        interaction_rules[_label] = [
            vertex
            for vertex in model.vertex_rules
            if sorted(vertex.particles) == sorted(_particles)
        ]
        assert len(interaction_rules[_label]) == 1
    return ghost, gluon, interaction_rules, quark


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and project the bare diagrams
    """)
    return


@app.cell
def _(ghost, gluon, interaction_rules, model, project_qcd_diagram, quark):
    bare_parts, diagrams, irr_inputs = {}, {}, {}
    for _kind, _incoming, _outgoing, _loops, _vertices, _count in [
        ("tree", [quark], [gluon, quark], 0, interaction_rules["qqg"], 1),
        ("quark", [quark], [quark], 1, interaction_rules["qqg"], 1),
        ("ghost", [ghost], [ghost], 1, interaction_rules["ccg"], 1),
        ("gluon_loop", [gluon], [gluon], 1, interaction_rules["ggg"], 1),
        ("ghost_loop", [gluon], [gluon], 1, interaction_rules["ccg"], 1),
        ("quark_loop", [gluon], [gluon], 1, interaction_rules["qqg"], 1),
        ("tadpole", [gluon], [gluon], 1, interaction_rules["gggg"], 1),
        (
            "vertex",
            [quark],
            [gluon, quark],
            1,
            interaction_rules["qqg"] + interaction_rules["ggg"],
            2,
        ),
    ]:
        _generated = model.process(
            _incoming, _outgoing, vertex_allow=_vertices
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
        assert len(_generated.diagrams) == _count
        diagrams[_kind] = _generated.diagrams
        for _number, _diagram in enumerate(_generated.diagrams):
            _terms, _irr_input = project_qcd_diagram(
                _kind, _number, _diagram, _incoming, _loops
            )
            bare_parts.update(_terms)
            irr_inputs.update(_irr_input)
    return bare_parts, diagrams, irr_inputs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fix the tree convention and reduce the integrals
    """)
    return


@app.cell
def _(CF, IBPFamily, Nc, Symbol, bare_parts, family, gs, oneloop, zero):
    _tree_terms = bare_parts["tree_0"]
    loop_parts = {name: terms for name, terms in bare_parts.items() if name != "tree_0"}
    assert len(_tree_terms) == 1 and _tree_terms[0][0] == [0]
    tree = _tree_terms[0][1]
    assert (tree + Symbol.I * gs * Nc * CF).together() == zero
    solution = IBPFamily(family, name="qcd_one_loop").reduce_laporta(
        sorted({tuple(p) for _terms in loop_parts.values() for p, c in _terms}),
        max_depth=2,
    )
    assert solution.residuals == [[1]]
    assert abs(complex(oneloop.a0(1.0, 1.0)[1]) - 1) < 1e-12
    return loop_parts, solution, tree


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extract the ultraviolet poles
    """)
    return


@app.cell
def _(
    CA,
    CF,
    D,
    M,
    Nc,
    Nf,
    Symbol,
    epsilon,
    gs,
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
    _reduced, diagram_poles = {}, {}
    for _label, _terms in loop_parts.items():
        _expression = sum(
            (c * solution.reduce(p, integral=integral) for p, c in _terms), zero
        ).together()
        _reduced[_label] = _expression
        _normalized = (
            _expression
            * (Symbol.I / tree if _label.startswith("vertex") else one)
            / gs**2
        )
        if _label == "quark_loop":
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
        # SU(N), T_R=1/2: C_A - 2 C_F = 1/N. Fierz output can
        # retain 1/N after division by the tree's color normalization.
        _pole = _pole.replace(1 / Nc, CA - 2 * CF).expand()
        diagram_poles[_label] = _pole
        assert _pole.derivative(M).expand() == zero
        assert _pole.derivative(mass).expand() == zero
        assert _pole.coefficient(epsilon**-2) == zero
    return (diagram_poles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the complete QCD pole coefficients
    """)
    return


@app.cell
def _(CA, CF, Nf, diagram_poles, epsilon, gmunu, ppmunu, s, xi, zero):
    uv_poles = dict(diagram_poles)
    uv_poles["gluon"] = sum(
        (
            uv_poles[_label]
            for _label in ("gluon_loop", "ghost_loop", "quark_loop", "tadpole")
        ),
        zero,
    ).expand()
    uv_poles["vertex"] = (uv_poles["vertex_0"] + uv_poles["vertex_1"]).expand()
    _expected = [
        CF * xi / epsilon,
        -CF * (xi + 3) / epsilon,
        CA * (xi - 3) * s / (4 * epsilon),
        ((13 - 3 * xi) * CA - 4 * Nf) * (s * gmunu - ppmunu) / (6 * epsilon),
        (CF * xi + CA * (xi + 3) / 4) / epsilon,
    ]
    for _label, _reference in zip(
        ("quark_p", "quark_m", "ghost", "gluon", "vertex"), _expected, strict=True
    ):
        assert (uv_poles[_label] - _reference).together() == zero
    return (uv_poles,)


@app.cell
def _(diagrams, mo, uv_poles):
    mo.vstack(
        [
            mo.md("## Generated diagrams and UV poles"),
            mo.ui.tabs(
                {
                    "Quark": mo.vstack(
                        [
                            mo.hstack(diagrams["quark"]),
                            uv_poles["quark_p"],
                            uv_poles["quark_m"],
                        ]
                    ),
                    "Ghost": mo.vstack(
                        [mo.hstack(diagrams["ghost"]), uv_poles["ghost"]]
                    ),
                    "Gluon": mo.vstack(
                        [
                            mo.hstack(diagrams["gluon_loop"] + diagrams["ghost_loop"]),
                            mo.hstack(diagrams["quark_loop"] + diagrams["tadpole"]),
                            uv_poles["gluon"],
                        ]
                    ),
                    "Quark–gluon vertex": mo.vstack(
                        [mo.hstack(diagrams["vertex"]), uv_poles["vertex"]]
                    ),
                }
            ),
            mo.md(
                r"The self-energies multiply $i a_4$ and the external color identity. Quark entries are coefficients of $\not p$ and $m$. The vertex entry multiplies its generated tree value times $a_4$. Both vertex diagrams contribute, and the gluon pole becomes transverse only after summing its loop classes."
            ),
        ]
    )
    return


@app.cell
def _(family, integral, mo, solution):
    mo.vstack(
        [
            mo.md("## One shared vacuum family"),
            family,
            mo.md(
                r"The longitudinal gluon denominators cancel after tensor projection. All scalar terms belong to this family; the calculation rejects hidden loop momenta in their coefficients. Native IBP reduces $I(1),\ldots,I(4)$ to $I(1)$:"
            ),
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.md(f"**I({_power})**"),
                            solution.reduce([_power], integral=integral),
                        ]
                    )
                    for _power in range(1, 5)
                ]
            ),
            mo.ui.table([solution.stats], selection=None),
            mo.md(
                "The tadpole pole is an analytic input. A finite-depth IBP residual alone does not certify a minimal master basis."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the local counterterm operators
    """)
    return


@app.cell
def _(S, interaction_rules, json, model, one):
    # Local counterterm operators, with Z=1+a4*deltaZ, a4=gs²/(16*pi²).
    # Unknowns: Zq, Zm, ZA, Zxi, Zc, Zg, ZAm, Zcm. The local operator basis
    # is an explicit model input; all matching entries come from generated graphs.
    _qqg = interaction_rules["qqg"]
    spec = json.loads(model.to_json())
    vertex_definition = next(
        v for v in spec["vertex_rules"] if v["name"] == _qqg[0].name
    )
    # Auxiliary operators follow the reference model:
    # M*(ZAm²-1)*A²/2 and M*(Zcm²-1)*cbar*c/2. Identical gluons supply
    # the factor two in their local rule; the distinct ghost fields do not.
    a4 = S("qcd_ct::a4")
    unknowns = list(
        S(
            "qcd_ct::deltaZq",
            "qcd_ct::deltaZm",
            "qcd_ct::deltaZA",
            "qcd_ct::deltaZxi",
            "qcd_ct::deltaZc",
            "qcd_ct::deltaZg",
            "qcd_ct::deltaZAm",
            "qcd_ct::deltaZcm",
        )
    )
    Zq, Zm, ZA, Zxi, Zc, Zg, ZAm, Zcm = (one + a4 * _delta for _delta in unknowns)
    return (
        ZA,
        ZAm,
        Zc,
        Zcm,
        Zg,
        Zm,
        Zq,
        Zxi,
        a4,
        spec,
        unknowns,
        vertex_definition,
    )


@app.cell
def _(qcd_two_point_definition, spec):
    operator_spec = qcd_two_point_definition(spec)
    return (operator_spec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Preserve the model’s quark–gluon vertex convention
    """)
    return


@app.cell
def _(ZA, Zg, Zq, a4, json, model, one, operator_spec, vertex_definition):
    completed_spec = json.loads(json.dumps(operator_spec))
    _ct_vertex = json.loads(json.dumps(vertex_definition))
    _ct_vertex["name"] = "CT_qqg"
    for _row, _couplings in enumerate(_ct_vertex["couplings"]):
        for _col, _coupling in enumerate(_couplings):
            if _coupling is None:
                continue
            _dressed = model.expand_couplings(model.coupling(_coupling).symbol) * (
                Zq * Zg * ZA.sqrt() - one
            )
            _name = f"CT_qqg_{_row}_{_col}"
            completed_spec["couplings"].append(
                {
                    "name": _name,
                    "expression": repr(_dressed.series(a4, 0, 1).to_expression()),
                    "orders": [["QCD", 3], ["CT", 1]],
                    "value": None,
                }
            )
            _couplings[_col] = _name
    completed_spec["vertex_rules"].append(_ct_vertex)
    return (completed_spec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select counterterm vertices
    """)
    return


@app.cell
def _(Model, completed_spec, diagrams, json):
    ct_model = Model.from_json(json.dumps(completed_spec))
    ct_quark, ct_gluon, ct_ghost = (
        ct_model.particle(_name) for _name in ("b", "g", "ghG")
    )
    ct_vertices = [v for v in ct_model.vertex_rules if v.name.startswith("CT_")]
    external_ordering = diagrams["tree"][0].overall_factor_expression(evaluate=True)
    return (
        ct_ghost,
        ct_gluon,
        ct_model,
        ct_quark,
        ct_vertices,
        external_ordering,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the local counterterm amplitudes
    """)
    return


@app.cell
def _(
    ct_ghost,
    ct_gluon,
    ct_model,
    ct_quark,
    ct_vertices,
    project_qcd_counterterm,
    zero,
):
    ct_diagrams, ct_coefficients = {}, {}
    for _kind, _incoming, _outgoing, _count, _qcd_order in [
        ("quark", [ct_quark], [ct_quark], 2, 2),
        ("gluon", [ct_gluon], [ct_gluon], 3, 2),
        ("ghost", [ct_ghost], [ct_ghost], 2, 2),
        ("vertex", [ct_quark], [ct_gluon, ct_quark], 1, 3),
    ]:
        # CT order is perturbative bookkeeping; two-point insertions have no loops.
        # Bound vertices explicitly to exclude arbitrarily long insertion chains.
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
        ).generate_diagrams(coupling_orders={"QCD": _qcd_order, "CT": 1}, **_options)
        assert len(_generated.diagrams) == _count
        assert (
            not ct_model.process(_incoming, _outgoing, vertex_allow=ct_vertices)
            .generate_diagrams(coupling_orders={"CT": 0}, **_options)
            .diagrams
        )
        ct_diagrams[_kind] = _generated.diagrams
        for _diagram in _generated.diagrams:
            for _label, _coefficient in project_qcd_counterterm(
                _kind, _diagram, _incoming
            ).items():
                ct_coefficients[_label] = (
                    ct_coefficients.get(_label, zero) + _coefficient
                )
    return ct_coefficients, ct_diagrams


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build the linear cancellation system
    """)
    return


@app.cell
def _(M, Matrix, ct_coefficients, gmunu, ppmunu, s, unknowns, xi, zero):
    _ct_g = ct_coefficients["gluon"].coefficient(gmunu)
    _ct_pp = ct_coefficients["gluon"].coefficient(ppmunu)
    _ct_rows = [
        ct_coefficients["quark_p"],
        ct_coefficients["quark_m"],
        _ct_g.coefficient(s),
        xi * _ct_pp,
        ct_coefficients["ghost"].coefficient(s),
        ct_coefficients["vertex"],
        _ct_g.replace(s, zero) / M,
        ct_coefficients["ghost"].replace(s, zero) / M,
    ]
    _entries = [r.derivative(z).together() for r in _ct_rows for z in unknowns]
    assert all(e.derivative(z) == zero for e in _entries for z in unknowns)
    for _i, _row in enumerate(_ct_rows):
        assert (
            _row - sum((_entries[8 * _i + j] * z for j, z in enumerate(unknowns)), zero)
        ).together() == zero
    ct_matrix = Matrix.from_linear(8, 8, _entries)
    assert ct_matrix[6, 6].to_expression() == 2
    assert ct_matrix[7, 7].to_expression() == 1
    assert (
        ct_coefficients["gluon"] - _ct_g * gmunu - _ct_pp * ppmunu
    ).together() == zero
    assert _ct_g.derivative(s).derivative(s) == zero
    assert ct_coefficients["ghost"].derivative(s).derivative(s) == zero
    return (ct_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve and check the renormalization constants
    """)
    return


@app.cell
def _(
    CA,
    CF,
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
    _gluon_g = uv_poles["gluon"].coefficient(gmunu)
    _gluon_pp = uv_poles["gluon"].coefficient(ppmunu)
    _rhs = Matrix.vec(
        [
            -uv_poles["quark_p"],
            -uv_poles["quark_m"],
            -_gluon_g.coefficient(s),
            -xi * _gluon_pp,
            -uv_poles["ghost"].coefficient(s),
            -uv_poles["vertex"],
            -_gluon_g.replace(s, zero) / M,
            -uv_poles["ghost"].replace(s, zero) / M,
        ]
    )
    deltas = ct_matrix.solve(_rhs)
    _references = [
        -CF * xi / epsilon,
        -3 * CF / epsilon,
        ((13 - 3 * xi) * CA - 4 * Nf) / (6 * epsilon),
        ((13 - 3 * xi) * CA - 4 * Nf) / (6 * epsilon),
        CA * (3 - xi) / (4 * epsilon),
        -(11 * CA - 2 * Nf) / (6 * epsilon),
        zero,
        zero,
    ]
    for _row, _reference in enumerate(_references):
        assert (deltas[_row, 0].to_expression() - _reference).together() == zero
    _residual = ct_matrix * deltas - _rhs
    assert all(
        _residual[_row, 0].to_expression().together() == zero for _row in range(8)
    )
    assert deltas[1, 0].to_expression().derivative(xi).expand() == zero
    assert deltas[5, 0].to_expression().derivative(xi).expand() == zero
    print(
        "Generated QCD one-loop: symbolic gauge, four IBP targets, eight generated counterterms passed",
        solution.stats,
    )

    # The completed MS/MSbar reference only needs the local UV poles. Use the
    # OneLOop measure conversion (4*pi)^eps*rGamma=1+cDelta*eps+O(eps²), where
    # cDelta=log(4*pi)-EulerGamma, with D=4-2eps. Keep it symbolic in exact checks.
    return (deltas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify MS and MSbar cancellation
    """)
    return


@app.cell
def _(D, S, ct_coefficients, deltas, epsilon, one, unknowns, uv_poles, zero):
    cDelta = S("qcd_ct::cDelta")
    assert (-2 / (D - 4)).replace(D, 4 - 2 * epsilon) == 1 / epsilon
    counterterms_by_scheme = {
        "MS": [deltas[_row, 0].to_expression().together() for _row in range(8)],
        "MSbar": [
            (
                epsilon * deltas[_row, 0].to_expression() * (1 / epsilon + cDelta)
            ).together()
            for _row in range(8)
        ],
    }
    for _scheme, _constants in counterterms_by_scheme.items():
        for _label, _generated_ct in ct_coefficients.items():
            for _delta, _constant in zip(unknowns, _constants, strict=True):
                _generated_ct = _generated_ct.replace(_delta, _constant)
            _subtraction_part = uv_poles[_label] * (
                one + epsilon * cDelta if _scheme == "MSbar" else one
            )
            assert (_generated_ct + _subtraction_part).together() == zero
            # In one common conventional measure MS retains the finite cDelta*P.
            _remainder = (
                uv_poles[_label] * (one + epsilon * cDelta) + _generated_ct
            ).together()
            _expected_remainder = (
                epsilon * cDelta * uv_poles[_label] if _scheme == "MS" else zero
            )
            assert (_remainder - _expected_remainder).together() == zero
    print("Generated QCD CT amplitudes: complete MS/MSbar tensor cancellation passed")
    return cDelta, counterterms_by_scheme


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Collect the derived residues
    """)
    return


@app.cell
def _(deltas, epsilon, uv_poles):
    counterterm_names = ["Zq", "Zm", "ZA", "Zξ", "Zc", "Zg", "ZAm", "Zcm"]
    residues = [
        (epsilon * deltas[_row, 0].to_expression()).together() for _row in range(8)
    ]
    vertex_residue = (epsilon * uv_poles["vertex"]).together()
    return counterterm_names, residues, vertex_residue


@app.cell(hide_code=True)
def _(ct_coefficients, ct_diagrams, ct_matrix, deltas, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Generated local counterterms

    Write $Z=1+a_4\delta Z$, with $q_0=\sqrt{Z_q}q$, $m_0=Z_m m$,
    $A_0=\sqrt{Z_A}A$, $\xi_0=Z_\xi\xi$, $c_0=\sqrt{Z_c}c$ and
    $g_{s,0}=Z_g g_s$. The local kinetic, mass and gauge-fixing operators are model
    inputs. Symbolica expands their bare factors to obtain the CT couplings.
    The quark–gluon rule copies the original model's ordered particle, color and
    Lorentz data and multiplies its actual coupling by $Z_qZ_g\sqrt{Z_A}-1$.

    These are tree topologies with **CT order one**, separate from loop count.
    An explicit vertex bound prevents arbitrarily long two-point insertion chains.
    The generator produces two quark, three gluon, two ghost and one vertex CT;
    setting CT order zero excludes them.
    """),
            mo.ui.tabs(
                {
                    _kind: mo.hstack(_diagrams)
                    for _kind, _diagrams in ct_diagrams.items()
                }
            ),
            mo.md(r"**Coefficients extracted from generated numerators**"),
            *[
                mo.hstack([mo.md("**" + _kind + "**"), _coefficient])
                for _kind, _coefficient in ct_coefficients.items()
            ],
            mo.md(r"""
    Shared particle color sums close the two-point color indices, and the vertex
    uses the conjugate tree color tensor. Native graph weights are retained.
    Only the named external-fermion convention is converted for the amputated
    quark two-point kernel; ghosts retain their own native sign.

    The columns below are $(\delta Z_q,\delta Z_m,\delta Z_A,\delta Z_\xi,
    \delta Z_c,\delta Z_g,\delta Z_{Am},\delta Z_{cm})$.
    Every matrix entry is a derivative of a generated coefficient, with linearity
    checked before solving. The complete tensor amplitudes cancel separately.

    The [reference model](https://raw.githubusercontent.com/FeynCalc/feyncalc/master/FeynCalc/Examples/FeynRules/QCD/QCD.fr)
    uses $M(Z_{Am}^2-1)A^2/2$ and $M(Z_{cm}^2-1)\bar c c/2$.
    Identical gluons supply a factor two; the distinct ghost fields do not.
    Their rows therefore contain $2\delta Z_{Am}$ and $\delta Z_{cm}$.
    """),
            ct_matrix,
            deltas,
            mo.md(r"""
    The result obeys $Z_\xi=Z_A$. Mass and coupling renormalization are gauge
    independent. The coupling pole gives $\beta_0=(11C_A-2N_f)/3$ in
    $\mu\,d g_s/d\mu=-\beta_0g_s^3/(16\pi^2)+O(g_s^5)$.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(counterterm_names, counterterms_by_scheme, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## MS and MSbar at the same scale

    Use $\Delta=1/\epsilon+c_\Delta$ with
    $c_\Delta=\log(4\pi)-\gamma_E$ and $D=4-2\epsilon$.
    [OneLOop](https://arxiv.org/abs/1007.4716) divides its master integrals by
    $r_\Gamma=\Gamma(1-\epsilon)^2\Gamma(1+\epsilon)/\Gamma(1-2\epsilon)$.
    The conventional loop measure restores $(4\pi)^\epsilon r_\Gamma=
    1+c_\Delta\epsilon+O(\epsilon^2)$, with $i/(16\pi^2)$ stripped off.

    If a normalized loop coefficient is $P/\epsilon+F$, this measure gives
    $P/\epsilon+F+c_\Delta P$. MS subtracts its pole; MSbar also subtracts
    $c_\Delta P$. The actual generated CT amplitudes cancel the corresponding
    terms in every sector. The finite scheme shift left in MS is checked too.

    The reference extracts UV terms after tensor reduction. Shared UV expansion
    provides those same local coefficients here; no full finite vertex is needed
    for this comparison. The direct selective-IRR calculation below derives the original reference's
    nonzero auxiliary gluon mass separately. Its physical pole constants agree.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the direct IR-rearrangement kinematics
    """)
    return


@app.cell
def _(D, K, Kinematics, P, S, Symbols, s, sp):
    # Direct n=0 infrared rearrangement: massify only massless propagators, then
    # Taylor-expand external momenta to the superficial divergence degree. This
    # follows the original massive reference independently of graph.uv_expansion.
    irr_edge_momentum, irr_dot, irr_coordinate, irr_scaling, irr_a, irr_b = (
        Symbols.edge_momentum,
        sp.TensorPattern.dot,
        S("qcd_irr::x"),
        S("qcd_irr::t"),
        S("qcd_irr::a_"),
        S("qcd_irr::b_"),
    )
    irr_kinematics = Kinematics(D, momenta=[K(0), P(0), P(1)]).with_scalar_product(
        P(0), P(0), s
    )
    beyond_uv = S("qcd_irr::beyond_uv")
    return (
        beyond_uv,
        irr_a,
        irr_b,
        irr_coordinate,
        irr_dot,
        irr_edge_momentum,
        irr_kinematics,
        irr_scaling,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Rearrange the massive and massless theories

    Set the quark mass to zero before massification in the massless theory. The folded per-diagram routine keeps primitive longitudinal propagators distinct and checks their reconstruction.
    """)
    return


@app.cell
def _(irr_inputs, rearrange_qcd_diagram, zero):
    irr_scalars_by_scenario = {}
    for _scenario in ("massive", "massless"):
        _massless = _scenario == "massless"
        _scalars = {}
        for (_kind, _number), _record in irr_inputs.items():
            for _label, _value in rearrange_qcd_diagram(
                _kind, _number, _record, _massless
            ).items():
                _scalars[_label] = (_scalars.get(_label, zero) + _value).together()
        irr_scalars_by_scenario[_scenario] = _scalars
    return (irr_scalars_by_scenario,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Decompose and reduce the rearranged integrals
    """)
    return


@app.cell
def _(IBPFamily, family, irr_scalars_by_scenario, rearranged_tadpole_terms):
    irr_reductions = {}
    for _scenario, _irr_scalars in irr_scalars_by_scenario.items():
        _massless = _scenario == "massless"
        _irr_terms, _irr_targets = {}, set()
        for _label, _scalar in _irr_scalars.items():
            _terms, _targets = rearranged_tadpole_terms(_scalar, _massless)
            _irr_terms[_label] = _terms
            _irr_targets.update(_targets)
        _solution = IBPFamily(
            family, name="qcd_direct_irr_" + _scenario
        ).reduce_laporta([list(p) for p in sorted(_irr_targets)], max_depth=2)
        assert _solution.residuals == [[1]]
        irr_reductions[_scenario] = (_irr_terms, _irr_targets, _solution)
    return (irr_reductions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Evaluate the ultraviolet poles
    """)
    return


@app.cell
def _(
    CA,
    CF,
    D,
    M,
    Nc,
    beyond_uv,
    epsilon,
    index,
    integral,
    irr_reductions,
    mass,
    sp,
    zero,
):
    irr_poles_by_scenario = {}
    for _scenario, (_irr_terms, _irr_targets, _solution) in irr_reductions.items():
        _irr_poles = {}
        for _label, _terms in _irr_terms.items():
            _expression = sum(
                (
                    _coefficient
                    * _solution.reduce([_power], integral=integral)
                    .replace(M, _squared_mass)
                    .replace(integral(1), _squared_mass / epsilon)
                    for _squared_mass, _power, _coefficient in _terms
                ),
                zero,
            ).together()
            _pole = (
                _expression.replace(D, 4 - 2 * epsilon)
                .series(epsilon, 0, -1)
                .to_expression()
                .expand()
                .replace(
                    sp.PortPattern.exact(sp.Representation.mink(4), index),
                    sp.PortPattern.exact(sp.Representation.mink(D), index),
                )
            )
            _irr_poles[_label] = _pole.together().replace(1 / Nc, CA - 2 * CF).expand()
            assert _pole.coefficient(epsilon**-2) == zero
            assert _pole.derivative(mass).together() == zero
            assert _pole.derivative(beyond_uv).together() == zero
        _irr_poles["gluon"] = sum(
            (
                _irr_poles[k]
                for k in ("gluon_loop", "ghost_loop", "quark_loop", "tadpole")
            ),
            zero,
        ).expand()
        _irr_poles["vertex"] = (
            _irr_poles["vertex_0"] + _irr_poles["vertex_1"]
        ).expand()

        irr_poles_by_scenario[_scenario] = _irr_poles
    return (irr_poles_by_scenario,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check physical and auxiliary-mass poles
    """)
    return


@app.cell
def _(CA, M, Nf, epsilon, gmunu, irr_poles_by_scenario, uv_poles, xi, zero):
    for _scenario, _irr_poles in irr_poles_by_scenario.items():
        _massless = _scenario == "massless"
        # Physical UV constants agree, but the auxiliary mass does depend on the
        # rearrangement prescription. The reference's quark loop retains its true mass.
        for _label in (
            ["quark_p", "ghost", "vertex"]
            if _massless
            else ["quark_p", "quark_m", "ghost", "vertex"]
        ):
            assert (_irr_poles[_label] - uv_poles[_label]).together() == zero
        _irr_auxiliary_pole = (
            (CA * (1 + 3 * xi) + (8 * Nf if _massless else zero))
            * M
            * gmunu
            / (4 * epsilon)
        )
        assert (
            _irr_poles["gluon"] - uv_poles["gluon"] - _irr_auxiliary_pole
        ).together() == zero
        if _massless:
            assert (
                _irr_poles["quark_loop"]
                - uv_poles["quark_loop"]
                - 2 * Nf * M * gmunu / epsilon
            ).together() == zero
        else:
            assert _irr_poles["quark_loop"].derivative(M) == zero
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Solve the active counterterm equations
    """)
    return


@app.cell
def _(
    CA,
    M,
    Matrix,
    Nf,
    ct_matrix,
    deltas,
    epsilon,
    gmunu,
    irr_poles_by_scenario,
    ppmunu,
    s,
    xi,
    zero,
):
    irr_linear_results = {}
    for _scenario, _irr_poles in irr_poles_by_scenario.items():
        _massless = _scenario == "massless"
        _active_indices = [i for i in range(8) if not (_massless and i == 1)]
        _irr_gluon_g = _irr_poles["gluon"].coefficient(gmunu)
        _irr_gluon_pp = _irr_poles["gluon"].coefficient(ppmunu)
        _irr_rhs = Matrix.vec(
            [
                -_irr_poles["quark_p"],
                -_irr_poles.get("quark_m", zero),
                -_irr_gluon_g.coefficient(s),
                -xi * _irr_gluon_pp,
                -_irr_poles["ghost"].coefficient(s),
                -_irr_poles["vertex"],
                -_irr_gluon_g.replace(s, zero) / M,
                -_irr_poles["ghost"].replace(s, zero) / M,
            ]
        )
        _active_matrix = Matrix.from_linear(
            len(_active_indices),
            len(_active_indices),
            [
                ct_matrix[i, j].to_expression()
                for i in _active_indices
                for j in _active_indices
            ],
        )
        _active_rhs = Matrix.vec(
            [_irr_rhs[i, 0].to_expression() for i in _active_indices]
        )
        _deltas = _active_matrix.solve(_active_rhs)
        for _row, _original_row in enumerate(_active_indices):
            _reference = (
                -(CA * (1 + 3 * xi) + (8 * Nf if _massless else zero)) / (8 * epsilon)
                if _original_row == 6
                else deltas[_original_row, 0].to_expression()
            )
            assert (_deltas[_row, 0].to_expression() - _reference).together() == zero
        irr_linear_results[_scenario] = {
            "poles": _irr_poles,
            "indices": _active_indices,
            "constants": _deltas,
            "matrix": _active_matrix,
        }
    return (irr_linear_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify MS and MSbar pole cancellation
    """)
    return


@app.cell
def _(
    Replacement,
    cDelta,
    ct_coefficients,
    epsilon,
    irr_linear_results,
    irr_reductions,
    one,
    unknowns,
    zero,
):
    irr_results = {}
    for _scenario, _linear in irr_linear_results.items():
        _massless = _scenario == "massless"
        _irr_poles, _active_indices = _linear["poles"], _linear["indices"]
        _deltas, _active_matrix = _linear["constants"], _linear["matrix"]
        _active_unknowns = [unknowns[i] for i in _active_indices]
        _irr_terms, _irr_targets, _solution = irr_reductions[_scenario]
        _irr_schemes = {}
        for _scheme, _measure in (("MS", one), ("MSbar", one + epsilon * cDelta)):
            _constants = [
                (_deltas[_row, 0].to_expression() * _measure).together()
                for _row in range(len(_active_indices))
            ]
            _irr_schemes[_scheme] = _constants
            _irr_ct_rules = [
                Replacement(delta, constant)
                for delta, constant in zip(_active_unknowns, _constants, strict=True)
            ]
            for _label, _generated_ct in ct_coefficients.items():
                if _massless and _label == "quark_m":
                    continue
                if _massless:
                    assert _generated_ct.derivative(unknowns[1]) == zero
                _actual_ct = _generated_ct.replace_multiple(_irr_ct_rules)
                assert (_measure * _irr_poles[_label] + _actual_ct).together() == zero
                _common_measure_remainder = (
                    (one + epsilon * cDelta) * _irr_poles[_label] + _actual_ct
                ).together()
                _expected_remainder = (
                    epsilon * cDelta * _irr_poles[_label] if _scheme == "MS" else zero
                )
                assert (
                    _common_measure_remainder - _expected_remainder
                ).together() == zero
        irr_results[_scenario] = {
            "poles": _irr_poles,
            "schemes": _irr_schemes,
            "constants": _deltas,
            "indices": _active_indices,
            "matrix": _active_matrix,
            "solution": _solution,
            "residues": [
                (epsilon * _deltas[_row, 0].to_expression()).together()
                for _row in range(len(_active_indices))
            ],
        }
        print(
            f"Direct {_scenario} QCD IRR: {len(_irr_targets)} IBP targets and {len(_active_indices)} generated CT equations passed",
            _solution.stats,
        )
    return (irr_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare with the massive reference
    """)
    return


@app.cell
def _(epsilon, irr_results):
    # Keep the massive result available for the notebook's original comparison.
    irr_deltas = irr_results["massive"]["constants"]
    irr_solution = irr_results["massive"]["solution"]

    irr_residues = [
        (epsilon * irr_deltas[_row, 0].to_expression()).together() for _row in range(8)
    ]
    return irr_deltas, irr_residues, irr_solution


@app.cell
def _(irr_deltas, irr_solution, mo):
    mo.vstack(
        [
            mo.md(r"""
    ## Direct infrared rearrangement with a massive quark

    The original reference replaces each **massless** propagator $q^2$ by
    $q^2-M$, then Taylor-expands the external momenta to the superficial degree
    of divergence. The massive quark denominators $q^2-m_q^2$ remain intact.
    This is calculated independently of the full UV expansion above.

    Each internal gluon's longitudinal numerator contains an additional $1/q^2$.
    The calculation isolates these factors before massification, including both
    internal gluons of the non-Abelian vertex and gluon loop. Recombining the
    pieces reproduces the original generated numerator exactly. Their powers
    are passed to the graph's existing denominator API.

    After tensor reduction, Symbolica partial fractions separate tadpoles at
    $M$ and $m_q^2$. Reconstructing the rational integrand checks the decomposition
    before IBP. Five powers reduce to one tadpole per mass, with the shared
    OneLOop pole $A_0(m^2)=m^2/\epsilon$.
    """),
            irr_solution,
            mo.md(r"""
    The physical poles agree with the full UV expansion. The extra local gluon
    pole is $C_A(1+3\xi)M g^{\mu\nu}/(4\epsilon)$. Solving the same matrix
    from the eight generated CT diagrams therefore gives
    $$\delta Z_{Am}=-\frac{C_A(1+3\xi)}{8\epsilon},\qquad \delta Z_{cm}=0.$$
    The complete projected tensor amplitudes cancel against those CT insertions.
    The quark loop contributes no auxiliary-mass term because its physical mass
    was retained. The [massless QCD notebook](?file=hep/qcd_massless_renormalization.py)
    sets the quark mass to zero before massification and displays the distinct
    result. Both calculations share this notebook's symbolic workflow.
    """),
            irr_deltas,
        ]
    )
    return


@app.cell
def _(mo):
    gauge_parameter = mo.ui.slider(
        0.0, 3.0, step=0.25, value=1.0, label="Gauge parameter ξ"
    )
    flavor_count = mo.ui.slider(0, 20, step=1, value=5, label="Quark flavors Nf")
    color_count = mo.ui.slider(2, 5, step=1, value=3, label="Colors Nc")
    subtraction_scheme = mo.ui.dropdown(
        ["MSbar", "MS"], value="MSbar", label="Subtraction scheme"
    )
    rearrangement = mo.ui.dropdown(
        ["Full UV expansion", "Direct massive IRR"],
        value="Full UV expansion",
        label="Rearrangement",
    )
    mo.vstack(
        [
            mo.hstack([gauge_parameter, flavor_count, color_count]),
            mo.hstack([subtraction_scheme, rearrangement]),
        ]
    )
    return (
        color_count,
        flavor_count,
        gauge_parameter,
        rearrangement,
        subtraction_scheme,
    )


@app.cell
def _(
    CA,
    CF,
    Nf,
    Symbol,
    color_count,
    flavor_count,
    gauge_parameter,
    irr_residues,
    rearrangement,
    residues,
    subtraction_scheme,
    vertex_residue,
    xi,
):
    _n = color_count.value
    _parameters = {
        CA: _n,
        CF: (_n**2 - 1) / (2 * _n),
        Nf: flavor_count.value,
        xi: gauge_parameter.value,
    }
    _c_delta = complex(((4 * Symbol.PI).log() - Symbol.EULER_GAMMA).evaluate({})).real
    finite_factor = _c_delta if subtraction_scheme.value == "MSbar" else 0.0
    _active_residues = (
        irr_residues if rearrangement.value == "Direct massive IRR" else residues
    )
    selected_residues = [
        complex(value.evaluate(_parameters)).real for value in _active_residues
    ]
    _vertex = complex(vertex_residue.evaluate(_parameters)).real
    vertex_error = abs(
        selected_residues[0] + selected_residues[5] + selected_residues[2] / 2 + _vertex
    )
    beta0 = -2 * selected_residues[5]
    assert vertex_error < 1e-12
    assert selected_residues[2] == selected_residues[3]
    assert abs(beta0 - (11 * _n - 2 * flavor_count.value) / 3) < 1e-12
    return beta0, finite_factor, selected_residues, vertex_error


@app.cell(hide_code=True)
def _(
    beta0,
    counterterm_names,
    finite_factor,
    mo,
    selected_residues,
    vertex_error,
):
    mo.vstack(
        [
            mo.md(
                r"**Counterterms** — vary gauge, flavor count, SU(Nc) color group and subtraction scheme."
            ),
            mo.ui.table(
                [
                    {
                        "Constant": _name,
                        "1/ε coefficient": value,
                        "Finite subtraction coefficient": finite_factor * value,
                    }
                    for _name, value in zip(
                        counterterm_names, selected_residues, strict=True
                    )
                ],
                selection=None,
            ),
            mo.md(
                f"Quark–gluon vertex cancellation residual: **{vertex_error:.1e}**.  "
                + rf"One-loop $\beta_0$: **{beta0:.6g}**."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
