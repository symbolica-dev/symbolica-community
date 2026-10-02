import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Gluon ghost subtraction")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gluon ghost subtraction

    [Browse all notebooks](/) · [Physical polarization calculation](/?file=hep/qcd_gluons.py)

    Generate $b\bar b\to gg$ with the bottom mass retained and use covariant
    gluon polarization sums. These include unphysical states. Generate both
    ghost–antighost orderings separately and subtract their squared amplitudes
    to recover the physical result.

    All three gluon diagrams and their interference are included. Shared
    `Particle.color_sum` and `Particle.spin_sum` provide the initial averages
    and completeness tensors; Idenso and Spenso perform the algebra.
    Ghosts carry color but have no external polarization wavefunctions.

    The checks compare each ghost contribution and the final massive $SU(N_c)$
    result with [FeynCalc](https://feyncalc.github.io/FeynCalcExamples/QCD/Tree/QQbar-GlGl-2),
    then verify the massless $SU(3)$ limit and exchange of the two final gluons.
    External spins are four-dimensional and $T_R=1/2$.
    The symbolic calculation takes a few minutes.
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
    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("qcd_ghosts")
    return E, Kinematics, Model, Replacement, S, TensorExpression, hep, mo, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    E,
    Nc,
    P,
    S,
    Symbols,
    TensorExpression,
    dA,
    ghost_model,
    gs,
    kinematics,
    mass,
    s,
    sp,
    t,
    u,
):
    def evaluate_ghost_channel(_outgoing, _count, _generated):
        """Reduce one channel, retaining covariant gluon polarization sums."""
        _ports = S("external_0", "external_1", "external_2", "external_3")
        # Give the conjugate amplitude distinct external labels; scope only its dummies.
        _bra_ports = {port: S(f"bra_{i}") for i, port in enumerate(_ports)}
        _a, _b, _c, _inverse, _index = S("a_", "b_", "c_", "inverse_", "index_")
        _conjugate, _adjoint_index = (
            sp.BroadcastFunction.conj().to_expression(),
            S("adjoint_index"),
        )
        _amplitude = E("0")
        _denominators = []
        for _diagram in _generated.diagrams:
            _numerator = ghost_model.expand_couplings(
                _diagram.numerator_expression(in_lmb=True).to_expression()
            )
            # Align external spin and color ports using the graph's native half-edge IDs.
            # Ghosts have no polarization wavefunctions; these IDs cover them as well.
            for _half in _diagram.half_edges:
                _edge = _half.edge.data
                if _edge.is_external:
                    _numerator = sp.TensorExpression(_numerator).rename_indices(
                        {Symbols.half_edge(_half.data, 1): _ports[_edge.external_index]}
                    )
            _denominator = kinematics.apply(
                _diagram.denominator_expression(dimension=4, in_lmb=True)
                .to_expression()
                .replace(Symbols.denominator(_a, _b, _c, _inverse), _inverse)
            ).expand()
            _denominators.append(_denominator)
            _amplitude += (
                _numerator
                * _diagram.overall_factor_expression(evaluate=True)
                * _diagram.numerator_prefactor_expression()
                / _denominator
            )
        assert all(
            _d in _denominators
            for _d in ((s, t - mass**2, u - mass**2) if _count == 3 else (s,))
        )
        _operator = _amplitude.expand()
        assert len(_operator.structure.slots) == (8 if _count == 3 else 6)
        _adjoint = (
            _operator.dirac_adjoint()
            .simplify_algebra(
                contract="dots",
                color=False,
                gamma=True,
                gamma0=True,
                gamma_evaluate_traces=False,
            )
            .expand()
            .to_expression()
        )
        _adjoint = _adjoint.replace(_conjugate(P(_a, _b)), P(_a, _b))
        for _real in (mass, gs, s, t, u):
            _adjoint = _adjoint.replace(_conjugate(_real), _real)
        _adjoint = (
            sp.TensorExpression(_adjoint)
            .wrap_indices(_adjoint_index, dummies_only=True)
            .rename_indices(_bra_ports)
            .to_expression()
        )

        # Match the shared completeness tensor against the dual of each external slot.
        # Matching selects color slots and determines the quark/antiquark orientation.
        _left, _right, _metric = (
            S("left_"),
            S("right_"),
            sp.TensorName.g().to_expression(),
        )
        _color_projector = E("1")
        _initial_colors = 1
        for _position, _name in enumerate(("b", "b~", *_outgoing)):
            _particle = ghost_model.particle(_name)
            if _position < 2:
                _initial_colors *= abs(_particle.color)
            _matches = []
            for _slot in _operator.structure.slots:
                _original = _slot.to_expression()
                if _original.replace(_ports[_position], E("0")) == _original:
                    continue
                _closure = _metric(
                    _slot.dual().to_expression(),
                    _original.replace(
                        _ports[_position],
                        _bra_ports[_ports[_position]],
                    ),
                )
                _match = next(
                    _closure.match(_particle.color_sum(_left, _right), max_level=0),
                    None,
                )
                if _match is not None:
                    _matches.append(_match)
            assert len(_matches) == 1
            _indices = dict(_matches[0])
            _color_projector *= _particle.color_sum(
                _indices[_left], _indices[_right], average=_position < 2
            )

        # Spenso dimensions are integers or symbols. Keep dA symbolic until color
        # contraction, then impose the SU(N) relation and convert scalar Casimirs.
        _generic = (
            (
                _operator.to_expression()
                * _adjoint
                * _color_projector
                * _initial_colors
                / Nc**2
            )
            .replace(
                sp.PortPattern.exact(sp.Representation.cof(3), _index),
                sp.PortPattern.exact(sp.Representation.cof(Nc), _index),
            )
            .replace(
                sp.PortPattern.exact(sp.Representation.coad(8), _index),
                sp.PortPattern.exact(sp.Representation.coad(dA), _index),
            )
        )
        _colored = (
            TensorExpression(_generic)
            .simplify_algebra(contract="dots", gamma=False, color=True)
            .to_expression()
            .replace(dA, Nc**2 - 1)
        )
        _colored = TensorExpression(_colored).simplify_algebra(
            contract="dots",
            gamma=False,
            color=True,
            color_substitute_cof_dimension_invariants=True,
        )

        # Dirac adjunction exchanges the two endpoints of the open fermion chain.
        # Reduce the fermion trace before introducing the physical gluon projectors.
        _spin_projector = ghost_model.particle("b").spin_sum(
            P(0),
            _ports[0],
            _bra_ports[_ports[1]],
            average=True,
        ) * ghost_model.particle("b~").spin_sum(
            P(1),
            _bra_ports[_ports[0]],
            _ports[1],
            average=True,
        )
        _spin_summed = (
            (_colored * _spin_projector)
            .simplify_algebra(contract="dots", color=False, gamma=True, epsilon=True)
            .expand()
            .to_expression()
        )
        _polarizations = E("1")
        if _count == 3:
            for _position in (2, 3):
                _polarizations *= ghost_model.particle("g").spin_sum(
                    P(_position),
                    _ports[_position],
                    _bra_ports[_ports[_position]],
                    covariant=True,
                )
        _scalar = (
            TensorExpression(_spin_summed * _polarizations)
            .contract()
            .to_dots()
            .expand()
            .to_expression()
        )
        assert TensorExpression(_scalar).is_scalar
        _squared = kinematics.apply(_scalar).replace(s, 2 * mass**2 - t - u).together()

        print("Finished channel", _outgoing, flush=True)

        return _squared

    return (evaluate_ghost_channel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model):
    ghost_model = Model.standard_model()
    return (ghost_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the annihilation kinematics
    """)
    return


@app.cell
def _(E, Kinematics, S, ghost_model, hep, sp):
    P = hep.Kinematics.external_momentum
    s, t, u = S("s", "t", "u")
    mass = ghost_model.particle("b").mass
    gs = ghost_model.parameter("G").symbol
    Nc, dA, cof, coad = (
        sp.Nc,
        S("dA"),
        sp.Representation.cof,
        sp.Representation.coad,
    )
    kinematics = Kinematics.mandelstam(
        [P(0), P(1), P(2), P(3)], [mass**2, mass**2, E("0"), E("0")], [s, t, u]
    )
    return Nc, P, dA, gs, kinematics, mass, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the gluon and both ghost channels
    """)
    return


@app.cell
def _(evaluate_ghost_channel, ghost_model):
    ghost_results = []
    ghost_generated = []
    for _outgoing, _vertices, _count in (
        (["ghG", "ghG~"], ["V_76", "V_35"], 1),
        (["ghG~", "ghG"], ["V_76", "V_35"], 1),
        (["g", "g"], ["V_76", "V_36"], 3),
    ):
        _generated = ghost_model.process(
            ["b", "b~"], _outgoing, vertex_allow=_vertices
        ).generate_diagrams(
            loops=0,
            max_vertices=2,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_generated.diagrams) == _count
        ghost_generated.append(_generated)
        ghost_results.append(evaluate_ghost_channel(_outgoing, _count, _generated))
    return ghost_generated, ghost_results


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Subtract ghosts and check the massive result
    """)
    return


@app.cell
def _(E, Nc, ghost_results, gs, mass, s, t, u):
    _expected = (
        (Nc**2 - 1)
        * gs**4
        * (
            mass**4 * (3 * t**2 + 14 * t * u + 3 * u**2)
            - mass**2 * (t**3 + 7 * t**2 * u + 7 * t * u**2 + u**3)
            - 6 * mass**8
            + t * u * (t**2 + u**2)
        )
        * (
            -2 * Nc**2 * mass**2 * (t + u)
            + 2 * Nc**2 * mass**4
            + Nc**2 * (t**2 + u**2)
            - s**2
        )
        / (2 * Nc**3 * s**2 * (u - mass**2) ** 2 * (t - mass**2) ** 2)
    )
    _ghost_reference = (
        (Nc**2 - 1) * gs**4 * (u - mass**2) * (t - mass**2) / (4 * Nc * s**2)
    )
    for _result in ghost_results[:2]:
        assert (
            _result - _ghost_reference.replace(s, 2 * mass**2 - t - u)
        ).together() == E("0")
    assert ghost_results[0] == ghost_results[1]
    ghost_physical = (ghost_results[2] - ghost_results[0] - ghost_results[1]).together()
    assert (ghost_physical - _expected.replace(s, 2 * mass**2 - t - u)).together() == E(
        "0"
    )
    print(
        "Both ghost contributions and massive SU(N) ghost-subtracted gluons passed",
        flush=True,
    )

    # The same massive physical reference was independently checked using two axial
    # polarization choices in installed_feyncalc_qcd_gluons.py.
    assert (ghost_results[2] - ghost_physical).together() != E("0")
    return (ghost_physical,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the massless SU(3) limit
    """)
    return


@app.cell
def _(E, Nc, Replacement, ghost_physical, gs, mass, s, t, u):
    ghost_massless = ghost_physical.replace(mass, E("0")).replace(Nc, E("3"))
    _expected_massless = (
        E("32/27") * gs**4 * (t**2 + u**2) / (t * u)
        - E("8/3") * gs**4 * (t**2 + u**2) / s**2
    )
    assert (ghost_massless - _expected_massless.replace(s, -t - u)).together() == E("0")
    # A labeled squared amplitude has no identical-particle phase-space factor.
    assert (
        ghost_physical
        - ghost_physical.replace_multiple([Replacement(t, u), Replacement(u, t)])
    ).together() == E("0")
    print(
        "Massless SU(3), Bose symmetry and nonzero ghost correction passed", flush=True
    )
    return (ghost_massless,)


@app.cell(hide_code=True)
def _(ghost_generated, ghost_massless, ghost_physical, ghost_results, mo):
    mo.vstack(
        [
            mo.md("**Covariant gluon amplitudes**"),
            mo.hstack(list(ghost_generated[2].diagrams)),
            mo.md("**Both generated ghost orderings**"),
            mo.hstack([ghost_generated[0].diagrams[0], ghost_generated[1].diagrams[0]]),
            mo.md(
                "**Each ghost contribution (including initial spin and color averages)**"
            ),
            ghost_results[0].factor(),
            mo.md("**Covariant result before subtraction**"),
            ghost_results[2].factor(),
            mo.md(
                "**Physical result: covariant − ghost ordering 1 − ghost ordering 2**"
            ),
            ghost_physical.factor(),
            mo.md("**Massless SU(3) limit**"),
            ghost_massless.factor(),
            mo.md(
                "All exact checks passed. These are labeled squared amplitudes; integrating over both identical final-gluon labels requires the usual factor of 1/2!."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
