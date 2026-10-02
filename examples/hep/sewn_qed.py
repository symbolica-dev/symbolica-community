import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Sewing-aware QED spin sums")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sewing-aware QED spin sums

    [Browse notebooks](/) · [Amplitude-based Compton calculation](/?file=hep/compton.py) ·
    [Sewn diphoton annihilation](/?file=hep/sewn_diphoton.py)

    Generate squared tree amplitudes as sewn forward graphs, retaining the fermion
    masses. Compare electron and positron Compton scattering, electron–positron
    annihilation into muons, and electron–muon scattering with the
    [FeynCalc tree examples](https://feyncalc.github.io/examples).

    The graph and generator preserve particle flow at the sewing boundary.
    `Particle.sum_spins` supplies the same physical completeness tensors used by
    ordinary amplitudes. Native graph factors include the relative permutation
    sign when open fermion chains close into cycles. The cut's physical particles
    and orientations determine the positive-energy final-state momenta.

    For Compton scattering, covariant and timelike axial photon projectors are
    checked separately against the complete massive result. The two choices
    must agree. These are squared matrix elements; phase-space integration is
    separate.
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
    import marimo as mo
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("sewn_qed")
    return E, Kinematics, Model, S, TensorExpression, hep, mo


@app.cell(hide_code=True)
def _():
    from symbolica.community.hep import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    E,
    K,
    P,
    Q,
    Symbols,
    TensorExpression,
    a,
    b,
    c,
    e,
    index,
    inverse,
    k1,
    me,
    mm,
    model,
    s,
    t,
    u,
):
    def evaluate_sewn_channel(
        _name, _incoming, _outgoing, _compton, _masses, _kin, _result
    ):
        """Check each polarization choice against the exact massive channel reference."""
        results = {}
        if _compton:
            _expected = (
                2
                * e**4
                * (
                    -(me**4) * (3 * s**2 + 14 * s * u + 3 * u**2)
                    + me**2 * (s**3 + 7 * s**2 * u + 7 * s * u**2 + u**3)
                    + 6 * me**8
                    - s * u * (s**2 + u**2)
                )
                / ((s - me**2) ** 2 * (u - me**2) ** 2)
            )
        elif _name == "annihilation":
            _expected = (
                2
                * e**4
                / s**2
                * (
                    2 * me**2 * (2 * mm**2 + s - t - u)
                    + 2 * me**4
                    + 2 * mm**4
                    + 2 * mm**2 * (s - t - u)
                    + t**2
                    + u**2
                )
            )
        else:
            _expected = (
                2
                * e**4
                / t**2
                * (
                    -2 * me**2 * (-2 * mm**2 + s - t + u)
                    + 2 * me**4
                    + 2 * mm**4
                    - 2 * mm**2 * (s - t + u)
                    + s**2
                    + u**2
                )
            )
        for _reference in [None, P(0)] if _compton else [None]:
            _squared = E("0")
            for _diagram in _result.diagrams:
                assert len(_diagram.cuts) == 1
                _cut = _diagram.cuts[0]
                assert sorted(p.name for p in _cut.particles) == sorted(_outgoing)
                _coordinate = next(
                    _edge.id
                    for _edge, particle in zip(_cut.edges, _cut.particles)
                    if particle.name == _outgoing[0]
                )
                _diagram = _diagram.with_loop_momentum_edges([_coordinate])
                _cut = _diagram.cuts[0]
                assert all(side.loop_count == 0 for side in (_cut.left, _cut.right))
                _orientation = _cut.orientations[_coordinate]
                _projector = _diagram.projector_expression()
                for _edge in _diagram.external_edges:
                    _projector = model.particle(_edge.particle_name).sum_spins(
                        _projector,
                        Q(_edge.id),
                        edge=_edge.id,
                        average=True,
                        reference=_reference,
                        covariant=_reference is None,
                    )
                _numerator = model.expand_couplings(
                    _diagram.numerator_expression().to_expression() * _projector
                )
                _numerator = (
                    TensorExpression(_numerator.expand())
                    .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                    .expand()
                    .to_expression()
                )
                _numerator = _kin.apply(
                    _diagram.loop_momentum_basis.route_expression(_numerator).replace(
                        K(0, index), _orientation * k1(index)
                    )
                )
                _denominator = _diagram.denominator_expression(
                    edge_powers={_edge.id: 0 for _edge in _cut.edges},
                    dimension=4,
                    in_lmb=True,
                ).to_expression()
                _denominator = _kin.apply(
                    _denominator.replace(
                        Symbols.denominator(a, b, c, inverse), inverse
                    ).replace(K(0, index), _orientation * k1(index))
                )
                _squared += (
                    _diagram.overall_factor_expression(evaluate=True)
                    * _diagram.numerator_prefactor_expression()
                    * _numerator
                    / _denominator
                )
            _difference = (
                (_squared - _expected).replace(t, sum(_masses) - s - u).together()
            )
            assert _difference == 0, (_name, _reference, _difference)
            if _compton:
                _massless = _squared.replace(t, 2 * me**2 - s - u).replace(me, E("0"))
                assert (_massless + 2 * e**4 * (s / u + u / s)).together() == 0
            _label = _name + (" · covariant" if _reference is None else " · axial n=p₁")
            results[_label] = {
                "result": _squared.replace(t, sum(_masses) - s - u).together(),
                "reference": _expected.replace(t, sum(_masses) - s - u).together(),
                "residual": _difference,
                "diagrams": _result.diagrams,
                "incoming": _incoming,
                "outgoing": _outgoing,
            }
            print(
                f"Sewn {_name}, reference={_reference}: exact massive result passed",
                flush=True,
            )
        return results

    return (evaluate_sewn_channel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define momenta and masses
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model, S, Symbols, hep):
    model = Model.standard_model()
    Q, P, K = (
        Symbols.edge_momentum,
        hep.Kinematics.external_momentum,
        hep.Kinematics.loop_momentum,
    )
    s, t, u = S("s", "t", "u")
    me = model.particle("e-").mass
    mm = model.particle("mu-").mass
    e = -model.particle("e-").electric_charge
    k1, k2, index = S("k1", "k2", "index_")
    a, b, c, inverse = S("a_", "b_", "c_", "inverse_")
    return K, P, Q, a, b, c, e, index, inverse, k1, k2, me, mm, model, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select the QED vertices
    """)
    return


@app.cell
def _(model):
    _electron, _muon, _photon = (model.particle(_name) for _name in ("e-", "mu-", "a"))
    vertices = [
        v
        for v in model.vertex_rules
        if sorted(v.particles)
        in (
            sorted([_electron.antiname, _electron.name, _photon.name]),
            sorted([_muon.antiname, _muon.name, _photon.name]),
        )
    ]
    assert len(vertices) == 2
    return (vertices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate and verify the sewn channels

    The folded routine performs spin sums, routes each directed cut, and checks the exact massive reference. Compton scattering is also checked with a physical axial projector.
    """)
    return


@app.cell
def _(
    E,
    Kinematics,
    P,
    evaluate_sewn_channel,
    k1,
    k2,
    me,
    mm,
    model,
    s,
    t,
    u,
    vertices,
):
    sewn_results = {}
    for _name, _incoming, _outgoing in [
        ("electron Compton", ["e-", "a"], ["e-", "a"]),
        ("positron Compton", ["e+", "a"], ["e+", "a"]),
        ("annihilation", ["e-", "e+"], ["mu-", "mu+"]),
        ("electron-muon scattering", ["e-", "mu-"], ["e-", "mu-"]),
    ]:
        _compton = _name.endswith("Compton")
        _masses = (
            [me**2, E("0"), me**2, E("0")]
            if _compton
            else [me**2, me**2, mm**2, mm**2]
            if _name == "annihilation"
            else [me**2, mm**2, me**2, mm**2]
        )
        _kin = Kinematics.mandelstam([P(0), P(1), k1, k2], _masses, [s, t, u])
        _result = model.process(
            _incoming, _outgoing, vertex_allow=vertices
        ).generate_cross_section(
            loops=1,
            max_vertices=4,
            maximum_bridges=None,
            numerator_grouping=None,
            progress=None,
        )
        assert len(_result.diagrams) == (4 if _compton else 1)
        sewn_results.update(
            evaluate_sewn_channel(
                _name, _incoming, _outgoing, _compton, _masses, _kin, _result
            )
        )
    return (sewn_results,)


@app.cell(hide_code=True)
def _(mo, sewn_results):
    selected_case = mo.ui.dropdown(
        list(sewn_results),
        value=next(iter(sewn_results)),
        label="Process and incoming-photon projector",
    )
    mo.vstack([selected_case])
    return (selected_case,)


@app.cell(hide_code=True)
def _(mo, selected_case, sewn_results):
    selected = sewn_results[selected_case.value]
    assert selected["residual"] == 0
    mo.vstack(
        [
            mo.md("## Generated sewn diagrams"),
            mo.hstack(selected["diagrams"]),
            mo.md(
                r"Each cut separates two tree amplitudes. Incoming spins are averaged; final spins are summed. Typed particles and vertex rules select the QED interactions."
            ),
            mo.md("**Massive squared matrix element**"),
            selected["result"],
            mo.md("**Independent reference**"),
            selected["reference"],
            mo.md("**Exact difference after the Mandelstam relation**"),
            selected["residual"],
        ]
    )
    return


if __name__ == "__main__":
    app.run()
