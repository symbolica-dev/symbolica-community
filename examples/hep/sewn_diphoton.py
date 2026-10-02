import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Diphoton annihilation through sewing",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diphoton annihilation through sewing

    [Browse notebooks](/) · [Ordinary diphoton amplitudes](/?file=hep/diphoton.py) ·
    [Other sewn QED processes](/?file=hep/sewn_qed.py)

    Generate $e^-e^+ \to \gamma\gamma$ as forward graphs cut through the two
    final photons. Keep the electron mass, all interference terms and graph
    factors, then compare with the
    [FeynCalc result](https://feyncalc.github.io/FeynCalcExamples/QED/Tree/ElAel-GaGa).

    A stored cut-edge momentum has a direction. `cut.orientations` converts it
    into a positive-energy outgoing momentum before imposing the Mandelstam
    relations. Both choices of cut photon as the loop coordinate must give the
    same result. Particle lookup uses model names throughout.

    Incoming spins use the shared `Particle.sum_spins`. The outgoing photons
    use covariant sums or physical axial projectors from `Particle.spin_sum`.
    Null and timelike gauge references agree exactly; replacing either photon
    polarization by its momentum gives zero.
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

    _set_namespace("sewn_diphoton")
    return E, Kinematics, Model, Replacement, S, TensorExpression, hep, mo, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(
    E,
    K,
    P,
    Q,
    Replacement,
    Symbols,
    TensorExpression,
    a,
    b,
    c,
    charge,
    expected,
    index,
    inverse,
    k1,
    kin,
    mass,
    metric,
    model,
    photon,
    result,
    s,
    sp,
    t,
    u,
):
    def check_cut_coordinate(coordinate_index):
        """Contract all polarization sums for one directed cut coordinate."""
        results, routing_rows = {}, []
        for mode in ("covariant", "null", "timelike", "Ward 0", "Ward 1"):
            total = E("0")
            for original in result.diagrams:
                coordinate = original.cuts[0].edges[coordinate_index].id
                diagram = original.with_loop_momentum_edges([coordinate])
                cut = diagram.cuts[0]
                assert [particle.name for particle in cut.particles] == ["a", "a"]
                assert all(side.loop_count == 0 for side in (cut.left, cut.right))
                orientation = cut.orientations[coordinate]
                if mode == "covariant":
                    routing_rows.append(
                        {
                            "Coordinate": coordinate_index + 1,
                            "Diagram": diagram.name,
                            "Cut edge": coordinate,
                            "K(0) / k1": orientation,
                        }
                    )
                projector = diagram.projector_expression()
                for edge in diagram.external_edges:
                    projector = model.particle(edge.particle_name).sum_spins(
                        projector, Q(edge.id), edge=edge.id, average=True
                    )
                numerator = model.expand_couplings(
                    diagram.numerator_expression().to_expression() * projector
                )
                for position, edge in enumerate(cut.edges):
                    if mode == "covariant" or (
                        mode.startswith("Ward") and mode != f"Ward {position}"
                    ):
                        continue
                    # The photon propagator numerator is -i*g. Replacing g by the
                    # negative physical density retains its original propagator phase.
                    edge_metric = E("1𝑖") * edge.numerator_expression().to_expression()
                    match = dict(
                        next(
                            edge_metric.match(
                                metric(
                                    sp.PortPattern.exact(sp.Representation.mink(4), a),
                                    sp.PortPattern.exact(sp.Representation.mink(4), b),
                                ),
                                max_level=0,
                            )
                        )
                    )
                    if mode.startswith("Ward"):
                        physical = Q(
                            edge.id,
                            sp.PortPattern.exact(sp.Representation.mink(4), match[a]),
                        ) * Q(
                            edge.id,
                            sp.PortPattern.exact(sp.Representation.mink(4), match[b]),
                        )
                    else:
                        reference = (
                            P(0)
                            if mode == "timelike"
                            else Q(cut.edges[1 - position].id)
                        )
                        physical = photon.spin_sum(
                            Q(edge.id), match[a], match[b], reference=reference
                        )
                    numerator = numerator.replace(edge_metric, -physical)
                scalar = (
                    TensorExpression(numerator.expand())
                    .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                    .expand()
                    .to_expression()
                )
                assert TensorExpression(scalar).is_scalar
                # A cut coordinate is a stored directed edge momentum. Convert it
                # to the positive-energy outgoing photon before imposing kinematics.
                contracted = kin.apply(
                    diagram.loop_momentum_basis.route_expression(scalar).replace(
                        K(0, index), orientation * k1(index)
                    )
                )
                denominator = kin.apply(
                    diagram.denominator_expression(
                        edge_powers={edge.id: 0 for edge in cut.edges},
                        dimension=4,
                        in_lmb=True,
                    )
                    .to_expression()
                    .replace(Symbols.denominator(a, b, c, inverse), inverse)
                    .replace(K(0, index), orientation * k1(index))
                )
                total += (
                    diagram.overall_factor_expression(evaluate=True)
                    * diagram.numerator_prefactor_expression()
                    * contracted
                    / denominator
                ).replace(s, 2 * mass**2 - t - u)
            # The forward graphs identify the two identical cut photons. Restore
            # both labeled final-state assignments for the labeled squared amplitude.
            squared = (
                total + total.replace_multiple([Replacement(t, u), Replacement(u, t)])
            ).together()
            target = E("0") if mode.startswith("Ward") else expected
            assert (squared - target).together() == 0, (coordinate_index, mode)
            if not mode.startswith("Ward"):
                assert (
                    squared.replace(mass, 0) - 2 * charge**4 * (t / u + u / t)
                ).together() == 0
                assert squared.replace(mass, 1).replace(charge, 1).replace(
                    t, -1
                ).replace(u, -7).together() == E("83/8")
            results[coordinate_index, mode] = squared
            print("PASS", coordinate_index, mode, flush=True)
        return results, routing_rows

    return (check_cut_coordinate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Select the QED interaction
    """)
    return


@app.cell
def _(Model):
    model = Model.standard_model()
    electron, photon = (model.particle(name) for name in ("e-", "a"))
    vertices = [
        vertex
        for vertex in model.vertex_rules
        if sorted(vertex.particles)
        == sorted([electron.name, electron.antiname, photon.name])
    ]
    assert len(vertices) == 1
    return model, photon, vertices


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the sewn forward graphs
    """)
    return


@app.cell
def _(model, vertices):
    result = model.process(
        ["e-", "e+"], ["a", "a"], vertex_allow=vertices
    ).generate_cross_section(
        loops=1,
        max_vertices=4,
        maximum_bridges=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(result.diagrams) == 2
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare physical cut kinematics
    """)
    return


@app.cell
def _(E, Kinematics, S, Symbols, hep, model, sp):
    Q, P, K = (
        Symbols.edge_momentum,
        hep.Kinematics.external_momentum,
        hep.Kinematics.loop_momentum,
    )
    s, t, u = S("s", "t", "u")
    _electron = model.particle("e-")
    mass, charge = _electron.mass, -_electron.electric_charge
    k1, k2, index = S("k1", "k2", "index_")
    a, b, c, inverse = S("a_", "b_", "c_", "inverse_")
    metric, mink = (sp.TensorName.g().to_expression(), sp.Representation.mink)
    kin = Kinematics.mandelstam(
        [P(0), P(1), k1, k2], [mass**2, mass**2, E("0"), E("0")], [s, t, u]
    )
    return (
        K,
        P,
        Q,
        a,
        b,
        c,
        charge,
        index,
        inverse,
        k1,
        kin,
        mass,
        metric,
        s,
        t,
        u,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Specify the independent massive reference
    """)
    return


@app.cell
def _(charge, mass, s, t, u):
    x, y = t - mass**2, u - mass**2
    expected = (
        2
        * charge**4
        * (
            x / y
            + y / x
            + 4 * mass**2 * s / (x * y)
            - 4 * mass**4 * s**2 / (x**2 * y**2)
        )
    )
    expected = expected.replace(s, 2 * mass**2 - t - u).together()
    return (expected,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check both cut coordinates

    For each coordinate, the folded routine tests covariant and physical polarization sums, both Ward identities, and the massive and massless references.
    """)
    return


@app.cell
def _(check_cut_coordinate, result):
    results = {}
    routing_rows = []
    for _coordinate in (0, 1):
        _squares, _routes = check_cut_coordinate(_coordinate)
        results.update(_squares)
        routing_rows.extend(_routes)
    assert results[0, "covariant"] == results[1, "covariant"]
    print(
        "Both cut coordinates, physical photon sums, Ward identities and massive diphoton reference passed",
        flush=True,
    )
    cross_section = result
    return cross_section, results, routing_rows


@app.cell
def _(cross_section):
    cross_section
    return


@app.cell(hide_code=True)
def _(mo, routing_rows):
    mo.vstack(
        [
            mo.md(r"""
        **Physical cut momenta and identical photons**

        The table gives the sign in $K(0)=\eta k_1$. Cut propagator denominators
        belong to the phase-space measure and are omitted from the squared
        amplitude. The two forward topologies identify the identical cut photons;
        adding the $t\leftrightarrow u$ assignment restores the fully labeled result.

        This labeled squared amplitude has no final-state $1/2!$ factor. The
        [ordinary-amplitude notebook](/?file=hep/diphoton.py) includes that factor
        when integrating an event rate over both photon labels.
        """),
            mo.ui.table(routing_rows, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    coordinate = mo.ui.dropdown(
        {"First cut photon": 0, "Second cut photon": 1},
        value="First cut photon",
        label="Loop coordinate",
    )
    polarization = mo.ui.dropdown(
        {
            "Covariant": "covariant",
            "Physical · null reference": "null",
            "Physical · timelike reference": "timelike",
            "Ward · first photon": "Ward 0",
            "Ward · second photon": "Ward 1",
        },
        value="Physical · timelike reference",
        label="Photon projectors",
    )
    massless = mo.ui.switch(value=False, label="Massless electron limit")
    mo.hstack([coordinate, polarization, massless])
    return coordinate, massless, polarization


@app.cell
def _(
    E,
    charge,
    coordinate,
    expected,
    mass,
    massless,
    mo,
    polarization,
    results,
    t,
    u,
):
    shown = results[coordinate.value, polarization.value]
    reference = E("0") if polarization.value.startswith("Ward") else expected
    if massless.value:
        shown = shown.replace(mass, 0)
        reference = reference.replace(mass, 0)
    residual = (shown - reference).together()
    assert residual == 0
    sample = (
        shown.replace(mass, 1)
        .replace(charge, 1)
        .replace(t, -1)
        .replace(u, -7)
        .together()
    )
    mo.vstack(
        [
            mo.md(r"**Spin-averaged labeled squared amplitude**"),
            shown.factor(),
            mo.md("**Exact difference from the reference**"),
            residual,
            mo.md(r"""
        With $e=m_e=1$, $t=-1$, $u=-7$ (hence $s=10$), the massive physical
        result is $83/8$. Ward projections vanish. In the massless limit,
        $\overline{|\mathcal M|^2}=2e^4(t/u+u/t)$ and $s=8$ at the same $t,u$.
        """),
            sample,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
