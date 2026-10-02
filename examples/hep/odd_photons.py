import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Odd photons and Furry cancellation",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Odd photons and Furry cancellation

    [Browse notebooks](/) · [Photon self-energy](/?file=hep/photon_self_energy.py) ·
    [Higgs to gluons with IBP](/?file=hep/higgs_gluons.py) ·
    [Tadpole mass insertions](/?file=hep/tadpole_mass_insertions.py)

    A massive charged-fermion loop coupled to an odd number of photons gives
    zero. Generate the diagrams, retain their native weights, and verify the
    cancellation with shared Dirac algebra and integral-family momentum maps.
    This follows FeynCalc's
    [one-photon](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/Ga),
    [three-photon](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/Ga-GaGa)
    and [five-photon](https://feyncalc.github.io/FeynCalcExamples/QED/OneLoop/Ga-GaGaGaGa)
    examples.

    The external momenta are generic and off shell, the electron mass remains
    symbolic, and the Lorentz dimension is $D$ with $\operatorname{tr}(1)=4$.
    Independent probe vectors contract every external Lorentz index; no
    transversality or physical polarization sum is imposed. Thus an exact
    cancellation in all probes also establishes the open-tensor identity.
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
    from symbolica import E, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import IntegralFamily, Kinematics, Model, TensorReducer
    from symbolica.community.tensor import TensorExpression

    _set_namespace("furry")
    return (
        E,
        IntegralFamily,
        Kinematics,
        Model,
        S,
        TensorExpression,
        TensorReducer,
        hep,
        mo,
        sp,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Model, mo):
    model = Model.standard_model()
    photons = mo.ui.dropdown(
        {
            "One photon: odd tadpole": 1,
            "Three photons: triangle pair": 3,
            "Five photons: twelve pentagon pairs": 5,
        },
        value="Three photons: triangle pair",
        label="External photons",
    )
    mo.vstack([photons])
    return model, photons


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the closed fermion loops
    """)
    return


@app.cell
def _(S, hep, model, photons, sp):
    count = photons.value
    electron, photon = (model.particle(_name) for _name in ("e-", "a"))
    vertices = [
        v
        for v in model.vertex_rules
        if sorted(v.particles)
        == sorted([electron.name, electron.antiname, photon.name])
    ]
    assert len(vertices) == 1
    D = S("D")
    K, P, mink = (
        hep.Kinematics.loop_momentum,
        hep.Kinematics.external_momentum,
        sp.Representation.mink,
    )
    generated = model.process(
        [photon], [photon] * (count - 1), vertex_allow=vertices
    ).generate_diagrams(
        loops=1,
        max_vertices=count,
        allow_self_loops=True,
        allow_zero_flow_edges=True,
        maximum_bridges=None,
        self_energy=None,
        tadpoles=None,
        zero_snails=None,
        numerator_grouping=None,
        progress=None,
    )
    assert len(generated.diagrams) == {1: 1, 3: 2, 5: 24}[count]
    return D, K, P, count, generated


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Identify their integral families
    """)
    return


@app.cell
def _(D, IntegralFamily, K, Kinematics, P, count, generated):
    probes = [P(100 + i) for i in range(count)]
    kinematics = Kinematics(
        D, momenta=[K(0)] + [P(i) for i in range(count - 1)] + probes
    )
    families = []
    for _diagram in generated.diagrams:
        raw = _diagram.propagator_family(kinematics=kinematics)
        # Include probes so the scalar map also transports each loop-probe product.
        families.append(
            IntegralFamily(
                raw.loop_momenta,
                raw.external_momenta + probes,
                raw.denominators,
                kinematics=kinematics,
            )
        )
    return families, kinematics


@app.cell
def _(K, count, families, generated):
    pairs = []
    if count == 1:
        _mapping = families[0].mapping_to(families[0], [-K(0)])
        assert _mapping is not None
        pairs.append((0, 0, _mapping))
    else:
        remaining = set(range(len(families)))
        while remaining:
            _left = min(remaining)
            remaining.remove(_left)
            candidates = []
            for _right in sorted(remaining):
                _mapping = families[_right].find_mapping(families[_left])
                if _mapping is not None:
                    candidates.append((_right, _mapping))
            assert len(candidates) == 1, (_left, candidates)
            _right, _mapping = candidates[0]
            assert dict(_mapping.momentum_rules)[K(0)].coefficient(K(0)) == -1
            assert _mapping.map_powers([1] * count) == [1] * count
            pairs.append((_left, _right, _mapping))
            remaining.remove(_right)
        assert len(pairs) == len(generated.diagrams) // 2
    return (pairs,)


@app.cell(hide_code=True)
def _(count, generated, mo, pairs):
    selected_pair = mo.ui.dropdown(
        {
            f"Diagram {left + 1}"
            if count == 1
            else f"Diagrams {left + 1} and {right + 1}": i
            for i, (left, right, _) in enumerate(pairs)
        },
        value="Diagram 1"
        if count == 1
        else f"Diagrams {pairs[0][0] + 1} and {pairs[0][1] + 1}",
        label="Evaluate orientation pair",
    )
    mo.vstack(
        [
            mo.md(
                f"**{len(generated.diagrams)} generated diagrams; "
                f"{'one odd tadpole' if count == 1 else str(len(pairs)) + ' orientation pairs'}.** "
                "Select a pair to compute its exact cancellation. "
                "A pentagon pair takes about twenty seconds."
            ),
            selected_pair,
        ]
    )
    return (selected_pair,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Trace the selected orientations
    """)
    return


@app.cell
def _(
    D,
    E,
    P,
    S,
    TensorExpression,
    count,
    generated,
    kinematics,
    model,
    pairs,
    selected_pair,
    sp,
):
    left, right, mapping = pairs[selected_pair.value]
    index, wave = S("index_", "wave_")
    traces = []
    for position in [left] if count == 1 else [left, right]:
        diagram = generated.diagrams[position]
        numerator = model.expand_couplings(
            diagram.numerator_expression().to_expression()
        )
        for edge in diagram.external_edges:
            port = dict(
                next(
                    diagram.projector_expression().match(
                        wave(
                            edge.id,
                            sp.PortPattern.exact(sp.Representation.mink(4), index),
                        ),
                        max_level=0,
                    )
                )
            )[index]
            numerator *= P(
                100 + edge.external_index,
                sp.PortPattern.exact(sp.Representation.mink(4), port),
            )
        numerator = numerator.replace(
            sp.PortPattern.exact(sp.Representation.mink(4), index),
            sp.PortPattern.exact(sp.Representation.mink(D), index),
        )
        # Trace compact edge momenta before expanding their routed linear combinations.
        tensor = (
            TensorExpression(numerator.expand())
            .simplify_algebra(contract="dots", gamma=True, epsilon=True)
            .expand()
            .to_expression()
        )
        assert TensorExpression(tensor).is_scalar
        trace = (
            kinematics.apply(diagram.momentum_basis().route_expression(tensor))
            * diagram.overall_factor_expression(evaluate=True)
            * diagram.numerator_prefactor_expression()
        ).expand()
        assert trace != E("0")
        traces.append(trace)
    return left, mapping, right, traces


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Verify Furry cancellation
    """)
    return


@app.cell
def _(D, E, K, TensorReducer, count, mapping, sp, traces):
    if count == 1:
        assert (mapping.apply(traces[0]) + traces[0]).expand() == E("0")
        vacuum = TensorReducer(
            D, integrated=[K(0, sp.PortPattern.exact(sp.Representation.mink(D)))]
        )
        residual = vacuum.reduce(traces[0]).expand()
    else:
        residual = (traces[0] + mapping.apply(traces[1])).expand()
    assert residual == E("0")
    return (residual,)


@app.cell(hide_code=True)
def _(count, families, generated, left, mapping, mo, residual, right, traces):
    mo.vstack(
        [
            mo.hstack(
                [
                    generated.diagrams[i]
                    for i in ([left] if count == 1 else [left, right])
                ]
            ),
            mo.md("**Verified loop-momentum substitution**"),
            mo.hstack([mo.hstack(list(rule)) for rule in mapping.momentum_rules]),
            mo.md(
                f"**Denominator permutation** (zero-based): `{mapping.denominator_map}`"
            ),
            mo.md(
                "**Result after odd vacuum projection**"
                if count == 1
                else "**Sum of the two weighted numerators in the same routing**"
            ),
            residual,
            mo.accordion(
                {
                    "Integral family": families[left],
                    "First weighted numerator": traces[0],
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The one-photon graph has a nonzero numerator odd in the loop momentum;
    vacuum tensor reduction gives zero. For three and five photons, each
    nonzero numerator has an opposite-orientation partner. The verified map
    preserves the integration measure and permutes the unit-power
    denominators. Their numerator sum is exactly zero in a common routing.

    This check happens before IBP or scalar-master evaluation. The
    [Higgs-to-gluons notebook](/?file=hep/higgs_gluons.py) follows a nonzero
    fermion loop through those later steps. Here the generator's tadpole and
    bridge filters are explicitly relaxed so the one-point graph is retained;
    every generated graph keeps its original fermion and symmetry factors.
    """)
    return


if __name__ == "__main__":
    app.run()
