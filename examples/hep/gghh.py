import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="gg → HH · one double box")


@app.cell(hide_code=True)
async def _():
    import marimo as mo
    import sys as _sys

    if _sys.platform == "emscripten":
        import hashlib as _hashlib
        import micropip as _micropip
        from pathlib import Path as _Path
        from pyodide.http import pyfetch as _pyfetch

        # Install the explicitly exported community wheel; no physics files.
        _base = f"{str(mo.notebook_location()).rstrip('/')}/public/fastsecdec"
        _response = await _pyfetch(f"{_base}/manifest.json")
        if not _response.ok:
            raise RuntimeError("Missing wheel manifest; export with --notebook gghh.")
        _wheel = (await _response.json())["wheel"]
        _response = await _pyfetch(f"{_base}/{_wheel['filename']}")
        if not _response.ok:
            raise RuntimeError("Unable to fetch the exported HEPKit wheel.")
        _data = await _response.bytes()
        if _hashlib.sha256(_data).hexdigest() != _wheel["sha256"]:
            raise RuntimeError("HEPKit wheel hash does not match the export manifest.")
        _wheel_path = _Path("/tmp") / _wheel["filename"]
        _wheel_path.write_bytes(_data)
        await _micropip.install(f"emfs:{_wheel_path}")

    from fractions import Fraction
    from symbolica import E, S
    from symbolica.community import hepkit as hep
    from symbolica.community.hepkit.sector_decomposition import (
        QmcSettings, HavanaDiscreteSettings, with_diagram_expressions,
    )
    from symbolica.community.tensor import Tensor, TensorName, Representation, dot
    return E, Fraction, HavanaDiscreteSettings, QmcSettings, Representation, S, Tensor, TensorName, dot, hep, mo, with_diagram_expressions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # $gg\to HH$: one top-quark double box

    Everything needed for this example is in this notebook. HEPKit supplies the
    Standard Model, graph generation, tensor algebra and numerical helicities;
    FastSecDec supplies sector decomposition and integration.

    We compute one color-projected $(+,+)$ diagram at $\sqrt{s}=300$ GeV,
    $m_t=172.5$ GeV and $m_H=125$ GeV. This is one contribution, not the complete
    gauge-invariant amplitude. The measure is $\prod_l d^Dk_l/(i\pi^{D/2})$,
    with $D=4-2\epsilon$, Feynman gauge, $\cos\theta=4/5$, no spin/color
    average and zero top/Higgs widths.

    Use **marimo's editor**. The expensive decomposition and integration cells
    start disabled; enable and run the desired cell. A browser needs the Pyodide community
    wheel; it uses one CPU and generation can take several minutes.
    """)
    return


@app.cell
def _(E, Fraction, hep, mo):
    # Exact transport of numerical external data; no algebra implementation.
    def exact(z):
        return E(str(Fraction(z.real))) + E("1i") * E(str(Fraction(z.imag)))

    mt, mH, sqrt_s = 172.5, 125.0, 300.0
    model = hep.Model.standard_model()
    values = model.scalar_bindings(overrides={
        model.parameter(name).symbol: exact(value)
        for name, value in dict(MT=mt, ymt=mt, MH=mH, WT=0.0, WH=0.0).items()
    })
    mo.ui.table([{"parameter": name, "value (GeV)": value}
                 for name, value in {"mt = ymt": mt, "mH": mH, "sqrt(s)": sqrt_s,
                                     "top/Higgs widths": 0}.items()])
    return exact, mH, model, mt, sqrt_s, values


@app.cell
def _(E, hep, model):
    vertices = [v for v in model.vertex_rules
                if sorted(model.particle(p).pdg_code for p in v.particles)
                in ([-6, 6, 21], [-6, 6, 25])]  # ttg and ttH
    process = model.process([21, 21], [25, 25], vertex_allow=vertices)
    diagrams = process.generate_diagrams(
        loops=2, max_vertices=6, threads=1, projector=E("1"),
        numerator_grouping=hep.NumeratorGrouping("none"),
    ).diagrams
    return diagrams, process


@app.cell
def _(diagrams, mo):
    def is_double_box(d):
        top = [e for e in d.internal_edges if abs(e.particle.pdg_code) == 6]
        gluons = [e for e in d.internal_edges if e.particle.pdg_code == 21]
        if len(top) != 6 or len(gluons) != 1 or len(d.vertices) != 6:
            return False
        if not d.subgraph(edges=[e.id for e in top]).is_connected():
            return False
        legs = {e.source if e.source is not None else e.target: e.particle.pdg_code
                for e in d.external_edges}
        pairs = [(e.source, e.target) for e in top if e.source in legs and e.target in legs]
        # Two disjoint g-H pairs on a connected top hexagon: the gluon is its
        # opposite chord, giving two boxes. All topology queries are native.
        return (len(pairs) == 2 and len({v for pair in pairs for v in pair}) == 4
                and all(sorted([legs[a], legs[b]]) == [21, 25] for a, b in pairs))

    double_boxes = [d for d in diagrams if is_double_box(d)]
    raw_diagram = double_boxes[0]
    mo.vstack([mo.md(f"**{len(double_boxes)} double boxes** among {len(diagrams)} diagrams; using {raw_diagram.name}."),
               mo.as_html(raw_diagram.render())])
    return double_boxes, raw_diagram


@app.cell
def _(E, Representation, S, Tensor, TensorName, dot, exact, hep, mH, raw_diagram, sqrt_s):
    eps, D = S("gghh::eps", "gghh::D")
    P, K = hep.Kinematics.external_momentum(), hep.Kinematics.loop_momentum()
    legs = sorted(hep.Amplitude.from_diagram(raw_diagram).legs, key=lambda leg: leg.index)
    gluons = [leg for leg in legs if leg.particle.pdg_code == 21]
    energy = exact(sqrt_s / 2)
    q = (energy**2 - exact(mH)**2) ** E("1/2")
    physical = [[energy, 0, 0, energy], [energy, 0, 0, -energy],
                [energy, 3*q/5, 0, 4*q/5], [energy, -3*q/5, 0, -4*q/5]]
    physical = dict(zip([leg.index for leg in legs if leg.state == "incoming"]
                       + [leg.index for leg in legs if leg.state == "outgoing"], physical))
    basis = raw_diagram.loop_momentum_basis
    external = {e.id: e.external_index for e in raw_diagram.external_edges}
    pol = [TensorName.vector(f"gghh::eps{i+1}") for i in range(2)]
    states = [hep.FourMomentum(sqrt_s/2, 0, 0, z).wavefunction("epsilon", hep.Helicity.PLUS)
              for z in (sqrt_s/2, -sqrt_s/2)]
    vectors = [(P(i), physical[external[e]]) for i, e in enumerate(basis.external_edges)
               if e not in basis.dependent_externals]
    auxiliaries = [name.to_expression() for name in pol]
    vectors += list(zip(auxiliaries, [[exact(z) for z in state.components] for state in states]))
    tensors = [Tensor.dense(TensorName.vector(f"point::v{i}")(Representation.mink(4)),
                           [E(str(x)) if isinstance(x, int) else x for x in vector])
               for i, (_, vector) in enumerate(vectors)]
    kinematics = hep.Kinematics(D, momenta=[K(i) for i in range(2)] + [p for p, _ in vectors])
    for i, (left, _) in enumerate(vectors):
        for j in range(i, len(vectors)):
            product = dot(tensors[i], tensors[j])
            product.execute()
            kinematics = kinematics.with_scalar_product(left, vectors[j][0], product.result_scalar())
    return D, auxiliaries, eps, gluons, kinematics, pol


@app.cell
def _(D, E, Representation, gluons, kinematics, mo, pol, raw_diagram, with_diagram_expressions):
    color = Representation.coad(8).id(*(leg.tensor_index for leg in gluons))
    polarization = pol[0](next(s for s in gluons[0].slots if s.representation == Representation.mink(4)))
    polarization *= pol[1](next(s for s in gluons[1].slots if s.representation == Representation.mink(4)))
    contracted = (raw_diagram.numerator_expression(in_lmb=True) * color * polarization
                  * raw_diagram.projector_expression()).with_lorentz_dimension(D)
    contracted = contracted.simplify_algebra(
        contract="minimal", color_substitute_cof_dimension_invariants=True,
    ).to_dots()
    assert contracted.is_scalar
    numerator = kinematics.apply(contracted).to_expression()
    # Projection is now in the numerator; keep native graph weights separate.
    diagram = with_diagram_expressions(
        raw_diagram, numerator=numerator, projector=E("1"),
        overall_factor=raw_diagram.overall_factor_expression(evaluate=True),
    )
    mo.accordion({"Contracted scalar numerator · preview": mo.as_html(
        numerator.formatted(max_terms=8, max_line_length=90, show_namespaces=False))})
    return diagram, numerator


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Decompose, then compile

    The input is the HEPKit diagram with its contracted numerator and kinematics.
    Its seven physical propagators retain their unit powers. Enable the next
    cell to generate through the finite coefficient. The usual domain check
    stays active; no threshold-free assumption is forced.
    """)
    return


@app.cell(disabled=True)
def _(auxiliaries, diagram, eps, kinematics, values):
    generated = diagram.sector_decompose(
        regulator=eps, dimension=4-2*eps, kinematics=kinematics,
        scalar_values=values, auxiliary_momenta=auxiliaries, max_order=0,
        coefficient_expansion="native_named", progress="auto",
    )
    kernels = generated.compile(progress="auto")
    return generated, kernels


@app.cell
def _(generated, kernels, mo):
    sector_choice = mo.ui.dropdown({f"Sector {s.index}": s.index for s in generated.sectors},
                                   value="Sector 0", label="Inspect")
    mo.vstack([mo.ui.table([
        {"sector": s.index, "dimension": s.dimension, "orders": generated.orders,
         "evaluator bytes": stats.exact_program_bytes, "SymJIT application bytes": stats.symjit_ir_bytes,
         "additions": stats.operations.additions, "multiplications": stats.operations.multiplications}
        for s, stats in zip(generated.sectors, kernels.sector_statistics)
    ]), mo.md("Expression sizes describe the shared sector evaluator. SymJIT application bytes measure its serialized program; browser kernels use the interpreter."), sector_choice])
    return (sector_choice,)


@app.cell
def _(generated, mo, sector_choice):
    charts = [c for c in generated.metadata.charts if c.kernel_sector == sector_choice.value]
    mo.accordion({f"Chart {c.source_index}": mo.vstack([
        mo.md("**Coordinate map**"),
        mo.ui.table([{"source": str(p), "image": str(v)}
                     for p, v in zip(c.coordinates.source_parameters, c.coordinates.images)]),
        *[mo.vstack([mo.md(f"**Mapped term {i}: prefactor**"), mo.as_html(t.prefactor),
                     mo.ui.table([{"coordinate": str(p), "epsilon-dependent exponent": str(a.exponent),
                                   "Taylor terms": a.subtraction_count}
                                  for p, a in zip(c.coordinates.target_parameters, t.powers)])])
          for i, t in enumerate(c.pre_subtraction.terms)],
    ]) for c in charts})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate

    Creating a session samples nothing. Enable the integration cell to advance
    it; interrupt with marimo's Stop control. Rerunning only that cell continues
    the same session. Long native calls can delay browser interruption.
    The small default allocation is an exploration, not a precision guarantee.
    """)
    return


@app.cell
def _(QmcSettings, kernels):
    session = kernels.session(QmcSettings(points=1024, shifts=8, seed=20261005))
    # Larger native observation: points=32768, shifts=16, rule="hkkn_alpha3", seed=20261007.
    return (session,)


@app.cell
def _(mo):
    def integrate(session):
        while not session.complete:
            snapshot = session.step()  # one QMC package or global Havana batch
            estimate = snapshot.estimate
            rows = [] if estimate is None else [
                {"epsilon power": n, "component": part, "value": value, "standard error": error,
                 "relative error (per mil)": 1000*error/abs(value) if value else None}
                for n, part, value, error in zip(estimate.orders, estimate.components,
                                                estimate.mean, estimate.standard_error)]
            mo.output.replace(mo.vstack([
                mo.hstack([mo.stat(label="Accepted points", value=f"{snapshot.completed_points:,}"),
                           mo.stat(label="Planned", value=f"{snapshot.planned_points:,}"),
                           mo.stat(label="Worker time", value=f"{snapshot.worker_seconds:.1f} s")]),
                mo.md(f"**{snapshot.method} · {snapshot.stage}** · uncertainty: {snapshot.uncertainty}"),
                mo.ui.table(rows) if rows else mo.md("Waiting for a native production estimate…"),
            ]))
        return session.snapshot()
    return (integrate,)


@app.cell(disabled=True)
def _(integrate, session):
    qmc_result = integrate(session)
    return (qmc_result,)


@app.cell
def _(HavanaDiscreteSettings, kernels):
    # Optional comparison: native discrete importance sampling over sectors.
    mc = kernels.mc_session(HavanaDiscreteSettings(points_per_batch=1024, batches=4), pilot=True)
    return (mc,)


@app.cell(disabled=True)
def _(integrate, mc):
    if mc.stage == "pilot":
        integrate(mc)
        mc.freeze_production(points_per_batch=1024, batches=8)
    mc_result = integrate(mc)
    return (mc_result,)


@app.cell
def _(kernels, mo):
    from symbolica import get_citations

    kernels  # Refresh the bibliography after native generation and compilation.
    citations = get_citations()
    mo.vstack([
        mo.md("## References"),
        *[mo.as_html(citation) for citation in citations],
        mo.download("\n\n".join(citation.to_bibtex() for citation in citations).encode(),
                    filename="fastsecdec-references.bib", label="Download BibTeX"),
    ])
    return (citations,)


if __name__ == "__main__":
    app.run()
