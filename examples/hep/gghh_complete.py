import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="gg → HH · from native diagrams to integrals")


@app.cell(hide_code=True)
async def _():
    import marimo as mo
    import sys

    if sys.platform == "emscripten":
        import hashlib
        from pathlib import Path
        from importlib import import_module

        # Only the installed HEPKit wheel is external. All notebook helpers,
        # native model setup and physical inputs are defined below in this file.
        # Browser-only packages stay outside native Marimo's static registry.
        _micropip = import_module("micropip")
        _pyfetch = import_module("pyodide.http").pyfetch
        _base = f"{str(mo.notebook_location()).rstrip('/')}/public/fastsecdec"
        _response = await _pyfetch(f"{_base}/manifest.json")
        if not _response.ok:
            raise RuntimeError("Export this notebook with the tested HEPKit Pyodide wheel.")
        _manifest = await _response.json()
        _entry = _manifest["wheel"]
        _filename = Path(_entry["filename"])
        if _filename.is_absolute() or len(_filename.parts) != 1 or _filename.name in {"", ".", ".."}:
            raise RuntimeError("Invalid exported wheel filename")
        _response = await _pyfetch(f"{_base}/{_filename.name}")
        if not _response.ok:
            raise RuntimeError("Missing exported wheel")
        _data = await _response.bytes()
        if hashlib.sha256(_data).hexdigest() != _entry["sha256"]:
            raise RuntimeError("Exported wheel hash differs")
        _root = Path("/fastsecdec-complete")
        _root.mkdir(exist_ok=True)
        _wheel = _root / _filename.name
        _wheel.write_bytes(_data)
        await _micropip.install(f"emfs:{_wheel}")

    # Make the wheel-install dependency explicit to cells importing HEPKit.
    runtime_ready = True
    return mo, runtime_ready


@app.cell(hide_code=True)
def _(runtime_ready):
    def _definitions():
        from types import SimpleNamespace
        from dataclasses import dataclass
        from symbolica import E, Expression
        from symbolica.community import hepkit as hep


        @dataclass(frozen=True)
        class ShowcaseInput:
            """Prepared native owners and explicit integral conventions for one example."""

            name: str
            model: hep.Model
            diagram: hep.FeynmanDiagram
            kinematics: hep.Kinematics
            regulator: Expression
            dimension: Expression
            scalar_values: dict[Expression, Expression]
            max_order: int

            def fixed_scalar_values(self):
                # The massless examples belong to the exact zero-mass stratum. A named
                # runtime mass may not cross into it without regenerating the template.
                return {symbol: value for symbol, value in self.scalar_values.items() if value == E("0")}

            def integral_arguments(self):
                """Native arguments for diagram sector decomposition; no serialization step."""
                return {
                    "diagram": self.diagram,
                    "kinematics": self.kinematics,
                    "regulator": self.regulator,
                    "dimension": self.dimension,
                    "model_parameters": "runtime",
                    "scalar_values": self.fixed_scalar_values(),
                    "powers": {},
                    "auxiliary_momenta": [],
                    "measure_multiplier": E("1"),
                }

            def scalar_numerator(self):
                """Display the fully weighted numerator through native tensor algebra."""
                numerator = (
                    self.diagram.numerator_expression(in_lmb=True)
                    * self.diagram.projector_expression()
                    * self.diagram.numerator_prefactor_expression()
                    * self.diagram.overall_factor_expression()
                ).with_lorentz_dimension(self.kinematics.dimension)
                numerator = numerator.simplify_algebra(contract="minimal").to_dots()
                if not numerator.is_scalar:
                    raise ValueError("The prepared numerator still has free tensor indices")
                return self.kinematics.apply(numerator).to_expression()

        return ShowcaseInput

    ShowcaseInput = _definitions()
    return (ShowcaseInput,)


@app.cell(hide_code=True)
def _():
    def _definitions():
        from types import SimpleNamespace
        """Formatting only; native values and uncertainty are never recomputed."""

        from decimal import Decimal, ROUND_HALF_EVEN, localcontext
        import math

        _SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        _COUNT_COLUMNS = {
            "evaluations", "conditioning_checks", "rescues", "failures",
            "weighted_checks", "additional_replays",
        }


        def scientific(value):
            """Signed, normalized human notation; never used for native calculations."""
            if not math.isfinite(value):
                return "unavailable"
            mantissa, exponent = format(abs(value), ".6e").split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            sign = "−" if value < 0 else "+"
            return f"{sign}{mantissa} ·10{str(int(exponent)).translate(_SUPERSCRIPTS)}"


        def uncertainty(value, error):
            """Two significant error digits, in units of the last shown central digit.

            The Python owner has no public mean/error formatter. Native float formatting
            rounds the supplied error; Decimal only places its display digits safely
            across the finite f64 exponent range. It does not estimate an uncertainty.
            """
            if not math.isfinite(value):
                return "unavailable"
            if error is None or not math.isfinite(error) or error < 0:
                return f"{scientific(value)} (σ unavailable)"
            central = Decimal(str(abs(value)))
            exponent = central.adjusted() if value else (Decimal(str(error)).adjusted() if error else 0)
            sign = "−" if value < 0 else "+"
            with localcontext() as context:
                # Exact binary64 decimal expansions and the largest relative exponent
                # separation fit here; the caller's Decimal context is never modified.
                context.prec = 1200
                if error == 0:
                    mantissa = format(central.scaleb(-exponent), "f")
                    if "." in mantissa:
                        mantissa = mantissa.rstrip("0").rstrip(".")
                    body = f"{mantissa}(0)"
                else:
                    rounded_error = Decimal(format(error, ".1e"))
                    decimals = max(1 - (rounded_error.adjusted() - exponent), 0)
                    mean = Decimal.from_float(float(abs(value))).scaleb(-exponent)
                    rounded_mean = mean.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
                    if rounded_mean >= 10:
                        exponent += 1
                        decimals = max(1 - (rounded_error.adjusted() - exponent), 0)
                        mean = mean.scaleb(-1)
                        rounded_mean = mean.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_EVEN)
                    digits = rounded_error.scaleb(decimals - exponent)
                    body = f"{rounded_mean:.{decimals}f}({digits:.0f})"
            return f"{sign}{body} ·10{str(exponent).translate(_SUPERSCRIPTS)}"


        def compact_count(count):
            """Four significant decimal digits, with base-1000 K/M/B suffixes."""
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("A count must be a nonnegative integer")
            if count < 1000:
                return str(count)
            scale = 10 ** max(len(str(count)) - 4, 0)
            rounded = (count + scale // 2) // scale * scale
            unit, suffix = ((1_000_000_000, "B") if rounded >= 1_000_000_000 else
                            (1_000_000, "M") if rounded >= 1_000_000 else (1000, "K"))
            whole = rounded // unit
            digits = max(4 - len(str(whole)), 0)
            if not digits:
                return f"{whole} {suffix}"
            fraction = (rounded % unit) // (unit // 10 ** digits)
            return f"{whole}.{fraction:0{digits}d} {suffix}"


        # User-selected allocations; these never run or change an existing native owner.


        def epsilon_label(order):
            return "ε" + str(order).translate(_SUPERSCRIPTS)

        def table(mo, rows, *, scientific_values=False):
            if not rows:
                return mo.md("No native observations yet.")
            float_format = scientific if scientific_values else lambda value: f"{value:.8g}"
            formats = {key: (lambda value: "—" if value is None else float_format(value))
                       for key in rows[0] if any(isinstance(row.get(key), float) for row in rows)}
            for key in rows[0]:
                if key in _COUNT_COLUMNS or any(word in key.split() for word in ("points", "batches", "shifts")):
                    formats[key] = lambda value: ("—" if value is None else compact_count(value)
                                                  if isinstance(value, int) and not isinstance(value, bool) else str(value))
            if "worker seconds" in rows[0]:
                formats["worker seconds"] = lambda value: "—" if value is None else f"{value:.3g}"
            if "relative error" in rows[0]:
                formats["relative error"] = lambda value: "—" if value is None else f"{100 * value:.4g}%"
            return mo.ui.table(rows, selection=None, show_column_summaries=False,
                               show_data_types=False, pagination=len(rows) > 12,
                               page_size=12, format_mapping=formats)


        def panel(mo, title, content, *, expanded=True):
            """An open, styled monitor with Marimo 0.24-compatible containers."""
            if not expanded:
                return mo.accordion({title: content})
            return mo.vstack([mo.md(f"### {title}"), content]).style({
                "border": "1px solid var(--gray-5)", "border-radius": "12px",
                "padding": "1rem", "background": "var(--gray-1)",
            })

        return SimpleNamespace(scientific=scientific, uncertainty=uncertainty, compact_count=compact_count, epsilon_label=epsilon_label, table=table, panel=panel)

    formatting = _definitions()
    return (formatting,)


@app.cell(hide_code=True)
def _(ShowcaseInput, runtime_ready):
    def _definitions():
        from types import SimpleNamespace
        """Native Standard Model gg → HH catalogue and selected-diagram preparation.

        Importing this module does no generation, contraction or sampling. The caller
        explicitly builds the catalogue, chooses a native diagram, then prepares it.
        """
        from dataclasses import dataclass

        from symbolica import E, S
        from symbolica.community import hepkit as hep
        from symbolica.community.tensor import Representation, Tensor, TensorName, dot



        def exact(value):
            value = complex(value)
            def real(number):
                numerator, denominator = float(number).as_integer_ratio()
                return E(str(numerator)) / E(str(denominator))
            return real(value.real) + E("1i") * real(value.imag)


        def standard_model():
            model = hep.Model.standard_model()
            card = model.default_parameter_card()
            for name, value in {"MT": 172.5, "ymt": 172.5, "MH": 125.0,
                                "WT": 0.0, "WH": 0.0}.items():
                card.set(name, value, 0.0)
            model = model.with_parameter_card(card)
            values = model.scalar_bindings(card)
            return model, values


        def process(model):
            """Restrict particles, never individual vertices of the allowed theory."""
            return model.process([21, 21], [25, 25], particle_selection=[6, 21, 25])


        @dataclass(frozen=True)
        class Catalogue:
            model: object
            scalar_values: dict
            process: object
            result: object

            @property
            def diagrams(self):
                return self.result.diagrams

            @property
            def default_diagram(self):
                return next(diagram for diagram in self.diagrams if diagram.loop_count == 1)

            def selected(self, identity=None):
                if identity is None:
                    return self.default_diagram
                return next(diagram for diagram in self.diagrams if diagram.id == identity)


        def catalogue(*, progress="auto"):
            model, values = standard_model()
            native_process = process(model)
            result = native_process.generate_diagrams(
                loops=(1, 2), coupling_orders={"QED": 2}, threads=1,
                symmetrize_initial=True, symmetrize_final=True, allow_zero_flow_edges=True,
                maximum_bridges=None, self_energy=None, tadpoles=None, zero_snails=None,
                numerator_grouping=None,
                projector=E("1"), progress=progress,
            )
            if not result.report.completed:
                raise RuntimeError("Native diagram generation did not complete")
            if not any(diagram.loop_count == 1 for diagram in result.diagrams):
                raise RuntimeError("The generated catalogue contains no one-loop diagram")
            return Catalogue(model, values, native_process, result)


        @dataclass(frozen=True)
        class GGHHInput(ShowcaseInput):
            auxiliary_momenta: tuple
            raw_diagram: object
            raw_numerator: object
            simplified_numerator: object
            gram_symbols: tuple

            def fixed_scalar_values(self):
                # Zero widths are the declared real-mass convention, not numerical
                # integration defaults for freely varying complex propagator masses.
                return {self.model.parameter(name).symbol: E("0") for name in ("WT", "WH")}

            def integral_arguments(self):
                arguments = super().integral_arguments()
                arguments["auxiliary_momenta"] = list(self.auxiliary_momenta)
                arguments["runtime_parameters"] = [symbol for _, _, symbol in self.gram_symbols]
                return arguments

            def runtime_point(self, point):
                """Bind the native physical Gram matrix without regenerating sectors."""
                named, _ = _external_data(self.raw_diagram, point)
                result = {}
                for left, right, symbol in self.gram_symbols:
                    value = complex(_dot(named[left][1], named[right][1]))
                    if value.imag != 0:
                        raise ValueError("The chosen scattering plane requires real Gram values")
                    result[symbol] = value.real
                return result

            def generation_arguments(self):
                return {"coefficient_expansion": "coefficient_series"}


        def _tensor(name, components):
            return Tensor.dense(TensorName.vector(name)(Representation.mink(4)), components)


        def _dot(left, right):
            product = dot(left, right)
            product.execute()
            return product.result_scalar().expand()


        def _external_data(raw, point):
            import math
            if point is None:
                e, mass, cosine = S("gghh_point::energy", "gghh_point::higgs_mass", "gghh_point::cos_theta")
                energy = 150.0  # Polarizations are dimensionless; native fixed-helicity convention.
            else:
                energy = float(point.get("sqrt_s", 300)) / 2
                mass = float(point.get("higgs_mass", 125))
                cosine = float(point.get("cos_theta", 0.8))
                if not all(math.isfinite(x) for x in (energy, mass, cosine)) or mass <= 0 or energy <= mass or abs(cosine) >= 1:
                    raise ValueError("Require sqrt(s) > 2 mH > 0 and -1 < cos(theta) < 1")
                e, mass, cosine = exact(energy), exact(mass), exact(cosine)
            legs = sorted(hep.Amplitude.from_diagram(raw).legs, key=lambda leg: leg.index)
            incoming = [leg.index for leg in legs if leg.state == "incoming"]
            outgoing = [leg.index for leg in legs if leg.state == "outgoing"]
            momentum = (e**2 - mass**2) ** E("1/2")
            longitudinal = momentum * cosine
            transverse = momentum * (1 - cosine**2) ** E("1/2")
            zero = E("0")
            vectors = [[e,zero,zero,e], [e,zero,zero,-e],
                       [e,transverse,zero,longitudinal], [e,-transverse,zero,-longitudinal]]
            physical = dict(zip(incoming + outgoing, vectors))
            external = {edge.id: edge.external_index for edge in raw.external_edges}
            basis = raw.loop_momentum_basis
            P = hep.Kinematics.external_momentum()
            polarizations = [TensorName.vector(f"gghh::eps{i+1}") for i in range(2)]
            states = [hep.FourMomentum(energy, 0, 0, z).wavefunction("epsilon", hep.Helicity.PLUS)
                      for z in (energy, -energy)]
            named = [(P(i), _tensor(f"gghh_data::p{i}", physical[external[edge]]))
                     for i, edge in enumerate(basis.external_edges) if edge not in basis.dependent_externals]
            named += [(name.to_expression(), _tensor(f"gghh_data::epsilon{i}", [exact(z) for z in state.components]))
                      for i, (name, state) in enumerate(zip(polarizations, states))]
            return named, polarizations


        def prepare(*, selected=None, source=None, observer=None):
            """Contract one chosen native owner, preserving all generated graph factors.

            The (+,+), delta_ab projection retains symbolic Gram products; the initial
            integration point is sqrt(s)=300 GeV, mt=172.5 GeV, mH=125 GeV and cos(theta)=4/5.
            It is a single diagram contribution, not a
            gauge-invariant amplitude or a claim of threshold regularization.
            """
            source = source if source is not None else catalogue(progress=observer or "auto")
            raw = source.selected(selected)
            raw.validate()
            legs = sorted(hep.Amplitude.from_diagram(raw).legs, key=lambda leg: leg.index)
            gluons = [leg for leg in legs if leg.particle.pdg_code == 21]
            incoming = [leg.index for leg in legs if leg.state == "incoming"]
            outgoing = [leg.index for leg in legs if leg.state == "outgoing"]
            if len(gluons) != 2 or len(incoming) != 2 or len(outgoing) != 2:
                raise ValueError("Expected native gg → HH external ports")
            K = hep.Kinematics.loop_momentum()
            regulator, dimension = S("gghh::eps", "gghh::D")
            named, polarization_names = _external_data(raw, None)
            auxiliary = tuple(name.to_expression() for name in polarization_names)
            kinematics = hep.Kinematics(dimension,
                momenta=[K(i) for i in range(raw.loop_count)] + [name for name, _ in named])
            gram_symbols = []
            for i, (left, a) in enumerate(named):
                for j, (right, b) in enumerate(named[i:], i):
                    value = _dot(a, b)
                    # Preserve only native exact structural zeros. Even dimensionless
                    # polarization products are runtime inputs: their numerical native
                    # wavefunctions must not freeze binary64 normalizations into algebra.
                    if value != E("0"):
                        value = S(f"gghh_kinematics::dot_{i}_{j}", is_real=True)
                        gram_symbols.append((i, j, value))
                    kinematics = kinematics.with_scalar_product(left, right, value)
            color = Representation.coad(8).id(*(leg.tensor_index for leg in gluons))
            slots = [[slot for slot in leg.slots if slot.representation == Representation.mink(4)]
                     for leg in gluons]
            if any(len(value) != 1 for value in slots):
                raise ValueError("Expected one Lorentz slot on each external gluon")
            polarization = polarization_names[0](slots[0][0]) * polarization_names[1](slots[1][0])
            raw_numerator = raw.numerator_expression(in_lmb=True)
            contracted = (raw_numerator * color * polarization * raw.projector_expression()).with_lorentz_dimension(dimension)
            contracted = contracted.simplify_algebra(
                contract="minimal", color_substitute_cof_dimension_invariants=True,
            ).to_dots()
            if not contracted.is_scalar:
                raise ValueError("Native numerator contraction left free tensor indices")
            simplified = kinematics.apply(contracted).to_expression()
            diagram = hep.sector_decomposition.with_diagram_expressions(
                raw, numerator=simplified, projector=E("1"),
                overall_factor=raw.overall_factor_expression(evaluate=True),
            )
            return GGHHInput(
                f"gg → HH · {raw.name} · {raw.loop_count} loop(s)", source.model, diagram,
                kinematics, regulator, 4 - 2 * regulator, source.scalar_values, 0,
                auxiliary, raw, raw_numerator, simplified, tuple(gram_symbols),
            )

        return SimpleNamespace(exact=exact, standard_model=standard_model, process=process, Catalogue=Catalogue, catalogue=catalogue, GGHHInput=GGHHInput, _tensor=_tensor, _dot=_dot, _external_data=_external_data, prepare=prepare)

    gghh_inputs = _definitions()
    return (gghh_inputs,)


@app.cell(hide_code=True)
def _(runtime_ready):
    def _definitions():
        from types import SimpleNamespace
        """Short native scientific calls shared by both notebook frontends."""
        from symbolica.community.hepkit import sector_decomposition as sd


        def generation(prepared, configuration):
            integral = sd.Integral(**prepared.integral_arguments())
            return integral.generation_session(
                max_order=configuration.get("max_order", 0),
                coefficient_expansion="coefficient_series",
                compilation_settings=sd.CompilationSettings(backend="eager"),
            )


        def integration(kernels, configuration):
            if configuration.get("method") == "havana_discrete_mc":
                return kernels.mc_session(sd.HavanaDiscreteSettings(
                    points_per_batch=configuration["pilot_points"],
                    batches=configuration["pilot_batches"], seed=configuration["seed"],
                    bins=configuration["bins"],
                    minimum_probability_density=configuration["minimum_probability_density"],
                    maximum_sector_probability_ratio=configuration["maximum_sector_probability_ratio"],
                ), pilot=True)
            return kernels.session(sd.QmcSettings(
                points=configuration["points"], shifts=configuration["shifts"],
                seed=configuration["seed"], package_points=configuration["package_points"],
                rule=configuration.get("rule", "kuo_33002"),
                periodization=configuration.get("periodization", "korobov3"),
            ))


        def step(session, method):
            if method == "havana_discrete_mc":
                return session.step(max_batches=1, evaluation_batch_size=256)
            return session.step(max_packages=1, evaluation_batch_size=256)


        def bind(generated, kernels, prepared, point):
            """Explicitly select one physical point on independent native evaluators."""
            from symbolica import S
            values = dict(generated.runtime_parameter_defaults)
            if hasattr(prepared, "runtime_point"):
                values.update(prepared.runtime_point(point))
                # These independent Standard Model leaves define this chosen point.
                for name, value in {"MT": point["top_mass"], "ymt": point["top_mass"],
                                    "MH": point["higgs_mass"]}.items():
                    symbol = S(f"model::{name}")
                    if symbol in kernels.runtime_parameters:
                        values[symbol] = float(value)
            values = {symbol: values[symbol] for symbol in kernels.runtime_parameters}
            return kernels.with_parameters(values, stability=sd.StabilitySettings(mode="distance")), values

        return SimpleNamespace(generation=generation, integration=integration, step=step, bind=bind)

    science = _definitions()
    return (science,)


@app.cell(hide_code=True)
def _(formatting):
    def _definitions():
        from types import SimpleNamespace
        table = formatting.table

        """Views of actual native generation events and timings."""


        def phase_rows(events):
            """Keep native event times; do not infer unobserved phase boundaries."""
            rows = []
            for event in events:
                if not rows or rows[-1]["phase"] != event.stage:
                    rows.append({"phase": event.stage, "first event (s)": event.elapsed_seconds})
                rows[-1].update({"last event (s)": event.elapsed_seconds,
                                 "completed": event.completed, "total": event.total,
                                 "sectors": event.sectors, "kernels": event.kernels})
            return rows

        def timing_rows(event):
            names = ["input", "parametrization", "domain", "geometry", "mapping", "symmetry", "subtraction", "laurent", "coefficient_expansion", "compilation", "total"]
            return [{"native phase": name, "seconds": getattr(event.timings, f"{name}_seconds")} for name in names]

        def generation_view(mo, state):
            if not state.events:
                return mo.md("Preparing the native input. Individual algebra units are atomic; Pause takes effect at the next retained boundary.")
            last = state.events[-1]
            progress = f"{last.completed:,} / {last.total:,}" if last.total is not None else f"{last.completed:,} observed"
            content = [
                mo.hstack([mo.stat(label=last.stage.replace("_", " ").title(), value=progress),
                           mo.stat(label="Native elapsed", value=f"{last.elapsed_seconds:.2f} s"),
                           mo.stat(label="Sectors / kernels", value=f"{last.sectors} / {last.kernels}")], widths="equal"),
                mo.md(last.detail),
            ]
            if last.total is not None and last.total > 0:
                content.append(mo.Html(f'<progress value="{last.completed}" max="{last.total}" style="width:100%;accent-color:#5b5bc4"></progress>'))
            detail = {"Observed phase timeline": table(mo, phase_rows(state.events)),
                      "Native phase timings": table(mo, timing_rows(last))}
            coefficient = last.coefficient_expansion
            if coefficient is not None:
                # The native detail above supplies stage-valid request/piece counts.
                # Raw zero-initialized counters are not measurements before their stage.
                detail["Coefficient expansion"] = table(mo, [{
                    "sector": coefficient.sector, "stage": coefficient.stage,
                    "requested method": coefficient.requested_method, "effective method": coefficient.effective_method,
                    **({"epsilon expansion pass": coefficient.attempt,
                        "relative epsilon depth": coefficient.relative_width} if coefficient.attempt else {}),
                }])
            if last.stage == "complete":
                t = last.timings
                content.append(mo.md(f"Geometry **{t.geometry_seconds:.3f} s** · Mapping **{t.mapping_seconds:.3f} s** · Symmetry **{t.symmetry_seconds:.3f} s** · Compilation **{t.compilation_seconds:.3f} s**"))
            content.append(mo.accordion(detail))
            return mo.vstack(content)


        def input_view(mo, state):
            if state.prepared is None:
                return mo.md("The native diagram appears after Generate.")
            prepared = state.prepared
            details = {"Generated configuration": table(mo, [{"parameter": key, "value": str(value)} for key, value in state.configuration.items()]),
                       "Input conventions": mo.md(
                "Native graph weights and the scalar numerator enter once, with measure "
                r"$\prod_\ell d^Dk_\ell/(i\pi^{D/2})$. No implicit Euler-gamma or scale factors are added."
            )}
            if state.input_events:
                details["HEPKit diagram-generation events"] = table(mo, [
                    {"stage": e.stage, "completed": e.completed, "total": e.total}
                    for e in state.input_events
                ])
            provenance = getattr(prepared, "provenance", None)
            if provenance:
                details["Input identity"] = mo.md(
                    f"Live label **{provenance['actual_diagram_name']}**, audited Rust label "
                    f"**{provenance['audited_diagram_name']}**; stable native ID `{prepared.raw_diagram.id}`. "
                    "Only the cosmetic name differs; every physical serialized field is checked."
                )
            if state.numerator_view is not None:
                details["Native weighted scalar numerator"] = state.numerator_view
            return mo.vstack([
                mo.md(f"**{prepared.name}** · {prepared.diagram.loop_count} loop(s) · input preparation {state.preparation_seconds:.3f} s"),
                state.drawing if state.drawing is not None else mo.md("Native drawing has not been requested."),
                mo.accordion(details),
            ])

        return SimpleNamespace(phase_rows=phase_rows, timing_rows=timing_rows, generation_view=generation_view, input_view=input_view)

    generation_views = _definitions()
    return (generation_views,)


@app.cell(hide_code=True)
def _(formatting):
    def _definitions():
        from types import SimpleNamespace
        compact_count = formatting.compact_count
        epsilon_label = formatting.epsilon_label
        table = formatting.table
        uncertainty = formatting.uncertainty

        """Full native Laurent-vector, covariance and accepted-coverage views."""

        from html import escape
        import math

        def vector_rows(estimate):
            if estimate is None:
                return []
            if any(len(values) != len(estimate.orders) for values in (estimate.components, estimate.mean, estimate.standard_error)):
                raise ValueError("Native Laurent-vector shape mismatch")
            return [
                {"epsilon order": order, "component": component, "mean": mean,
                 "standard error": error,
                 "relative error": error / abs(mean) if mean != 0 else None}
                for order, component, mean, error in zip(
                    estimate.orders, estimate.components, estimate.mean, estimate.standard_error
                )
            ]

        def highest_order_rows(estimate):
            rows = vector_rows(estimate)
            highest = max((row["epsilon order"] for row in rows), default=None)
            return [row for row in rows if row["epsilon order"] == highest]


        def estimate_table(mo, estimate):
            """Format a copy for display; raw vectors/history/export retain native floats."""
            rows = [{"epsilon order": row["epsilon order"], "component": row["component"],
                     "estimate ± 1σ": uncertainty(row["mean"], row["standard error"]),
                     "relative error": row["relative error"]}
                    for row in vector_rows(estimate)]
            return table(mo, rows)

        def sector_rows(snapshot):
            discrete = snapshot.method == "havana_discrete_mc"
            rows = []
            for sector in snapshot.sectors:
                row = {"sector": sector.id, "dimension": sector.dimension,
                       "accepted points": sector.completed_points,
                       "planned points": "as allocated" if sector.planned_points is None else sector.planned_points,
                       "global batches seen" if discrete else "complete shifts": sector.complete_replicas,
                       "planned global batches" if discrete else "planned shifts": sector.planned_replicas,
                       "worker seconds": sector.worker_seconds}
                allocation = getattr(sector, "discrete_allocation", None)
                if allocation is not None:
                    row.update({"native selection probability": allocation.probability,
                                "global points per batch": allocation.points_per_batch})
                rows.append(row)
            return rows


        def coverage_summary(snapshot):
            if snapshot.method == "havana_discrete_mc":
                # Each sector carries the same global batch count, not an independent replica.
                first = next(iter(snapshot.sectors), None)
                return "Complete global batches", (first.complete_replicas if first else 0), (first.planned_replicas if first else 0)
            return "Complete sector shifts", sum(s.complete_replicas for s in snapshot.sectors), sum(s.planned_replicas for s in snapshot.sectors)

        def covariance_rows(estimate):
            if estimate is None:
                return []
            if len(estimate.components) != len(estimate.orders) or len(estimate.covariance_of_mean) != len(estimate.orders) ** 2:
                raise ValueError("Native covariance shape mismatch")
            labels = [f"{epsilon_label(order)} {component}" for order, component in zip(estimate.orders, estimate.components)]
            n = len(labels)
            return [{"component": labels[i], **dict(zip(labels, estimate.covariance_of_mean[i * n:(i + 1) * n]))} for i in range(n)]

        def history_plot(mo, history):
            """An SVG view of actual native means and one-standard-error intervals."""
            if not history:
                return mo.md("Waiting for a valid native estimate. History contains no provisional zeros.")
            chunks = []
            recorded_zero = []
            for component in dict.fromkeys(row["component"] for row in history):
                rows = [row for row in history if row["component"] == component]
                rows = [row for row in rows if math.isfinite(row["mean"]) and math.isfinite(row["standard error"])]
                if not rows:
                    continue
                base = rows[-1]["mean"]
                lo = min(row["mean"] - row["standard error"] for row in rows)
                hi = max(row["mean"] + row["standard error"] for row in rows)
                margin = (hi - lo) * 0.1 or max(abs(lo) * 0.01, 1e-12)
                lo, hi = lo - margin, hi + margin
                xmin, xmax = rows[0]["accepted points"], rows[-1]["accepted points"]
                x = lambda point: 76 + (point - xmin) / max(xmax - xmin, 1) * 490
                y = lambda value: 172 - (value - lo) / (hi - lo) * 135
                marks, path = [], []
                for row in rows:
                    xx, yy = x(row["accepted points"]), y(row["mean"])
                    path.append(f"{xx:.2f},{yy:.2f}")
                    marks.append(f'<path d="M {xx:.2f} {y(row["mean"] - row["standard error"]):.2f} V {y(row["mean"] + row["standard error"]):.2f}" stroke="#9276ce"/><circle cx="{xx:.2f}" cy="{yy:.2f}" r="3" fill="#6b46b0"><title>{escape(str(row))}</title></circle>')
                title = f"{epsilon_label(rows[-1]['epsilon order'])} {component}: mean ± 1 standard error"
                plot = mo.Html(f'<svg viewBox="0 0 620 220" role="img" aria-label="{escape(title)}" style="width:100%;color:var(--text-primary)"><text x="76" y="18" fill="currentColor" font-size="13">{escape(title)}</text><text x="76" y="30" fill="currentColor" font-size="10">Vertical offset from {base:.17g}</text><path d="M76 32 V172 H580" fill="none" stroke="currentColor" opacity=".3"/><text x="4" y="42" fill="currentColor" font-size="10">{hi - base:.3g}</text><text x="4" y="172" fill="currentColor" font-size="10">{lo - base:.3g}</text><polyline points="{" ".join(path)}" fill="none" stroke="#6b46b0" opacity=".6"/>{"".join(marks)}<text x="76" y="194" fill="currentColor" font-size="11">{compact_count(xmin)}</text><text x="566" y="194" text-anchor="end" fill="currentColor" font-size="11">{compact_count(xmax)}</text><text x="320" y="213" text-anchor="middle" fill="currentColor" font-size="11">Accepted points in this phase</text></svg>')
                if all(row["mean"] == 0 and row["standard error"] == 0 for row in rows):
                    recorded_zero.append(plot)
                else:
                    chunks.append(plot)
            if recorded_zero:
                chunks.append(mo.accordion({"Recorded zero-valued component histories": mo.vstack([
                    mo.md("Every displayed mean and standard error in these recorded histories is zero. This does not establish symbolic exactness; all components remain in the Laurent vector and covariance."),
                    *recorded_zero,
                ])}))
            return mo.vstack(chunks)

        def result_view(mo, state):
            snapshot = state.snapshot
            if snapshot is None:
                if state.kernels is not None:
                    return mo.callout("Kernels are ready. No integration session exists and zero points have been sampled. Select Integrate when ready.", kind="success")
                return mo.md("The Laurent vector will appear when native coverage permits an estimate.")
            estimate = snapshot.estimate
            coverage = sector_rows(snapshot)
            coverage_label, complete, planned = coverage_summary(snapshot)
            worker_seconds = snapshot.worker_seconds
            diagnostic = snapshot.evaluation_diagnostics
            details = {"Sector coverage": table(mo, coverage), "Full covariance of the mean": table(mo, covariance_rows(estimate), scientific_values=True)}
            status_rows = [{"field": "backend", "value": state.kernels.backend},
                           {"field": "method", "value": snapshot.method},
                           {"field": "stage", "value": snapshot.stage},
                           {"field": "uncertainty", "value": snapshot.uncertainty},
                           {"field": "stop_reason", "value": snapshot.stop_reason}]
            if snapshot.stop_detail is not None:
                status_rows.append({"field": "stop_detail", "value": snapshot.stop_detail})
            status_rows += [{"field": key, "value": str(value)} for key, value in state.configuration.items()]
            details["Run settings and native status"] = table(mo, status_rows)
            uncertainty_label = {"available": "Available", "exact": "Exact", "waiting_for_coverage": "Waiting for native coverage", "pilot_only": "Pilot only", "statistical_failure": "Statistical failure"}.get(snapshot.uncertainty, snapshot.uncertainty)
            stop_label = {None: "No native stop recorded", "planned_work_complete": "Allocation complete", "cancelled": "Cancelled", "target_reached": "Accuracy target reached", "work_limit": "Work limit", "time_limit": "Time limit", "numerical_failure": "Numerical failure"}.get(snapshot.stop_reason, snapshot.stop_reason)
            backend_label = {"native_o2": "Native O2", "portable_interpreted": "Portable interpreter"}.get(state.kernels.backend, state.kernels.backend)
            if diagnostic is not None:
                details["Native evaluation diagnostics"] = table(mo, [{name: getattr(diagnostic, name) for name in ("evaluations", "conditioning_checks", "rescues", "max_precision_bits", "failures", "weighted_checks", "additional_replays")}])
            return mo.vstack([
                mo.hstack([mo.stat(label="Accepted points", value=f"{compact_count(snapshot.completed_points)} / {compact_count(snapshot.planned_points)}"), mo.stat(label=coverage_label, value=f"{compact_count(complete)} / {compact_count(planned)}"), mo.stat(label="Allocation", value="Complete" if state.phase == "complete" else "In progress" if state.active else "Paused")], widths="equal"),
                mo.callout("Pilot training only. This action freezes production after training; pilot samples stay excluded. Pause retains the same native pilot owner.", kind="info") if snapshot.stage == "pilot" else mo.md(""),
                mo.md(f"**Active wall time:** {state.integration_wall_seconds:.2f} s · **Native worker time:** {worker_seconds:.2f} s. Active time includes refresh waits and excludes caller-cancelled intervals."),
                mo.md(f"**Uncertainty:** {uncertainty_label} · **Native stop:** {stop_label} · **Backend:** {backend_label}"),
                mo.callout(snapshot.uncertainty_detail, kind="warn") if snapshot.uncertainty_detail else mo.md(""),
                estimate_table(mo, estimate) if estimate is not None else mo.callout("The native session has no valid full-vector estimate yet. Missing uncertainty is not zero uncertainty.", kind="info"),
                mo.md("Allocation completion does not assert an accuracy target. All displayed uncertainties and covariance come from the native session."),
                history_plot(mo, state.history),
                mo.accordion(details),
            ])


        def previous_result_view(mo, state):
            snapshot = state.previous_snapshot
            if snapshot is None:
                return mo.md("")
            configuration = state.previous_configuration
            return mo.accordion({"Previous allocation · retained report": mo.vstack([
                mo.md(f"**{configuration['example']}** · {snapshot.method} · {snapshot.stage} · {state.previous_phase}. This is the saved prior allocation, not the current session."),
                mo.md(f"Accepted points: **{compact_count(snapshot.completed_points)}**. Its full report is available below."),
                estimate_table(mo, snapshot.estimate) if snapshot.estimate is not None else mo.md("That allocation had no valid native estimate."),
            ])})

        return SimpleNamespace(vector_rows=vector_rows, highest_order_rows=highest_order_rows, estimate_table=estimate_table, sector_rows=sector_rows, coverage_summary=coverage_summary, covariance_rows=covariance_rows, history_plot=history_plot, result_view=result_view, previous_result_view=previous_result_view)

    integration_views = _definitions()
    return (integration_views,)


@app.cell(hide_code=True)
def _(formatting):
    def _definitions():
        from types import SimpleNamespace
        panel = formatting.panel
        table = formatting.table
        epsilon_label = formatting.epsilon_label

        """Read-only native sector inspection; no JSON parsing or coefficient expansion."""

        from collections import Counter
        from html import escape


        def _monomials(chart):
            """Group native retained monomial factors, not inferred integrand limits."""
            from symbolica import E
            groups = {}
            record = getattr(chart, "pre_subtraction", None)
            if record is None:
                return []
            for term in record.terms:
                value = E("1")
                for parameter, power in zip(chart.coordinates.target_parameters, term.powers):
                    value *= parameter ** power.exponent
                groups[value] = groups.get(value, 0) + 1
            return list(groups.items())


        def overview(mo, generated, kernels=None):
            if generated is None:
                return mo.md("Generate first to inspect the decomposition.")
            charts = generated.metadata.charts
            statistics = getattr(kernels, "sector_statistics", ())
            chart_counts = Counter(chart.kernel_sector for chart in charts)
            indexed = {sector.index: sector for sector in generated.sectors}
            ranked = sorted(indexed, key=lambda index: (
                -statistics[index].exact_program_bytes if index < len(statistics) else 0, index))
            rows = []
            for index in ranked[:10]:
                sector = indexed[index]
                stats = statistics[index] if index < len(statistics) else None
                matching = [chart for chart in charts if chart.kernel_sector == index]
                representative = next((chart for chart in matching if chart.source_index == chart.representative), matching[0] if matching else None)
                groups = _monomials(representative) if representative is not None else []
                monomials = mo.vstack([mo.hstack([_formula(mo, value), mo.md(f"× {count} mapped term(s)")], justify="start")
                                      for value, count in groups[:3]]) if groups else mo.md("Not retained")
                if len(groups) > 3:
                    monomials = mo.vstack([monomials, mo.md(f"+ {len(groups)-3} more; inspect sector {index}")])
                rows.append({"Sector ID": index, "Coordinates": sector.dimension,
                             "Evaluator bytes": stats.exact_program_bytes if stats else "Not recorded",
                             "Native operations": _operation_total(stats) if stats else "Not recorded",
                             "Retained pre-subtraction monomial factors": monomials})
            sizes = [stat.exact_program_bytes for stat in statistics]
            return mo.vstack([
                mo.md(f"### {len(indexed)} numerical sectors · {len(charts)} source charts"),
                _static_table(mo, [{"Backend": getattr(kernels, "backend", "Not compiled"),
                                   "Evaluator storage": f"{sum(sizes):,} bytes",
                                   "Largest evaluator": f"{max(sizes, default=0):,} bytes",
                                   "Orders": str(generated.orders)}]),
                mo.md("**Ten largest shared evaluators**, ordered by serialized native program size. IDs are the saved kernel indices."),
                _static_table(mo, rows),
                mo.md("Monomials precede endpoint subtraction and symmetry multiplicity; the regular body may still vanish. Operation counts follow native Horner/CPE optimization. No numerical session is created by inspection."),
                mo.md(f"{chart_counts[None]} chart(s) have no numerical kernel at these orders; this alone does not distinguish exact, cancelled or truncated terms.") if chart_counts[None] else mo.md(""),
            ])


        def coefficient_index(generated, index, order):
            """Resolve a physical Laurent order from the native retained schema."""
            if not 0 <= index < len(generated.sectors):
                raise ValueError("Choose an existing generated sector")
            coefficients = generated.sectors[index].aliased_coefficients
            for position, coefficient in enumerate(coefficients):
                if coefficient.order == order:
                    return position
            available = ", ".join(str(value.order) for value in coefficients)
            raise ValueError(f"Sector {index} has no epsilon order {order}; available orders: {available}")


        def expression_viewer(generated, index, coefficient_index):
            """Open only the explicitly selected native coefficient's scoped pager."""
            from symbolica.community.spenso import TensorExpression
            native_sectors = generated.sectors
            if not 0 <= index < len(native_sectors):
                raise ValueError("Choose an existing generated sector")
            coefficients = native_sectors[index].aliased_coefficients
            if not 0 <= coefficient_index < len(coefficients):
                raise ValueError(f"Coefficient index must be between 0 and {len(coefficients)-1}")
            # Native alias materialization is an explicit inspection operation. The
            # native Pager bounds rendering and owns navigation/cache lifetimes.
            return TensorExpression(coefficients[coefficient_index].expression()).paged(page_size=25)


        def inspection(mo, generated, kernels, index, coefficient_index=0, viewer=None):
            if not generated.sectors:
                return mo.vstack([overview(mo, generated, kernels),
                                  mo.md("There are no numerical sectors at these orders. Bind the physical point and integrate to obtain the native exact contribution, if any.")])
            views = [overview(mo, generated, kernels),
                     detail(mo, generated, index, coefficient_index=coefficient_index, kernels=kernels)]
            if viewer is not None:
                coefficient = generated.sectors[index].aliased_coefficients[coefficient_index]
                views.append(panel(mo, f"Sector {index} integrand · {epsilon_label(coefficient.order)}", mo.vstack([
                    mo.md("The actual native Symbolica Laurent coefficient after endpoint subtraction, including the retained sector's symmetry multiplicity. The global coordinate-independent exact contribution is separate. HEPKit renders one scoped page at a time; use the viewer controls to explore the full expression."),
                    mo.as_html(viewer),
                ])))
            return mo.vstack(views)


        def _static_table(mo, rows):
            """Small selected-detail tables, with native rich math cells preserved."""
            if not rows:
                return mo.md("No retained entries.")
            columns = list(rows[0])

            def cell(value):
                if value is None:
                    return "—"
                if isinstance(value, (str, int, float, bool)):
                    return escape(str(value))
                return mo.as_html(value).text

            header = "".join(f"<th>{escape(str(key))}</th>" for key in columns)
            body = "".join("<tr>" + "".join(f"<td>{cell(row.get(key))}</td>" for key in columns) + "</tr>" for row in rows)
            return mo.Html("""<style>
                .fsd-inspection-table {width:100%;border-collapse:collapse}
                .fsd-inspection-table th,.fsd-inspection-table td {padding:.45rem .8rem;text-align:left;vertical-align:top}
                .fsd-inspection-table th {border-bottom:1px solid var(--gray-5,#e2e8f0)}
                .fsd-inspection-table .katex-display {margin:.25rem 0;text-align:left}
                </style><div style="overflow:auto"><table class="fsd-inspection-table"><thead><tr>""" + header + '</tr></thead><tbody>' + body + '</tbody></table></div>')


        def _geometry(mo, geometry):
            matrix = geometry.exponent_matrix
            rows = [{"source coordinate": i, **{f"target {j}": str(value) for j, value in enumerate(row)}} for i, row in enumerate(matrix)]
            return mo.vstack([
                _static_table(mo, [{"source dimension": geometry.source_dimension, "target dimension": geometry.dimension,
                            "fixed parameter": str(geometry.fixed_parameter), "determinant": str(geometry.determinant),
                            "Jacobian powers": str(geometry.jacobian_powers), "factor valuations": str(geometry.factor_valuations)}]),
                _static_table(mo, rows),
            ])


        def _preview(value):
            # Reuse Symbolica's bounded native printer; this does not expand aliases.
            text = str(value.formatted(max_terms=12, max_line_length=80, show_namespaces=True))
            return text if len(text) <= 2000 else text[:2000] + " … [display preview clipped]"


        def _operation_total(stats):
            operations = stats.operations
            return sum(getattr(operations, key) for key in ("additions", "multiplications", "inversions", "function_calls"))


        def _compact(value):
            return str(value.formatted(max_terms=12, max_line_length=80, show_namespaces=True))


        def _formula(mo, value):
            # Symbolica supplies the bounded LaTeX; no expression reconstruction or CAS.
            # Qualified identities remain in the exact-source metadata and downloads.
            formatted = value.formatted(max_terms=12, max_line_length=80, show_namespaces=False)
            latex = formatted._repr_latex_()
            if latex and len(latex) <= 4000:
                return mo.md(latex)
            return mo.Html('<pre style="white-space:pre-wrap;overflow:auto">' + escape(_preview(value)) + '</pre>')


        def _pre_subtraction(mo, chart, term_index):
            record = getattr(chart, "pre_subtraction", None)
            if record is None:
                return mo.md("Pre-subtraction factors were not retained by this artifact or wheel. They cannot be recovered from the expanded coefficients without new algebra.")
            terms = record.terms
            if not terms:
                return mo.md("This source chart had no nonzero mapped terms before subtraction.")
            if not 0 <= term_index < len(terms):
                raise ValueError(f"Mapped term index must be between 0 and {len(terms)-1}")
            term = terms[term_index]
            parameters = chart.coordinates.target_parameters
            if len(parameters) != len(term.powers):
                raise ValueError("Native pre-subtraction power dimension differs")
            powers = [{"coordinate": _formula(mo, parameter), "exponent b + c ε": _formula(mo, power.exponent),
                       "b": _formula(mo, power.constant), "c": _formula(mo, power.slope),
                       "Taylor coefficients required": power.subtraction_count}
                      for parameter, power in zip(parameters, term.powers)]
            source = [{"coordinate": _compact(parameter), "exact exponent": _preview(power.exponent)}
                      for parameter, power in zip(parameters, term.powers)]
            return mo.vstack([
                mo.md(f"**Mapped term {term_index} / {len(terms)-1}** · metadata v{record.version} · regulator {_compact(record.regulator)}"),
                mo.md("Each term has the form $P(\\epsilon)\\,\\prod_i t_i^{b_i+c_i\\epsilon}\\,R(t,\\epsilon)$. The native prefactor and powers below precede symmetry multiplicity, endpoint subtraction and Laurent expansion."),
                mo.md("**Native prefactor** $P(\\epsilon)$"),
                _formula(mo, term.prefactor),
                _static_table(mo, powers),
                mo.md(f"Regular body native Atom storage: **{term.regular_expression_bytes:,} bytes**. Its full body is not duplicated in this metadata. Native Taylor endpoint subtraction implements the corresponding plus-distribution continuation. Taylor counts are endpoint admission requirements, not a count of surviving poles or a floating-point conditioning estimate."),
                mo.accordion({"Exact native names and prefactor source": mo.vstack([
                    _static_table(mo, source),
                    mo.Html('<pre style="white-space:pre-wrap;overflow:auto">' + escape(_preview(term.prefactor)) + '</pre>'),
                    mo.download(str(term.prefactor.formatted(show_namespaces=True)).encode(), filename=f"chart-{chart.source_index}-term-{term_index}-prefactor.txt", label="Download exact prefactor"),
                ])}),
            ])


        def _statistics(mo, kernels, index):
            statistics = getattr(kernels, "sector_statistics", ())
            if index >= len(statistics):
                return mo.md("Compile with the updated bindings to retain evaluator size statistics.")
            stats = statistics[index]
            return mo.vstack([
                _static_table(mo, [{"backend": stats.backend, "arithmetic": stats.arithmetic,
                            "inputs": stats.inputs, "shared outputs": stats.outputs,
                            "exact program bytes": stats.exact_program_bytes,
                            "SymJIT application bytes": stats.symjit_ir_bytes}]),
                _static_table(mo, [{key: getattr(stats.operations, key) for key in ("additions", "multiplications", "inversions", "function_calls")}]),
                mo.md("Counts belong to the actual shared complete-vector evaluator after native Horner/CPE optimization and before backend lowering. Complex outputs split into real/imaginary components during numerical evaluation. No per-coefficient compiled cost is inferred from shared expressions."),
            ])


        def detail(mo, generated, index, coefficient_index=0, alias_page=0, chart_index=0, term_index=0, kernels=None):
            """Inspect one coefficient page and one chart after an explicit action."""
            native_sectors = generated.sectors
            if not 0 <= index < len(native_sectors):
                raise ValueError("Choose an existing generated sector")
            sector = native_sectors[index]
            coefficients = sector.aliased_coefficients
            if not 0 <= coefficient_index < len(coefficients):
                raise ValueError(f"Coefficient index must be between 0 and {len(coefficients)-1}")
            coefficient = coefficients[coefficient_index]
            definitions = coefficient.aliases
            page_count = max(1, (len(definitions) + 19) // 20)
            if not 0 <= alias_page < page_count:
                raise ValueError(f"Alias page must be between 0 and {page_count-1}")
            selected = definitions[alias_page * 20:(alias_page + 1) * 20]
            aliases = [{"alias": str(key), "native definition preview": _preview(value)} for key, value in selected]
            coefficient_view = mo.vstack([
                mo.md(f"**{epsilon_label(coefficient.order)}** · coefficient index {coefficient_index} · {coefficient.alias_count} stored definitions"),
                mo.Html('<pre style="white-space:pre-wrap;overflow:auto;max-height:18rem">' + escape(_preview(coefficient.root)) + '</pre>'),
                mo.md(f"Alias page **{alias_page} / {page_count-1}** · at most 20 definitions. Native formatting shows at most 12 terms and 2,000 characters per preview; the full expressions remain in the native owner."),
                _static_table(mo, aliases),
            ])
            matching = [chart for chart in generated.metadata.charts if chart.kernel_sector == sector.index]
            chart_view = mo.md("No source chart points to this numerical sector.")
            if matching:
                if not 0 <= chart_index < len(matching):
                    raise ValueError(f"Chart index must be between 0 and {len(matching)-1}")
                chart = matching[chart_index]
                coordinates = chart.coordinates
                if len(coordinates.source_parameters) != len(coordinates.images):
                    raise ValueError("Native coordinate-map shape mismatch")
                images = [{"source parameter": _formula(mo, source), "native image": _formula(mo, image)}
                          for source, image in zip(coordinates.source_parameters, coordinates.images)]
                chart_view = mo.vstack([
                    mo.md(f"**Source chart {chart.source_index}** · chart selection {chart_index} / {len(matching)-1}"),
                    _static_table(mo, [{"representative": chart.representative,
                                "representative permutation": str(chart.representative_permutation),
                                "source domain": coordinates.source_domain,
                                "gauge-fixed parameter": str(coordinates.projective_fixed_parameter)}]),
                    _static_table(mo, images),
                    mo.md("Positive real measure Jacobian:"),
                    _formula(mo, coordinates.measure_jacobian),
                    _pre_subtraction(mo, chart, term_index),
                    _geometry(mo, chart.geometry),
                ])
            return panel(mo, f"Sector {sector.index} · {sector.dimension} coordinates", mo.vstack([
                mo.md("Maps describe the density pullback **before endpoint subtraction**. Generated coefficients may be complex; compiled results can split real and imaginary components."),
                _static_table(mo, [{"native parameter IDs": ", ".join(_compact(value) for value in sector.parameters),
                            "conditioning basis": sector.conditioning_basis,
                            "cancellation degree": sector.cancellation_degree,
                            "conditioning rows": str(sector.cancellation_terms)}]),
                mo.accordion({"Actual evaluator size": _statistics(mo, kernels, index),
                              "Selected compact coefficient": coefficient_view,
                              "Selected source chart": chart_view,
                              "Sector geometry": _geometry(mo, sector.map)}),
            ]))

        return SimpleNamespace(_monomials=_monomials, overview=overview, coefficient_index=coefficient_index, expression_viewer=expression_viewer, inspection=inspection, _static_table=_static_table, _geometry=_geometry, _preview=_preview, _operation_total=_operation_total, _compact=_compact, _formula=_formula, _pre_subtraction=_pre_subtraction, _statistics=_statistics, detail=detail)

    sector_views = _definitions()
    return (sector_views,)


@app.cell(hide_code=True)
def _():
    def _definitions():
        from types import SimpleNamespace
        """Download native getter values without reconstructing estimates or expressions."""

        import json


        def _fields(owner, names):
            return {name: getattr(owner, name) for name in names.split()}


        def _symbol_name(symbol):
            # Symbolica's default str() hides namespaces; preserve parameter identity.
            return str(symbol.formatted(show_namespaces=True)) if hasattr(symbol, "formatted") else str(symbol)


        def run_report(state):
            """A presentation record; native kernel/checkpoint codecs remain separate."""
            result = {
                "schema": "fastsecdec-showcase-report-1",
                "configuration": state.configuration,
                "runtime_parameter_point": {_symbol_name(symbol): value for symbol, value in state.parameter_point.items()},
                "phase": state.phase,
                "active": state.active,
                "session_created": state.session is not None,
                "error": state.error,
                "checkpoint_warning": state.checkpoint_warning,
                "preparation_seconds": state.preparation_seconds,
                "completed_pilot_active_seconds": state.pilot_seconds,
                "integration_active_seconds": state.integration_wall_seconds,
                "time_semantics": "Caller active time includes refresh waits and excludes paused intervals; native worker time is separate.",
                "generation_events": [],
                "kernels": None,
                "generated": None,
                "snapshot": None,
                "history": state.history,
            }
            phases = "input parametrization domain geometry mapping symmetry subtraction laurent coefficient_expansion compilation total"
            for event in state.events:
                row = _fields(event, "stage completed total sectors kernels elapsed_seconds detail")
                row["timings"] = _fields(event.timings, " ".join(name + "_seconds" for name in phases.split()))
                coefficient = event.coefficient_expansion
                row["coefficient_expansion"] = None
                if coefficient is not None:
                    row["coefficient_expansion"] = _fields(coefficient, "sector stage requested_method effective_method attempt relative_width formal_pieces")
                    row["coefficient_expansion"]["requests"] = _fields(coefficient.requests, "source_bodies unique_requests cached_partials aliases interleaved_requests fallback_requests")
                result["generation_events"].append(row)
            if state.kernels is not None:
                result["kernels"] = _fields(state.kernels, "content_id backend orders components sector_count")
            if state.generated is not None:
                generated = state.generated
                result["generated"] = {"orders": generated.orders}
                if hasattr(generated, "metadata"):
                    domain = generated.metadata.domain
                    result["generated"]["domain"] = _fields(domain, "domain branch_policy")
                    result["generated"]["sectors"] = [_fields(s, "index dimension coefficient_count alias_counts conditioning_basis cancellation_degree") for s in generated.sectors]
                    result["generated"]["charts"] = [_fields(c, "source_index representative representative_permutation kernel_sector") for c in generated.metadata.charts]
            if state.snapshot is not None:
                snapshot = state.snapshot
                row = _fields(snapshot, "method stage completed_points planned_points complete_sectors worker_seconds uncertainty uncertainty_detail stop_reason stop_detail")
                row["sectors"] = []
                for sector in snapshot.sectors:
                    sector_row = _fields(sector, "id dimension completed_points planned_points complete_replicas planned_replicas worker_seconds")
                    allocation = getattr(sector, "discrete_allocation", None)
                    sector_row["discrete_allocation"] = None if allocation is None else _fields(allocation, "probability points_per_batch")
                    row["sectors"].append(sector_row)
                estimate = snapshot.estimate
                row["estimate"] = None if estimate is None else _fields(estimate, "orders components mean standard_error covariance_of_mean production_complete")
                diagnostics = snapshot.evaluation_diagnostics
                row["evaluation_diagnostics"] = None if diagnostics is None else _fields(diagnostics, "evaluations conditioning_checks rescues max_precision_bits failures weighted_checks additional_replays")
                result["snapshot"] = row
            return result


        def report_bytes(state):
            return (json.dumps(run_report(state), indent=2, allow_nan=False) + "\n").encode("utf-8")

        return SimpleNamespace(_fields=_fields, _symbol_name=_symbol_name, run_report=run_report, report_bytes=report_bytes)

    reports = _definitions()
    return (reports,)


@app.cell(hide_code=True)
def _(integration_views, reports):
    def _definitions():
        from types import SimpleNamespace
        highest_order_rows = integration_views.highest_order_rows
        report_bytes = reports.report_bytes

        """Explicit caller-owned actions; no work runs merely by creating this state."""

        from dataclasses import dataclass, field
        from time import perf_counter


        @dataclass
        class RunState:
            prepared: object = None
            generated: object = None
            generation_session: object = None
            generation_active: bool = False
            generation_display: object = None
            generation_last_display: float = -float("inf")
            kernels: object = None
            parameter_point: dict = field(default_factory=dict)
            session: object = None
            snapshot: object = None
            observation: object = None
            live_observation: object = None
            configuration: dict | None = None
            events: list = field(default_factory=list)
            input_events: list = field(default_factory=list)
            history: list = field(default_factory=list)
            phase: str = "draft"
            active: bool = False
            active_started: float | None = None
            active_seconds: float = 0.0
            preparation_seconds: float = 0.0
            pilot_seconds: float = 0.0
            message: str = "Submit inputs, then select Generate. Integration starts separately."
            error: str | None = None
            checkpoint_warning: str | None = None
            checkpoint_bytes: bytes | None = None
            previous_report_bytes: bytes | None = None
            previous_snapshot: object = None
            previous_configuration: dict | None = None
            previous_phase: str | None = None
            inspected_sector: object = None
            numerator_view: object = None
            drawing: object = None
            seen: dict = field(default_factory=lambda: {"generate": 0, "integrate": 0, "new": 0, "cancel": 0, "resume": 0, "adapt": 0, "freeze": 0, "inspect": 0, "numerator": 0, "tick": ""})

            @property
            def is_mc(self):
                return self.configuration is not None and self.configuration.get("method", "qmc") == "havana_discrete_mc"

            @property
            def pilot(self):
                return self.is_mc and self.session is not None and self.session.stage == "pilot"

            def save_checkpoint(self):
                if not getattr(self.session, "checkpoint_available", True):
                    self.checkpoint_bytes = None
                else:
                    self.checkpoint_bytes = self.session.checkpoint()

            @property
            def integration_wall_seconds(self):
                return self.active_seconds + (perf_counter() - self.active_started if self.active_started is not None else 0.0)

            def stop_clock(self):
                self.active_seconds = self.integration_wall_seconds
                self.active_started = None

            def observe_generation(self, event, display=None):
                self.events.append(event)
                now = perf_counter()
                if display is not None and (now - self.generation_last_display >= 1.0 or event.stage == "complete"):
                    display(event)
                    self.generation_last_display = now
                return True

            @property
            def work_active(self):
                return self.active or self.generation_active

            def start_generation(self, prepare, configuration, create_session, *, display=None):
                """Construct a retained native generation owner; sampling stays absent."""
                if self.work_active:
                    self.message = "Pause the active calculation before choosing another input."
                    return
                self.prepared = self.generated = self.kernels = self.session = self.snapshot = None
                self.generation_session = None
                self.observation = self.live_observation = None
                self.parameter_point.clear()
                self.previous_report_bytes = self.previous_snapshot = None
                self.previous_configuration = self.previous_phase = None
                self.preparation_seconds = 0.0
                self.configuration = dict(configuration)
                self.events.clear()
                self.history.clear()
                self.input_events.clear()
                self.inspected_sector = self.numerator_view = self.drawing = None
                self.checkpoint_bytes = None
                self.error = self.checkpoint_warning = None
                self.active_seconds = self.pilot_seconds = 0.0
                self.active_started = None
                self.phase = "preparing"
                self.message = "Preparing the selected native diagram. Pause is serviced at the next native unit boundary."
                self.generation_display = display
                self.generation_last_display = -float("inf")
                if display is not None:
                    display()
                started = perf_counter()
                try:
                    self.prepared = prepare()
                    self.preparation_seconds = perf_counter() - started
                    self.generation_session = create_session(self.prepared, configuration)
                    self.generation_active = not self.generation_session.complete
                    self.phase = "generating"
                    self.message = "Generating sectors and eager evaluators. Pause retains this native owner."
                    if self.generation_active:
                        self.advance_generation()
                    else:
                        self.generated = self.generation_session.generated
                        self.kernels = self.generation_session.kernels
                        self.phase = "ready"
                        self.message = "Native generation is already complete. No points sampled."
                except (Exception, KeyboardInterrupt) as error:
                    self.preparation_seconds = self.preparation_seconds or perf_counter() - started
                    self.generation_active = False
                    self.fail(error)

            def advance_generation(self, *, display=True):
                if not self.generation_active:
                    return
                try:
                    self.generation_session.step(max_units=1, observer=lambda event: self.observe_generation(
                        event, (lambda _: self.generation_display()) if display and self.generation_display is not None else None))
                    self.generated = self.generation_session.generated
                    self.kernels = self.generation_session.kernels
                    if self.generation_session.complete:
                        self.generation_active = False
                        self.phase = "ready"
                        self.message = "Sectors and eager evaluators ready. Inspect them, then integrate explicitly. No points sampled."
                except KeyboardInterrupt:
                    self.generation_active = False
                    self.phase = "generation_paused"
                    self.message = "Generation paused; Resume continues the retained native owner."
                except Exception as error:
                    self.generated = self.generation_session.generated
                    self.kernels = self.generation_session.kernels
                    self.generation_active = False
                    if self.generation_session.failed is None:
                        self.phase = "generation_paused"
                        self.error = f"{type(error).__name__}: {error}"
                        self.message = "Caller/presentation interruption; the native unit is retained. Resume retries the next unit."
                    else:
                        self.fail(error)

            def pause(self):
                if self.generation_active:
                    self.generation_active = False
                    self.phase = "generation_paused"
                    self.message = "Generation paused between native units. Resume continues the same owner."
                else:
                    self.cancel()

            def integrate(self, create_session, settings=None):
                """Start a fresh native session only from already prepared kernels."""
                if self.kernels is None:
                    self.message = "Generate and compile the input before integration."
                    return
                if self.session is not None:
                    self.message = "This allocation already exists. Resume it, or choose New integration after stopping to reuse these kernels with different settings."
                    return
                try:
                    configuration = dict(self.configuration)
                    if settings is not None:
                        allowed = {"method", "points", "shifts", "seed", "package_points", "rule", "periodization",
                                   "pilot_points", "pilot_batches", "points_per_batch", "batches", "bins",
                                   "minimum_probability_density", "maximum_sector_probability_ratio"}
                        if set(settings) - allowed:
                            raise ValueError("Integrate can change allocation settings only; Generate binds the physics input")
                        configuration.update(settings)
                    if configuration.get("method", "qmc") not in {"qmc", "havana_discrete_mc"}:
                        raise ValueError("Choose QMC or Havana discrete Monte Carlo")
                    inactive_keys = ({"points", "shifts", "package_points", "rule", "periodization"}
                                     if configuration.get("method", "qmc") == "havana_discrete_mc" else
                                     {"pilot_points", "pilot_batches", "points_per_batch", "batches", "bins",
                                      "minimum_probability_density", "maximum_sector_probability_ratio"})
                    for key in inactive_keys:
                        configuration.pop(key, None)
                    self.session = create_session(self.kernels, configuration)
                    self.configuration = configuration
                    self.snapshot = self.session.snapshot()
                    self.capture_observation()
                    self.active = not self.session.complete
                    self.active_started = perf_counter() if self.active else None
                    self.phase = ("pilot" if self.pilot else "integrating") if self.active else ("pilot_ready" if self.pilot else "complete")
                    self.error = None
                    self.message = ("Havana pilot. Production starts after this pilot; its samples stay excluded." if self.pilot else "Integrating with short caller work budgets; native observations refresh once per second.") if self.active else "The native allocation is exact and already complete."
                    if not self.active:
                        self.save_checkpoint()
                except (Exception, KeyboardInterrupt) as error:
                    self.fail(error)

            def new_integration(self):
                """Release only inactive sampling state; retain the generated native owners."""
                if self.active:
                    self.message = "Cancel the active allocation before choosing New integration."
                    return
                if self.kernels is None:
                    self.message = "Generate and compile an input first."
                    return
                if self.session is not None:
                    try:
                        self.previous_report_bytes = report_bytes(self)
                        self.previous_snapshot = self.snapshot
                        self.previous_configuration = dict(self.configuration)
                        self.previous_phase = self.phase
                    except (Exception, KeyboardInterrupt) as error:
                        self.fail(error)
                        return
                self.session = self.snapshot = None
                self.observation = self.live_observation = None
                self.history.clear()
                self.checkpoint_bytes = None
                self.error = self.checkpoint_warning = None
                self.active_seconds = self.pilot_seconds = 0.0
                self.active_started = None
                self.phase = "ready"
                self.message = "Same generated input and compiled kernels retained. Choose a method/allocation, then Integrate explicitly. The previous allocation report remains downloadable."

            def fail(self, error):
                failed_phase = self.phase
                self.stop_clock()
                self.active = False
                stage = getattr(error, "stage", failed_phase)
                if isinstance(error, KeyboardInterrupt):
                    self.error = None
                    self.phase = "interrupted"
                    self.message = f"Caller interruption during {failed_phase}. Completed owners and accepted coverage are retained."
                else:
                    self.error = f"{type(error).__name__} [{stage}]: {error}"
                    self.phase = "failed"
                    self.message = f"Stopped during {failed_phase}; no numerical value replaces this failure."
                if self.session is not None:
                    warnings = []
                    try:
                        self.snapshot = self.session.snapshot()
                        self.capture_observation()
                    except (Exception, KeyboardInterrupt) as secondary:
                        warnings.append(f"Snapshot capture failed: {type(secondary).__name__}: {secondary}")
                    try:
                        self.save_checkpoint()
                    except (Exception, KeyboardInterrupt) as secondary:
                        warnings.append(f"Checkpoint capture failed: {type(secondary).__name__}: {secondary}")
                    self.checkpoint_warning = " ".join(warnings) or None

            def cancel(self):
                if self.session is None or not self.active:
                    return
                self.stop_clock()
                self.active = False
                try:
                    self.save_checkpoint()
                    self.snapshot = self.session.snapshot()
                    self.capture_observation()
                    self.phase = "paused"
                    self.message = "Pilot paused in memory; Resume continues this native session. Downloadable checkpoints become available after freezing production." if self.pilot else "Cancelled by the caller between packages. Accepted coverage is saved."
                except (Exception, KeyboardInterrupt) as error:
                    self.fail(error)

            def resume(self):
                if self.phase == "generation_paused" and self.generation_session is not None:
                    self.generation_active = True
                    self.phase = "generating"
                    self.error = None
                    self.message = "Resuming native generation and eager compilation."
                    return
                if self.active or self.kernels is None:
                    return
                if self.pilot:
                    try:
                        self.snapshot = self.session.snapshot()
                        self.capture_observation()
                        self.active = not self.session.complete
                        self.active_started = perf_counter() if self.active else None
                        self.phase = "pilot" if self.active else "pilot_ready"
                        self.error = None
                        self.message = "Resumed the retained native pilot in memory." if self.active else "Pilot complete. Adapt another pilot or freeze production explicitly."
                    except (Exception, KeyboardInterrupt) as error:
                        self.fail(error)
                    return
                if self.checkpoint_bytes is None:
                    return
                if self.checkpoint_warning:
                    self.message = "Checkpoint capture failed. The native session is retained; an older checkpoint will not be resumed silently."
                    return
                try:
                    restore = self.kernels.restore_mc if self.is_mc else self.kernels.restore
                    self.session = restore(self.checkpoint_bytes)
                    self.snapshot = self.session.snapshot()
                    self.capture_observation()
                    self.active = not self.session.complete
                    self.active_started = perf_counter() if self.active else None
                    self.error = None
                    self.phase = "integrating" if self.active else "complete"
                    self.message = "Resumed from the native checkpoint." if self.active else "The saved allocation is already complete."
                except (Exception, KeyboardInterrupt) as error:
                    self.fail(error)

            def pilot_action(self, freeze=False):
                """Explicit native adaptation/freeze; pilot estimates never enter production."""
                if self.active or not self.pilot or not self.session.complete:
                    self.message = "Complete and stop a Havana pilot before adapting or freezing production."
                    return
                try:
                    elapsed = self.integration_wall_seconds
                    if freeze:
                        self.snapshot = self.session.freeze_production(
                            points_per_batch=self.configuration["points_per_batch"], batches=self.configuration["batches"],
                        )
                    else:
                        self.snapshot = self.session.adapt_pilot()
                    self.capture_observation()
                    self.pilot_seconds += elapsed
                    self.history.clear()
                    self.checkpoint_bytes = None
                    self.active_seconds = 0.0
                    self.active = not self.session.complete
                    self.active_started = perf_counter() if self.active else None
                    self.phase = "pilot" if self.pilot else "integrating"
                    self.error = self.checkpoint_warning = None
                    self.message = "Another native pilot epoch started." if self.pilot else "Production grids frozen; pilot estimates discarded. Sampling in short caller work budgets."
                    if not self.active:
                        self.phase = "pilot_ready" if self.pilot else "complete"
                        self.save_checkpoint()
                except (Exception, KeyboardInterrupt) as error:
                    self.fail(error)

            def capture_observation(self):
                # Read-only native views; these never enter checkpoints or convergence.
                self.observation = self.session.observation() if hasattr(self.session, "observation") else None
                self.live_observation = self.session.live_observation() if hasattr(self.session, "live_observation") else None

            def record_estimate(self):
                if self.snapshot.estimate is not None:
                    for row in highest_order_rows(self.snapshot.estimate):
                        self.history.append({"accepted points": self.snapshot.completed_points, **row})

            def advance(self, step, *, observe=True):
                """Only an explicit Integrate or Resume action can arm these steps."""
                if not self.active:
                    return
                try:
                    self.snapshot = step(self.session, self.configuration.get("method", "qmc"))
                    if observe or self.session.complete:
                        self.capture_observation()
                        self.record_estimate()
                    if self.session.complete:
                        self.stop_clock()
                        self.active = False
                        self.phase = "pilot_ready" if self.pilot else "complete"
                        self.save_checkpoint()
                        self.message = "Pilot complete. Adapt another pilot or Freeze production explicitly; pilot values are not production estimates." if self.pilot else "Planned allocation complete. Check the accuracy target separately."
                except (Exception, KeyboardInterrupt) as error:
                    self.fail(error)

        return RunState

    RunState = _definitions()
    return (RunState,)


@app.cell(hide_code=True)
def _(RunState, formatting, generation_views, gghh_inputs, integration_views, reports, science, sector_views):
    def _definitions():
        from types import SimpleNamespace
        panel = formatting.panel
        generation = generation_views
        integration = integration_views
        sectors = sector_views
        catalogue = gghh_inputs.catalogue
        prepare = gghh_inputs.prepare
        report_bytes = reports.report_bytes

        """Notebook presentation and explicit caller actions, independent of numerics."""
        from time import perf_counter
        from html import escape


        class Study:
            def __init__(self, kind="gghh"):
                if kind != "gghh":
                    raise ValueError("Unknown notebook study")
                self.kind = kind
                self.catalogue = None
                self.run = RunState()
                self.seen = {key: 0 for key in ("build", "generate", "inspect", "qmc", "havana", "pause", "resume", "export")}
                self.last_tick = None
                self.inspection = None
                self.inspection_request = None
                self.expression_viewer = None
                self.prepared_panel = None
                self.prepared_owner = None
                self.numerator_viewers = {}
                self.numerator_content = {}
                self.work_budget_seconds = 0.05
                self.work_unit_limit = 2048
                self.observation_interval = 1.0
                self.last_publication = -float("inf")
                self.revision = 0
                self.view_revisions = {name: 0 for name in ("prepared", "inspection", "artifact")}
                self.pending_views = set()
                self.prepared_downloads = {}

            def _repr_html_(self):
                title = "gg → HH · native diagram study" if self.kind == "gghh" else "FastSecDec · native integral study"
                return f'<div style="padding:1rem;border:1px solid #7583a0;border-radius:12px"><b>{title}</b><br>Single caller · eager Symbolica evaluator · explicit scientific actions</div>'

            def close_inspection(self):
                if self.expression_viewer is not None:
                    self.expression_viewer.close()
                    self.expression_viewer = None
                self.inspection = None

            def close_numerators(self):
                for viewer in self.numerator_viewers.values():
                    viewer.close()
                self.numerator_viewers.clear()
                self.numerator_content.clear()
                self.prepared_owner = self.prepared_panel = None

            def invalidate_views(self, *names):
                for name in names:
                    self.view_revisions[name] += 1
                    self.pending_views.add(name)

            def take_view_updates(self):
                updates = {name: self.view_revisions[name] for name in self.pending_views}
                self.pending_views.clear()
                return updates

            def build(self, count, mo):
                if count != self.seen["build"]:
                    self.seen["build"] = count
                    if self.run.work_active:
                        self.run.message = "Pause the calculation before rebuilding diagrams."
                    else:
                        def progress(event):
                            mo.output.replace(panel(mo, "Building native diagram catalogue", mo.vstack([
                                mo.md(f"**{event.stage}** · {event.completed} / {event.total if event.total is not None else '…'}"),
                                mo.md("One and two loops · QED² · Higgs, gluon, top · initial and final symmetries"),
                            ])))
                            return True
                        self.catalogue = catalogue(progress=progress)
                return self.catalogue

            def choices(self):
                if self.catalogue is None:
                    return {"Build diagrams first": None}
                return {f"{d.loop_count} loop · {d.name} · {i}": d.id
                        for i, d in enumerate(self.catalogue.diagrams)}

            def default_choice(self):
                options = self.choices()
                if self.catalogue is None:
                    return next(iter(options))
                identity = self.catalogue.default_diagram.id
                return next(label for label, value in options.items() if value == identity)

            def diagram_view(self, mo, identity):
                if self.kind != "gghh" or self.catalogue is None or identity is None:
                    return mo.md("Build the native catalogue, then choose a diagram." if self.kind == "gghh" else "Generate prepares the selected scalar example.")
                diagram = self.catalogue.selected(identity)
                return mo.vstack([
                    mo.md(f"**{diagram.name}** · {diagram.loop_count} loop(s) · native ID `{diagram.id}`"),
                    mo.as_html(diagram),
                ])

            def dispatch(self, actions, tick, selection, settings, mo):
                changed = {key for key, value in actions.items() if value != self.seen[key]}
                self.seen.update(actions)
                tick_changed = tick is not None and tick != self.last_tick
                self.last_tick = tick
                run = self.run
                previous_phase = run.phase
                was_generating = run.generation_active
                if changed.intersection({"generate", "qmc", "havana", "resume"}):
                    self.prepared_downloads.clear()
                if "pause" in changed:
                    run.pause()
                elif "resume" in changed:
                    run.resume()
                elif "generate" in changed:
                    if run.work_active:
                        run.message = "Pause the active calculation before choosing another input."
                        return self.publish()
                    self.inspection_request = None
                    if self.catalogue is None or selection is None:
                        run.message = "Build the diagram catalogue first."
                    else:
                        configuration = {"example": "gghh", "diagram_id": selection, "max_order": 0}
                        run.start_generation(lambda: prepare(selected=selection, source=self.catalogue), configuration, science.generation,
                                             display=lambda: mo.output.replace(self.monitor(mo)))
                    self.invalidate_views("prepared", "inspection", "artifact")
                elif "qmc" in changed or "havana" in changed:
                    if run.kernels is None:
                        run.message = "Generate the selected input before starting integration."
                    elif "qmc" in changed and int(settings["points"]) < 1024:
                        run.message = "The selected native Kuo-33002 rule requires at least 1024 points per shift. Increase Points, then select Integrate QMC."
                    elif run.work_active:
                        run.message = "Pause the active calculation before starting another integration."
                    else:
                        if run.session is not None:
                            run.new_integration()
                            if run.session is not None:
                                # A failed report/checkpoint handoff retains the old
                                # owner and point; do not bind a new point around it.
                                return self.publish()
                        try:
                            run.kernels, point = science.bind(run.generated, run.kernels, run.prepared, settings)
                            self.invalidate_views("artifact")
                            run.parameter_point = point
                            if self.kind == "gghh":
                                run.configuration["physical_point"] = {
                                    key: float(settings[key]) for key in ("sqrt_s", "higgs_mass", "top_mass", "cos_theta")
                                }
                        except (Exception, KeyboardInterrupt) as error:
                            run.fail(error)
                            return self.publish()
                        method = "havana_discrete_mc" if "havana" in changed else "qmc"
                        common = {"method": method, "seed": int(settings["seed"])}
                        allocation = ({"pilot_points": 64, "pilot_batches": 2,
                                       "points_per_batch": int(settings["points"]), "batches": int(settings["replicas"]),
                                       "bins": 16, "minimum_probability_density": 0.01,
                                       "maximum_sector_probability_ratio": 100.0}
                                      if method == "havana_discrete_mc" else
                                      {"points": int(settings["points"]), "shifts": int(settings["replicas"]),
                                       "package_points": 4096, "rule": "kuo_33002", "periodization": "korobov3"})
                        run.integrate(science.integration, {**common, **allocation})
                elif "inspect" in changed:
                    if run.generated is None or run.kernels is None:
                        run.message = "Finish generation before inspecting the numerical sectors."
                    else:
                        # Widgets must be created AND displayed by the dedicated cell:
                        # Marimo disposes comms belonging to a cell whenever it reruns.
                        self.inspection_request = (int(settings["sector"]), int(settings.get("epsilon_order", 0)))
                        self.invalidate_views("inspection")
                elif "export" in changed:
                    self.prepare_downloads()
                elif tick_changed:
                    self.advance_budget(science)
                if was_generating and not run.generation_active:
                    self.invalidate_views("artifact")
                if changed or previous_phase != run.phase or perf_counter() - self.last_publication >= self.observation_interval:
                    if run.active:
                        try:
                            run.capture_observation()
                            run.record_estimate()
                        except (Exception, KeyboardInterrupt) as error:
                            run.fail(error)
                    return self.publish()
                return None

            def publish(self):
                self.last_publication = perf_counter()
                self.revision += 1
                return self.revision

            def advance_budget(self, science):
                """Spend a short caller budget, then return control to the frontend."""
                deadline = perf_counter() + self.work_budget_seconds
                for _ in range(self.work_unit_limit):
                    if self.run.generation_active:
                        self.run.advance_generation()
                    elif self.run.active:
                        self.run.advance(science.step, observe=False)
                        # The explicit Havana action authorizes pilot then production.
                        # Native freeze discards the pilot estimator before continuing.
                        if self.run.phase == "pilot_ready":
                            self.run.pilot_action(freeze=True)
                    else:
                        break
                    if not self.run.work_active or perf_counter() >= deadline:
                        break

            def monitor(self, mo):
                run = self.run
                content = [mo.callout(run.error or run.message, kind="danger" if run.error else "info")]
                if run.generation_session is not None or run.phase == "preparing":
                    content.append(panel(mo, "Generation · retained native units", generation.generation_view(mo, run), expanded=run.generation_active or run.phase == "generation_paused"))
                if run.snapshot is not None:
                    content.append(panel(mo, "Integration · accepted native work", integration.result_view(mo, run)))
                    if run.live_observation is not None:
                        content.append(mo.as_html(run.live_observation))
                if run.checkpoint_warning:
                    content.append(mo.callout(run.checkpoint_warning, kind="warn"))
                return mo.vstack(content)

            def prepared_view(self, mo):
                # This runs only in the stable prepared-input cell. If that cell is
                # intentionally rerun, create new widgets instead of stale cached comms.
                self.close_numerators()
                prepared = self.run.prepared
                if prepared is None:
                    return mo.md("")
                raw = getattr(prepared, "raw_numerator", None)
                simplified = getattr(prepared, "simplified_numerator", None)
                items = {}
                if raw is not None:
                    self.numerator_viewers["raw"] = raw.paged(page_size=25)
                    items["Raw numerator · native tensor expression"] = mo.as_html(self.numerator_viewers["raw"])
                if simplified is not None:
                    from symbolica.community.spenso import TensorExpression
                    self.numerator_viewers["simplified"] = TensorExpression(simplified).paged(page_size=25)
                    items["Simplified projected numerator · native Symbolica expression"] = mo.as_html(self.numerator_viewers["simplified"])
                self.prepared_owner = prepared
                # Marimo's accordion unmounts closed content. Native Pager responds by
                # disposing its final widget view, so cached markup cannot reopen it.
                # HTML details only hides descendants: both widget DOM trees stay
                # mounted, and their native owners/UI wrappers remain retained here.
                self.numerator_content = items
                panels = [mo.Html(
                    '<details style="border:1px solid var(--gray-5);border-radius:10px;padding:0.8rem">'
                    '<summary style="cursor:pointer;font-weight:600">' + escape(title) + '</summary>'
                    '<div style="padding-top:0.8rem">' + content.text + '</div></details>'
                ) for title, content in items.items()]
                content = panels or [mo.as_html(prepared.diagram)]
                self.prepared_panel = mo.vstack([mo.md(f"**Prepared input:** {prepared.name}"), *content])
                return self.prepared_panel

            def inspection_view(self, mo):
                """Create native pagers only in their stable inspection display cell."""
                self.close_inspection()
                if self.inspection_request is None:
                    return mo.md("Select Inspect after generation to open the evaluator overview and sector detail.")
                try:
                    index, order = self.inspection_request
                    coefficient_index = 0
                    if self.run.generated.sectors:
                        coefficient_index = sectors.coefficient_index(self.run.generated, index, order)
                        self.expression_viewer = sectors.expression_viewer(self.run.generated, index, coefficient_index)
                    self.inspection = sectors.inspection(mo, self.run.generated, self.run.kernels, index,
                        coefficient_index=coefficient_index, viewer=self.expression_viewer)
                except Exception as error:
                    self.close_inspection()
                    self.inspection = mo.callout(f"Inspection failed: {error}", kind="danger")
                return self.inspection

            def prepare_downloads(self):
                """Explicit native serialization on the notebook caller, never a download worker."""
                if self.run.work_active:
                    self.run.message = "Pause the calculation before preparing downloads."
                    return
                if self.run.kernels is None:
                    self.run.message = "Finish generation before preparing downloads."
                    return
                try:
                    artifact = self.run.kernels.to_bytes()
                    report = report_bytes(self.run)
                    self.prepared_downloads = {"artifact": artifact, "report": report}
                    self.run.message = "Downloads are ready. Their bytes describe this retained artifact and current allocation."
                except (Exception, KeyboardInterrupt) as error:
                    self.run.message = f"Could not prepare downloads: {type(error).__name__}: {error}"

            def result_view(self, mo):
                run = self.run
                native = mo.as_html(run.observation or run.snapshot) if run.snapshot is not None else mo.md("No integration has run.")
                previous = integration.previous_result_view(mo, run)
                downloads = []
                if self.prepared_downloads and not run.work_active:
                    downloads.append(mo.download(self.prepared_downloads["artifact"], filename="fastsecdec-kernels.dat", label="Evaluator artifact"))
                    downloads.append(mo.download(self.prepared_downloads["report"], filename="fastsecdec-report.json", label="Full run report"))
                if run.previous_report_bytes is not None:
                    downloads.append(mo.download(run.previous_report_bytes, filename="fastsecdec-previous-report.json", label="Previous run report"))
                if run.checkpoint_bytes is not None and not run.work_active:
                    downloads.append(mo.download(run.checkpoint_bytes, filename="fastsecdec-checkpoint.json", label="Accepted checkpoint"))
                point = mo.accordion({"Bound evaluator parameter point": sectors._static_table(mo, [
                    {"Native symbol": sectors._formula(mo, symbol), "Value": value}
                    for symbol, value in run.parameter_point.items()
                ])}) if run.parameter_point else mo.md("")
                return mo.vstack([native, point, previous, mo.hstack(downloads, justify="start") if downloads else mo.md("")])

        return Study

    Study = _definitions()
    return (Study,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # gg → HH · complete, self-contained notebook

    This file contains its own input preparation, workflow and presentation code.
    It needs only installed packages (`marimo` and Symbolica with HEPKit), with no
    sibling Python modules, parameter cards, DOT files or fixture folders.
    The folded definition cells expose every native HEPKit call used below.

    All one- and two-loop Standard Model diagrams with $QED=2$, restricted to Higgs, gluon and top particles. Initial and final states are symmetrized. The first one-loop diagram is selected by default. The initial integration point uses $\sqrt{s}=300$ GeV, $m_H=125$ GeV, $m_t=172.5$ GeV and $\cos\theta=4/5$, with $(+,+)$ gluon helicities and color projection $\delta_{ab}$. Each calculation is one diagram contribution.

    **Build diagrams → Generate → Inspect → Integrate QMC → Integrate Havana.**
    Every calculation starts with a button. Pause retains the native owner;
    Resume continues it. Changing a selection starts no calculation.
    One caller, eager Symbolica arithmetic, no SymJIT — the same workflow on
    native Python and Pyodide.
    """)
    return


@app.cell
def _(Study):
    study = Study("gghh")
    study
    return (study,)


@app.cell(hide_code=True)
def _(mo, study):
    build = mo.ui.button(value=0, on_click=lambda n: n + 1,
                         label="Build diagrams", kind="success")
    build if study.kind == "gghh" else mo.md("Choose one of the native scalar examples below.")
    return (build,)


@app.cell
def _(build, mo, study):
    catalogue = study.build(build.value, mo) if study.kind == "gghh" else None
    mo.md(f"**{len(catalogue.diagrams)} diagrams** · " + " · ".join(
        f"{sum(d.loop_count == loops for d in catalogue.diagrams)} at {loops} loop(s)"
        for loops in (1, 2))) if catalogue is not None else mo.md("")
    return (catalogue,)


@app.cell(hide_code=True)
def _(catalogue, mo, study):
    catalogue
    diagram = mo.ui.dropdown(study.choices(), value=study.default_choice(),
                             allow_select_none=False, label="Native diagram / input")
    diagram
    return (diagram,)


@app.cell
def _(diagram, mo, study):
    study.diagram_view(mo, diagram.value)
    return


@app.cell(hide_code=True)
def _(mo, study):
    generate = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Generate sectors", kind="success")
    inspect = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Inspect", kind="neutral")
    qmc = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Integrate QMC", kind="success")
    havana = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Integrate Havana", kind="success")
    pause = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Pause", kind="warn")
    resume = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Resume")
    export = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Prepare downloads")
    points = mo.ui.dropdown({str(n): n for n in (64, 256, 1024, 4096, 16384)}, value="1024", label="Points per shift / global MC batch")
    replicas = mo.ui.number(start=2, stop=64, value=4, step=1, label="Shifts / global batches")
    seed = mo.ui.number(start=0, stop=2**32-1, value=20261007, step=1, label="Seed")
    sector = mo.ui.number(start=0, value=0, step=1, label="Sector ID")
    epsilon_order = mo.ui.number(value=0, step=1, label="ε order")
    sqrt_s = mo.ui.number(start=251, value=300, label="sqrt(s) [GeV]")
    higgs_mass = mo.ui.number(start=1, value=125, label="mH [GeV]")
    top_mass = mo.ui.number(start=1, value=172.5, label="mt = Yukawa mass [GeV]")
    cos_theta = mo.ui.number(start=-0.999, stop=0.999, value=0.8, step=0.1, label="cos(theta)")
    get_active, set_active = mo.state(False)
    get_revision, set_revision = mo.state(0)
    get_prepared_revision, set_prepared_revision = mo.state(0)
    get_inspection_revision, set_inspection_revision = mo.state(0)
    get_artifact_revision, set_artifact_revision = mo.state(0)
    mo.vstack([
        mo.hstack([generate, inspect, qmc, havana], justify="start", wrap=True),
        mo.hstack([pause, resume, sector, epsilon_order], justify="start", wrap=True),
        mo.accordion({"Optional exports": mo.vstack([export, mo.md("Prepare portable evaluator and run-report bytes on the notebook caller, then use the download links below.")])}),
        mo.accordion({"Runtime physical point": mo.vstack([
            mo.hstack([sqrt_s, higgs_mass, top_mass, cos_theta], justify="start", wrap=True),
            mo.md("Bound only at Integrate. Native model leaves and momentum Gram products remain evaluator parameters; changing this point reuses generated sectors."),
        ])}) if study.kind == "gghh" else mo.md(""),
        mo.accordion({"Integration allocation": mo.vstack([
            mo.hstack([points, replicas, seed], justify="start", wrap=True),
            mo.md("QMC uses Kuo-33002 (at least 1024 points), Korobov-3 and independent shifted lattices. Havana trains two 64-point pilot batches, then freezes production; pilot samples are excluded. A complete allocation is not an accuracy guarantee."),
        ])}),
    ])
    return cos_theta, epsilon_order, export, generate, get_active, get_artifact_revision, get_inspection_revision, get_prepared_revision, get_revision, havana, higgs_mass, inspect, pause, points, qmc, replicas, resume, sector, seed, set_active, set_artifact_revision, set_inspection_revision, set_prepared_revision, set_revision, sqrt_s, top_mass


@app.cell(hide_code=True)
def _(get_active, mo):
    refresh = mo.ui.refresh(options=["125ms", "250ms", "1s"], default_interval="125ms", label="Caller steps") if get_active() else None
    refresh
    return (refresh,)


@app.cell(hide_code=True)
def _(cos_theta, diagram, epsilon_order, export, generate, havana, higgs_mass, inspect, mo, pause, points, qmc, refresh, replicas, resume, sector, seed, set_active, set_artifact_revision, set_inspection_revision, set_prepared_revision, set_revision, sqrt_s, study, top_mass):
    _before = study.run.work_active
    _revision = study.dispatch(
        {"generate": generate.value, "inspect": inspect.value, "qmc": qmc.value,
         "havana": havana.value, "pause": pause.value, "resume": resume.value, "export": export.value},
        refresh.value if refresh is not None else None,
        diagram.value, {"points": points.value, "replicas": replicas.value,
                        "seed": seed.value, "sector": sector.value, "epsilon_order": epsilon_order.value,
                        "sqrt_s": sqrt_s.value, "higgs_mass": higgs_mass.value,
                        "top_mass": top_mass.value, "cos_theta": cos_theta.value}, mo,
    )
    if study.run.work_active != _before:
        set_active(study.run.work_active)
    if _revision is not None:
        set_revision(_revision)
    _views = study.take_view_updates()
    if "prepared" in _views:
        set_prepared_revision(_views["prepared"])
    if "inspection" in _views:
        set_inspection_revision(_views["inspection"])
    if "artifact" in _views:
        set_artifact_revision(_views["artifact"])
    return


@app.cell(hide_code=True)
def _(get_revision, mo, study):
    get_revision()
    study.monitor(mo)
    return


@app.cell(hide_code=True)
def _(get_prepared_revision, mo, study):
    get_prepared_revision()
    study.prepared_view(mo)
    return


@app.cell
def _(get_artifact_revision, study):
    get_artifact_revision()
    generated = study.run.generated
    generated
    return (generated,)


@app.cell
def _(get_artifact_revision, study):
    get_artifact_revision()
    artifact = study.run.kernels
    artifact
    return (artifact,)


@app.cell(hide_code=True)
def _(get_inspection_revision, mo, study):
    get_inspection_revision()
    study.inspection_view(mo)
    return


@app.cell
def _(get_revision, mo, study):
    get_revision()
    study.result_view(mo)
    return


@app.cell(hide_code=True)
def _(get_artifact_revision, mo):
    from symbolica import get_citations
    get_artifact_revision()
    citations = get_citations()
    mo.accordion({"Native citations": mo.vstack([
        *[mo.as_html(citation) for citation in citations],
        mo.download("\n\n".join(c.to_bibtex() for c in citations).encode(), filename="fastsecdec-references.bib", label="Download BibTeX"),
    ])})
    return


if __name__ == "__main__":
    app.run()
