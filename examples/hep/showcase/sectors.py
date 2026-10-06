"""Read-only native sector inspection; no JSON parsing or coefficient expansion."""

from collections import Counter
from html import escape
from .presentation import table, epsilon_label


def overview(mo, generated, kernels=None):
    if generated is None:
        return mo.md("Generate first to inspect the decomposition.")
    if not hasattr(generated, "sectors") or not hasattr(generated, "metadata"):
        return mo.callout("This wheel lacks the native sector-inspection API. Install the updated community wheel.", kind="warn")
    charts = generated.metadata.charts
    statistics = getattr(kernels, "sector_statistics", ())
    chart_counts = Counter(chart.kernel_sector for chart in charts)
    rows = []
    for sector in generated.sectors:
        stats = statistics[sector.index] if sector.index < len(statistics) else None
        rows.append({
            "sector": sector.index, "dimension": sector.dimension,
            "generated orders": ", ".join(epsilon_label(order) for order in generated.orders),
            "source charts": chart_counts[sector.index],
            "shared evaluator operations": _operation_total(stats) if stats else None,
            "exact program bytes": stats.exact_program_bytes if stats else None,
            "SymJIT application bytes": stats.symjit_ir_bytes if stats else None,
            "conditioning basis": sector.conditioning_basis.replace("_", " "),
        })
    no_kernel = chart_counts[None]
    domain = generated.metadata.domain
    certificates = [{"term": factor.term_index, "factor": factor.factor_index,
                     "native certificate": factor.certificate.replace("_", " ")}
                    for factor in domain.factors]
    return mo.vstack([
        mo.md(f"**{len(rows)} numerical sectors** · {len(charts)} retained charts · branch: {domain.branch_policy.replace('_', ' ')}"),
        mo.md(f"Native domain: **{domain.domain.replace('_', ' ')}** · caller assertion: **{'yes' if domain.caller_asserted else 'no'}** · admission relies on assertion: **{'yes' if domain.relies_on_assertion else 'no'}**"),
        mo.accordion({f"Domain certificates · {len(certificates)} factors": table(mo, certificates)}),
        table(mo, rows),
        mo.md(f"{no_kernel} chart(s) retain no numerical kernel at the requested orders. This may mean exact, cancelled or truncated contributions; the native metadata does not classify them further.") if no_kernel else mo.md(""),
        mo.md("Operations describe the complete shared Symbolica evaluator before real/complex SymJIT lowering and optimization. SymJIT application bytes measure its compressed serialized application, not machine code; portable execution has no SymJIT application. Older wheels may lack these statistics."),
    ])


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
    return str(value.formatted(max_terms=12, max_line_length=80, show_namespaces=False))


def _formula(mo, value):
    # Symbolica supplies the bounded LaTeX; no expression reconstruction or CAS.
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
    source = [{"coordinate": str(parameter), "exact exponent": _preview(power.exponent)}
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
            mo.download(lambda: str(term.prefactor).encode(), filename=f"chart-{chart.source_index}-term-{term_index}-prefactor.txt", label="Download exact prefactor"),
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
        mo.md("Counts belong to the actual shared complete-vector evaluator before SymJIT lowering/optimization. Complex outputs split into real/imaginary components during numerical evaluation. No per-coefficient compiled cost is inferred from shared expressions."),
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
    return mo.accordion({f"Sector {sector.index} · {sector.dimension} dimensions": mo.vstack([
        mo.md("Maps describe the density pullback **before endpoint subtraction**. Generated coefficients may be complex; compiled results can split real and imaginary components."),
        _static_table(mo, [{"parameters": ", ".join(str(value) for value in sector.parameters),
                    "conditioning basis": sector.conditioning_basis,
                    "cancellation degree": sector.cancellation_degree,
                    "conditioning rows": str(sector.cancellation_terms)}]),
        mo.accordion({"Actual evaluator size": _statistics(mo, kernels, index),
                      "Selected compact coefficient": coefficient_view,
                      "Selected source chart": chart_view,
                      "Sector geometry": _geometry(mo, sector.map)}),
    ])})
