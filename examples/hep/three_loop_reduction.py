import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="Three-loop massive vacuum reduction",
)

with app.setup(hide_code=True):
    from time import perf_counter
    from typing import Sequence, TypedDict, Union

    from symbolica import Expression
    from symbolica.community.hepkit.rustred import CandidateArtifact

    import marimo as mo
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    class IntegralKey(TypedDict):
        values: list[int]
        symbolic: list[bool]

    class FixedPower(TypedDict):
        axis: int
        value: int

    class RuleCase(TypedDict, total=False):
        fixed: list[FixedPower]
        affine_equation_count: int
        affine_zero_equations: list[int]

    class RuleTerm(TypedDict):
        coefficient_id: int
        integral: IntegralKey

    class Rule(TypedDict, total=False):
        ordinal: int
        target: IntegralKey
        case: RuleCase
        rhs: list[RuleTerm]
        excluded_all_zero_conjunctions: list[list[int]]

    class RuleExpressions(TypedDict):
        target: Expression
        rhs: Expression
        affine: list[Expression]
        excluded: list[list[Expression]]

    def rule_expressions(
        artifact: CandidateArtifact, rule: Rule, integral: Expression,
        parameter_bindings: Sequence[tuple[Expression, Expression]] = (),
    ) -> RuleExpressions:
        """Decode only the selected rule, binding its indices and dimension."""
        from symbolica import E, S

        indices = [S(f"n_{axis}") for axis in range(len(rule["target"]["values"]))]
        bindings = [(S(f"rustred::n{axis}"), index)
                    for axis, index in enumerate(indices)] + list(parameter_bindings)
        coefficients: dict[int, Expression] = {}

        def coefficient(cid: int) -> Expression:
            if cid not in coefficients:
                detail = artifact.coefficient(cid, max_output_bytes=65536)
                value = (E(detail["numerator"], default_namespace="rustred")
                         / E(detail["denominator"], default_namespace="rustred"))
                for source, display in bindings:
                    value = value.replace(source, display)
                coefficients[cid] = value
            return coefficients[cid]

        def key_expression(key: IntegralKey) -> Expression:
            return integral(*(indices[axis] + value if symbolic else value
                              for axis, (value, symbolic) in enumerate(
                                  zip(key["values"], key["symbolic"]))))

        return {
            "target": key_expression(rule["target"]),
            "rhs": sum((coefficient(term["coefficient_id"]) * key_expression(term["integral"])
                        for term in rule["rhs"]), E("0")),
            "affine": [coefficient(cid) for cid in rule["case"]["affine_zero_equations"]],
            "excluded": [[coefficient(cid) for cid in branch]
                         for branch in rule["excluded_all_zero_conjunctions"]],
        }

    def integral_notation(key: Union[IntegralKey, tuple[int, ...]]) -> str:
        """Format native integer structure without parsing coefficient text."""
        if isinstance(key, dict):
            values, symbolic = key["values"], key["symbolic"]
            if len(values) != len(symbolic):
                raise ValueError("Integral flags and values have different arities")
        else:
            values, symbolic = key, [False] * len(key)
        powers = []
        for axis, (value, variable) in enumerate(zip(values, symbolic)):
            if type(value) is not int or type(variable) is not bool:
                raise ValueError("Expected exact native integer powers and boolean flags")
            if not variable:
                powers.append(str(value))
            else:
                suffix = f" + {value}" if value > 0 else f" - {-value}" if value < 0 else ""
                powers.append(f"n_{axis}{suffix}")
        return "I(" + ", ".join(powers) + ")"

    def rule_fixed_conditions(rule: Rule) -> list[FixedPower]:
        """Only conditions not already displayed as literal target powers."""
        target = rule["target"]
        return [item for item in rule["case"]["fixed"]
                if target["symbolic"][item["axis"]]
                or target["values"][item["axis"]] != item["value"]]

    def rule_label(rule: Rule) -> str:
        """Show the target and only the case conditions it does not encode."""
        case = rule["case"]
        parts = [f"n_{item['axis']} = {item['value']}"
                 for item in rule_fixed_conditions(rule)]
        affine = case.get("affine_equation_count", len(case.get("affine_zero_equations", [])))
        if affine:
            parts.append(f"{affine} affine {'condition' if affine == 1 else 'conditions'}")
        label = f"{rule['ordinal']} · {integral_notation(rule['target'])}"
        return label + (" · Case: " + "; ".join(parts) if parts else "")


@app.cell(hide_code=True)
def _():
    from symbolica import E, S
    from symbolica.community.hepkit import FeynmanDiagram, IntegralFamily, Kinematics, Model
    from symbolica.community.hepkit.ibp import IBPFamily
    from symbolica.community.hepkit.rustred import (
        certify_candidates,
        family_candidates,
        inspect_closing_artifact,
        reduce_with_closing_artifact,
    )
    return (
        E, S, FeynmanDiagram, IBPFamily, IntegralFamily, Kinematics, Model,
        certify_candidates, family_candidates, inspect_closing_artifact,
        reduce_with_closing_artifact,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Three-loop massive vacuum reduction

    [Browse notebooks](/) · [Integral families](/?file=hep/integral_families.py) ·
    [Four-loop candidate search](/?file=hep/four_loop_reduction.py)

    **Draw the graph → generate IBP rules → certify closure → reduce to masters.**

    The equal-mass **Mercedes graph** has four vertices, six edges and three
    loops. We compute exact coefficients in symbolic dimension $d$, including
    raised propagators, pinches and numerator insertions. The recursive
    reducer uses the artifact generated and certified in this notebook.

    Generation runs automatically on one core in Python and WASM. The result
    contains **38 raw terminal keys**, grouped into five topology types at the
    end; no numerical master values or minimal-basis claim are needed here.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Route the graph

    With $M=m^2$ and $D_i=q_i^2-M$, define
    $I_M(n)=\int_{k_1,k_2,k_3}\prod_{i=1}^{6}D_i^{-n_i}$ in a common
    dimensionally regulated measure. Positive powers are denominators,
    zero removes an edge, and negative powers insert numerator factors.
    Set $M=1$ for the IBP calculation and restore it by homogeneity below.
    All six scalar products are covered: no auxiliary propagator is needed.
    """)
    return


@app.cell
def _(E, FeynmanDiagram, IBPFamily, IntegralFamily, Kinematics, Model, S):
    dimension, mass_squared, integral = S("d", "M", "I")
    scalar_model = Model.phi_3_4()
    dot_input = """
    digraph Mercedes {
        // Three loop-basis chords; edge IDs fix the denominator order.
        B -> A [id=0, particle="phi", lmb_id=0];
        A -> C [id=1, particle="phi", lmb_id=1];
        C -> B [id=2, particle="phi", lmb_id=2];
        D -> B [id=3, particle="phi"];
        A -> D [id=4, particle="phi"];
        C -> D [id=5, particle="phi"];
    }
    """
    diagram = FeynmanDiagram.from_dot(scalar_model, dot_input)
    routed = diagram.integral_family(kinematics=Kinematics(dimension))
    model_mass = scalar_model.particle("phi").mass
    family = IntegralFamily(
        routed.loop_momenta, routed.external_momenta,
        [den.replace(model_mass, E("1")) for den in routed.denominators],
        kinematics=routed.kinematics,
    )
    # Validate the routed family with the IBP backend.
    assert IBPFamily(family, name="Mercedes").denominator_count == 6
    # The source API preserves d; explicitly map its native symbol for display.
    parameter_bindings = ((S("rustred::d"), dimension),)
    mo.vstack([diagram, family])
    return (
        dimension,
        dot_input,
        integral,
        mass_squared,
        parameter_bindings,
    )


@app.cell(hide_code=True)
def _():
    # The closure API takes a source family in the graph's denominator order.
    generation_source = """
    schema = "rustred.project.toml.v1"

    # The same ordered family as the graph, with common mass set to one.
    # Family input: no reduction rules or hints.
    [family]
    name = "rustred_three_loop_unit_mass_vacuum_k6_v1"
    loop_momenta = ["k1", "k2", "k3"]
    external_momenta = []
    dimension = "d"

    [[family.denominators]]
    id = "D1"
    expression = "k1^2-1"

    [[family.denominators]]
    id = "D2"
    expression = "k2^2-1"

    [[family.denominators]]
    id = "D3"
    expression = "k3^2-1"

    [[family.denominators]]
    id = "D4"
    expression = "(k1-k3)^2-1"

    [[family.denominators]]
    id = "D5"
    expression = "(k1-k2)^2-1"

    [[family.denominators]]
    id = "D6"
    expression = "(k2-k3)^2-1"

    [target]
    powers = [1, 1, 1, 1, 1, 1]
    numerator = "1"
    """
    return (generation_source,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2. Generate and certify the rules

    These calls generate the rules from the family, then verify coverage and
    termination before reduction. They run on one core in both Python and WASM.
    Changing a sector, rule or target below reuses this result.
    """)
    return


@app.cell
def _(family_candidates, generation_source):
    _started = perf_counter()
    candidates = family_candidates(
        generation_source, input_format="toml", n_cores=1,
        exact_backend="sparse", numerical_depth=2,
    )
    candidate_artifact = candidates.artifact()
    generation_seconds = perf_counter() - _started
    return candidate_artifact, candidates, generation_seconds


@app.cell(hide_code=True)
def _(candidate_artifact, generation_seconds):
    mo.hstack([
        mo.stat(candidate_artifact.metadata()["total_rules"], label="Generated rules"),
        mo.stat(f"{generation_seconds:.2f} s", label="Generation time"),
    ], widths="equal")
    return


@app.cell
def _(candidates, certify_candidates, inspect_closing_artifact):
    certificate = certify_candidates(candidates.bundle)
    closing_artifact = certificate.artifact
    inspected = inspect_closing_artifact(closing_artifact)
    inspection = tomllib.loads(inspected.to_toml())
    master_powers = {tuple(item["powers"]) for item in inspection["artifact"]["masters"]}
    assert certificate.status == "generated-durable" and inspected.status == "inspected"
    assert inspection["artifact"]["arity"] == 6 and master_powers
    reductions = {}
    return closing_artifact, inspection, master_powers, reductions


@app.cell(hide_code=True)
def _(candidate_artifact):
    _sectors = candidate_artifact.sectors(start=0, limit=64)["items"]
    _options = {
        f"{''.join('1' if b else '0' for b in row['sector'])} · {row['total_rules']} rules": row["ordinal"]
        for row in _sectors
    }
    _default = next(row["ordinal"] for row in _sectors
                    if all(row["sector"]) and row["total_rules"])
    sector_choice = mo.ui.dropdown(options=_options,
        value=next(label for label, ordinal in _options.items() if ordinal == _default),
        label="Sector", allow_select_none=False, searchable=True, full_width=True)
    mo.vstack([
        mo.md("""
        ## 3. Inspect a generated recurrence

        Select a sector, then a rule. The label shows the target and any extra
        case conditions. Only the selected rule's coefficients are decoded.
        """),
        sector_choice,
    ])
    return (sector_choice,)


@app.cell(hide_code=True)
def _(candidate_artifact, sector_choice):
    _sector = int(sector_choice.value)
    _page = candidate_artifact.rules(_sector, start=0, limit=1000)
    _summaries = list(_page["items"])
    for _start in range(1000, _page["total"], 1000):
        _summaries.extend(candidate_artifact.rules(_sector, start=_start, limit=1000)["items"])
    _options = {rule_label(rule): rule["ordinal"] for rule in _summaries}
    mo.stop(not _options, mo.md("This sector has no recurrence rules."))
    rule_choice = mo.ui.dropdown(options=_options, value=next(iter(_options)),
        label="Rule", allow_select_none=False, searchable=True, full_width=True)
    rule_choice
    return (rule_choice,)


@app.cell
def _(candidate_artifact, rule_choice, sector_choice):
    rule_detail = candidate_artifact.rule(
        int(sector_choice.value), int(rule_choice.value), max_output_bytes=65536,
    )
    return (rule_detail,)


@app.cell(hide_code=True)
def _(
    candidate_artifact,
    integral,
    parameter_bindings,
    rule_detail,
    sector_choice,
):
    _sector = candidate_artifact.sectors(start=int(sector_choice.value), limit=1)["items"][0]
    _expressions = rule_expressions(candidate_artifact, rule_detail, integral, parameter_bindings)
    _conditions = [mo.md("**Sector:** " + ", ".join(
        f"n_{axis} {'> 0' if active else '≤ 0'}" for axis, active in enumerate(_sector["sector"])))]
    _case = [mo.md(f"$n_{item['axis']} = {item['value']}$")
             for item in rule_fixed_conditions(rule_detail)]
    _case.extend(mo.hstack([equation.formatted(max_terms=None), mo.md("**= 0**")],
                          justify="start") for equation in _expressions["affine"])
    for _number, _branch in enumerate(_expressions["excluded"], 1):
        _conditions.append(mo.md(f"**Excluded branch {_number}:** all expressions below vanish"
                                 if _branch else f"**Excluded branch {_number}:** always true"))
        _conditions.extend(expr.formatted(max_terms=None) for expr in _branch)
    _conditions.append(mo.md("Any excluded branch forbids the rule; equations within a branch "
        "are joined by **AND**. Denominator poles, source conditions and rule priority "
        "also govern applicability."))
    mo.vstack([
        mo.md(f"**Rule {rule_detail['ordinal']} · {len(rule_detail['rhs'])} RHS terms**"),
        *([mo.hstack([mo.md("**Case:**"), *_case], justify="start", wrap=True)] if _case else []),
        mo.hstack([_expressions["target"], mo.md("**=**")], justify="start"),
        _expressions["rhs"].formatted(max_terms=None, max_line_length=None, terms_on_new_line=True),
        mo.accordion({"Applicability conditions": mo.vstack(_conditions)}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Reduce completely and restore the mass

    Selecting an integral calls the reducer below. It follows the certified
    rules recursively, combines coefficients
    exactly and returns only raw terminal keys. Results are cached per target.

    For three loops, $I_M(n)=M^{3d/2-\sum_i n_i}I_1(n)$. Thus a unit-mass
    coefficient $c_a(d)$ becomes
    $c_a(d)\,M^{\sum_i a_i-\sum_i n_i}$ multiplying $I_M(a)$.
    The returned terms include this integer mass exponent.
    """)
    return


@app.cell(hide_code=True)
def _(closing_artifact):
    mo.stop(not closing_artifact)
    targets = {
        "Doubled propagator · I(2,1,1,1,1,1)": (2, 1, 1, 1, 1, 1),
        "Two doubled propagators · I(2,2,1,1,1,1)": (2, 2, 1, 1, 1, 1),
        "Pinched graph with dots · I(2,2,1,0,0,0)": (2, 2, 1, 0, 0, 0),
        "Numerator insertion · I(1,1,1,-1,0,0)": (1, 1, 1, -1, 0, 0),
        "Scaleless sector · I(0,0,0,0,0,0)": (0, 0, 0, 0, 0, 0),
    }
    target_choice = mo.ui.dropdown(options=targets, value=next(iter(targets)),
        label="Target integral", allow_select_none=False)
    target_choice
    return (target_choice,)


@app.cell
def _(
    closing_artifact,
    reductions,
    reduce_with_closing_artifact,
    target_choice,
):
    selected_powers = tuple(target_choice.value)
    if selected_powers not in reductions:
        reductions[selected_powers] = reduce_with_closing_artifact(
            closing_artifact, list(selected_powers))
    reduction = reductions[selected_powers]
    return reduction, selected_powers


@app.cell(hide_code=True)
def _(E, inspection, master_powers, parameter_bindings, reduction, selected_powers):
    assert reduction.status == "reduced"
    assert reduction.family_fingerprint == inspection["artifact"]["family_fingerprint"]
    reduced_terms = []
    for _term in reduction.terms:
        _master = tuple(_term.master_powers)
        assert _master in master_powers
        assert _term.common_mass_squared_power == sum(_master) - sum(selected_powers)
        # Native strings cross only the display boundary; native RustRed reduced them.
        _coefficient = E(_term.unit_mass_coefficient)
        for _internal, _original in parameter_bindings:
            _coefficient = _coefficient.replace(_internal, _original)
        reduced_terms.append((_master, _coefficient, _term.common_mass_squared_power))
    return (reduced_terms,)


@app.cell
def _(E, integral, mass_squared, reduced_terms, selected_powers):
    reduced_expression = sum(
        (coefficient * mass_squared**power * integral(*master)
         for master, coefficient, power in reduced_terms), E("0"),
    )
    mo.vstack([
        mo.hstack([integral(*selected_powers), mo.md("**=**")], justify="start"),
        reduced_expression.formatted(
            max_terms=None, max_line_length=None, terms_on_new_line=True,
        ),
    ])
    return (reduced_expression,)


@app.cell
def _(E, closing_artifact, dimension, parameter_bindings, reduce_with_closing_artifact, reductions):
    # Independent factorized check: three one-loop tadpoles with two raised powers.
    _target = (2, 2, 1, 0, 0, 0)
    if _target not in reductions:
        reductions[_target] = reduce_with_closing_artifact(closing_artifact, list(_target))
    _terms = reductions[_target].terms
    assert len(_terms) == 1 and _terms[0].master_powers == [1, 1, 1, 0, 0, 0]
    _coefficient = E(_terms[0].unit_mass_coefficient)
    for _internal, _original in parameter_bindings:
        _coefficient = _coefficient.replace(_internal, _original)
    assert (_coefficient - ((dimension - 2) / 2)**2).together() == 0
    assert _terms[0].common_mass_squared_power == -2
    tadpole_checked = True
    mo.callout(r"Independent check passed: the factorized tadpole ratio is "
               r"$I_M(2,2,1,0,0,0)=\frac{(d-2)^2}{4M^2}I_M(1,1,1,0,0,0)$.", kind="success")
    return (tadpole_checked,)


@app.cell(hide_code=True)
def _(master_powers):
    _types = [
        {"Type": "T3,1", "Name": "Three one-loop tadpoles", "Raw keys": 16,
         "Representative": "I(1,1,1,0,0,0)"},
        {"Type": "T4,1", "Name": "Two-loop sunset × one-loop tadpole", "Raw keys": 12,
         "Representative": "I(1,1,1,1,0,0)"},
        {"Type": "T4,2", "Name": "Three-loop basketball (four-line banana)", "Raw keys": 3,
         "Representative": "I(0,1,1,1,1,0)"},
        {"Type": "T5,1", "Name": "Connected five-line vacuum", "Raw keys": 6,
         "Representative": "I(0,1,1,1,1,1)"},
        {"Type": "T6,1", "Name": "Mercedes (tetrahedron / K4)", "Raw keys": 1,
         "Representative": "I(1,1,1,1,1,1)"},
    ]
    _rows = [{"Raw terminal key": integral_notation(powers), "Total power": sum(powers)}
             for powers in sorted(master_powers)]
    mo.vstack([
        mo.md(r"""
        ## 5. Terminal topologies

        For this equal-mass input, changes of loop-momentum variables identify
        the raw keys in the following groups: $16+12+3+6+1=38$.
        The labels follow [R. N. Lee, Figure 2](https://arxiv.org/pdf/1203.4868#page=5).
        They name graph types, not a conversion to that paper's normalization.
        Each representative uses this notebook's denominator order, not a
        relabeling of the artifact. The first two types factorize into
        lower-loop integrals; the last three are connected three-loop graphs.
        """),
        mo.ui.table(_types, selection=None, pagination=False,
                    show_column_summaries=False, show_download=False),
        mo.md("""
        **The certified artifact is unchanged.** The following table and the
        reductions retain all 38 raw keys: the display does not apply the
        topology identifications, merge coefficients, or supply an additional
        proof of master independence. Closure means reduction terminates in
        these keys; independence and basis minimization are different questions.
        """),
        mo.ui.table(_rows, selection=None, pagination=True, page_size=10,
                    show_column_summaries=False, show_download=False),
    ])
    return


@app.cell
def _(reduction, tadpole_checked):
    from symbolica import get_citations

    _ = reduction, tadpole_checked  # Collect after the calculations register their citations.
    get_citations()
    return


if __name__ == "__main__":
    app.run()
