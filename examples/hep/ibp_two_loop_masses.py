import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="IBP: massive two-loop 2- and 3-point functions",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parametric IBP for massive two-loop functions
    [Browse all notebooks](/) · [Unequal-mass bubble](/?file=hep/ibp_bubble.py) ·
    [Massless triangle](/?file=hep/ibp_triangle.py)

    Generate reusable **symbolic-index recurrences** for a two-point sunrise
    and a six-propagator three-point vertex, then apply them to concrete
    raised powers and linear combinations of integrals. All propagator masses
    are independent: `m1sq`, …, `m6sq` denote **squared masses**. The dimension
    $D$ and external invariants also remain symbolic throughout the reduction.

    We use $D_j=r_j^2-m_j^2$ and
    $$I(n_1,\ldots,n_N)=\int\frac{d^Dk\,d^Dl}{(i\pi^{D/2})^2}
       \prod_{j=1}^{N}D_j^{-n_j}.$$
    The common normalization and causal prescriptions do not affect the IBP
    algebra. Positive powers are propagators; zero pinches a propagator;
    negative powers represent numerator factors.

    `IntegralFamily.complete()` appends irreducible scalar products (ISPs).
    We fix their powers to zero when deriving the scalar recurrences, leaving
    **every physical propagator power symbolic**. Right-hand sides may still
    contain numerator integrals. RustRed's `solve_parametric` discovers a
    recurrence, and `reduce` applies one step: this notebook does not claim a
    complete reduction of every sector to an independent master basis.

    Run with a community build providing `IBPFamily` and Marimo. The
    fully symbolic vertex search can take about a minute on a native build;
    changing the target below reuses the generated rules.
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
    from collections.abc import Iterable, Sequence
    from time import perf_counter

    import marimo as mo
    from marimo import Html
    from symbolica import E, Expression, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hep import (
        FeynmanDiagram,
        IBPFamily,
        IBPSolution,
        IntegralFamily,
        Kinematics,
        Model,
    )

    _set_namespace("massive_ibp")
    return (
        E,
        Expression,
        FeynmanDiagram,
        Html,
        IBPFamily,
        IBPSolution,
        IntegralFamily,
        Iterable,
        Kinematics,
        Model,
        S,
        Sequence,
        mo,
        perf_counter,
    )


@app.cell(hide_code=True)
def _(E, Expression, Html, IBPSolution, Iterable, Sequence, mo):
    def integral_sum(
        terms: Iterable[tuple[Sequence[int], Expression]], head: Expression
    ) -> Expression:
        return sum(
            (coefficient * head(*powers) for powers, coefficient in terms), E("0")
        )

    def show_recurrence(
        solution: IBPSolution, head: Expression, elapsed: float
    ) -> Html:
        # Print full expressions rather than the abbreviated rich repr, and keep
        # the rules visible without requiring an accordion to be opened.
        def full(expression: Expression) -> str:
            return expression.format(
                max_terms=None,
                max_line_length=100,
                show_namespaces=False,
                num_exp_as_superscript=False,
            )

        printed_rules = []
        for number, rule in enumerate(solution.rules, 1):
            lines = [f"Rule {number}", f"{full(head(*rule.target))} ="]
            for i, (powers, coefficient) in enumerate(rule.terms):
                prefix = "    " if i == 0 else "  + "
                lines.append(
                    f"{prefix}({full(coefficient.factor())}) * {full(head(*powers))}"
                )
            lines.extend(["", "Index domain (all indices are integers):"])
            for i, (power, positive) in enumerate(zip(rule.target, rule.sector), 1):
                if power == E("0"):
                    lines.append(f"  n{i} = 0 (fixed ISP power)")
                else:
                    lines.append(f"  n{i} {'> 0' if positive else '<= 0'}")
            lines.extend(["", "Nonzero conditions (ALL must hold):"])
            lines.extend(
                f"  ({full(condition.factor())}) != 0"
                for condition in rule.nonzero_conditions
            )
            if not rule.nonzero_conditions:
                lines.append("  None.")
            lines.extend(["", "Excluded branches (ANY branch forbids application):"])
            for branch in rule.exceptions:
                lines.append(
                    "  "
                    + " AND ".join(
                        f"({full(condition.factor())}) = 0" for condition in branch
                    )
                )
            if not rule.exceptions:
                lines.append("  None.")
            printed_rules.append("\n".join(lines))
        rules_text = "\n\n".join(printed_rules)
        return mo.vstack(
            [
                mo.md(
                    f"Generated **{len(solution.rules)}** recurrence(s) in **{elapsed:.2f} s**."
                ),
                mo.ui.table([solution.stats], selection=None),
                mo.md("### Generated parametric rules and validity conditions"),
                mo.md(f"```text\n{rules_text}\n```"),
                mo.download(
                    data=rules_text.encode("utf-8"),
                    filename=f"{full(head)}_parametric_rules.txt",
                    label="Download the complete rules and conditions",
                ),
            ]
        )

    def apply_checked(
        solution: IBPSolution, powers: Sequence[int], head: Expression
    ) -> tuple[list[tuple[list[int], Expression]], Expression]:
        # An unmatched target is returned unchanged by solution.reduce().
        # Assert that the selected example actually uses a recurrence.
        terms = solution.reduce(list(powers))
        assert terms and all(tuple(p) != tuple(powers) for p, _ in terms)
        expression = solution.reduce(list(powers), integral=head)
        assert (expression - integral_sum(terms, head)).expand() == E("0")
        return terms, expression

    return apply_checked, show_recurrence


@app.cell(hide_code=True)
def _(E, IBPFamily, IntegralFamily):
    def check_identities(family: IntegralFamily, ibp: IBPFamily) -> int:
        kin = family.kinematics
        loops = family.loop_momenta
        external = family.external_momenta
        pairs = [(a, b) for i, a in enumerate(loops) for b in loops[i:]]
        pairs += [(a, b) for a in loops for b in external]
        indices = ibp.index_symbols
        rows = ibp.ibp_identities()
        assert len(rows) == len(loops) * (len(loops) + len(external))
        # Native rows group by contraction vector, then differentiated loop.
        for row, (loop, vector) in zip(
            rows, [(a, b) for b in loops + external for a in loops]
        ):
            expected = kin.dimension if loop == vector else E("0")
            for index, denominator in zip(indices, family.denominators):
                derivative = E("0")
                for a, b in pairs:
                    contraction = E("0")
                    if a == loop:
                        contraction += kin.scalar_product(vector, b)
                    if b == loop:
                        contraction += kin.scalar_product(a, vector)
                    derivative += (
                        denominator.expand().coefficient(kin.scalar_product(a, b))
                        * contraction
                    )
                expected -= index * derivative / denominator
            actual = E("0")
            for powers, coefficient in row:
                term = coefficient
                for denominator, power, index in zip(
                    family.denominators, powers, indices
                ):
                    shift = int(str((index - power).expand()))
                    term *= denominator**shift
                actual += term
            assert (actual - expected).together() == E("0")
        return len(rows)

    return (check_identities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(S):
    D, k, l, p, q, s, t, u = S(
        "D",
        "k",
        "l",
        "p",
        "q",
        "s",
        "t",
        "u",
    )
    masses = S(*(f"m{i}sq" for i in range(1, 7)))
    J, V = S("J", "V")
    return D, J, V, k, l, masses, p, q, s, t, u


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Two-point function: unequal-mass sunrise
    The external legs carry $p$ and $-p$, with $p^2=s$:
    $$D_1=k^2-m_1^2,\qquad D_2=l^2-m_2^2,\qquad
      D_3=(p-k-l)^2-m_3^2.$$
    There are $L(L+1)/2+LE=5$ independent loop scalar products for
    $L=2$, $E=1$. Completion adds two ISP slots after these three propagators;
    their precise definitions and order are displayed below.
    """)
    return


@app.cell
def _(D, IBPFamily, IntegralFamily, Kinematics, k, l, masses, p, s):
    sunrise_kinematics = Kinematics(D, momenta=[k, l, p]).with_scalar_product(p, p, s)
    sunrise_family = IntegralFamily(
        [k, l],
        [p],
        [
            sunrise_kinematics.scalar_product(r, r) - mass
            for r, mass in zip([k, l, p - k - l], masses[:3])
        ],
        kinematics=sunrise_kinematics,
    ).complete()
    assert sunrise_family.is_complete and sunrise_family.is_independent
    assert len(sunrise_family.denominators) == 5
    sunrise_ibp = IBPFamily(sunrise_family, name="massive_sunrise")
    sunrise_family
    return sunrise_family, sunrise_ibp


@app.cell
def _(J, perf_counter, show_recurrence, sunrise_ibp):
    _start = perf_counter()
    sunrise_solution = sunrise_ibp.solve_parametric(
        [True, True, True, False, False],
        fixed=[None, None, None, 0, 0],
        max_depth=1,
    )
    assert sunrise_solution.rules
    show_recurrence(sunrise_solution, J, perf_counter() - _start)
    return (sunrise_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start from a Feynman diagram

    Use **`diagram.integral_family()`** to obtain a complete family.
    It reads the stored momentum routing and model masses, retains the physical
    propagators first, and automatically appends suitable ISPs. Pass preferred
    dot products as its optional argument to choose the auxiliary entries.

    This example loads a sunrise from DOT using `Model.phi4()`, so all three
    propagators have the model's common mass. The unequal-mass example above
    instead specifies each denominator explicitly.

    `IntegralFamily.from_diagram(diagram, ...)` offers the same completion.
    The returned family exposes the routed momentum names, which we use
    below to impose $p^2=s$. Use `diagram.propagator_family()` when you need
    only the physical propagators, for example before partial fractioning.
    The original scalar integral has powers `[1, 1, 1, 0, 0]` in either
    completed family; the last two entries are auxiliary denominators.
    """)
    return


@app.cell
def _(D, FeynmanDiagram, Kinematics, Model, mo, s):
    sunrise_diagram = FeynmanDiagram.from_dot(
        Model.phi4(),
        """digraph sunrise {
            ext [style=invis];
            ext -> a [particle="phi"];
            a -> b [particle="phi", lmb_id=0];
            a -> b [particle="phi", lmb_id=1];
            a -> b [particle="phi"];
            b -> ext [particle="phi"];
        }""",
    )
    _routed = sunrise_diagram.integral_family(kinematics=Kinematics(D))
    _p = _routed.external_momenta[0]
    _kin = _routed.kinematics.with_scalar_product(_p, _p, s)
    automatic_diagram_family = sunrise_diagram.integral_family(kinematics=_kin)
    _dot_products = [_kin.scalar_product(_k, _p) for _k in _routed.loop_momenta]
    diagram_family = sunrise_diagram.integral_family(
        independent_dot_products=_dot_products, kinematics=_kin
    )
    assert (
        automatic_diagram_family.is_complete and automatic_diagram_family.is_independent
    )
    assert diagram_family.is_complete and diagram_family.is_independent
    assert len(diagram_family.denominators) == 5
    assert diagram_family.denominators[:3] == automatic_diagram_family.denominators[:3]
    assert diagram_family.denominators[3:] == _dot_products
    mo.vstack(
        [
            mo.md("**Automatically completed family**"),
            automatic_diagram_family,
            mo.md("**Family with the two loop–external dot products as ISPs**"),
            diagram_family,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Three-point function: six unequal masses
    Take incoming momenta $p,q,-p-q$ with $p^2=s$, $q^2=t$ and
    $(p+q)^2=u$, so $p\cdot q=(u-s-t)/2$. No external leg is put on shell.
    The connected two-loop vertex has
    $$\begin{aligned}
    D_1&=k^2-m_1^2,& D_2&=(k-p)^2-m_2^2,& D_3&=(k-p-q)^2-m_3^2,\\
    D_4&=l^2-m_4^2,& D_5&=(l-p-q)^2-m_5^2,& D_6&=(k-l)^2-m_6^2.
    \end{aligned}$$
    The mixed propagator $D_6$ couples the two loops. Here $L=2$, $E=2$,
    so completion adds one ISP to span all seven loop scalar products.
    """)
    return


@app.cell
def _(D, IBPFamily, IntegralFamily, Kinematics, k, l, masses, p, q, s, t, u):
    vertex_kinematics = (
        Kinematics(D, momenta=[k, l, p, q])
        .with_scalar_product(p, p, s)
        .with_scalar_product(q, q, t)
        .with_scalar_product(p, q, (u - s - t) / 2)
    )
    vertex_family = IntegralFamily(
        [k, l],
        [p, q],
        [
            vertex_kinematics.scalar_product(r, r) - mass
            for r, mass in zip([k, k - p, k - p - q, l, l - p - q, k - l], masses)
        ],
        kinematics=vertex_kinematics,
    ).complete()
    assert vertex_family.is_complete and vertex_family.is_independent
    assert len(vertex_family.denominators) == 7
    vertex_ibp = IBPFamily(vertex_family, name="massive_vertex")
    vertex_family
    return vertex_family, vertex_ibp


@app.cell
def _(V, perf_counter, show_recurrence, vertex_ibp):
    _start = perf_counter()
    vertex_solution = vertex_ibp.solve_parametric(
        [True, True, True, True, True, True, False],
        fixed=[None, None, None, None, None, None, 0],
        max_depth=1,
    )
    assert vertex_solution.rules
    show_recurrence(vertex_solution, V, perf_counter() - _start)
    return (vertex_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Apply and reuse the reductions
    Each preset has zero ISP powers. The sunrise presets raise $n_2,n_3$;
    the vertex presets raise $n_6$, avoiding the exceptional integer indices
    of the discovered rules. Select a target to see its exact reduction.
    The code also applies each recurrence to **all three presets** and to a
    linear combination, keeping the masses, invariants and $D$ symbolic.
    """)
    return


@app.cell
def _(mo):
    sunrise_targets = [(1, 2, 2, 0, 0), (2, 2, 3, 0, 0), (3, 3, 3, 0, 0)]
    vertex_targets = [
        (1, 1, 1, 1, 1, 2, 0),
        (2, 1, 1, 1, 1, 3, 0),
        (2, 2, 2, 2, 2, 2, 0),
    ]
    sunrise_choice = mo.ui.dropdown(
        {str(powers): powers for powers in sunrise_targets},
        value=str(sunrise_targets[0]),
        label="Sunrise powers",
    )
    vertex_choice = mo.ui.dropdown(
        {str(powers): powers for powers in vertex_targets},
        value=str(vertex_targets[0]),
        label="Vertex powers",
    )
    mo.vstack([sunrise_choice, vertex_choice])
    return sunrise_choice, sunrise_targets, vertex_choice, vertex_targets


@app.cell
def _(
    J,
    V,
    apply_checked,
    sunrise_solution,
    sunrise_targets,
    vertex_solution,
    vertex_targets,
):
    sunrise_reductions = {
        powers: apply_checked(sunrise_solution, powers, J) for powers in sunrise_targets
    }
    vertex_reductions = {
        powers: apply_checked(vertex_solution, powers, V) for powers in vertex_targets
    }
    return sunrise_reductions, vertex_reductions


@app.cell
def _(
    J,
    V,
    mo,
    sunrise_choice,
    sunrise_reductions,
    vertex_choice,
    vertex_reductions,
):
    mo.vstack(
        [
            mo.md("**Sunrise: input and reduced expression**"),
            J(*sunrise_choice.value),
            sunrise_reductions[sunrise_choice.value][1].collect_factors().collect_num(),
            mo.md("**Vertex: input and reduced expression**"),
            V(*vertex_choice.value),
            vertex_reductions[vertex_choice.value][1].collect_factors().collect_num(),
        ]
    )
    return


@app.cell
def _(
    J,
    V,
    mo,
    s,
    sunrise_reductions,
    sunrise_targets,
    u,
    vertex_reductions,
    vertex_targets,
):
    # Each second target has two additional propagator powers; s**2/u**2
    # give the two terms in each combination the same mass dimension.
    sunrise_input = J(*sunrise_targets[0]) + s**2 * J(*sunrise_targets[1])
    sunrise_reduced = (
        sunrise_reductions[sunrise_targets[0]][1]
        + s**2 * sunrise_reductions[sunrise_targets[1]][1]
    ).expand()
    vertex_input = V(*vertex_targets[0]) + u**2 * V(*vertex_targets[1])
    vertex_reduced = (
        vertex_reductions[vertex_targets[0]][1]
        + u**2 * vertex_reductions[vertex_targets[1]][1]
    ).expand()
    mo.accordion(
        {
            "Reduced sunrise combination": mo.vstack([sunrise_input, sunrise_reduced]),
            "Reduced vertex combination": mo.vstack([vertex_input, vertex_reduced]),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the defining IBP identities independently
    For every loop $k_i$ and vector $v\in\{k,l,p,q\}$ present in the family,
    direct differentiation gives
    $$0=\int\prod_j D_j^{-n_j}\left[
       D\,\delta_{v,k_i}-\sum_j\frac{n_j}{D_j}
           v\cdot\frac{\partial D_j}{\partial k_i}\right].$$
    Below, differentiate the original denominators using
    $v\cdot\partial_{k_i}(a\cdot b)
      =\delta_{a,k_i}\,v\cdot b+\delta_{b,k_i}\,a\cdot v$.
    Separately convert each native IBP row back to its integrand by replacing
    $I(n+\Delta)/I(n)$ with $\prod_j D_j^{-\Delta_j}$.
    Their difference must vanish **symbolically**, checking the mass signs,
    external kinematics, index shifts and ISP ordering for all 6 sunrise and
    8 vertex identities. No numerical parameter choices enter these checks.
    """)
    return


@app.cell
def _(
    check_identities,
    mo,
    sunrise_family,
    sunrise_ibp,
    vertex_family,
    vertex_ibp,
):
    identity_checks = {
        "Sunrise": check_identities(sunrise_family, sunrise_ibp),
        "Vertex": check_identities(vertex_family, vertex_ibp),
    }
    mo.md(
        f"**Exact checks passed:** {identity_checks['Sunrise']} sunrise and "
        f"{identity_checks['Vertex']} vertex IBP identities agree with direct differentiation."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Domain and further reduction
    `rule.apply(powers)` rejects incompatible sectors, fixed indices and
    exceptional integer powers. `solution.reduce(powers)` tries the available
    rules and leaves an unmatched integral unchanged. The examples assert
    that every selected target really changes.

    **Kinematic conditions remain your responsibility:** before specializing
    masses, invariants or $D$, check the displayed nonzero conditions after
    substituting the target indices. For exceptional kinematics, construct
    the specialized family and solve it separately. Keep $D$ symbolic until
    after reduction if a dimensional-regularization expansion is needed.

    One recurrence need not cover its own right-hand side. Boundary indices,
    pinched sectors and numerator sectors require additional rules. For a
    finite set of targets, `ibp.reduce_laporta(targets, max_depth=...)` also
    performs back-substitution and exposes its remaining basis as `residuals`;
    those residuals are not certified independent masters. Applying these
    algebraic reductions does not numerically evaluate the two-loop integrals.
    """)
    return


if __name__ == "__main__":
    app.run()
