# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "symbolica==3.0.1",
#     "marimo==0.24.0",
#     "typst==0.15.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Gamma algebra")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gamma algebra

    `simplify_algebra(gamma=True, epsilon=True)`
    enables Dirac identities and epsilon reduction, including the structural
    contractions they require. The identity cards explicitly request both
    families; color identities remain disabled. `contract()` performs structural
    work, while collecting a trace alone does not evaluate it.

    Every example below runs the installed implementation; the
    identity cards check the result against an explicit expected expression
    and check that a second pass does nothing.

    | Capability | Scope |
    |:--|:--|
    | Clifford contractions and ordinary traces | Compatible symbolic Lorentz dimensions |
    | Slashed momenta and chain joining | Registered Spenso tensor notation |
    | Chisholm, gamma5, chiral projectors, charge conjugation | Four dimensions |
    | Canonical chain ordering | Opt-in; may produce more terms |
    | Three-gamma epsilon expansion | Opt-in for open chains; automatic in short 4D trace kernels |
    | Short trace kernels | Ordinary and gamma5 traces, 1–14 ordinary gammas |
    | Expanded output | Explicit `result.expand()` materializes the complete result |

    This is a partial algebraic normal form. It does not impose on-shell
    kinematics, external Dirac equations or a dimensional-regularization
    gamma5 scheme. Run with the **combined Symbolica community host** containing
    Spenso and Idenso, plus Marimo and Typst; the ordinary Symbolica package
    alone does not provide these community bindings.
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
    from symbolica.community import tensor as sp
    from functools import partial

    import marimo as mo
    from symbolica import E, S
    from symbolica.community.tensor import (
        AUTO,
        DisplaySettings,
        Representation,
        TensorExpression,
        TensorName,
        chain,
        trace,
    )

    return (
        AUTO,
        DisplaySettings,
        E,
        Representation,
        S,
        TensorExpression,
        TensorName,
        chain,
        mo,
        partial,
        sp,
        trace,
    )


@app.cell(hide_code=True)
def _(DisplaySettings, E, TensorExpression, mo, show_dimensions):
    display_settings = DisplaySettings(show_dimensions=show_dimensions.value)

    def identity_card(title, expression, expected, settings=None):
        result = expression.simplify_algebra(
            contract="dots", **settings or dict(gamma=True), epsilon=True
        )
        expected_atom = (
            expected.to_expression()
            if isinstance(expected, TensorExpression)
            else expected
        )
        # Expand only the small verification difference, not the input numerator.
        expected_atom = TensorExpression(expected_atom).to_dots().to_expression()
        assert (result.to_expression() - expected_atom).expand() == E("0"), title
        assert (
            result.simplify_algebra(
                contract="dots", **settings or dict(gamma=True), epsilon=True
            )
            == result
        ), title
        return mo.vstack(
            [
                mo.md(f"**{title}** — identity and fixed point checked"),
                mo.hstack(
                    [
                        mo.vstack(
                            [
                                mo.md("Input"),
                                mo.Html(expression.to_html(settings=display_settings)),
                            ]
                        ),
                        mo.vstack(
                            [
                                mo.md("Result"),
                                mo.Html(result.to_html(settings=display_settings)),
                            ]
                        ),
                    ],
                    widths="equal",
                    align="start",
                ),
            ]
        )

    return display_settings, identity_card


@app.cell(hide_code=True)
def _(TensorExpression, TensorName, sp, spin):
    p, q = (
        sp.TensorName.vector("gamma_tutorial::p").to_expression(),
        sp.TensorName.vector("gamma_tutorial::q").to_expression(),
    )
    gamma_head = TensorName.dirac_gamma().to_expression()
    metric_head = TensorName.g().to_expression()

    def slash(momentum):
        return TensorExpression(
            gamma_head(spin.to_expression(), spin.to_expression(), momentum)
        )

    return gamma_head, metric_head, p, q, slash


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(AUTO, Representation, S, TensorExpression, chain, partial, trace):
    spin = Representation.bis(4)
    lorentz = Representation.mink(4)
    a, b, c, mu, nu, rho, sigma, D = S(
        *(
            f"gamma_tutorial::{name}"
            for name in ("a", "b", "c", "mu", "nu", "rho", "sigma", "D")
        )
    )
    gamma = TensorExpression.dirac_gamma(4)
    gm, gn, gr, gs = (
        gamma(AUTO, AUTO, lorentz(index)) for index in (mu, nu, rho, sigma)
    )
    g5 = TensorExpression.gamma5(4)(AUTO, AUTO)
    plus = TensorExpression.projp(4)(AUTO, AUTO)
    minus = TensorExpression.projm(4)(AUTO, AUTO)
    word = partial(chain, spin(a), spin(b))
    tr = partial(trace, spin)
    identity = spin.g(a, b)
    return (
        D,
        a,
        b,
        c,
        g5,
        gamma,
        gm,
        gn,
        gr,
        gs,
        identity,
        lorentz,
        minus,
        mu,
        nu,
        plus,
        rho,
        sigma,
        spin,
        tr,
        word,
    )


@app.cell(hide_code=True)
def _(mo):
    show_dimensions = mo.ui.switch(value=False, label="Show representation dimensions")
    show_dimensions
    return (show_dimensions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Open chains and contracted indices

    Gamma's argument order is `(spinor-in, spinor-out, Lorentz)`.
    `AUTO` supplies local ports to the public `chain` and `trace` builders;
    `word` below simply binds the two external spinor indices with `partial`.
    Repeating a Lorentz label contracts it. In four dimensions,
    $\gamma^\mu\gamma_\mu=4\mathbf{1}$ and
    $\gamma^\mu\gamma^\nu\gamma_\mu=-2\gamma^\nu$.

    Multiply indexed `TensorExpression` objects directly. Repeated explicit
    indices sew the factors, and contraction collects the resulting chains.
    """)
    return


@app.cell
def _(
    a,
    b,
    c,
    gamma,
    gm,
    gn,
    gr,
    gs,
    identity,
    identity_card,
    mo,
    mu,
    word,
):
    joined = gamma(a, c, mu) * gamma(c, b, mu)
    open_checks = mo.vstack(
        [
            identity_card("Adjacent contraction", word(gm, gm), 4 * identity),
            identity_card("Sandwich contraction", word(gm, gn, gm), -2 * word(gn)),
            identity_card(
                "Four-dimensional Chisholm reversal",
                word(gm, gn, gr, gs, gm),
                -2 * word(gs, gr, gn),
            ),
            identity_card("Collect explicitly sewn matrices", joined, 4 * identity),
        ],
        gap=2,
    )
    open_checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Symbolic dimension and trace normalization

    Change **Lorentz** dimension before simplifying a contracted expression.
    `with_lorentz_dimension(D)` leaves the spinor dimension unchanged, so
    $\gamma^\mu\gamma_\mu=D\mathbf{1}$ while $\mathrm{tr}(\mathbf{1})=4$.
    If you instead explicitly use `bis(D)`, the empty trace is $D$.
    The trace normalization comes from the spin representation; it is not
    inferred as $2^{D/2}$.
    """)
    return


@app.cell
def _(
    D,
    Representation,
    gm,
    gn,
    identity,
    identity_card,
    mo,
    mu,
    nu,
    tr,
    trace,
    word,
):
    dimension_checks = mo.vstack(
        [
            identity_card(
                "D-dimensional contraction",
                word(gm, gm).with_lorentz_dimension(D),
                D * identity,
            ),
            identity_card(
                "D-dimensional sandwich",
                word(gm, gn, gm).with_lorentz_dimension(D),
                (2 - D) * word(gn).with_lorentz_dimension(D),
            ),
            identity_card(
                "Lorentz D, spinor 4",
                tr(gm, gn).with_lorentz_dimension(D),
                4 * Representation.mink(D).g(mu, nu),
            ),
            identity_card(
                "Explicit bis(D) trace normalization", trace(Representation.bis(D)), D
            ),
        ],
        gap=2,
    )
    dimension_checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Slashed momenta

    Compact momenta inside a gamma factor represent a slash. This example
    constructs that registered Symbolica form directly:
    `gamma(bis(4,a), bis(4,b), p(mink(4)))`.
    No mass-shell relation is assumed: $p^2$ remains a scalar product.
    """)
    return


@app.cell
def _(a, b, identity_card, lorentz, metric_head, p, q, slash, sp, word):
    p_compact, q_compact = p(lorentz.to_expression()), q(lorentz.to_expression())
    slash_p = slash(p_compact)(a, b)
    slash_q = slash(q_compact)(a, b)
    slash_sandwich = word(slash(p_compact), slash(q_compact), slash(p_compact))
    incoming, outgoing, chain_head = (
        sp.PortPattern.chain_in(),
        sp.PortPattern.chain_out(),
        sp.TensorPattern.chain,
    )
    p_squared = metric_head(p_compact, p_compact)
    p_dot_q = metric_head(p_compact, q_compact)
    slash_check = identity_card(
        "Slash sandwich: 2(p·q) slash(p) − p² slash(q)",
        slash_sandwich,
        2 * p_dot_q * word(slash_p) - p_squared * word(slash_q),
    )
    slash_check
    return chain_head, incoming, outgoing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Ordinary traces

    Odd ordinary traces vanish. Even traces recurse to pairwise metrics;
    four gammas give the familiar three signed pairings. Disable trace
    evaluation when retaining closed spin chains is preferable downstream.

    Gamma simplification returns a `TensorExpression` and leaves scalar spectators
    factored. Call `result.expand()` explicitly when a polynomial is needed;
    that request materializes the complete result, including scalar spectators.
    `evaluate_traces=False` retains closed spin chains.
    """)
    return


@app.cell
def _(gm, gn, gr, gs, identity_card, lorentz, mo, mu, nu, rho, sigma, tr):
    trace_four_expected = 4 * (
        lorentz.g(mu, nu) * lorentz.g(rho, sigma)
        - lorentz.g(mu, rho) * lorentz.g(nu, sigma)
        + lorentz.g(mu, sigma) * lorentz.g(nu, rho)
    )
    trace_checks = mo.vstack(
        [
            identity_card("Empty trace", tr(), 4),
            identity_card("Odd trace", tr(gm, gn, gr), 0),
            identity_card("Four-gamma trace", tr(gm, gn, gr, gs), trace_four_expected),
            identity_card(
                "Keep a trace inert",
                tr(gm, gn),
                tr(gm, gn),
                dict(gamma=True, gamma_evaluate_traces=False),
            ),
        ],
        gap=2,
    )
    trace_checks
    return


@app.cell
def _(
    AUTO,
    S,
    a,
    b,
    c,
    display_settings,
    gamma,
    lorentz,
    mo,
    mu,
    nu,
    rho,
    sigma,
    tr,
):
    # Polynomial materialization is an explicit operation on the whole result.
    _trace = tr(
        *(gamma(AUTO, AUTO, lorentz(_i)) for _i in (mu, a, b, mu, c, nu, rho, sigma))
    )
    _x, _y = S("expanded_trace_example::x", "expanded_trace_example::y")
    _spectator = (_x + _y) ** 8
    _expected_body = (
        _trace.simplify_algebra(contract="dots", gamma=True, epsilon=True)
        .expand()
        .to_expression()
    )
    _source = _spectator * _trace
    _reduced = _source.simplify_algebra(contract="dots", gamma=True, epsilon=True)
    _expanded = _reduced.expand()
    assert _expanded.to_expression() == (_spectator * _expected_body).expand()
    assert (
        _expanded.simplify_algebra(contract="dots", gamma=True, epsilon=True).expand()
        == _expanded
    )
    expanded_trace_check = mo.vstack(
        [
            mo.md(
                "**Explicit polynomial materialization includes scalar $(x+y)^8$.** Exact result and rerun checked."
            ),
            mo.Html(_expanded.to_html(settings=display_settings)),
        ]
    )
    expanded_trace_check
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Conservative versus canonical ordering

    The default `gamma_ordering="repeated_pairs"` strategy moves gammas toward repeated
    partners and avoids ordering every distinct factor. `gamma_ordering="canonical"` also
    applies adjacent Clifford swaps. It is useful for cancellations between
    different orders, but can increase intermediate expression size.

    The residual below is the Clifford anticommutator minus its expected
    value. Canonical mode reduces it to zero; the default leaves a residual.
    Neither option promises a globally minimal expression or a complete Fierz basis.
    """)
    return


@app.cell
def _(mo):
    ordering = mo.ui.radio(
        {"Repeated pairs (default)": "repeated_pairs", "Canonical": "canonical"},
        value="Repeated pairs (default)",
        inline=True,
        label="Open-chain ordering",
    )
    ordering
    return (ordering,)


@app.cell
def _(
    E,
    display_settings,
    gm,
    gn,
    identity,
    lorentz,
    mo,
    mu,
    nu,
    ordering,
    word,
):
    anticommutator = word(gm, gn) + word(gn, gm) - 2 * lorentz.g(mu, nu) * identity
    assert anticommutator.simplify_algebra(
        contract="dots", gamma=True, epsilon=True
    ).to_expression() != E("0")
    assert anticommutator.simplify_algebra(
        contract="dots", gamma=True, gamma_ordering="canonical", epsilon=True
    ).to_expression() == E("0")
    ordering_result = anticommutator.simplify_algebra(
        contract="dots", gamma=True, gamma_ordering=ordering.value, epsilon=True
    )
    mo.vstack(
        [
            mo.Html(anticommutator.to_html(settings=display_settings)),
            mo.md("**Residual after the selected pass**"),
            mo.Html(ordering_result.to_html(settings=display_settings)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Four-dimensional gamma5 and projectors

    Gamma5 anticommutes through ordinary 4D gammas and squares to one;
    $P_\pm=(1\pm\gamma_5)/2$ obey projector identities. The internal epsilon
    normalization is fixed by
    $\mathrm{tr}(\gamma_5\gamma^\mu\gamma^\nu\gamma^\rho\gamma^\sigma)
    =4\,\epsilon(\mu,\nu,\rho,\sigma)$, **without an extra factor of i**.
    Convert conventions explicitly when comparing to other packages.
    """)
    return


@app.cell
def _(
    TensorExpression,
    g5,
    gm,
    gn,
    gr,
    gs,
    identity,
    identity_card,
    lorentz,
    minus,
    mo,
    mu,
    nu,
    plus,
    rho,
    sigma,
    tr,
    word,
):
    axial_expected = 4 * TensorExpression.levi_civita(lorentz)(mu, nu, rho, sigma)
    chiral_checks = mo.vstack(
        [
            identity_card("Gamma5 square", word(g5, g5), identity),
            identity_card("Gamma5 anticommutation", word(g5, gm), -word(gm, g5)),
            identity_card(
                "Axial four-gamma trace", tr(g5, gm, gn, gr, gs), axial_expected
            ),
            identity_card("Orthogonal projectors", word(plus, minus), 0),
            identity_card("Projector trace", tr(plus), 2),
        ],
        gap=2,
    )
    chiral_checks
    return


@app.cell
def _(display_settings, gm, gn, gr, mo, word):
    epsilon_settings = dict(gamma=True, gamma_expand_three_gamma_epsilon=True)
    epsilon_input = word(gm, gn, gr)
    epsilon_result = epsilon_input.simplify_algebra(
        contract="dots", **epsilon_settings, epsilon=True
    )
    assert (
        epsilon_input.simplify_algebra(contract="dots", gamma=True, epsilon=True)
        == epsilon_input
    )
    assert epsilon_result.to_expression() != epsilon_input.to_expression()
    assert (
        epsilon_result.simplify_algebra(
            contract="dots", **epsilon_settings, epsilon=True
        )
        == epsilon_result
    )
    mo.vstack(
        [
            mo.md(
                "**Optional three-gamma expansion** — disabled by default. This rewrites the word into metric terms and a gamma5–epsilon term; more terms need not mean a better result."
            ),
            mo.Html(epsilon_result.to_html(settings=display_settings)),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Charge conjugation and deliberate stopping points

    In the registered Weyl convention $C=-i\gamma^2\gamma^0$,
    $C^2=-\mathbf{1}$ and $C\gamma^\mu C=(\gamma^\mu)^T$.
    Charge conjugation can produce transposed words; ordinary Clifford rules
    do not treat those words as forward gammas. Mixed dimensions and
    D-dimensional gamma5 also remain explicit. The cards verify that behavior.
    """)
    return


@app.cell
def _(
    D,
    TensorExpression,
    a,
    b,
    chain_head,
    g5,
    gamma_head,
    gm,
    gn,
    identity,
    identity_card,
    incoming,
    lorentz,
    mo,
    mu,
    outgoing,
    sp,
    spin,
    word,
):
    _left, _right = spin(a).to_expression(), spin(b).to_expression()
    _c = sp.TensorName.charge_conjugation().to_expression()(incoming, outgoing)
    _forward = gamma_head(incoming, outgoing, lorentz(mu).to_expression())
    _transpose = gamma_head(outgoing, incoming, lorentz(mu).to_expression())
    _square = TensorExpression(chain_head(_left, _right, _c, _c))
    _sandwich = TensorExpression(chain_head(_left, _right, _c, _forward, _c))
    _transposed_word = TensorExpression(chain_head(_left, _right, _transpose, _forward))
    _mixed = word(gm, gn.with_lorentz_dimension(D), gm)
    _dimensional_g5 = word(g5, gm).with_lorentz_dimension(D)
    boundary_checks = mo.vstack(
        [
            identity_card("Charge-conjugation square", _square, -identity),
            identity_card(
                "Charge-conjugation sandwich",
                _sandwich,
                TensorExpression(chain_head(_left, _right, _transpose)),
            ),
            identity_card(
                "Transposed gamma word stays explicit",
                _transposed_word,
                _transposed_word,
            ),
            identity_card("Mixed Lorentz dimensions stay explicit", _mixed, _mixed),
            identity_card(
                "No D-dimensional gamma5 prescription", _dimensional_g5, _dimensional_g5
            ),
        ],
        gap=2,
    )
    boundary_checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Continue with measurements

    [Tensor reduction benchmarks](/?file=hep/tensor_benchmarks.py) contains trace-growth,
    contraction-order, FORM, and ladder comparisons with independent checks.
    """)
    return


if __name__ == "__main__":
    app.run()
