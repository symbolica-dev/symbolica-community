import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="DOT → Feynman graph")


with app.setup(hide_code=True):
    import anywidget
    import marimo as mo
    import traitlets
    from symbolica import S
    from symbolica.community.hepkit import FeynkitError, FeynmanDiagram, Model
    from symbolica.community.tensor import TensorExpression

    class DotEditor(anywidget.AnyWidget):
        value = traitlets.Unicode().tag(sync=True)
        _esm = r"""
        import {EditorState} from "https://esm.sh/@codemirror/state@6.7.5";
        import {EditorView, keymap, lineNumbers, drawSelection, rectangularSelection,
                highlightActiveLine} from "https://esm.sh/@codemirror/view@6.43.12?deps=@codemirror/state@6.7.5";
        import {defaultKeymap, history, historyKeymap, indentWithTab,
                addCursorAbove, addCursorBelow} from "https://esm.sh/@codemirror/commands@6.11.1?deps=@codemirror/state@6.7.5,@codemirror/view@6.43.12,@codemirror/language@6.12.4";
        import {StreamLanguage, bracketMatching, syntaxHighlighting,
                defaultHighlightStyle} from "https://esm.sh/@codemirror/language@6.12.4?deps=@codemirror/state@6.7.5,@codemirror/view@6.43.12";
        import {closeBrackets, closeBracketsKeymap, autocompletion,
                completionKeymap} from "https://esm.sh/@codemirror/autocomplete@6.20.3?deps=@codemirror/state@6.7.5,@codemirror/view@6.43.12,@codemirror/language@6.12.4";
        import {searchKeymap, selectNextOccurrence,
                selectSelectionMatches} from "https://esm.sh/@codemirror/search@6.7.1?deps=@codemirror/state@6.7.5,@codemirror/view@6.43.12,@codemirror/language@6.12.4";

        const keywords = new Set(["strict", "graph", "digraph", "subgraph", "node", "edge"]);
        const attributes = new Set(["particle", "lmb_id", "label", "style", "shape",
            "color", "fontcolor", "penwidth", "weight", "pos", "dir", "arrowhead",
            "arrowtail", "rank", "rankdir"]);
        const dot = StreamLanguage.define({
            startState: () => ({comment: false, quoted: false, html: 0, depth: 0}),
            token(stream, state) {
                if (state.comment) {
                    if (stream.skipTo("*/")) {stream.match("*/"); state.comment = false;}
                    else stream.skipToEnd();
                    return "comment";
                }
                if (state.quoted) {
                    let ch;
                    while ((ch = stream.next()) !== undefined) {
                        if (ch === "\\") stream.next();
                        else if (ch === '"') {state.quoted = false; break;}
                    }
                    return "string";
                }
                if (state.html) {
                    let ch;
                    while ((ch = stream.next()) !== undefined) {
                        if (ch === "<") state.html++;
                        if (ch === ">" && --state.html === 0) break;
                    }
                    return "string";
                }
                if (stream.eatSpace()) return null;
                if (stream.match("//") || stream.match("#")) {
                    stream.skipToEnd(); return "comment";
                }
                if (stream.match("/*")) {state.comment = true; return "comment";}
                if (stream.match('"')) {state.quoted = true; return "string";}
                if (stream.match("<")) {state.html = 1; return "string";}
                if (stream.match(/^(->|--)/)) return "operator";
                if (stream.match(/^-?(?:\.\d+|\d+(?:\.\d*)?)/)) return "number";
                if (stream.match(/^[_a-zA-Z\u0080-\uffff][\w\u0080-\uffff]*/)) {
                    const word = stream.current().toLowerCase();
                    return keywords.has(word) ? "keyword" :
                        attributes.has(word) ? "propertyName" : "variableName";
                }
                const ch = stream.next();
                if (ch === "{" || ch === "[") state.depth++;
                if (ch === "}" || ch === "]") state.depth = Math.max(0, state.depth - 1);
                return "punctuation";
            },
            indent: (state, after) => Math.max(0, state.depth - (/^\s*[}\]]/.test(after) ? 1 : 0)) * 4,
            languageData: {commentTokens: {line: "//", block: {open: "/*", close: "*/"}}}
        });

        function render({model, el}) {
            el.classList.add("dot-editor");
            el.dataset.language = "dot";
            const label = document.createElement("div");
            label.textContent = "DOT source";
            label.className = "dot-editor-label";
            const host = document.createElement("div");
            el.append(label, host);
            let timer;
            const sync = () => {model.set("value", view.state.doc.toString()); model.save_changes();};
            const view = new EditorView({
                parent: host,
                state: EditorState.create({doc: model.get("value"), extensions: [
                    dot, syntaxHighlighting(defaultHighlightStyle), bracketMatching(),
                    lineNumbers(), highlightActiveLine(), drawSelection(), rectangularSelection(),
                    EditorState.allowMultipleSelections.of(true), history(), closeBrackets(),
                    autocompletion({override: [context => {
                        const word = context.matchBefore(/[\w]*/);
                        if (!word || (!context.explicit && word.from === word.to)) return null;
                        return {from: word.from, options: [...keywords, ...attributes].map(label =>
                            ({label, type: keywords.has(label) ? "keyword" : "property"}))};
                    }]}),
                    keymap.of([
                        {key: "Mod-d", run: selectNextOccurrence},
                        {key: "Mod-Shift-l", run: selectSelectionMatches},
                        {key: "Ctrl-Alt-ArrowUp", run: addCursorAbove},
                        {key: "Ctrl-Alt-ArrowDown", run: addCursorBelow},
                        ...closeBracketsKeymap, ...completionKeymap, ...historyKeymap,
                        ...searchKeymap, ...defaultKeymap, indentWithTab
                    ]),
                    EditorView.contentAttributes.of({"aria-label": "DOT source"}),
                    EditorView.domEventHandlers({keydown: event => {event.stopPropagation(); return false;}}),
                    EditorView.updateListener.of(update => {
                        if (update.docChanged) {clearTimeout(timer); timer = setTimeout(sync, 400);}
                    })
                ]})
            });
            const changed = () => {
                const value = model.get("value");
                if (value !== view.state.doc.toString()) {
                    clearTimeout(timer);
                    view.dispatch({changes: {from: 0, to: view.state.doc.length, insert: value}});
                }
            };
            model.on("change:value", changed);
            return () => {clearTimeout(timer); model.off("change:value", changed); view.destroy();};
        }
        export default {render};
        """
        _css = """
        .dot-editor {min-width: 0; width: 100%;}
        .dot-editor-label {font-weight: 500; margin-bottom: .4rem;}
        .dot-editor .cm-editor {border: 1px solid #8886; border-radius: 6px;}
        .dot-editor .cm-scroller {min-height: 320px; max-height: 600px; overflow: auto;}
        .dot-editor .cm-content {font-family: ui-monospace, monospace; font-size: 14px;}
        """


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # DOT → Feynman graph

    Edit or paste a graph on the left to update the Feynman diagram on the right.
    Choose the model whose particles appear in your DOT. For example,
    `particle="phi"` selects the scalar particle in the default model.

    Below the graph, build and simplify its numerator using the same momentum
    routing, Lorentz dimension, and algebra operations as the four-loop notebook.

    [Browse notebooks](/) · [Integral families](/?file=hep/integral_families.py)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.accordion(
        {
            "DOT conventions": mo.md("""
        ## Setup and notebook helpers

        Mark external vertices with `style=invis`. Optional `lmb_id=0`,
        `lmb_id=1`, … annotations choose the loop-momentum basis; enable
        momentum labels to inspect the routing. Particle names must belong
        to the selected model; expand the list below the editor to find them.

        Use Ctrl/Cmd+D to select the next occurrence, Ctrl/Cmd+Shift+L to
        select all matches, or Alt-click to add a cursor. Ctrl+Alt+Up/Down
        adds cursors on adjacent lines. Ctrl+Space opens DOT completions.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def _():
    model_choice = mo.ui.dropdown(
        {"Scalar φ³ + φ⁴": "scalar", "QCD": "qcd", "Standard Model": "standard"},
        value="Scalar φ³ + φ⁴",
        label="Model",
        allow_select_none=False,
    )
    show_momenta = mo.ui.checkbox(label="Show momentum labels")
    mo.hstack([model_choice, show_momenta], justify="start", align="center")
    return model_choice, show_momenta


@app.cell(hide_code=True)
def _(model_choice):
    if model_choice.value == "standard":
        model = Model.standard_model()
    elif model_choice.value == "qcd":
        model = Model.qcd()
    else:
        model = Model.phi_3_4()
    return (model,)


@app.cell(hide_code=True)
def _():
    dot_source = mo.ui.anywidget(
        DotEditor(
            value="""digraph sunrise {
    ext [style=invis];
    ext -> a [particle="phi"];
    a -> b [particle="phi", lmb_id=0];
    a -> b [particle="phi", lmb_id=1];
    a -> b [particle="phi"];
    b -> ext [particle="phi"];
}""",
        )
    )
    return (dot_source,)


@app.cell
def _(dot_source, model):
    diagram = None
    parse_error = None
    if dot_source.value["value"].strip():
        try:
            diagram = FeynmanDiagram.from_dot(
                model, dot_source.value["value"]
            ).apply_feynman_rules()
            diagram.validate()
        except FeynkitError as _error:
            diagram = None
            parse_error = str(_error)
    return diagram, parse_error


@app.cell
def _(diagram, show_momenta):
    drawing = None
    render_error = None
    if diagram is not None:
        try:
            drawing = diagram.render(momenta=show_momenta.value)
        except (FeynkitError, ValueError) as _error:
            render_error = str(_error)
    return drawing, render_error


@app.cell(hide_code=True)
def _(diagram, dot_source, drawing, model, parse_error, render_error):
    if parse_error is not None or render_error is not None:
        _preview = mo.callout(
            mo.vstack(
                [
                    mo.md(
                        "**Could not parse the graph.**"
                        if parse_error is not None
                        else "**Could not render the graph.**"
                    ),
                    mo.plain_text(
                        parse_error if parse_error is not None else render_error
                    ),
                ]
            ),
            kind="danger",
        )
    elif drawing is None:
        _preview = mo.callout(
            "Paste a DOT graph to see its Feynman diagram.", kind="info"
        )
    else:
        _preview = mo.vstack(
            [
                mo.md(
                    f"**Loops:** {diagram.loop_count} · **Vertices:** {len(diagram.vertices)} · **Edges:** {len(diagram.edges)}"
                ),
                mo.iframe(drawing.to_html(), height="420px"),
                mo.hstack(
                    [
                        mo.download(
                            drawing.to_svg(),
                            filename="diagram.svg",
                            mimetype="image/svg+xml",
                            label="Download SVG",
                        ),
                        mo.download(
                            diagram.to_dot(),
                            filename="diagram.dot",
                            mimetype="text/plain",
                            label="Download DOT",
                        ),
                    ],
                    justify="start",
                ),
            ]
        )
    mo.vstack(
        [
            mo.hstack(
                [
                    dot_source.style({"min-width": "0", "overflow": "auto"}),
                    _preview.style({"min-width": "0", "overflow": "auto"}),
                ],
                widths=[1, 1],
                align="start",
            ),
            mo.accordion(
                {
                    "Particle names in the selected model": mo.md(
                        ", ".join(
                            f"`{_particle.name}`" for _particle in model.particles
                        )
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build the numerator and projector

    The selected model supplies the Feynman rules from the particle assignments
    in DOT. Use the graph's loop-momentum routing before contraction, keeping
    momentum differences compact. Color and Lorentz factors stay together;
    couplings, propagator phases, and the graph weight are retained.

    The cells below follow the
    [four-loop numerator notebook](/?file=hep/four_loop_numerator.py).
    For two external gluons, apply $g_{\mu\nu}\delta_{ab}/8$ directly to
    the external ports. Other graphs keep their external indices open.
    """)
    return


@app.cell
def _(diagram, model):
    mo.stop(diagram is None)
    graph_weight = diagram.overall_factor_expression(evaluate=True)
    _massless = model.expand_couplings(
        diagram.numerator_expression(in_lmb=True) * graph_weight
    )
    numerator = _massless.with_lorentz_dimension(S("D")).collect_factors()
    numerator.structure
    return (numerator,)


@app.cell
def _(diagram, numerator):
    _slots = numerator.structure.slots()
    projector = TensorExpression(1)
    if (
        len(diagram.external_edges) == 2
        and all(_edge.particle_name == "g" for _edge in diagram.external_edges)
        and len(_slots) == 4
    ):
        projector = (
            TensorExpression.g(_slots[0], _slots[1])
            * TensorExpression.g(_slots[2], _slots[3])
            / 8
        )
    projector
    return (projector,)


@app.cell
def _(numerator, projector):
    projected_numerator = projector * numerator
    projected_numerator
    return (projected_numerator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Perform simple contractions that do not generate new terms
    """)
    return


@app.cell
def _(projected_numerator):
    projected_numerator.contract().to_dots()
    return


@app.cell
def _(projected_numerator):
    projected_numerator.simplify_algebra(
        contract="minimal", color=True, gamma=False
    ).to_dots()
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Contract to dot products

    Reduce the projected tensor in one call, including color and Dirac identities
    and Lorentz contractions. The order of contractions is optimized.
    """)
    return


@app.cell
def _(projected_numerator):
    reduced = projected_numerator.simplify_algebra(contract="dots")
    reduced
    return (reduced,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    Number of terms after expanding the numerator:
    """)
    return


@app.cell
def _(reduced):
    len(reduced.expand().to_expression())
    return


if __name__ == "__main__":
    app.run()
