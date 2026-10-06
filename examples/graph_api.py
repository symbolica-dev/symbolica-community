"""Interactive tour of the unified Community graph API."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", app_title="Community graph API")


@app.cell
def _():
    import marimo as mo
    from symbolica import E
    from symbolica.community import graph, hepkit, tensor

    return E, graph, hepkit, mo, tensor


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Community graph API

    Build, inspect, generate, and render graphs through **`symbolica.community.graph`**.
    The same `RenderSettings` works for graphs, subgraphs, Feynman diagrams,
    tensor networks, and amplitudes. Every `.render()` returns a `DiagramRender`.

    Change the controls, open the example tabs, or use **View code** to inspect
    this notebook. The diagrams follow your light/dark theme; hover their controls
    for zoom and pan help.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    layout_steps = mo.ui.slider(20, 160, step=20, value=40, label="Layout steps")
    node_radius = mo.ui.slider(0.04, 0.2, step=0.02, value=0.08, label="Vertex radius")
    stroke_width = mo.ui.slider(0.5, 3, step=0.5, value=1, label="Stroke width (pt)")
    accent = mo.ui.dropdown(
        {"Blue": "#528bc5", "Teal": "#23998e", "Amber": "#c98c36"},
        value="Blue",
        label="Vertex fill",
    )
    momentum = mo.ui.checkbox(value=False, label="Momentum labels and arrows")
    mo.vstack(
        [
            mo.md("## Shared rendering settings"),
            mo.hstack(
                [layout_steps, node_radius, stroke_width, accent, momentum],
                wrap=True,
                justify="start",
                gap=2,
            ),
        ]
    )
    return accent, layout_steps, momentum, node_radius, stroke_width


@app.cell
def _(accent, graph, layout_steps, node_radius, stroke_width):
    settings = graph.RenderSettings(
        layouts=graph.LayoutSettings(impred_steps=layout_steps.value),
        drawing=graph.DrawOptions(
            node_radius=node_radius.value,
            node_fill=graph.Color(accent.value),
            edge_stroke=graph.Stroke(thickness=graph.Length.pt(stroke_width.value)),
        ),
    )
    return (settings,)


@app.cell(hide_code=True)
def _(accent, layout_steps, mo, node_radius, stroke_width):
    mo.md(f"""
    ```python
    from symbolica.community import graph

    settings = graph.RenderSettings(
        layouts=graph.LayoutSettings(impred_steps={layout_steps.value}),
        drawing=graph.DrawOptions(
            node_radius={node_radius.value},
            node_fill=graph.Color("{accent.value}"),
            edge_stroke=graph.Stroke(thickness=graph.Length.pt({stroke_width.value})),
        ),
    )
    result = my_graph.render(config=settings)
    ```
    """)
    return


@app.cell
def _(graph):
    a = graph.node("a", data=0, label=graph.TextLabel("a"))
    b = graph.node("b", data=0, label=graph.TextLabel("b"))
    c = graph.node("c", data=0, label=graph.TextLabel("c"))
    triangle = graph.build(
        a,
        b,
        c,
        graph.edge(graph.source(a), "ab", graph.sink(b), data=1),
        graph.edge(graph.source(b), "bc", graph.sink(c), data=1),
        graph.edge(graph.source(c), "ca", graph.sink(a), data=1),
    )
    selected = triangle.subgraph(edges=[0, 1])
    return selected, triangle


@app.cell(hide_code=True)
def _(mo, selected, settings, triangle):
    # Give each reactive preview its own SVG viewer state.
    mo.vstack(
        [
            mo.md("## Graphs and live subgraphs"),
            mo.md(
                "`graph.build(...)` combines node and edge specifications. "
                "`triangle.subgraph(edges=[0, 1])` selects two edges of the same graph."
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Graph**"),
                            mo.iframe(triangle.render(config=settings).to_html()),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Subgraph**"),
                            mo.iframe(selected.render(config=settings).to_html()),
                        ]
                    ),
                ],
                widths="equal",
                align="start",
            ),
        ]
    )
    return


@app.cell
def _(E, hepkit, tensor):
    model = hepkit.Model.phi3()
    amplitude = model.process(["phi", "phi"], ["phi", "phi"]).generate_amplitude(
        loops=0,
        progress=None,
    )
    diagram = amplitude.diagrams[0]
    network = tensor.TensorNetwork(
        E(
            "g(mink(4,1),mink(4,2))*p(mink(4,1))*q(mink(4,2))",
            default_namespace="spenso",
        )
    )
    return amplitude, diagram, network


@app.cell
def _(amplitude, diagram, hepkit, momentum, network, settings):
    physics_style = hepkit.DiagramStyle(
        show_momentum=momentum.value,
        momentum_arrows=momentum.value,
    )
    diagram_render = diagram.render(config=settings, style=physics_style)
    network_render = network.render(config=settings)
    amplitude_render = amplitude.render(config=settings, style=physics_style)
    return amplitude_render, diagram_render, network_render


@app.cell(hide_code=True)
def _(amplitude_render, diagram_render, graph, mo, network_render):
    assert all(
        type(result) is graph.DiagramRender
        for result in (diagram_render, network_render, amplitude_render)
    )
    mo.vstack(
        [
            mo.md("## One render result across Community"),
            mo.md(
                "Physics-specific presentation stays in `hepkit.DiagramStyle`. "
                "The layout and drawing settings above apply to every tab."
            ),
            mo.ui.tabs(
                {
                    "FeynmanDiagram": mo.vstack(
                        [
                            mo.md(
                                "`diagram.render(config=settings, style=physics_style)`"
                            ),
                            mo.iframe(diagram_render.to_html()),
                        ]
                    ),
                    "TensorNetwork": mo.vstack(
                        [
                            mo.md("`network.render(config=settings)`"),
                            mo.iframe(network_render.to_html()),
                        ]
                    ),
                    "Amplitude": mo.vstack(
                        [
                            mo.md(
                                "`amplitude.render(config=settings, style=physics_style)`"
                            ),
                            mo.iframe(amplitude_render.to_html()),
                        ]
                    ),
                }
            ),
            mo.md(
                "All three are **`graph.DiagramRender`**. An amplitude's configured "
                "child snapshots are available through `result.diagrams`. "
                "`diagram.to_graph()` returns the same generic `graph.Graph` used above."
            ),
        ]
    )
    return


@app.cell
def _(graph):
    signature = graph.EdgeSignature(0)
    generated = graph.Graph.generate(
        [(index, signature) for index in range(4)],
        [[signature] * 3],
        max_loops=0,
    )
    canonical, vertex_map, group_size, orbits = generated[0][0].canonize()
    return canonical, generated, group_size, orbits, vertex_map


@app.cell(hide_code=True)
def _(canonical, generated, group_size, mo, orbits, settings, vertex_map):
    mo.vstack(
        [
            mo.md("""
        ## Generate and canonicalize

        ```python
        signature = graph.EdgeSignature(0)
        generated = graph.Graph.generate(
            [(i, signature) for i in range(4)], [[signature] * 3], max_loops=0,
        )
        canonical, vertex_map, group_size, orbits = generated[0][0].canonize()
        ```

        Generation returns `(Graph, automorphism_group_size)` pairs.
        `HalfEdge` is the live graph-element view; `EdgeSignature` describes a
        generation port. Custom payloads can supply `node_key`, `edge_key`,
        and `half_edge_key` to the graph algorithms.
        """),
            mo.ui.table(
                [
                    {
                        "Diagram": index + 1,
                        "Vertices": value.n_nodes,
                        "Edges": value.n_edges,
                        "Automorphism group size": size,
                    }
                    for index, (value, size) in enumerate(generated)
                ],
                selection=None,
            ),
            mo.hstack(
                [
                    mo.iframe(value.render(config=settings).to_html())
                    for value, _ in generated
                ],
                widths="equal",
                align="start",
            ),
            mo.md(
                f"First graph: canonical vertex map `{vertex_map}`, "
                f"group size `{group_size}`, orbits `{orbits}`. "
                f"Isomorphic to its canonical form: **{generated[0][0].is_isomorphic(canonical)}**."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    dot_input = mo.ui.text_area(
        value="digraph first { a -> b; b -> c; c -> a; }\n"
        "digraph second { a -> b; a -> b; }",
        label="DOT document",
        full_width=True,
    )
    mo.vstack(
        [
            mo.md("## Multiple graphs from DOT"),
            mo.md(
                "Edit the document to try triangles, parallel edges, or self-loops. "
                "`Graph.from_dot_set(source, graph.DotCodec.linnest())` returns "
                "one `Graph` per declaration, in document order."
            ),
            dot_input,
        ]
    )
    return (dot_input,)


@app.cell(hide_code=True)
def _(dot_input, graph, mo, settings):
    try:
        _parsed = graph.Graph.from_dot_set(dot_input.value, graph.DotCodec.linnest())
        _output = mo.vstack(
            [
                mo.md(f"**{len(_parsed)} graphs** parsed."),
                mo.hstack(
                    [
                        mo.iframe(value.render(config=settings).to_html())
                        for value in _parsed
                    ],
                    widths="equal",
                    align="start",
                ),
            ]
        )
    except ValueError as _error:
        _output = mo.callout(str(_error), kind="warn")
    _output
    return


@app.cell(hide_code=True)
def _(diagram_render, mo):
    mo.vstack(
        [
            mo.md("## Inspect and export the snapshot"),
            mo.md(
                "`DiagramRender` supports SVG, HTML, Typst, PDF, and PNG exports. "
                "It displays directly in IPython and Marimo."
            ),
            mo.hstack(
                [
                    mo.download(
                        diagram_render.to_svg().encode(),
                        filename="diagram.svg",
                        mimetype="image/svg+xml",
                        label="Download SVG",
                    ),
                    mo.download(
                        diagram_render.to_html().encode(),
                        filename="diagram.html",
                        mimetype="text/html",
                        label="Download HTML",
                    ),
                    mo.download(
                        diagram_render.to_linnest().encode(),
                        filename="diagram.typ",
                        mimetype="text/plain",
                        label="Download Typst",
                    ),
                ],
                justify="start",
            ),
            mo.accordion(
                {
                    "Prepared Typst source": mo.md(
                        "```typst\n" + diagram_render.typst_source + "\n```"
                    )
                }
            ),
            mo.md("""
        ```python
        result.save("diagram.pdf")
        result.save_pages("pages", format="png")
        pages = result.to_svg_pages()
        ```
        Multipage results expose every page through `to_svg_pages()`;
        `to_svg()` reports a clear error unless there is exactly one page.
        """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(graph, mo, triangle):
    original_settings = graph.RenderSettings(
        layouts=graph.LayoutSettings(impred_steps=20),
        drawing=graph.DrawOptions(node_fill=graph.Color("#528bc5")),
    )
    frozen_result = triangle.render(config=original_settings)
    original_svg = frozen_result.to_svg()
    original_settings.drawing = graph.DrawOptions(node_fill=graph.Color("#c98c36"))
    assert frozen_result.to_svg() == original_svg
    mo.vstack(
        [
            mo.md("## Render results are snapshots"),
            mo.md(
                "This blue snapshot was created before changing its settings to amber. "
                "The saved result remains blue; rendering again uses the new settings."
            ),
            mo.hstack(
                [
                    mo.iframe(frozen_result.to_html()),
                    mo.iframe(triangle.render(config=original_settings).to_html()),
                ],
                widths="equal",
                align="start",
            ),
            mo.callout(
                "Snapshot stability checked: later settings changes did not "
                "alter the existing result.",
                kind="success",
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
