"""Automatic MathML paging must bound work and preserve the source expression."""

import json
import re
import sys
import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest
from symbolica import E, MathMLViewer, S
from symbolica.community.tensor import TensorExpression


def polynomial(n=301):
    return E("+".join(f"{i + 1}*page_test::x^{i}" for i in range(n)))


def assert_page(page, *, max_nodes=10_000):
    assert "error" not in page, page
    assert "Page rendering failed" not in page["html"], page["html"]
    assert len(page["html"].encode()) <= 256 * 1024
    math = re.findall(r"<math\b.*?</math>", page["html"], re.DOTALL)
    assert math
    assert sum(sum(1 for _ in ET.fromstring(m).iter()) for m in math) <= max_nodes


def action(pager, name, **parameters):
    field, page = pager.handle_message({"action": name, **parameters})
    assert field == "page" and "error" not in page, page
    return page


def math_nodes(html, tag):
    return [
        node
        for markup in re.findall(r"<math\b.*?</math>", html, re.DOTALL)
        for node in ET.fromstring(markup).iter()
        if node.tag.rsplit("}", 1)[-1] == tag
    ]


def sum_marker(page):
    # Responsive layouts retain their own omission capabilities. The compact
    # default may show only a product marker while a wider layout shows its sums.
    return next(
        (layout["layout"], path)
        for layout in [page, *page["layouts"]]
        for path, label in layout["markers"]
        if label.startswith("Inspect sum")
    )


class TensorPageSource:
    """Observable public provider for the shared viewer's cache/error checks."""

    def __init__(self, value):
        self.value = value
        self.calls = []

    def page(self, path, start, size, layout, expansions):
        assert path == [] and layout == 0 and expansions == []
        self.calls.append(start)
        viewer = self.value.paged(page_size=size)
        try:
            page = viewer.page()
            while page["start"] < start:
                page = action(viewer, "next")
            assert page["start"] == start
            return page
        finally:
            viewer.close()


def test_pages_are_contiguous_bounded_and_reversible():
    value = TensorExpression(polynomial())
    original = value.to_expression()
    pager = value.paged(page_size=100)
    pages = []
    while True:
        page = pager.page()
        assert_page(page)
        pages.append((page["start"], page["end"]))
        if page["next"] is None:
            break
        action(pager, "next")
    assert pages[0][0] == 0 and pages[-1][1] == 301
    assert all(a[1] == b[0] for a, b in zip(pages, pages[1:]))
    for start, end in reversed(pages[:-1]):
        page = action(pager, "previous")
        assert (page["start"], page["end"]) == (start, end)
    assert value.to_expression() == original
    pager.close()
    with pytest.raises(RuntimeError, match="viewer is closed"):
        pager.page()


def test_nested_sums_keep_factors_and_subexpression_navigation():
    left, right = (
        polynomial(201),
        polynomial(203).replace(S("page_test::x"), S("page_test::y")),
    )
    value = TensorExpression(7 * left * right)
    before = value.to_expression()
    pager = value.paged()
    page = pager.page()
    assert_page(page)
    assert page["kind"] == "Product" and page["total"] == 3
    assert "<mn>7</mn>" in page["html"]
    layout, target = sum_marker(page)
    child = action(pager, "open", value=target, layout=layout)
    assert_page(child)
    assert child["kind"] == "Sum" and child["total"] in (201, 203)
    assert child["back"]
    action(pager, "size", value=100)
    parent = action(pager, "back")
    assert parent["page_size"] == 25
    assert parent["html"] == page["html"]
    assert value.to_expression() == before


def test_automatic_and_formatted_outputs_are_bounded():
    value = TensorExpression(polynomial(2001))
    assert value.paged().page()["end"] == 25
    assert len(str(value)) < 200
    for output in (value, value.formatted()):
        bundle = output._repr_mimebundle_(include=["text/plain", "text/html"])
        html = bundle["text/html"]
        assert len(html.encode()) < 270_000
        assert "Navigation requires a live notebook" in html
        assert "items 1–25 of 2001" in html
        assert set(bundle) == {"text/plain", "text/html"}
        assert len(bundle["text/plain"]) < 200
    assert value.to_expression() == polynomial(2001)


def test_export_and_small_output_keep_existing_behavior():
    small = TensorExpression(E("a+b"))
    html = small._repr_mimebundle_(include=["text/html"])["text/html"]
    assert "omitted" not in html
    for tag in ("mi", "mo", "mn"):
        assert [n.text for n in math_nodes(html, tag)] == [
            n.text for n in math_nodes(small.to_html(), tag)
        ]
    large = TensorExpression(polynomial(101))
    assert "Navigation requires" not in large.to_html()
    assert len(large.to_html()) > 1000


def test_missing_widget_dependency_static_preview_and_disposal():
    pager = TensorExpression(polynomial()).paged()
    with patch.dict(sys.modules, {"anywidget": None}):
        bundle = pager._repr_mimebundle_()
    assert set(bundle) == {"text/plain", "text/html"}
    widget = pager._get_widget()
    assert len(widget.page["html"].encode()) <= 256 * 1024
    pager._on_message(widget, {"action": "attach", "view": "test"}, [])
    pager._on_message(widget, {"action": "dispose", "view": "test"}, [])
    assert pager._widget is None
    # Disposal drops frontend state; a later display can recreate the same page.
    assert_page(pager.page())
    pager.close()
    with pytest.raises(RuntimeError, match="viewer is closed"):
        pager.page()


def test_renderer_failure_does_not_fall_back_to_full_output():
    source = TensorPageSource(TensorExpression(polynomial()))
    pager = MathMLViewer(source)
    first = pager.page()
    with patch.object(source, "page", side_effect=TimeoutError("deadline")):
        field, page = pager.handle_message({"action": "next"})
    assert field == "page" and page["error"] == "deadline"
    assert page["html"] == first["html"]
    assert (page["start"], page["end"]) == (first["start"], first["end"])
    assert len(json.dumps(page).encode()) < 256 * 1024
    assert action(pager, "next")["start"] == first["end"]


def test_page_budget_shrinks_and_rejects_invalid_navigation():
    # Enough work per term must exhaust the 64 KiB initial preview budget
    # before its requested 25 terms. Explicit pages use a larger bounded budget.
    value = TensorExpression(
        E(
            "+".join(
                f"page_budget::f({','.join(f'a{i}_{j}' for j in range(256))})"
                for i in range(1001)
            )
        )
    )
    original = value.to_expression()
    pager = value.paged()
    page = pager.page()
    assert_page(page)
    assert 0 < page["end"] < 25 and page["total"] == 1001
    for layout in page["layouts"]:
        assert_page(layout)
    for message in (
        {"action": "size", "value": 1000000},
        {"action": "open", "value": [123456789]},
    ):
        field, failed = pager.handle_message(message)
        assert field == "page" and failed["error"]
        assert failed["html"] == page["html"]
    assert pager.page()["html"] == page["html"]
    following = action(pager, "next")
    assert_page(following)
    assert following["start"] == page["end"]
    assert 0 < following["end"] - following["start"] < 25
    assert action(pager, "previous")["html"] == page["html"]
    assert value.to_expression() == original
    pager.close()


def test_explicit_large_page_uses_a_larger_bounded_budget():
    value = TensorExpression(
        E(
            "+".join(
                f"page_large_budget::f({','.join(f'a{i}_{j}' for j in range(8))})"
                for i in range(1001)
            )
        )
    )
    original = value.to_expression()
    pager = value.paged()
    first = pager.page()
    assert_page(first)
    assert first["end"] == 25
    larger = action(pager, "size", value=500)
    assert (larger["start"], larger["end"], larger["total"]) == (0, 500, 1001)
    # The 256 KiB transport ceiling still applies. For this eight-argument
    # fixture, also bound MathML structure by 32 elements per displayed term.
    for layout in [larger, *larger["layouts"]]:
        assert_page(layout, max_nodes=500 * 32 + 16)
    assert action(pager, "size", value=25)["html"] == first["html"]
    assert value.to_expression() == original
    pager.close()


def test_shared_page_cache_reuses_24_renders_and_evicts_the_oldest():
    source = TensorPageSource(TensorExpression(polynomial(651)))
    # One layout makes each requested page exactly one cache entry. These short
    # pages stay below the separate 2 MiB bound, isolating the entry limit.
    pager = MathMLViewer(source, page_size=25, layout_count=1)
    first = pager.page()
    pages = [first]
    for _ in range(24):
        pages.append(action(pager, "next"))
    assert sum(len(json.dumps(page).encode()) for page in pages) < 1024 * 1024
    starts = list(range(0, 601, 25))
    assert source.calls == starts
    for _ in range(23):
        action(pager, "previous")
    assert source.calls == starts
    assert action(pager, "previous")["html"] == first["html"]
    assert source.calls == [*starts, 0]
    assert action(pager, "next")["start"] == 25
    assert source.calls == [*starts, 0]
    pager.close()


def test_oversized_function_has_bounded_argument_navigation():
    # A single term exceeds the node budget; entering it pages its arguments.
    value = TensorExpression(
        E("page_test::f(" + ",".join(f"a{i}" for i in range(3000)) + ")")
    )
    before = value.to_expression()
    pager = value.paged(page_size=100)
    page = pager.page()
    assert_page(page)
    assert page["unit"] == "arguments" and page["total"] == 3000
    assert page["end"] <= 100
    assert value.to_expression() == before
    page = action(pager, "next")
    assert_page(page)
    assert page["start"] == 100


def test_signs_and_tensor_ports_survive_pages():
    from symbolica.community.tensor import Representation, TensorName

    rep = Representation.mink(4)
    tensor = TensorName("page_test::T")
    # Fix the logical interface before selecting any sum fragment.
    value = sum(
        (-1) ** i * E(f"page_test::z^{i}") * tensor(rep("mu"), rep("nu"))
        for i in range(130)
    )
    before = value.to_expression()
    pager = value.paged(page_size=25)
    first = pager.page()
    assert_page(first)
    assert {"mu", "nu"} <= {n.text for n in math_nodes(first["html"], "mi")}
    assert "−" in first["html"]
    second = action(pager, "next")
    assert_page(second)
    assert {"mu", "nu"} <= {n.text for n in math_nodes(second["html"], "mi")}
    assert value.to_expression() == before
    assert action(pager, "previous")["html"] == first["html"]


def test_serialized_page_never_contains_undisplayed_terms():
    value = TensorExpression(E("+".join(f"hidden_term_{i:05d}" for i in range(6001))))
    payload = value.paged(page_size=25).page()
    assert payload["end"] == 25 and payload["total"] == 6001
    serialized = json.dumps(payload)
    assert len(serialized.encode()) < 64 * 1024
    # Inspect the actual transport payload: hidden expression trees are private.
    assert "tree" not in payload
    displayed = set(re.findall(r"hidden_term_\d{5}", serialized))
    assert len(displayed) == 25


def test_jupyter_bundle_contains_live_widget_and_bounded_fallbacks():
    value = TensorExpression(polynomial(1001))
    for output in (value, value.formatted()):
        bundle = output._repr_mimebundle_()
        assert "application/vnd.jupyter.widget-view+json" in bundle
        assert len(bundle["text/plain"]) < 200
        assert len(bundle["text/html"].encode()) < 270_000


def test_page_source_can_serve_jupyter_comm_thread():
    from concurrent.futures import ThreadPoolExecutor

    pager = TensorExpression(polynomial(301)).paged()
    with ThreadPoolExecutor(max_workers=1) as worker:
        page = worker.submit(pager.page).result(timeout=15)
    assert_page(page)


def test_three_rung_ladder_pages_keep_factorization_and_budgets():
    """The 6,001-term example that formerly produced 380,000 MathML nodes."""
    from symbolica import Graph
    from symbolica.community.hepkit import Model

    target = Graph()
    for _ in range(8):
        target.add_node(0)
    target.add_node(-1)
    target.add_node(2)
    for i in range(8):
        target.add_edge(i, (i + 1) % 8, data=21)
    for i, j in ((1, 7), (2, 6), (3, 5), (0, 8), (4, 9)):
        target.add_edge(i, j, data=21)
    target = target.canonize()[0]

    def accept(graph, completed):
        if completed < len(graph):
            return True
        if len(graph) != 10 or graph.num_edges() != 13:
            return False
        for edge in range(graph.num_edges()):
            graph.set_directed(edge, False)
        return graph.canonize()[0] == target

    model = Model.qcd()
    diagram = model.process(["g"], ["g"], vertex_allow=["V_36"]).generate_diagrams(
        loops=4, max_vertices=8, filter=accept
    )[0]
    dimension = S("paging_ladder::D")
    numerator = TensorExpression(
        model.expand_couplings(
            diagram.numerator_expression(in_lmb=True).to_expression()
        )
    ).with_lorentz_dimension(dimension)
    slots = [slot.dual() for slot in numerator.structure.slots()]
    projector = (
        TensorExpression.g(slots[0], slots[1])
        * TensorExpression.g(slots[2], slots[3])
        / 8
    )
    value = (
        (projector * numerator)
        .simplify_algebra(
            contract="dots", color_substitute_cof_dimension_invariants=True
        )
        .expand()
        .collect_factors()
        .collect_num()
    )
    original = value.to_expression()
    pager = value.paged()
    outline = pager.page()
    assert_page(outline)
    assert outline["kind"] == "Product"
    layout, target = sum_marker(outline)
    selected = next(
        candidate
        for candidate in [outline, *outline["layouts"]]
        if candidate["layout"] == layout
    )
    end = next(
        start
        for (path, _), start in zip(selected["markers"], selected["marker_starts"])
        if path == target
    )
    child = action(pager, "open", value=target, layout=layout)
    assert child["start"] == end
    for _ in range(8):
        page = pager.page()
        assert_page(page)
        assert page["total"] == 6001
        assert page["start"] == end
        assert 0 < page["end"] - page["start"] <= 25
        end = page["end"]
        action(pager, "next")
    assert action(pager, "back")["html"] == outline["html"]
    assert value.to_expression() == original


def test_generated_index_names_survive_cache_eviction():
    from symbolica.community.tensor import TensorName, TensorStructure

    TensorName("page_indices::v")
    TensorName("page_indices::w")
    left, right, rep = S("page_indices::v", "page_indices::w", "spenso::mink")
    expression = sum(
        left(rep(4, S(f"page_indices::i{i}", tags=["spenso::dummy-index"])))
        * right(rep(4, S(f"page_indices::i{i}", tags=["spenso::dummy-index"])))
        for i in range(130)
    )
    # Establish the scalar interface before display, preserving the dummy names.
    value = TensorExpression(expression, structure=TensorStructure([]))
    original = value.to_expression()
    pager = value.paged(page_size=25)
    first = pager.page()["html"]
    for _ in range(4):
        action(pager, "next")
    for _ in range(4):
        action(pager, "previous")
    assert pager.page()["html"] == first
    assert value.to_expression() == original


def test_one_render_oversized_term_becomes_an_inspectable_placeholder():
    value = TensorExpression(
        E("page_test::f(page_test::g(" + ",".join(f"a{i}" for i in range(3000)) + "))")
    )
    original = value.to_expression()
    pager = value.paged()
    page = pager.page()
    assert_page(page)
    assert page["total"] == 1 and page["markers"]
    assert "data-symbolica-marker" in page["html"]
    child = action(pager, "open", value=page["markers"][0][0])
    assert_page(child)
    assert child["unit"] == "arguments" and child["total"] == 3000
    assert child["start"] == page["marker_starts"][0]
    assert 0 < child["end"] - child["start"] <= 25
    assert action(pager, "back")["html"] == page["html"]
    assert value.to_expression() == original


def test_two_expanded_binomials_have_a_compact_outline_and_separate_sum():
    value = TensorExpression((E("x+1") ** 100).expand() * (E("y+1") ** 100).expand())
    original = value.to_expression()
    viewer = value.paged()
    for size in (100, 250):
        page = action(viewer, "size", value=size)
        assert_page(page)
        assert page["kind"] == "Product" and page["total"] == 2
        context = next(
            layout
            for layout in [page, *page["layouts"]]
            if len(layout["markers"]) == 2
            and all(label.startswith("Inspect sum") for _, label in layout["markers"])
        )
        assert_page(context)
        # Each factor remains a separate bounded sum with its own navigation.
        assert [n.text for n in math_nodes(context["html"], "mo")].count("(") == 2
        assert {"x", "y"} <= {n.text for n in math_nodes(context["html"], "mi")}
        for target, label in context["markers"]:
            assert label.startswith("Inspect sum")
            child = action(viewer, "open", value=target, layout=context["layout"])
            assert_page(child)
            assert child["kind"] == "Sum" and child["total"] == 101
            assert len({n.text for n in math_nodes(child["html"], "mi")}) == 1
            assert action(viewer, "back")["html"] == page["html"]
        assert value.to_expression() == original


def test_dot_product_power_keeps_compact_tensor_notation():
    from symbolica.community.tensor import Representation, TensorName, dot

    rep = Representation.mink(4)
    left = TensorName.vector("page_power::k")(rep)
    right = TensorName.vector("page_power::p")(rep)
    value = dot(left, right) ** 2
    page = value.paged().page()
    assert_page(page)
    # Dot operands are vector labels; the exponent applies to their contraction.
    powers = math_nodes(page["html"], "msup")
    assert len(powers) == 1
    base, exponent = powers[0]
    assert "".join(base.itertext()) == "(k·p)"
    assert "".join(exponent.itertext()) == "2"
    assert not math_nodes(page["html"], "mmultiscripts")
