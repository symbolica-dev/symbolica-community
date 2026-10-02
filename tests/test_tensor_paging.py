"""Automatic MathML paging must bound work and preserve the source expression."""

import re
import sys
import xml.etree.ElementTree as ET
from unittest.mock import patch

import pytest
from symbolica import E, S
from symbolica.community.tensor import TensorExpression


def polynomial(n=301):
    return E("+".join(f"{i + 1}*page_test::x^{i}" for i in range(n)))


def assert_page(page):
    assert "Page rendering failed" not in page["html"], page["html"]
    assert len(page["html"].encode()) <= 256 * 1024
    math = re.findall(r"<math\b.*?</math>", page["html"], re.DOTALL)
    assert math
    assert sum(sum(1 for _ in ET.fromstring(m).iter()) for m in math) <= 10_000


def test_pages_are_contiguous_bounded_and_reversible():
    value = TensorExpression(polynomial())
    original = value.to_expression()
    pager = value.paged(page_size=100)
    pages = []
    while True:
        page = pager._page()
        assert_page(page)
        pages.append((page["start"], page["end"]))
        if page["next"] is None:
            break
        pager._action({"action": "next"})
    assert pages[0][0] == 0 and pages[-1][1] == 301
    assert all(a[1] == b[0] for a, b in zip(pages, pages[1:]))
    assert len(pager._cache) <= 3
    for start, end in reversed(pages[:-1]):
        page = pager._action({"action": "previous"})
        assert (page["start"], page["end"]) == (start, end)
    assert value.to_expression() == original
    pager.close()
    assert not pager._cache and pager._source is None


def test_nested_sums_keep_factors_and_subexpression_navigation():
    left, right = (
        polynomial(201),
        polynomial(203).replace(S("page_test::x"), S("page_test::y")),
    )
    value = TensorExpression(7 * left * right)
    before = value.to_expression()
    pager = value.paged()
    page = pager._page()
    assert_page(page)
    assert page["total"] in (201, 203)
    assert "<mn>7</mn>" in page["html"]
    assert "⋯" in page["html"]
    target = next(t for t, _ in page["holes"] if t != 0)
    child = pager._action({"action": "open", "value": target})
    assert_page(child)
    assert child["breadcrumbs"]
    pager._action({"action": "size", "value": 100})
    parent = pager._action({"action": "back"})
    assert parent["page_size"] == 25
    assert parent["html"] == page["html"]
    assert value.to_expression() == before


def test_automatic_and_formatted_outputs_are_bounded():
    value = TensorExpression(polynomial(2001))
    assert value.paged()._page()["end"] == 25
    assert len(str(value)) < 200
    for output in (value, value.formatted()):
        html = output._repr_html_()
        assert len(html.encode()) < 270_000
        assert "Navigation requires a live notebook" in html
        assert "Terms 1–25 of 2001" in html
        bundle = output._repr_mimebundle_(include=["text/plain", "text/html"])
        assert set(bundle) == {"text/plain", "text/html"}
        assert len(bundle["text/plain"]) < 200
    assert value.to_expression() == polynomial(2001)


def test_export_and_small_output_keep_existing_behavior():
    small = TensorExpression(E("a+b"))
    assert small._repr_html_() == small.to_html()
    assert "omitted" not in small._repr_html_()
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
    assert pager._widget is None and not pager._cache


def test_renderer_failure_does_not_fall_back_to_full_output():
    pager = TensorExpression(polynomial()).paged()
    import _spenso_paging

    with patch.object(
        _spenso_paging, "compile_page", side_effect=TimeoutError("deadline")
    ):
        page = pager._page()
    assert "Page rendering failed" in page["html"]
    assert len(page["html"]) < 300


def test_page_budget_shrinks_and_cache_is_limited():
    pager = TensorExpression(polynomial(1001)).paged(page_size=500)
    import _spenso_paging

    real = _spenso_paging.compile_page
    seen = []

    def limited(payload):
        seen.append(payload["end"] - payload["start"])
        return None if seen[-1] > 25 else real(payload)

    with patch.object(_spenso_paging, "compile_page", side_effect=limited):
        page = pager._page()
    assert_page(page)
    assert len(seen) > 1 and page["end"] <= 25
    with pytest.raises(ValueError):
        pager._action({"action": "size", "value": 1000000})
    with pytest.raises(ValueError):
        pager._action({"action": "open", "value": 123456789})


def test_oversized_function_has_bounded_argument_navigation():
    # A single term exceeds the node budget; entering it pages its arguments.
    value = TensorExpression(
        E("page_test::f(" + ",".join(f"a{i}" for i in range(3000)) + ")")
    )
    before = value.to_expression()
    pager = value.paged(page_size=100)
    page = pager._page()
    assert_page(page)
    assert page["unit"] == "Arguments" and page["total"] == 3000
    assert page["end"] <= 100
    assert value.to_expression() == before
    page = pager._action({"action": "next"})
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
    first = pager._page()
    assert_page(first)
    assert "<mtext>mu</mtext>" in first["html"] and "<mtext>nu</mtext>" in first["html"]
    assert "−" in first["html"]
    second = pager._action({"action": "next"})
    assert_page(second)
    assert (
        "<mtext>mu</mtext>" in second["html"] and "<mtext>nu</mtext>" in second["html"]
    )
    assert value.to_expression() == before
    assert pager._action({"action": "previous"})["html"] == first["html"]


def test_serialized_page_never_contains_undisplayed_terms():
    pager = TensorExpression(polynomial(6001)).paged(page_size=25)
    payload = pager._source.page(0, None, 25)
    assert payload["end"] == 25 and payload["total"] == 6001
    assert len(payload["tree"]) < 64 * 1024
    # The render tree has no copies of the other 5,976 terms.
    assert payload["tree"].count(b"kind") < 200


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
        page = worker.submit(pager._page).result(timeout=15)
    assert_page(page)


def test_three_rung_ladder_pages_keep_factorization_and_budgets():
    """The 6,001-term example that formerly produced 380,000 MathML nodes."""
    from symbolica import Graph
    from symbolica.community.hep import Model
    from symbolica.community.tensor import Representation

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
    projector = (
        TensorExpression.g(Representation.mink(dimension))
        * TensorExpression.g(Representation.coad(8))
        / 8
    ).index(*(slot.dual() for slot in numerator.structure.slots))
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
    end = 0
    for _ in range(8):
        page = pager._page()
        assert_page(page)
        assert page["total"] == 6001
        assert page["start"] == end
        assert 0 < page["end"] - page["start"] <= 25
        assert len(pager._cache) <= 3
        end = page["end"]
        pager._action({"action": "next"})
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
    first = pager._page()["html"]
    for _ in range(4):
        pager._action({"action": "next"})
    assert len(pager._cache) == 3
    for _ in range(4):
        pager._action({"action": "previous"})
    assert pager._page()["html"] == first
    assert value.to_expression() == original


def test_one_render_oversized_term_becomes_an_inspectable_placeholder():
    pager = TensorExpression(E("page_test::f(a+b+c)")).paged()
    import _spenso_paging

    with patch.object(_spenso_paging, "compile_page", return_value=None):
        page = pager._page()
    assert "exceeds the display budget" in page["html"]
    assert page["holes"]
    child = pager._action({"action": "open", "value": page["holes"][0][0]})
    assert_page(child)


def test_two_expanded_binomials_have_a_compact_outline_and_separate_sum():
    value = TensorExpression((E("x+1") ** 100).expand() * (E("y+1") ** 100).expand())
    original = value.to_expression()
    viewer = value.paged()
    for size in (100, 250):
        page = viewer._action({"action": "size", "value": size})
        assert_page(page)
        assert page["selection"] == "Sum [1]"
        assert page["total"] == 101
        assert len(page["holes"]) == 2
        math = re.findall(r"<math\b.*?</math>", page["html"], re.DOTALL)
        assert len(math) == 2
        # The context keeps both factors; only its tiny markers are fenced.
        context = ET.fromstring(math[0])
        assert [n.text for n in context.iter("mo")].count("(") == 2
        assert "[1]" in math[0] and "[2]" in math[0]
        assert "(" not in [n.text for n in ET.fromstring(math[1]).iter("mo")]
        assert "Expression structure" in page["html"]
        assert value.to_expression() == original


def test_dot_product_power_keeps_compact_tensor_notation():
    from symbolica.community.tensor import Representation, TensorName, dot

    rep = Representation.mink(4)
    left = TensorName.vector("page_power::k")(rep)
    right = TensorName.vector("page_power::p")(rep)
    value = dot(left, right) ** 2
    page = value.paged()._page()
    assert_page(page)
    math = ET.fromstring(
        re.search(r"<math\b.*?</math>", page["html"], re.DOTALL).group()
    )
    # Dot operands are vector labels; the exponent applies to their contraction.
    assert "(" not in [n.text for n in math.iter("mo")]
    assert ")" not in [n.text for n in math.iter("mo")]
    assert any(n.tag == "msup" for n in math.iter())
