"""Keep grouped tensor expressions on the notebook's rich display path."""

import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from symbolica import E, Expression, MathMLViewer, PrintMode, S
from symbolica.community.hepkit import Model
from symbolica.community.tensor import Representation, TensorExpression, TensorName


def test_tensor_sum_factors_render_as_grouped_mathml():
    mink = Representation.mink(4)
    k = TensorName.vector("display_regression::k")
    p = TensorName.vector("display_regression::p")
    expression = (k(mink("mu")) + p(mink("mu"))).outer(k(mink("nu")) + p(mink("nu")))
    assert isinstance(expression, TensorExpression)
    original = expression.to_expression()

    html = expression._repr_html_()
    assert html is not None, (
        "Rich rendering failed and would silently fall back to LaTeX"
    )
    math_start = html.index("<math")
    math_end = html.index("</math>", math_start) + len("</math>")
    math = ET.fromstring(html[math_start:math_end])
    operators = [
        node.text for node in math.iter() if node.tag.rsplit("}", 1)[-1] == "mo"
    ]
    assert operators.count("(") == operators.count(")") == 2
    assert operators.count("+") == 2
    assert "overflow-x: auto" in html
    assert expression.to_expression() == original


def test_color_chain_stack_keeps_mathml_without_embedded_fonts():
    mo = pytest.importorskip("marimo")
    generator = TensorExpression.color_t(8, 3)
    explicit_word = generator("a", "i", "k") * generator("b", "k", "j")
    color_word = explicit_word.contract(
        representations=[Representation.cof(3)], metrics=False, rank_one=False
    )
    color_conjugate = color_word.dirac_adjoint()
    expressions = [explicit_word, color_word, color_conjugate]
    originals = [expression.to_expression() for expression in expressions]

    stack = mo.vstack(
        [
            mo.md("**Explicit indexed network**"),
            explicit_word,
            mo.md("**Collected chain**"),
            color_word,
            mo.md("**Complex conjugate**"),
            color_conjugate,
        ]
    )
    # Marimo mounts each shared widget; its model carries the bounded MathML page.
    assert stack.text.count("<marimo-anywidget") == 3
    viewers = [expression.paged() for expression in expressions]
    try:
        html = "".join(viewer._page()["html"] for viewer in viewers)
    finally:
        for viewer in viewers:
            viewer.close()
    assert html.count("<math") == 3
    assert html.count('<math data-spenso-math=""') == 3
    assert "<mi>" in html
    assert "data:font/" not in html
    # The shared viewer registers the external font once in the document.
    # Individual expression pages must not embed or register another font.
    javascript = MathMLViewer.javascript()
    assert '@font-face{font-family:"STIX Two Math"' in javascript
    assert "font-display:swap" in javascript
    assert "@font-face" not in html
    assert sys.getsizeof(html) < 100_000
    assert [expression.to_expression() for expression in expressions] == originals


@pytest.mark.parametrize(
    ("source", "case"),
    [
        (
            (
                "spenso::dot(gammalooprs::ϵ(0,spenso::mink(4)),"
                "gammalooprs::ϵbar(1,spenso::mink(4)))"
            ),
            "dot",
        ),
        ("gammalooprs::ϵbar(1,spenso::mink(4,mu))", "abstract"),
        ("gammalooprs::ϵbar(1,spenso::cind(2))", "component"),
    ],
)
def test_graph_polarization_bar_keeps_indices_in_static_and_paged_output(source, case):
    # Register the same graph symbols used by the four-loop numerator notebook.
    Model.qcd()
    expression = TensorExpression(E(source))
    original, structure = expression.to_expression(), expression.structure
    viewer = expression.paged()
    try:
        outputs = (expression.to_html(), viewer.page()["html"])
    finally:
        viewer.close()

    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        accents = list(math.iter("mover"))
        assert len(accents) == 1
        assert (
            unicodedata.normalize("NFKC", "".join(accents[0][0].itertext())).strip()
            == "ε"
        )
        assert "bar" not in "".join(math.itertext())
        script_tag = "msup" if case == "component" else "msubsup"
        scripted_bar = [
            node for node in math.iter(script_tag) if accents[0] in node[0].iter()
        ]
        assert len(scripted_bar) == 1
        scripted_bar = scripted_bar[0]
        if case == "component":
            assert "1" in "".join(scripted_bar[0].itertext())
            assert "".join(scripted_bar[1].itertext()).strip() == "2"
        else:
            assert "1" in "".join(scripted_bar[1].itertext())
            if case == "abstract":
                assert "mu" in "".join(scripted_bar[2].itertext())
            else:
                assert "0" in "".join(math.itertext())
                assert "⋅" in "".join(math.itertext())
    assert expression.to_expression() == original
    assert expression.structure == structure


@pytest.mark.parametrize(
    ("source", "head", "object_index", "tensor_indices"),
    [
        ("gammalooprs::Q(27,spenso::mink(4,mu))", "q", "27", {"mu"}),
        ("gammalooprs::Q(27,spenso::cind(2))", "q", "27", {"2"}),
        ("gammalooprs::K(13,spenso::mink(4,mu))", "k", "13", {"mu"}),
        ("gammalooprs::K(13,spenso::cind(2))", "k", "13", {"2"}),
        ("gammalooprs::ϵbar(1,spenso::mink(4,mu))", "ε", "1", {"mu"}),
        ("gammalooprs::ϵbar(1,spenso::cind(2))", "ε", "1", {"2"}),
        (
            "spenso::f(spenso::coad(8,a),spenso::coad(8,b),spenso::coad(8,c))",
            "f",
            None,
            {"a", "b", "c"},
        ),
    ],
)
def test_raw_graph_expression_custom_mathml_keeps_its_indices(
    source, head, object_index, tensor_indices
):
    Model.qcd()
    expression = E(source)
    assert type(expression) is Expression
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        fragment = expression.format(mode=PrintMode.MathML)
        outputs = (f"<math>{fragment}</math>", viewer.page()["html"])
    finally:
        viewer.close()

    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        tokens = {
            unicodedata.normalize("NFKC", node.text or "").strip()
            for node in math.iter()
            if node.tag in ("mi", "mn", "mtext")
        }
        assert head in tokens
        assert not tokens.intersection({"Q", "K", "εbar", "mink", "coad", "cind"})
        if head == "ε":
            assert len(list(math.iter("mover"))) == 1
        if object_index is not None:
            assert any(
                "".join(node[1].itertext()).strip() == object_index
                for node in math.iter("msub")
            )
        upper_indices = {
            "".join(script.itertext()).strip()
            for node in math.iter("mmultiscripts")
            for script in list(node)[2::2]
        }
        assert tensor_indices <= upper_indices
    assert type(expression) is Expression
    assert expression.__getstate__() == original


@pytest.mark.parametrize("power", [1, 2])
def test_raw_polarization_dot_uses_custom_heads_and_groups_powers(power):
    Model.qcd()
    expression = (
        E(
            "spenso::dot(gammalooprs::ϵ(0,spenso::mink(4)),"
            "gammalooprs::ϵbar(1,spenso::mink(4)))"
        )
        ** power
    )
    assert type(expression) is Expression
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        assert len(list(math.iter("mover"))) == 1
        text = "".join(math.itertext())
        assert "bar" not in text and "dot" not in text and "mink" not in text
        assert {"".join(node[1].itertext()).strip() for node in math.iter("msub")} == {
            "0",
            "1",
        }
        assert any(node.text in ("·", "⋅") for node in math.iter("mo"))
        if power == 2:
            square = next(
                node
                for node in math.iter("msup")
                if "".join(node[1].itertext()).strip() == "2"
            )
            operators = [node.text for node in square[0].iter("mo")]
            assert operators[0] == "(" and operators[-1] == ")"
    assert expression.__getstate__() == original


def test_raw_custom_mathml_keeps_ordinary_function_arguments():
    Model.qcd()
    expression = E("raw_display_regression::ordinary(a,7)")
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        text = "".join(math.itertext())
        assert all(token in text for token in ("ordinary", "a", "7", "(", ")"))
    assert type(expression) is Expression
    assert expression.__getstate__() == original


def test_raw_dot_preserves_a_complete_custom_mathml_printer():
    Model.qcd()

    def printer(expression, *, mode, **options):
        return "<mi>CUSTOM</mi>" if mode == PrintMode.MathML else None

    custom = S(
        "raw_display_regression::CompleteVector",
        tags=["spenso::tensor", "spenso::rank1", "spenso::print-callback"],
        print=printer,
    )
    representation = S("spenso::mink")(4)
    expression = S("spenso::dot")(
        custom(1, representation), S("gammalooprs::Q")(2, representation)
    )
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        text = "".join(math.itertext())
        # A complete printer owns its argument's display. The compact dot rule
        # must defer instead of replacing that output with the vector's name.
        assert "CUSTOM" in text
        assert "CompleteVector" not in text
        assert "dot" in text
    assert type(expression) is Expression
    assert expression.__getstate__() == original


def test_raw_dot_groups_a_custom_head_before_attaching_the_object_index():
    Model.qcd()
    primed = S(
        "raw_display_regression::PrimedVector",
        tags=[
            "spenso::tensor",
            "spenso::rank1",
            "spenso::print-head:mathml:<mi>A</mi><mo>′</mo>",
        ],
    )
    representation = S("spenso::mink")(4)
    expression = S("spenso::dot")(
        primed(1, representation), S("gammalooprs::Q")(2, representation)
    )
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        scripted = [
            node for node in math.iter("msub") if "A" in "".join(node[0].itertext())
        ]
        assert len(scripted) == 1
        assert len(scripted[0]) == 2
        head, index = scripted[0]
        assert head.tag == "mrow" and len(head) == 2
        assert [node.text for node in head] == ["A", "′"]
        assert "".join(index.itertext()).strip() == "1"
    assert type(expression) is Expression
    assert expression.__getstate__() == original


@pytest.mark.parametrize("kind", ["tensor", "native"])
@pytest.mark.parametrize("case", ["single", "square", "product"])
def test_scalar_dot_factors_are_parenthesized_and_products_are_juxtaposed(kind, case):
    Model.qcd()
    left = E(
        "spenso::dot(gammalooprs::ϵ(0,spenso::mink(4)),"
        "gammalooprs::ϵbar(1,spenso::mink(4)))"
    )
    right = E(
        "spenso::dot(gammalooprs::Q(2,spenso::mink(4)),"
        "gammalooprs::K(3,spenso::mink(4)))"
    )
    expression = {"single": left, "square": left**2, "product": 7 * left * right}[case]
    original = expression.__getstate__()
    tensor = TensorExpression(expression)
    structure = tensor.structure
    viewer = tensor.paged() if kind == "tensor" else expression.mathml_viewer()
    try:
        static = (
            tensor.to_html()
            if kind == "tensor"
            else f"<math>{expression.format(mode=PrintMode.MathML)}</math>"
        )
        outputs = (static, viewer.page()["html"])
    finally:
        viewer.close()

    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        tokens = [(node.text or "").strip() for node in math.iter()]
        dot_count = 2 if case == "product" else 1
        assert sum(token in ("·", "⋅") for token in tokens) == dot_count
        assert "×" not in tokens and "*" not in tokens
        fences = 0 if kind == "native" and case == "single" else dot_count
        assert tokens.count("(") == tokens.count(")") == fences
        assert len(list(math.iter("mover"))) == 1
        if case == "square":
            square = next(
                node
                for node in math.iter("msup")
                if "".join(node[1].itertext()).strip() == "2"
            )
            base = [(node.text or "").strip() for node in square[0].iter()]
            assert base.count("(") == base.count(")") == 1
            assert sum(token in ("·", "⋅") for token in base) == 1
        if case == "product":
            assert "7" in tokens
            parent_of = {child: parent for parent in math.iter() for child in parent}
            for dot in (
                node for node in math.iter() if (node.text or "").strip() in ("·", "⋅")
            ):
                parent = parent_of[dot]
                while parent is not math:
                    leaves = [
                        node
                        for node in parent.iter()
                        if len(node) == 0 and (node.text or "").strip()
                    ]
                    texts = [(node.text or "").strip() for node in leaves]
                    if texts[0] == "(" and texts[-1] == ")":
                        assert sum(text in ("·", "⋅") for text in texts) == 1
                        break
                    parent = parent_of[parent]
                else:
                    pytest.fail("A scalar dot factor is missing its own parentheses")
    assert type(expression) is Expression
    assert expression.__getstate__() == original
    assert tensor.to_expression().__getstate__() == original
    assert tensor.structure == structure


def test_native_color_generator_keeps_upper_and_dual_indices():
    Model.qcd()
    expression = E(
        "spenso::t(spenso::coad(8,a),spenso::cof(3,i),spenso::dind(spenso::cof(3,j)))"
    )
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        indexed = list(math.iter("mmultiscripts"))
        assert len(indexed) == 1
        head, *scripts = indexed[0]
        assert "".join(head.itertext()).strip() == "t"
        assert ["".join(node.itertext()).strip() for node in scripts[::2]] == [
            "",
            "",
            "j",
        ]
        assert ["".join(node.itertext()).strip() for node in scripts[1::2]] == [
            "a",
            "i",
            "",
        ]
    assert type(expression) is Expression
    assert expression.__getstate__() == original


@pytest.mark.parametrize(
    ("source", "lower", "upper", "squared"),
    [
        ("spenso::g(spenso::coad(8,a),spenso::coad(8,b))", [], ["a", "b"], False),
        (
            "spenso::g(spenso::dind(spenso::cof(3,i)),spenso::dind(spenso::cof(3,j)))",
            ["i", "j"],
            [],
            False,
        ),
        (
            "spenso::g(spenso::cof(3,i),spenso::dind(spenso::cof(3,j)))",
            ["j"],
            ["i"],
            False,
        ),
        (
            "7*spenso::g(spenso::coad(8,a),spenso::coad(8,b))^2",
            [],
            ["a", "b"],
            True,
        ),
    ],
)
def test_native_metric_matches_typed_indices_and_groups_powers(
    source, lower, upper, squared
):
    Model.qcd()
    expression = E(source)
    original = expression.__getstate__()
    tensor = TensorExpression(expression)
    structure = tensor.structure
    native_viewer, tensor_viewer = expression.mathml_viewer(), tensor.paged()
    try:
        outputs = (
            (True, f"<math>{expression.format(mode=PrintMode.MathML)}</math>"),
            (True, native_viewer.page()["html"]),
            (False, tensor.to_html()),
            (False, tensor_viewer.page()["html"]),
        )
    finally:
        native_viewer.close()
        tensor_viewer.close()
    for native, html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        for node in math.iter():
            for child in list(node):
                if child.tag == "mphantom":
                    node.remove(child)
        metrics = [
            node
            for node in math.iter()
            if node.tag in ("mmultiscripts", "msubsup")
            and unicodedata.normalize("NFKC", "".join(node[0].itertext())).strip()
            == "g"
        ]
        assert len(metrics) == 1
        metric = metrics[0]
        if metric.tag == "mmultiscripts":
            rows = (list(metric)[1::2], list(metric)[2::2])
        else:
            rows = ([metric[1]], [metric[2]])
        for row, expected in zip(rows, (lower, upper)):
            indices = [
                unicodedata.normalize("NFKC", "".join(node.itertext())).strip()
                for node in row
            ]
            assert "".join("".join(indices).split()) == "".join(expected)
        tokens = {
            unicodedata.normalize("NFKC", node.text or "").strip()
            for node in math.iter()
        }
        assert not tokens.intersection({"coad", "cof", "dind", "δ", "η"})
        if squared:
            square = next(
                node
                for node in math.iter("msup")
                if "".join(node[1].itertext()).strip() == "2"
            )
            assert metric in square[0].iter()
            assert "7" in tokens
            assert not tokens.intersection({"·", "⋅", "×", "*"})
            if native:
                operators = [node.text for node in square[0].iter("mo")]
                assert operators == ["(", ")"]
    assert type(expression) is Expression
    assert expression.__getstate__() == original
    assert tensor.to_expression().__getstate__() == original
    assert tensor.structure == structure


@pytest.mark.parametrize(
    ("source", "family", "dimension", "index", "dual"),
    [
        ("spenso::coad(8)", "coad", "8", None, False),
        ("spenso::mink(D)", "mink", "D", None, False),
        ("spenso::mink(4,mu)", "mink", "4", "mu", False),
        ("spenso::dind(spenso::cof(3,i))", "cof", "3", "i", True),
    ],
)
def test_native_standalone_representation_keeps_family_dimension_and_variance(
    source, family, dimension, index, dual
):
    Model.qcd()
    expression = E(source)
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        spaces = [
            node
            for node in math.iter("msub")
            if "".join(node[0].itertext()).strip() == family
        ]
        assert len(spaces) == 1
        assert "".join(spaces[0][1].itertext()).strip() == dimension
        assert "dind" not in "".join(math.itertext())
        assert not any(node.text in ("(", ")") for node in math.iter("mo"))
        dual_markers = [node for node in math.iter("msup") if node[1].text == "*"]
        assert len(dual_markers) == int(dual)
        slots = list(math.iter("mmultiscripts"))
        assert len(slots) == int(index is not None)
        if index is not None:
            lower, upper = slots[0][1:]
            assert "".join((lower if dual else upper).itertext()).strip() == index
            assert (upper if dual else lower).tag == "none"
    assert type(expression) is Expression
    assert expression.__getstate__() == original


@pytest.mark.parametrize(
    ("source", "arguments"),
    [
        (
            "spenso::g(spenso::coad(8,a),spenso::coad(7,b))",
            {"coad", "8", "7", "a", "b"},
        ),
        (
            "spenso::g(spenso::coad(8,a,b),metric_display_control::coad(8,c))",
            {"coad", "8", "a", "b", "c"},
        ),
    ],
)
def test_native_metric_defers_incompatible_or_malformed_slots(source, arguments):
    Model.qcd()
    expression = E(source)
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        tokens = {(node.text or "").strip() for node in math.iter()}
        assert {"g", "(", ")", ","} | arguments <= tokens
        assert not any(
            "".join(node[0].itertext()).strip() == "g"
            for node in math.iter("mmultiscripts")
        )
    assert type(expression) is Expression
    assert expression.__getstate__() == original


def test_native_color_index_prints_its_stored_expression_without_renaming():
    Model.qcd()
    payload = E("gammalooprs::hedge(17,2)")
    index = S(
        "native_index_data_regression::cooked_original",
        tags=["spenso::index"],
        data=payload,
    )
    assert type(index.get_symbol_data()) is Expression
    assert index.get_symbol_data() == payload
    coad = S("spenso::coad")
    expression = S("spenso::f")(
        coad(8, index), coad(8, S("index_a")), coad(8, S("index_b"))
    )
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        outputs = (
            f"<math>{expression.format(mode=PrintMode.MathML)}</math>",
            viewer.page()["html"],
        )
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        for node in math.iter():
            node.tag = node.tag.rsplit("}", 1)[-1]
        scripts = list(math.iter("mmultiscripts"))
        assert len(scripts) == 1
        assert "".join(scripts[0][0].itertext()).strip() == "f"
        upper = scripts[0][2]
        tokens = {(node.text or "").strip() for node in upper.iter()}
        assert {"h", "17", "2"} <= tokens
        assert "cooked_original" not in "".join(math.itertext())
        assert "hedge" not in "".join(math.itertext())
    assert type(expression) is Expression
    assert expression.__getstate__() == original
    assert index.get_symbol_data() == payload
    assert "cooked_original" in str(index)


def test_custom_mathml_labels_do_not_reuse_native_navigation_paths():
    Model.qcd()
    arguments = ",".join(f"x{index}" for index in range(1, 41))
    expression = E(f"F(H({arguments}),gammalooprs::Q(g(h(y)),spenso::mink(4,mu)))")
    original = expression.__getstate__()
    viewer = expression.mathml_viewer()
    try:
        for layout in range(8):
            page = viewer._render_page(layout)
            start = page["html"].index("<math")
            end = page["html"].index("</math>", start) + len("</math>")
            math = ET.fromstring(page["html"][start:end])
            for node in math.iter():
                node.tag = node.tag.rsplit("}", 1)[-1]
            labels = list(math.iter("mmultiscripts"))
            assert len(labels) == 1
            # Q's complete custom label contains g(h(y)); its locally rooted
            # paths must not collide with native H's enclosing path [0].
            assert not any(
                "data-symbolica-path" in node.attrib for node in labels[0].iter()
            )
            native_h = [
                node for node in math.iter() if node.get("data-symbolica-path") == "0"
            ]
            assert len(native_h) == 1
            assert "H" in {(node.text or "").strip() for node in native_h[0].iter("mi")}
            if page["markers"]:
                assert [0] in [path for path, _ in page["markers"]]
    finally:
        viewer.close()
    assert type(expression) is Expression
    assert expression.__getstate__() == original


@pytest.mark.parametrize("vector", [False, True])
def test_tensor_name_does_not_install_a_general_native_printer(vector):
    constructor = TensorName.vector if vector else TensorName
    name = constructor(f"native_registration_guard::Fresh{int(vector)}")
    value = name(Representation.mink(4)("mu"))
    expression = value.to_expression()
    original = expression.__getstate__()
    before = expression.format(mode=PrintMode.MathML)
    viewer = value.paged()
    try:
        assert "data-spenso-math" in viewer.page()["html"]
        assert "<math" in value.to_html()
    finally:
        viewer.close()
    after = expression.format(mode=PrintMode.MathML)
    assert before == after
    assert "mink" in after and "mu" in after
    assert type(expression) is Expression
    assert expression.__getstate__() == original


@pytest.mark.parametrize("kind", ["tensor", "native"])
def test_scalar_multiplication_uses_juxtaposition_in_static_and_paged_mathml(kind):
    expression = E("juxtaposition_regression::a*juxtaposition_regression::b")
    original = expression.__getstate__()
    tensor = TensorExpression(expression)
    viewer = tensor.paged() if kind == "tensor" else expression.mathml_viewer()
    try:
        static = (
            tensor.to_html()
            if kind == "tensor"
            else f"<math>{expression.format(mode=PrintMode.MathML)}</math>"
        )
        outputs = (static, viewer.page()["html"])
    finally:
        viewer.close()
    for html in outputs:
        start = html.index("<math")
        end = html.index("</math>", start) + len("</math>")
        math = ET.fromstring(html[start:end])
        tokens = {
            unicodedata.normalize("NFKC", node.text or "").strip()
            for node in math.iter()
        }
        assert {"a", "b"} <= tokens
        assert not tokens.intersection({"·", "⋅", "×", "*"})
    assert expression.__getstate__() == original
    assert tensor.to_expression().__getstate__() == original


@pytest.mark.parametrize("source", ["(a+b)*(c+d)", "a*(b+c)", "(a+b*(c+d))*(e+f)"])
def test_scalar_tensor_rich_display_survives_symbolica_updates(source):
    expression = TensorExpression(E(source))
    html = expression.to_html()
    assert "data-spenso-math" in html
    assert "<math" in html
    assert expression._repr_html_() is not None


def test_rendering_without_python_rendering_packages():
    # The standalone check asserts a clean import registry. Other tests exercise
    # native Linnet interoperability and legitimately import its Python package.
    # Keep the no-rendering-dependency assertions in a fresh interpreter.
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("check_offline_rendering.py"))],
        check=True,
        capture_output=True,
        text=True,
    )
