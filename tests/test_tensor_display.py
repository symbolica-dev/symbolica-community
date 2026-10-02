"""Keep grouped tensor expressions on the notebook's rich display path."""

import sys
import xml.etree.ElementTree as ET

import pytest
from symbolica import E
from symbolica.community.tensor import Representation, TensorExpression, TensorName


def test_tensor_sum_factors_render_as_grouped_mathml():
    pytest.importorskip("typst")
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
    pytest.importorskip("typst")
    mo = pytest.importorskip("marimo")
    generator = TensorExpression.color_t(8, 3)
    explicit_word = generator("a", "i", "k") * generator("b", "k", "j")
    color_word = explicit_word.contract(
        representations=[Representation.cof(3)], metrics=False, rank_one=False
    ).to_expression()
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
    html = stack.text
    assert html.count("<math") == 3
    assert html.count("<div data-spenso-math>") == 3
    assert "<mi>" in html
    assert "data:font/" not in html
    # The current renderer references the full math font externally; notebook
    # output must stay small and must never embed a base64 font per expression.
    assert 'font-family:"STIX Two Math"' in html
    assert "font-display:swap" in html
    assert sys.getsizeof(html) < 100_000
    assert [expression.to_expression() for expression in expressions] == originals


@pytest.mark.parametrize("source", ["(a+b)*(c+d)", "a*(b+c)", "(a+b*(c+d))*(e+f)"])
def test_scalar_tensor_rich_display_survives_symbolica_updates(source):
    pytest.importorskip("typst")
    expression = TensorExpression(E(source))
    html = expression.to_html()
    assert "data-spenso-math" in html
    assert "<math" in html
    assert expression._repr_html_() is not None
