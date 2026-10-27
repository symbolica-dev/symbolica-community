"""Keep grouped tensor expressions on the notebook's rich display path."""

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from symbolica import E
from symbolica.community.tensor import Representation, TensorExpression, TensorName


def test_tensor_sum_factors_render_as_grouped_mathml():
    mink = Representation.mink(4)
    k = TensorName.vector("display_regression::k")
    p = TensorName.vector("display_regression::p")
    expression = (k(mink("mu")) + p(mink("mu"))).outer(k(mink("nu")) + p(mink("nu")))
    assert isinstance(expression, TensorExpression)
    original = expression.to_expression()

    bundle = expression._repr_mimebundle_(include=["text/html"])
    assert "text/html" in bundle, (
        "Rich rendering failed and would silently fall back to LaTeX"
    )
    html = bundle["text/html"]
    math_start = html.index("<math")
    math_end = html.index("</math>", math_start) + len("</math>")
    math = ET.fromstring(html[math_start:math_end])
    operators = [
        node.text for node in math.iter() if node.tag.rsplit("}", 1)[-1] == "mo"
    ]
    assert operators.count("(") == operators.count(")") == 2
    assert operators.count("+") == 2
    assert "overflow-x:auto" in html.replace(" ", "")
    assert expression.to_expression() == original


def test_color_chain_stack_keeps_mathml_without_embedded_fonts():
    mo = pytest.importorskip("marimo")
    widgets = pytest.importorskip("ipywidgets")
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
    html = stack.text
    # Marimo retains the rich display widgets; their MathML is in widget state.
    model_ids = re.findall(r"data-model-id='&quot;([^&]+)&quot;'", html)
    assert len(model_ids) == 3
    state = widgets.Widget.get_manager_state()["state"]
    presentations = [state[model_id]["state"] for model_id in model_ids]
    mathml = "".join(presentation["page"]["html"] for presentation in presentations)
    assert mathml.count("<math") == 3
    assert "<mi>" in mathml
    assert "data:font/" not in html
    assert "data:font/" not in mathml
    # The shared viewer loads the full math font externally once per document;
    # neither its source nor expression payloads may embed a base64 font.
    for presentation in presentations:
        viewer = presentation["_esm"]
        assert 'font-family:"STIX Two Math"' in viewer
        assert "font-display:swap" in viewer
        assert "data:font/" not in viewer
    assert sys.getsizeof(html) < 100_000
    assert sys.getsizeof(mathml) < 100_000
    assert [expression.to_expression() for expression in expressions] == originals


@pytest.mark.parametrize("source", ["(a+b)*(c+d)", "a*(b+c)", "(a+b*(c+d))*(e+f)"])
def test_scalar_tensor_rich_display_survives_symbolica_updates(source):
    expression = TensorExpression(E(source))
    original = expression.to_expression()
    html = expression.to_html()
    assert "data-spenso-math" in html
    assert "<math" in html
    bundle = expression._repr_mimebundle_(include=["text/html"])
    assert "<math" in bundle["text/html"]
    assert expression.to_expression() == original


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
