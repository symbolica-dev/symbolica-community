"""Selected-rule display uses Symbolica, with no coefficient-ID placeholders."""

import ast
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("symbolica")
pytest.importorskip("marimo")
from symbolica import E, S

HERE = Path(__file__).parents[1] / "examples/hep"
SPEC = importlib.util.spec_from_file_location("three_loop_display_support",
                                            HERE / "three_loop_reduction.py")
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


class Coefficients:
    def __init__(self):
        self.calls = []

    def coefficient(self, cid, **options):
        self.calls.append(cid)
        return {"numerator": "d-2+n0" if cid == 0 else "1", "denominator": "2*n1"}


def key(values, symbolic):
    return {"values": values, "symbolic": symbolic}


def rule(rhs):
    return {"target": key([1, 2], [True, False]), "rhs": rhs,
            "case": {"affine_zero_equations": []}, "excluded_all_zero_conjunctions": []}


def test_rule_expression_binds_parameters_indices_fixed_powers_and_guards():
    artifact, integral = Coefficients(), S("I_display_test")
    selected = rule([
        {"coefficient_id": 0, "integral": key([-1, 0], [True, False])},
        {"coefficient_id": 1, "integral": key([1, -2], [True, True])},
    ])
    selected["case"]["affine_zero_equations"] = [0]
    selected["excluded_all_zero_conjunctions"] = [[0, 1], []]
    dimension, n0, n1 = S("display_dimension", "n_0", "n_1")
    shown = support.rule_expressions(artifact, selected, integral,
                                     [(S("rustred::d"), dimension)])
    c0, c1 = (dimension - 2 + n0) / (2 * n1), 1 / (2 * n1)
    assert shown["target"] == integral(n0 + 1, 2)
    assert (shown["rhs"] - c0 * integral(n0 - 1, 0)
            - c1 * integral(n0 + 1, n1 - 2)).expand() == 0
    assert shown["affine"] == [c0]
    assert shown["excluded"] == [[c0, c1], []]
    assert artifact.calls == [0, 1]  # Decode repeated RHS/guard coefficients once.
    output = str(shown["rhs"].formatted(max_terms=None, max_line_length=None,
                                        terms_on_new_line=True))
    assert len(output.splitlines()) == 2
    assert "display_dimension" in output and "c_0" not in output


def test_zero_rhs_remains_zero_without_coefficient_fetches():
    artifact = Coefficients()
    shown = support.rule_expressions(artifact, rule([]), S("I_display_zero"))
    assert shown["rhs"] == 0
    assert str(shown["rhs"].formatted(terms_on_new_line=True)) == "0"
    assert artifact.calls == []


def test_full_rhs_is_not_truncated_at_ten_or_one_hundred_terms():
    selected = rule([{"coefficient_id": 1, "integral": key([i, 0], [False, False])}
                     for i in range(105)])
    artifact = Coefficients()
    rhs = support.rule_expressions(artifact, selected, S("I_display_many"))["rhs"]
    output = rhs.formatted(max_terms=None, max_line_length=None, terms_on_new_line=True)
    assert len(str(output).splitlines()) == 105
    assert str(output).count("I_display_many(") == 105
    assert "white-space: pre-wrap" in output._repr_html_()
    assert artifact.calls == [1]


def test_real_generated_k6_rule_displays_all_terms():
    from symbolica.community import hepkit as hep

    _, source = support.graph_inputs()
    session = hep.rustred.start_family_candidates(source, input_format="toml", n_cores=1)
    assert session.wait(timeout=30)
    artifact = session.result().artifact()
    sector = next(item for item in artifact.sectors(0, 100)["items"] if all(item["sector"]))
    selected = artifact.rule(sector["ordinal"], 0)
    assert len(selected["rhs"]) > 10
    expressions = support.rule_expressions(artifact, selected, S("I_k6_display"),
                                           [(S("rustred::d"), S("d"))])
    output = str(expressions["rhs"].formatted(max_terms=None, max_line_length=None,
                                              terms_on_new_line=True))
    assert len(output.splitlines()) == len(selected["rhs"])
    assert output.count("I_k6_display(") == len(selected["rhs"])
    assert "n_0" in output and "rustred::" not in output
    assert len(expressions["excluded"]) == len(selected["excluded_all_zero_conjunctions"])


def test_notebook_starts_once_in_setup_and_keeps_polling_separate():
    source = (HERE / "three_loop_reduction.py").read_text()
    tree = ast.parse(source)
    cells = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    starts = [cell for cell in cells if any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "run"
        and node.func.attr == "start" for node in ast.walk(cell))]
    assert len(starts) == 1
    assert "heartbeat" not in {arg.arg for arg in starts[0].args.args}
    assert 'label="Generate"' not in source
    assert "session = rustred.start_family_candidates(" in source
    assert "render_coefficient" not in source and "rhs_offset" not in source
    assert "terms_on_new_line=True" in source and "max_terms=None" in source


def test_notebook_has_no_local_helper_or_graph_file_dependencies(tmp_path, monkeypatch):
    source = (HERE / "three_loop_reduction.py").read_text()
    tree = ast.parse(source)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "three_loop_reduction_support" not in imports
    assert "rustred_campaign_support" not in imports
    monkeypatch.chdir(tmp_path)
    dot, family_source = support.graph_inputs()
    assert "digraph Mercedes" in dot
    assert support.tomllib.loads(family_source)["target"]["powers"] == [1] * 6


def test_inlined_native_fallback_supports_hosts_without_capability_query():
    run = support.ThreeLoopRun(object(), "unused")
    assert run.capabilities["background_sessions"] is True
    assert run.capabilities["live_event_polling"] is True
    assert run.poll()["state"] == "ready"
