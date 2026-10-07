"""Selected-rule display uses Symbolica, with no coefficient-ID placeholders."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    artifact = hep.rustred.family_candidates(
        source, input_format="toml", n_cores=1,
        exact_backend="sparse", numerical_depth=2,
    ).artifact()
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


def test_notebook_has_sector_then_rule_dropdowns_with_no_offset_selector():
    source = (HERE / "three_loop_reduction.py").read_text()
    tree = ast.parse(source)
    labels = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dropdown"):
            labels.extend(keyword.value.value for keyword in node.keywords
                          if keyword.arg == "label" and isinstance(keyword.value, ast.Constant))
    assert "Sector" in labels and "Rule" in labels
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert names.isdisjoint({"rule_offset", "rule_table", "rhs_offset"})
    assert "render_coefficient" not in source
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


def summary(target, fixed=(), affine_count=0, ordinal=0):
    return {"ordinal": ordinal, "target": target,
            "case": {"fixed": list(fixed), "affine_equation_count": affine_count,
                     "kind": "generic"},
            "rhs_terms": 12, "retained_source_count": 3, "guard_count": 0}


def test_rule_label_omits_fixed_case_already_visible_in_target():
    selected = summary(key([0, 2], [True, False]), [{"axis": 1, "value": 2}], ordinal=7)
    assert support.rule_case_summary(selected) == ""
    label = support.rule_label(selected)
    assert support.integral_notation(selected["target"]) in label
    assert "7" in label and "n_1 = 2" not in label


def test_rule_label_preserves_case_not_encoded_by_target():
    selected = summary(key([1, 2], [True, False]),
                       [{"axis": 0, "value": 1}, {"axis": 1, "value": 2}])
    case = support.rule_case_summary(selected)
    assert "n_0 = 1" in case and "n_1 = 2" not in case
    assert case in support.rule_label(selected)


def test_rule_case_keeps_affine_restrictions_even_with_fixed_target():
    selected = summary(key([1, 2], [False, False]),
                       [{"axis": 0, "value": 1}, {"axis": 1, "value": 2}],
                       affine_count=2)
    case = support.rule_case_summary(selected)
    assert "2" in case and "affine" in case.lower()
    assert "n_0 = 1" not in case and "n_1 = 2" not in case
    assert case in support.rule_label(selected)


def test_rule_ordinals_disambiguate_equal_targets():
    target = key([0, 0], [True, True])
    assert (support.rule_label(summary(target, ordinal=3))
            != support.rule_label(summary(target, ordinal=4)))


def test_full_rule_case_keeps_affine_conditions_without_summary_count():
    selected = summary(key([0, 2], [True, False]), [{"axis": 1, "value": 2}])
    selected["case"].pop("affine_equation_count")
    selected["case"]["affine_zero_equations"] = [7]
    case = support.rule_case_summary(selected)
    assert "1" in case and "affine" in case.lower()
    assert "n_1 = 2" not in case


def notebook_cell_producing(name):
    """Exercise a notebook cell without starting its other scientific work."""
    tree = ast.parse((HERE / "three_loop_reduction.py").read_text())
    cells = [cell for cell in tree.body if isinstance(cell, ast.FunctionDef) and any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
        for node in ast.walk(cell))]
    assert len(cells) == 1
    cell = cells[0]
    cell.decorator_list = []
    namespace = vars(support).copy()
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(SPEC.origin), "exec"), namespace)
    return namespace[cell.name]


def test_rule_dropdown_loads_all_summaries_but_only_selected_rhs():
    summaries = [summary(key([0, 2], [True, False]), ordinal=index) for index in range(1013)]

    class Artifact:
        def __init__(self):
            self.summary_calls, self.rule_calls = [], []

        def rules(self, sector, *, start, limit):
            assert sector == 14
            self.summary_calls.append((start, limit))
            return {"items": summaries[start:start + limit], "total": len(summaries)}

        def rule(self, sector, ordinal, **options):
            self.rule_calls.append((sector, ordinal))
            return {"ordinal": ordinal, "rhs": ["selected-rule-only"]}

    artifact, sector = Artifact(), SimpleNamespace(value=14)
    picker, = notebook_cell_producing("rule_choice")(
        candidate_artifact=artifact, sector_choice=sector)
    assert len(artifact.summary_calls) > 1
    assert set(picker.options.values()) == set(range(len(summaries)))
    assert support.rule_label(summaries[-1]) in picker.options
    assert picker.value == 0 and artifact.rule_calls == []

    selected, = notebook_cell_producing("rule_detail")(
        candidate_artifact=artifact, sector_choice=sector,
        rule_choice=SimpleNamespace(value=1007))
    assert artifact.rule_calls == [(14, 1007)]
    assert selected == {"ordinal": 1007, "rhs": ["selected-rule-only"]}
