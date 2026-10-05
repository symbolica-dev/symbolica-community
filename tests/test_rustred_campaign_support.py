"""Cheap lifecycle tests use explicit fakes; they are not solver evidence."""

import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "rustred_campaign_support",
    Path(__file__).parents[1] / "examples/hep/rustred_campaign_support.py",
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


class FakeResult:
    bundle = b"test-only-not-a-native-artifact"
    status = "uncertified-candidates"

    def artifact(self):
        return self

    def to_toml(self):
        return 'status = "test-only-not-native"\n'


class FakeSession:
    def __init__(self):
        self.state = "running"
        self.cancelled = False
        self.polls = 0

    def cancel(self):
        self.cancelled = True
        self.state = "cancelling"

    def poll_events(self, *, max_events, timeout):
        assert max_events == 128 and timeout == 0
        self.polls += 1
        return {
            "snapshot": {
                "state": self.state,
                "done": self.state in {"completed", "cancelled", "failed"},
                "elapsed_seconds": 1.25,
                "counts": {"sectors_total": 9, "generated": 3, "rules": 7},
                "active_jobs": [], "last_error": "test failure",
            },
            "events": [{"sequence": self.polls, "kind": "test-only"}],
            "dropped_events": 2,
        }

    def result(self):
        assert self.state == "completed"
        return FakeResult()


class FakeFamily:
    def __init__(self):
        self.calls = []
        self.sessions = []

    def start_generation(self, **options):
        self.calls.append(options)
        session = FakeSession()
        self.sessions.append(session)
        return session


def campaign(tmp_path):
    families = {name: FakeFamily() for name in support.FAMILY_NAMES}
    return support.FourLoopCampaign(
        families, {name: [9] for name in families}, output_directory=tmp_path / "run"
    ), families


def test_no_native_call_or_directory_before_explicit_start(tmp_path):
    run, families = campaign(tmp_path)
    assert run.poll()["state"] == "ready"
    assert not (tmp_path / "run").exists()
    assert all(not family.calls for family in families.values())


def test_numerator_evaluation_requires_explicit_action_after_complete_generation(tmp_path):
    run, _ = campaign(tmp_path)
    calls = []
    evaluation = support.ExplicitNumeratorEvaluation(
        lambda integral, path: calls.append((integral, path)) or {"metrics": {"reference_matches": True}})
    assert not calls and evaluation.state == "ready"
    assert not evaluation.run(run, "test-only-input")
    assert not calls
    run.state = "running"
    assert not evaluation.run(run, "test-only-input")
    run.state = "completed"
    run.started = 0
    run.output_directory.mkdir()
    assert evaluation.run(run, "test-only-input")
    assert not evaluation.run(run, "test-only-input")
    assert len(calls) == 1 and evaluation.state == "completed"
    evidence = support.json.loads((run.output_directory / "vakint-h-numerator.json").read_text())
    assert evidence["metrics"]["reference_matches"] is True
    assert not any(family.calls for family in run.families.values())


def test_numerator_failure_is_preserved_without_reactive_retry(tmp_path):
    run, _ = campaign(tmp_path)
    run.state = "completed"
    run.started = 0
    run.output_directory.mkdir()
    calls = []

    def refuse(integral, path):
        calls.append(integral)
        raise ValueError("native test-only refusal")

    evaluation = support.ExplicitNumeratorEvaluation(refuse)
    assert not evaluation.run(run, "test-input")
    assert not evaluation.run(run, "test-input")
    assert len(calls) == 1 and evaluation.state == "failed"
    assert "native test-only refusal" in evaluation.error
    assert run.state == "completed"


def test_complete_all_four_sequentially_and_persist_only_done(tmp_path):
    run, families = campaign(tmp_path)
    assert run.start(n_cores=1, event_capacity=256)
    assert not run.start(n_cores=1)  # a reactive re-run is not a restart
    assert len(families["H"].calls) == 1
    assert not families["X"].calls
    run.poll()
    assert not list((tmp_path / "run").glob("*/candidate.rrbin"))
    for name in support.FAMILY_NAMES:
        assert run.active_name == name
        options = families[name].calls[0]
        assert options["n_cores"] == 1 and options["resume"] is False
        assert options["nonpositive_indices"] == [9]
        run.session.state = "completed"
        run.poll()
    snapshot = run.snapshot()
    assert snapshot["state"] == "completed"
    assert snapshot["completed_families"] == 4
    assert len(list((tmp_path / "run").glob("*/candidate.rrbin"))) == 4
    assert tuple(run.artifacts) == support.FAMILY_NAMES
    recorded = support.json.loads((run.output_directory / "snapshot.json").read_text())
    assert recorded["state"] == "completed" and recorded["completed_families"] == 4
    controls = support.json.loads((run.output_directory / "input-controls.json").read_text())
    assert controls["options"]["n_cores"] == 1
    assert len(list(run.output_directory.glob("*/generation-report.toml"))) == 4


def test_cancel_waits_for_drain_and_never_starts_next_family(tmp_path):
    run, families = campaign(tmp_path)
    run.start(n_cores=1)
    run.cancel()
    assert run.session.cancelled
    assert run.poll()["state"] == "cancelling"
    assert not families["X"].calls
    run.session.state = "cancelled"
    snapshot = run.poll()
    assert snapshot["state"] == "cancelled"
    assert not run.results and not run.artifacts
    assert all(row["state"] == "not started" for row in snapshot["families"][1:])


def test_cancel_racing_with_completed_sector_does_not_start_next(tmp_path):
    run, families = campaign(tmp_path)
    run.start(n_cores=1)
    run.cancel()
    run.session.state = "completed"
    snapshot = run.poll()
    assert snapshot["state"] == "cancelled"
    assert snapshot["completed_families"] == 1
    assert not families["X"].calls


def test_native_failure_preserves_error_and_stops_queue(tmp_path):
    run, families = campaign(tmp_path)
    run.start(n_cores=1)
    run.session.state = "failed"
    snapshot = run.poll()
    assert snapshot["state"] == "failed"
    assert snapshot["last_error"] == "test failure"
    assert not families["X"].calls and not run.artifacts


def test_output_and_family_scope_not_silently_reused(tmp_path):
    run, _ = campaign(tmp_path)
    run.output_directory.mkdir(parents=True)
    (run.output_directory / "existing").touch()
    with pytest.raises(ValueError, match="empty output"):
        run.start(n_cores=1)
    with pytest.raises(ValueError, match="retain H, X, BMW and FG"):
        support.FourLoopCampaign({"H": FakeFamily()}, {"H": [9]})


def test_ui_history_is_bounded_and_counts_not_inferred_from_events(tmp_path):
    run, _ = campaign(tmp_path)
    run.start(n_cores=1)
    for _ in range(125):
        run.poll()
    snapshot = run.snapshot()
    assert len(snapshot["events"]) == 80
    summary = support.summary_rows(snapshot)
    assert summary[0]["Rules"] == 7 and summary[0]["Sectors"] == "3 / 9"
    assert snapshot["families"][0]["dropped_events"] == 2
    assert summary[1]["Sectors"] == "— / —"


def test_packaged_graphs_are_available_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sources = support.dot_sources()
    assert tuple(sources) == support.FAMILY_NAMES
    for name, source in sources.items():
        edge_count = source.count('particle="phi"')
        assert edge_count == (9 if name in {"H", "X"} else 8)
        for loop in range(4):
            assert source.count(f"lmb_id={loop}") == 1


def test_integral_notation_preserves_fixed_replacements_and_symbolic_shifts():
    assert support.integral_notation({
        "symbolic": [True, False, True, True], "values": [-2, -1, 0, 3]
    }) == "I(n_0 - 2, -1, n_2, n_3 + 3)"
    assert support.integral_notation([1, 0, -2]) == "I(1, 0, -2)"
    with pytest.raises(ValueError, match="arities"):
        support.integral_notation({"symbolic": [True], "values": [0, 1]})
    with pytest.raises(ValueError, match="integer"):
        support.integral_notation([True])


def test_condition_presentation_preserves_and_or_and_lazy_ids():
    rule = {
        "case": {"kind": "affine", "fixed": [{"axis": 2, "value": -1}],
                 "affine_zero_equations": [7, 9]},
        "rhs": [{"coefficient_id": 3}, {"coefficient_id": 7}],
        "excluded_all_zero_conjunctions": [[10, 11], [12], []],
    }
    rows = support.rule_condition_rows(rule, [True, False, False])
    assert rows[0]["Meaning"] == "n_0 > 0, n_1 ≤ 0, n_2 ≤ 0"
    assert rows[1]["Meaning"] == "n_2 = -1"
    assert rows[2]["Meaning"] == "c_7 = 0 AND c_9 = 0"
    assert rows[3]["Meaning"] == "c_10 = 0 AND c_11 = 0"
    assert rows[4]["Meaning"] == "c_12 = 0"
    assert rows[5]["Meaning"] == "Always true (empty conjunction)"
    assert support.rule_coefficient_ids(rule) == [3, 7, 9, 10, 11, 12]


def test_terminal_ordinals_and_parameter_legend_do_not_reinterpret_values():
    assert support.terminal_rows({"start": 25, "items": [[1, 0], [2, -1]]}) == [
        {"ordinal": 25, "Integral": "I(1, 0)"},
        {"ordinal": 26, "Integral": "I(2, -1)"},
    ]
    assert support.parameter_rows([("internal_0", "s(p,q)")]) == [
        {"Artifact variable": "internal_0", "Original HEPKit expression": "s(p,q)"}
    ]


def test_native_printer_text_is_escaped_not_parsed():
    class HtmlOnly:
        @staticmethod
        def Html(value):
            return value
    text = "unsafe_<p> & n_0/unknown(name)"
    rendered = support._code(HtmlOnly, text)
    assert "unsafe_&lt;p&gt; &amp; n_0/unknown(name)" in rendered
    assert "<p>" not in rendered


def test_readable_panels_use_only_supplied_structural_fixture():
    mo = pytest.importorskip("marimo")
    rule = {
        "ordinal": 2, "retained_source_count": 1,
        "case": {"kind": "general", "fixed": [], "affine_zero_equations": []},
        "target": {"symbolic": [True, False], "values": [0, 1]},
        "rhs": [{"ordinal": 0, "coefficient_id": 3,
                 "integral": {"symbolic": [True, False], "values": [-1, 2]}}],
        "excluded_all_zero_conjunctions": [[4]],
    }
    assert support.rule_view(mo, rule, [True, True]).text
    assert support.coefficient_view(mo, {
        "id": 3, "variables": ["test-only"], "numerator": "test < numerator",
        "denominator": "unparsed(test)", "numerator_terms": 1, "denominator_terms": 1,
    }).text
    assert support._table(mo, []).text


def test_view_evidence_does_not_call_other_native_operations(tmp_path):
    run, _ = campaign(tmp_path)
    run.start(n_cores=1)
    calls = []
    value = run.observe_view("H", "metadata", lambda: calls.append("metadata") or
                             {"decoded_coefficients": 0})
    assert calls == ["metadata"] and value["decoded_coefficients"] == 0
    row = support.json.loads((run.output_directory / "views.jsonl").read_text())
    assert row["operation"] == "metadata" and row["metadata"] == value


def test_large_rule_initial_html_and_raw_preview_are_bounded():
    mo = pytest.importorskip("marimo")
    term = {"ordinal": 0, "coefficient_id": 3,
            "integral": {"symbolic": [True, False], "values": [-1, 2]}}
    rule = {
        "ordinal": 2, "retained_source_count": 1,
        "case": {"kind": "affine", "fixed": [], "affine_zero_equations": list(range(10000))},
        "target": {"symbolic": [True, False], "values": [0, 1]},
        "rhs": [term] * 10000,
        "excluded_all_zero_conjunctions": [list(range(10000))] * 10000,
    }
    preview = support.rule_page_payload(rule)
    assert len(preview["rhs"]) == 10
    assert len(preview["case"]["affine_zero_equations"]) == 8
    assert len(preview["excluded_branches_preview"]) == 10
    assert all(len(branch["equations"]) == 8 for branch in preview["excluded_branches_preview"])
    assert preview["rhs_total"] == preview["excluded_branches_total"] == 10000
    assert len(support.json.dumps(preview)) < 5000
    assert len(support.rule_coefficient_ids(rule)) <= 98
    html = support.rule_view(mo, rule, [True, False]).text
    assert len(html) < 60000
    assert "preview only" in html
    assert "c_9999" not in html


def test_rule_pages_select_only_requested_slices():
    rule = {
        "ordinal": 0, "target": [1],
        "case": {"kind": "coordinate", "fixed": [], "affine_zero_equations": []},
        "rhs": [{"ordinal": i, "coefficient_id": i, "integral": [i]} for i in range(35)],
        "excluded_all_zero_conjunctions": [[100 + i] for i in range(35)],
    }
    page = support.rule_page_payload(rule, rhs_start=20, condition_start=10)
    assert [term["ordinal"] for term in page["rhs"]] == list(range(20, 30))
    assert support.rule_coefficient_ids(rule, rhs_start=20, condition_start=10) == (
        list(range(20, 30)) + list(range(110, 120)))


def test_coefficient_metadata_does_not_duplicate_expression_strings():
    class Recorder:
        def __init__(self):
            self.ui = self
            self.json_values = []
        def json(self, value):
            self.json_values.append(value)
            return value
        def md(self, value): return value
        def Html(self, value): return value
        def accordion(self, value): return value
        def vstack(self, value): return value
    mo = Recorder()
    support.coefficient_view(mo, {
        "id": 3, "variables": ["d"], "numerator": "UNIQUE_NATIVE_NUMERATOR",
        "denominator": "UNIQUE_NATIVE_DENOMINATOR", "numerator_terms": 1, "denominator_terms": 1,
    })
    assert len(mo.json_values) == 1
    assert "numerator" not in mo.json_values[0] and "denominator" not in mo.json_values[0]


def test_normalization_is_explicit_waits_for_drain_and_reuses_results(tmp_path, monkeypatch):
    calls = []
    class Normalized:
        def metadata(self):
            return {"raw_terminal_records": 3, "unique_raw_terminals": 3,
                    "after_unit_aliases": 2, "canonical_terminals": 1}
    def normalize(family, artifact):
        calls.append((family, artifact))
        return Normalized()
    monkeypatch.setattr(FakeFamily, "normalize_candidate_terminals", normalize, raising=False)
    run, _ = campaign(tmp_path)
    assert not run.normalize_completed()
    run.start(n_cores=1)
    assert not run.normalize_completed() and not calls
    for _ in support.FAMILY_NAMES:
        run.session.state = "completed"
        run.poll()
    assert not calls  # poll and artifact collection never normalize implicitly
    assert run.normalize_completed() and len(calls) == 4
    assert run.normalize_completed() and len(calls) == 4
    assert run.snapshot()["state"] == "completed"
    assert [row["Weighted outputs"] for row in support.normalization_summary_rows(run)] == [1] * 4
    assert (run.output_directory / "terminal-normalization.json").is_file()


def test_normalization_refusal_does_not_change_generation_result(tmp_path, monkeypatch):
    def refuse(*args):
        raise ValueError("test-only native admission refusal")
    monkeypatch.setattr(FakeFamily, "normalize_candidate_terminals", refuse, raising=False)
    run, _ = campaign(tmp_path)
    run.start(n_cores=1)
    for _ in support.FAMILY_NAMES:
        run.session.state = "completed"
        run.poll()
    run.normalize_completed()
    assert not run.normalizations
    assert all(row["State"] == "refused" for row in support.normalization_summary_rows(run))
    assert run.snapshot()["state"] == "completed" and len(run.artifacts) == 4
    assert run.last_error is None


def test_normalization_relation_initial_preview_is_bounded():
    mo = pytest.importorskip("marimo")
    relation = {"ordinal": 1, "integral": [1, 0],
                "rhs": [{"integral": [1, -1], "coefficient_id": 9}] * 10000}
    html = support.normalization_relation_view(mo, relation).text
    assert len(html) < 20000
    assert "of 10000" in html


def test_evidence_failure_keeps_live_session_cancellable(tmp_path, monkeypatch):
    run, _ = campaign(tmp_path)
    run.start(n_cores=1)
    session = run.session
    def refuse(*args, **kwargs):
        raise OSError("test-only evidence failure")
    monkeypatch.setattr(Path, "write_text", refuse)
    snapshot = run.poll()
    assert snapshot["state"] == "running"
    assert snapshot["evidence_error"] == "test-only evidence failure"
    assert run.session is session
    run.cancel()
    assert session.cancelled and run.state == "cancelling"


def test_failed_coefficient_view_is_not_recorded_as_loaded(tmp_path):
    run, _ = campaign(tmp_path)
    run.start(n_cores=1)
    session = run.session
    def refuse():
        raise RuntimeError("test-only native printer refusal")
    with pytest.raises(RuntimeError, match="printer refusal"):
        run.observe_view("H", "coefficient", refuse)
    assert not (run.output_directory / "views.jsonl").exists()
    assert run.session is session and run.state == "running"
