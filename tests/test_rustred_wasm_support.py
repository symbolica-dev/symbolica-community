"""Portable notebook lifecycle checks; fakes are not solver evidence."""

import ast
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("marimo")


EXAMPLES = Path(__file__).parents[1] / "examples/hep"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FOUR = load("rustred_campaign_support")
SYNCHRONOUS = {
    "execution_mode": "synchronous", "background_sessions": False,
    "live_event_polling": False, "cancellation_in_flight": False, "max_workers": 1,
}


class Result:
    status = "uncertified-candidates"
    bundle = b"test-only-not-native"

    def artifact(self):
        return self

    def to_toml(self):
        return 'status = "test-only-not-native"\n'


class Session:
    def __init__(self, state="completed"):
        self.state = state
        self.results = 0

    def poll_events(self, *, max_events, timeout):
        assert max_events == 128 and timeout == 0
        return {
            "snapshot": {
                "state": self.state, "done": True, "elapsed_seconds": 1,
                "counts": {"generated": 2, "sectors_total": 2, "rules": 3},
                "active_jobs": [], "last_error": "test-only failure",
            },
            "events": [], "dropped_events": 0,
        }

    def result(self):
        self.results += 1
        return Result()

    def cancel(self):
        raise AssertionError("Synchronous sessions cannot cancel in flight")


def test_three_loop_api_cells_are_visible_and_do_not_require_native_capabilities():
    source = (EXAMPLES / "three_loop_reduction.py").read_text()
    tree = ast.parse(source)
    cells = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for method in ("family_candidates", "certify_candidates", "inspect_closing_artifact"):
        matching = [cell for cell in cells if any(
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "rustred" and node.func.attr == method
            for node in ast.walk(cell))]
        assert len(matching) == 1, f"{method} must have one direct notebook call"
        assert not any(
            isinstance(decorator, ast.Call)
            and any(keyword.arg == "hide_code" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value for keyword in decorator.keywords)
            for decorator in matching[0].decorator_list)

    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert attributes.isdisjoint({"execution_capabilities", "start_family_candidates",
                                  "poll_events"})
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "native_available" not in names and "ThreeLoopRun" not in names
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "family_candidates"]
    options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in calls[0].keywords}
    assert options == {"input_format": "toml", "n_cores": 1,
                       "exact_backend": "sparse", "numerical_depth": 2}


def test_four_loop_synchronous_queue_starts_only_on_request(tmp_path):
    class Family:
        def __init__(self):
            self.calls = 0

        def start_generation(self, **options):
            assert options["n_cores"] == 1
            self.calls += 1
            return Session()

    families = {name: Family() for name in FOUR.FAMILY_NAMES}
    run = FOUR.FourLoopCampaign(
        families, {name: [9] for name in families},
        output_directory=tmp_path / "campaign", capabilities=SYNCHRONOUS,
    )
    assert run.poll()["state"] == "ready"
    assert not (tmp_path / "campaign").exists()
    assert all(family.calls == 0 for family in families.values())
    assert run.start(n_cores=1)
    assert run.poll()["state"] == "completed"
    assert tuple(run.results) == FOUR.FAMILY_NAMES
    assert all(family.calls == 1 for family in families.values())
    assert not run.start(n_cores=1) and not run.cancel()
    assert len(list((tmp_path / "campaign").glob("*/candidate.rrbin"))) == 4
