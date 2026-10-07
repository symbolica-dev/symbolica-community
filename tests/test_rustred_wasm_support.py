"""Portable notebook lifecycle checks; fakes are not solver evidence."""

import importlib.util
from pathlib import Path


EXAMPLES = Path(__file__).parents[1] / "examples/hep"


def load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


THREE = load("three_loop_reduction_support")
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


class Native:
    def __init__(self, *, state="completed", error=None):
        self.calls, self.state, self.error = 0, state, error

    def execution_capabilities(self):
        return dict(SYNCHRONOUS)

    def start_family_candidates(self, source, **options):
        assert source == "test-only-source" and options["n_cores"] == 1
        self.calls += 1
        if self.error:
            raise self.error
        self.session = Session(self.state)
        return self.session


def test_synchronous_generation_is_explicit_and_collects_once():
    native = Native()
    run = THREE.ThreeLoopRun(native, "test-only-source")
    assert run.poll()["state"] == "ready" and native.calls == 0
    assert not run.cancel()
    assert run.start(n_cores=1)
    assert run.poll()["state"] == "generated"
    assert run.session is None and run.result is run.candidate
    assert run.closing is run.inspection is None and run.reductions == {}
    assert not run.start(n_cores=1) and not run.cancel()
    run.poll()
    assert native.calls == native.session.results == 1


def test_synchronous_failure_retains_error_without_restarting():
    native = Native(error=ValueError("test-only rejection"))
    run = THREE.ThreeLoopRun(native, "test-only-source")
    assert not run.start(n_cores=1)
    assert run.poll()["state"] == "failed"
    assert run.error == "test-only rejection" and run.finished is not None
    assert not run.start(n_cores=1) and native.calls == 1
    assert run.result is run.candidate is run.closing is None


def test_completed_failure_never_fetches_a_result():
    native = Native(state="failed")
    run = THREE.ThreeLoopRun(native, "test-only-source")
    run.start(n_cores=1)
    assert run.poll()["state"] == "failed"
    assert run.error == "test-only failure" and native.session.results == 0


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
