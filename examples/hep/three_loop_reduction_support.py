"""Small notebook lifecycle helpers; all IBP algebra stays in native RustRed."""

from collections import deque
from pathlib import Path
from time import monotonic
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "rustred_three_loop"


def graph_inputs():
    """Load the graph and its explicitly checked native source, from any cwd."""
    return tuple((DATA_DIRECTORY / name).read_text() for name in ("k6.dot", "k6.toml"))


def rule_expressions(artifact, rule, integral, parameter_bindings=()):
    """Materialize only the selected source-input rule for Symbolica display.

    The native polynomial printer uses bare n0, n1, ... and d. Parse them in
    rustred's namespace and bind the indices to the displayed n_0, n_1, ... .
    This display conversion does not apply or certify a reduction rule.
    """
    from symbolica import E, S

    indices = [S(f"n_{axis}") for axis in range(len(rule["target"]["values"]))]
    bindings = [(S(f"rustred::n{axis}"), index)
                for axis, index in enumerate(indices)] + list(parameter_bindings)
    coefficients = {}

    def coefficient(cid):
        if cid not in coefficients:
            detail = artifact.coefficient(cid, max_output_bytes=65536)
            value = (E(detail["numerator"], default_namespace="rustred")
                     / E(detail["denominator"], default_namespace="rustred"))
            for source, display in bindings:
                value = value.replace(source, display)
            coefficients[cid] = value
        return coefficients[cid]

    def key_expression(key):
        return integral(*(indices[axis] + value if symbolic else value
                          for axis, (value, symbolic) in enumerate(
                              zip(key["values"], key["symbolic"]))))

    return {
        "target": key_expression(rule["target"]),
        "rhs": sum((coefficient(term["coefficient_id"]) * key_expression(term["integral"])
                    for term in rule["rhs"]), E("0")),
        "affine": [coefficient(cid) for cid in rule["case"]["affine_zero_equations"]],
        "excluded": [[coefficient(cid) for cid in branch]
                     for branch in rule["excluded_all_zero_conjunctions"]],
    }


def assert_source_matches_family(source, family):
    """Check this example's fixed source against native routed scalar products.

    No general graph serializer or polynomial parser lives here. The six
    source expressions are checked literally; Symbolica performs the exact
    comparison with the corresponding native HEPKit scalar products.
    """
    parsed = tomllib.loads(source)
    definition = parsed["family"]
    assert definition["dimension"] == "d"
    assert definition["loop_momenta"] == ["k1", "k2", "k3"]
    assert definition["external_momenta"] == []
    assert [item["expression"] for item in definition["denominators"]] == [
        "k1^2-1", "k2^2-1", "k3^2-1", "(k1-k3)^2-1",
        "(k1-k2)^2-1", "(k2-k3)^2-1",
    ]
    assert len(family.loop_momenta) == 3 and not family.external_momenta
    assert family.is_complete and family.is_independent
    k1, k2, k3 = family.loop_momenta
    momenta = [k1, k2, k3, k1 - k3, k1 - k2, k2 - k3]
    assert len(family.denominators) == len(momenta)
    for denominator, momentum in zip(family.denominators, momenta):
        expected = family.kinematics.scalar_product(momentum, momentum) - 1
        assert (denominator - expected).expand() == 0


class ThreeLoopRun:
    """One explicit native generation; no work starts when constructing this.

    Certification and recursive reduction are visible notebook operations.
    Their results are kept here so UI changes cannot repeat expensive algebra.
    No artifact is loaded from a precomputed catalog or another notebook run.
    """

    def __init__(self, native, source, *, clock=monotonic):
        self.native, self.source, self.clock = native, source, clock
        self.capabilities = native.execution_capabilities()
        self.state, self.session = "ready", None
        self.started, self.finished = None, None
        self.result, self.candidate = None, None
        self.closing, self.inspection = None, None
        self.reductions = {}
        self.events = deque(maxlen=20)
        self.counts, self.active_jobs = {}, []
        self.dropped_events, self.error = 0, None

    def start(self, **options):
        if self.state != "ready":
            return False
        self.started, self.state = self.clock(), "running"
        try:
            self.session = self.native.start_family_candidates(
                self.source, input_format="toml", **options)
            if not self.capabilities["background_sessions"]:
                # WASM completes inside start(); collect only real finished work.
                self.poll()
        except Exception as error:
            self._fail(error)
            return False
        return True

    def cancel(self):
        if (not self.capabilities["cancellation_in_flight"] or self.session is None
                or self.state not in {"running", "cancelling"}):
            return False
        self.session.cancel()
        self.state = "cancelling"
        return True

    def _fail(self, error):
        self.error, self.state = str(error), "failed"
        self.finished, self.session = self.clock(), None

    def poll(self):
        """Drain at most 128 native events, without waiting or decoding algebra."""
        if self.session is not None:
            try:
                batch = self.session.poll_events(max_events=128, timeout=0.0)
                native = batch["snapshot"]
                self.counts = dict(native["counts"])
                self.active_jobs = native["active_jobs"]
                self.events.extend(batch["events"])
                self.dropped_events = batch["dropped_events"]
                if native["done"]:
                    if native["state"] == "completed":
                        self.result = self.session.result()
                        self.candidate = self.result.artifact()
                        self.state = "generated"
                    elif native["state"] == "cancelled":
                        self.state = "cancelled"
                    else:
                        raise RuntimeError(native.get("last_error") or "Generation failed")
                    self.finished, self.session = self.clock(), None
            except Exception as error:
                self._fail(error)
        end = self.finished if self.finished is not None else self.clock()
        return {
            "state": self.state,
            "elapsed_seconds": 0 if self.started is None else end - self.started,
            "counts": dict(self.counts), "active_jobs": self.active_jobs,
            "events": list(self.events), "dropped_events": self.dropped_events,
            "error": self.error,
        }
