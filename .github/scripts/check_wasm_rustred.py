"""Actual Pyodide acceptance for the shared HEPKit/RustRed implementation.

The JS runner supplies ordinary DOT/TOML inputs, never saved reduction rules.
All artifact bytes below are generated within this WASM interpreter.
"""

from time import perf_counter
import importlib
import sys
import tomllib

from symbolica import E, S, get_citations
from symbolica.community import hepkit as hep


assert sys.platform == "emscripten", "Run this gate inside actual Pyodide"
capabilities = hep.rustred.execution_capabilities()
package = importlib.import_module("symbolica.community.hepkit.rustred")
assert package.CandidateGenerationSession is hep.rustred.CandidateGenerationSession
assert package.execution_capabilities() == capabilities
assert capabilities["execution_mode"] == "synchronous"
assert capabilities["max_workers"] == 1
assert not capabilities["background_sessions"]
assert not capabilities["live_event_polling"]
assert not capabilities["cancellation_in_flight"]

model = hep.Model.phi_3_4()
diagram = hep.FeynmanDiagram.from_dot(model, rustred_k6_dot)
routed = diagram.integral_family(kinematics=hep.Kinematics(S("d")))
family = hep.IntegralFamily(
    routed.loop_momenta, routed.external_momenta,
    [den.replace(model.particle("phi").mass, E("1")) for den in routed.denominators],
    kinematics=routed.kinematics,
)
assert diagram.loop_count == 3
assert family.is_complete and family.is_independent
k1, k2, k3 = family.loop_momenta
for denominator, momentum in zip(
    family.denominators, [k1, k2, k3, k1 - k3, k1 - k2, k2 - k3],
):
    assert (denominator - family.kinematics.scalar_product(momentum, momentum) + 1).expand() == 0
ibp = hep.IBPFamily(family, name="wasm_k6_graph")
assert ibp.denominator_count == 6 and len(ibp.ibp_identities()) == 9
assert "https://github.com/alphal00p/rustred" in {c.id for c in get_citations()}

# Exercise shared-expression finite reduction independently of source parsing.
d, k = S("wasm_ibp_d", "wasm_ibp_k")
kinematics = hep.Kinematics(d, momenta=[k])
tadpole = hep.IBPFamily(hep.IntegralFamily(
    [k], [], [kinematics.scalar_product(k, k) - 1], kinematics=kinematics,
))
finite = tadpole.reduce_laporta([[3]], max_depth=1)
assert finite.reduce([3])[0][0] == [1]
assert (finite.reduce([3])[0][1] - (d - 2) * (d - 4) / 8).together() == 0

started = perf_counter()
session = hep.rustred.start_family_candidates(
    rustred_k6_source, input_format="toml", n_cores=1, event_capacity=16,
)
assert session.done and session.wait(timeout=0)
assert session.execution_mode == "synchronous"
events = session.poll_events(max_events=16, timeout=0)
assert events["snapshot"]["done"]
assert events["snapshot"]["state"] == "completed"
candidate = session.result()
assert candidate.status == "uncertified-candidates"
candidate_bytes = bytes(bytearray(candidate.bundle))
view = hep.rustred.CandidateArtifact.open(candidate_bytes)
assert view.metadata()["arity"] == 6
assert view.metadata()["closure_claim"] is False
assert view.metadata()["total_rules"] > 0
generated_seconds = perf_counter() - started
print(f"WASM RustRed: fresh K6 candidates generated in {generated_seconds:.2f}s", flush=True)

certificate = hep.rustred.certify_candidates(candidate_bytes)
assert certificate.status == "generated-durable"
artifact = bytes(bytearray(certificate.artifact))
inspection = tomllib.loads(hep.rustred.inspect_closing_artifact(artifact).to_toml())
masters = {tuple(row["powers"]) for row in inspection["artifact"]["masters"]}
assert masters and inspection["artifact"]["arity"] == 6
assert inspection["validation"]["replayed_source_rows"] > 0
assert inspection["validation"]["guarded_rules"] > 0
assert inspection["artifact"]["root_power_lower"] == [-(2**63)] * 6
assert inspection["artifact"]["root_power_upper"] == [2**63 - 1] * 6
print(f"WASM RustRed: certified and cold-inspected {len(masters)} master terminals", flush=True)

integral = S("wasm_k6_integral")
dimension = E("rustred::{}::d")


def reduce_expression(powers):
    result = hep.rustred.reduce_with_closing_artifact(artifact, powers)
    assert result.status == "reduced"
    for term in result.terms:
        assert tuple(term.master_powers) in masters
        assert term.common_mass_squared_power == sum(term.master_powers) - sum(powers)
    return sum((E(term.unit_mass_coefficient) * integral(*term.master_powers)
                for term in result.terms), E("0"))


scalar = reduce_expression([1] * 6)
dotted = E("0")
for index in range(6):
    powers = [1] * 6
    powers[index] = 2
    dotted += reduce_expression(powers)
assert (dotted - (3 * dimension / 2 - 6) * scalar).together() == 0
tadpoles = reduce_expression([1, 1, 1, 0, 0, 0])
assert (reduce_expression([2, 2, 1, 0, 0, 0])
        - (dimension - 2)**2 / 4 * tadpoles).together() == 0
assert (reduce_expression([1, 1, 1, -2, 0, 0])
        - (1 + 4 / dimension) * tadpoles).together() == 0
assert reduce_expression([0] * 6) == 0

rustred_wasm_validation = {
    "execution_mode": capabilities["execution_mode"],
    "generated_seconds": generated_seconds,
    "total_seconds": perf_counter() - started,
    "arity": 6,
    "terminal_masters": len(masters),
    "generated_and_certified_k6": True,
    "graph_ibp": True,
    "exact_homogeneity": True,
    "pinch_and_numerator": True,
}
print("WASM RustRed: DOT/IBP, fresh K6 generation, closure, cold loading and exact reductions passed.")
