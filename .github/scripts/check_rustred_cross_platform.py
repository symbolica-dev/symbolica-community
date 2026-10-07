"""Optional actual-WASM check of externally supplied native64 K=6 evidence.

The JS runner mounts both files only when explicitly requested. No artifact or
native expectation is bundled in the repository or browser notebook export.
"""

from hashlib import sha256
import json
from pathlib import Path
import struct
import tomllib

from symbolica import E
from symbolica.community import hepkit as hep


assert struct.calcsize("P") == 4, "Cross-platform consumer must be WASM32"
artifact = Path("/tmp/rustred-native-canary/artifact.rr").read_bytes()
expected = json.loads(Path("/tmp/rustred-native-canary/expected.json").read_text())
assert expected["schema"] == "rustred.cross-platform-reduction.v1"
assert expected["producer_pointer_bits"] == 64
assert len(artifact) == expected["artifact_bytes"]
assert sha256(artifact).hexdigest() == expected["artifact_sha256"]
assert len(expected["masters"]) == 38
masters = {tuple(powers) for powers in expected["masters"]}
assert len(masters) == 38
inspection = tomllib.loads(hep.rustred.inspect_closing_artifact(artifact).to_toml())
metadata = inspection["artifact"]
assert metadata["arity"] == 6
assert metadata["family_fingerprint"] == expected["family_fingerprint"]
assert {tuple(row["powers"]) for row in metadata["masters"]} == masters

targets = {(1,) * 6, (2, 2, 1, 1, 1, 1), (2, 2, 1, 0, 0, 0),
           (1, 1, 1, -2, 0, 0), (0,) * 6}
for index in range(6):
    powers = [1] * 6
    powers[index] = 2
    targets.add(tuple(powers))
assert len(expected["cases"]) == 11
assert {tuple(case["target_powers"]) for case in expected["cases"]} == targets

for case in expected["cases"]:
    target = case["target_powers"]
    result = hep.rustred.reduce_with_closing_artifact(artifact, target)
    assert result.status == "reduced"
    assert result.family_fingerprint == expected["family_fingerprint"]
    actual_terms = {
        tuple(term.master_powers): (E(term.unit_mass_coefficient), term.common_mass_squared_power)
        for term in result.terms
    }
    expected_terms = {
        tuple(term["master_powers"]): (E(term["coefficient"]), term["common_mass_squared_power"])
        for term in case["terms"]
    }
    assert len(actual_terms) == len(result.terms)
    assert len(expected_terms) == len(case["terms"])
    assert actual_terms.keys() == expected_terms.keys()
    assert actual_terms.keys() <= masters
    for powers, (coefficient, mass_power) in actual_terms.items():
        expected_coefficient, expected_mass_power = expected_terms[powers]
        assert mass_power == expected_mass_power == sum(powers) - sum(target)
        assert (coefficient - expected_coefficient).together() == 0, (target, powers)

rustred_cross_platform_validation = {
    "producer_pointer_bits": 64, "consumer_pointer_bits": 32,
    "artifact_sha256": expected["artifact_sha256"],
    "expectations_sha256": sha256(Path("/tmp/rustred-native-canary/expected.json").read_bytes()).hexdigest(),
    "family_fingerprint": expected["family_fingerprint"],
    "masters": len(masters), "exact_cases": len(expected["cases"]), "status": "passed",
}
