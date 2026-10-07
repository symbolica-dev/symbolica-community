"""Write exact native64 expectations for the optional WASM K=6 artifact gate."""

from hashlib import sha256
import argparse
import json
from pathlib import Path
import struct
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from symbolica.community import hepkit as hep


def export(artifact_path, output_path):
    if struct.calcsize("P") != 8:
        raise ValueError("Produce this fixture in a native 64-bit interpreter")
    artifact_path, output_path = Path(artifact_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError("Choose a fresh expectation file")
    artifact = artifact_path.read_bytes()
    inspection = tomllib.loads(hep.rustred.inspect_closing_artifact(artifact).to_toml())
    metadata = inspection["artifact"]
    masters = sorted(row["powers"] for row in metadata["masters"])
    assert metadata["arity"] == 6 and len(masters) == 38
    targets = [[1] * 6]
    for index in range(6):
        target = [1] * 6
        target[index] = 2
        targets.append(target)
    targets += [[2, 2, 1, 1, 1, 1], [2, 2, 1, 0, 0, 0], [1, 1, 1, -2, 0, 0], [0] * 6]
    cases = []
    for target in targets:
        reduced = hep.rustred.reduce_with_closing_artifact(artifact, target)
        assert reduced.status == "reduced"
        terms = [{
            "master_powers": list(term.master_powers),
            "coefficient": term.unit_mass_coefficient,
            "common_mass_squared_power": term.common_mass_squared_power,
        } for term in reduced.terms]
        assert all(term["master_powers"] in masters for term in terms)
        cases.append({"target_powers": target, "terms": terms})
    result = {
        "schema": "rustred.cross-platform-reduction.v1", "producer_pointer_bits": 64,
        "artifact_sha256": sha256(artifact).hexdigest(), "artifact_bytes": len(artifact),
        "family_fingerprint": metadata["family_fingerprint"],
        "masters": masters, "cases": cases,
    }
    with output_path.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(cases)} exact native cases and {len(masters)} master keys to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.artifact, args.output)
