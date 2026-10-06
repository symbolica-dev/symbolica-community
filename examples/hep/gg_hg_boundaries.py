"""Load portable, previously computed starting values for the live demonstration.

These are supplied numerical inputs, not a binary-cache compatibility override.
Their exact dyadic values enter the current solver through its ordinary supplied
boundary API, with the original errors and accuracy cap. No destination values
are included.
"""

import gzip
import hashlib
import json
from pathlib import Path


def _number(record):
    from symbolica import Float

    if not isinstance(record, list) or len(record) != 3:
        raise ValueError("Invalid boundary number encoding")
    numerator, denominator, bits = record
    if (not isinstance(numerator, str) or not isinstance(denominator, str)
            or len(numerator) > 4096 or len(denominator) > 4096
            or type(bits) is not int or not 2 <= bits <= 4096):
        raise ValueError("Invalid boundary number precision or integer encoding")
    numerator, denominator = int(numerator), int(denominator)
    if denominator <= 0 or denominator & (denominator - 1):
        raise ValueError("Boundary values must use exact binary rationals")
    magnitude = abs(numerator)
    if magnitude:
        trailing_zeros = (magnitude & -magnitude).bit_length() - 1
        if magnitude.bit_length() - trailing_zeros > bits:
            raise ValueError("Boundary number exceeds its recorded significand precision")
    value = Float.from_ratio(numerator, denominator, precision=bits)
    if value.as_integer_ratio() != (numerator, denominator):
        raise ValueError("Boundary number cannot be represented at its recorded precision")
    return value


def load_boundary_bundle(directory, systems, *, control=None):
    """Validate the entire bundle and return a new cache, without changing a session."""
    from symbolica import ComplexFloat, E
    from symbolica.community.hep.integration import BoundaryCache, CalculationCancelled

    directory = Path(directory)
    manifest = json.loads((directory / "boundaries-manifest.json").read_text())
    if manifest.get("schema") != "higgs-jet-boundary-bundle-v1":
        raise ValueError("Unsupported boundary bundle manifest")
    packed = (directory / "boundaries.json.gz").read_bytes()
    digest = hashlib.sha256(packed).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Boundary bundle checksum mismatch")
    document = json.loads(gzip.decompress(packed))
    if (document.get("schema") != "higgs-jet-supplied-boundaries-v1"
            or document.get("encoding") != "exact-integer-ratio-with-binary-precision"):
        raise ValueError("Unsupported boundary bundle encoding")
    if set(systems) != {"planar", "nonplanar"}:
        raise ValueError("Both canonical Higgs-jet systems are required")
    for name, system in systems.items():
        if system.mathematical_fingerprint != manifest["mathematical_fingerprints"][name]:
            raise ValueError(f"Boundary basis, normalization or equation mismatch: {name}")
    expected = {
        configuration.label: (topology, system, configuration)
        for topology, system in systems.items()
        for configuration in system.configurations()
    }
    records = document["boundaries"]
    if len(records) != len(expected) or len(expected) != 16:
        raise ValueError("Boundary bundle must contain all sixteen starting configurations")
    pending = BoundaryCache()
    results = {}
    for record in records:
        if control is not None and control.cancelled:
            raise CalculationCancelled("Supplied boundary loading cancelled")
        label = record["label"]
        if label in results or label not in expected:
            raise ValueError("Duplicate or unknown boundary configuration")
        topology, system, configuration = expected[label]
        if (record["topology"] != topology or record["dimension"] != system.dimension
                or record["mass"] != configuration.mass
                or record["permutation"] != configuration.permutation
                or [E(x) for x in record["coordinates"]]
                != [configuration.start[x] for x in system.coordinates]
                or record["root_sheets"]
                != {x.get_name().rsplit("::", 1)[-1]: sign
                    for x, sign in configuration.root_sheets.items()}):
            raise ValueError(f"Boundary coordinate, basis or root-sheet mismatch: {label}")
        cap = min(record["verified_digits"], record["input_verified_digits"])
        if (type(cap) is not int or cap < 30 or record["leading_power"] != 0
                or len(record["coefficients"]) != 5
                or len(record["comparison_errors"]) != 5):
            raise ValueError(f"Insufficient boundary accuracy or epsilon range: {label}")
        if any(len(row) != system.dimension
               for key in ("coefficients", "comparison_errors") for row in record[key]):
            raise ValueError(f"Boundary coefficient dimensions disagree: {label}")
        coefficients = [
            [ComplexFloat(_number(z[0]), _number(z[1])) for z in row]
            for row in record["coefficients"]
        ]
        errors = [[_number(x) for x in row] for row in record["comparison_errors"]]
        if any(z.real.precision != record["working_bits"]
               or z.imag.precision != record["working_bits"]
               for row in coefficients for z in row):
            raise ValueError(f"Boundary working precision disagrees: {label}")
        provenance = (
            f"Supplied notebook starting values; bundle sha256={digest}; "
            f"original native boundary identity={record['origin_identity']}; "
            + record["provenance"]
        )
        result = system.kinematic_transport().add_boundary(
            pending, configuration.start, coefficients, 0,
            verified_digits=cap, comparison_errors=errors,
            provenance=provenance, root_sheets=configuration.root_sheets,
        )
        results[label] = result
    if control is not None and control.cancelled:
        raise CalculationCancelled("Supplied boundary loading cancelled")
    return pending, results
