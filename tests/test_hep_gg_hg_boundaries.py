"""Portable scientific seeds: no reduction, boundary generation or transport work."""

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from symbolica.community.hep import integration as numerical

EXAMPLES = Path(__file__).parents[1] / "examples" / "hep"
BUNDLE = EXAMPLES / "data" / "gg_hg"
SPEC = importlib.util.spec_from_file_location("gg_hg_seed_loader", EXAMPLES / "gg_hg_boundaries.py")
LOADER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOADER)


@pytest.fixture(scope="module")
def systems():
    return {name: numerical.HiggsJetIntegralSystem(name) for name in ("planar", "nonplanar")}


def test_shipped_values_round_trip_through_native_evidence_and_binary_restart(systems, tmp_path):
    cache, results = LOADER.load_boundary_bundle(BUNDLE, systems)
    assert len(cache) == len(results) == 16
    records = json.loads(gzip.decompress((BUNDLE / "boundaries.json.gz").read_bytes()))["boundaries"]
    for record in records:
        result = results[record["label"]]
        assert result.verified_digits == result.input_verified_digits == 40
        assert result.working_bits == 415
        assert "Supplied notebook starting values" in result.provenance
        for values, encoded in zip(result.coefficients, record["coefficients"], strict=True):
            for value, pair in zip(values, encoded, strict=True):
                assert value.real.as_integer_ratio() == tuple(map(int, pair[0][:2]))
                assert value.imag.as_integer_ratio() == tuple(map(int, pair[1][:2]))
                assert value.real.precision == pair[0][2]
                assert value.imag.precision == pair[1][2]
        for values, encoded in zip(result.comparison_errors, record["comparison_errors"], strict=True):
            for value, number in zip(values, encoded, strict=True):
                assert value.as_integer_ratio() == tuple(map(int, number[:2]))
                assert value.precision == number[2]
    cache.save(tmp_path)
    restored = numerical.BoundaryCache.load(tmp_path)
    assert len(restored) == 16
    assert sorted((entry.identity, entry.provenance) for entry in restored.entries()) == sorted(
        (entry.identity, entry.provenance) for entry in cache.entries()
    )
    for system in systems.values():
        for configuration in system.configurations():
            result = system.evaluate(restored, configuration.start, configuration.root_sheets)
            assert result.cache_hit and result.steps == 0 and result.inserted_points == 0
            expected = results[configuration.label]
            assert result.coefficients == expected.coefficients
            assert result.comparison_errors == expected.comparison_errors
            assert result.verified_digits == expected.verified_digits
            assert result.input_verified_digits == expected.input_verified_digits
            assert result.working_bits == expected.working_bits
            # Evaluation adds its cache-hit history; stored provenance above is exact.
            assert expected.provenance in result.provenance
            assert [[(z.real.precision, z.imag.precision) for z in row]
                    for row in result.coefficients] == [
                [(z.real.precision, z.imag.precision) for z in row]
                for row in expected.coefficients
            ]
            assert [[error.precision for error in row] for row in result.comparison_errors] == [
                [error.precision for error in row] for row in expected.comparison_errors
            ]


def _copy_bundle(tmp_path, change=None, manifest_change=None):
    document = json.loads(gzip.decompress((BUNDLE / "boundaries.json.gz").read_bytes()))
    manifest = json.loads((BUNDLE / "boundaries-manifest.json").read_text())
    if change:
        change(document)
    packed = gzip.compress(json.dumps(document).encode(), mtime=0)
    manifest["sha256"] = hashlib.sha256(packed).hexdigest()
    if manifest_change:
        manifest_change(manifest)
    (tmp_path / "boundaries.json.gz").write_bytes(packed)
    (tmp_path / "boundaries-manifest.json").write_text(json.dumps(manifest))
    return tmp_path


@pytest.mark.parametrize("field, value, error", [
    ("label", "unknown", "unknown"),
    ("root_sheets", {"root1": -1, "root2": -1}, "root-sheet"),
    ("coordinates", ["0", "0", "0"], "coordinate"),
    ("input_verified_digits", 10, "accuracy"),
    ("leading_power", -1, "epsilon"),
    ("working_bits", 500, "precision"),
])
def test_wrong_scientific_metadata_is_rejected(systems, tmp_path, field, value, error):
    path = _copy_bundle(tmp_path, lambda d: d["boundaries"][0].__setitem__(field, value))
    with pytest.raises(ValueError, match=error):
        LOADER.load_boundary_bundle(path, systems)


def test_wrong_mathematics_is_rejected_before_import(systems, tmp_path):
    path = _copy_bundle(tmp_path, manifest_change=lambda m:
                        m["mathematical_fingerprints"].__setitem__("planar", "different basis"))
    with pytest.raises(ValueError, match="equation mismatch"):
        LOADER.load_boundary_bundle(path, systems)


def test_corrupt_download_is_rejected(systems, tmp_path):
    path = _copy_bundle(tmp_path)
    with (path / "boundaries.json.gz").open("ab") as output:
        output.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        LOADER.load_boundary_bundle(path, systems)


def test_cancelled_import_returns_no_partial_cache(systems):
    control = numerical.ComputationControl()
    control.cancel()
    with pytest.raises(numerical.CalculationCancelled):
        LOADER.load_boundary_bundle(BUNDLE, systems, control=control)


@pytest.mark.parametrize("record", [
    ["1", "3", 200],  # Non-dyadic input would require rounding.
    ["1025", "1", 2],  # Declared precision cannot preserve this exact value.
    ["1", "0", 200],
    ["1", "1", 0],
])
def test_numeric_import_rejects_lossy_encoding(record):
    with pytest.raises(ValueError):
        LOADER._number(record)
