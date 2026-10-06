"""Reject mismatched native graph owners and unverified wheel substitutions."""

import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest

CHECKS = runpy.run_path(
    str(Path(__file__).parents[1] / ".github/scripts/check_native_graph_package.py")
)
REVISION = "a" * 40
REPOSITORY = "https://github.com/alphal00p/gammaloop"


def manifests(tmp_path):
    manifest, lock = tmp_path / "Cargo.toml", tmp_path / "Cargo.lock"
    pin = f'{{ git = "{REPOSITORY}", branch = "feynkit" }}'
    manifest.write_text(f"[dependencies]\nfeynkit-py = {pin}\nspynso3 = {pin}\n")
    lock.write_text(
        "".join(
            f'[[package]]\nname = "{name}"\nsource = "git+{REPOSITORY}?branch=feynkit#{REVISION}"\n'
            for name in ("feynkit-py", "linnet", "spynso3")
        )
    )
    return manifest, lock


def test_owner_requires_matching_manifest_sources(tmp_path):
    manifest, lock = manifests(tmp_path)
    assert CHECKS["owner_pin"](manifest, lock) == (REPOSITORY, REVISION)
    prefix, suffix = manifest.read_text().rsplit('branch = "feynkit"', 1)
    manifest.write_text(prefix + 'branch = "feature"' + suffix)
    with pytest.raises(ValueError, match="manifest owners differ"):
        CHECKS["owner_pin"](manifest, lock)


@pytest.mark.parametrize("failure", ["wrong_source", "duplicate", "missing_active"])
def test_invalid_locked_owner_is_rejected(tmp_path, failure):
    manifest, lock = manifests(tmp_path)
    source = lock.read_text()
    if failure == "wrong_source":
        source = source.replace(f"#{REVISION}", f"#{'b' * 40}", 2)
    elif failure == "duplicate":
        source += '[[package]]\nname = "linnet"\nsource = "registry+foreign"\n'
    else:
        source = source.replace("[[package]]", "[[patch.unused]]", 1)
    lock.write_text(source)
    with pytest.raises(ValueError, match="differs from the manifest owner"):
        CHECKS["owner_pin"](manifest, lock)


def wheel_fixture(tmp_path, *, change=None):
    owner = tmp_path / "owner"
    project = owner / "crates/linnet-py/pyproject.toml"
    project.parent.mkdir(parents=True)
    project.write_text(
        '[project]\nname = "linnet"\nversion = "0.1.0"\nrequires-python = ">=3.10"\ndependencies = []\n'
    )
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    files = {
        "linnet-0.1.0.dist-info/METADATA": b"Name: linnet\nVersion: 0.1.0\nRequires-Python: >=3.10\n",
        "linnet-0.1.0.dist-info/WHEEL": b"Root-Is-Purelib: false\nTag: cp310-abi3-linux_x86_64\n",
        "linnet/linnet.abi3.so": b"native test payload",
    }
    if change:
        change(files)
    with ZipFile(
        wheel_dir / "linnet-0.1.0-cp310-abi3-linux_x86_64.whl", "w"
    ) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return wheel_dir, owner


def test_native_wheel_hashes_and_metadata(tmp_path):
    wheel_dir, owner = wheel_fixture(tmp_path)
    record = CHECKS["check_wheel"](wheel_dir, owner)
    assert record["sha256"] == CHECKS["digest"](Path(record["wheel"]).read_bytes())
    assert record["native_sha256"] == CHECKS["digest"](b"native test payload")


@pytest.mark.parametrize(
    "case", ["duplicate_wheel", "version", "dependency", "pure", "tag", "extension"]
)
def test_unexpected_wheel_is_rejected(tmp_path, case):
    def mutate(files):
        metadata, wheel = (
            "linnet-0.1.0.dist-info/METADATA",
            "linnet-0.1.0.dist-info/WHEEL",
        )
        if case == "version":
            files[metadata] = files[metadata].replace(b"0.1.0", b"9.0.0")
        elif case == "dependency":
            files[metadata] += b"Requires-Dist: typst==0.15.0\n"
        elif case == "pure":
            files[wheel] = files[wheel].replace(b"false", b"true")
        elif case == "tag":
            files[wheel] = files[wheel].replace(
                b"cp310-abi3-linux_x86_64", b"py3-none-any"
            )
        elif case == "extension":
            files["foreign.so"] = b"unexpected"

    wheel_dir, owner = wheel_fixture(tmp_path, change=mutate)
    if case == "duplicate_wheel":
        (wheel_dir / "foreign.whl").write_bytes(b"unexpected")
    with pytest.raises(ValueError):
        CHECKS["check_wheel"](wheel_dir, owner)
