"""Reject mismatched graph binding and consumer revisions."""

import runpy
from pathlib import Path

import pytest

CHECKS = runpy.run_path(
    str(Path(__file__).parents[1] / ".github/scripts/check_native_graph_package.py")
)
REVISION = "a" * 40
REPOSITORY = "https://github.com/alphal00p/gammaloop"


def manifests(tmp_path):
    manifest, lock = tmp_path / "Cargo.toml", tmp_path / "Cargo.lock"
    pin = f'{{ git = "{REPOSITORY}", branch = "feynkit" }}'
    manifest.write_text(
        f"[dependencies]\nfeynkit-py = {pin}\nspynso3 = {pin}\nlinnet-py = {pin}\n"
    )
    lock.write_text(
        "".join(
            f'[[package]]\nname = "{name}"\nsource = "git+{REPOSITORY}?branch=feynkit#{REVISION}"\n'
            for name in ("feynkit-py", "linnet", "spynso3", "linnet-py")
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
