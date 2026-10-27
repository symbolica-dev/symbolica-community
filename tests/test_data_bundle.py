"""Exercise the installed bundle and recoverable external-data boundary."""

import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import sys

from symbolica import data

STIX_FONT = "render/fonts/STIXTwoMath-Regular.woff2"


def test_resource_distributions_have_no_file_collisions():
    owned = set()
    for name, bundle in (
        ("symbolica", None),
        ("symbolica-fonts", "fonts"),
        ("symbolica-vakint-data", "vakint-data"),
    ):
        distribution = importlib.metadata.distribution(name)
        assert "Root-Is-Purelib: false" in distribution.read_text("WHEEL")
        files = {str(path) for path in distribution.files}
        assert not owned.intersection(files), (name, owned.intersection(files))
        owned.update(files)
        if bundle:
            payloads = {path for path in files if ".dist-info/" not in path}
            assert payloads
            assert all(path.startswith(f"symbolica/_data/{bundle}/") for path in payloads)
            assert f"symbolica/_data/{bundle}/manifest.json" in payloads


def test_font_bundle_contains_only_the_browser_math_font():
    distribution = importlib.metadata.distribution("symbolica-fonts")
    payloads = {
        str(path) for path in distribution.files if ".dist-info/" not in str(path)
    }
    manifest_path = "symbolica/_data/fonts/manifest.json"
    assert payloads == {manifest_path, f"symbolica/_data/fonts/{STIX_FONT}"}
    manifest = json.loads(distribution.locate_file(manifest_path).read_text())
    assert [asset["name"] for asset in manifest["assets"]] == [STIX_FONT]
    assert [
        asset["name"]
        for asset in data.manifest()["assets"]
        if asset["name"].startswith("render/fonts/")
    ] == [STIX_FONT]


def test_installed_data_matches_the_compiled_consumer_catalog():
    asyncio.run(data.prepare("render/fonts"))
    for asset in data.manifest()["assets"]:
        if asset["name"].startswith("render/fonts/"):
            payload = data.read(asset["name"])
            assert len(payload) == asset["size"]
            assert hashlib.sha256(payload).hexdigest() == asset["sha256"]
    # Numerical data remains text, including the original precision markers.
    expression = data.read("vakint/numerics/matad_numerics/0000.txt").decode()
    assert len(expression) > 80_000
    assert "`20096." in expression


def test_missing_or_corrupt_data_can_be_supplied_without_restarting(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from pathlib import Path
import asyncio
import sys
import symbolica
from symbolica import E, S, data

name = 'render/fonts/STIXTwoMath-Regular.woff2'
assert not next(a for a in data.manifest()['assets'] if a['name'] == name)['loaded']
data.set_directory(sys.argv[1])
assert E('1/(1+x^2)').integrate(S('x')) == E('atan(x)')
assert not next(a for a in data.manifest()['assets'] if a['name'] == name)['loaded']
try:
    asyncio.run(data.prepare(name, cache_dir=Path(sys.argv[1]) / 'cache'))
except FileNotFoundError as error:
    assert 'pip install symbolica-fonts' in str(error)
else:
    raise AssertionError('Missing resource was accepted')
for operation in [lambda: data.read(name), lambda: data.install(name, b'damaged')]:
    try:
        operation()
    except ValueError:
        pass
    else:
        raise AssertionError('Missing/corrupt resource was accepted')
assert not next(a for a in data.manifest()['assets'] if a['name'] == name)['loaded']
payload = (Path(symbolica.__file__).parent / '_data' / 'fonts' / name).read_bytes()
data.install(name, payload)
assert data.read(name) == payload
assert next(a for a in data.manifest()['assets'] if a['name'] == name)['loaded']
""",
            str(tmp_path),
        ],
        check=True,
    )
