"""Exercise the installed bundle and recoverable external-data boundary."""

import asyncio
import hashlib
import importlib.metadata
import subprocess
import sys

from symbolica import data


def test_resource_distributions_have_no_file_collisions():
    owned = set()
    for name, bundle in (
        ("symbolica", None),
        ("symbolica-fonts", "fonts"),
        ("symbolica-vakint-data", "vakint-data"),
    ):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            if name != "symbolica-vakint-data":
                raise
            # The optional historical pure-data distribution remains supported;
            # current installs need only fonts and explicit raw-data requests.
            continue
        assert "Root-Is-Purelib: false" in distribution.read_text("WHEEL")
        files = {str(path) for path in distribution.files}
        assert not owned.intersection(files), (name, owned.intersection(files))
        owned.update(files)
        if bundle:
            payloads = {path for path in files if ".dist-info/" not in path}
            assert payloads
            assert all(path.startswith(f"symbolica/_data/{bundle}/") for path in payloads)
            assert f"symbolica/_data/{bundle}/manifest.json" in payloads


def test_installed_data_matches_the_compiled_consumer_catalog():
    asyncio.run(data.prepare("render/fonts"))
    for asset in data.manifest()["assets"]:
        if asset["name"].startswith("render/fonts/"):
            payload = data.read(asset["name"])
            assert len(payload) == asset["size"]
            assert hashlib.sha256(payload).hexdigest() == asset["sha256"]
    # Numerical data remains text, including the original precision markers.
    name = "vakint/numerics/matad_numerics/0000.txt"
    asyncio.run(data.prepare(name))
    expression = data.read(name).decode()
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

data.set_directory(sys.argv[1])
assert E('1/(1+x^2)').integrate(S('x')) == E('atan(x)')
name = 'render/fonts/NewCMMath-Book.otf'
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
""",
            str(tmp_path),
        ],
        check=True,
    )


def test_default_preparation_never_fetches_historical_reduction_packs(monkeypatch, tmp_path):
    before = {
        asset["name"]: asset["loaded"]
        for asset in data.manifest()["assets"]
        if asset["name"].startswith("vakint/")
    }
    assert before

    def no_implicit_download(*args, **kwargs):
        raise AssertionError("default preparation fetched compatibility data")

    monkeypatch.setattr(data, "urlopen", no_implicit_download)
    asyncio.run(data.prepare(cache_dir=tmp_path))
    after = {
        asset["name"]: asset["loaded"]
        for asset in data.manifest()["assets"]
        if asset["name"].startswith("vakint/")
    }
    assert after == before
