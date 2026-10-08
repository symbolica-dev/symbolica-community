"""Remote loading, authentication failures and cache reuse without live HTTP."""
import asyncio
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def loader(monkeypatch):
    payloads = {"integral-systems.json": '{"system": 1}', "generic-form-factors.json": '{"coefficients": 2}'}
    loaded = {}
    native = ModuleType("symbolica.community.hep_integration_native")
    class InvalidInputError(ValueError):
        pass
    native.InvalidInputError = InvalidInputError
    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()
    native.higgs_jet_data_manifest = lambda: {
        name: (f"https://example.invalid/pinned/{name}", digest(text), name in loaded)
        for name, text in payloads.items()
    }
    def install(documents):
        for name, text in documents.items():
            if text != payloads[name]:
                raise InvalidInputError("checksum mismatch")
        loaded.update(documents)
    native.install_higgs_jet_data = install
    monkeypatch.setitem(sys.modules, native.__name__, native)
    path = Path(__file__).parents[1] / "python/symbolica/community/hep/integration/data.py"
    spec = importlib.util.spec_from_file_location("higgs_jet_data_loader_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, payloads, loaded


def test_native_download_is_optional_and_cached_by_authenticated_content(monkeypatch, tmp_path):
    module, payloads, loaded = loader(monkeypatch)
    requests = []
    def download(url):
        requests.append(url)
        return payloads[url.rsplit("/", 1)[-1]]
    monkeypatch.setattr(module, "_download", download)
    asyncio.run(module.load_higgs_jet_data(cache_dir=tmp_path))
    assert set(loaded) == {"integral-systems.json"}
    assert len(requests) == 1
    asyncio.run(module.load_higgs_jet_data(cache_dir=tmp_path, form_factors=True))
    assert loaded == payloads
    assert len(requests) == 2
    loaded.clear()
    asyncio.run(module.load_higgs_jet_data(cache_dir=tmp_path, form_factors=True))
    assert loaded == payloads
    assert len(requests) == 2
    loaded.clear()
    next(tmp_path.glob("*-integral-systems.json")).write_text("corrupt")
    asyncio.run(module.load_higgs_jet_data(cache_dir=tmp_path, form_factors=True))
    assert loaded == payloads
    assert len(requests) == 3
    loaded.clear()
    next(tmp_path.glob("*-integral-systems.json")).write_bytes(b"\xff")
    asyncio.run(module.load_higgs_jet_data(cache_dir=tmp_path, form_factors=True))
    assert loaded == payloads
    assert len(requests) == 4


def test_browser_fetch_rejects_corrupt_content_before_caching(monkeypatch, tmp_path):
    module, payloads, loaded = loader(monkeypatch)
    monkeypatch.setattr(module.sys, "platform", "emscripten")
    http = ModuleType("pyodide.http")
    async def bad_fetch(url):
        async def string():
            return "corrupt"
        return SimpleNamespace(ok=True, status=200, string=string)
    http.pyfetch = bad_fetch
    monkeypatch.setitem(sys.modules, "pyodide.http", http)
    with pytest.raises(module.InvalidInputError, match="checksum"):
        asyncio.run(module.load_higgs_jet_data(cache_dir=tmp_path, form_factors=True))
    assert not loaded
    assert not list(tmp_path.iterdir())
    async def good_fetch(url):
        async def string():
            return payloads[url.rsplit("/", 1)[-1]]
        return SimpleNamespace(ok=True, status=200, string=string)
    http.pyfetch = good_fetch
    asyncio.run(module.load_higgs_jet_data(cache_dir=tmp_path, form_factors=True))
    assert loaded == payloads
    assert len(list(tmp_path.iterdir())) == 2
