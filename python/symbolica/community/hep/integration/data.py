"""Fetch authenticated example mathematics without bundling it in the extension."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from urllib.request import urlopen

from symbolica.community.hep_integration_native import (
    InvalidInputError, higgs_jet_data_manifest, install_higgs_jet_data,
)


def _default_cache() -> Path:
    if sys.platform == "emscripten":
        return Path("/tmp/symbolica-higgs-jet-data")
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return root / "symbolica" / "higgs-jet-data"


def _download(url: str) -> str:
    with urlopen(url, timeout=60) as response:
        return response.read().decode("utf-8")


async def load_higgs_jet_data(
    *, form_factors: bool = False, cache_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Load the published Higgs+jet inputs on demand in native Python or WASM.

    URLs are pinned to the data revision and Rust authenticates the exact bytes
    before installing them. Files are cached by their content hashes. General
    integral reduction and transport do not require this download. Set
    ``form_factors=True`` before constructing ``HiggsJetFormFactorProjector``
    to fetch the additional 5.5 MB coefficient document.
    """
    directory = _default_cache() if cache_dir is None else Path(cache_dir)
    pending = {
        name: (url, digest)
        for name, (url, digest, loaded) in higgs_jet_data_manifest().items()
        if not loaded and (form_factors or name != "generic-form-factors.json")
    }
    if not pending:
        return
    directory.mkdir(parents=True, exist_ok=True)
    downloads = {}
    for name, (url, digest) in pending.items():
        path = directory / f"{digest}-{name}"
        if path.exists():
            try:
                install_higgs_jet_data({name: path.read_text(encoding="utf-8")})
                continue
            except (InvalidInputError, UnicodeDecodeError):
                path.unlink()
        downloads[name] = (url, path)

    async def fetch(name: str, url: str) -> tuple[str, str]:
        if sys.platform == "emscripten":
            from pyodide.http import pyfetch
            response = await pyfetch(url)
            if not response.ok:
                raise OSError(f"Higgs-jet data download failed: HTTP {response.status}: {url}")
            text = await response.string()
        else:
            text = await asyncio.to_thread(_download, url)
        return name, text

    documents = dict(await asyncio.gather(*(fetch(name, url) for name, (url, _) in downloads.items())))
    install_higgs_jet_data(documents)
    # Write only validated content. Interrupted writes are detected and fetched
    # again by the checksum check on the next call.
    for name, text in documents.items():
        downloads[name][1].write_text(text, encoding="utf-8")
