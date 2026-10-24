"""Prepare verified external resources before synchronous native/WASM operations."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

from . import core

_directories: tuple[Path, ...] = ()

@lru_cache(maxsize=1)
def _compatibility_sources() -> dict:
    """Historical raw aliases share the generated native integrity catalog."""
    declaration = json.loads(
        Path(__file__).with_name("data_compatibility.json").read_text()
    )
    return {
        entry["name"]: declaration["remote_base_url"] + entry["source_path"]
        for entry in declaration["assets"]
    }



def manifest() -> dict:
    """Describe the resources expected by this engine, including loaded status."""
    return json.loads(core._data_manifest())


def set_directory(directory: str | os.PathLike[str]) -> None:
    """Use an extracted bundle; payloads remain unloaded until first needed."""
    set_directories(directory)


def set_directories(*directories: str | os.PathLike[str]) -> None:
    """Search several extracted resource bundles without copying their files."""
    global _directories
    _directories = tuple(Path(directory) for directory in directories)
    core._set_data_directories([str(directory) for directory in _directories])


def configure() -> None:
    """Locate the installed data distribution without reading large resources."""
    if directory := os.environ.get("SYMBOLICA_DATA_DIRECTORY"):
        set_directories(*directory.split(os.pathsep))
    else:
        package_data = Path(__file__).parent / "_data"
        set_directories(
            *(
                package_data / bundle
                for bundle in ("fonts", "vakint-data")
            )
        )


def install(name: str, payload: bytes) -> None:
    """Supply one resource; Rust checks its exact size and pinned checksum."""
    core._install_data_asset(name, payload)


def read(name: str) -> bytes:
    """Read verified bytes, including native-produced RustRed assets in WASM."""
    return core._read_data_asset(name)


async def prepare(
    *datasets: str,
    source: str | os.PathLike[str] | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Load named resources or prefixes from a directory or HTTP asset mirror.

    No download occurs during import or mathematical computation. Browser hosts
    may mount persistent storage at ``cache_dir``; the default browser cache is
    session-local. With no dataset arguments, prepare all resources supported
    by this particular native/WASM engine. Historical raw-data aliases require
    an explicit dataset name; they are never prepared by default or used as
    current RustRed reduction programs.
    """
    records = manifest()["assets"]
    compatibility = _compatibility_sources()
    for dataset in datasets:
        if not any(
            record["name"] == dataset
            or record["name"].startswith(dataset.rstrip("/") + "/")
            for record in records
        ):
            raise ValueError(f"unknown or unsupported dataset: {dataset}")
    pending = [
        record
        for record in records
        if not record["loaded"]
        and (datasets or record["name"] not in compatibility)
        and (
            not datasets
            or any(
                record["name"] == dataset
                or record["name"].startswith(dataset.rstrip("/") + "/")
                for dataset in datasets
            )
        )
    ]
    if not pending:
        return
    roots = _directories if source is None else (Path(source),)
    explicit_source = source is not None
    source = "" if source is None else os.fspath(source)
    remote = source.startswith(("https://", "http://"))
    directory = (
        Path(cache_dir)
        if cache_dir is not None
        else (
            Path("/tmp/symbolica-data")
            if sys.platform == "emscripten"
            else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
            / "symbolica"
            / "data"
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    # Limit simultaneous transfers and WASM memory copies for large programs.
    for record in pending:
        name = record["name"]
        cached = directory / record["sha256"]
        if cached.exists():
            try:
                install(name, cached.read_bytes())
                continue
            except ValueError:
                cached.unlink()
        path = next((root / name for root in roots if (root / name).is_file()), None)
        url = (
            source.rstrip("/") + "/" + name if remote
            else compatibility.get(name) if path is None and not explicit_source
            else None
        )
        if url is not None:
            if sys.platform == "emscripten":
                from pyodide.http import pyfetch

                response = await pyfetch(url)
                response.raise_for_status()
                payload = await response.bytes()
            else:

                def download(url=url, size=record["size"]):
                    with urlopen(url, timeout=60) as response:
                        return response.read(size + 1)

                payload = await asyncio.to_thread(download)
        else:
            if path is None:
                package = (
                    "symbolica-fonts" if name.startswith("render/fonts/")
                    else "symbolica-vakint-data" if name.startswith("vakint/")
                    else "the resource wheel"
                )
                raise FileNotFoundError(
                    f"Install {package} with `python -m pip install {package}` or supply a source for {name}"
                )
            with path.open("rb") as stream:
                payload = stream.read(record["size"] + 1)
        install(name, payload)
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        try:
            temporary.replace(cached)
        finally:
            temporary.unlink(missing_ok=True)
