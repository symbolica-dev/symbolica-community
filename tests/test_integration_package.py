"""Keep host build dependencies visible without admitting foreign owners."""

import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "integration_package_check",
    Path(__file__).parents[1] / ".github/scripts/check_integration_package.py",
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def package(name, source="registry+test"):
    return {"id": f"{source}#{name}", "name": name, "source": source}


def test_wasm_graph_retains_host_build_dependencies_and_filters_inactive_names(monkeypatch):
    root, build, host, disabled, dev = [
        package(name) for name in ("host", "blake3", "cpufeatures", "rug", "dev-only")
    ]

    def edge(p, kind=None, target=None):
        return {"pkg": p["id"], "dep_kinds": [{"kind": kind, "target": target}]}

    nodes = [
        {"id": root["id"], "deps": [edge(build, "build"), edge(disabled), edge(dev, "dev")]},
        {"id": build["id"], "deps": [edge(host, target='cfg(target_arch = "x86_64")')]},
        *({"id": p["id"], "deps": []} for p in (host, disabled, dev)),
    ]
    metadata = {"packages": [root, build, host, disabled, dev], "resolve": {"root": root["id"], "nodes": nodes}}

    def output(command):
        if command[1] == "metadata":
            assert "--filter-platform" not in command
            return json.dumps(metadata)
        assert command[command.index("--target") + 1] == "wasm32-unknown-emscripten"
        assert command[command.index("--edges") + 1] == "normal,build"
        return "host v1.0.0\nblake3 v1.0.0\ncpufeatures v1.0.0\n"

    monkeypatch.setattr(CHECK, "output", output)
    active, _, _ = CHECK.active_graph("wasm32-unknown-emscripten", ["--no-default-features", "--features", "wasm"])
    assert {p["name"] for p in active} == {"host", "blake3", "cpufeatures"}
    # A tree entry reachable only through a dev edge must still fail closed.
    monkeypatch.setattr(CHECK, "output", lambda command: output(command) + ("dev-only v1.0.0\n" if command[1] == "tree" else ""))
    with pytest.raises(AssertionError, match="tree_only.*dev-only"):
        CHECK.active_graph("wasm32-unknown-emscripten", [])


def browser_graph():
    algebra = "git+https://github.com/symbolica-dev/symbolica?branch=community#" + "a" * 40
    hepkit = "git+https://github.com/ValentinHirschi/gammaloop?rev=" + "b" * 40
    packages = [package(name, algebra) for name in CHECK.ALGEBRA]
    packages += [package(name, hepkit + "#" + "b" * 40) for name in CHECK.HEPKIT]
    packages += [package(name) for name in ("pyo3", "hyperbolica", "symbolica-amflow")]
    nodes = {p["id"]: {"features": []} for p in packages}
    root = {"dependencies": [{"name": "feynkit-py", "source": hepkit}]}
    return packages, nodes, root


def test_browser_requires_supplied_transport():
    packages, nodes, root = browser_graph()
    CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    packages[:] = [p for p in packages if p["name"] != "symbolica-amflow"]
    with pytest.raises(AssertionError, match="symbolica-amflow"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


@pytest.mark.parametrize("name", [*CHECK.NATIVE_ONLY, "gmp-mpfr-sys", "rug"])
def test_browser_still_rejects_every_native_only_dependency(name):
    packages, nodes, root = browser_graph()
    packages.append(package(name))
    with pytest.raises(AssertionError, match="native-only crates"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


def test_core_build_still_excludes_transport():
    packages, nodes, root = browser_graph()
    packages = [p for p in packages if p["name"] in (*CHECK.ALGEBRA, "pyo3", "symbolica-amflow")]
    with pytest.raises(AssertionError, match="community crates in core-only build"):
        CHECK.check_graph("wasm-core", packages, nodes, root, community=False, native=False)


def test_duplicate_and_mixed_owners_still_fail():
    packages, nodes, root = browser_graph()
    packages.append(package("symbolica", "git+foreign"))
    with pytest.raises(AssertionError, match="symbolica.*foreign"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    packages.pop()
    next(p for p in packages if p["name"] == "feynkit-model")["source"] = "git+foreign"
    with pytest.raises(AssertionError, match="mixed HEPKit owners"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
