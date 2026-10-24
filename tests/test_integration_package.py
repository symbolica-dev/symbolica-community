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
        {"id": root["id"], "deps": [edge(build, "build"), edge(disabled), edge(dev, "dev")], "features": ["native", "wasm"]},
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
        assert command[command.index("--format") + 1] == "{p}|{f}"
        return "host v1.0.0|wasm\nblake3 v1.0.0|std\ncpufeatures v1.0.0|\n"

    monkeypatch.setattr(CHECK, "output", output)
    active, target_nodes, _ = CHECK.active_graph("wasm32-unknown-emscripten", ["--no-default-features", "--features", "wasm"])
    assert {p["name"] for p in active} == {"host", "blake3", "cpufeatures"}
    assert target_nodes[root["id"]]["features"] == ["wasm"]
    # A tree entry reachable only through a dev edge must still fail closed.
    monkeypatch.setattr(CHECK, "output", lambda command: output(command) + ("dev-only v1.0.0|\n" if command[1] == "tree" else ""))
    with pytest.raises(AssertionError, match="tree_only.*dev-only"):
        CHECK.active_graph("wasm32-unknown-emscripten", [])


def browser_graph():
    algebra = CHECK.algebra_sources()
    hepkit = "git+https://github.com/alphal00p/gammaloop?branch=preserve-canonical-numerator-interfaces"
    packages = [package(name, algebra[name] + "#" + algebra[name].rsplit("=", 1)[1]) for name in CHECK.ALGEBRA]
    packages += [package(name, hepkit + "#" + "b" * 40) for name in CHECK.HEPKIT]
    rustred = "git+https://github.com/alphal00p/rustred?branch=main"
    packages += [package(name, rustred + "#" + "c" * 40) for name in CHECK.RUSTRED]
    packages += [package(name) for name in ("pyo3", "hyperbolica", "symbolica-amflow")]
    nodes = {p["id"]: {"features": []} for p in packages}
    bridge = next(p for p in packages if p["name"] == "rustred-feynkit")
    nodes[bridge["id"]]["features"] = ["campaign-api", "wasm"]
    flow = next(p for p in packages if p["name"] == "symbolica-amflow")
    nodes[flow["id"]]["features"] = ["automatic", "python_wasm", "wasm"]
    root = {"dependencies": [
        {"name": "feynkit-py", "source": hepkit},
        {"name": "rustred-feynkit", "source": rustred},
    ]}
    return packages, nodes, root


@pytest.mark.parametrize("name", CHECK.ALGEBRA)
def test_algebra_rejects_sources_outside_the_host_declaration(name):
    packages, nodes, root = browser_graph()
    next(p for p in packages if p["name"] == name)["source"] = "git+foreign#" + "a" * 40
    with pytest.raises(AssertionError, match="algebra source differs"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


def test_numeric_owners_still_require_one_revision():
    packages, nodes, root = browser_graph()
    graphica = next(p for p in packages if p["name"] == "graphica")
    graphica["source"] = CHECK.algebra_sources()["graphica"] + "#" + "d" * 40
    with pytest.raises(AssertionError, match="numeric revisions differ"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


def test_browser_requires_automatic_evaluation_and_transport():
    packages, nodes, root = browser_graph()
    CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    packages[:] = [p for p in packages if p["name"] != "symbolica-amflow"]
    with pytest.raises(AssertionError, match="symbolica-amflow"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


@pytest.mark.parametrize("features, message", [
    (["python_wasm", "wasm"], "Automatic loop evaluation disabled"),
    (["automatic", "wasm"], "Browser loop API disabled"),
    (["automatic", "python_wasm", "wasm", "native"], "Native loop feature"),
])
def test_browser_rejects_disabled_or_native_automatic_evaluation(features, message):
    packages, nodes, root = browser_graph()
    flow = next(p for p in packages if p["name"] == "symbolica-amflow")
    nodes[flow["id"]]["features"] = features
    with pytest.raises(AssertionError, match=message):
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


def test_browser_requires_rustred_with_one_owner_and_portable_features():
    packages, nodes, root = browser_graph()
    core = next(p for p in packages if p["name"] == "rustred")
    nodes[core["id"]]["features"] = ["native"]
    with pytest.raises(AssertionError, match="native RustRed feature"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    nodes[core["id"]]["features"] = []
    core["source"] = "git+foreign"
    with pytest.raises(AssertionError, match="mixed RustRed"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    packages.remove(core)
    with pytest.raises(AssertionError, match="rustred"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


def test_duplicate_and_mixed_owners_still_fail():
    packages, nodes, root = browser_graph()
    packages.append(package("symbolica", "git+foreign"))
    with pytest.raises(AssertionError, match="symbolica.*foreign"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    packages.pop()
    next(p for p in packages if p["name"] == "feynkit-model")["source"] = "git+foreign"
    with pytest.raises(AssertionError, match="mixed HEPKit owners"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


def test_full_graph_requires_the_same_native_owner():
    packages, nodes, root = browser_graph()
    CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    graph = next(p for p in packages if p["name"] == "linnet-py")
    graph["source"] = "git+foreign#" + "d" * 40
    with pytest.raises(AssertionError, match="mixed HEPKit owners"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
    packages.remove(graph)
    with pytest.raises(AssertionError, match="linnet-py"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)


def test_kernel_requires_the_same_immutable_numeric_revision():
    packages, nodes, root = browser_graph()
    symbolica = next(p for p in packages if p["name"] == "symbolica")
    symbolica["source"] = CHECK.algebra_sources()["symbolica"] + "#" + "d" * 40
    with pytest.raises(AssertionError, match="kernel and numeric revisions differ"):
        CHECK.check_graph("pyodide", packages, nodes, root, community=True, native=False)
