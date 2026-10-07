"""Check wheel namespaces and Cargo ownership in native and browser builds."""

import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ALGEBRA = ("symbolica", "numerica", "graphica")
HEPKIT = (
    "feynkit-amplitude", "feynkit-cff", "feynkit-generator", "feynkit-graph",
    "feynkit-kinematics", "feynkit-model", "feynkit-py", "feynkit-tensor",
    "feynkit-ufo", "linnet", "spenso", "spenso-macros", "spenso-hep-lib",
    "idenso", "spynso3", "symbolica-utils",
)
OPTIONAL_HEPKIT = (
    "gammaloop-workspace-hack", "linnest", "linnet-py", "typst-renderer",
    "three-dimensional-reps", "kurvst",
)
RUSTRED = ("rustred", "rustred-order", "rustred-app", "rustred-feynkit", "rustred-python")
NATIVE_ONLY = ("vakint", "oneloop", "oneloop-python")


def output(command):
    return subprocess.check_output(command, text=True)


def active_graph(target, feature_args):
    """Traverse metadata, retaining packages activated by Cargo's target resolver.

    Metadata over-approximates weak optional and target-specific features. Keep
    its unfiltered catalogue: filtering for WASM also drops host-only children of
    build dependencies (for example BLAKE3's x86 cpufeatures). Cargo tree supplies
    actual active names for both host and target, excluding dev dependencies.
    Target-active features also come from Cargo tree: metadata's resolver node
    may include native features activated only by an inactive Vakint dependency.
    Exact sources still come from metadata, never abbreviated tree output or
    Cargo.lock.
    """
    metadata = json.loads(output([
        "cargo", "metadata", "--locked", "--format-version", "1", *feature_args,
    ]))
    tree = output([
        "cargo", "tree", "--locked", "--target", target, *feature_args,
        "--edges", "normal,build", "--prefix", "none", "--format", "{p}|{f}",
    ])
    target_features = {}
    for line in tree.splitlines():
        if not line:
            continue
        package_label, separator, features = line.partition("|")
        assert separator, ("Cargo tree omitted target features", line)
        name = package_label.split(" ", 1)[0]
        target_features.setdefault(name, set()).update(
            feature for feature in features.removesuffix(" (*)").strip().split(",")
            if feature
        )
    names = set(target_features)
    packages = {p["id"]: p for p in metadata["packages"]}
    nodes = {n["id"]: n for n in metadata["resolve"]["nodes"]}
    root = metadata["resolve"]["root"]
    assert root is not None, "run from the community host workspace"
    pending, reachable = [root], set()
    while pending:
        package_id = pending.pop()
        if package_id in reachable or packages[package_id]["name"] not in names:
            continue
        reachable.add(package_id)
        pending.extend(
            dep["pkg"] for dep in nodes[package_id]["deps"]
            if any(kind["kind"] != "dev" for kind in dep["dep_kinds"])
        )
    active = [packages[p] for p in reachable]
    active_names = {p["name"] for p in active}
    assert names == active_names, (
        "Cargo tree/metadata traversal disagrees",
        {"tree_only": sorted(names - active_names), "metadata_only": sorted(active_names - names)},
    )
    nodes = {
        package_id: {**node, "features": sorted(target_features[packages[package_id]["name"]])}
        if package_id in reachable else node
        for package_id, node in nodes.items()
    }
    return active, nodes, packages[root]


def singleton(packages, name):
    matches = [p for p in packages if p["name"] == name]
    assert len(matches) == 1, (name, [p["id"] for p in matches])
    return matches[0]


def owner(package):
    """Source revision, or common root of a local crates/ checkout."""
    if package["source"] is not None:
        return package["source"]
    manifest = Path(package["manifest_path"]).resolve()
    assert manifest.parent.parent.name == "crates", (package["name"], manifest)
    return str(manifest.parent.parent.parent)


def declared_source(root, name):
    matches = [d for d in root["dependencies"] if d["name"] == name]
    assert len(matches) == 1, (name, matches)
    return matches[0]["source"]


def check_graph(label, packages, nodes, root, *, community, native):
    algebra = {singleton(packages, name)["source"] for name in ALGEBRA}
    assert len(algebra) == 1, (label, "algebra revisions differ", algebra)
    source = next(iter(algebra))
    assert source and re.fullmatch(
        r"git\+https://github.com/symbolica-dev/symbolica\?branch=community#[0-9a-f]{40}",
        source,
    ), (label, "expected official Symbolica community sources", source)
    singleton(packages, "pyo3")
    symbolica = singleton(packages, "symbolica")
    assert "faster_alloc" not in nodes[symbolica["id"]]["features"], "host allocator policy changed"
    names = {p["name"] for p in packages}
    if not community:
        forbidden = (*HEPKIT, *OPTIONAL_HEPKIT, "hyperbolica", "symbolica-amflow", *RUSTRED, *NATIVE_ONLY)
        assert not names.intersection(forbidden), (
            label, "community crates in core-only build", names.intersection(forbidden),
        )
    else:
        owners = {
            owner(singleton(packages, name)) for name in (*HEPKIT, *OPTIONAL_HEPKIT)
            if name in HEPKIT or name in names
        }
        assert len(owners) == 1, (label, "mixed HEPKit owners", owners)
        hepkit_owner = next(iter(owners))
        feynkit_source = singleton(packages, "feynkit-py")["source"]
        if feynkit_source is not None:
            assert feynkit_source.startswith(declared_source(root, "feynkit-py") + "#"), (
                label, "HEPKit source differs from the host declaration", feynkit_source,
            )
        singleton(packages, "hyperbolica")
        # Supplied-boundary transport is shared by native and browser hosts.
        singleton(packages, "symbolica-amflow")
        reducers = {owner(singleton(packages, name)) for name in RUSTRED}
        assert len(reducers) == 1, (label, "mixed RustRed core/app/bridge", reducers)
        reducer_source = next(iter(reducers))
        assert reducer_source.startswith("git+https://github.com/alphal00p/rustred?"), (
            label, "expected official RustRed source", reducer_source,
        )
        assert reducer_source.startswith(declared_source(root, "rustred-feynkit") + "#"), (
            label, "RustRed source differs from host declaration", reducer_source,
        )
        bridge = singleton(packages, "rustred-feynkit")
        assert "campaign-api" in nodes[bridge["id"]]["features"], "RustRed campaign API disabled"
        for name in RUSTRED:
            package = singleton(packages, name)
            features = nodes[package["id"]]["features"]
            assert "reconstruction" not in features, (
                label, "experimental RustRed reconstruction enabled", name,
            )
            if not native:
                assert "native" not in features, (label, "native RustRed feature in browser build", name)
        if not native:
            assert "wasm" in nodes[bridge["id"]]["features"], "RustRed WASM API disabled"
        if native:
            vakint = singleton(packages, "vakint")
            assert owner(vakint) != hepkit_owner, "Vakint must retain its separate owner revision"
            if vakint["source"] is not None:
                declared = declared_source(root, "vakint")
                assert declared and re.search(r"\?rev=[0-9a-f]{40}$", declared), (
                    "Vakint requires an immutable host pin", declared,
                )
                assert vakint["source"] == declared + "#" + declared.rsplit("=", 1)[1], (
                    "Vakint resolved outside its declared pin", vakint["source"], declared,
                )
    if not native:
        forbidden = (*NATIVE_ONLY, "gmp-mpfr-sys", "rug")
        assert not names.intersection(forbidden), (
            label, "native-only crates in browser build", names.intersection(forbidden),
        )
    print(f"{label}: shared sources and {len(packages)} active packages checked")


def check_wheel(wheel):
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        base = "symbolica/community/hepkit/integration/"
        assert {base + "__init__.py", base + "__init__.pyi", "symbolica/py.typed"} <= names
        source = archive.read(base + "__init__.pyi").decode()
        assert "class IntegrationOptions" in source and "class IntegrationError" in source
        assert "class Expression" not in source
        rustred_base = "symbolica/community/hepkit/rustred/"
        assert {rustred_base + "__init__.py", rustred_base + "__init__.pyi"} <= names
        rustred_stub = archive.read(rustred_base + "__init__.pyi").decode()
        assert "def execution_capabilities(" in rustred_stub
        assert "class CandidateGenerationSession" in rustred_stub
        assert not any(n.startswith("hyperbolica/") for n in names)
        base = "symbolica/community/hep/integration/"
        assert {base + "__init__.py", base + "__init__.pyi"} <= names
        source = archive.read(base + "__init__.pyi").decode()
        for name in (
            "IntegralEvaluator", "PreparedIntegralFamily", "KinematicTransport",
            "BoundaryCache", "EvaluationOptions", "DifferentialSystem",
            "BoundaryData", "LaurentExpansion", "TransportResult",
        ):
            assert f"class {name}" in source, name
        assert "class Expression" not in source


def main():
    host = next(
        line.removeprefix("host: ") for line in output(["rustc", "-vV"]).splitlines()
        if line.startswith("host: ")
    )
    variants = (
        ("native", host, [], True, True),
        ("stubgen", host, ["--no-default-features", "--features", "python_stubgen"], True, True),
        ("core-only", host, ["--no-default-features", "--features", "native"], False, True),
        ("wasm", "wasm32-unknown-unknown", ["--no-default-features", "--features", "wasm"], True, False),
        ("pyodide", "wasm32-unknown-emscripten", ["--no-default-features", "--features", "wasm"], True, False),
        ("wasm-core", "wasm32-unknown-unknown", ["--no-default-features", "--features", "wasm-core"], False, False),
    )
    for label, target, features, community, native in variants:
        check_graph(label, *active_graph(target, features), community=community, native=native)
    for wheel in sys.argv[1:]:
        check_wheel(wheel)
    print("Integration wheel and shared-kernel checks passed")


if __name__ == "__main__":
    main()
