"""Build provenance guards for the matching owner-native graph wheel in CI.

This is a separate Python >=3.10 test dependency, not the unrelated PyPI linnet
distribution or a change to the community kernel's Python >=3.9 policy.
"""

import argparse
import hashlib
import json
import re
import subprocess
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_toml(path):
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)


def owner_pin(manifest, lock):
    host = read_toml(manifest)
    pin = host["dependencies"]["feynkit-py"]
    repository = pin["git"]
    require(
        repository == "https://github.com/alphal00p/gammaloop",
        "unexpected native graph owner repository",
    )
    require(
        pin.get("branch") == "feynkit" and "rev" not in pin and "tag" not in pin,
        "native graph owner must use the feynkit branch",
    )
    declarations = list(host["dependencies"].values())
    declarations += [
        pin for table in host.get("patch", {}).values() for pin in table.values()
    ]
    for declaration in declarations:
        if isinstance(declaration, dict) and declaration.get("git") == repository:
            require(
                declaration.get("branch") == "feynkit"
                and "rev" not in declaration
                and "tag" not in declaration,
                "Linnet and HEPKit manifest owners differ",
            )
    locked = read_toml(lock)
    packages = locked["package"]
    owners = [p for p in packages if p["name"] == "feynkit-py"]
    require(len(owners) == 1, "locked feynkit-py differs from the manifest owner")
    source = owners[0].get("source", "")
    prefix = f"git+{repository}?branch=feynkit#"
    require(
        source.startswith(prefix)
        and re.fullmatch(r"[0-9a-f]{40}", source[len(prefix) :]),
        "locked feynkit-py differs from the manifest owner",
    )
    revision = source[len(prefix) :]
    for name in ("feynkit-py", "linnet"):
        matches = [p for p in packages if p["name"] == name]
        require(
            len(matches) == 1 and matches[0].get("source") == source,
            f"locked {name} differs from the manifest owner",
        )
    # Standalone Linnet bindings are built from this exact owner checkout, even
    # though they are intentionally absent from the host dependency graph.
    for package in packages:
        if package.get("source", "").startswith(f"git+{repository}?"):
            require(
                package["source"] == source,
                f"locked {package['name']} differs from the manifest owner",
            )
    return repository, revision


def check_owner(owner, revision):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(owner), *args], text=True
        ).strip()

    require(
        git("rev-parse", "HEAD") == revision,
        "native graph checkout differs from the host pin",
    )
    require(
        not git("status", "--porcelain", "--untracked-files=no"),
        "native graph owner has tracked source changes",
    )


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_wheel(wheel_dir, owner):
    wheels = sorted(Path(wheel_dir).glob("*.whl"))
    require(len(wheels) == 1, "expected exactly one freshly built native graph wheel")
    wheel = wheels[0].resolve()
    project = read_toml(owner / "crates/linnet-py/pyproject.toml")["project"]
    require(
        project["name"] == "linnet" and project["dependencies"] == [],
        "owner graph package contract changed; review dependencies explicitly",
    )
    require(
        project["requires-python"] == ">=3.10",
        "owner Python policy changed; review CI matrix",
    )
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = [n for n in names if n.endswith(".dist-info/METADATA")]
        require(len(metadata) == 1, "expected exactly one wheel metadata record")
        info = BytesParser().parsebytes(archive.read(metadata[0]))
        for field, key in (
            ("Name", "name"),
            ("Version", "version"),
            ("Requires-Python", "requires-python"),
        ):
            require(
                info.get_all(field) == [project[key]],
                f"wheel {field} differs from the owner",
            )
        require(
            not info.get_all("Requires-Dist"),
            "owner graph wheel must not install dependencies",
        )
        wheel_info = BytesParser().parsebytes(
            archive.read(metadata[0].removesuffix("METADATA") + "WHEEL")
        )
        require(wheel_info.get("Root-Is-Purelib") == "false", "expected a native wheel")
        tags = wheel_info.get_all("Tag", [])
        require(
            tags
            and all(
                re.fullmatch(
                    r"cp310-abi3-(?:manylinux[^ ]+|linux_[^ ]+|macosx_[^ ]+)", tag
                )
                for tag in tags
            ),
            "expected a Linux/macOS cp310-abi3 owner wheel",
        )
        extensions = [n for n in names if n.endswith((".so", ".pyd", ".dylib"))]
        require(
            extensions == ["linnet/linnet.abi3.so"],
            "unexpected native graph extension layout",
        )
        native_hash = digest(archive.read(extensions[0]))
    return {
        "wheel": str(wheel),
        "sha256": digest(wheel.read_bytes()),
        "native": extensions[0],
        "native_sha256": native_hash,
        "version": project["version"],
        "tags": tags,
    }


def check_installed(record):
    import importlib.metadata

    import linnet

    distribution = importlib.metadata.distribution("linnet")
    require(
        distribution.version == record["version"],
        "installed graph version differs from the wheel",
    )
    native = Path(distribution.locate_file(record["native"])).resolve()
    require(
        native.parent == Path(linnet.__file__).resolve().parent,
        "imported graph package differs from its installed distribution",
    )
    require(
        digest(native.read_bytes()) == record["native_sha256"],
        "installed native graph extension differs from the verified wheel",
    )
    require(
        importlib.metadata.distribution("symbolica").version,
        "community host is not installed",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pin", "owner", "wheel", "installed"))
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--lock", type=Path, default=Path("Cargo.lock"))
    parser.add_argument("--owner", type=Path, default=Path("native-owners/hepkit"))
    parser.add_argument("--wheel-dir", type=Path, default=Path("linnet-dist"))
    parser.add_argument("--record", type=Path, default=Path("target/linnet-wheel.json"))
    parser.add_argument(
        "--requirements", type=Path, default=Path("target/linnet-wheel.txt")
    )
    args = parser.parse_args()
    repository, revision = owner_pin(args.manifest, args.lock)
    if args.mode == "pin":
        print(f"repository={repository.removeprefix('https://github.com/')}")
        print(f"revision={revision}")
        return
    check_owner(args.owner, revision)
    if args.mode == "wheel":
        record = check_wheel(args.wheel_dir, args.owner)
        record.update(repository=repository, revision=revision)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n")
        args.requirements.parent.mkdir(parents=True, exist_ok=True)
        args.requirements.write_text(
            f"linnet @ {Path(record['wheel']).as_uri()} --hash=sha256:{record['sha256']}\n"
        )
        print(json.dumps(record, indent=2))
    elif args.mode == "installed":
        record = json.loads(args.record.read_text())
        require(
            (record["repository"], record["revision"]) == (repository, revision),
            "wheel record differs from the current host owner",
        )
        check_installed(record)
        print("Installed native graph bytes match the verified owner wheel")


if __name__ == "__main__":
    main()
