"""Provenance guard for the Community graph binding library and its consumers."""

import argparse
import re
import subprocess
from pathlib import Path

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
    for name in ("feynkit-py", "linnet", "linnet-py", "spynso3"):
        matches = [p for p in packages if p["name"] == name]
        require(
            len(matches) == 1 and matches[0].get("source") == source,
            f"locked {name} differs from the manifest owner",
        )
    # The entire embedded owner workspace must share one revision.
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


def main():
    parser = argparse.ArgumentParser(
        description="Verify the canonical graph owner revision"
    )
    parser.add_argument("mode", choices=("pin", "owner"))
    parser.add_argument("--manifest", type=Path, default=Path("Cargo.toml"))
    parser.add_argument("--lock", type=Path, default=Path("Cargo.lock"))
    parser.add_argument("--owner", type=Path, default=Path("native-owners/hepkit"))
    args = parser.parse_args()
    repository, revision = owner_pin(args.manifest, args.lock)
    if args.mode == "pin":
        print(f"repository={repository.removeprefix('https://github.com/')}")
        print(f"revision={revision}")
    else:
        check_owner(args.owner, revision)


if __name__ == "__main__":
    main()
