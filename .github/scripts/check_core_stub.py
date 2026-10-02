"""Reject source stubs or wheels that omit the community citation API."""

import argparse
import ast
from pathlib import Path
from zipfile import ZipFile


def check_core_stub(path: Path) -> None:
    if path.suffix == ".whl":
        with ZipFile(path) as wheel:
            source = wheel.read("symbolica/core.pyi").decode("utf-8")
    else:
        source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "get_citations"
        for node in module.body
    ):
        raise SystemExit(
            f"{path}: core.pyi is missing the top-level get_citations declaration. "
            "Preserve the community citation API when regenerating or replacing the core stub."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+", help="core.pyi files or built wheels")
    for path in parser.parse_args().paths:
        check_core_stub(path)
