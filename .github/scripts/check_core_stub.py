"""Check the community citation API and optional canonical Symbolica stub parity."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from zipfile import ZipFile


def check_core_stub(path: Path, canonical: Path | None = None) -> None:
    if path.suffix == ".whl":
        with ZipFile(path) as wheel:
            source = wheel.read("symbolica/core.pyi").decode("utf-8")
    else:
        source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    citations = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Citation"
    ]
    functions = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_citations"
    ]
    if len(citations) != 1 or len(functions) != 1:
        raise SystemExit(
            f"{path}: core.pyi must declare exactly one Citation class and get_citations function. "
            "Preserve the community citation API when regenerating or replacing the core stub."
        )
    function = functions[0]
    expected = ast.parse("def get_citations() -> list[Citation]: ...").body[0]
    if (
        ast.dump(function.args) != ast.dump(expected.args)
        or function.returns is None
        or ast.dump(function.returns) != ast.dump(expected.returns)
        or function.decorator_list
        or getattr(function, "type_params", [])
    ):
        raise SystemExit(f"{path}: expected get_citations() -> list[Citation].")
    if canonical is not None:
        reference = ast.parse(
            canonical.read_text(encoding="utf-8"), filename=str(canonical)
        )
        module.body.remove(function)
        if ast.dump(module) != ast.dump(reference):
            raise SystemExit(
                f"{path}: core.pyi differs from {canonical} beyond community get_citations."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+", help="core.pyi files or built wheels"
    )
    parser.add_argument(
        "--canonical", type=Path, help="canonical Symbolica stub to compare"
    )
    args = parser.parse_args()
    for path in args.paths:
        check_core_stub(path, args.canonical)
