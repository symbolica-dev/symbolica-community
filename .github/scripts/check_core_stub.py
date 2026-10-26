"""Check Community core stub additions and optional canonical Symbolica parity."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from zipfile import ZipFile

REFLECTED_OPERATORS = {
    "__add__": ("_SupportsRAdd", "__radd__"),
    "__sub__": ("_SupportsRSub", "__rsub__"),
    "__mul__": ("_SupportsRMul", "__rmul__"),
    "__truediv__": ("_SupportsRTrueDiv", "__rtruediv__"),
    "__pow__": ("_SupportsRPow", "__rpow__"),
}
SUPPORT_DECLARATIONS = {
    "_TypeVar": "from typing import TypeVar as _TypeVar",
    "_ArithmeticT_co": '_ArithmeticT_co = _TypeVar("_ArithmeticT_co", bound="Expression", covariant=True)',
    **{
        protocol: (
            f"class {protocol}(Protocol[_ArithmeticT_co]):\n"
            f"    def {reflected}(self, left: Expression, /) -> _ArithmeticT_co: ..."
        )
        for protocol, reflected in REFLECTED_OPERATORS.values()
    },
}
SCALAR_OPERAND = "Expression | int | float | complex | Float | ComplexFloat | Decimal"
RESOURCE_DECLARATIONS = (
    "def _set_data_directories(directories: list[str]) -> None: ...",
    "def _data_manifest() -> str: ...",
    "def _install_data_asset(name: str, payload: bytes) -> None: ...",
    "def _read_data_asset(name: str) -> bytes: ...",
)


def _same_signature(actual: ast.FunctionDef, expected: ast.FunctionDef) -> bool:
    return (
        ast.dump(actual.args) == ast.dump(expected.args)
        and actual.returns is not None
        and ast.dump(actual.returns) == ast.dump(expected.returns)
        and [ast.dump(node) for node in actual.decorator_list]
        == [ast.dump(node) for node in expected.decorator_list]
        and not getattr(actual, "type_params", [])
    )


def _declared_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    targets = (
        node.targets
        if isinstance(node, ast.Assign)
        else [node.target]
        if isinstance(node, ast.AnnAssign)
        else []
    )
    return {
        name.id
        for target in targets
        for name in ast.walk(target)
        if isinstance(name, ast.Name) and isinstance(name.ctx, ast.Store)
    }


def _check_support_declarations(
    module: ast.Module, path: Path, *, missing_ok: bool = False
) -> dict[str, ast.stmt | None]:
    # Core-only distributions need these annotations without optional modules.
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = ("symbolica." if node.level == 1 else "") + (node.module or "")
            imports = [
                base,
                *(base.rstrip(".") + "." + alias.name for alias in node.names),
            ]
        else:
            continue
        if any(
            name == "symbolica.community" or name.startswith("symbolica.community.")
            for name in imports
        ):
            raise SystemExit(
                f"{path}: core arithmetic annotations must not import optional Community modules."
            )
    found = {}
    for name, declaration in SUPPORT_DECLARATIONS.items():
        expected = ast.parse(declaration).body[0]
        nodes = [node for node in module.body if name in _declared_names(node)]
        if (
            (not nodes and not missing_ok)
            or len(nodes) > 1
            or (nodes and ast.dump(nodes[0]) != ast.dump(expected))
        ):
            raise SystemExit(
                f"{path}: expected exactly one unchanged reflected arithmetic helper {name}; "
                "cannot repair a malformed or duplicate declaration."
            )
        found[name] = nodes[0] if nodes else None
    return found


def _operator_declarations(name: str) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    parameter = "exp" if name == "__pow__" else "other"
    protocol = REFLECTED_OPERATORS[name][0]
    reflected = ast.parse(
        f"@overload\ndef {name}(self, {parameter}: {protocol}[_ArithmeticT_co]) -> _ArithmeticT_co: ..."
    ).body[0]
    scalar = ast.parse(
        f"@overload\ndef {name}(self, {parameter}: {SCALAR_OPERAND}) -> Expression: ..."
    ).body[0]
    return reflected, scalar


def _check_reflected_arithmetic(module: ast.Module, path: Path) -> None:
    """Validate the refinements, then undo only those additions for parity."""
    support = _check_support_declarations(module, path)
    expressions = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Expression"
    ]
    if len(expressions) != 1:
        raise SystemExit(f"{path}: core.pyi must declare exactly one Expression class.")
    expression = expressions[0]
    for name in REFLECTED_OPERATORS:
        methods = [
            node
            for node in expression.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        parameter = "exp" if name == "__pow__" else "other"
        expected_reflected, expected_scalar = _operator_declarations(name)
        if (
            len(methods) != 2
            or not all(isinstance(node, ast.FunctionDef) for node in methods)
            or ast.dump(methods[0]) != ast.dump(expected_reflected)
            or not _same_signature(methods[1], expected_scalar)
            or expression.body.index(methods[1])
            != expression.body.index(methods[0]) + 1
        ):
            raise SystemExit(
                f"{path}: Expression.{name} must have exactly two adjacent @overload declarations: "
                f"first (self, {parameter}: {REFLECTED_OPERATORS[name][0]}[_ArithmeticT_co]) -> _ArithmeticT_co, "
                f"then the unchanged scalar ({parameter}: {SCALAR_OPERAND}) -> Expression fallback. "
                "Preserve the Community reflected arithmetic refinements when refreshing core.pyi."
            )
        expression.body.remove(methods[0])
        methods[1].decorator_list.clear()
    for node in support.values():
        module.body.remove(node)


def _check_resource_transport(module: ast.Module, path: Path) -> None:
    """Allow only the known host transport declarations in canonical parity."""
    for declaration in RESOURCE_DECLARATIONS:
        expected = ast.parse(declaration).body[0]
        functions = [
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == expected.name
        ]
        if len(functions) > 1 or (
            functions and ast.dump(functions[0]) != ast.dump(expected)
        ):
            raise SystemExit(
                f"{path}: expected the Community resource declaration {declaration}"
            )
        if functions:
            module.body.remove(functions[0])


def _check_source(source: str, path: Path, canonical: Path | None = None) -> None:
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
    if not _same_signature(function, expected):
        raise SystemExit(f"{path}: expected get_citations() -> list[Citation].")
    _check_reflected_arithmetic(module, path)
    _check_resource_transport(module, path)
    if canonical is not None:
        reference = ast.parse(
            canonical.read_text(encoding="utf-8"), filename=str(canonical)
        )
        module.body.remove(function)
        if ast.dump(module) != ast.dump(reference):
            raise SystemExit(
                f"{path}: core.pyi differs from {canonical} beyond the validated Community "
                "citation, resource transport, and reflected arithmetic additions."
            )


def check_core_stub(path: Path, canonical: Path | None = None) -> None:
    if path.suffix == ".whl":
        with ZipFile(path) as wheel:
            source = wheel.read("symbolica/core.pyi").decode("utf-8")
    else:
        source = path.read_text(encoding="utf-8")
    _check_source(source, path, canonical)


def fix_core_stub(path: Path, canonical: Path | None = None) -> None:
    """Restore missing reflected arithmetic refinements without rewriting existing stub text."""
    if path.suffix == ".whl":
        raise SystemExit(
            f"{path}: --fix accepts source stubs only; rebuild the wheel afterward."
        )
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits: list[tuple[int, str]] = []
    support = _check_support_declarations(module, path, missing_ok=True)
    expressions = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Expression"
    ]
    if len(expressions) != 1:
        raise SystemExit(f"{path}: cannot repair without exactly one Expression class.")
    expression = expressions[0]
    declarations = list(SUPPORT_DECLARATIONS.items())
    helper_edits: dict[int, list[str]] = {}
    for index, (name, declaration) in enumerate(declarations):
        if support[name] is not None:
            continue
        next_present = next(
            (
                support[later]
                for later, _ in declarations[index + 1 :]
                if support[later] is not None
            ),
            expression,
        )
        # Group helpers sharing an insertion point to preserve their dependency order.
        helper_edits.setdefault(offsets[next_present.lineno - 1], []).append(
            declaration
        )
    edits.extend(
        (offset, "\n\n".join(declarations) + "\n\n")
        for offset, declarations in helper_edits.items()
    )

    for name in REFLECTED_OPERATORS:
        methods = [
            node
            for node in expression.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        expected_reflected, expected_scalar = _operator_declarations(name)
        if not 1 <= len(methods) <= 2 or not all(
            isinstance(node, ast.FunctionDef) for node in methods
        ):
            raise SystemExit(
                f"{path}: cannot repair missing or duplicate Expression.{name} declarations."
            )
        scalar = methods[-1]
        if not scalar.decorator_list:
            expected_scalar.decorator_list.clear()
        if not _same_signature(scalar, expected_scalar):
            raise SystemExit(
                f"{path}: cannot repair changed Expression.{name} scalar signature."
            )
        indentation = lines[scalar.lineno - 1][: scalar.col_offset]
        missing_decorator = "" if scalar.decorator_list else indentation + "@overload\n"
        if len(methods) == 2:
            if (
                ast.dump(methods[0]) != ast.dump(expected_reflected)
                or expression.body.index(scalar)
                != expression.body.index(methods[0]) + 1
            ):
                raise SystemExit(
                    f"{path}: cannot repair malformed Expression.{name} reflected overload."
                )
            if missing_decorator:
                edits.append((offsets[scalar.lineno - 1], missing_decorator))
        else:
            start = (
                scalar.decorator_list[0].lineno
                if scalar.decorator_list
                else scalar.lineno
            )
            parameter = "exp" if name == "__pow__" else "other"
            addition = (
                f"{indentation}@overload\n"
                f"{indentation}def {name}(self, {parameter}: {REFLECTED_OPERATORS[name][0]}[_ArithmeticT_co]) -> _ArithmeticT_co: ...\n\n"
                + missing_decorator
            )
            edits.append((offsets[start - 1], addition))
    for offset, addition in sorted(edits, reverse=True):
        source = source[:offset] + addition + source[offset:]
    _check_source(source, path, canonical)
    if edits:
        path.write_text(source, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", type=Path, nargs="+", help="core.pyi files or built wheels"
    )
    parser.add_argument(
        "--canonical", type=Path, help="canonical Symbolica stub to compare"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="restore missing reflected arithmetic refinements in source stubs",
    )
    args = parser.parse_args()
    if args.fix and any(path.suffix == ".whl" for path in args.paths):
        parser.error("--fix accepts source stubs only, not wheels")
    for path in args.paths:
        if args.fix:
            fix_core_stub(path, args.canonical)
        else:
            check_core_stub(path, args.canonical)
