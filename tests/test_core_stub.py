"""Preserve Community stub additions without hiding canonical declaration drift."""

import ast
import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest

checker = runpy.run_path(
    str(Path(__file__).parents[1] / ".github/scripts/check_core_stub.py")
)
check_core_stub = checker["check_core_stub"]
fix_core_stub = checker["fix_core_stub"]

OPERATORS = ("__add__", "__sub__", "__mul__", "__truediv__", "__pow__")
SCALAR = "Expression | int | float | complex | Float | ComplexFloat | Decimal"
PROTOCOLS = {
    "__add__": ("_SupportsRAdd", "__radd__"),
    "__sub__": ("_SupportsRSub", "__rsub__"),
    "__mul__": ("_SupportsRMul", "__rmul__"),
    "__truediv__": ("_SupportsRTrueDiv", "__rtruediv__"),
    "__pow__": ("_SupportsRPow", "__rpow__"),
}
SUPPORT = {
    "_TypeVar": "from typing import TypeVar as _TypeVar",
    "_ArithmeticT_co": '_ArithmeticT_co = _TypeVar("_ArithmeticT_co", bound="Expression", covariant=True)',
    **{
        protocol: f"class {protocol}(Protocol[_ArithmeticT_co]):\n    def {reflected}(self, left: Expression, /) -> _ArithmeticT_co: ..."
        for protocol, reflected in PROTOCOLS.values()
    },
}
CITATIONS = "\ndef get_citations() -> list[Citation]: ...\n"
TRANSPORT = """
def _set_data_directories(directories: list[str]) -> None: ...
def _data_manifest() -> str: ...
def _install_data_asset(name: str, payload: bytes) -> None: ...
def _read_data_asset(name: str) -> bytes: ...
"""
CANONICAL = """from typing import Protocol, overload

class Citation:
    @property
    def id(self) -> str: ...

class Expression:
    def __getitem__(self, index: int, /) -> Expression: ...
""" + "".join(
    f"    def {name}(self, {'exp' if name == '__pow__' else 'other'}: {SCALAR}) -> Expression: ...\n"
    for name in OPERATORS
)
COMMUNITY = (
    CANONICAL.replace(
        "class Expression:", "\n\n".join(SUPPORT.values()) + "\n\nclass Expression:"
    )
    + CITATIONS
    + TRANSPORT
)
for operator in OPERATORS:
    parameter = "exp" if operator == "__pow__" else "other"
    COMMUNITY = COMMUNITY.replace(
        f"    def {operator}(",
        f"    @overload\n"
        f"    def {operator}(self, {parameter}: {PROTOCOLS[operator][0]}[_ArithmeticT_co]) -> _ArithmeticT_co: ...\n"
        f"    @overload\n"
        f"    def {operator}(",
    )


def write_stub(tmp_path, source, wheel=False):
    path = tmp_path / ("symbolica.whl" if wheel else "core.pyi")
    if wheel:
        with ZipFile(path, "w") as archive:
            archive.writestr("symbolica/core.pyi", source)
    else:
        path.write_text(source)
    return path


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize("compare_canonical", [False, True])
def test_canonical_core_with_community_additions(tmp_path, wheel, compare_canonical):
    canonical = tmp_path / "symbolica.pyi"
    canonical.write_text(CANONICAL)
    path = write_stub(tmp_path, COMMUNITY, wheel)
    check_core_stub(path, canonical if compare_canonical else None)


@pytest.mark.parametrize(
    "source",
    [
        CITATIONS,
        COMMUNITY.replace(CITATIONS, ""),
        COMMUNITY + "\nclass Citation: ...\n",
        COMMUNITY + CITATIONS,
        COMMUNITY.replace(
            CITATIONS,
            "\ndef get_citations(reset: bool = False) -> list[Citation]: ...\n",
        ),
        COMMUNITY.replace(CITATIONS, "\ndef get_citations() -> list[str]: ...\n"),
    ],
)
def test_missing_duplicate_or_incorrect_citation_api_is_rejected(tmp_path, source):
    path = write_stub(tmp_path, source)
    with pytest.raises(SystemExit, match="Citation|get_citations"):
        check_core_stub(path)


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize(
    "source",
    [
        COMMUNITY.replace("index: int", "index: str"),
        COMMUNITY + "\ndef extra_core_api() -> None: ...\n",
        COMMUNITY.replace("class Citation:", "class Citation:\n    extra: str"),
        COMMUNITY.replace(
            "other: " + SCALAR + ") -> Expression: ...",
            "other: " + SCALAR + ') -> Expression: "Changed upstream documentation."',
            1,
        ),
    ],
)
def test_canonical_declaration_drift_is_rejected(tmp_path, source, wheel):
    canonical = tmp_path / "symbolica.pyi"
    canonical.write_text(CANONICAL)
    path = write_stub(tmp_path, source, wheel)
    with pytest.raises(SystemExit, match="differs from"):
        check_core_stub(path, canonical)


def mutate_operator(name, mutation):
    module = ast.parse(COMMUNITY)
    expression = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Expression"
    )
    reflected, scalar = [
        node
        for node in expression.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if mutation == "missing_reflected":
        expression.body.remove(reflected)
    elif mutation == "missing_scalar":
        expression.body.remove(scalar)
    elif mutation == "duplicate":
        expression.body.append(reflected)
    elif mutation == "order":
        reflected_index = expression.body.index(reflected)
        expression.body.remove(scalar)
        expression.body.insert(reflected_index, scalar)
    elif mutation == "separated":
        expression.body.insert(
            expression.body.index(scalar), ast.parse("def unrelated(self): ...").body[0]
        )
    elif mutation == "reflected_parameter":
        reflected.args.args[1].annotation = ast.Name(id="Expression")
    elif mutation == "reflected_parameter_name":
        reflected.args.args[1].arg = "rhs"
    elif mutation == "reflected_return":
        reflected.returns = ast.Name(id="Expression")
    elif mutation == "scalar_parameter":
        scalar.args.args[1].annotation = ast.Name(id="Expression")
    elif mutation == "scalar_return":
        scalar.returns = ast.Name(id="_ArithmeticT_co")
    elif mutation == "reflected_decorator":
        reflected.decorator_list.clear()
    elif mutation == "scalar_decorator":
        scalar.decorator_list.clear()
    elif mutation == "extra_decorator":
        scalar.decorator_list.append(ast.Name(id="staticmethod"))
    elif mutation == "default":
        reflected.args.defaults.append(ast.Constant(value=None))
    elif mutation == "positional_only":
        reflected.args.posonlyargs = reflected.args.args
        reflected.args.args = []
    else:
        raise AssertionError(mutation)
    return ast.unparse(module)


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_reflected",
        "missing_scalar",
        "duplicate",
        "order",
        "separated",
        "reflected_parameter",
        "reflected_parameter_name",
        "reflected_return",
        "scalar_parameter",
        "scalar_return",
        "reflected_decorator",
        "scalar_decorator",
        "extra_decorator",
        "default",
        "positional_only",
    ],
)
def test_reflected_arithmetic_regression_is_rejected_without_canonical(
    tmp_path, wheel, name, mutation
):
    path = write_stub(tmp_path, mutate_operator(name, mutation), wheel)
    with pytest.raises(SystemExit, match=f"Expression.{name}"):
        check_core_stub(path)


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize("helper", SUPPORT)
@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_missing_or_duplicate_helper_is_rejected(tmp_path, wheel, helper, mutation):
    declaration = SUPPORT[helper]
    source = (
        COMMUNITY.replace(declaration, "", 1)
        if mutation == "missing"
        else COMMUNITY + "\n" + declaration + "\n"
    )
    path = write_stub(tmp_path, source, wheel)
    with pytest.raises(SystemExit, match="reflected arithmetic helper"):
        check_core_stub(path)


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize(
    "source",
    [
        COMMUNITY.replace(
            "from typing import TypeVar as _TypeVar",
            "from wrong.module import TypeVar as _TypeVar",
        ),
        COMMUNITY.replace("TypeVar as _TypeVar", "TypeVar as WrongAlias"),
        COMMUNITY.replace("TypeVar as _TypeVar", "TypeVar as _TypeVar, Any"),
        COMMUNITY.replace('bound="Expression", ', ""),
        COMMUNITY.replace('bound="Expression"', 'bound="object"'),
        COMMUNITY.replace("covariant=True", "covariant=False"),
        COMMUNITY.replace("covariant=True", "contravariant=True"),
    ],
)
def test_bound_covariant_typevar_and_import_are_required(tmp_path, wheel, source):
    path = write_stub(tmp_path, source, wheel)
    with pytest.raises(SystemExit, match="reflected arithmetic helper"):
        check_core_stub(path)


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize("protocol,method", PROTOCOLS.values())
@pytest.mark.parametrize(
    "mutation",
    [
        "operand",
        "positional_only",
        "return",
        "method",
        "extra_member",
        "decorator",
        "base",
    ],
)
def test_protocol_shape_is_required(tmp_path, wheel, protocol, method, mutation):
    original = SUPPORT[protocol]
    if mutation == "operand":
        changed = original.replace("left: Expression", "left: object")
    elif mutation == "positional_only":
        changed = original.replace(", /", "")
    elif mutation == "return":
        changed = original.replace("-> _ArithmeticT_co", "-> Expression")
    elif mutation == "method":
        changed = original.replace(method, "__symbolica_rmul__")
    elif mutation == "extra_member":
        changed = original + "\n    unexpected: str"
    elif mutation == "decorator":
        changed = original.replace("    def", "    @staticmethod\n    def")
    else:
        changed = original.replace("Protocol[_ArithmeticT_co]", "Protocol")
    path = write_stub(tmp_path, COMMUNITY.replace(original, changed), wheel)
    with pytest.raises(SystemExit, match="reflected arithmetic helper"):
        check_core_stub(path)


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize(
    "dependency",
    [
        "from symbolica.community.tensor import TensorExpression as _TensorExpression",
        "from symbolica import community",
        "import symbolica.community.tensor",
        "from .community.tensor import TensorExpression",
        "from . import community",
    ],
)
def test_core_stub_must_not_depend_on_optional_community_modules(
    tmp_path, wheel, dependency
):
    path = write_stub(tmp_path, COMMUNITY + "\n" + dependency + "\n", wheel)
    with pytest.raises(SystemExit, match="must not import optional Community"):
        check_core_stub(path)


@pytest.mark.parametrize("wheel", [False, True])
@pytest.mark.parametrize("declaration", TRANSPORT.strip().splitlines())
@pytest.mark.parametrize("mutation", ["duplicate", "signature", "body"])
def test_resource_declarations_are_not_an_unchecked_parity_exception(
    tmp_path, wheel, declaration, mutation
):
    if mutation == "duplicate":
        source = COMMUNITY + declaration + "\n"
    elif mutation == "signature":
        source = COMMUNITY.replace(
            declaration, declaration.replace("(", "(extra: str, ", 1)
        )
    else:
        source = COMMUNITY.replace(
            declaration, declaration.replace("...", '"Unexpected body"')
        )
    path = write_stub(tmp_path, source, wheel)
    with pytest.raises(SystemExit, match="resource declaration"):
        check_core_stub(path)


def test_fix_restores_reflected_refinements_and_preserves_source(tmp_path):
    original = (
        CANONICAL.replace(
            "class Expression:",
            'class Expression:\n    """Keep this docstring, comments, and formatting."""\n    # Keep this comment.',
        )
        + CITATIONS
        + TRANSPORT
    )
    canonical = tmp_path / "symbolica.pyi"
    canonical.write_text(original.removesuffix(CITATIONS + TRANSPORT))
    path = write_stub(tmp_path, original)
    fix_core_stub(path, canonical)
    fixed = path.read_text()
    check_core_stub(path, canonical)
    for line in original.splitlines():
        assert line in fixed
    fix_core_stub(path, canonical)
    assert path.read_text() == fixed


@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize("mutation", ["missing_reflected", "scalar_decorator"])
def test_fix_restores_partial_refinements(tmp_path, name, mutation):
    path = write_stub(tmp_path, mutate_operator(name, mutation))
    fix_core_stub(path)
    check_core_stub(path)


@pytest.mark.parametrize("helper", SUPPORT)
def test_fix_restores_missing_helper(tmp_path, helper):
    path = write_stub(tmp_path, COMMUNITY.replace(SUPPORT[helper], "", 1))
    fix_core_stub(path)
    check_core_stub(path)


def test_fix_is_verbatim_idempotent(tmp_path):
    path = write_stub(tmp_path, COMMUNITY)
    fix_core_stub(path)
    assert path.read_text() == COMMUNITY


@pytest.mark.parametrize("name", OPERATORS)
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_scalar",
        "duplicate",
        "order",
        "reflected_return",
        "scalar_parameter",
        "extra_decorator",
    ],
)
def test_fix_refuses_malformed_declarations_without_modifying_source(
    tmp_path, name, mutation
):
    original = mutate_operator(name, mutation)
    path = write_stub(tmp_path, original)
    with pytest.raises(SystemExit, match="cannot repair"):
        fix_core_stub(path)
    assert path.read_text() == original


def test_fix_refuses_malformed_import_without_modifying_source(tmp_path):
    original = COMMUNITY.replace(
        "from typing import TypeVar as _TypeVar",
        "from wrong.module import TypeVar as _TypeVar",
    )
    path = write_stub(tmp_path, original)
    with pytest.raises(SystemExit, match="cannot repair"):
        fix_core_stub(path)
    assert path.read_text() == original


def test_fix_rejects_wheels(tmp_path):
    path = write_stub(tmp_path, COMMUNITY, wheel=True)
    original = path.read_bytes()
    with pytest.raises(SystemExit, match="source stubs only"):
        fix_core_stub(path)
    assert path.read_bytes() == original
