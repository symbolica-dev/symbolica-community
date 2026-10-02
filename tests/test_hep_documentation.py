"""Keep the public HEP reference complete and its examples executable."""

import ast
import doctest
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

HEP = Path(__file__).parents[1] / "python/symbolica/community/hepkit"
PARSER = doctest.DocTestParser()
# These examples explicitly require a user-supplied UFO directory. Their
# documentation is audited below; the other examples use built-in models.
UFO_CLASSES = {"LoadedModel", "UfoLoader", "UfoLoadDiagnostics"}
# Notebook display requires an active frontend and the optional renderer.
DISPLAY_METHODS = {"render", "to_html", "to_linnest"}


def entries(path):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef):
            yield node.name, node
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    yield f"{node.name}.{method.name}", method
        elif isinstance(node, ast.FunctionDef):
            yield node.name, node


@pytest.mark.parametrize(
    "filename", ["__init__.pyi", "ibp/__init__.pyi", "oneloop/__init__.pyi"]
)
def test_public_documentation_has_examples_and_uses_public_namespace(filename):
    for name, node in entries(HEP / filename):
        doc = ast.get_docstring(node) or ""
        assert "Examples\n--------" in doc, name
        examples = PARSER.get_examples(doc)
        assert examples, f"{name}: Examples must contain usable Python code"
        assert not re.search(r"\bfk\b|\bfeynkit\b", doc, re.IGNORECASE), name
        for example in examples:
            compile(example.source, f"{filename}:{name}", "exec")


@pytest.mark.parametrize(
    "filename", ["__init__.pyi", "ibp/__init__.pyi", "oneloop/__init__.pyi"]
)
def test_parameters_are_documented_after_examples(filename):
    for name, node in entries(HEP / filename):
        if not isinstance(node, ast.FunctionDef):
            continue
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg]
        expected = {arg.arg for arg in arguments} - {"self", "cls"}
        if not expected:
            continue

        doc = ast.get_docstring(node) or ""
        assert "Examples\n--------" in doc, name
        assert "Parameters\n----------" in doc, name
        assert doc.index("Examples\n--------") < doc.index("Parameters\n----------"), (
            name
        )
        section = doc.split("Parameters\n----------", 1)[1]
        # Stop at the next NumPy-style section, so a Returns or Raises entry
        # cannot accidentally satisfy a missing parameter description.
        section = re.split(r"\n\S[^\n]*\n-{3,}(?:\n|$)", section, maxsplit=1)[0]
        documented = {
            parameter.strip().lstrip("*")
            for line in section.splitlines()
            if line and not line[0].isspace() and ":" in line
            for parameter in line.split(":", 1)[0].split(",")
        }
        assert expected <= documented, (
            f"{name}: missing {sorted(expected - documented)}"
        )


def runnable_examples():
    for filename in ("__init__.pyi", "oneloop/__init__.pyi"):
        tree = ast.parse((HEP / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                yield filename, node.name, "", ast.get_docstring(node)
            elif isinstance(node, ast.ClassDef) and node.name not in UFO_CLASSES:
                setup = ast.get_docstring(node)
                yield filename, node.name, "", setup
                for method in node.body:
                    if not isinstance(method, ast.FunctionDef):
                        continue
                    if (
                        method.name.startswith("_repr_")
                        or method.name in DISPLAY_METHODS
                    ):
                        continue
                    yield (
                        filename,
                        f"{node.name}.{method.name}",
                        setup,
                        ast.get_docstring(method),
                    )


EXAMPLES = list(runnable_examples())


@pytest.mark.parametrize(
    "filename,name,setup,doc",
    EXAMPLES,
    ids=[f"{filename}:{name}" for filename, name, _, _ in EXAMPLES],
)
def test_reference_examples_execute(filename, name, setup, doc, tmp_path, monkeypatch):
    # Give each example its own setup and directory: mutable builders, model
    # cards, and serialization examples must not depend on earlier examples.
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for source in (setup, doc):
        for example in PARSER.get_examples(source or ""):
            mode = "single" if example.want else "exec"
            output = io.StringIO()
            with redirect_stdout(output):
                # These are trusted, source-controlled documentation snippets.
                exec(  # noqa: S102
                    compile(example.source, f"{filename}:{name}", mode), namespace
                )
            if example.want:
                assert doctest.OutputChecker().check_output(
                    example.want, output.getvalue(), doctest.ELLIPSIS
                ), f"{name}: expected {example.want!r}, got {output.getvalue()!r}"
