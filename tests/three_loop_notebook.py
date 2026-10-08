"""Read the notebook's actual inputs and exercise individual display cells."""

import ast
import importlib.util
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "examples/hep/three_loop_reduction.py"
SPEC = importlib.util.spec_from_file_location("three_loop_reduction_notebook", NOTEBOOK)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def notebook_input(name):
    """Read a literal input from its notebook cell, without a separate fixture."""
    tree = ast.parse(NOTEBOOK.read_text())
    values = [node.value for cell in tree.body if isinstance(cell, ast.FunctionDef)
              for node in ast.walk(cell) if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name
                      for target in node.targets)]
    assert len(values) == 1, f"Expected one notebook assignment for {name}"
    value = ast.literal_eval(values[0])
    assert isinstance(value, str)
    return value


def notebook_cell_producing(name):
    """Exercise a notebook cell without starting its other scientific work."""
    tree = ast.parse(NOTEBOOK.read_text())
    cells = [cell for cell in tree.body if isinstance(cell, ast.FunctionDef) and any(
        isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
        for node in ast.walk(cell))]
    assert len(cells) == 1
    cell = cells[0]
    cell.decorator_list = []
    namespace = vars(support).copy()
    exec(compile(ast.Module(body=[cell], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace[cell.name]
