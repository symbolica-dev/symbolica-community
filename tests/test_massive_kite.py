"""Physical checks for the two-mass kite, without a cold AMFlow calculation."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_massive_kite_mass_assignment_and_exchange_symmetry():
    pytest.importorskip("marimo", minversion="0.24.0")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    notebook = Path(__file__).parents[1] / "examples/hep/massive_kite.py"
    code = """
import runpy
import sys
from types import SimpleNamespace
from symbolica import E

app = runpy.run_path(sys.argv[1], run_name='notebook_check')['app']
_, definitions = app.run()
state = dict(definitions)
family = state['family']
assert family.is_complete and family.is_independent
assert len(family.denominators) == 5

# Compare with the massless graph to check both physical mass terms in F.
hep = state['hep']
rho, r = state['rho'], state['r']
kinematics = hep.Kinematics(E('4'), momenta=[state['p']]).with_scalar_product(
    state['p'], state['p'], -rho,
)
graph_family = state['diagram'].integral_family(kinematics=kinematics)
_, graph_F = graph_family.symanzik(state['x'])
massless_F = graph_F.replace(state['model'].particle('phi').mass, E('0'))
a, b = state['massive_slots']
mass_terms = state['U'] * (state['x'][a] + r * state['x'][b])
assert (state['F'] - massless_F - mass_terms).expand() == 0

cell = next(cell for cell in app._cell_manager.cells() if 'exact_results' in cell._cell.defs)
arguments = {name: state[name] for name in cell._cell.refs if name in state}
values = {}
for ratio in ('1/2', '1', '2'):
    arguments['mass_ratio'] = SimpleNamespace(value=ratio)
    _, definitions = cell.run(**arguments)
    values[ratio] = definitions['exact_results']

# Raising a positive Euclidean mass suppresses the integrand point by point.
for point in state['comparison_points']:
    assert values['1/2'][point][0] > values['1'][point][0] > values['2'][point][0] > 0

# Exchanging the two masses and restoring dimensions gives
# f(rho, r) = f(rho/r, 1/r) / r, independently of integration coordinates.
first, first_error = state['exact_value'](
    state['exact_expression'].replace(r, E('2')).replace(rho, E('4')),
)
second, second_error = values['1/2'][2]
assert abs(first - second/2) <= 5 * (first_error + second_error/2)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(notebook)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_massless_toggle_preserves_the_reference_scale_and_kite_period():
    pytest.importorskip("marimo", minversion="0.24.0")
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    notebook = Path(__file__).parents[1] / "examples/hep/massive_kite.py"
    code = """
import runpy
import sys
from types import SimpleNamespace
from symbolica import E

app = runpy.run_path(sys.argv[1], run_name='notebook_check')['app']
_, state = app.run()
state = dict(state)
state['massless'] = SimpleNamespace(value=True)
cells = list(app._cell_manager.cells())
def run_cell(definition):
    cell = next(cell for cell in cells if definition in cell._cell.defs)
    _, definitions = cell.run(**{name: state[name] for name in cell._cell.refs if name in state})
    state.update(definitions)

run_cell('family')
run_cell('exact_expression')
run_cell('exact_results')
assert state['family'].is_complete and state['family'].is_independent
graph = state['diagram'].integral_family(kinematics=state['family'].kinematics)
mass = state['model'].particle('phi').mass
assert state['denominators'] == [denominator.replace(mass, E('0')) for denominator in graph.denominators]
assert not state['exact_expression'].contains(state['r'])
reference = float((E('6') * E('3').zeta()).evaluate({}).real)
for rho, (value, uncertainty) in state['exact_results'].items():
    assert abs(value * float(rho) - reference) <= 5 * float(rho) * uncertainty
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(notebook)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
