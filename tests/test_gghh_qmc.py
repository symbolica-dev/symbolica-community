"""Compare symbolic runtime bindings against the original numeric notebook."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("topology", ["1pi", "triangle"])
def test_gghh_qmc_matches_fixed_kinematics(topology):
    pytest.importorskip("marimo", minversion="0.24.0")
    root = Path(__file__).parents[1]
    script = r"""
import asyncio
from dataclasses import replace
import inspect
import math
import runpy
import sys

from symbolica import E
from symbolica.community.hepkit import sector_decomposition as sd

directory = sys.argv[1]
old_app = runpy.run_path(directory + '/gghh.py')['app']
_, old = old_app.run()
_, current = runpy.run_path(directory + '/gghh_complete.py')['app'].run()
inputs, science = current['gghh_inputs'], current['science']
catalogue = inputs.catalogue(progress=None)
if sys.argv[2] == 'triangle':
    # Retain the explicit reducible-graph branch regression independently of
    # the complete notebook catalogue's default selection.
    result = catalogue.process.generate_diagrams(
        loops=1, coupling_orders={'QED': 2}, threads=1,
        symmetrize_initial=True, symmetrize_final=True,
        maximum_bridges=None, filter_zero_color=True,
        numerator_grouping=None, projector=E('1'), progress=None)
    catalogue = replace(catalogue, result=result)

def integrate(kernels):
    assert 0 in kernels.orders and max(kernels.orders) == 0
    session = kernels.session(sd.QmcSettings(points=1024, shifts=2, seed=1))
    while not session.complete:
        session.step()
    mean = session.snapshot().estimate.mean
    assert all(math.isfinite(value) for value in mean)
    assert any(value != 0 for value in mean), mean
    assert len(mean) == len(kernels.orders) == len(kernels.components)
    result = dict(zip(zip(kernels.orders, kernels.components), mean))
    assert len(result) == len(mean), "Native coefficient slots must be unique"
    return result

point = {'sqrt_s': 300, 'higgs_mass': 125, 'cos_theta': 0.8, 'top_mass': 172.5}
topology = dict(old)
selection_cell = next(cell for _, cell in old_app._cell_manager.valid_cells()
                      if 'is_double_box' in cell.defs)
exec(selection_cell._cell.code, topology)
if sys.argv[2] == 'triangle':
    raws = [next(raw for raw in catalogue.diagrams
                 if sum(abs(edge.particle.pdg_code) == 6 for edge in raw.internal_edges) == 3
                 and sum(edge.particle.pdg_code == 25 for edge in raw.internal_edges) == 1)]
else:
    # The complete catalogue includes zero candidates; choose the same nonzero
    # top box explicitly for this numerical comparison.
    box = next(raw for raw in catalogue.diagrams if raw.loop_count == 1
               and len(raw.internal_edges) == 4
               and all(abs(edge.particle.pdg_code) == 6 for edge in raw.internal_edges))
    raws = [box,
            next(raw for raw in catalogue.diagrams if topology['is_double_box'](raw))]
    assert raws[0].loop_count == 1 and raws[1].loop_count == 2
    assert len(raws[0].internal_edges) == 4
    assert all(abs(edge.particle.pdg_code) == 6 for edge in raws[0].internal_edges)
for raw in raws:
    assert all(edge.momentum_expression(in_lmb=True) != E('0') for edge in raw.internal_edges)
    prepared = inputs.prepare(source=catalogue, selected=raw.id)
    assert prepared.simplified_numerator != E('0')
    generated = science.generation(prepared, {'max_order': 0})
    assert generated.mode == 'symbolic' and generated.subtraction == 'taylor'
    while not generated.complete:
        generated.step(max_units=1)
    template = sd.Kernels.from_bytes(generated.kernels.to_bytes())
    kernels, _ = science.bind(generated.generated, template, prepared, point)
    runtime_mean = integrate(kernels)

    # Run the original notebook's numeric contraction on the identical graph,
    # rather than comparing unrelated graph IDs or recorded sample estimates.
    context = dict(old, raw_diagram=raw)
    for definition in ('kinematics', 'numerator'):
        cell = next(cell for _, cell in old_app._cell_manager.valid_cells()
                    if definition in cell.defs)
        result = cell.run(**{name: context[name] for name in cell.refs if name in context})
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        _, definitions = result
        context.update(definitions)
    fixed = context['diagram'].sector_decompose(
        regulator=context['eps'], dimension=4 - 2*context['eps'],
        kinematics=context['kinematics'], scalar_values=catalogue.scalar_values,
        auxiliary_momenta=context['auxiliaries'], max_order=0,
        coefficient_expansion='native_named', progress=None)
    fixed_kernels = fixed.compile(backend='eager', progress=None)
    if fixed_kernels.runtime_parameters:
        fixed_kernels = fixed_kernels.with_parameters(fixed.runtime_parameter_defaults)
    fixed_mean = integrate(fixed_kernels)
    print(raw.name, 'runtime', runtime_mean, 'fixed', fixed_mean, flush=True)
    # Runtime and fixed preparation can retain different vanishing pole slots.
    # Compare every native Laurent/component key, including those extra slots
    # against their absent exact coefficient.
    slots = runtime_mean.keys() | fixed_mean.keys()
    assert all(math.isclose(runtime_mean.get(slot, 0.0), fixed_mean.get(slot, 0.0),
                            rel_tol=1e-8, abs_tol=1e-8)
               for slot in slots), (runtime_mean, fixed_mean)

    if raw.loop_count == 1:
        changed = dict(point, sqrt_s=320, cos_theta=0.4)
        rebound, _ = science.bind(
            generated.generated, template, prepared, changed)
        assert integrate(rebound) != runtime_mean
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(root / "examples/hep"), topology],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
