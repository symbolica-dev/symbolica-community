"""Keep the HEP examples executable with the installed community extension."""

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
SOURCES = sorted((ROOT / "examples/hep").glob("*.py")) + sorted(
    (ROOT / "examples").glob("hep*.py")
)
NOTEBOOKS = [path for path in SOURCES if "app = marimo.App(" in path.read_text()]
REMOVED = {
    "reduce_algebra",
    "simplify_gamma",
    "simplify_color",
    "simplify_epsilon",
    "schoonschip_net",
    "expand_mink",
    "expand_color",
    "undo_all",
    "CookSettings",
    "SchoonschipSettings",
    "simplify_metrics",
    "collect_chains",
    "spenso_conjugate",
    "to_color_casimir",
    "SimplificationSettings",
    "ContractSettings",
    "AlgebraSettings",
    "GammaSimplifySettings",
    "ColorSimplifySettings",
}


@pytest.mark.parametrize("path", SOURCES, ids=lambda path: path.stem)
def test_current_tensor_api(path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in REMOVED, f"{path}:{node.lineno}: {name}"


def run_notebook(path, controls=None, models=()):
    pytest.importorskip("marimo", minversion="0.24.0")
    script = """
import json, runpy, sys
from types import SimpleNamespace
from symbolica.community.hepkit import Model
sys.path.insert(0, sys.argv[1])
controls, models = json.loads(sys.argv[3])
definitions = {name: SimpleNamespace(value=value) for name, value in controls.items()}
definitions.update({name: Model.standard_model() for name in models})
app = runpy.run_path(sys.argv[2], run_name="notebook_check")["app"]
app.run(defs=definitions)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(path.parent),
            str(path),
            json.dumps([controls or {}, models]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_runs_from_clean_session(path):
    if path.stem == "gg_hg":
        pytest.skip("Cold transport is a long acceptance; rendering has a separate lightweight fixture.")
    run_notebook(path)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_helpers_have_one_folded_preamble(path):
    cells = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    setup = [
        i
        for i, cell in enumerate(cells)
        if any(
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and "## Setup and notebook helpers" in n.value
            for n in ast.walk(cell)
        )
    ]
    assert len(setup) == 1, path
    for i, cell in enumerate(cells):
        helpers = [
            n for n in cell.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        ]
        if not helpers:
            continue
        assert i > setup[0], (path, cell.lineno)
        assert any(
            isinstance(d, ast.Call)
            and any(
                kw.arg == "hide_code"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value
                for kw in d.keywords
            )
            for d in cell.decorator_list
        ), (path, cell.lineno)
        # The preamble ends when visible calculation code begins.
        assert all(
            any(
                isinstance(d, ast.Call)
                and any(
                    kw.arg == "hide_code"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value
                    for kw in d.keywords
                )
                for d in earlier.decorator_list
            )
            for earlier in cells[setup[0] + 1 : i]
        ), (path, cell.lineno)


CHANNELS = (
    [
        ("higgs_decay", choice, "model", "channel")
        for choice in (
            "Electrons",
            "Charm quarks",
            "Bottom quarks",
            "W bosons",
            "Z bosons",
        )
    ]
    + [
        ("z_decay", choice, "model", "channel")
        for choice in (
            "Electron neutrinos",
            "Electrons",
            "Charm quarks",
            "Bottom quarks",
        )
    ]
    + [
        ("identical_leptons", choice, "lepton_model", "reaction")
        for choice in ("Bhabha", "Møller")
    ]
    + [
        ("polarized_spin", choice, "spin_model", "species")
        for choice in ("Tau", "Antitau")
    ]
)


@pytest.mark.parametrize("notebook,choice,model,control", CHANNELS)
def test_selectable_particle_channels(notebook, choice, model, control):
    run_notebook(ROOT / "examples/hep" / f"{notebook}.py", {control: choice}, [model])


@pytest.mark.parametrize("particle", ["g", "a", "Z"])
@pytest.mark.parametrize("dimension", ["Symbolic D", "4", "6"])
@pytest.mark.parametrize(
    "normalization",
    [
        "Sum over states",
        "Average in D dimensions",
        "Average over four-dimensional states",
    ],
)
def test_polarization_conventions(particle, dimension, normalization):
    run_notebook(
        ROOT / "examples/hep/polarization_sums.py",
        {
            "species": particle,
            "dimension_choice": dimension,
            "normalization": normalization,
        },
    )


@pytest.mark.parametrize("ratio", [-1.0, 0.0, 1.0, 2.0])
@pytest.mark.parametrize("gauge", [0.0, 1.0, 3.0])
def test_self_energy_limits_and_physical_branch(ratio, gauge):
    run_notebook(
        ROOT / "examples/hep/electron_self_energy.py",
        {
            "ratio_control": ratio,
            "gauge_control": gauge,
            "mass_control": 2.0,
            "scale_control": 3.0,
        },
    )
