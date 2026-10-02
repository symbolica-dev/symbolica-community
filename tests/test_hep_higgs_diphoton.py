"""Independent physics checks for the complete diphoton decay notebook."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "examples/hep/Higgs_diphoton_decay.py"


@pytest.mark.parametrize(
    "mass,fermion,charge_color,seed",
    [
        (125.0, 173.0, 4 / 3, 2026),
        (80.0, 173.0, 1.0, 17),
        (250.0, 173.0, 4 / 3, 41),
        (125.0, 250.0, 4 / 3, 93),
    ],
)
def test_full_rate_and_independent_momentum_space_oracles(
    mass, fermion, charge_color, seed
):
    pytest.importorskip("marimo", minversion="0.24.0")
    parameters = {
        "higgs_mass": mass,
        "fermion_mass": fermion,
        "charge_color": charge_color,
        "vev": 246.22,
        "alpha": 1 / 137.035999084,
        "seed": seed,
        "iterations": 8,
        "samples_per_iteration": 20000,
    }
    script = r"""
import os
import json, math, runpy, sys
import numpy as np
from symbolica import set_license_key
if os.environ.get("SYMBOLICA_LICENSE_KEY"):
    set_license_key(os.environ["SYMBOLICA_LICENSE_KEY"])
from symbolica import E
parameters = json.loads(sys.argv[2])
app = runpy.run_path(sys.argv[1])["app"]
_, data = app.run(defs={"parameters": parameters})
M, m = parameters["higgs_mass"], parameters["fermion_mass"]

# Numerator algebra and the independent closed colour loop finish in one call.
from symbolica.community.tensor import ReductionStatus
for diagram in data["generated"].diagrams:
    numerator = data["prepared_numerator"](
        diagram, data["model"], data["D"], (data["mu"], data["nu"]), data["y"]
    )
    reduced = numerator.simplify_algebra(contract="dots", **data["algebra"])
    assert reduced.reduction_status == ReductionStatus.Complete
    assert reduced.simplify_algebra(contract="dots", **data["algebra"]).to_expression() == reduced.to_expression()

# Independent Feynman-parameter quadrature of the fermion form factor.
# It uses neither the native tensor intake/reducer nor the closed arcsine formula.
nodes, weights = np.polynomial.legendre.leggauss(48)
x = (nodes[:, None] + 1) / 2
y = (1-x) * (nodes[None, :] + 1) / 2
parameter_A = float(np.sum(
    weights[:, None] * weights[None, :] * (1-x)
    * (1-4*x*y)/(1-M*M/(m*m)*x*y)
))
assert abs(data["analytic_A"]-parameter_A) < 3e-11
reference_width = (
    parameters["alpha"]**2 * parameters["charge_color"]**2 * M**3
    * parameter_A**2 / (256*math.pi**3*parameters["vev"]**2)
)
assert abs(data["analytic_width"] / reference_width - 1) < 3e-11
assert abs(data["mc_width"]-reference_width) < 5*data["mc_width_error"]
assert 0 < data["mc_width_error"]/reference_width < 0.005

# Direct lower-half-plane residues: an independent local check of the CFF
# energy integration, routing, overall sign, and loop-measure normalization.
external = np.array([[M, 0., 0., 0.], [M/2, 0., 0., M/2], [M/2, 0., 0., -M/2]])
for diagram, cff in zip(data["generated"].diagrams, data["cff_results"]):
    for spatial in (np.array([31., 19., 47.]), np.array([-83., 23., -62.])):
        energies, shifts = [], []
        for edge in diagram.internal_edges:
            signature = edge.momentum_signature()
            shift = np.array(signature.external) @ external
            vector = signature.loops[0] * spatial + shift[1:]
            energies.append(math.sqrt(float(vector @ vector) + m*m))
            shifts.append(shift[0]/signature.loops[0])
        residue = sum(
            1/(2*energy*math.prod(
                (energy-shifts[i]+shifts[j])**2-energies[j]**2
                for j in range(3) if j != i
            )) for i, energy in enumerate(energies)
        )
        radius = float(np.linalg.norm(spatial))
        cff_value = complex(data["cff_triangle_density"](
            cff, diagram, E(str(radius)), E(str(spatial[2]/radius)), E(str(M)), E(str(m))
        ).evaluate({}))
        assert abs(cff_value / (-2*residue/math.pi) - 1) < 2e-10

# The rational density is an actual integrable contribution, not a fitted
# offset. Its exact radial normalization is one, independently of the Higgs.
x = (nodes + 1)/2
radial = m*x/(1-x)
vacuum_radial = 3*m*m*radial**2/(radial**2+m*m)**2.5 * m/(1-x)**2
assert abs(float(np.dot(weights/2, vacuum_radial))-1) < 2e-13
assert data["rational_term"] == 2
assert abs(data["wrong_A"]-parameter_A) > 1
assert data["R1"] == data["R2"] == 1
assert abs(data["only_R2_A"]-parameter_A) > 1
assert data["four_space"].dimension == 4
assert data["epsilon_space"].dimension == data["n_epsilon"]
assert data["four_transverse_dimension"] == 2
assert (data["dimensional_weight"]-(data["D"]-4)/(data["D"]-2)).together() == 0
assert abs(complex(data["vacuum_master"].evaluate({data["m"]: m})) + 1/(2*m*m)) < 1e-18

# The four-dimensional numerator is generated independently with physical
# Lorentz slots. It must agree with the epsilon-norm-zero limit of the split
# D-dimensional numerator, before either numerator is integrated.
for split, four in zip(data["split_traces"], data["four_traces"]):
    assert (split.replace(data["mu_squared"], E("0"))-four).expand() == 0

# Independent four-by-four Dirac matrices, without Symbolica's trace or tensor
# reducer. Evaluate the physical transverse projection on a complex triple cut.
# The propagator mass is shifted; the numerator fermion mass stays fixed.
identity2, zero2 = np.eye(2), np.zeros((2, 2))
pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
gamma = [np.block([[identity2, zero2], [zero2, -identity2]])] + [
    np.block([[zero2, sigma], [-sigma, zero2]]) for sigma in pauli
]
identity4 = np.eye(4)
def slash(momentum):
    return momentum[0]*gamma[0] - sum(momentum[i]*gamma[i] for i in range(1, 4))
p, q = external[1], external[2]
cut_coefficients = []
for t in (0., m*m/7, -m*m/5):
    loop = np.array([M/2, 1j*math.sqrt(m*m-t), 0., -M/2])
    # These solve k²=(k-q)²=(k-p-q)²=m²-t. Averaging x and y photon
    # polarizations removes the azimuthal spurious part of the triangle cut.
    chains = [slash(momentum)+m*identity4 for momentum in (loop, loop-q, loop-p-q)]
    projected = sum(np.trace(chains[0]@gamma[i]@chains[1]@gamma[i]@chains[2])
                    for i in (1, 2))/(2*m)
    cut_coefficient = -projected/2  # both orientations and the K normalization
    native_coefficient = complex(data["shifted_triangle_weight"].evaluate(
        {data["m"]: m, data["s"]: M*M, data["tilde_k2"]: t}
    ))
    assert abs(cut_coefficient-native_coefficient) < 1e-9
    cut_coefficients.append(cut_coefficient)
cut_slope = (cut_coefficients[1]-cut_coefficients[0])/(m*m/7)
assert abs(cut_slope+2) < 2e-14

# A fifth Clifford direction i*gamma5 has square -1. At zero physical loop
# momentum it isolates the extra-dimensional numerator coefficient (t=-u²).
gamma5 = 1j*gamma[0]@gamma[1]@gamma[2]@gamma[3]
def evanescent_trace(u):
    fermion = identity4 + 1j*u*gamma5
    return -np.trace(fermion@gamma[1]@fermion@gamma[1]@fermion)
transverse_t_coefficient = evanescent_trace(1)-evanescent_trace(0)
assert abs(transverse_t_coefficient-4) < 1e-14
assert all(c == E(str(int(transverse_t_coefficient.real)))
           for c in data["evanescent_coefficients"])

# Independent dimension-shift/Schwinger-parameter evaluation of I_t. Its UV
# residue is -epsilon*Gamma(epsilon) times the simplex volume, hence -1/2.
epsilon = 1e-8
fx = (nodes[:, None]+1)/2
fy = (1-fx)*(nodes[None, :]+1)/2
I_t_reference = -epsilon*math.gamma(epsilon)*np.sum(
    weights[:, None]*weights[None, :]/4*(1-fx)*(m*m-M*M*fx*fy)**(-epsilon)
)
assert abs(I_t_reference-complex(data["I_t"].evaluate({}))) < 1e-6
assert abs(cut_slope*I_t_reference-1) < 2e-6

# Equations use the native MathML renderer, including the equals sign, instead
# of silently falling back to a printed Symbolica function call.
rich_equation = data["equation"]("R_2", data["R2"])._repr_html_()
assert rich_equation and "<math" in rich_equation and "<mo>=</mo>" in rich_equation
assert "diphoton_notation" not in rich_equation

# Reactive notebook sessions can re-execute the preamble. Custom display
# callbacks must not redefine immutable Symbolica printers on the second run.
if parameters["seed"] == 2026:
    _, rerun = app.run(defs={"parameters": parameters})
    assert rerun["analytic_width"] == data["analytic_width"]
    assert rerun["mc_width"] == data["mc_width"]

# Heavy-mass limit from the independent parameter integral and from the native
# master evaluator, with no cancellation-prone textual normal-form comparison.
heavy = complex(data["master_evaluator"].evaluate_complex([complex(M*M), complex(100*M)])[0, 0])
assert abs(heavy-4/3) < 1e-5
print(json.dumps({"parameters": parameters, "analytic_width": data["analytic_width"],
                  "mc_width": data["mc_width"], "mc_width_error": data["mc_width_error"],
                  "pull": data["width_pull"], "mc_seconds": data["mc_seconds"]}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(NOTEBOOK), json.dumps(parameters)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())
