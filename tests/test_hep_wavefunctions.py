"""Shared GammaLoop external states through the installed community host."""

import math

import pytest

from symbolica.community import hepkit as hep


def minkowski(left, right):
    return left[0] * right[0] - sum(a * b for a, b in zip(left[1:], right[1:]))


@pytest.mark.parametrize("mass", [0.0, 4.0])
def test_vector_states_and_scalar(mass):
    momentum = hep.FourMomentum(math.sqrt(25.0 + mass * mass), 3.0, 0.0, 4.0)
    helicities = [hep.Helicity.PLUS, hep.Helicity.MINUS]
    if mass:
        helicities.append(hep.Helicity.ZERO)
    for helicity in helicities:
        state = momentum.wavefunction("epsilon", helicity)
        assert isinstance(state, hep.Wavefunction)
        assert type(state).__module__ == "symbolica.community.hepkit"
        assert len(state) == 4
        assert all(isinstance(c, complex) for c in state.components)
        assert minkowski(momentum.components(), state.components) == pytest.approx(0)
        assert minkowski(state.components, state.bar().components) == pytest.approx(-1)
        assert momentum.wavefunction("epsilon_bar", helicity) == state.bar()
        assert state.bar().bar() == state
        copy = state.components
        copy[0] = 123j
        assert state.components[0] != 123j
    scalar = momentum.wavefunction("scalar", hep.Helicity.ZERO)
    assert scalar.components == [1 + 0j]
    assert scalar.kind == "scalar" and len(scalar) == 1
    assert scalar.bar() == scalar


@pytest.mark.parametrize("energy", [5.0, math.sqrt(41.0)])
@pytest.mark.parametrize("kind", ["u", "v"])
@pytest.mark.parametrize("helicity", [hep.Helicity.PLUS, hep.Helicity.MINUS])
def test_spinor_states_and_adjoint(energy, kind, helicity):
    momentum = hep.FourMomentum(energy, 3.0, 0.0, 4.0)
    state = momentum.wavefunction(kind, helicity)
    assert state.kind == kind
    assert len(state) == 4
    assert sum(abs(c) ** 2 for c in state.components) == pytest.approx(2 * energy)
    expected_bar = [c.conjugate() for c in state.components[2:] + state.components[:2]]
    assert state.bar().components == expected_bar
    assert state.bar() == momentum.wavefunction(kind + "_bar", helicity)
    assert state.bar().bar() == state


@pytest.mark.parametrize(
    "kind,helicity",
    [
        ("epsilon", hep.Helicity.ZERO),
        ("u", hep.Helicity.ZERO),
        ("scalar", hep.Helicity.PLUS),
        ("unknown", hep.Helicity.PLUS),
    ],
)
def test_invalid_state_uses_native_kinematics_error(kind, helicity):
    with pytest.raises(hep.KinematicsError):
        hep.FourMomentum(5, 3, 0, 4).wavefunction(kind, helicity)
