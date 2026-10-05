"""The notebook's DOTs preserve the independently frozen family coordinates."""

import importlib.util
from pathlib import Path
import tomllib

import pytest

symbolica = pytest.importorskip("symbolica")
from symbolica import E, S
from symbolica.community import hepkit as hep


EXAMPLE = Path(__file__).parents[1] / "examples/hep"
SPEC = importlib.util.spec_from_file_location(
    "rustred_campaign_support", EXAMPLE / "rustred_campaign_support.py"
)
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


@pytest.mark.parametrize("name", support.FAMILY_NAMES)
def test_dot_routing_and_denominator_order_equal_reference(name):
    model = hep.Model.phi_3_4()
    diagram = hep.FeynmanDiagram.from_dot(model, support.dot_sources()[name])
    physical = diagram.propagator_family(kinematics=hep.Kinematics(S("d")))
    completed = diagram.integral_family(
        support.preferred_auxiliaries(name, physical), kinematics=physical.kinematics
    )
    mass = model.particle("phi").mass
    actual = [den.replace(mass, E("1")) for den in completed.denominators]
    family = hep.IntegralFamily(
        completed.loop_momenta, completed.external_momenta,
        actual, kinematics=completed.kinematics,
    )
    assert family.is_complete and family.is_independent
    assert len(family.loop_momenta) == 4 and not family.external_momenta
    assert len(actual) == 10
    reference = tomllib.loads(
        (support.DATA_DIRECTORY / f"{name.lower()}.reference.toml").read_text()
    )
    assert reference["target"]["powers"] == (
        [1] * len(physical.denominators) + [0] * (10 - len(physical.denominators))
    )
    # All frozen reference entries are q^2-1. Symbolica and HEPKit perform
    # substitution and bilinear contraction; no test-side polynomial kernel.
    for index, entry in enumerate(reference["family"]["denominators"]):
        expression = entry["expression"]
        assert expression.endswith("^2-1")
        momentum = E(expression.removesuffix("^2-1"))
        for label, routed in zip(reference["family"]["loop_momenta"], family.loop_momenta):
            momentum = momentum.replace(S(label), routed)
        expected = family.kinematics.scalar_product(momentum, momentum) - 1
        assert (actual[index] - expected).expand() == E("0"), (name, index)
    assert hep.IBPFamily(family, name=name).denominator_count == 10
