"""Native HEPKit inputs for the FastSecDec notebook.

These builders only prepare diagrams and kinematics. Generation, uncertainty
and integration results belong to the Rust FastSecDec bridge. Compact DOT and
the scalar model are the existing native FastSecDec fixtures, read by HEPKit.
All integrals use prod d^D k / (i*pi^(D/2)) with no extra Euler-gamma factor.
"""

from dataclasses import dataclass
import math
from pathlib import Path

from symbolica import E, S, Expression
from symbolica.community import hepkit as hep


FIXTURES = Path(__file__).with_name("fixtures") / "fastsecdec"


@dataclass(frozen=True)
class ShowcaseInput:
    """Prepared native owners and explicit integral conventions for one example."""

    name: str
    model: hep.Model
    diagram: hep.FeynmanDiagram
    kinematics: hep.Kinematics
    regulator: Expression
    dimension: Expression
    scalar_values: dict[Expression, Expression]
    max_order: int

    def integral_arguments(self):
        """Arguments for ``hep.fastsecdec.Integral``; no serialization step."""
        return {
            "diagram": self.diagram,
            "kinematics": self.kinematics,
            "regulator": self.regulator,
            "dimension": self.dimension,
            "scalar_values": dict(self.scalar_values),
            "powers": {},
            "auxiliary_momenta": [],
            "measure_multiplier": E("1"),
        }

    def scalar_numerator(self):
        """Display the fully weighted numerator through native tensor algebra."""
        numerator = (
            self.diagram.numerator_expression(in_lmb=True)
            * self.diagram.projector_expression()
            * self.diagram.numerator_prefactor_expression()
            * self.diagram.overall_factor_expression()
        ).with_lorentz_dimension(self.kinematics.dimension)
        numerator = numerator.simplify_algebra(contract="minimal").to_dots()
        if not numerator.is_scalar:
            raise ValueError("The prepared numerator still has free tensor indices")
        return self.kinematics.apply(numerator).to_expression()


def _number(value, name, *, positive=False, spacelike=False):
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if positive and numeric <= 0:
        raise ValueError(f"{name} must be positive")
    if spacelike and numeric >= 0:
        raise ValueError(f"{name} must be negative in this Euclidean example")
    # Match the native model card's f64 value exactly before polynomial
    # extraction. Decimal Float coefficients are not affine exact-ring input.
    numerator, denominator = numeric.as_integer_ratio()
    return E(str(numerator)) / E(str(denominator))


def _model(mass):
    model = hep.Model(FIXTURES / "scalar.json")
    card = model.default_parameter_card()
    card.set("mt", float(mass), 0.0)
    return model.with_parameter_card(card)


def _build(name, filename, products, mass, max_order):
    regulator, tensor_dimension = S("fastsecdec_showcase::eps", "fastsecdec_showcase::D")
    p = hep.Kinematics.external_momentum()
    model = _model(mass)
    diagram = hep.FeynmanDiagram.from_dot(model, (FIXTURES / filename).read_text())
    diagram.validate()
    k = hep.Kinematics.loop_momentum()
    momenta = [k(index) for index in range(diagram.loop_count)]
    momenta += [p(index) for index in sorted({i for pair in products for i in pair})]
    kinematics = hep.Kinematics(tensor_dimension, momenta=momenta)
    for (left, right), value in products.items():
        kinematics = kinematics.with_scalar_product(p(left), p(right), value)
    return ShowcaseInput(
        name, model, diagram, kinematics, regulator, 4 - 2 * regulator,
        {S("UFO::mt"): _number(mass, "mass")}, max_order,
    )


def massive_triangle(*, mass=1, s=-1):
    """Equal internal masses; two massless legs and spacelike invariant ``s``."""
    _number(mass, "mass", positive=True)
    invariant = _number(s, "s", spacelike=True)
    return _build(
        "Massive triangle", "triangle.dot",
        {(1, 1): E("0"), (2, 2): E("0"), (1, 2): invariant / 2}, mass, 1,
    )


def _box(rank_two, s12, s23):
    s = _number(s12, "s12", spacelike=True)
    t = _number(s23, "s23", spacelike=True)
    return _build(
        "Rank-two box numerator" if rank_two else "Massless box",
        "box_rank2_numerator.dot" if rank_two else "box.dot",
        {(0, 0): E("0"), (1, 1): E("0"), (2, 2): E("0"),
         (0, 1): s / 2, (1, 2): t / 2, (0, 2): -(s + t) / 2}, 0, 1,
    )


def massless_box(*, s12=-1, s23=-1):
    """Massless external/internal lines, with both independent invariants negative."""
    return _box(False, s12, s23)


def rank_two_box(*, s12=-1, s23=-1):
    """The same box with numerator k² + 3(k·p0)(k·p1), in its stored loop basis."""
    return _box(True, s12, s23)


def coupled_sunset(*, s=-1):
    """Two-loop massless sunset with k0·k1 + 2k0·p0 and spacelike p0²=s."""
    invariant = _number(s, "s", spacelike=True)
    return _build(
        "Coupled two-loop sunset numerator", "sunset_2loop_numerator.dot",
        {(0, 0): invariant}, 0, 1,
    )
