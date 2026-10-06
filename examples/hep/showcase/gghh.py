"""gg -> HH input, rebuilt with HEPKit on every explicit Generate action.

The archived raw diagram is an identity guard for the topology selected by the
native Rust/Linnet example, never a substitute for diagram generation. See the
asset origin manifest. No kernels or integration results are loaded here.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from symbolica import E, S, Replacement
from symbolica.community import hepkit as hep
from symbolica.community.tensor import (
    ReductionStatus, Representation, Tensor, TensorName, dot,
)

from .inputs import ShowcaseInput


ASSETS = Path(__file__).resolve().parents[1] / "fixtures" / "gghh"


def _assets(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "origin.json").read_text())
    for name, expected in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid ggHH input manifest path")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"ggHH input identity changed: {name}")
    return manifest


def _rational(value):
    numerator, denominator = float(value).as_integer_ratio()
    return E(str(numerator)) / E(str(denominator))


def _components(state):
    # Exact transport of the binary64 GammaLoop convention, not a decimal fit.
    return [_rational(z.real) + E("1i") * _rational(z.imag) for z in state.components]


def _tensor(name, components):
    return Tensor.dense(TensorName.vector(name)(Representation.mink(4)), components)


def _dot(left, right):
    value = dot(left, right)
    value.execute()
    return value.result_scalar().expand()


def _near(value, expected):
    if abs(complex(value.evaluate({})) - expected) > 2e-12:
        raise ValueError("Native external-state Gram check failed")


@dataclass(frozen=True)
class GGHHInput(ShowcaseInput):
    auxiliary_momenta: tuple
    raw_diagram: object
    preparation_events: tuple
    provenance: dict

    def integral_arguments(self):
        arguments = super().integral_arguments()
        arguments["auxiliary_momenta"] = list(self.auxiliary_momenta)
        return arguments

    def generation_arguments(self):
        return {"coefficient_expansion": "native_named"}

    def scalar_numerator(self):
        return super().scalar_numerator().replace_multiple(
            [Replacement(symbol, value) for symbol, value in self.scalar_values.items()]
        )


def prepare(*, observer=None, assets=ASSETS):
    """Reproduce the audited native (+,+), delta_ab single-diagram input.

    ``observer`` receives HEPKit's original typed GenerationProgress objects.
    All expensive preparation is explicit; importing this module does no work.
    """
    fs = hep.sector_decomposition
    if not hasattr(fs, "with_diagram_expressions"):
        raise RuntimeError("Rebuild the community wheel with the ggHH expression-copy API")
    origin = _assets(assets)
    assets = Path(assets)
    model = hep.Model(assets / "model.json")
    card = hep.ParameterCard.from_json((assets / "parameters.json").read_text())
    model = model.with_parameter_card(card)
    scalar_values = model.scalar_bindings(card)
    vertices = [v for v in model.vertex_rules
                if sorted(model.particle(p).pdg_code for p in v.particles)
                in ([-6, 6, 21], [-6, 6, 25])]
    if len(vertices) != 2:
        raise ValueError("The bound SM input no longer has the two expected top vertices")
    process = model.process([21, 21], [25, 25], vertex_allow=vertices)
    events = []

    def progress(event):
        events.append(event)
        if observer is not None:
            observer(event)

    generated = process.generate_diagrams(
        loops=2, max_vertices=6, threads=1, allow_self_loops=False,
        allow_zero_flow_edges=False, symmetrize_initial=False,
        symmetrize_final=False, symmetrize_left_right=False,
        symmetrize_external_fermions=False, graph_prefix="FK",
        numerator_grouping=hep.NumeratorGrouping("none"),
        numerator_prefactor=E("1"), projector=E("1"), progress=progress,
    )
    if not generated.report.completed:
        raise ValueError("HEPKit diagram generation did not complete")
    identity = json.loads((assets / "generation.json").read_text())
    matches = [d for d in generated.diagrams if d.id == identity["diagram_id"]]
    if len(matches) != 1:
        raise ValueError("Native generation did not reproduce the audited double-box identity")
    raw = matches[0]
    # Rehydrate the identity through the same native owner: canonical Symbolica
    # tag ordering may differ between a Rust process and the Python host.
    expected = hep.FeynmanDiagram.from_json(model, (assets / "raw-diagram.json").read_text())
    actual_identity, expected_identity = json.loads(raw.to_json()), json.loads(expected.to_json())
    # The generated-order display counter differs across frontends. The native
    # content ID and every physical field below must still match exactly.
    origin = dict(origin, actual_diagram_name=actual_identity.pop("name"),
                  audited_diagram_name=expected_identity.pop("name"))
    if actual_identity != expected_identity:
        raise ValueError("Generated raw graph/model/numerator/weight differs from the native input")
    raw.validate()

    # Amplitude::legs keeps each raw half-edge port; no label guessing or graph
    # parsing. The amplitude expression is not used and no weight is reapplied.
    legs = sorted(hep.Amplitude.from_diagram(raw).legs, key=lambda leg: leg.index)
    gluons = [leg for leg in legs if leg.particle.pdg_code == 21]
    if len(gluons) != 2 or any(leg.state != "incoming" for leg in gluons):
        raise ValueError("Expected two unsewn incoming gluon ports")
    indices = [leg.tensor_index for leg in gluons]
    color_projection = Representation.coad(8).id(*indices)
    source = raw.numerator_expression() * color_projection
    policy = dict(gamma=False, color=True, epsilon=False, contract="none")
    symbolic = source.simplify_algebra(**policy)
    explicit = source.simplify_algebra(**policy, color_substitute_cof_dimension_invariants=True)
    closure = symbolic.simplify_algebra(**policy, color_substitute_cof_dimension_invariants=True)
    if any(value.reduction_status != ReductionStatus.Complete for value in (symbolic, explicit, closure)):
        raise ValueError("Native color reduction did not complete")
    if explicit.to_expression() != closure.to_expression():
        raise ValueError("Native symbolic and explicit color reductions disagree")

    epsilon_names = tuple(TensorName.vector(f"gghh::eps{i + 1}") for i in range(2))
    lorentz_slots = [[slot for slot in leg.slots if slot.representation == Representation.mink(4)]
                    for leg in gluons]
    if any(len(slots) != 1 for slots in lorentz_slots):
        raise ValueError("Native gluon ports do not have one four-dimensional Lorentz slot")
    projector = epsilon_names[0](lorentz_slots[0][0])
    projector *= epsilon_names[1](lorentz_slots[1][0])
    diagram = fs.with_diagram_expressions(
        raw, numerator=explicit.to_expression(), projector=projector.to_expression(),
        overall_factor=raw.overall_factor_expression(evaluate=True),
    )

    incoming = sorted(leg.index for leg in legs if leg.state == "incoming")
    outgoing = sorted(leg.index for leg in legs if leg.state == "outgoing")
    vectors = [[E(x) for x in row] for row in (
        ["150", "0", "0", "150"], ["150", "0", "0", "-150"],
        ["150", "15*11^(1/2)", "0", "20*11^(1/2)"],
        ["150", "-15*11^(1/2)", "0", "-20*11^(1/2)"],
    )]
    if len(incoming) != 2 or len(outgoing) != 2:
        raise ValueError("Expected two incoming and two outgoing external states")
    physical = dict(zip(incoming + outgoing, vectors))
    external = {edge.id: edge.external_index for edge in raw.external_edges}
    basis = raw.loop_momentum_basis
    coordinates = [physical[external[edge]] for edge in basis.external_edges]
    P, Q = hep.Kinematics.external_momentum(), hep.Symbols.edge_momentum()
    for edge in basis.external_edges:
        routed = basis.route_expression(Q(edge))
        for component in range(4):
            actual = routed
            for index, vector in enumerate(coordinates):
                actual = actual.replace(P(index), vector[component])
            if (actual - physical[external[edge]][component]).expand() != E("0"):
                raise ValueError("Native external routing changed the physical point")

    states = [hep.FourMomentum(150, 0, 0, z).wavefunction("epsilon", hep.Helicity.PLUS)
              for z in (150, -150)]
    polarizations = [_tensor(f"gghh_data::eps{i}", _components(state)) for i, state in enumerate(states)]
    for index, (state, polarization) in enumerate(zip(states, polarizations)):
        if _dot(polarization, polarization) != E("0"):
            raise ValueError("The transported circular polarization is not exactly null")
        _near(_dot(polarization, _tensor(f"gghh_data::bar{index}", _components(state.bar()))), -1)
        _near(_dot(_tensor(f"gghh_data::incoming{index}", physical[incoming[index]]), polarization), 0)
    _near(_dot(*polarizations), -1)
    named = [(P(i), _tensor(f"gghh_data::p{i}", coordinates[i]))
             for i, edge in enumerate(basis.external_edges) if edge not in basis.dependent_externals]
    auxiliaries = tuple(name.to_expression() for name in epsilon_names)
    named.extend(zip(auxiliaries, polarizations))
    regulator, dimension = S("gghh::eps", "gghh::D")
    K = hep.Kinematics.loop_momentum()
    kinematics = hep.Kinematics(dimension, momenta=[K(i) for i in range(raw.loop_count)] + [name for name, _ in named])
    for i, (left, a) in enumerate(named):
        for right, b in named[i:]:
            kinematics = kinematics.with_scalar_product(left, right, _dot(a, b))
    return GGHHInput(
        "gg → HH · single top double box (+,+)", model, diagram, kinematics,
        regulator, 4 - 2 * regulator, scalar_values, 0, auxiliaries, raw, tuple(events), origin,
    )
