"""Native loop integration annotations reuse the host's symbolic and HEP types."""
from symbolica import ComplexFloat, Expression, Float
from symbolica.community.hepkit import FeynmanDiagram, IntegralFamily, Kinematics
from symbolica.community.hep.integration import (
    BoundaryCache,
    BoundaryData,
    ComputationControl,
    DifferentialSystem,
    EvaluationOptions,
    IntegralEvaluator,
    KinematicTransport,
    LaurentExpansion,
    PreparedIntegralFamily,
    TransportResult,
)


def evaluate_family(
    family: IntegralFamily,
    point: dict[Expression, Expression],
    epsilon: Expression,
    variable: Expression,
    sample: Expression,
) -> tuple[LaurentExpansion, PreparedIntegralFamily]:
    evaluator = IntegralEvaluator(
        options=EvaluationOptions(digits=20), reduction_batch_size=4
    )
    results: list[LaurentExpansion] = evaluator.evaluate(
        family, [[1]], point, epsilon, last=0, control=ComputationControl()
    )
    coefficient: ComplexFloat = results[0].coefficients[-1]
    error: Float = results[0].comparison_errors[-1]
    samples: list[list[ComplexFloat]] = evaluator.evaluate_samples(
        family, [[1]], point, epsilon, [sample]
    )
    prepared: PreparedIntegralFamily = evaluator.prepare(
        family, [[1]], [variable], epsilon, branch_domain="positive real variable"
    )
    return results[0], prepared


def evaluate_diagram(
    diagram: FeynmanDiagram, kinematics: Kinematics,
    point: dict[Expression, Expression], epsilon: Expression,
) -> LaurentExpansion:
    return IntegralEvaluator().evaluate_diagram(diagram, kinematics, point, epsilon)


def propagate(
    system: DifferentialSystem, flow: KinematicTransport, boundary: BoundaryData,
    cache: BoundaryCache, point: dict[Expression, Expression], endpoint: ComplexFloat,
) -> tuple[TransportResult, TransportResult]:
    cache.extend(BoundaryCache())
    direct: TransportResult = system.transport(boundary, [endpoint])
    cached: TransportResult = flow.evaluate(cache, point, 0, 2, admit_straight_path=True)
    return direct, cached
