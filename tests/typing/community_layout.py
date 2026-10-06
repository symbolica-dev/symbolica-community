"""Type-check public packages from their shipped Python/stub layout."""

from symbolica import E, Expression, get_citations
from symbolica.community import graph, tensor
from symbolica.community import hepkit as hep
from symbolica.community.hepkit import ibp
from symbolica.community.hepkit.vakint import (
    Vakint,
    VakintEvaluationMethod,
    VakintExpression,
    VakintNumericalResult,
)
from symbolica.community.spenso import TensorExpression as LegacyTensorExpression
from symbolica.community.vakint import Vakint as LegacyVakint

expression: tensor.TensorExpression = tensor.TensorExpression(E("1"))
scalar: Expression = expression.to_expression()
layout: graph.LayoutSettings = graph.LayoutSettings(impred_steps=1)
stroke: graph.Stroke = graph.Stroke(thickness=1)
tensor_drawing: graph.DiagramRender = tensor.TensorNetwork(E("1")).render(
    config=graph.RenderSettings(
        layouts=layout, drawing=graph.DrawOptions(edge_stroke=stroke)
    )
)
legacy_tensor_type: type[tensor.TensorExpression] = LegacyTensorExpression
legacy_vakint_type: type[Vakint] = LegacyVakint
vakint_type: type[Vakint] = hep.vakint.Vakint
vakint_expression: VakintExpression = VakintExpression(E("0"))
references: list[str] = [citation.to_bibtex() for citation in get_citations()]

flat_ibp_family: type[hep.IBPFamily] = ibp.IBPFamily
subpackage_ibp_family: type[ibp.IBPFamily] = hep.IBPFamily
flat_ibp_rule: type[hep.IBPRule] = ibp.IBPRule
subpackage_ibp_rule: type[ibp.IBPRule] = hep.IBPRule
flat_ibp_solution: type[hep.IBPSolution] = ibp.IBPSolution
subpackage_ibp_solution: type[ibp.IBPSolution] = hep.IBPSolution


def make_vakint(method: VakintEvaluationMethod) -> Vakint:
    return Vakint(evaluation_order=[method])


def as_expression(engine: Vakint, result: VakintNumericalResult) -> Expression:
    return engine.numerical_result_to_expression(result)


def factor_projectors(
    symbolic: tensor.TensorExpression, concrete: tensor.Tensor
) -> None:
    # The Python 3.9-compatible Unpack spelling must retain concrete overloads.
    symbolic_group: tensor.FactorProjector[tensor.TensorExpression] = (
        tensor.FactorProjector.symmetric(symbolic, symbolic)
    )
    symmetric: tensor.FactorProjector[tensor.TensorNetwork] = (
        tensor.FactorProjector.symmetric(concrete, symbolic)
    )
    antisymmetric: tensor.FactorProjector[tensor.TensorNetwork] = (
        tensor.FactorProjector.antisymmetric(symbolic, concrete)
    )
    cyclic: tensor.FactorProjector[tensor.TensorNetwork] = (
        tensor.FactorProjector.cyclic(symbolic, concrete, symbolic)
    )


def routed_campaign(
    family: hep.IBPFamily,
    artifact: hep.rustred.CandidateArtifact,
) -> tuple[
    list[tuple[Expression, Expression]],
    hep.rustred.CandidateGenerationSession,
    hep.rustred.TerminalNormalization,
]:
    bindings: list[tuple[Expression, Expression]] = family.parameter_bindings
    generation: hep.rustred.CandidateGenerationSession = family.start_generation(
        event_capacity=64,
        nonpositive_indices=[0],
    )
    terminals: hep.rustred.TerminalNormalization = family.normalize_candidate_terminals(
        artifact,
        max_terminals=10,
        max_supports=10,
        max_matrix_cells=100,
        max_output_terms=100,
    )
    return bindings, generation, terminals
