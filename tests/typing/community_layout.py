"""Type-check public packages from their shipped Python/stub layout."""

from symbolica import E, Expression, get_citations
from symbolica.community import hep, tensor
from symbolica.community.hep.vakint import (
    Vakint,
    VakintEvaluationMethod,
    VakintExpression,
    VakintNumericalResult,
)
from symbolica.community.spenso import TensorExpression as LegacyTensorExpression
from symbolica.community.vakint import Vakint as LegacyVakint

expression: tensor.TensorExpression = tensor.TensorExpression(E("1"))
scalar: Expression = expression.to_expression()
legacy_tensor_type: type[tensor.TensorExpression] = LegacyTensorExpression
legacy_vakint_type: type[Vakint] = LegacyVakint
vakint_type: type[Vakint] = hep.vakint.Vakint
vakint_expression: VakintExpression = VakintExpression(E("0"))
references: list[str] = [citation.to_bibtex() for citation in get_citations()]


def make_vakint(method: VakintEvaluationMethod) -> Vakint:
    return Vakint(evaluation_order=[method])


def as_expression(engine: Vakint, result: VakintNumericalResult) -> Expression:
    return engine.numerical_result_to_expression(result)
