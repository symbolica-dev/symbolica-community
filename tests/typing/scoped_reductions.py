"""Supplied tables borrow native families and preserve exact symbolic outputs."""

from symbolica import Expression
from symbolica.community.hepkit import IntegralFamily
from symbolica.community.hep.integration import (
    IntegralEvaluator,
    PreparedIntegralFamily,
    ReductionTables,
)


def prepare_supplied(
    family: IntegralFamily, epsilon: Expression, mass_squared: Expression,
) -> PreparedIntegralFamily:
    tables: ReductionTables = ReductionTables().with_family(
        family, epsilon,
        [([2], [([1], (1 - epsilon) / mass_squared)])],
        [[1]], nonzero_conditions=[mass_squared],
    )
    prepared = IntegralEvaluator(reductions=tables).prepare(
        family, [[1], [2]], [mass_squared], epsilon,
        branch_domain="positive mass squared",
    )
    reductions: list[list[tuple[list[int], Expression]]] = prepared.target_reductions
    conditions: list[Expression] = prepared.nonzero_conditions
    return prepared
