"""Minimal physics DOT gets the same local rules as generated diagrams."""

import pytest
from symbolica import S
from symbolica.community.hepkit import FeynmanDiagram, Model
from symbolica.community.tensor import Representation, TensorExpression


@pytest.mark.parametrize(
    "model_factory,particle", [(Model.phi3, "phi"), (Model.qcd, "g")]
)
def test_minimal_dot_instantiates_and_simplifies_numerator(model_factory, particle):
    model = model_factory()
    topology = FeynmanDiagram.from_dot(
        model,
        f'''digraph bubble {{
        ext [style=invis];
        ext -> a [particle="{particle}"];
        a -> b [particle="{particle}", lmb_id=0];
        a -> b [particle="{particle}"];
        b -> ext [particle="{particle}"];
    }}''',
    )
    assert topology.numerator_expression().to_expression() == 1
    diagram = topology.apply_feynman_rules()
    diagram.validate()
    assert diagram.id == topology.id
    assert diagram.loop_count == 1
    assert topology.numerator_expression().to_expression() == 1

    numerator = (
        model.expand_couplings(
            diagram.numerator_expression(in_lmb=True)
            * diagram.overall_factor_expression(evaluate=True)
        )
        .with_lorentz_dimension(S("D"))
        .collect_factors()
    )
    assert numerator.to_expression() != 1
    if particle == "g":
        slots = numerator.structure.slots()
        lorentz = [
            s for s in slots if s.representation.name == Representation.mink(4).name
        ]
        color = [
            s for s in slots if s.representation.name == Representation.coad(8).name
        ]
        assert len(slots) == 4
        assert len(lorentz) == len(color) == 2
        numerator = (
            TensorExpression.g(*lorentz) * TensorExpression.g(*color) / 8 * numerator
        )
    for mode in ("dots", "minimal"):
        reduced = numerator.simplify_algebra(
            contract=mode, color=True, gamma=mode == "dots"
        )
        assert reduced.is_scalar
        assert reduced.to_expression() != 0


def test_reapplying_rules_preserves_generated_diagram():
    model = Model.qcd()
    diagram = model.process(["g"], ["g"], particle_selection=["g"]).generate_diagrams(
        loops=1,
        progress=None,
    )[0]
    restored = FeynmanDiagram.from_dot(model, diagram.to_dot()).apply_feynman_rules()
    assert restored.to_json() == diagram.to_json()
