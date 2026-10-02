"""Exercise the actual one-loop notebook cells and independent UV identities."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from symbolica import E, S
from symbolica.community.hepkit import Model
from symbolica.community.tensor import TensorExpression


@pytest.fixture(scope="module")
def notebook():
    pytest.importorskip("marimo", minversion="0.24.0")
    from examples.hep_showcase import app

    _, definitions = app.run()
    return SimpleNamespace(**definitions)


def projected_integrand(n, diagram, *, longitudinal=False):
    numerator = (
        n.project_color(diagram).simplify_algebra(gamma=True, epsilon=True).contract()
    )
    routed = diagram.loop_momentum_basis.route_expression(
        numerator, loop_momenta=[n.K], external_momenta=[n.P]
    )
    projected = n.apply_projector(
        routed, n.external_projector(routed, longitudinal=longitudinal)
    )
    assert projected.is_scalar
    weighted = TensorExpression(
        n.model.expand_couplings(projected.to_expression() * n.diagram_weight(diagram))
    )
    return weighted * n.graph_propagators(diagram)


def test_tensor_helpers_preserve_external_interface(notebook):
    n = notebook
    for diagram in n.diagrams:
        numerator = n.project_color(diagram)
        contracted = numerator.simplify_algebra(gamma=True, epsilon=True).contract()
        projector = n.external_projector(contracted)
        assert numerator.rank == contracted.rank == projector.rank == 2
        assert numerator.structure.slots == contracted.structure.slots
        assert set(projector.list_dangling()) == set(numerator.list_dangling())
        assert n.apply_projector(contracted, projector).is_scalar


def test_uv_stages_preserve_tensor_type(notebook):
    n = notebook
    for name in (
        "graph_propagator_product",
        "graph_denominator",
        "projected_integrand",
        "uv_expression_copy",
        "uv_deformed_expression",
        "uv_scaled_expression",
        "uv_with_measure",
        "uv_tensor_reduced",
        "uv_reduced_series",
        "uv_reduced_expression",
        "uv_counterterm_integrand",
        "uv_residue",
        "uv_counterterm",
        "total_uv_residue",
    ):
        tensor = getattr(n, name)
        assert isinstance(tensor, TensorExpression), name
        assert tensor.is_scalar, name
    assert n.uv_expression_copy is not n.projected_integrand


def test_expression_copy_uv_poles_and_subtraction(notebook):
    n = notebook
    expected = {"ghG": E("1/4"), "g": E("19/4"), "b": E("-2/3")}
    total, ward = E("0"), E("0")
    for diagram in n.diagrams:
        original_graph = diagram.to_json()
        expression = projected_integrand(n, diagram)
        result = n.uv_expansion_data(expression, diagram)
        assert all(
            isinstance(value, TensorExpression) and value.is_scalar
            for value in result.values()
        )
        assert result["copy"] == expression
        assert result["copy"] is not expression
        assert diagram.to_json() == original_graph
        residue = result["residue"].to_expression()
        reference = (
            expected[diagram.internal_edges[0].particle_name]
            * n.p2.replace(n.D, 4)
            * S("UFO::G") ** 2
        )
        assert (residue - reference).expand() == E("0")
        assert (residue.replace(S("UFO::MB"), 0) - residue).expand() == E("0")
        assert (residue.replace(n.mUV, 0) - residue).expand() == E("0")
        assert (
            n.evaluate_propagators(result["deformed"]).to_expression().replace(n.t, 1)
            - n.evaluate_propagators(expression).to_expression()
        ).cancel() == E("0")
        # Independently Taylor-expand the original integrand's radial UV tail.
        tail = (
            n.evaluate_propagators(expression)
            .to_expression()
            .replace(n.k, n.k / n.uv_probe)
        )
        tail = TensorExpression(tail.series(n.uv_probe, 0, 4).to_expression())
        averaged = n.uv_scalar_invariants(
            diagram.tensor_reduce(n.D, expression=n.uv_tensor_input(tail, diagram)),
            diagram,
        ).to_expression()
        counterterm = result["counterterm"].to_expression()
        ct_tail = (
            counterterm.replace(n.k, n.k / n.uv_probe)
            .series(n.uv_probe, 0, 4)
            .to_expression()
        )
        assert (ct_tail + averaged).expand().cancel() == E("0")
        assert counterterm.contains(n.mUV)
        total += residue
        ward += n.uv_expansion_data(
            projected_integrand(n, diagram, longitudinal=True), diagram
        )["residue"].to_expression()
    assert (total - E("13/3") * n.p2.replace(n.D, 4) * S("UFO::G") ** 2).expand() == E(
        "0"
    )
    assert ward.expand() == E("0")


def test_uv_measure_and_logarithmic_truncation(notebook):
    n = notebook
    assert n.uv_measure_factor == n.t ** (-4 * n.diagram.loop_count)
    assert n.uv_series.get_trailing_exponent() == (-2, 1)
    assert n.uv_series.get_absolute_order() == (1, 1)
    series = n.uv_series.to_expression() / n.uv_measure_factor
    reference = n.uv_scaled_expression.to_expression().series(n.t, 0, 4).to_expression()
    assert (series - reference).expand() == E("0")
    assert series.coefficient(n.t**4) != E("0")


def test_scalar_scaling_factors_out_of_dots(notebook):
    n = notebook
    kp = n.dot(n.K(n.lorentz), n.P(n.lorentz)).to_expression()
    assert n.t.is_scalar()
    assert (
        n.dot(n.K(n.lorentz) / n.t, n.K(n.lorentz) / n.t).to_expression()
        == n.k2 / n.t**2
    )
    assert n.dot(n.K(n.lorentz) / n.t, n.P(n.lorentz)).to_expression() == kp / n.t
    evaluated = n.evaluate_propagators(n.projected_integrand).to_expression()
    scalar_scaled = evaluated.replace(n.k2, n.k2 / n.t**2).replace(kp, kp / n.t)
    assert evaluated.replace(n.k, n.k / n.t) == scalar_scaled


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 6])
def test_feynkit_tensor_reduction(notebook, rank):
    n = notebook
    kp = n.dot(n.K(n.lorentz), n.P(n.lorentz)).to_expression()
    expected = {
        1: E("0"),
        2: n.k2 * n.p2 / n.D,
        3: E("0"),
        4: 3 * n.k2**2 * n.p2**2 / (n.D * (n.D + 2)),
        6: 15 * n.k2**3 * n.p2**3 / (n.D * (n.D + 2) * (n.D + 4)),
    }[rank]
    indexed = n.uv_tensor_input(TensorExpression(kp**rank), n.diagram)
    reduced = n.uv_scalar_invariants(
        n.diagram.tensor_reduce(n.D, expression=indexed), n.diagram
    )
    assert (reduced.to_expression() - expected).cancel() == E("0")


def test_uv_rejects_unsupported_topology(notebook):
    model = Model(Path(__file__).parents[1] / "examples/hep/scalar_phi3.json")
    triangle = (
        model.process(["scalar_0"], ["scalar_0", "scalar_0"])
        .generate_diagrams(
            loops=1,
            max_vertices=3,
            allow_self_loops=False,
            progress=None,
        )
        .diagrams[0]
    )
    with pytest.raises(ValueError, match="two-propagator bubbles"):
        notebook.graph_propagators(triangle)


def test_massive_uv_expansion_point(notebook):
    n = notebook
    p = n.P(n.lorentz).to_expression()
    mass = S("hep_uv_test::mass", is_scalar=True)
    original = TensorExpression(n.prop(n.k + p, mass**2))
    deformed = n.evaluate_propagators(n.uv_deform(original)).to_expression()
    scalar = n.dot(
        n.K(n.lorentz) + n.t * n.P(n.lorentz), n.K(n.lorentz) + n.t * n.P(n.lorentz)
    ).to_expression()
    expected = n.t**2 / (scalar - n.t**2 * mass**2 - (1 - n.t**2) * n.mUV**2)
    assert (deformed - expected).together() == E("0")
    assert (
        (deformed / n.t**2).together().replace(n.t, 0) - 1 / (n.k2 - n.mUV**2)
    ).together() == E("0")
    joint = TensorExpression(n.k2 * n.prop(n.k, mass**2) * original.to_expression())
    series = (
        n.evaluate_propagators(n.uv_deform(joint))
        .to_expression()
        .series(n.t, 0, 4)
        .to_expression()
    )
    assert (series.coefficient(n.t**2) - n.k2 / (n.k2 - n.mUV**2) ** 2).cancel() == E(
        "0"
    )
    assert (
        n.evaluate_propagators(n.uv_deform(joint)).to_expression().replace(n.t, 1)
        - n.evaluate_propagators(joint).to_expression()
    ).cancel() == E("0")


def test_denominators_come_from_model_propagators(notebook):
    n = notebook
    source = json.loads(n.model.to_json())
    extra_mass = S("hep_uv_test::extra_mass", is_scalar=True)
    for item in source["propagators"]:
        if item["particle"] == "g":
            item["denominator"] += "-hep_uv_test::extra_mass^2"
    model = Model.from_json(json.dumps(source))
    diagrams = (
        model.process(["g"], ["g"], particle_veto=["c", "t", "s", "u", "d", "b", "ghG"])
        .generate_diagrams(
            loops=1,
            coupling_orders={"QCD": 2, "QED": 0},
            progress=None,
        )
        .diagrams
    )
    diagram = next(d for d in diagrams if len(d.internal_edges) == 2)
    expected = E("1")
    for edge in diagram.internal_edges:
        momentum = diagram.loop_momentum_basis.route_expression(
            edge.momentum_expression(dimension=n.D),
            loop_momenta=[n.K],
            external_momenta=[n.P],
        )
        expected /= n.dot(momentum, momentum).to_expression() - extra_mass**2
    actual = n.evaluate_propagators(n.graph_propagators(diagram)).to_expression()
    assert (actual - expected).cancel() == E("0")
