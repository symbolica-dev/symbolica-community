"""Exercise the actual inlined marimo cells and their UV expansion."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from symbolica import E, S, Expression
from symbolica.community.hep import Model, SnailFilterOptions
from symbolica.community.spenso import TensorExpression, as_tensor


@pytest.fixture(scope='module')
def notebook():
    pytest.importorskip('marimo', minversion='0.24.2', reason='Notebook examples require marimo')
    from examples.hep_showcase import app
    # Keep regression inputs stable while the interactive generation cell is edited.
    model = Model(str(Path(__file__).parents[1] / 'examples/hep_sm.json'))
    diagrams = model.generate_diagrams(
        ['g'], ['g'], loops=1, coupling_orders={'QCD': 2, 'QED': 0},
        particle_veto=['c', 't', 's', 'u', 'd'], zero_snails=SnailFilterOptions(),
    )
    gluon_index = next(i for i, d in enumerate(diagrams) if d.internal_edges[0].particle_name == 'g')
    _, definitions = app.run(defs={
        'diagrams': diagrams, 'loops': 1, 'report': None, 'status': None,
        'diagram_index': SimpleNamespace(value=gluon_index),
    })
    return SimpleNamespace(**definitions)


def test_projected_numerators_have_no_external_indices(notebook):
    n = notebook
    for diagram in n.diagrams:
        projected = n.scalar_products(n.apply_projector(n.route_numerator(diagram), n.external_projector(diagram)))
        assert not as_tensor(projected).list_dangling()


def test_tensor_helpers_preserve_external_interface(notebook):
    n = notebook
    for diagram in n.diagrams:
        numerator = n.numerator_in_d(diagram)
        contracted = n.contract_indices(numerator)
        routed = n.route_numerator(diagram)
        projector = n.external_projector(diagram)
        for tensor in (numerator, contracted, routed, projector):
            assert isinstance(tensor, TensorExpression)
            assert tensor.rank == 4
            assert set(tensor.list_dangling()) == set(numerator.list_dangling())
        projected = n.scalar_products(n.apply_projector(routed, projector))
        weighted = n.resolve_qcd(as_tensor(projected * n.diagram_weight(diagram, n.model)), n.model)
        for tensor in (projected, weighted):
            assert isinstance(tensor, TensorExpression)
            assert tensor.is_scalar
        scalar = n.invariants(weighted)
        assert isinstance(scalar, TensorExpression)
        assert scalar.is_scalar
        with pytest.raises(ValueError, match='external indices'):
            n.invariants(contracted)


def test_uv_stages_preserve_tensor_type(notebook):
    n = notebook
    for name in (
        'graph_propagator_product', 'graph_denominator', 'projected_integrand',
        'uv_expression_copy', 'uv_deformed_expression', 'uv_scaled_expression', 'uv_with_measure',
        'uv_indexed_series', 'uv_tensor_reduced',
        'uv_reduced_series', 'uv_reduced_expression', 'uv_counterterm_integrand',
        'uv_residue', 'uv_counterterm', 'total_uv_residue', 'total_uv_counterterm',
    ):
        tensor = getattr(n, name)
        assert isinstance(tensor, TensorExpression), name
        assert tensor.is_scalar, name
    assert n.uv_expression_copy is not n.projected_integrand


def test_expression_copy_uv_poles_and_subtraction(notebook):
    n = notebook
    expected = {'ghG': E('1/4'), 'g': E('19/4'), 'b': E('-2/3')}
    total, ward = E('0'), E('0')
    for diagram in n.diagrams:
        original_graph = diagram.to_json()
        numerator = n.resolve_qcd(n.scalar_products(n.apply_projector(n.route_numerator(diagram), n.external_projector(diagram))) * n.diagram_weight(diagram, n.model), n.model)
        expression = n.invariants(numerator) * n.graph_propagators(diagram, n.model)
        result = n.uv_expansion_data(expression, loop_count=diagram.loop_count)
        for tensor in result.values():
            assert isinstance(tensor, TensorExpression)
            assert tensor.is_scalar
        assert result['copy'] == expression
        assert result['copy'] is not expression
        assert diagram.to_json() == original_graph
        assert result['residue'] == expected[diagram.internal_edges[0].particle_name] * n.p2.replace(n.D, 4) * S('UFO::G')**2
        assert result['residue'].replace(S('UFO::MB'), 0) == result['residue']
        assert not result['residue'].contains(n.mUV)
        assert (n.evaluate_propagators(result['deformed']).replace(n.t, 1)
                - n.evaluate_propagators(expression)).cancel() == E('0')
        # The directly expanded massive counterterm cancels the original UV tail.
        k_vector = n.K(n.mink(n.D))
        original_tail = n.evaluate_propagators(expression).replace(k_vector, k_vector/n.uv_probe)
        original_tail = original_tail.series(n.uv_probe, 0, 4).to_expression().expand()
        averaged_tail = n.uv_scalar_invariants(n.tensor_reducer.reduce(n.uv_tensor_input(original_tail)))
        ct_tail = result['counterterm'].replace(k_vector, k_vector/n.uv_probe)
        ct_tail = ct_tail.series(n.uv_probe, 0, 4).to_expression()
        assert (ct_tail + averaged_tail).expand().cancel() == E('0')
        assert result['counterterm'].contains(n.mUV)
        total += result['residue']
        longitudinal = n.resolve_qcd(n.scalar_products(n.apply_projector(n.route_numerator(diagram), n.external_projector(diagram, longitudinal=True))) * n.diagram_weight(diagram, n.model), n.model)
        ward += n.uv_expansion_data(n.invariants(longitudinal) * n.graph_propagators(diagram, n.model), loop_count=diagram.loop_count)['residue']
    assert total.expand() == E('13/3') * n.p2.replace(n.D, 4) * S('UFO::G')**2
    assert ward.expand() == E('0')


def test_uv_measure_and_logarithmic_truncation(notebook):
    n = notebook
    assert n.uv_measure_factor == n.t**(-4*n.diagram.loop_count)
    assert n.uv_series.get_trailing_exponent() == (-2, 1)
    assert n.uv_series.get_absolute_order() == (1, 1)
    # Removing measure bookkeeping must keep the same integrand terms as before.
    integrand_series = n.uv_series.to_expression()/n.uv_measure_factor
    previous = n.uv_scaled_expression.to_expression().series(n.t, 0, 4).to_expression()
    assert (integrand_series-previous).expand() == E('0')
    assert integrand_series.coefficient(n.t**4) != E('0')


def test_scalar_scaling_factors_out_of_dots(notebook):
    n = notebook
    k_vector, p_vector = n.K(n.mink(n.D)), n.P(n.mink(n.D))
    assert n.t.is_scalar()
    assert n.k2 == n.dot(k_vector, k_vector)
    assert n.kp == n.dot(k_vector, p_vector)
    assert n.p2 == n.dot(p_vector, p_vector)
    assert n.dot(k_vector/n.t, k_vector/n.t) == n.k2/n.t**2
    assert n.dot(k_vector/n.t, p_vector) == n.kp/n.t
    evaluated = n.evaluate_propagators(n.projected_integrand)
    scalar_scaled = evaluated.replace(n.k2, n.k2/n.t**2).replace(n.kp, n.kp/n.t)
    vector_scaled = evaluated.replace(k_vector, k_vector/n.t)
    assert vector_scaled == scalar_scaled


@pytest.mark.parametrize('rank', [1, 2, 3, 4, 6])
def test_feynkit_tensor_reduction(notebook, rank):
    n = notebook
    expected = {
        1: E('0'),
        2: n.k2*n.p2/n.D,
        3: E('0'),
        4: 3*n.k2**2*n.p2**2/(n.D*(n.D+2)),
        6: 15*n.k2**3*n.p2**3/(n.D*(n.D+2)*(n.D+4)),
    }[rank]
    indexed = n.uv_tensor_input(n.kp**rank)
    assert isinstance(indexed, TensorExpression)
    reduced = n.uv_scalar_invariants(n.tensor_reducer.reduce(indexed))
    assert (reduced-expected).cancel() == E('0')


def test_uv_rejects_unsupported_topology(notebook):
    model = Model(str(Path(__file__).parents[1] / 'examples/hep/scalar_phi3.json'))
    triangle = model.generate_diagrams(
        ['scalar_0'], ['scalar_0', 'scalar_0'], loops=1,
        max_vertices=3, allow_self_loops=False,
    )[0]
    with pytest.raises(ValueError, match='two-propagator bubbles'):
        notebook.graph_propagators(triangle, model)


def test_massive_uv_expansion_point(notebook):
    n = notebook
    k, p = n.K(n.mink(n.D)), n.P(n.mink(n.D))
    mass = S('hep_uv_test::mass', is_scalar=True)
    original = n.prop(k+p, mass**2)
    deformed = n.evaluate_propagators(n.uv_deform(original))
    expected = n.t**2 / (n.dot(k+n.t*p, k+n.t*p) - n.t**2*mass**2 - (1-n.t**2)*n.mUV**2)
    assert (deformed-expected).together() == E('0')
    assert ((deformed/n.t**2).together().replace(n.t, 0) - 1/(n.k2-n.mUV**2)).together() == E('0')
    joint = n.k2 * n.prop(k, mass**2) * original
    series = n.evaluate_propagators(n.uv_deform(joint)).series(n.t, 0, 4).to_expression()
    assert (series.coefficient(n.t**2) - n.k2/(n.k2-n.mUV**2)**2).cancel() == E('0')
    assert (n.evaluate_propagators(n.uv_deform(joint)).replace(n.t, 1)
            - n.evaluate_propagators(joint)).cancel() == E('0')


def test_denominators_come_from_model_propagators(notebook):
    import json
    n = notebook
    source = json.loads(n.model.to_json())
    extra_mass = S('hep_uv_test::extra_mass', is_scalar=True)
    for item in source['propagators']:
        if item['particle'] == 'g':
            item['denominator'] += '-hep_uv_test::extra_mass^2'
    modified_model = Model.from_json(json.dumps(source))
    diagrams = modified_model.generate_diagrams(
        ['g'], ['g'], loops=1, coupling_orders={'QCD': 2, 'QED': 0},
        particle_veto=['c', 't', 's', 'u', 'd', 'b', 'ghG'],
    )
    diagram = next(d for d in diagrams if len(d.internal_edges) == 2)
    assert modified_model.particle('g').is_massless
    expected = E('1')
    for edge in diagram.internal_edges:
        a, b = n.routing_coefficients(diagram)[edge.id]
        q = a*n.K(n.mink(n.D)) + b*n.P(n.mink(n.D))
        expected /= n.dot(q, q)-extra_mass**2
    actual = n.evaluate_propagators(n.graph_propagators(diagram, modified_model))
    assert (actual-expected).cancel() == E('0')
