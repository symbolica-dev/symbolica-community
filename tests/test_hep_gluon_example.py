"""Exercise the actual inlined marimo cells and their UV expansion."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from symbolica import E, S, Expression
from symbolica.community.hep import Model
from symbolica.community.spenso import TensorExpression, as_tensor


@pytest.fixture(scope='module')
def notebook():
    pytest.importorskip('marimo', minversion='0.24.2', reason='Notebook examples require marimo')
    from examples.hep_showcase import app
    _, definitions = app.run()
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
        assert isinstance(scalar, Expression)
        assert not isinstance(scalar, TensorExpression)
        with pytest.raises(ValueError, match='external indices'):
            n.invariants(contracted)


def test_expression_copy_uv_poles_and_subtraction(notebook):
    n = notebook
    expected = {'ghG': E('1/4'), 'g': E('19/4'), 'b': E('-2/3')}
    total, ward = E('0'), E('0')
    for diagram in n.diagrams:
        original_graph = diagram.to_json()
        numerator = n.resolve_qcd(n.scalar_products(n.apply_projector(n.route_numerator(diagram), n.external_projector(diagram))) * n.diagram_weight(diagram, n.model), n.model)
        expression = n.invariants(numerator) / n.bubble_denominator(diagram, n.model)
        result = n.uv_expansion_data(expression)
        assert result['copy'] == expression
        assert result['copy'] is not expression
        assert diagram.to_json() == original_graph
        assert result['residue'] == expected[diagram.internal_edges[0].particle_name] * n.p2 * S('UFO::G')**2
        assert result['residue'].replace(S('UFO::MB'), 0) == result['residue']
        a, b = result['quadratic'], result['logarithmic']
        ct = -a/(n.k2-n.Muv2) - (b-a*n.Muv2)/(n.k2-n.Muv2)**2
        # The regulated, angular-averaged counterterm cancels all UV orders.
        k_vector = n.K(n.mink(n.D))
        ct_series = ct.replace(k_vector, k_vector/n.t).series(n.t, 0, 4).to_expression()
        assert (ct_series + result['reduced']).expand().cancel() == E('0')
        # Its integrated pole is independent of the arbitrary IR regulator.
        assert (-a*n.Muv2-(b-a*n.Muv2)+b).expand() == E('0')
        total += result['residue']
        longitudinal = n.resolve_qcd(n.scalar_products(n.apply_projector(n.route_numerator(diagram), n.external_projector(diagram, longitudinal=True))) * n.diagram_weight(diagram, n.model), n.model)
        ward += n.uv_expansion_data(n.invariants(longitudinal) / n.bubble_denominator(diagram, n.model))['residue']
    assert total.expand() == E('13/3') * n.p2 * S('UFO::G')**2
    assert ward.expand() == E('0')


def test_scalar_scaling_factors_out_of_dots(notebook):
    n = notebook
    k_vector, p_vector = n.K(n.mink(n.D)), n.P(n.mink(n.D))
    assert n.t.is_scalar()
    assert n.k2 == n.dot(k_vector, k_vector)
    assert n.kp == n.dot(k_vector, p_vector)
    assert n.dot(k_vector/n.t, k_vector/n.t) == n.k2/n.t**2
    assert n.dot(k_vector/n.t, p_vector) == n.kp/n.t
    scalar_scaled = n.projected_integrand.replace(n.k2, n.k2/n.t**2).replace(n.kp, n.kp/n.t)
    vector_scaled = n.projected_integrand.replace(k_vector, k_vector/n.t)
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
        notebook.bubble_denominator(triangle, model)
