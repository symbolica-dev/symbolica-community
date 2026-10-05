"""Checked scalar-vacuum ingress using HEPKit's graph and Symbolica primitives.

This is a representation adapter, not a momentum router, tensor reducer or IBP
engine. No topology, coefficient or momentum expression is parsed from text.
"""

from symbolica import N, Expression, Replacement, S


def integral_from_diagram(
    diagram, family, numerator, *, powers=None, parameter_substitutions=None,
    external_momenta=(),
):
    """Build a native :class:`VakintExpression` from a routed vacuum graph.

    ``family`` must retain the graph's ordered physical denominator prefix in
    its stored loop basis; additional auxiliary slots are permitted only with
    nonpositive powers. Physical powers must be positive integers (not bools).
    The default powers are one for physical propagators and zero for auxiliaries.

    ``numerator`` is a scalar Symbolica expression built from HEPKit
    ``Kinematics.scalar_product`` in the family's dimension. Spectator vectors
    must be listed explicitly in ``external_momenta``; their order defines
    Vakint's numerical external-vector IDs 1, 2, ... . Integrated loop momenta
    retain family order and map to Vakint k(1), k(2), ... . Unconverted Lorentz
    tensors or undeclared momentum products are refused, not guessed.

    ``parameter_substitutions`` is an explicit mapping of scalar Symbolica
    variables to scalar expressions. It is applied simultaneously to the
    graph's physical denominators and masses, and to the supplied numerator.
    The supplied family must already contain these same substituted physical
    denominators. No mass or dimension is inferred from a prior IBP artifact.

    Cuts, external propagators, dummy/dangling edges and non-vacuum families
    are outside this initial adapter. Graph symmetry factors and interaction
    numerators are not inserted: ``numerator`` is the complete requested factor.
    Existing raw Vakint APIs remain available for other supported inputs.
    """
    from symbolica.community import hepkit as hep
    from symbolica.community.hepkit_vakint_native import VakintExpression

    if not isinstance(diagram, hep.FeynmanDiagram) or not isinstance(family, hep.IntegralFamily):
        raise TypeError("expected a HEPKit FeynmanDiagram and IntegralFamily")
    if not isinstance(numerator, Expression):
        raise TypeError("numerator must be a native scalar Symbolica Expression")
    if diagram.external_edges or diagram.cuts or family.external_momenta:
        raise ValueError("only uncut vacuum graphs are supported; numerator spectators are explicit")
    diagram.validate()
    edges = sorted(diagram.internal_edges, key=lambda edge: edge.id)
    if not edges or any(edge.is_dummy or edge.is_dangling for edge in edges):
        raise ValueError("a vacuum topology needs non-dummy, non-dangling propagators")
    if len({edge.id for edge in edges}) != len(edges):
        raise ValueError("duplicate graph edge IDs")
    # Vakint integrates unrestricted loop momenta. Do not silently transport
    # an on-shell/scalar-product assumption on an integrated momentum.
    graph_kinematics = hep.Kinematics(family.kinematics.dimension)
    physical = diagram.propagator_family(kinematics=graph_kinematics)
    loops = family.loop_momenta
    if loops != physical.loop_momenta or len(loops) != diagram.loop_count:
        raise ValueError("family loop basis does not match the stored graph routing")
    spectators = tuple(external_momenta)
    if any(not isinstance(momentum, Expression) for momentum in spectators):
        raise TypeError("external_momenta must contain native Symbolica momentum names")
    momenta = [*loops, *spectators]
    if len(set(momenta)) != len(momenta):
        raise ValueError("integrated and spectator momentum names must be distinct")
    # get_head accepts concrete names only; sums/non-momentum descriptors refuse.
    momentum_heads = {momentum.get_head() for momentum in momenta}

    def scalar_symbols(expression):
        symbols = set(expression.get_all_symbols(include_function_symbols=True))
        return not (symbols & momentum_heads) and not any(
            symbol.get_name().startswith("spenso::") for symbol in symbols
        )

    substitutions = dict(parameter_substitutions or {})
    for source, target in substitutions.items():
        if not isinstance(source, Expression) or not isinstance(target, Expression):
            raise TypeError("parameter substitutions must map native Expressions to Expressions")
        # A scalar variable is its own head; a function call is not.
        if source != source.get_head() or not scalar_symbols(source) or not scalar_symbols(target):
            raise ValueError("parameter substitutions must be scalar, not momentum/representation rewrites")
    replacements = [Replacement(source, target) for source, target in substitutions.items()]

    def substitute(expression):
        return expression.replace_multiple(replacements) if replacements else expression

    denominators = family.denominators
    expected = [substitute(denominator) for denominator in physical.denominators]
    if len(denominators) < len(edges) or len(expected) != len(edges) or any(
        (actual - wanted).expand() != N(0)
        for actual, wanted in zip(denominators, expected)
    ):
        raise ValueError("family physical denominator order/sign/masses do not match the graph")
    if powers is None:
        powers = [1] * len(edges) + [0] * (len(denominators) - len(edges))
    else:
        powers = list(powers)
    if len(powers) != len(denominators) or any(type(power) is not int for power in powers):
        raise ValueError("powers must be one integer (not bool) per family denominator")
    if any(power <= 0 for power in powers[:len(edges)]) or any(power > 0 for power in powers[len(edges):]):
        raise ValueError("physical powers must be positive; auxiliary powers must be nonpositive")

    # Build incidence from exact graph IDs and momenta from its integer signatures.
    k, p, dot, prop, edge_head, topo = S(
        "vakint::k", "vakint::p", "vakint::dot", "vakint::prop", "vakint::edge", "vakint::topo"
    )
    mapped = [k(i + 1) for i in range(len(loops))] + [p(i + 1) for i in range(len(spectators))]
    topology = N(1)
    for index, (edge, power) in enumerate(zip(edges, powers)):
        signature = edge.momentum_signature()
        if len(signature.loops) != len(loops) or any(signature.external):
            raise ValueError("stored graph routing is not the declared vacuum loop basis")
        if edge.source is None or edge.target is None:
            raise ValueError("propagator has an incomplete incidence")
        momentum = sum((coefficient * mapped[i] for i, coefficient in enumerate(signature.loops)), N(0))
        original_momentum = sum((coefficient * loops[i] for i, coefficient in enumerate(signature.loops)), N(0))
        # Reuse the native denominator convention, including its UFO ZERO
        # handling, instead of reimplementing model mass extraction.
        mass_squared = substitute((physical.kinematics.scalar_product(
            original_momentum, original_momentum) - physical.denominators[index]).expand())
        if not scalar_symbols(mass_squared):
            raise ValueError("graph denominator does not have a scalar mass squared")
        topology *= prop(edge.id, edge_head(edge.source, edge.target), momentum,
                         mass_squared, power)

    scalar = substitute(numerator)
    for denominator, power in zip(denominators[len(edges):], powers[len(edges):]):
        scalar *= denominator ** (-power)
    free = hep.Kinematics(family.kinematics.dimension, momenta=momenta)
    products = [
        Replacement(free.scalar_product(left, right), dot(mapped[i], mapped[j]))
        for i, left in enumerate(momenta) for j, right in enumerate(momenta[i:], i)
    ]
    scalar = scalar.expand().replace_multiple(products)
    if not scalar_symbols(scalar):
        raise ValueError("numerator contains unconverted momentum/tensor notation; use declared scalar products")
    return VakintExpression(scalar * topo(topology))
