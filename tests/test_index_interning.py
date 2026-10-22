"""Index interning admits native labels and encodes only unsupported payloads."""

import json

import pytest
from symbolica import E, S
from symbolica.community.hepkit import FeynmanDiagram, Model
from symbolica.community.tensor import (
    CookingError,
    Representation,
    TensorExpression,
    TensorName,
)

NATIVE_LABELS = [
    "index_intern_regression::mu",
    "0",
    "17",
    "-7",
    "3/5",
    "gammalooprs::hedge(17,2)",
    "gammalooprs::edge(7,0)",
    "gammalooprs::vertex(3,1)",
    "spenso::open_index(17,2)",
    "spenso::index_scope(index_intern_regression::scope,17)",
    "spenso::index_scope(index_intern_regression::scope,gammalooprs::hedge(17,2))",
    (
        "spenso::index_scope(index_intern_regression::outer,"
        "spenso::index_scope(index_intern_regression::scope,gammalooprs::edge(7,0)))"
    ),
]

MALFORMED_LABELS = [
    "gammalooprs::hedge()",
    "gammalooprs::hedge(17)",
    "gammalooprs::hedge(17,2,3)",
    "gammalooprs::hedge(-1,2)",
    "gammalooprs::hedge(17,-1)",
    "gammalooprs::edge(1/2,0)",
    "gammalooprs::edge(7,1/2)",
    "gammalooprs::edge(18446744073709551616,0)",
    "gammalooprs::vertex(index_intern_regression::owner,0)",
    "gammalooprs::vertex(3,index_intern_regression::local)",
    "gammalooprs::vertex(3,index_intern_regression::opaque(1))",
    "spenso::open_index(17)",
    "spenso::open_index(17,-1)",
    "spenso::index_scope(index_intern_regression::scope)",
    "spenso::index_scope(1,17)",
    ("spenso::index_scope(index_intern_regression::scope,gammalooprs::hedge(17))"),
]


@pytest.fixture(scope="module", autouse=True)
def qcd_model():
    # This registers the actual tagged graph labels, including edge and vertex.
    return Model.qcd()


@pytest.fixture
def vector():
    return TensorName.vector("index_intern_regression::v")


@pytest.fixture(scope="module")
def three_gluon_graph(qcd_model):
    return FeynmanDiagram.from_dot(
        qcd_model,
        """digraph native_indices {
            ext [style=invis];
            ext -> a [particle="g"];
            ext -> a [particle="g"];
            a -> ext [particle="g"];
        }""",
    ).apply_feynman_rules()


@pytest.mark.parametrize("intern", ["indices", "flattened"])
@pytest.mark.parametrize("label", NATIVE_LABELS)
def test_constructor_preserves_native_index_bytes(vector, label, intern):
    payload = E(label)
    scalar = S("index_intern_regression::parameter")(9)
    raw = S("index_intern_regression::v")(scalar, S("spenso::euc")(3, payload))
    original = raw.__getstate__()
    native = TensorExpression(raw)

    admitted = TensorExpression(raw, intern=intern)

    assert admitted.to_expression().__getstate__() == original
    assert admitted.structure == native.structure
    assert admitted.axes == native.axes
    assert admitted.rank == 1
    assert admitted.arguments == (scalar,)
    assert raw.__getstate__() == original


@pytest.mark.parametrize("intern", ["indices", "flattened"])
@pytest.mark.parametrize("label", NATIVE_LABELS)
def test_explicit_index_preserves_native_index_bytes(vector, label, intern):
    payload = E(label)
    scalar = S("index_intern_regression::parameter")(9)
    raw = S("index_intern_regression::v")(scalar, S("spenso::euc")(3, payload))

    indexed = vector(scalar, Representation.euc(3)).index(payload, intern=intern)

    assert indexed.to_expression().__getstate__() == raw.__getstate__()
    assert indexed.axes[0].index == payload
    assert indexed.rank == 1
    assert indexed.arguments == (scalar,)


@pytest.mark.parametrize("intern", ["indices", "flattened"])
@pytest.mark.parametrize("label", MALFORMED_LABELS)
@pytest.mark.parametrize("entrypoint", ["constructor", "index"])
def test_malformed_known_indices_are_rejected(vector, label, intern, entrypoint):
    payload = E(label)
    raw = S("index_intern_regression::v")(S("spenso::euc")(3, payload))
    with pytest.raises(CookingError):
        if entrypoint == "constructor":
            TensorExpression(raw, intern=intern)
        else:
            vector(Representation.euc(3)).index(payload, intern=intern)


@pytest.mark.parametrize("intern", ["indices", "flattened"])
def test_mixed_native_and_unsupported_indices_keep_scalar_arguments(vector, intern):
    native = E("gammalooprs::hedge(17,2)")
    opaque = E("index_intern_regression::opaque(17,2)")
    matrix = TensorName("index_intern_regression::mixed")
    head = S("index_intern_regression::mixed")
    euc = S("spenso::euc")
    raw = head(opaque, euc(3, native), euc(3, opaque))
    with pytest.raises(ValueError):
        TensorExpression(raw)

    admitted = TensorExpression(raw, intern=intern)
    encoded = vector(Representation.euc(3)).index(opaque, intern=intern).axes[0].index

    assert encoded != opaque
    assert admitted.to_expression() == head(opaque, euc(3, native), euc(3, encoded))
    assert admitted.axes[0].index == native
    assert admitted.axes[1].index == encoded
    assert admitted.arguments == (opaque,)
    assert admitted.name.to_expression() == matrix.to_expression()
    if intern == "indices":
        assert encoded.get_symbol_data() == opaque
    assert (
        TensorExpression(admitted.to_expression(), intern=intern)
        .to_expression()
        .__getstate__()
        == admitted.to_expression().__getstate__()
    )


@pytest.mark.parametrize("intern", ["indices", "flattened"])
def test_mixed_native_and_unsupported_indices_contract_identically(vector, intern):
    space = Representation.euc(3)
    native = E("gammalooprs::edge(7,0)")
    opaque = E("index_intern_regression::opaque(17,2)")
    left, right = (
        S("index_intern_regression::left"),
        S("index_intern_regression::right"),
    )
    metric = TensorExpression.g(space(left), space(right)).to_expression()
    reference = (
        metric
        * vector(space(left)).to_expression()
        * vector(space(right)).to_expression()
    )
    raw = reference.replace(left, native).replace(right, opaque)

    actual = TensorExpression(raw, intern=intern)
    expected = TensorExpression(reference).contract()

    assert actual.rank == 0
    assert actual.contract().to_expression() == expected.to_expression()


@pytest.mark.parametrize("intern", ["indices", "flattened"])
def test_explicit_index_still_encodes_unsupported_functions(vector, intern):
    payload = E("index_intern_regression::opaque(17,2)")
    unresolved = vector(Representation.euc(3))
    with pytest.raises(ValueError):
        unresolved.index(payload)

    indexed = unresolved.index(payload, intern=intern)

    assert indexed.rank == 1
    assert indexed.axes[0].index != payload
    if intern == "indices":
        assert indexed.axes[0].index.get_symbol_data() == payload
    assert (
        TensorExpression(indexed.to_expression(), intern=intern).to_expression()
        == indexed.to_expression()
    )


@pytest.mark.parametrize("intern", ["indices", "flattened"])
def test_native_renaming_preserves_nonzero_and_ranked_zero_interfaces(vector, intern):
    old = E("gammalooprs::hedge(17,2)")
    new = E("gammalooprs::vertex(3,1)")
    original = vector(Representation.euc(3)).index(old)
    expected = vector(Representation.euc(3)).index(new)
    zero = TensorExpression(0, structure=original.structure)

    renamed = original.rename_indices({old: new}, intern=intern)
    renamed_zero = zero.rename_indices({old: new}, intern=intern)

    assert (
        renamed.to_expression().__getstate__()
        == expected.to_expression().__getstate__()
    )
    assert renamed.axes == expected.axes
    assert renamed_zero.to_expression() == 0
    assert renamed_zero.rank == 1
    assert renamed_zero.axes == expected.axes
    assert renamed_zero.structure == expected.structure
    assert original.axes[0].index == old
    assert zero.axes == original.axes


@pytest.mark.parametrize("intern", ["indices", "flattened"])
def test_index_only_interning_preserves_scalar_algebra(intern):
    raw = E(
        "index_intern_regression::opaque(gammalooprs::hedge(17),"
        "gammalooprs::vertex(3,1))+index_intern_regression::parameter(9)^2"
    )

    tensor = TensorExpression(raw, intern=intern)

    assert tensor.is_scalar
    assert tensor.to_expression().__getstate__() == raw.__getstate__()


def test_graph_json_import_preserves_native_labels(qcd_model, three_gluon_graph):
    encoded = three_gluon_graph.to_json()
    native = TensorExpression(E(json.loads(encoded)["numerator"]))

    restored = FeynmanDiagram.from_json(qcd_model, encoded)
    numerator = restored.numerator_expression()

    assert numerator.rank == 6
    assert numerator.axes == native.axes
    assert numerator.to_expression() == native.to_expression()
    roundtrip = FeynmanDiagram.from_json(qcd_model, restored.to_json())
    assert roundtrip.numerator_expression().axes == numerator.axes
    assert (
        roundtrip.numerator_expression().to_expression().__getstate__()
        == numerator.to_expression().__getstate__()
    )


@pytest.mark.parametrize("intern", ["indices", "flattened"])
def test_graph_json_ranked_zero_retains_declared_port_order(
    qcd_model, three_gluon_graph, intern
):
    payload = json.loads(three_gluon_graph.to_json())
    native = TensorExpression(E(payload["numerator"]))
    declared = tuple(reversed(native.axes))
    ports = [slot.to_expression().to_canonical_string() for slot in declared]
    # Mirror the Rust with_numerator owner: one ranked anchor and unit spectators.
    for vertex in payload["vertices"]:
        vertex["numerator"] = "1"
        vertex.pop("num_interface", None)
    for _, edge in payload["edges"]:
        edge["numerator"] = "1"
        edge.pop("num_interface", None)
    anchor = next(
        vertex for vertex in payload["vertices"] if vertex["interaction"] is not None
    )
    anchor["numerator"] = "0"
    anchor["num_interface"] = ports
    payload["numerator"] = "0"
    payload["num_interface"] = ports

    restored = FeynmanDiagram.from_json(qcd_model, json.dumps(payload))
    zero = restored.numerator_expression()

    assert zero.to_expression() == 0
    assert zero.rank == 6
    assert zero.axes == declared
    old, new = E("gammalooprs::hedge(0,1)"), E("gammalooprs::vertex(99,0)")
    renamed = zero.rename_indices({old: new}, intern=intern)
    assert renamed.to_expression() == 0
    assert renamed.rank == zero.rank
    assert tuple(slot.index for slot in renamed.axes) == tuple(
        new if slot.index == old else slot.index for slot in declared
    )
    assert tuple(slot.representation for slot in renamed.axes) == tuple(
        slot.representation for slot in declared
    )
    roundtrip = FeynmanDiagram.from_json(qcd_model, restored.to_json())
    assert roundtrip.numerator_expression().axes == declared
