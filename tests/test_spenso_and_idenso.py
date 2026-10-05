"""Tests for Spenso and Idenso"""

import random

import pytest
from symbolica import E, Expression, S
from symbolica.community.spenso import (
    ExecutionMode,
    Representation,
    Slot,
    Tensor,
    TensorExpression,
    TensorLibrary,
    TensorNetwork,
)
from symbolica.community.spenso import TensorName as N


class TestNotebookBasics:
    """Tests for basic setup and imports from notebook."""

    def test_imports(self):
        """Test that all required imports work (Cell 1)."""
        # This test validates that all imports are working
        assert Expression is not None
        assert S is not None
        assert E is not None
        assert N is not None
        assert TensorExpression is not None
        assert TensorNetwork is not None
        assert Representation is not None
        assert Tensor is not None
        assert Slot is not None
        assert TensorLibrary is not None


class TestRepresentations:
    """Tests for tensor representations."""

    def test_create_representations(self):
        """Test creating different representations"""
        mink = Representation.mink(4)
        bis = Representation.bis(4)
        custom = Representation("custom", 4, is_self_dual=False)

        assert mink is not None
        assert bis is not None
        assert custom is not None
        # Test that string representation works
        assert str(custom) is not None

    def test_create_slot_from_representation(self):
        """Test creating a slot from representation"""
        mink = Representation.mink(4)
        mu = mink("mu")

        assert mu is not None
        assert str(mu) is not None

    def test_slot_to_expression(self):
        """Test converting slot to expression"""
        mink = Representation.mink(4)
        mu = mink("mu")
        mue = mu.to_expression()

        assert mue is not None

    def test_create_slot_directly(self):
        """Test creating slot directly"""
        nu = Slot("mink", 4, "nu")
        assert nu is not None

    def test_create_slots_various_ways(self):
        """Test creating slots with different index types"""
        bis = Representation.bis(4)

        # The index can be a string (that could be parsed into a symbolica symbol)
        i = bis("i")
        # The index can also be an integer
        j = bis(2)

        k = S("k")
        # The index can also directly a symbolica expression
        k = bis(k)

        assert i is not None
        assert j is not None
        assert k is not None
        assert str(k) is not None


class TestTensorNames:
    """Tests for tensor names and basic operations."""

    def test_create_tensor_names(self):
        """Test creating tensor names"""
        gamma = N.dirac_gamma()
        p = N("P")
        w = N("w")
        g = N.g()
        mq = S("mq")

        assert gamma is not None
        assert p is not None
        assert w is not None
        assert g is not None
        assert mq is not None


class TestTensorIndices:
    """Tests for tensor indices operations."""

    def setup_method(self):
        """Set up common objects for tensor indices tests."""
        self.bis = Representation.bis(4)
        self.mink = Representation.mink(4)
        self.i = self.bis("i")
        self.j = self.bis(2)
        self.k = self.bis(S("k"))
        self.mu = self.mink("mu")
        self.nu = self.mink("nu")
        self.gamma = TensorExpression.dirac_gamma(4)
        self.p = N("P")
        self.w = N("w")
        self.g = TensorExpression.g(self.bis)
        self.mq = S("mq")

    def test_create_tensor_indices(self):
        """Test creating tensor indices"""
        other_g = self.gamma(self.i, self.k, self.mu)
        g_muik = self.gamma(self.i, self.k, self.mu)

        assert other_g is not None
        assert g_muik is not None
        assert str(g_muik) is not None
        assert str(other_g) is not None

    def test_tensor_indices_indexing(self):
        """Test indexing tensor indices"""
        g_muik = self.gamma(self.i, self.k, self.mu)
        result = g_muik[2]
        assert result is not None

    def test_tensor_to_expression(self):
        """Test converting tensor to expression"""
        expr = self.gamma(self.k, self.j, self.mu).to_expression()
        assert expr is not None

    def test_tensor_slicing(self):
        """Test tensor slicing"""
        g_muik = self.gamma(self.i, self.k, self.mu)
        result = g_muik[45:63:3]
        assert result is not None

    def test_tensor_multi_indexing(self):
        """Test tensor multi-indexing"""
        g_muik = self.gamma(self.i, self.k, self.mu)
        result = g_muik[[2, 2, 2]]
        assert result is not None


class TestTensorNetwork:
    """Tests for tensor network operations."""

    def setup_method(self):
        """Set up common objects for tensor network tests."""
        self.bis = Representation.bis(4)
        self.mink = Representation.mink(4)
        self.i = self.bis("i")
        self.j = self.bis(2)
        self.k = self.bis(S("k"))
        self.mu = self.mink("mu")
        self.nu = self.mink("nu")
        self.gamma = TensorExpression.dirac_gamma(4)
        self.p = N("P")
        self.w = N("w")
        self.g = TensorExpression.g(self.bis)
        self.mq = S("mq")

    def test_tensor_network_creation(self):
        """Test creating tensor network from expression"""
        g_muik = self.gamma(self.i, self.k, self.mu)
        x = (
            g_muik
            * (
                self.p(2, self.nu) * self.gamma(self.k, self.j, self.nu)
                + self.mq * self.g(self.k, self.j)
            )
            * self.w(1, self.i)
            * self.w(3, self.mu)
        )

        canonical_str = x.to_canonical_string()
        assert canonical_str is not None

    def test_tensor_network_graph(self):
        """Test tensor network graph creation"""
        g_muik = self.gamma(self.i, self.k, self.mu)
        x = (
            g_muik
            * (
                self.p(2, self.nu) * self.gamma(self.k, self.j, self.nu)
                + self.mq * self.g(self.k, self.j)
            )
            * self.w(1, self.i)
            * self.w(3, self.mu)
        )

        tn = TensorNetwork(x)
        # prints the rich graph associated to the network
        assert tn is not None
        assert str(tn) is not None

    def test_tensor_network_execution_scalar(self):
        """Test tensor network execution in scalar mode"""
        g_muik = self.gamma(self.i, self.k, self.mu)
        x = (
            g_muik
            * (
                self.p(2, self.nu) * self.gamma(self.k, self.j, self.nu)
                + self.mq * self.g(self.k, self.j)
            )
            * self.w(1, self.i)
            * self.w(3, self.mu)
        )
        tn = TensorNetwork(x)

        result = tn.execute(n_steps=2, mode=ExecutionMode.Scalar)
        # Should not raise an exception
        assert True

    def test_tensor_network_arithmetic(self):
        """Test tensor network arithmetic operations"""
        t = (
            TensorNetwork.one() * TensorNetwork.zero()
            + TensorNetwork.one() * TensorNetwork.zero()
        )

        assert t is not None
        assert str(t) is not None

        t.execute()
        assert str(t) is not None

    def test_tensor_network_full_execution(self):
        """Test full tensor network execution and result"""
        g_muik = self.gamma(self.i, self.k, self.mu)
        x = (
            g_muik
            * (
                self.p(2, self.nu) * self.gamma(self.k, self.j, self.nu)
                + self.mq * self.g(self.k, self.j)
            )
            * self.w(1, self.i)
            * self.w(3, self.mu)
        )
        tn = TensorNetwork(x)

        tn.execute()
        t = tn.result_tensor()

        assert t is not None

        # Test structure
        structure = t.expression()
        assert structure is not None


class TestTensorEvaluation:
    """Tests for tensor evaluation and compilation."""

    def setup_method(self):
        """Set up tensor for evaluation tests."""
        self.bis = Representation.bis(4)
        self.mink = Representation.mink(4)
        self.i = self.bis("i")
        self.j = self.bis(2)
        self.k = self.bis(S("k"))
        self.mu = self.mink("mu")
        self.nu = self.mink("nu")
        self.gamma = TensorExpression.dirac_gamma(4)
        self.p = N("P")
        self.w = N("w")
        self.g = TensorExpression.g(self.bis)
        self.mq = S("mq")

        # Create tensor network and execute
        g_muik = self.gamma(self.i, self.k, self.mu)
        x = (
            g_muik
            * (
                self.p(2, self.nu) * self.gamma(self.k, self.j, self.nu)
                + self.mq * self.g(self.k, self.j)
            )
            * self.w(1, self.i)
            * self.w(3, self.mu)
        )
        tn = TensorNetwork(x)
        tn.execute()
        self.t = tn.result_tensor()

    def test_tensor_evaluator_creation(self):
        """Test tensor evaluator creation"""
        params = [Expression.I]
        params += TensorNetwork(self.w(1, self.i)).result_tensor()
        params += TensorNetwork(self.w(3, self.mu)).result_tensor()
        params += TensorNetwork(self.p(2, self.nu)).result_tensor()
        constants = {self.mq: E("173")}

        # Much like the expressions, tensors have the same evaluation api
        fixed = self.t.map_components(lambda value: value.replace(self.mq, constants[self.mq]))
        e = fixed.evaluator(params=params, functions=[])
        assert e is not None

        # Test evaluation without compilation
        e_params = [random.random() + 1j * random.random() for _ in range(len(params))]
        eval_res = e.evaluate_complex([e_params])[0]

        assert eval_res is not None
        assert eval_res.expression() is not None

    def test_tensor_compilation(self, tmp_path):
        """Test tensor compilation"""
        params = [Expression.I]
        params += TensorNetwork(self.w(1, self.i)).result_tensor()
        params += TensorNetwork(self.w(3, self.mu)).result_tensor()
        params += TensorNetwork(self.p(2, self.nu)).result_tensor()
        constants = {self.mq: E("173")}

        fixed = self.t.map_components(lambda value: value.replace(self.mq, constants[self.mq]))
        e = fixed.evaluator(params=params, functions=[])

        # The evaluator can be compiled to a shared library
        c = e.compile(
            function_name="f",
            filename=str(tmp_path / "test_expression.cpp"),
            library_name=str(tmp_path / "test_expression.so"),
            number_type="complex",
            inline_asm="none",
        )

        assert c is not None


class TestLibraryTensors:
    """Tests for library tensor operations."""

    def test_sparse_library_tensor(self):
        """Test creating and manipulating sparse library tensors"""
        custom = Representation("custom", 4, is_self_dual=False)
        mq = S("mq")

        t = Tensor.sparse(N("sparse_test")(custom, custom), type(mq))
        # Note that the structure is a list of representations, not slots
        structure = t.expression()
        assert structure is not None

        # Set individual elements
        t[6] = E("f(x)*(1+y)")

        print(t[6])
        assert t is not None

        t[[3, 2]] = E("sin(alpha)")
        assert t is not None

        # Convert to dense
        t.to_dense()
        assert t is not None

    def test_dense_library_tensor_and_network(self):
        """Test creating dense library tensor and tensor network"""
        d = Representation("newrep", 3)

        # Dense tensors are built from a list of values in row-major order.
        t = Tensor.dense(
            N("dense_library_test")(d, d),
            [0, 0, 123, 11, 3, 234, 234, 23, 44],
        )

        # Test element assignment
        t[[1, 2]] = 3 / 34
        assert t is not None
        assert t.expression() is not None

        lib = TensorLibrary.hep_lib()
        lib.register(t)

        new_t = t.expression()

        x = new_t(1, 2) * new_t(2, 3) * new_t(3, 1)
        n = TensorNetwork(x, library=lib)
        n.execute(library=lib)
        result_t = n.result_tensor(library=lib)

        assert result_t is not None


class TestSymbolicOperations:
    """Tests for symbolic operations and simplifications."""

    def setup_method(self):
        """Set up for symbolic operations tests."""
        self.ag = S("spenso::gamma")
        self.minkd = Representation("mink", "D")
        self.fc = S("spenso::f")
        self.ps = S("p")
        self.coad = Representation("coad", 8)
        self.lib = TensorLibrary.hep_lib()

        def to_expression(t):
            if isinstance(t, Expression):
                return t
            elif isinstance(t, Slot):
                return t.to_expression()
            else:
                raise TypeError(f"Expected Expression or Slot, got {type(t)}")

        self.to_expression = to_expression

        self.gam = TensorExpression.dirac_gamma(4)

        def p(i):
            m = to_expression(self.minkd(i))
            return self.ps(
                m,
            )

        def f(i, j, k):
            return self.fc(
                to_expression(self.coad(i)),
                to_expression(self.coad(j)),
                to_expression(self.coad(k)),
            )

        self.p_func = p
        self.f_func = f

    def test_symbolic_setup(self):
        """Test symbolic setup (Cell 43)."""
        assert self.ag is not None
        assert self.minkd is not None
        assert self.fc is not None
        assert self.ps is not None
        assert self.coad is not None
        assert self.gam is not None
        assert callable(self.p_func)
        assert callable(self.f_func)


class TestIdensoSimplifications:
    """Tests for idenso simplification functions."""

    def setup_method(self):
        """Set up for idenso tests."""
        self.bis = Representation.bis(4)
        self.lib = TensorLibrary.hep_lib()
        self.gam = TensorExpression.dirac_gamma(4)
        self.fc = S("spenso::f")
        self.ps = N("idenso_tests::p")
        self.coad = Representation("coad", 8)

        def f(i, j, k):
            return self.fc(
                self.coad(i).to_expression(),
                self.coad(j).to_expression(),
                self.coad(k).to_expression(),
            )

        self.f_func = f

    def test_simplify_metrics_tensor(self):
        """Test metric simplification with tensor"""
        result = (self.bis.g(4, 2) * self.gam(2, 3, 1)).contract()
        assert result is not None

    def test_simplify_metrics_bis_trace(self):
        """Test metric simplification with bis trace"""
        result = self.bis.g(1, 1).contract()
        assert result is not None

    def test_simplify_metrics_euclidean_trace(self):
        """Test metric simplification with euclidean trace"""
        result = Representation.euc("d").g(1, 1).contract()
        assert result is not None

    @pytest.mark.parametrize("dimension", [4, S("D")])
    def test_simplify_gamma_chain(self, dimension):
        """Tr(gamma_mu gamma_nu gamma^mu gamma_rho) in a consistent dimension."""
        gamma = TensorExpression.dirac_gamma(dimension)
        mink = Representation.mink(dimension)
        # Multiply explicit indexed expressions to retain this contracted basis.
        chain = TensorExpression(
            gamma(1, 2, "mu").to_expression()
            * gamma(2, 3, "nu").to_expression()
            * gamma(3, 4, "mu").to_expression()
            * gamma(4, 1, "rho").to_expression()
        )
        expected = 4 * (2 - dimension) * mink.g("nu", "rho").to_expression()
        assert (chain.simplify_algebra(color=False).to_expression() - expected).expand() == 0

    @pytest.mark.parametrize("dimension", [4, S("D")])
    def test_to_dots_conversion(self, dimension):
        """Contract the gamma trace with two momenta and check its scalar value."""
        gamma = TensorExpression.dirac_gamma(dimension)
        mink = Representation.mink(dimension)
        chain = TensorExpression(
            gamma(1, 2, "mu").to_expression()
            * gamma(2, 3, "nu").to_expression()
            * gamma(3, 4, "mu").to_expression()
            * gamma(4, 1, "rho").to_expression()
            * self.ps(mink("nu")).to_expression()
            * self.ps(mink("rho")).to_expression()
        )
        result = chain.simplify_algebra(color=False).expand().contract().to_dots()
        momentum = self.ps(mink).to_expression()
        expected = 4 * (2 - dimension) * S("spenso::dot")(momentum, momentum)
        assert result.rank == 0
        # Self-contractions may remain indexed squares. Compare canonical
        # contractions so the identity is independent of notation/dummy names.
        actual = result.canonize().to_expression()
        expected = TensorExpression(expected).canonize().to_expression()
        assert (actual - expected).expand() == 0

    def test_simplify_color_structure(self):
        """Test color structure simplification"""
        result = TensorExpression(
            self.f_func(1, 2, 3) * self.f_func(3, 2, 1)
        ).simplify_algebra(gamma=False)
        assert result is not None


# Integration test that runs a subset of operations together
def test_notebook_integration():
    """Integration test that combines multiple notebook operations."""
    # Basic setup
    mink = Representation.mink(4)
    bis = Representation.bis(4)

    # Create indices
    mu = mink("mu")
    i = bis("i")

    # Create tensor names
    gamma = N.dirac_gamma()
    w = N("w")

    # Create simple tensor expression
    # Keep explicit indices for component-network evaluation.
    expr = TensorExpression(
        TensorExpression.dirac_gamma(4)(i, i, mu).to_expression()
        * w(1, mu).to_expression()
    )

    # Create and execute tensor network
    tn = TensorNetwork(expr)
    tn.execute()

    result = tn.result_tensor()
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
