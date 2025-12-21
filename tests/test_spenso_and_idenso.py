"""Tests for Spenso and Idenso"""

import random

import pytest
from symbolica import E, Expression, S
from symbolica.community.spenso import (
    ExecutionMode,
    LibraryTensor,
    Representation,
    Slot,
    Tensor,
    TensorIndices,
    TensorLibrary,
    TensorNetwork,
    TensorStructure,
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
        assert LibraryTensor is not None
        assert TensorNetwork is not None
        assert Representation is not None
        assert TensorStructure is not None
        assert TensorIndices is not None
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
        gamma = N.gamma()
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
        self.gamma = N.gamma()
        self.p = N("P")
        self.w = N("w")
        self.g = N.g()
        self.mq = S("mq")

    def test_create_tensor_indices(self):
        """Test creating tensor indices"""
        other_g = self.gamma(self.i, self.k, self.mu)
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)

        assert other_g is not None
        assert g_muik is not None
        assert str(g_muik) is not None
        assert str(other_g) is not None

    def test_tensor_indices_indexing(self):
        """Test indexing tensor indices"""
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
        result = g_muik[2]
        assert result is not None

    def test_tensor_to_expression(self):
        """Test converting tensor to expression"""
        expr = self.gamma(self.k, self.j, self.mu).to_expression()
        assert expr is not None

    def test_tensor_slicing(self):
        """Test tensor slicing"""
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
        result = g_muik[45:63:3]
        assert result is not None

    def test_tensor_multi_indexing(self):
        """Test tensor multi-indexing"""
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
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
        self.gamma = N.gamma()
        self.p = N("P")
        self.w = N("w")
        self.g = N.g()
        self.mq = S("mq")

    def test_tensor_network_creation(self):
        """Test creating tensor network from expression"""
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
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
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
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
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
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
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
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
        structure = t.structure()
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
        self.gamma = N.gamma()
        self.p = N("P")
        self.w = N("w")
        self.g = N.g()
        self.mq = S("mq")

        # Create tensor network and execute
        g_muik = TensorIndices(self.i, self.k, self.mu, name=self.gamma)
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
        e = self.t.evaluator(constants=constants, params=params, funs={})
        assert e is not None

        # Test evaluation without compilation
        e_params = [
            random.random() + 1j * random.random() for _ in range(len(params))
        ]
        eval_res = e.evaluate_complex([e_params])[0]

        assert eval_res is not None
        assert eval_res.structure() is not None

    def test_tensor_compilation(self):
        """Test tensor compilation"""
        params = [Expression.I]
        params += TensorNetwork(self.w(1, self.i)).result_tensor()
        params += TensorNetwork(self.w(3, self.mu)).result_tensor()
        params += TensorNetwork(self.p(2, self.nu)).result_tensor()
        constants = {self.mq: E("173")}

        e = self.t.evaluator(constants=constants, params=params, funs={})

        # The evaluator can be compiled to a shared library
        c = e.compile(
            function_name="f",
            filename="test_expression.cpp",
            library_name="test_expression.so",
            inline_asm="none",
        )

        assert c is not None


class TestLibraryTensors:
    """Tests for library tensor operations."""

    def test_sparse_library_tensor(self):
        """Test creating and manipulating sparse library tensors"""
        custom = Representation("custom", 4, is_self_dual=False)
        mq = S("mq")

        t = LibraryTensor.sparse([custom, custom], type(mq))
        # Note that the structure is a list of representations, not slots
        structure = t.structure()
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
        tname = S("test")

        # Dense tensors are built from a list of values in row-major order.
        t = LibraryTensor.dense(
            TensorStructure(d, d, name=tname),
            [0, 0, 123, 11, 3, 234, 234, 23, 44],
        )

        # Test element assignment
        t[[1, 2]] = 3 / 34
        assert t is not None
        assert t.structure() is not None

        lib = TensorLibrary.hep_lib()
        lib.register(t)

        new_t = t.structure()

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

        self.gam = self.lib["spenso::gamma"]

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
        self.gam = self.lib["spenso::gamma"]
        self.fc = S("spenso::f")
        self.ps = S("p")
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
        from symbolica.community.idenso import simplify_metrics

        result = simplify_metrics(self.bis.g(4, 2) * self.gam(2, 3, 1))
        assert result is not None

    def test_simplify_metrics_bis_trace(self):
        """Test metric simplification with bis trace"""
        from symbolica.community.idenso import simplify_metrics

        result = simplify_metrics(self.bis.g(1, 1).to_expression())
        assert result is not None

    def test_simplify_metrics_euclidean_trace(self):
        """Test metric simplification with euclidean trace"""
        from symbolica.community.idenso import simplify_metrics

        result = simplify_metrics(
            Representation.euc("d").g(1, 1).to_expression()
        )
        assert result is not None

    def test_simplify_gamma_chain(self):
        """Test gamma matrix simplification"""
        from symbolica.community.idenso import simplify_gamma

        # Define p function as in the notebook
        def p(i):
            from symbolica import Expression

            minkd = Representation("mink", "D")
            if isinstance(i, str):
                m = minkd(i).to_expression()
            elif isinstance(i, int):
                m = minkd(i).to_expression()
            else:
                m = minkd(i).to_expression()
            return self.ps(m)

        a = simplify_gamma(
            self.gam(1, 2, 1)
            * self.gam(2, 3, "mu")
            * self.gam(3, 4, 1)
            * self.gam(4, 1, 2)
            * p("mu")
            * p(2)
        )
        assert a is not None

    def test_to_dots_conversion(self):
        """Test conversion to dots notation"""
        from symbolica.community.idenso import simplify_gamma, to_dots

        # Define p function as in the notebook
        def p(i):
            from symbolica import Expression

            minkd = Representation("mink", "D")
            if isinstance(i, str):
                m = minkd(i).to_expression()
            elif isinstance(i, int):
                m = minkd(i).to_expression()
            else:
                m = minkd(i).to_expression()
            return self.ps(m)

        a = simplify_gamma(
            self.gam(1, 2, 1)
            * self.gam(2, 3, "mu")
            * self.gam(3, 4, 1)
            * self.gam(4, 1, 2)
            * p("mu")
            * p(2)
        )

        dots_result = to_dots(a)
        assert dots_result is not None

    def test_simplify_color_structure(self):
        """Test color structure simplification"""
        from symbolica.community.idenso import simplify_color

        result = simplify_color(self.f_func(1, 2, 3) * self.f_func(3, 2, 1))
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
    gamma = N.gamma()
    w = N("w")

    # Create simple tensor expression
    expr = gamma(i, i, mu) * w(1, mu)

    # Create and execute tensor network
    tn = TensorNetwork(expr)
    tn.execute()

    result = tn.result_tensor()
    assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])
