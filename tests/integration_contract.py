"""Small exact fixtures shared by native pytest and the Pyodide wheel test."""
import copy
import pickle
from symbolica import E, S, Expression, Symbol
from symbolica.community import hepkit as hep
from symbolica.community.hepkit import integration as api


def check_integration_contract():
    x, a = S("integration_contract::x", "integration_contract::a")
    expr = 1 / (x + 1)**2
    for parallel in (False, True):
        options = api.IntegrationOptions(parallel=parallel, check_divergences=True)
        assert api.integrate(expr, [x], options) == E("1")
        prepared = api.prepare(expr, [x], options)
        options.parallel = not parallel
        assert prepared.options.parallel == parallel
        assert copy.deepcopy(prepared).integrate() == E("1")
        detail = prepared.integrate_detailed()
        assert type(detail.expression) is Expression
        assert detail.variable_count == 1 and not detail.is_zero
        assert copy.copy(detail).expression == detail.expression
        assert api.integrate_over(E("1"), [x], [(E("5"), E("2"))]) == E("-3")
        tail = api.integrate_detailed_over(expr, [x], [(a, Symbol.INFINITY)])
        assert (tail.expression - 1/(a+1)).cancel() == E("0")
        assert a in tail.indeterminates
    assert type(pickle.loads(pickle.dumps(expr))) is Expression
    assert pickle.loads(pickle.dumps(expr)) == expr
    assert copy.deepcopy(expr) == expr
    assert (x**2).integrate(x) == x**3/3
    for cls in (api.IntegrationOptions, api.PreparedIntegral, api.IntegrationResult,
                api.AlgebraicLetter, api.IntegrationError, api.InputError):
        assert cls.__module__ == "symbolica.community.hepkit.integration"
    try:
        api.prepare(expr, [x, x])
    except api.DuplicateVariableError as error:
        assert isinstance(error, api.IntegrationError)
        assert error.variable == "x"
        restored = pickle.loads(pickle.dumps(error))
        assert type(restored) is type(error) and restored.variable == error.variable
    else:
        raise AssertionError("duplicate variable accepted")
    try:
        api.integrate(1/x, [x], api.IntegrationOptions(check_divergences=True))
    except api.DivergentIntegralError as error:
        assert error.variable == "x" and isinstance(error.power, int)
    else:
        raise AssertionError("divergence was not reported")
    try:
        api.prepare(expr, [x + 1])
    except api.InputError:
        pass
    else:
        raise AssertionError("non-symbol integration variable accepted")
    assert not hasattr(api, "Expression")  # constructors belong to Symbolica
    assert api.__all__ == sorted(set(api.__all__))


def check_symanzik_example():
    # Equal unit masses, p^2=0, D=2, propagator powers (1,1).
    # Strip the loop-measure prefactor; Gamma(1)=1. With x2=1 the
    # projective parameter integral is integral_0^infinity dx1 / F(x1,1).
    k, p, x1, x2 = S("integration_example::k", "integration_example::p",
                     "integration_example::x1", "integration_example::x2")
    kin = hep.Kinematics(E("2"), momenta=[k, p]).with_scalar_product(p, p, E("0"))
    family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k)-1,
        kin.scalar_product(k-p, k-p)-1], kinematics=kin)
    U, F = family.symanzik([x1, x2])
    assert (U - x1 - x2).expand() == E("0")
    assert (F - (x1+x2)**2).expand() == E("0")
    expression = 1/F.replace(x2, E("1"))
    result = api.integrate(expression, [x1], api.IntegrationOptions(check_divergences=True))
    assert type(result) is Expression and result == E("1")
    return result
