"""Basic tests for symbolica core functionality."""
import pytest
from symbolica import E, S


def test_expand():
    """Test expansion of (x+1)^2."""
    assert E('(x+1)^2').expand() == E('x^2+2*x+1')


def test_multiprecision_rational_arithmetic():
    """Exercise the numeric backend beyond machine-sized integers."""
    assert E(str(2**256)) + E('1/3') + E('2/3') == E(str(2**256 + 1))


def test_symbolic_integration():
    assert E('x^2').integrate(S('x')) == E('x^3/3')


def test_integration_step_metadata():
    """Compressed rule metadata must survive decompression into the Python API."""
    expression = E('1/(1+x^2)')
    result, overview, steps = expression.integrate_with_steps(S('x'))
    assert result == expression.integrate(S('x')) == E('atan(x)')
    assert overview.strip()
    assert any(step.rule is not None and step.source and step.description for step in steps)
