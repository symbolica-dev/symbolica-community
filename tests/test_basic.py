"""Basic tests for symbolica core functionality."""
import pytest
from symbolica import E


def test_expand():
    """Test expansion of (x+1)^2."""
    assert E('(x+1)^2').expand() == E('x^2+2*x+1')
