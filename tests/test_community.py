"""Tests for symbolica-community extensions."""
import pytest


def test_spenso_import():
    """Test that spenso module can be imported."""
    from symbolica.community.spenso import Tensor, TensorIndices, Representation
    assert Tensor is not None
    assert TensorIndices is not None
    assert Representation is not None
