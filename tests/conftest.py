"""Pytest configuration and fixtures."""
import pytest

from tests._license import configure_license_key


@pytest.fixture(scope="session", autouse=True)
def set_license_key():
    """Set the symbolica license key for all tests."""
    configure_license_key()
    yield
