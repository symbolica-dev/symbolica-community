"""Pytest configuration and fixtures."""
import os
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_license_key():
    """Set the symbolica license key for all tests."""
    # Check if license key is set in environment variable first
    license_key = os.environ.get('SYMBOLICA_LICENSE_KEY')

    if license_key:
        from symbolica import set_license_key as set_key
        set_key(license_key)

    yield
