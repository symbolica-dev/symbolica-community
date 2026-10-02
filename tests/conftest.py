"""Pytest configuration and fixtures."""
import os


def pytest_configure(config):
    """Make the license available before collection and in child processes."""
    license_key = os.environ.get("SYMBOLICA_LICENSE") or os.environ.get(
        "SYMBOLICA_LICENSE_KEY"
    )
    if license_key:
        # Symbolica reads SYMBOLICA_LICENSE automatically. Keep the old test
        # variable working, including in subprocesses that do not run pytest.
        os.environ["SYMBOLICA_LICENSE"] = license_key
