"""Shared license setup for pytest and isolated Python test processes."""

import os


def configure_license_key():
    license_key = os.environ.get("SYMBOLICA_LICENSE_KEY")
    if license_key:
        from symbolica import set_license_key

        set_license_key(license_key)
