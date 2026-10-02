"""The configured license key is applied in each Python process."""

import sys
from types import ModuleType
from unittest.mock import Mock

from tests._license import configure_license_key


def test_configure_license_key_uses_environment(monkeypatch):
    set_license_key = Mock()
    symbolica = ModuleType("symbolica")
    symbolica.set_license_key = set_license_key
    monkeypatch.setitem(sys.modules, "symbolica", symbolica)
    monkeypatch.setenv("SYMBOLICA_LICENSE_KEY", "test-key")

    configure_license_key()

    set_license_key.assert_called_once_with("test-key")


def test_configure_license_key_is_optional(monkeypatch):
    set_license_key = Mock()
    symbolica = ModuleType("symbolica")
    symbolica.set_license_key = set_license_key
    monkeypatch.setitem(sys.modules, "symbolica", symbolica)
    monkeypatch.delenv("SYMBOLICA_LICENSE_KEY", raising=False)

    configure_license_key()

    set_license_key.assert_not_called()
