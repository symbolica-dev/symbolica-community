"""Exact definite integration and access to HEPkit's existing IBP tools.

``integrate`` uses hyperlogarithms over [0, +infinity), in variable order;
``integrate_over`` accepts explicit directed intervals. Symbolica's
``Expression.integrate(x)`` continues to compute an antiderivative.

Browser execution is serial, including when ``parallel=True``. IBP remains
available only in native installations.
"""
from symbolica.community.hepkit_integration_native import *
from symbolica.community.hepkit_integration_native import __all__ as _native_all
from symbolica.community import hepkit_native as _hep

__all__ = list(_native_all)
if hasattr(_hep, "IBPFamily"):
    from .. import ibp as ibp
    __all__.append("ibp")
__all__.sort()
del _native_all, _hep
