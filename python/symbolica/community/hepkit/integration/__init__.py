"""Exact definite integration using hyperlogarithms.

``integrate`` uses hyperlogarithms over [0, +infinity), in variable order;
``integrate_over`` accepts explicit directed intervals. Symbolica's
``Expression.integrate(x)`` continues to compute an antiderivative.

Browser execution is serial, including when ``parallel=True``. HEPkit's
existing native IBP tools are available separately through ``hepkit.ibp``.
"""
from symbolica.community.hepkit_integration_native import *
from symbolica.community.hepkit_integration_native import __all__ as __all__
