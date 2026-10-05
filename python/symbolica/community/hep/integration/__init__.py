"""Numerical loop integration and cached kinematic continuation.

Families, diagrams, and symbolic kinematics are supplied by
``symbolica.community.hepkit``. Numerical values retain Symbolica's native
arbitrary precision. These solvers require a native community build.
"""

from symbolica.community.hep_integration_native import *
from symbolica.community.hep_integration_native import __all__ as __all__
