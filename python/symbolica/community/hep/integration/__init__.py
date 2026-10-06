"""Numerical loop integration and cached kinematic continuation.

Families, diagrams, and symbolic kinematics are supplied by
``symbolica.community.hepkit``. Numerical values retain Symbolica's native
arbitrary precision. Native builds include automatic reduction and boundary
generation. Browser builds provide transport from supplied boundaries and cached
intermediate points; inspect ``automatic_boundary_generation_available`` before
requesting automatic generation.
"""

from symbolica.community.hep_integration_native import *
from symbolica.community.hep_integration_native import __all__ as __all__
