"""Numerical loop integration and cached kinematic continuation.

Families, diagrams, and symbolic kinematics are supplied by
``symbolica.community.hepkit``. Numerical values retain Symbolica's native
arbitrary precision. Native and browser builds include automatic reduction and
boundary generation, supplied-boundary transport and reusable intermediate
points. Inspect ``automatic_boundary_generation_available`` in custom builds.
Browser calls use one worker and run synchronously; queued progress can be read
after completion, and cancellation is checked between operations. Browser cache
files live in the virtual filesystem and must be exported for durable storage.
"""

from symbolica.community.hep_integration_native import *
from symbolica.community.hep_integration_native import __all__ as __all__
