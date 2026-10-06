"""Sector decomposition of native HEPKit diagrams and integral families.

``sector_decompose`` generates a Laurent integrand. Compilation and numerical
integration remain explicit, caller-owned operations on the returned objects.
"""

from symbolica.community import hepkit_native as _native

if not hasattr(_native, "_fastsecdec_native"):
    raise ImportError(
        "Sector decomposition requires a community wheel built with "
        "experimental-fastsecdec; see examples/hep/FASTSECDEC_BUILD.md"
    )
del _native

from symbolica.community.hepkit_fastsecdec_native import *
