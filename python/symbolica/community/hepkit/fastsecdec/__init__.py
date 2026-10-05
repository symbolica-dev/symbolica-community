"""Native graph-to-Laurent integration with caller-owned QMC stepping.

Generation and compilation are synchronous. Their observer callbacks transport
native progress; event-loop responsiveness depends on the host. ``step`` bounds
numerical work and returns control to the Python caller after each batch.
"""

from symbolica.community import hepkit_native as _native

if not hasattr(_native, "_fastsecdec_native"):
    raise ImportError(
        "FastSecDec requires a community wheel built with experimental-fastsecdec; "
        "see examples/hep/FASTSECDEC_BUILD.md"
    )
del _native

from symbolica.community.hepkit_fastsecdec_native import *
