"""High-energy physics tools for symbolic and numerical calculations.

Models, Feynman diagrams, generation, CFFs, tensor reduction, and kinematics
are available directly in this namespace, for example ``hepkit.FeynmanDiagram``.
``hepkit.oneloop`` provides symbolic one-loop reduction, with scalar master
evaluation available in native builds. Native builds also provide exact
parametric and Laporta IBP solving through ``hepkit.IBPFamily``.
"""

from symbolica.community.hepkit_native import *

initialize_module()
del initialize_module

from . import oneloop as oneloop

from symbolica.community import hepkit_native as _native

if hasattr(_native, "_fastsecdec_native"):
    from . import sector_decomposition as sector_decomposition
    from . import fastsecdec as fastsecdec
del _native
