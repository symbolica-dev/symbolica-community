"""High-energy physics tools for symbolic and numerical calculations.

Models, Feynman diagrams, generation, CFFs, tensor reduction, and kinematics
are available directly in this namespace, for example ``hep.FeynmanDiagram``.
``hep.oneloop`` provides symbolic one-loop reduction, with scalar master
evaluation available in native builds. Native builds also provide exact
parametric and Laporta IBP solving through ``hep.IBPFamily``.
"""

from ..hep_native import *

initialize_module()
del initialize_module
