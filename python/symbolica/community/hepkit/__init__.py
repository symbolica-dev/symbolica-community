"""High-energy physics tools for symbolic and numerical calculations.

Models, Feynman diagrams, generation, CFFs, tensor reduction, and kinematics
are available directly in this namespace, for example ``hepkit.FeynmanDiagram``.
``hepkit.oneloop`` provides symbolic one-loop reduction, with scalar master
evaluation available in native builds. Exact parametric and Laporta IBP solving
through ``hepkit.IBPFamily`` is available in native and Pyodide builds.
"""

from symbolica.community.hepkit_native import *

initialize_module()
del initialize_module

from . import oneloop as oneloop

from . import sector_decomposition as sector_decomposition
from .sector_decomposition import (
    CompilationSettings as CompilationSettings,
    StabilitySettings as StabilitySettings,
    GenerationSession as GenerationSession,
    IntegrationObservation as IntegrationObservation,
    SectorContribution as SectorContribution,
    LiveObservation as LiveObservation,
    LiveSector as LiveSector,
    LiveEstimate as LiveEstimate,
    EvaluatorTiming as EvaluatorTiming,
)
from . import integration as integration
