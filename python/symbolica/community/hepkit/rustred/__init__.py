"""Native RustRed generation and lazy artifact views in HEPKit's shared kernel.

Start from ``hepkit.IBPFamily(...).start_generation(...)`` to retain the usual
HEPKit Graph, routing, and integral-family objects. This namespace exposes the
same session/result types and optional text-input entry point, without importing
a second Symbolica extension. Availability requires a native community build.
"""

from symbolica.community.hepkit_native import rustred as _native

CandidateArtifact = _native.CandidateArtifact
CandidateBundleResult = _native.CandidateBundleResult
CandidateGenerationRequest = _native.CandidateGenerationRequest
CandidateGenerationSession = _native.CandidateGenerationSession
TerminalNormalization = _native.TerminalNormalization
candidate_generation_request = _native.candidate_generation_request
start_family_candidates = _native.start_family_candidates
RustRedError = _native.RustRedError
RustRedInputError = _native.RustRedInputError
RustRedSchemaError = _native.RustRedSchemaError
RustRedLimitError = _native.RustRedLimitError
RustRedLoweringError = _native.RustRedLoweringError
RustRedDerivationError = _native.RustRedDerivationError
RustRedExecutionError = _native.RustRedExecutionError
RustRedLicenseError = _native.RustRedLicenseError
RustRedSerializationError = _native.RustRedSerializationError
RustRedOutputLimitError = _native.RustRedOutputLimitError
RustRedInternalError = _native.RustRedInternalError
RustRedCoordinatorPoisonedError = _native.RustRedCoordinatorPoisonedError

__all__ = [
    "CandidateArtifact", "CandidateBundleResult", "CandidateGenerationRequest",
    "CandidateGenerationSession", "TerminalNormalization", "candidate_generation_request",
    "start_family_candidates",
    "RustRedError", "RustRedInputError", "RustRedSchemaError", "RustRedLimitError",
    "RustRedLoweringError", "RustRedDerivationError", "RustRedExecutionError",
    "RustRedLicenseError", "RustRedSerializationError", "RustRedOutputLimitError",
    "RustRedInternalError", "RustRedCoordinatorPoisonedError",
]


def __getattr__(name):
    # Importing this convenience package must not hide the rest of the embedded
    # native API previously reachable through hepkit.rustred.
    return getattr(_native, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_native)))
