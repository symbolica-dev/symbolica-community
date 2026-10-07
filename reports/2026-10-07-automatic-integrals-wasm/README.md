# Automatic loop evaluation in Pyodide

This milestone enables `IntegralEvaluator`, `PreparedIntegralFamily` and scoped
`ReductionTables` in the Community WASM build. Automatic evaluation uses the
shared RustRed backend and portable Symbolica arithmetic. The shared dependency
guards require this feature while continuing to exclude native GMP/MPFR and
Vakint dependencies.

RustFlow is pinned to `b34ff6b3261bcc48f867d704e1d560e1bc4a4fec`.

## Validation status at publication

- All six dependency ownership/feature graphs passed.
- The 17 focused graph and runner-scope regressions passed, including refusal
  to issue a successful receipt when automatic numerical validation fails.
- A complete development-profile Community wheel compiled successfully. Its
  actual Pyodide run passed shared HEPKit graph/tensor operations, fresh RustRed
  K6 generation and exact reductions, supplied-boundary transport, binary
  restart, nearby-source reuse, cancellation and Standard Model integration.
- The diagnostic automatic numerical gate also passed: exact tadpole IBP and
  scoped supplied reductions, finite-epsilon tadpole and bubble values, a
  20-digit Laurent expansion, a freshly generated 30-digit boundary and
  transport/restart. The complete default Pyodide gate took 544.05 seconds in
  the development profile. The wheel SHA-256 is
  `e4b649dc696f410e2deee2fdefff15ed32fe72520f41a8e58546849d2c9c9fc3`.
  This wheel was built from working sources, so these observations do not
  certify the final immutable dependency pin. A freshly pinned wheel and its
  complete numerical gate remain required.

Precision stress tests near negative integer gamma-function poles are also
being checked against explicitly rounded inputs. Requested or working precision
must not be interpreted as verified accuracy. No new complete Higgs-jet boundary or amplitude acceptance
is claimed by this milestone.

The default `.github/scripts/test_pyodide.mjs` gate now includes the automatic
checks. It writes the `automatic_boundary_generation` capability and numerical
evidence to the wheel-hash-bound `loop-transport-validation.json` only after
all gates succeed. `--rustred-only` retains its explicitly narrower scope.

Browser evaluation is synchronous and restricted to one worker. Progress can
be polled after a call returns; pre-cancelled controls stop a subsequent call.
A callback in the same worker cannot interrupt synchronous work already in
progress. Cache files use Pyodide's virtual filesystem and need explicit export
for persistence across browser sessions.
