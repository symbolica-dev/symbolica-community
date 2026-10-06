# Native Standard Model helper in actual Pyodide

The complete community WASM wheel passes the installed transport/helper smoke,
an exact Higgs-jet model comparison, and all **89 maintained FastSecDec tests**
with no failures or skips. It contains the merged upstream sector-decomposition
features and the new native `HiggsJetAmplitude.with_form_factor_vertices`
helper and cumulative usage citations.

| Recorded operation | Seconds | Scope |
| --- | ---: | --- |
| Optimized WASM compilation | 1447 (24m07s) | Cargo duration; CPUs 60–67, four jobs |
| Installed community smoke | 22.672 | Whole Node process, CPU 60 |
| Standard Model amplitude component | 39.464 | Whole Node process, CPU 60; component code 31.808 s |
| Maintained FastSecDec tests | 14.040 | Whole Node process, CPU 60; maintained runner 5.401 s |

The smoke checks supplied transport, exact binary restart, nearby reuse,
cancellation, shared native types, typed declaration collisions, cumulative
citations, offline rendering, Hyperbolica, and integration/Symanzik fixtures.
The resulting capability marker includes `higgs_standard_model: true` and
authenticates the exact installed wheel.

The component gate starts with native `Model.standard_model()` and extends it
through the helper. All three scalar kernels are exactly equal to those built
from the legacy scientific fixture. Using the **archived eight W/Z form
factors**, all three observable values, errors, arithmetic changes and precision
metadata match the legacy construction exactly at 322 bits. Comparisons to the
archived observable references pass their combined-error bounds and 30-digit
relative agreement checks. Admitted relative-digit estimates remain **47 for
the effective square, 19 for the electroweak square, and 20 for interference**.
Working precision does not increase the archived inputs' accuracy.

This report does not claim fresh full Higgs-jet transport, new boundary
generation, Chromium UI acceptance, or a matched native/WASM timing comparison.
The separate full notebook and native-host gates retain their own evidence.

The compiled source is community `2f5d6033696c06e76c1b89faa69b5196ac64f925`:
RustFlow `9599e358`, Symbolica/Numerica/Graphica `58652fab`, HEPKit `41253945`,
and FastSecDec `9e4b5897`, with one shared native owner per package. All 263
frozen source files and 23 protected prior artifact/evidence files were
rechecked unchanged. Full revisions, wheel/extension hashes, exit records and
scope are in [report.json](report.json).

`evidence.tar.gz` contains small logs, gate harnesses, selected compiled-source
snapshots, source/protected-artifact hashes, the compact owner audit and member
checksums in `SHA256.json`. Wheel bytes, compiled binaries, large dependency
metadata and private environment files are excluded. The archive was scanned
for available credential values; no credential-free runtime claim is made.

To reproduce, check out the recorded community commit, provide the matching
wheel and Pyodide 314.0.7 distribution, and run the archived community smoke and
component harnesses on one CPU. The maintained FastSecDec wrapper requires
`FASTSECDEC_CHECKOUT` to identify the recorded exact owner revision. Process
JSON files retain the original commands; the component driver expects the
recorded community checkout as its working directory.
