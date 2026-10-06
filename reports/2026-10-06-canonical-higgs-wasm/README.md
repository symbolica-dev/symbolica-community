# Canonical Community Higgs-jet WASM validation

The complete Community wheel built from frozen source
`18571143b38097fc791bc102c23bf55b7037157c` passed actual Pyodide and a fresh,
single-core Chrome execution of the supplied-boundary `gg_hg.py` notebook.
All 296 tracked source hashes remained unchanged during this run.
The final publication subsequently changes only the citation test to exercise
an actual contraction and adds validation reports; those changes do not alter
this executable or notebook source.

The wheel uses canonical FeynKit/Linnet/Spenso/Idenso `9d1d0cb4`, Symbolica
community `58652fab`, and RustFlow `9599e358`. No local dependency override or
individual extension replacement was used. The earlier fork-owner validation
remains separate historical evidence.

| Check | Result | Wall time |
| --- | --- | ---: |
| Complete performance wheel build and packaging | Passed | 805.13 s |
| Six feature dependency graphs | Passed | 4.79 s |
| Wheel namespace and generated-stub layout | Passed | 0.05 s |
| Actual Pyodide smoke, shared graph types, transport/restart/reuse/cancellation | Passed | 24.91 s |
| Exact Standard Model kernels and archived-input amplitude component | Passed | 44.17 s |
| Fresh Chrome notebook through repeat cache hit | Passed | 345.98 s |
| Notebook's physical transport, 16 configurations, no cache hits | Passed | 288.62 s |

Chrome 153.0.8010.12 ran on CPU44 with one-core affinity and no license
environment. The notebook reached the coherent observables and literature
citations, then confirmed an exact repeated cache hit with zero ODE steps.
All 55 inspected scientific-notation table cells use 21 significant
digits. There were no browser console errors, page errors, or failed requests.
Runtime measurements include concurrent work on other cores and are not a
controlled performance comparison.

The Standard Model helper returns the shared native `Model` and leaves its
input unchanged. Its three scalar kernels exactly equal the earlier model's
kernels. Values, uncertainty estimates, arithmetic changes, and precision
match exactly for the component fixture at 322 bits. That fixture retains its
recorded 19-digit electroweak-square reference cap; this component check does
not create a higher-accuracy boundary certificate.

The full notebook uses supplied 40-digit boundary data. This validation does
not claim fresh automatic boundary generation from empty caches. The only
failed setup attempt was an HTML export with `uv` absent from the shell PATH;
its log and process result are preserved, and the successful export used the
existing `uv` executable with no source change.

Notebook SHA-256:
`4609e39028a3cbaf3069c65121abc3552a834ee52a01902f00040d4ec71489c4`.

Wheel SHA-256 (`45973377` bytes):
`04faf243f1ce1e89aa56df1249478827b22280ab430b67b3af50b9dc0f264484`.

Core module SHA-256:
`004336ee483b93b73918f1db8ecdb129b0425af5dc5cb19e1f8ac78a020b09e6`.

`report.json` records owners, timings, tool versions, precision evidence, and
process statuses. `evidence.tar.gz` contains 51 entries, including the
hash manifest, logs, harnesses, source snapshots, source-freeze hashes, table
cells, and browser screenshots. It excludes compiled wheels, local environment
files, and credentials. Its members were hash-verified after creation.

Archive SHA-256:
`ee5643e64f4177d3a8ced94cfc33a3fbc3db4b5e639964b3db2db26e4bf4598b`.
