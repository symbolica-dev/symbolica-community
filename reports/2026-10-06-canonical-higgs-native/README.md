# Final canonical-owner native validation

The complete Community wheel built from the final merge with canonical FeynKit
`9d1d0cb`, Symbolica `58652fa`, and RustFlow `9599e35` passed 156 focused Python
tests, all six dependency-graph checks, wheel/stub checks, and offline rendering.
The shared graph module and native types from community main are preserved.
The inherited tensor citation test now checks Spenso on construction and Idenso
after contraction, retaining the original citation assertions.

The unmodified gg→Hg notebook passed from supplied 40-digit starting boundaries:
all 4,360 transport coefficients, eight form factors, and three observables agree
with their references at the recorded accuracy. Every repeated destination is an
exact cache hit, with unchanged values and error estimates.

| Native run, one CPU | Notebook wall time | Physical transport | Cache hits |
| --- | ---: | ---: | ---: |
| Cold destination cache | 97.25 s | 81.90 s | 0/16 |
| Reloaded destination cache | 19.27 s | 1.73 s | 16/16 |

This validates the complete installed final native wheel; it does not reuse an
extension from the preceding fork-owner validation. The earlier reports remain
historical evidence with their original dependency versions. The final browser
package has a separate validation report.

`report.json` records exact numerical comparisons, accuracy metadata, source
hashes, and package provenance. `evidence.tar.gz` contains 31 verified members,
including complete test/build logs, exact cold/warm results, source snapshots,
and the independent merge audit. The cold run uses supplied boundaries, not a
fresh automatic-boundary calculation. Timings are local measurements; other
compilation ran on separate CPUs.
