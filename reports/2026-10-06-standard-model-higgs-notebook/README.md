# Standard Model gg→Hg notebook validation

The complete native Community wheel and a single-core Chrome/Pyodide notebook
passed with the visible transport API, `Model.standard_model()`, final literature
citations, and scientific table formatting. The browser inspected 55 table cells
with 21-significant-digit scientific notation. Supplied 40-digit starting
boundaries were used; this is not a fresh automatic-boundary calculation.

| Execution | Notebook wall time | Physical transport |
| --- | ---: | ---: |
| Native, cold destination cache | 95.99 s | 80.52 s |
| Native, persisted destination cache | 18.03 s | 1.38 s |
| Chrome/Pyodide, cold destination cache | 335.01 s | 280.58 s |

All 4,360 transport coefficients, eight form factors, and three observables were
compared with their independent references and recorded precision. The repeated
native evaluation reused all sixteen cached destinations with identical values
and error estimates. The browser also verified an exact binary-restart hit.
The native tests passed 126 Community checks and 118 maintained FastSecDec checks;
eight exporter checks and all six dependency-graph variants passed separately.

These artifacts use RustFlow `9599e35`, Symbolica `58652fa`, and FeynKit fork
revision `4125394`. They record the tested source snapshot, rather than certifying
any subsequent dependency update. Native and browser builds were complete wheels
in private environments; no individual extension was overlaid. Timing is local
single-core evidence, not a hardware-normalized comparison with upstream tools.

`report.json` contains exact values, uncertainties, versions, and checksums.
`evidence.tar.gz` contains 46 verified members: source snapshots, runners, build
and test logs, test XML, exact transport outputs, and browser captures. It also
preserves two corrected harness failures: a target-filtered Cargo metadata check,
and a table check that initially omitted widget shadow DOM. Neither was a
numerical failure. See the separate WASM validation report for its component and
FastSecDec tests.
