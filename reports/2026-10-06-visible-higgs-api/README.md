# Visible Higgs-jet API validation

The notebook shows native calls for loading boundaries, physical transport,
form-factor projection, coherent amplitude evaluation and binary restart. It
has no stage-button dashboard or hidden numerical controller. The model and
prepared kernels do not depend on editable kinematics. Tables render after
amplitude evaluation so their browser requests cannot overlap that long call.

| Actual execution | Cold transport | Warm transport | Scope |
| --- | ---: | ---: | --- |
| Native Python, one CPU | 76.070 s | 1.354 s | Unmodified notebook cells; all 4,360 coefficients, eight form factors, three observables, binary repeat and exact numerical reuse |
| Chromium WASM, one CPU | 280.67 s | Not timed in this gate | Visible code, all numerical stages, populated observable table, binary repeat and no console/page/network errors |

Native whole-notebook execution took 115.341 s cold and 43.207 s warm. Each
`app.run()` rebuilds the model and amplitude; this is not the cost of changing
only kinematics in an already open notebook. Warm transport has sixteen exact
hits, zero ODE steps and zero inserted points. Cold transport adds 32 points to
the sixteen supplied seeds. Exact coefficient/error values, individual
precisions, achieved digits, identities and root sheets agree across cold/warm
runs. The selected source's accuracy cap correctly changes from 40 for supplied
seeds to 28 for cached destinations. Observable estimates are 35/36/47 digits;
the external references retain their separate 19/20/39-digit limits.

Chromium reached the completed page and binary-repeat assertion in 339.705 s,
including setup and all numerical work. The final check also waits for the
observable table to contain actual data and monitors the console for another
five seconds. The API code is present in the rendered page, and the old
transport button is absent. This run uses no license credentials. Browser files
remain local to that page's virtual filesystem.

The native test uses the previously built `7096ba8` shared-host development
extension. The browser uses the validated `d81dae9` WASM wheel with the staged
amplitude-contraction optimization. Their constructor costs and end-to-end
times are not a matched backend comparison. The separate
[Node Pyodide scientific gate](../2026-10-06-pyodide-gg-hg/README.md) checks all
4,360 WASM coefficients and coherent outputs against native results with exact
rational arithmetic; this DOM test does not replace that scientific check.

The archive records both source hashes. Native execution precedes the final
presentation-only table dependency; the exact diff is included. Lightweight
checks cover the real Marimo dependency graph, cold/warm source selection,
displayed reference compatibility after changing masses or couplings, bundle
validation and export. Marimo lint and whitespace checks pass.

Retained failures are evidence, not passing gates: an initial native harness
incorrectly equated cold/warm source caps (corrected in a fresh warm run), and
the first code-first browser run finished numerically but failed because table
requests overlapped amplitude preparation. The final browser gate requires no
such errors. Original reports, logs, source snapshots and exact native outputs
are in `evidence.tar.gz`; member hashes and lengths are in `SHA256.json`.
Archive identity and scoped results are recorded in [report.json](report.json).

No fresh boundary generation is claimed by this live demonstration. Native
empty-cache generation, forced recomputation and cancellation remain in the
separate long acceptance runner.
