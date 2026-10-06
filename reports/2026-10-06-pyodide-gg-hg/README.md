# Actual Pyodide Higgs-jet scientific acceptance

The supplied-boundary gg→hg calculation passes in **actual Node-hosted Pyodide on one CPU, without license credentials**. The gate checks all 4,360 final complex master coefficients, eight W/Z form factors and three observables against independently admitted native results. Every difference is inside the combined admitted errors, and both backends' errors and differences meet the requested 20-digit tolerances. A separate host checker repeats those inequalities with exact integers and rational numbers.

| Measured operation | Seconds |
| --- | ---: |
| Import all sixteen supplied starting configurations | 2.077 |
| Sixteen unique initial transport calls, including notebook cache persistence | **275.747** |
| Native owner evaluation within those calls | 267.079 |
| First amplitude stage, including kernel/projector construction and projection | **24.113** |
| Warm transport calls, all sixteen exact hits | **1.477** |
| Warm amplitude stage, reusing kernel/projector | **7.776** |

The sixteen initial calls comprise a two-case pilot and fourteen new calculations in a fresh process; the continuation also hits the two pilot destinations. This is not an uninterrupted-session timing. Each process uses logical CPU 43 on the shared EPYC 9754 host, with physical options requested digits 20, initial guard 20, initial order 16 and one worker. Other work continued on the host. Interpreter/runtime setup, model construction and benchmark validation are separate. No native-to-WASM speed ratio is inferred from these observations.

The masters retain 28 verified digits at 216 requested working bits, with original input cap 40. The EW square, interference and HEFT square retain conditional relative-accuracy estimates of **35, 36 and 47 digits**. The archived external EW comparison still has its separate **19-digit reference limit**. Working precision is not interpreted as established physical accuracy.

All sixteen supplied configurations preserve every value, error and individual precision exactly: 4,360 complex coefficients, 40-digit caps and 415-bit input records. The current runtime's binary bank is saved and reloaded through the native owner. A second Pyodide process authenticates the same wheel/runtime/source identity, imports that bank and reproduces the pilot's 545 computed coefficients and errors exactly, including reported precision and achieved evidence. After completing all sixteen configurations, another binary reload and warm repeat preserve the whole 48-entry bank and all numerical evidence. Warm form factors and observables also agree exactly with their first evaluation.

This is **scientific/runtime acceptance, not browser UI acceptance**. Node Pyodide 314.0.7 does not exercise Marimo RPC scheduling, rendering, network delivery or Chromium's runtime. In particular, synchronous physical calls lasting more than twenty seconds can affect a browser controller even when these numerical checks pass. The actual Marimo export and UI are checked separately.

## Frozen identities

- Wheel: `symbolica-3.0.0-cp314-abi3-pyemscripten_2026_0_wasm32.whl`, SHA256 `fcc2367342481ebfac79ceb8b14f87e9a9afb1bcb326c54106340825576bfac2`.
- The installed `symbolica/core.abi3.so` matches that exact archive member: 174,186,956 bytes, SHA256 `72f8fa41626030c8405a27ee282c885319438dc92d0d65901c3671fffa9b716c`.
- Notebook/controller/loader: community `9507de0edbbc9214f5d8e1ee8ff82e286024877a`, extracted with `git show`, independently of a mutable checkout.
- Runtime driver: community `63c6c2e3`; final host checker/harness: `0cf7de154b7f605c6155af6995119411a0f98021`. Exact file hashes are retained for every invocation.
- RustFlow: `d81dae94a40e41e5d2dcd978f623e6e83abb3e4f`; Symbolica/Numerica: `6defcca968ca8411977fb1f641a9dee49ee7b7a7`; HEPKit: `b96600b0085d9ddfa9e6acbc11fa72ec6163253c`.
- Host compiled Cargo/Rust and packaged integration sources match community `acc0eb576b0196b86faa8eb2cde2a341d9addceb`. The wheel was built from the earlier base plus the pending host patch; the archived build-input manifest and patch give the exact attribution. Later docs/smoke additions did not change the wheel.
- Node 24.18.0, Pyodide 314.0.7; exact runtime JS/WASM/stdlib/lock hashes are in `report.json`. Both license environment variables were unset. The wheel's prior credential-free smoke marker was required and matched before installation.

Only supplied starting values enter the evaluator. Native equations, path selection, uncertainty propagation, exact basis maps, form-factor projection and amplitude construction remain in their existing owners. Comparison files are read by the acceptance layer after an actual result exists and are never passed as destination values or form factors to the evaluator. Source fingerprints and original boundary provenance remain distinct; no binary compatibility override is used.

## Timing boundaries and retained failures

Per-configuration controller clocks include the native call and the notebook's bank persistence, but stop before benchmark checking, exact-number serialization and the host checkpoint bridge. These checks are substantial: the resumed stage's wall time is 279.252 seconds, and the repeated-bank and warm stages' instrumented wall times are 52.236 and 48.442 seconds. Their timed transport calls total only 1.354 and 1.477 seconds, respectively. The 1.354-second figure excludes the preceding explicit binary save/load and must not be quoted as a complete restart time. Full phase timings, setup costs and host bridge times are recorded separately in `report.json`.

The first pilot and first full attempt remain in the archive because they exposed two harness assumptions:

1. Astro preserves requested precision metadata while rounding mantissa storage to machine words. Computed wasm32 values at requested precision 216 can have 224 significant bits. The first host checker incorrectly required at most 216. It now compares the complete exact dyadic value and permits only the documented 32-bit storage rounding, while retaining the strict supplied-input decoder and native MPFR bound. All scientific inequalities are unchanged.
2. A trajectory endpoint checkpoint and final refined boundary can coexist with identical numerical evidence but different lineage and input caps. The first resume had already passed exact raw-bank reload, then failed a numerical-uniqueness assertion. Hit validation now matches the complete numerical evidence, owner provenance prefix and selected source's achieved cap against a retained record.

The corrected runtime driver uses a new pilot/full pair. Earlier artifacts and failure logs are unchanged; no checkpoint identity was bypassed. Seven host comparison tests cover exact arithmetic, insufficient error bounds, sub-binary64 differences, relative-zero rejection and the backend storage distinction.

## Reproduce

The runnable driver and command details are in [`scripts/gg_hg_pyodide.md`](../../scripts/gg_hg_pyodide.md). The actual scientific run requires a matching transport-enabled wheel, its successful smoke marker and Pyodide distribution. It has no native-Python or simulated-runtime fallback.

To repeat the independent comparison using only the archived exact outputs:

```bash
mkdir extracted
cd extracted
tar -xzf ../evidence.tar.gz
python scripts/gg_hg_pyodide_compare.py pilot-v2 pilot-rechecked.json
python scripts/gg_hg_pyodide_compare.py full-v2 full-rechecked.json \
  --previous-checkpoint pilot-v2/checkpoint
```

`evidence.tar.gz` contains the latest complete checkpoint generation for each of the four retained attempts, steering sources, numerical outputs, exact checkers, initial failures/corrections, compiled-input provenance and `SHA256.json`. Intermediate checkpoint generations remain in the raw evidence. Wheel and Pyodide distribution bytes are identified by hashes and excluded from this source report. Archive SHA256: `d5b4ebeefc33322e1dcb044170e2f378b65b380380b80693197589278b0ba284` (37,451,174 bytes).
