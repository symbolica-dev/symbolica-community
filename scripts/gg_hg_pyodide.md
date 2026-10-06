This runner measures the supplied-boundary gg→hg demonstration in actual Node-hosted Pyodide on one CPU. It calls the notebook's cooperative controller, portable loader, native transport, form-factor projector and amplitude APIs. It does not implement the physics or substitute reference values. Node execution does not measure browser rendering, network download or Marimo startup.

The source is frozen with `git show` from community commit `9507de0edbbc9214f5d8e1ee8ff82e286024877a`. The selected wheel must have its adjacent `loop-transport-validation.json`, produced by the complete community Pyodide smoke and matching the wheel's SHA256. Both license environment variables must be unset. The runner has no native-Python or simulated runtime fallback. **Preparation and syntax checks alone are not successful Pyodide evidence.**

Use the final validated wheel and its matching Pyodide distribution:

```bash
mkdir -p /common/dev/amflow/target/gg-hg-pyodide-acceptance
# Set GG_WHEEL and GG_RUNTIME to the final wheel and matching runtime directory.
# The wheel build owner supplies these only after the actual runtime smoke.
taskset -c 43 env -u SYMBOLICA_LICENSE -u SYMBOLICA_LICENSE_KEY \
  node scripts/gg_hg_pyodide.mjs \
  --wheel "$GG_WHEEL" --runtime "$GG_RUNTIME" \
  --source /common/dev/symbolica-community/notebook-supplied-boundaries \
  --native-reference /common/dev/amflow/target/gg-hg-browser-seeds/full16-guard20-order16.json \
  --out /common/dev/amflow/target/gg-hg-pyodide-acceptance/pilot \
  --stage pilot
python scripts/gg_hg_pyodide_compare.py \
  /common/dev/amflow/target/gg-hg-pyodide-acceptance/pilot \
  /common/dev/amflow/target/gg-hg-pyodide-acceptance/pilot-exact-comparison.json
```

The pilot authenticates all sixteen supplied configurations, preserves all 4,360 complex coefficients and errors with their individual binary precisions, checks a binary save/reload, and computes one planar and one nonplanar destination at requested 20 digits, initial guard 20/order 16. These destinations are checked against independently admitted native outputs and their combined error allowances. The supplied 40-digit starting evidence is unchanged.

Only after the pilot passes, start a new process with the same runner bytes, wheel, runtime and steering source:

```bash
taskset -c 43 env -u SYMBOLICA_LICENSE -u SYMBOLICA_LICENSE_KEY \
  node scripts/gg_hg_pyodide.mjs \
  --wheel "$GG_WHEEL" --runtime "$GG_RUNTIME" \
  --source /common/dev/symbolica-community/notebook-supplied-boundaries \
  --native-reference /common/dev/amflow/target/gg-hg-browser-seeds/full16-guard20-order16.json \
  --out /common/dev/amflow/target/gg-hg-pyodide-acceptance/full \
  --stage full --resume /common/dev/amflow/target/gg-hg-pyodide-acceptance/pilot/checkpoint
python scripts/gg_hg_pyodide_compare.py \
  /common/dev/amflow/target/gg-hg-pyodide-acceptance/full \
  /common/dev/amflow/target/gg-hg-pyodide-acceptance/full-exact-comparison.json
```

The full stage checks all sixteen destinations, eight W/Z form factors and three coherent observables. Every fresh result must meet the requested 20 digits and the native cross-backend differences must lie inside combined admitted errors. The independent host checker repeats those inequalities on serialized exact rationals. Existing acceptance helpers additionally compare the archived external fixtures, retaining their documented precision limits, including the 19-digit EW reference. Source fingerprints and provenance of the old native comparison run remain distinct from the new WASM runtime; no binary compatibility override is used.

Computed Astro values retain their requested precision metadata while the owner rounds internal mantissa storage up to a machine word (32 bits in wasm32). The checker retains the full exact dyadic value and permits only that documented storage rounding; it never rounds the value to its metadata precision. The native MPFR reference uses its exact precision bound. The supplied-boundary decoder and exact 415-bit input checks are unchanged. An initial host-checker assumption that computed mantissas could not exceed requested precision was rejected by the first actual pilot and corrected against the pinned owner sources; it was not a numerical-error failure.

The full stage reloads the current runtime's binary bank and repeats all sixteen destinations, then performs another warm repeat and recomputes the amplitude with retained kernel/projector owners. Values, errors, per-number precisions, achieved digits and bank evidence must remain exact. Cache-hit wrapper source caps and added provenance are checked against the selected raw cache record; they need not equal a fresh solve's wrapper fields.

A trajectory endpoint checkpoint and the final refined boundary can share coordinates and numerical evidence while retaining different provenance and input caps. Hit validation therefore matches the complete numerical evidence, owner provenance prefix and selected source's achieved cap against a retained record. An initial full-resume checker incorrectly required numerical uniqueness before examining lineage; its failed artifact is retained. Corrected validation uses a new frozen pilot/full pair and never bypasses checkpoint identity checks.

`run.json` records the actual status, wheel/runtime/source hashes, CPU affinity, Pyodide/Node versions, phase timings, coefficient and error records, achieved digits and detailed diagnostics. Each completed configuration publishes a fully written checkpoint generation before atomically changing `checkpoint/CURRENT`. A new process checks every member's SHA256 and the full runtime/source identity before loading it. A failed run retains its failure and last complete generation; it is never reported as passed.

A resumed full run includes two cached pilot destinations and fourteen new destinations. Report cold work as the two pilot non-hit timings plus the fourteen resumed non-hit timings; do not label the full continuation as sixteen cold solves. `controller_step_ns` includes the native call and notebook cache persistence, and closes before benchmark validation/encoding and host checkpoint copies. `owner_evaluation_ns` is the owner-reported interval. Phase wall time includes those benchmark checks; host checkpoint copy time is separately recorded. Amplitude phase time includes first-use construction and projection; its owner evaluation interval is separately recorded. Setup, seed import, binary restart and warm phases remain separate. No speed comparison to the earlier native 7096 run should conflate its old constructor/cache format with the final WASM source.

The runner refuses to overwrite an output directory. Freeze the scripts before the pilot: changing them deliberately invalidates resume. No Cargo build is part of this procedure.
