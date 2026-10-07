# IBP build features

Default native and WASM builds retain exact arity dispatch for physical arities
1–16. Two independent, opt-in Cargo features change the build policy:

- `ibp-capacity-dispatch` compiles shared storage capacities 4, 8 and 16.
  All physical arities remain supported. Inactive storage coordinates stay
  fixed to zero and do not appear in public powers, cuts or coefficients.
  Generation, matching, routing, both walking policies, feedback, checkpoint
  restore and cold closure verification also compile only these three widths.
  Rules, source-row IDs, reports and certified artifacts retain physical
  coordinates. Private checkpoints record their storage width; readers check
  the manifest and section headers before converting zero padding. Padded
  coordinates cannot introduce denominators, shifts or extra proof scope.
- `ibp-runtime-arity-selection` permits the build-time environment variable
  `RUSTRED_RUNTIME_ARITIES` to select a finite list of supported physical arities.
  Setting that variable without this feature fails the build. A smaller list
  deliberately limits finite reducers and the saved campaign entry points;
  it is never enabled by default. Campaigns retain their existing maximum of
  16 coordinates and return a typed input error for an omitted arity.

For example, a full WASM build with shared capacities:

```sh
WASM_FEATURES=wasm,ibp-capacity-dispatch bash scripts/build_wasm_performance.sh dist/wasm-capacity
python scripts/compress_wheel_zstd.py dist/wasm-capacity/*.whl
```

For an explicitly restricted build, enable both features and set, for example,
`RUSTRED_RUNTIME_ARITIES=1,2,6,13,14,15`. The variable is read during compilation,
not at Python runtime. Query `hep.IBPFamily.compiled_runtime_arities()` for the
actual supported physical sizes and `compiled_runtime_capacities()` for storage
sizes. Capacity padding applies throughout finite reducers and routed campaigns;
public candidate formats retain exact physical arities. The feature forwards to
both RustRed's Feynkit bridge and its Python campaign bindings.
Exact generic Rust entry points and `dispatch_arity!` keep their existing
meaning. RustRed and RustFlow expose the same features without the `ibp-` prefix.

To verify the compiled campaign code, inspect the production `rustred_app`
archive with the LLVM tools from the same Emscripten SDK:

```sh
python scripts/audit_ibp_campaign_capacities.py path/to/librustred_app-HASH.rlib \
  --llvm-nm path/to/emsdk/upstream/bin/llvm-nm
```

The audit checks generic campaign symbols and cold-verifier symbols, requiring
only widths 4, 8 and 16. It reports function sizes in the archive before linking;
these are not compressed wheel sizes. Use the archive from the active build,
not a unit-test binary or an older cached build. Restricted registries can supply
`--expected-widths`; a build without capacity dispatch normally expects 1–16.

A controlled `release-performance` WASM comparison, with all other locked
sources unchanged, gives these results for RustRed `49a97604` versus `9cf14d3a`:

| Measurement | Previous dispatch | Shared campaign capacities |
| --- | ---: | ---: |
| Campaign function bytes before linking | 48,028,717 | 10,727,592 |
| Linked WASM module bytes | 219,527,383 | 209,110,642 |
| Wheel bytes, ZIP zstd level 22 | 28,170,633 | 27,274,861 |

The campaign archive shrinks by 77.7%, while the complete compressed wheel
shrinks by 3.18%. The Python bindings expose generation and certification but
not the routed walking/verification entry points; the linker already removes
those unreferenced functions from the wheel. The archive audit verifies that
these APIs no longer compile for all 16 arities, including for native CLI users.
It must not be used to attribute their entire pre-link size to the wheel.
