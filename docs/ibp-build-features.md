# IBP build features

Default native and WASM builds retain exact arity dispatch for physical arities
1–16. Two independent, opt-in Cargo features change the build policy:

- `ibp-capacity-dispatch` compiles shared storage capacities 4, 8 and 16.
  All physical arities remain supported. Inactive storage coordinates stay
  fixed to zero and do not appear in public powers, cuts or coefficients.
  Campaign generation and feedback searches also share these capacities;
  rules, source-row IDs, progress events and errors are converted back to
  physical coordinates before persistence or certification. Campaign formats
  and checkpoints keep their original arities.
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
sizes. Capacity padding applies inside finite reducers and campaign search
queues; saved campaign formats retain exact physical arities.
Exact generic Rust entry points and `dispatch_arity!` keep their existing
meaning. RustRed and RustFlow expose the same features without the `ibp-` prefix.
