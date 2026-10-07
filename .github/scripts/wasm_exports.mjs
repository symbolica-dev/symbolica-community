import assert from "node:assert/strict";

const allowedExports = new Set([
  "PyInit_core",
  "__wasm_call_ctors",
  "__wasm_apply_data_relocs",
]);
// Rust retains these inventory registration globals even with an explicit
// function export list. Accept Rust's legacy and v0 symbol mangling, keeping
// the crate/module paths and constructor name restricted in both formats.
const inventoryConstructors = [
  /^_ZN(?:9symbolica(?:14transcendental|5state)|19symbolica_integrate|6idenso|6spenso9shadowing|17feynkit_generator)1_6__CTOR17h[0-9a-f]{16}E$/,
  /^_RNvNv(?:Cs[0-9A-Za-z]+_(?:6idenso|19symbolica_integrate|17feynkit_generator)|NtCs[0-9A-Za-z]+_(?:6spenso9shadowing|9symbolica(?:14transcendental|5state)))1__6___CTOR$/,
  /^_RNvNv(?:Cs[0-9A-Za-z]+_13feynkit_graph|NtCs[0-9A-Za-z]+_11feynkit_cff7symbols)1__6___CTOR$/,
  /^_RNvNv(?:Nt)*Cs[0-9A-Za-z]+_11hyperbolica(?:7symbols|6python)[0-9A-Za-z_]*1__6___CTOR$/,
  /^_ZN11hyperbolica7symbols1_6__CTOR17h[0-9a-f]{16}E$/,
  // multiple-pymethods registers Python method blocks through inventory.
  // Only accept constructor globals from the known binding namespaces.
  // Rust v0 inserts an extra separator before `_rustred`'s leading underscore.
  /^_RNvNv(?:Nt)*Cs[0-9A-Za-z]+_(?:10feynkit_py|15rustred_feynkit|8__?rustred|17fastsecdec_python|7spynso3|9linnet_py|20oneloopreduce_python|16symbolica_amflow6python|8numerica7domains5float6python|9symbolica3api6python)[0-9A-Za-z_]*1__6___CTOR$/,
];

export function assertWasmExports(exports) {
  assert(exports.some(({ name }) => name === "PyInit_core"), "Missing Python module entry point");
  assert.deepEqual(
    exports.filter(({ name, kind }) =>
      !allowedExports.has(name) && !(kind === "global" && inventoryConstructors.some(pattern => pattern.test(name))),
    ),
    [],
    "Unexpected public WebAssembly exports",
  );
}
