"""Check the real JS export policy against Rust-v0 inventory names."""

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node is needed to check the JS export policy")
def test_inventory_names_are_allowed_only_for_known_constructor_globals():
    policy = (ROOT / ".github/scripts/wasm_exports.mjs").as_uri()
    source = f"""
import assert from 'node:assert/strict';
import {{ assertWasmExports }} from {policy!r};
const entry = {{name: 'PyInit_core', kind: 'function'}};
const names = [
  ...['s0_', 's3_', 's6_', 's9_', 'se_', 'sf_', 'si_', 'sk_', 'sn_'].map(
    item => '_RNvNvCskbLkFmFuiyt_8__rustred' + item + '1__6___CTOR'),
  ...['9streamings0_', '9streamings3_', '13normalizations0_', '10candidatess0_', '10candidatess2_'].map(
    item => '_RNvNvNtCskbLkFmFuiyt_8__rustred' + item + '1__6___CTOR'),
];
assert.equal(names.length, 14);
assertWasmExports([entry, ...names.map(name => ({{name, kind: 'global'}}))]);
// Keep the previously admitted spelling and known FeynKit binding namespace.
assertWasmExports([entry, {{name: names[0].replace('8__rustred', '8_rustred'), kind: 'global'}}]);
assertWasmExports([entry, {{name: names[0].replace('8__rustred', '15rustred_feynkit'), kind: 'global'}}]);
for (const rejected of [
  {{name: names[0], kind: 'function'}},
  {{name: names[0].replace('8__rustred', '8_unknown'), kind: 'global'}},
  {{name: names[0].replace('___CTOR', '___OTHER'), kind: 'global'}},
  {{name: names[0] + '_extra', kind: 'global'}},
  {{name: 'arbitrary_function', kind: 'function'}},
]) {{
  assert.throws(() => assertWasmExports([entry, rejected]), /Unexpected public WebAssembly exports/);
}}
assert.throws(() => assertWasmExports([]), /Missing Python module entry point/);
"""
    subprocess.run([NODE, "--input-type=module", "-e", source], check=True, capture_output=True, text=True, timeout=10)
