"""Runner-scope regressions with a fake interpreter, not WASM runtime evidence."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node is needed to check the JS runner")
@pytest.mark.parametrize("focused, fail_transport", [(False, False), (True, False), (False, True)])
def test_pyodide_scope_preserves_gates_and_receipts(tmp_path, focused, fail_transport):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "pyodide.mjs").write_text("""
import { appendFile } from 'node:fs/promises';
const values = new Map();
const files = new Map();
const wasm = Uint8Array.of(
  0,97,115,109,1,0,0,0, 1,4,1,96,0,0, 3,2,1,0,
  7,15,1,11,80,121,73,110,105,116,95,99,111,114,101,0,0,
  10,4,1,2,0,11,
);
export async function loadPyodide() {
  return {
    loadPackage: async () => {},
    globals: { set: (k, v) => values.set(k, v), get: (k) => values.get(k) },
    FS: {
      writeFile: (k, v) => files.set(k, v), mkdirTree: () => {},
      readFile: (k) => k === '/core.wasm' ? wasm : files.get(k),
    },
    runPythonAsync: async (source) => {
      await appendFile(new URL('calls.jsonl', import.meta.url), JSON.stringify(source) + '\\n');
      if (process.env.TEST_FAIL_TRANSPORT === '1' && source.includes('def check_higgs_standard_model')) {
        throw new Error('simulated independent loop-transport failure');
      }
    },
    runPython: (source) => {
      if (source.includes('rustred_cross_platform_validation')) return '{"cases":11}';
      if (source.includes('rustred_wasm_validation')) return '{"generated_and_certified_k6":true}';
      if (source.includes('symbolica.core.__file__')) return '/core.wasm';
      throw new Error('Unexpected fake-runtime call: ' + source);
    },
  };
}
""")
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel_name = "symbolica-3.0.1-cp314-abi3-pyemscripten_2026_0_wasm32.whl"
    (wheel_dir / wheel_name).write_bytes(b"synthetic archive, never installed")
    native_artifact = tmp_path / "native.rr"
    native_artifact.write_bytes(b"synthetic native artifact")
    expected = tmp_path / "expected.json"
    expected.write_text("{}")
    env = dict(os.environ)
    for key in ("SYMBOLICA_WHEEL_URL", "SYMBOLICA_ZSTD_URL"):
        env.pop(key, None)
    env.update({
        "PYODIDE_DIST_DIR": str(runtime), "SYMBOLICA_EXPECT_COMMUNITY": "1",
        "RUSTRED_NATIVE_ARTIFACT": str(native_artifact),
        "RUSTRED_NATIVE_REDUCTIONS": str(expected),
        "TEST_FAIL_TRANSPORT": str(int(fail_transport)),
    })
    command = [NODE, str(ROOT / ".github/scripts/test_pyodide.mjs"), str(wheel_dir)]
    if focused:
        command.append("--rustred-only")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=15)
    calls = [json.loads(line) for line in (runtime / "calls.jsonl").read_text().splitlines()]
    assert any('assert E("(x+1)^2")' in source for source in calls)
    assert any("alignment_reducers" in source and "alignment_members" in source for source in calls)
    assert (ROOT / "tests/check_offline_rendering.py").read_text() in calls
    assert (ROOT / ".github/scripts/check_wasm_rustred.py").read_text() in calls
    assert (ROOT / ".github/scripts/check_rustred_cross_platform.py").read_text() in calls
    transport = (ROOT / ".github/scripts/check_wasm_loop_transport.py").read_text()
    assert (transport in calls) is (not focused)
    rustred_receipt = wheel_dir / "rustred-wasm-validation.json"
    loop_receipt = wheel_dir / "loop-transport-validation.json"
    if fail_transport:
        assert result.returncode != 0
        assert "simulated independent loop-transport failure" in result.stderr
        assert not rustred_receipt.exists() and not loop_receipt.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "WebAssembly exports:" in result.stdout
        assert any("check_integration_contract()" in source for source in calls) is (not focused)
        receipt = json.loads(rustred_receipt.read_text())
        assert receipt["scope"] == ("rustred-only" if focused else "full-community")
        assert receipt["native_artifact_cross_platform"]["cases"] == 11
        assert loop_receipt.exists() is (not focused)
