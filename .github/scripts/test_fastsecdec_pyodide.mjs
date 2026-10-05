// Run the same installed scientific bridge gates in the actual Pyodide runtime.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const wheelDir = process.argv[2];
const runtimeDir = process.env.PYODIDE_DIST_DIR;
assert(wheelDir && runtimeDir, "Pass a wheel directory and set PYODIDE_DIST_DIR");
const wheels = (await readdir(wheelDir)).filter(name => name.endsWith("-pyemscripten_2026_0_wasm32.whl"));
assert.equal(wheels.length, 1, "Expected exactly one PyEmscripten wheel");
const bytes = await readFile(join(wheelDir, wheels[0]));
const root = fileURLToPath(new URL("../../", import.meta.url));
const { loadPyodide } = await import(pathToFileURL(join(runtimeDir, "pyodide.mjs")));
const pyodide = await loadPyodide({
  indexURL: runtimeDir,
  env: { SYMBOLICA_LICENSE_KEY: process.env.SYMBOLICA_LICENSE_KEY || "" },
});
await pyodide.loadPackage("micropip");
pyodide.FS.writeFile(`/${wheels[0]}`, bytes);
pyodide.globals.set("bridge_wheel_uri", `emfs:/${wheels[0]}`);
const tests = ["test_hep_fastsecdec.py", "test_hep_fastsecdec_inputs.py", "test_hep_wavefunctions.py"];
const fixtures = (await readdir(join(root, "examples/hep/fixtures/fastsecdec"))).sort();
const paths = [
  ...tests.map(name => `tests/${name}`),
  "examples/hep/fastsecdec_inputs.py",
  ...fixtures.map(name => `examples/hep/fixtures/fastsecdec/${name}`),
];
for (const path of paths) {
  const destination = `/bridge/${path}`;
  pyodide.FS.mkdirTree(dirname(destination));
  pyodide.FS.writeFile(destination, await readFile(join(root, path)));
}
pyodide.globals.set("bridge_test_paths", tests.map(name => `/bridge/tests/${name}`));
const result = await pyodide.runPythonAsync(`
import json
import os
import sys
import time
import micropip
await micropip.install([bridge_wheel_uri, "pytest"])
from symbolica import set_license_key
if os.environ.get("SYMBOLICA_LICENSE_KEY"):
    set_license_key(os.environ["SYMBOLICA_LICENSE_KEY"])
assert sys.platform == "emscripten"
import pytest
started = time.perf_counter()
class Report:
    passed = failed = skipped = 0
    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.passed += report.passed
            self.failed += report.failed
            self.skipped += report.skipped
        elif report.failed:
            self.failed += 1
report = Report()
code = pytest.main(["-q", "-p", "no:cacheprovider", *bridge_test_paths.to_py()], plugins=[report])
json.dumps({"exit_code": int(code), "passed": report.passed,
            "failed": report.failed, "skipped": report.skipped,
            "seconds": time.perf_counter() - started,
            "python": sys.version, "platform": sys.platform})
`);
const report = JSON.parse(result);
report.wheel = wheels[0];
report.wheel_sha256 = createHash("sha256").update(bytes).digest("hex");
console.log(JSON.stringify(report, null, 2));
assert.equal(report.exit_code, 0, "Installed Pyodide bridge gates failed");
assert.equal(report.passed, 39, "Expected all scientific bridge, input and shared wavefunction controls");
assert.equal(report.skipped, 0);
