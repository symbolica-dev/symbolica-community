import assert from "node:assert/strict";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [wheelDir, output] = process.argv.slice(2);
const runtimeDir = process.env.PYODIDE_DIST_DIR;
assert(wheelDir && output && runtimeDir, "Pass wheel directory, output JSON, and set PYODIDE_DIST_DIR");
const wheels = (await readdir(wheelDir)).filter(name => name.endsWith("-pyemscripten_2026_0_wasm32.whl"));
assert.equal(wheels.length, 1);
const { loadPyodide } = await import(pathToFileURL(join(runtimeDir, "pyodide.mjs")));
const pyodide = await loadPyodide({ indexURL: runtimeDir, env: { SYMBOLICA_LICENSE_KEY: process.env.SYMBOLICA_LICENSE_KEY || "" } });
await pyodide.loadPackage("micropip");
const wheelBytes = await readFile(join(wheelDir, wheels[0]));
pyodide.FS.writeFile(`/${wheels[0]}`, wheelBytes);
pyodide.globals.set("wheel_uri", `emfs:/${wheels[0]}`);
const start = performance.now();
await pyodide.runPythonAsync(`
import micropip, os
await micropip.install(wheel_uri)
from symbolica import E, S, set_license_key
if os.environ.get("SYMBOLICA_LICENSE_KEY"):
    set_license_key(os.environ["SYMBOLICA_LICENSE_KEY"])
from symbolica.community import tensor, hep
`);
const installImportMs = performance.now() - start;
pyodide.globals.set("hep_model_json", await readFile(new URL("../../examples/hep/scalar_phi3.json", import.meta.url), "utf8"));
await pyodide.runPythonAsync(await readFile(new URL("benchmark_pyodide.py", import.meta.url), "utf8"));
const result = JSON.parse(pyodide.globals.get("benchmark_result"));
Object.assign(result, { wheel: wheels[0], wheel_bytes: wheelBytes.length, pyodide: pyodide.version, node: process.version, node_flags: process.execArgv, v8: process.versions.v8, install_import_ms: installImportMs, network_time_included: false });
await writeFile(output, JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify(result));
