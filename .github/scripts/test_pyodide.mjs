import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const wheelDir = process.argv[2];
const runtimeDir = process.env.PYODIDE_DIST_DIR;
assert(wheelDir && runtimeDir, "Pass the wheel directory and set PYODIDE_DIST_DIR");
const wheels = (await readdir(wheelDir)).filter((name) =>
  name.endsWith("-pyemscripten_2026_0_wasm32.whl"),
);
assert.equal(wheels.length, 1, "Expected exactly one PyEmscripten wheel");

const { loadPyodide } = await import(pathToFileURL(join(runtimeDir, "pyodide.mjs")));
const pyodide = await loadPyodide({
  indexURL: runtimeDir,
  env: { SYMBOLICA_LICENSE_KEY: process.env.SYMBOLICA_LICENSE_KEY || "" },
});
await pyodide.loadPackage("micropip");
const wheel = wheels[0];
const wheelBytes = await readFile(join(wheelDir, wheel));
pyodide.FS.writeFile(`/${wheel}`, wheelBytes);
pyodide.globals.set("wheel_uri", `emfs:/${wheel}`);
pyodide.globals.set("expected_version", wheel.split("-")[1]);
await pyodide.runPythonAsync(`
import os
import sys
from importlib.metadata import version
import micropip
await micropip.install(wheel_uri)

from symbolica import E, S, set_license_key
if os.environ.get("SYMBOLICA_LICENSE_KEY"):
    set_license_key(os.environ["SYMBOLICA_LICENSE_KEY"])
assert E("(x+1)^2").expand() == E("x^2+2*x+1")
assert E(str(2**256)) + E("1/3") + E("2/3") == E(str(2**256 + 1))
assert E("sin(0)") == E("0")
assert E("x^2").integrate(S("x")) == E("x^3/3")
expression = E("1/(1+x^2)")
result, overview, steps = expression.integrate_with_steps(S("x"))
assert result == expression.integrate(S("x")) == E("atan(x)")
assert overview.strip()
assert any(step.rule is not None and step.source and step.description for step in steps)
assert version("symbolica") == expected_version
from symbolica.community.example_extension import add_two
from symbolica.community.idenso import simplify_metrics
assert add_two(E("x")) == E("x+2")
assert simplify_metrics(E("g(bis(4,1),bis(4,1))", default_namespace="spenso")) == E("4")
import symbolica.community.spenso as spenso
from symbolica.community.spenso import Representation, Tensor, TensorLibrary, TensorName, TensorNetwork
rep = Representation.euc(2)
tensor = Tensor.dense(TensorName("wasm_matrix")(rep, rep), [E("x"), E("2"), E("3"), E("4")])
evaluator = tensor.evaluator(params=[S("x")], constants={}, funs={})
assert list(evaluator.evaluate_complex([[5.0]])[0]) == [5.0, 2.0, 3.0, 4.0]
assert list(evaluator.evaluate_complex([[1.0 + 2.0j]])[0]) == [1.0 + 2.0j, 2.0, 3.0, 4.0]
library = TensorLibrary.hep_lib()
library.register(tensor)
network = TensorNetwork(tensor.structure()(1, 1), library=library)
network.execute(library=library)
assert list(network.result_tensor(library=library)) == [E("x+4")]
assert "symbolica.community.spenso_native" in sys.modules
assert not hasattr(evaluator, "compile")
assert not hasattr(spenso, "CompiledTensorEvaluator")
assert "symbolica.community.vakint_native" not in sys.modules
try:
    import symbolica.community.vakint
except ImportError as error:
    assert "native Symbolica installation" in str(error)
else:
    raise AssertionError("vakint should require a native installation")
`);
const modulePath = pyodide.runPython("import symbolica.core; symbolica.core.__file__");
const wasmBytes = pyodide.FS.readFile(modulePath);
const wasm = await WebAssembly.compile(wasmBytes);
const exports = WebAssembly.Module.exports(wasm);
const allowedExports = new Set([
  "PyInit_core",
  "__wasm_call_ctors",
  "__wasm_apply_data_relocs",
]);
// Rust retains these inventory registration globals even with an explicit
// function export list. The hash varies between builds; their identity does not.
const inventoryConstructor = /^_ZN(?:9symbolica(?:14transcendental|5state)|19symbolica_integrate|6idenso|6spenso9shadowing)1_6__CTOR17h[0-9a-f]{16}E$/;
assert(exports.some(({ name }) => name === "PyInit_core"), "Missing Python module entry point");
assert.deepEqual(
  exports.filter(({ name, kind }) =>
    !allowedExports.has(name) && !(kind === "global" && inventoryConstructor.test(name)),
  ),
  [],
  "Unexpected public WebAssembly exports",
);
console.log(`WebAssembly exports (${exports.length}): ${exports.map(({ name }) => name).join(", ")}`);
console.log(`Wheel: ${wheelBytes.length} bytes; WebAssembly module: ${wasmBytes.length} bytes.`);
console.log("PyEmscripten wheel installed with micropip; smoke test passed.");
