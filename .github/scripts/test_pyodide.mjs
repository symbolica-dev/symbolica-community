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
pyodide.globals.set("hep_model_json", await readFile(new URL("../../examples/hep/scalar_phi3.json", import.meta.url), "utf8"));
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
from symbolica.community import hep
assert hep.FeynmanDiagram.__module__ == "symbolica.community.hep"
model = hep.Model.from_json(hep_model_json)
options = hep.GenerationOptions(max_vertices=3, allow_self_loops=False)
generated = model.generate_diagrams(["scalar_0"], ["scalar_0", "scalar_0"], loops=1, options=options)
assert generated.report.completed and len(generated) > 0
diagram = generated[0]
assert isinstance(diagram, hep.FeynmanDiagram) and diagram.loop_count == 1
diagram.validate()
restored = hep.FeynmanDiagram.from_json(model, diagram.to_json())
assert restored.loop_count == 1
cff = restored.build_cff()
assert len(cff) > 0
from symbolica import Expression
assert isinstance(cff.to_expression(), Expression)
D, mu, nu = S("hep_smoke::D", "hep_smoke::mu", "hep_smoke::nu")
k, p = S("hep_smoke::k", "hep_smoke::p")
mink, dot = S("spenso::mink", "spenso::dot")
kv, pv = k(mink(D)), p(mink(D))
numerator = k(mink(D, mu)) * k(mink(D, nu)) * p(mink(D, mu)) * p(mink(D, nu))
assert hep.TensorReducer(D).with_integrated_vector(kv).reduce(numerator) == dot(kv, kv) * dot(pv, pv) / D
assert hep.ThreeMomentum(3.0, 4.0, 0.0).on_shell().components() == (5.0, 3.0, 4.0, 0.0)
assert "symbolica.community.hep_native" in sys.modules
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
// function export list. Accept Rust's legacy and v0 symbol mangling, keeping
// the crate/module paths and constructor name restricted in both formats.
const inventoryConstructors = [
  /^_ZN(?:9symbolica(?:14transcendental|5state)|19symbolica_integrate|6idenso|6spenso9shadowing|17feynkit_generator)1_6__CTOR17h[0-9a-f]{16}E$/,
  /^_RNvNv(?:Cs[0-9A-Za-z]+_(?:6idenso|19symbolica_integrate|17feynkit_generator)|NtCs[0-9A-Za-z]+_(?:6spenso9shadowing|9symbolica(?:14transcendental|5state)))1__6___CTOR$/,
];
assert(exports.some(({ name }) => name === "PyInit_core"), "Missing Python module entry point");
assert.deepEqual(
  exports.filter(({ name, kind }) =>
    !allowedExports.has(name) && !(kind === "global" && inventoryConstructors.some(pattern => pattern.test(name))),
  ),
  [],
  "Unexpected public WebAssembly exports",
);
console.log(`WebAssembly exports (${exports.length}): ${exports.map(({ name }) => name).join(", ")}`);
console.log(`Wheel: ${wheelBytes.length} bytes; WebAssembly module: ${wasmBytes.length} bytes.`);
console.log("PyEmscripten wheel installed with micropip; smoke test passed.");
