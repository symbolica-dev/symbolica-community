import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const wheelDir = process.argv[2];
const runtimeDir = process.env.PYODIDE_DIST_DIR;
const expectCommunity = process.env.SYMBOLICA_EXPECT_COMMUNITY !== "0";
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
pyodide.globals.set("wheel_uri", process.env.SYMBOLICA_WHEEL_URL || `emfs:/${wheel}`);
assert(!(process.env.SYMBOLICA_WHEEL_URL && process.env.SYMBOLICA_ZSTD_URL), "Choose one HTTP delivery mode");
pyodide.globals.set("wheel_zstd_url", process.env.SYMBOLICA_ZSTD_URL || "");
pyodide.globals.set("expected_version", wheel.split("-")[1]);
pyodide.globals.set("hep_model_json", await readFile(new URL("../../examples/hep/scalar_phi3.json", import.meta.url), "utf8"));
await pyodide.runPythonAsync(`
import os
import sys
from importlib.metadata import version
import micropip
if wheel_zstd_url:
    from compression import zstd
    from pathlib import Path
    from pyodide.http import pyfetch
    response = await pyfetch(wheel_zstd_url)
    assert response.status == 200
    compressed = await response.bytes()
    Path(wheel_uri.removeprefix("emfs:")).write_bytes(zstd.decompress(
        compressed, options={zstd.DecompressionParameter.window_log_max: 27},
    ))
    print(f"Downloaded zstd archive: {len(compressed)} bytes")
    del compressed
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
from symbolica import get_citations
assert any(citation.id == "https://github.com/symbolica-dev/symbolica-integrate" for citation in get_citations())
`);
if (expectCommunity) {
await pyodide.runPythonAsync(`
import importlib.util
assert importlib.util.find_spec("numpy") is None
`);
await pyodide.runPythonAsync(
  await readFile(new URL("../../tests/check_offline_rendering.py", import.meta.url), "utf8"),
);
await pyodide.runPythonAsync(`
assert importlib.util.find_spec("numpy") is None
# Array conversion and evaluator tests opt into the NumPy extra separately.
await micropip.install("numpy")
import numpy
`);
await pyodide.runPythonAsync(`
import symbolica.community.tensor as tensor_module
from symbolica.community.tensor import Representation, Tensor, TensorExpression, TensorLibrary, TensorName, TensorNetwork, dot
metric = TensorExpression(E("g(bis(4,1),bis(4,1))", default_namespace="spenso"))
assert metric.simplify_algebra(contract="dots").to_expression() == E("4")
rep = Representation.euc(2)
tensor = Tensor.dense(TensorName("wasm_matrix")(rep, rep), [E("x"), E("2"), E("3"), E("4")])
evaluator = tensor.evaluator(params=[S("x")], jit_compile=False)
assert list(evaluator.evaluate_complex([[5.0]])[0]) == [5.0, 2.0, 3.0, 4.0]
assert list(evaluator.evaluate_complex([[1.0 + 2.0j]])[0]) == [1.0 + 2.0j, 2.0, 3.0, 4.0]
library = TensorLibrary.hep_lib()
library.register(tensor)
network = TensorNetwork(tensor.expression()(1, 1), library=library)
network.execute(library=library)
assert list(network.result_tensor(library=library)) == [E("x+4")]
assert "symbolica.community.tensor_native" in sys.modules
# Large outputs must render and navigate without launching a Typst process.
preview_value = TensorExpression(E("+".join(f"{i + 1}*wasm_paging::x^{i}" for i in range(301))))
preview_original = preview_value.to_expression()
preview = preview_value.paged()
widget = preview._get_widget()
preview._on_message(widget, {"action": "attach", "view": "wasm-smoke"}, [])
assert widget.page["connected"] == "wasm-smoke"
first_html = widget.page["html"]
assert "<math" in first_html and "Page rendering failed" not in first_html
assert len(first_html.encode()) <= 256 * 1024
preview._on_message(widget, {"action": "next", "request": "next"}, [])
assert widget.page["start"] == 25 and widget.page["request"] == "next"
assert "<math" in widget.page["html"]
preview._on_message(widget, {"action": "previous"}, [])
assert widget.page["start"] == 0 and widget.page["html"] == first_html
preview._on_message(widget, {"action": "size", "value": 100}, [])
assert widget.page["page_size"] == 100 and widget.page["end"] == 100
assert preview_value.to_expression() == preview_original
preview.close()
from symbolica.community import hepkit as hep
assert hep.FeynmanDiagram.__module__ == "symbolica.community.hepkit"
model = hep.Model.from_json(hep_model_json)
process = model.process(["scalar_0"], ["scalar_0", "scalar_0"])
generated = process.generate_diagrams(loops=1, max_vertices=3, allow_self_loops=False)
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
k, p = TensorName.vector("hep_smoke::k"), TensorName.vector("hep_smoke::p")
space = Representation.mink(D)
kv, pv = k(space), p(space)
numerator = (k(space(mu)) * k(space(nu)) * p(space(mu)) * p(space(nu))).to_expression()
reduced = hep.TensorReducer(D, integrated=[kv.to_expression()]).reduce(numerator)
expected = (dot(kv, kv) * dot(pv, pv) / D).to_expression()
assert (reduced - expected).expand() == E("0")
assert hep.ThreeMomentum(3.0, 4.0, 0.0).on_shell().components() == (5.0, 3.0, 4.0, 0.0)
from symbolica.community.hepkit import oneloop
d, ell, mass = S("oneloop_smoke::d", "oneloop_smoke::ell", "oneloop_smoke::m2")
kinematics = hep.Kinematics(d, momenta=[ell])
family = hep.IntegralFamily(
    [ell], [], [kinematics.scalar_product(ell, ell) - mass], kinematics=kinematics,
)
reduction = oneloop.reduce(family, [1])
assert reduction.to_expression() == E("oneloopmaster::A0")(mass, E("1"))
assert not hasattr(oneloop, "Evaluator")
assert not hasattr(oneloop, "compile_native")
assert "symbolica.community.hepkit_native" in sys.modules
assert not hasattr(evaluator, "compile")
assert not hasattr(tensor_module, "CompiledTensorEvaluator")
assert "symbolica.community.hepkit_vakint_native" not in sys.modules
try:
    import symbolica.community.vakint
except ImportError as error:
    assert "native Symbolica installation" in str(error)
else:
    raise AssertionError("vakint should require a native installation")
`);
} else {
await pyodide.runPythonAsync(`
import importlib.util
import zipfile
assert importlib.util.find_spec("symbolica.community") is None
assert not any(name.startswith("symbolica.community") for name in sys.modules)
with zipfile.ZipFile(wheel_uri.removeprefix("emfs:")) as archive:
    assert not any(name.startswith("symbolica/community/") for name in archive.namelist())
print("Core-only wheel: algebra and integration passed; community packages absent.")
`);
}
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
  /^_RNvNv(?:Cs[0-9A-Za-z]+_13feynkit_graph|NtCs[0-9A-Za-z]+_11feynkit_cff7symbols)1__6___CTOR$/,
  // multiple-pymethods registers Python method blocks through inventory.
  // Only accept constructor globals from the known binding namespaces.
  /^_RNvNv(?:Nt)*Cs[0-9A-Za-z]+_(?:10feynkit_py|7spynso3|9linnet_py|20oneloopreduce_python|8numerica7domains5float6python|9symbolica3api6python)[0-9A-Za-z_]*1__6___CTOR$/,
];
assert(exports.some(({ name }) => name === "PyInit_core"), "Missing Python module entry point");
assert.deepEqual(
  exports.filter(({ name, kind }) =>
    !allowedExports.has(name) && !(kind === "global" && inventoryConstructors.some(pattern => pattern.test(name))),
  ),
  [],
  "Unexpected public WebAssembly exports",
);
console.log(`WebAssembly exports: ${exports.length}; functions: ${exports.filter(({ kind }) => kind === "function").map(({ name }) => name).join(", ")}`);
console.log(`Wheel: ${wheelBytes.length} bytes; WebAssembly module: ${wasmBytes.length} bytes.`);
console.log("PyEmscripten wheel installed with micropip; smoke test passed.");
