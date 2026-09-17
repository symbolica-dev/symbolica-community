import { createEditor } from "./assets/editor.js";
import { appendText, appendHtml } from "./assets/output.js";

const examples = [
  { name: "Algebra", subtitle: "Expand & factor", description: "Expand a polynomial and recover its factors.", code: `from symbolica import E, S

expression = E("(x + y)^3 * (x - y)")
expanded = expression.expand()

print("Expanded:")
print(expanded)
print("\\nFactored:")
expanded.factor()` },
  { name: "Integration", subtitle: "See the steps", description: "Integrate a rational function, with an explanation.", code: `from symbolica import E, S

expression = E("1 / (1 + x^2)")
result, overview, steps = expression.integrate_with_steps(S("x"))

print("Integral:", result)
print("\\n" + overview)
print("\\nNumber of steps:", len(steps))` },
  { name: "Tensors", subtitle: "Explore Spenso", description: "Build a symbolic matrix and evaluate it numerically.", code: `from symbolica import E, S
from symbolica.community.spenso import Representation, Tensor, TensorName

rep = Representation.euc(2)
matrix = Tensor.dense(
    TensorName("matrix")(rep, rep),
    [E("x"), E("2"), E("3"), E("x^2")],
)
evaluator = matrix.evaluator(params=[S("x")], constants={}, funs={})
values = list(evaluator.evaluate_complex([[5.0]])[0])
print("Matrix at x = 5:")
print(values[:2])
print(values[2:])` },
  { name: "Indices", subtitle: "Simplify with Idenso", description: "Contract a metric and take its trace.", code: `from symbolica import E
from symbolica.community.idenso import simplify_metrics

metric_trace = E("g(bis(4,1), bis(4,1))", default_namespace="spenso")
print("Metric trace:", metric_trace)
print("Simplified:", simplify_metrics(metric_trace))` },
];
const $ = id => document.getElementById(id);
const output = $("output");
const code = createEditor($("editor"), { onRun: () => run(), onTypeStatus: text => { $("type-status").textContent = text; } });
let worker, config, ready = false, running = false, empty = true;
function status(text, state = "busy") {
  $("status").textContent = text;
  $("status-dot").className = `dot ${state}`;
}
function controls() {
  $("run").disabled = !ready || running;
  $("stop").hidden = !running;
}
function append(text) { if (empty) output.textContent = ""; empty = false; appendText(output, text); }
function display(html) { if (empty) output.textContent = ""; empty = false; appendHtml(output, html); }
function clear() { output.textContent = "Output will appear here."; output.classList.remove("error"); empty = true; $("elapsed").textContent = ""; }
function select(index) {
  code.value = examples[index].code;
  $("example-description").textContent = examples[index].description;
  document.querySelectorAll(".example").forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
}
examples.forEach((example, index) => {
  const button = document.createElement("button");
  button.className = "example";
  button.innerHTML = `<span class="number">0${index + 1}</span><span><strong>${example.name}</strong><small>${example.subtitle}</small></span>`;
  button.onclick = () => select(index);
  $("examples").append(button);
});
function start() {
  worker?.terminate();
  ready = false; running = false; controls(); clear();
  status("Starting Python…");
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "status") status(data.text);
    if (data.type === "ready") { ready = true; status("Python ready", ""); $("session-note").textContent = "Local runtime · Variables persist between runs"; }
    if (data.type === "output") append(data.text);
    if (data.type === "display") display(data.html);
    if (data.type === "done") { running = false; status("Python ready", ""); $("elapsed").textContent = `${(data.elapsed / 1000).toFixed(3)} s`; if (empty) append("Finished with no output.\n"); }
    if (data.type === "error") { append(data.text + "\n"); output.classList.add("error"); running = false; ready = !data.fatal; status(data.fatal ? "Runtime failed · Try reset" : "Python ready · Last run failed", data.fatal ? "error" : ""); }
    controls();
  };
  worker.onerror = event => { append(event.message + "\n"); status("Runtime failed · Try reset", "error"); ready = false; running = false; controls(); };
  worker.postMessage({ type: "init", wheel: config.wheel });
}
function run() {
  if (!ready || running || !code.value.trim()) return;
  clear(); running = true; status("Running Python…"); controls();
  worker.postMessage({ type: "run", code: code.value });
}
$("run").onclick = run;
$("clear").onclick = clear;
$("reset").onclick = () => { if (config) start(); };
$("stop").onclick = () => { worker.terminate(); ready = false; running = false; append("\nExecution stopped. Reset the session to continue.\n"); status("Stopped · Reset to continue", "error"); controls(); };
select(0);
try {
  const response = await fetch("/config.json");
  if (!response.ok) throw new Error(`Configuration request failed: ${response.status}`);
  config = await response.json();
  $("version").textContent = config.version;
  $("size").textContent = `${(config.bytes / 1e6).toFixed(2)} MB community wheel`;
  start();
} catch (error) { status("Could not load local configuration", "error"); output.textContent = String(error); }
