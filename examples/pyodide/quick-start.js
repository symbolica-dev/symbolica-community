import { createEditor } from "./assets/editor.js";
import { appendText, appendHtml } from "./assets/output.js";
import { examples } from "./quick-start-examples.js";

const $ = id => document.getElementById(id);
const cells = [];
let worker, config, ready = false, active = null, queue = [];

function status(text, state = "busy") {
  $("status").textContent = text;
  $("status-dot").className = `dot ${state}`;
}
function controls() {
  $("run-all").disabled = !ready || !!active;
  $("stop").hidden = !active;
  for (const cell of cells) {
    cell.run.disabled = !ready || !!active;
    cell.restore.disabled = active === cell || queue.includes(cell);
  }
}
function append(cell, text) {
  appendText(cell.output, text);
}
function run(cell) {
  if (!ready || active) return;
  active = cell;
  cell.output.textContent = "";
  cell.output.classList.remove("error");
  cell.elapsed.textContent = "Running…";
  cell.section.setAttribute("aria-busy", "true");
  status(`Running: ${cell.example.title}`);
  controls();
  worker.postMessage({ type: "run", code: cell.editor.value, isolated: true });
}
function finish() {
  active?.section.setAttribute("aria-busy", "false");
  active = null;
  if (ready && queue.length) run(queue.shift());
  else { queue = []; controls(); }
}
function failure(text, fatal) {
  if (active) {
    append(active, text + "\n");
    active.output.classList.add("error");
    active.elapsed.textContent = "Failed";
  }
  ready = !fatal;
  // Stop Run all on an error so the failed example remains easy to inspect.
  queue = [];
  status(fatal ? "Runtime failed · Reset to retry" : "Python ready · Example failed", fatal ? "error" : "");
  finish();
}
function start() {
  worker?.terminate();
  if (active) {
    append(active, "\nRun cancelled. Restarting Python.\n");
    active.elapsed.textContent = "Cancelled";
  }
  ready = false; queue = []; finish();
  status("Starting Python…");
  worker = new Worker("worker.js", { type: "module" });
  worker.onmessage = ({ data }) => {
    if (data.type === "status") status(data.text);
    if (data.type === "ready") { ready = true; status("Python ready", ""); controls(); }
    if (data.type === "output" && active) append(active, data.text);
    if (data.type === "display" && active) appendHtml(active.output, data.html);
    if (data.type === "done" && active) {
      if (!active.output.hasChildNodes()) append(active, "Finished with no output.\n");
      active.elapsed.textContent = `${(data.elapsed / 1000).toFixed(3)} s`;
      status("Python ready", "");
      finish();
    }
    if (data.type === "error") failure(data.text, data.fatal);
  };
  worker.onerror = event => failure(event.message, true);
  worker.postMessage({ type: "init", wheel: config.wheel });
}

examples.forEach((example, index) => {
  const section = document.createElement("section");
  section.id = example.id;
  section.className = "live-example";
  const number = String(index + 1).padStart(2, "0");
  // These titles and labels are static page content; rich Python output is sanitized.
  section.innerHTML = `
    <div class="section-heading"><span class="section-number">${number}</span><div><p class="eyebrow">${example.label}</p><h2>${example.title}</h2></div></div>
    <p class="section-description"></p>
    <div class="panel">
      <div class="panel-head"><span><span class="file-dot"></span>${example.id.replaceAll("-", "_")}.py</span><button class="text-button restore">Restore example</button></div>
      <div class="cell-editor"></div>
      <div class="editor-foot"><span class="experiment"></span><div class="actions"><button class="primary run" disabled>▶ Run example</button></div></div>
      <div class="cell-output-head"><span>OUTPUT</span><span class="elapsed"></span></div>
      <div class="cell-output" role="log" aria-live="polite" aria-label="${example.title} output">Run this example to see the result.</div>
    </div>`;
  section.querySelector(".section-description").textContent = example.description;
  section.querySelector(".experiment").textContent = example.experiment;
  if (example.note) {
    const note = document.createElement("p");
    note.className = "browser-note";
    note.textContent = example.note;
    section.append(note);
  }
  $("live-examples").append(section);
  const cell = {
    section, example, run: section.querySelector(".run"), restore: section.querySelector(".restore"),
    output: section.querySelector(".cell-output"), elapsed: section.querySelector(".elapsed"),
  };
  cell.editor = createEditor(section.querySelector(".cell-editor"), {
    onRun: () => run(cell),
    onTypeStatus: text => { $("type-status").textContent = text; },
    height: `${Math.min(340, example.code.split("\n").length * 24 + 44)}px`,
    ariaLabel: `${example.title} Python code`,
    filename: `${example.id.replaceAll("-", "_")}.py`,
  });
  cell.editor.value = example.code;
  cell.run.onclick = () => run(cell);
  cell.restore.onclick = () => { cell.editor.value = example.code; };
  cells.push(cell);
  const link = document.createElement("a");
  link.href = `#${example.id}`;
  link.innerHTML = `<span>${number}</span>${example.title}`;
  $("contents").append(link);
});

$("run-all").onclick = () => {
  if (!ready || active) return;
  queue = [...cells];
  run(queue.shift());
};
$("stop").onclick = () => {
  worker?.terminate();
  if (active) { append(active, "\nExecution stopped. Reset the runtime to continue.\n"); active.elapsed.textContent = "Stopped"; }
  ready = false; queue = []; finish();
  status("Stopped · Reset to continue", "error");
};
$("reset").onclick = () => { if (config) start(); };
try {
  const response = await fetch("/config.json");
  if (!response.ok) throw new Error(`Configuration request failed: ${response.status}`);
  config = await response.json();
  $("version").textContent = config.version;
  $("size").textContent = `${(config.bytes / 1e6).toFixed(2)} MB wheel`;
  start();
} catch (error) { status(String(error), "error"); }
