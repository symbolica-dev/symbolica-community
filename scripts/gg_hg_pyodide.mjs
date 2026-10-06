// Actual Pyodide acceptance. No native-Python or simulated-runtime fallback.
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { availableParallelism } from "node:os";
import { readFileSync, writeFileSync, mkdirSync, renameSync } from "node:fs";
import { readFile, mkdir } from "node:fs/promises";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";

const arguments_ = process.argv.slice(2);
if (arguments_.includes("--help")) {
  console.log("taskset -c 43 env -u SYMBOLICA_LICENSE -u SYMBOLICA_LICENSE_KEY node scripts/gg_hg_pyodide.mjs --wheel WHEEL --runtime PYODIDE_DIST --source NOTEBOOK_CHECKOUT --native-reference FULL16_JSON --out NEW_DIRECTORY [--source-commit 1e1c2366] [--stage pilot|full] [--resume PREVIOUS_CHECKPOINT]");
  process.exit(0);
}
const options = {};
for (let i = 0; i < arguments_.length; i += 2) {
  assert(arguments_[i].startsWith("--") && arguments_[i + 1], "Expected --option value");
  const name = arguments_[i].slice(2);
  assert(["wheel", "runtime", "source", "source-commit", "native-reference", "out", "stage", "resume"].includes(name), `Unknown option ${name}`);
  assert(!(name in options), `Duplicate option ${name}`);
  options[name] = arguments_[i + 1];
}
for (const name of ["wheel", "runtime", "source", "native-reference", "out"]) {
  assert(options[name], `Missing --${name}`);
  options[name] = resolve(options[name]);
}
options.stage ??= "pilot";
options["source-commit"] ??= "1e1c2366";
assert(/^[0-9a-f]{7,40}$/.test(options["source-commit"]), "Use an immutable source commit");
const sourceCommit = execFileSync("git", ["-C", options.source, "rev-parse", `${options["source-commit"]}^{commit}`], { encoding: "utf8" }).trim();
assert(["pilot", "full"].includes(options.stage), "Stage must be pilot or full");
assert(!options.resume || options.stage === "full", "Resume continues to the full gate");
assert(!process.env.SYMBOLICA_LICENSE && !process.env.SYMBOLICA_LICENSE_KEY,
  "Unset both license environment variables; this acceptance models an unconfigured page");
assert.equal(availableParallelism(), 1, "Pin this process to one CPU with taskset");
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
const json = (path) => JSON.parse(readFileSync(path, "utf8"));
const wheelBytes = await readFile(options.wheel);
const validation = json(join(dirname(options.wheel), "loop-transport-validation.json"));
assert.equal(validation.schema, "supplied-loop-transport-runtime-v1");
assert.equal(validation.supplied_loop_transport, true);
assert.equal(validation.wheel, basename(options.wheel));
assert.equal(validation.wheel_sha256, hash(wheelBytes), "Wheel differs from the actual-Pyodide smoke proof");
const driver = fileURLToPath(new URL("./gg_hg_pyodide.py", import.meta.url));
const sourceFiles = [
  "gg_hg.py", "gg_hg_support.py", "gg_hg_boundaries.py", "gg_hg_acceptance.py",
  "gg_hg_anchor_acceptance.py", "data/gg_hg/native-model.json",
  "data/gg_hg/boundaries-manifest.json", "data/gg_hg/boundaries.json.gz",
  "data/gg_hg/amplitude-validation.json", "data/gg_hg/coherent-reference.json",
];
await mkdir(options.out); // Deliberately refuses to overwrite an earlier observation.
const frozen = join(options.out, "sources");
await mkdir(frozen);
const sources = {};
for (const name of sourceFiles) {
  const bytes = execFileSync("git", ["-C", options.source, "show", `${sourceCommit}:examples/hep/${name}`], { maxBuffer: 32 * 1024 * 1024 });
  const path = join(frozen, name);
  await mkdir(dirname(path), { recursive: true });
  writeFileSync(path, bytes);
  sources[name] = { bytes: bytes.length, sha256: hash(bytes) };
}
const driverBytes = await readFile(driver);
const jsBytes = await readFile(fileURLToPath(import.meta.url));
writeFileSync(join(frozen, "gg_hg_pyodide.py"), driverBytes);
writeFileSync(join(frozen, "gg_hg_pyodide.mjs"), jsBytes);
sources["gg_hg_pyodide.py"] = { bytes: driverBytes.length, sha256: hash(driverBytes) };
sources["gg_hg_pyodide.mjs"] = { bytes: jsBytes.length, sha256: hash(jsBytes) };
const referenceBytes = await readFile(options["native-reference"]);
writeFileSync(join(frozen, "native-reference.json"), referenceBytes);
const runtimeFiles = {};
for (const name of ["pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json"]) {
  const bytes = await readFile(join(options.runtime, name));
  runtimeFiles[name] = { bytes: bytes.length, sha256: hash(bytes) };
}
const identity = {
  wheel_sha256: hash(wheelBytes), sources,
  native_reference_sha256: hash(referenceBytes), runtime_files: runtimeFiles,
};
const manifest = {
  schema: "gg-hg-actual-pyodide-run-v1", status: "starting", stage: options.stage,
  identity, wheel: basename(options.wheel), validation,
  source_checkout: options.source,
  source_commit: sourceCommit,
  node: process.version, available_parallelism: availableParallelism(),
  cpu_affinity: readFileSync("/proc/self/status", "utf8").split("\n").find((line) => line.startsWith("Cpus_allowed_list:")),
  license_environment_supplied: false,
  scope: "Actual Node-hosted Pyodide on one CPU; notebook controller and native physics APIs. No browser DOM, network download or rendering timing claim.",
  timings_seconds: {}, checkpoint_publications: [], started_utc: new Date().toISOString(),
};
const atomicJson = (path, value) => {
  writeFileSync(`${path}.tmp`, JSON.stringify(value, null, 2) + "\n");
  renameSync(`${path}.tmp`, path);
};
const saveManifest = () => atomicJson(join(options.out, "run.json"), manifest);
saveManifest();
let pyodide;
let publication = 0;
// Each generation is written completely before CURRENT points to it. A killed
// process leaves the preceding generation readable, never a mixed bank/report.
function checkpoint() {
  const started = performance.now();
  const root = join(options.out, "checkpoint");
  mkdirSync(root, { recursive: true });
  const name = `generation-${String(publication++).padStart(4, "0")}`;
  const destination = join(root, name);
  mkdirSync(destination);
  const members = {};
  function copyTree(source, target, relative = "") {
    for (const entry of pyodide.FS.readdir(source)) {
      if (entry === "." || entry === "..") continue;
      const from = `${source}/${entry}`;
      const to = join(target, entry);
      const rel = relative ? `${relative}/${entry}` : entry;
      if (pyodide.FS.isDir(pyodide.FS.stat(from).mode)) {
        mkdirSync(to);
        copyTree(from, to, rel);
      } else {
        const bytes = pyodide.FS.readFile(from);
        writeFileSync(to, bytes);
        members[rel] = { bytes: bytes.length, sha256: hash(bytes) };
      }
    }
  }
  copyTree("/acceptance/output", destination);
  atomicJson(join(destination, "checkpoint.json"), { schema: "gg-hg-pyodide-checkpoint-v1", identity, members });
  atomicJson(join(root, "CURRENT"), { generation: name });
  const record = { generation: name, seconds: (performance.now() - started) / 1000 };
  manifest.checkpoint_publications.push(record);
  saveManifest();
}
try {
  const started = performance.now();
  const { loadPyodide } = await import(pathToFileURL(join(options.runtime, "pyodide.mjs")));
  pyodide = await loadPyodide({ indexURL: options.runtime, env: {} });
  manifest.timings_seconds.pyodide_initialization = (performance.now() - started) / 1000;
  manifest.pyodide_version = pyodide.version;
  await pyodide.loadPackage("micropip");
  pyodide.FS.writeFile(`/${basename(options.wheel)}`, wheelBytes);
  pyodide.globals.set("acceptance_wheel_uri", `emfs:/${basename(options.wheel)}`);
  const install = performance.now();
  await pyodide.runPythonAsync(`
import os, sys
assert sys.platform == "emscripten"
assert not os.environ.get("SYMBOLICA_LICENSE")
assert not os.environ.get("SYMBOLICA_LICENSE_KEY")
import micropip
await micropip.install(acceptance_wheel_uri)
`);
  manifest.timings_seconds.wheel_installation = (performance.now() - install) / 1000;
  pyodide.FS.mkdirTree("/acceptance/inputs");
  pyodide.FS.mkdirTree("/acceptance/output");
  for (const name of sourceFiles) {
    const path = `/acceptance/inputs/${name}`;
    pyodide.FS.mkdirTree(dirname(path));
    pyodide.FS.writeFile(path, readFileSync(join(frozen, name)));
  }
  pyodide.FS.writeFile("/acceptance/inputs/native-reference.json", referenceBytes);
  pyodide.FS.writeFile("/acceptance/gg_hg_pyodide.py", driverBytes);
  if (options.resume) {
    const root = resolve(options.resume);
    const current = json(join(root, "CURRENT"));
    assert(/^generation-[0-9]+$/.test(current.generation));
    const previous = join(root, current.generation);
    const saved = json(join(previous, "checkpoint.json"));
    assert.equal(saved.schema, "gg-hg-pyodide-checkpoint-v1");
    assert.deepEqual(saved.identity, identity, "Resume requires identical wheel, runtime, source and comparison input");
    for (const [name, entry] of Object.entries(saved.members)) {
      assert(!name.startsWith("/") && !name.split("/").includes(".."));
      const bytes = readFileSync(join(previous, name));
      assert.equal(hash(bytes), entry.sha256);
      assert.equal(bytes.length, entry.bytes);
      const path = `/acceptance/output/${name}`;
      pyodide.FS.mkdirTree(dirname(path));
      pyodide.FS.writeFile(path, bytes);
    }
    manifest.resumed_from = { directory: root, generation: current.generation };
  }
  pyodide.globals.set("publish_checkpoint", checkpoint);
  pyodide.globals.set("acceptance_stage", options.stage);
  pyodide.globals.set("acceptance_resume", Boolean(options.resume));
  saveManifest();
  await pyodide.runPythonAsync("exec(compile(open('/acceptance/gg_hg_pyodide.py').read(), '/acceptance/gg_hg_pyodide.py', 'exec'))\nawait main()");
  checkpoint();
  const report = JSON.parse(new TextDecoder().decode(pyodide.FS.readFile("/acceptance/output/acceptance.json")));
  assert.equal(report.status, "passed");
  manifest.status = "passed";
  manifest.acceptance = report;
} catch (error) {
  manifest.status = "failed";
  manifest.error = String(error);
  if (pyodide?.FS.analyzePath("/acceptance/output").exists) {
    try { checkpoint(); } catch (checkpointError) { manifest.checkpoint_error = String(checkpointError); }
  }
  throw error;
} finally {
  manifest.finished_utc = new Date().toISOString();
  saveManifest();
}
