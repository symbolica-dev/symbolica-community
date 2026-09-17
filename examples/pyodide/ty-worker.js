import init, { Workspace, Position, PositionEncoding, version } from "/ty/ty_wasm.js";

let workspace, document;
const initialized = (async () => {
  await init();
  const response = await fetch("/type-stubs.json");
  if (!response.ok) throw new Error(`Could not load wheel stubs (${response.status})`);
  const { files } = await response.json();
  workspace = new Workspace("/project", PositionEncoding.Utf16, { environment: { "python-version": "3.14" } });
  for (const [path, contents] of Object.entries(files)) {
    // Keep these files in ty's virtual filesystem for import resolution.
    workspace.openFile(`/project/${path}`, contents).free();
  }
  document = workspace.openFile("/project/playground.py", "");
  postMessage({ type: "ready", version: version() });
})();
initialized.catch(error => postMessage({ type: "error", message: String(error) }));

self.onmessage = async ({ data }) => {
  if (data.type !== "hover" && data.type !== "complete") return;
  try {
    await initialized;
    workspace.updateFile(document, data.source);
    if (data.type === "complete") {
      const results = workspace.completions(document, new Position(data.line, data.column));
      const completions = [];
      try {
        for (const item of results) {
          const edits = item.additional_text_edits;
          if (edits) {
            for (const edit of edits) edit.free();
            // Only offer in-scope names: accepting a completion does not add imports.
            if (edits.length) continue;
          }
          completions.push({ label: item.name, kind: item.kind, insertText: item.insert_text,
            detail: item.detail, documentation: item.documentation });
        }
        postMessage({ type: "complete", id: data.id, completions });
      } finally { for (const item of results) item.free(); }
      return;
    }
    // Position is passed by value: wasm-bindgen consumes its handle.
    const result = workspace.hover(document, new Position(data.line, data.column));
    if (!result) { postMessage({ type: "hover", id: data.id, hover: null }); return; }
    const range = result.range;
    const start = range.start;
    const end = range.end;
    try {
      postMessage({ type: "hover", id: data.id, hover: {
        markdown: result.markdown,
        start: { line: start.line, column: start.column },
        end: { line: end.line, column: end.column },
      } });
    } finally { start.free(); end.free(); range.free(); result.free(); }
  } catch (error) {
    postMessage({ type: "error", message: String(error) });
  }
};
