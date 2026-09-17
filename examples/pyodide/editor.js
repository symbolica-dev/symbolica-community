import { EditorState } from "@codemirror/state";
import { EditorView, hoverTooltip, keymap, lineNumbers, highlightActiveLine, drawSelection } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { python, pythonLanguage } from "@codemirror/lang-python";
import { HighlightStyle, syntaxHighlighting, syntaxTree } from "@codemirror/language";
import { autocompletion, acceptCompletion, startCompletion, completionStatus } from "@codemirror/autocomplete";
import { tags, classHighlighter, highlightCode } from "@lezer/highlight";
import { marked } from "marked";
import DOMPurify from "dompurify";

const pythonHighlighting = HighlightStyle.define([
  { tag: tags.keyword, color: "#e8b4d9" },
  { tag: tags.string, color: "#c3df99" },
  { tag: [tags.number, tags.bool, tags.null], color: "#efbc83" },
  { tag: tags.comment, color: "#9cb6a8", fontStyle: "italic" },
  { tag: tags.function(tags.variableName), color: "#a9d8ef" },
  { tag: [tags.typeName, tags.className], color: "#f0d49b" },
  { tag: [tags.operator, tags.punctuation], color: "#c5d8ce" },
  { tag: tags.meta, color: "#a9d8ef" },
]);

// All editors on a page share the same WASM instance and stub workspace.
let typeClient;
function createTypeClient() {
  const worker = new Worker("ty-worker.js", { type: "module" });
  const requests = new Map();
  const listeners = new Set();
  let nextId = 0, ready = false;
  let status = "Loading ty type hints…";
  const notify = message => { status = message; for (const listener of listeners) listener(message); };
  const fail = message => {
    ready = false;
    notify(`Type hints unavailable: ${message}`);
    for (const { resolve, timer } of requests.values()) { clearTimeout(timer); resolve(null); }
    requests.clear();
  };
  worker.onmessage = ({ data }) => {
    if (data.type === "ready") { ready = true; notify(`ty ${data.version} · Hover for types · Tab to complete · Alt+/ for suggestions`); return; }
    if (data.type === "error") { fail(data.message); return; }
    const request = requests.get(data.id);
    if (request) { clearTimeout(request.timer); requests.delete(data.id); request.resolve(data.type === "complete" ? data.completions : data.hover); }
  };
  worker.onerror = event => fail(event.message);
  function request(type, source, line, column) {
    if (!ready) return Promise.resolve(null);
    return new Promise(resolve => {
      const id = ++nextId;
      const timer = setTimeout(() => { requests.delete(id); resolve(null); }, 10000);
      requests.set(id, { resolve, timer });
      worker.postMessage({ type, id, source, line, column });
    });
  }
  return {
    getHover: (...args) => request("hover", ...args),
    getCompletions: (...args) => request("complete", ...args),
    subscribe(listener) { listeners.add(listener); listener(status); },
    unsubscribe(listener) {
      listeners.delete(listener);
      if (!listeners.size) { worker.terminate(); fail("Editors closed"); typeClient = undefined; }
    },
  };
}

function highlightPython(parent, code) {
  highlightCode(code, pythonLanguage.parser.parse(code), classHighlighter, (text, classes) => {
    const span = document.createElement("span");
    span.className = classes;
    span.textContent = text;
    parent.append(span);
  }, () => parent.append(document.createTextNode("\n")));
}

export function renderTypeMarkdown(markdown) {
  const content = document.createElement("div");
  content.className = "type-hint-body";
  content.innerHTML = DOMPurify.sanitize(marked.parse(markdown, { async: false }), {
    ALLOWED_TAGS: ["p", "br", "hr", "pre", "code", "strong", "em", "del", "ul", "ol", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "a", "table", "thead", "tbody", "tr", "th", "td"],
    ALLOWED_ATTR: ["class", "href", "title", "start"],
  });
  for (const link of content.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  for (const block of content.querySelectorAll("pre > code")) {
    if (block.className && !/language-(python|py)\b/.test(block.className)) continue;
    const code = block.textContent;
    block.replaceChildren();
    highlightPython(block, code);
  }
  return content;
}

export function createEditor(parent, { onRun, onTypeStatus = () => {}, height = "340px", ariaLabel = "Python code", filename = "playground.py" }) {
  const client = typeClient ??= createTypeClient();
  // Wrap callbacks so two editors can safely subscribe with the same callback.
  const listener = message => onTypeStatus(message);
  client.subscribe(listener);
  const completionTypes = { 1: "method", 2: "function", 3: "class", 4: "property", 5: "variable", 6: "class", 7: "interface", 8: "namespace", 9: "property", 12: "enum", 13: "keyword", 20: "constant", 21: "variable", 24: "type" };
  const complete = async context => {
    const { state, pos } = context;
    const node = syntaxTree(state).resolveInner(pos, -1);
    if (/Comment|String|Number/.test(node.name)) return null;
    const line = state.doc.lineAt(pos);
    const before = state.doc.sliceString(line.from, pos);
    if (!context.explicit && !/[\p{ID_Continue}.]$/u.test(before)) return null;
    const word = context.matchBefore(/[\p{ID_Continue}]+/u);
    const results = await client.getCompletions(state.doc.toString(), line.number, pos - line.from + 1);
    if (context.aborted || !results?.length) return null;
    const source = state.doc.toString();
    const from = word?.from ?? pos;
    const wordEnd = pos + (source.slice(pos).match(/^[\p{ID_Continue}]*/u)?.[0].length ?? 0);
    const documentation = new Map();
    const completionInfo = item => {
      // The completion API returns plaintext docstrings. Ask hover for the
      // selected name's structured Markdown in a temporary source snapshot.
      // This never inserts text into the editor or runs Python.
      if (!documentation.has(item.label)) {
        const completedSource = source.slice(0, from) + item.label + source.slice(wordEnd);
        const column = from - line.from + 1 + Math.min(1, item.label.length - 1);
        documentation.set(item.label, client.getHover(completedSource, line.number, column));
      }
      return documentation.get(item.label);
    };
    return {
      from,
      validFor: /^[\p{ID_Continue}]*$/u,
      options: results.map(item => ({
        label: item.label, type: completionTypes[item.kind], apply: item.insertText ?? item.label,
        // Public names come first for equally good matches. Explicitly typing
        // an underscore still benefits from the normal prefix-match ranking.
        sortText: `${item.label.startsWith("_") ? "1" : "0"}${item.label}`,
        // Render the signature ourselves; CodeMirror's detail field is plain text.
        signature: item.detail,
        info: item.documentation || item.detail ? async () => {
          const dom = document.createElement("div");
          dom.className = "type-hint";
          const result = await completionInfo(item);
          if (result?.markdown) {
            dom.append(renderTypeMarkdown(result.markdown));
          } else {
            const signature = item.detail ? "```python\n" + item.detail + "\n```" : "";
            const body = renderTypeMarkdown(signature);
            if (item.documentation) {
              const text = document.createElement("div");
              text.className = "type-hint-plain";
              text.textContent = item.documentation;
              body.append(text);
            }
            dom.append(body);
          }
          return dom;
        } : undefined,
      })),
    };
  };
  const completeWithTab = view => {
    if (acceptCompletion(view)) return true;
    if (completionStatus(view.state)) return true;
    const selection = view.state.selection.main;
    if (/Comment|String|Number/.test(syntaxTree(view.state).resolveInner(selection.head, -1).name)) return false;
    const line = view.state.doc.lineAt(selection.head);
    const before = view.state.doc.sliceString(line.from, selection.head);
    // Let the usual Tab binding indent whitespace and selected lines.
    if (!selection.empty || !/[\p{ID_Continue}.]$/u.test(before)) return false;
    return startCompletion(view);
  };
  const saveCode = view => {
    const url = URL.createObjectURL(new Blob([view.state.doc.toString()], { type: "text/x-python;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    try { link.click(); }
    finally {
      link.remove();
      // Give the browser time to start reading the download before releasing it.
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    return true;
  };
  const hover = hoverTooltip(async (view, pos) => {
    const document = view.state.doc;
    const line = document.lineAt(pos);
    const result = await client.getHover(document.toString(), line.number, pos - line.from + 1);
    if (!result || view.state.doc !== document) return null;
    const start = document.line(result.start.line).from + result.start.column - 1;
    const end = document.line(result.end.line).from + result.end.column - 1;
    return {
      pos: start, end, above: true,
      create() {
        const dom = window.document.createElement("div");
        dom.className = "type-hint";
        const title = dom.appendChild(window.document.createElement("div"));
        title.className = "type-hint-title";
        title.textContent = "ty · Type information";
        dom.append(renderTypeMarkdown(result.markdown));
        return { dom };
      },
    };
  }, { hoverTime: 300 });

  const view = new EditorView({
    parent,
    state: EditorState.create({ extensions: [
      python(), syntaxHighlighting(pythonHighlighting), lineNumbers(), highlightActiveLine(), drawSelection(), history(), hover,
      autocompletion({ override: [complete], activateOnTyping: true, activateOnTypingDelay: 200,
        addToOptions: [{ position: 80, render(completion) {
          if (!completion.signature) return null;
          const signature = document.createElement("span");
          signature.className = "cm-completionSignature";
          highlightPython(signature, completion.signature);
          return signature;
        } }],
      }),
      keymap.of([{ key: "Mod-Enter", run: () => { onRun(); return true; } },
        { key: "Mod-s", run: saveCode, preventDefault: true },
        { key: "Tab", run: completeWithTab }, { key: "Alt-/", run: startCompletion },
        indentWithTab, ...defaultKeymap, ...historyKeymap]),
      EditorView.contentAttributes.of({ "aria-label": ariaLabel, spellcheck: "false" }),
      EditorView.theme({
        "&": { height, backgroundColor: "#20342e", color: "#ebeee4" },
        ".cm-scroller": { overflow: "auto", fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace", fontSize: "13px", lineHeight: "1.8" },
        ".cm-content": { padding: "22px 0", caretColor: "#fff" },
        ".cm-line": { padding: "0 22px 0 8px" },
        ".cm-gutters": { backgroundColor: "#20342e", color: "#819f90", border: "none", padding: "0 8px" },
        // Selection rectangles sit behind the line: an opaque background hides them.
        ".cm-activeLine": { backgroundColor: "#8fbea014" },
        ".cm-activeLineGutter": { backgroundColor: "#294339" },
        "& .cm-selectionBackground": { backgroundColor: "#3c5169" },
        "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground": { backgroundColor: "#375f8d" },
        ".cm-cursor": { borderLeftColor: "#fff" },
        ".cm-tooltip": { backgroundColor: "#fffefb", color: "#253a34", border: "1px solid #c7d2c2", borderRadius: "8px", boxShadow: "0 6px 22px #0003" },
      }, { dark: true }),
    ] }),
  });
  return {
    get value() { return view.state.doc.toString(); },
    set value(text) { view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text }, selection: { anchor: 0 } }); view.scrollDOM.scrollTop = 0; },
    destroy() { client.unsubscribe(listener); view.destroy(); },
  };
}
