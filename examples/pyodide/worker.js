import { loadPyodide } from "/runtime/pyodide.mjs";

let pyodide, formatResult;
const send = (type, data = {}) => postMessage({ type, ...data });
self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      send("status", { text: "Loading Python 3.14…" });
      pyodide = await loadPyodide({ indexURL: "/runtime/" });
      send("status", { text: "Loading micropip…" });
      await pyodide.loadPackage("micropip");
      send("status", { text: "Installing Symbolica community…" });
      pyodide.globals.set("wheel_url", new URL(data.wheel, self.location.origin).href);
      await pyodide.runPythonAsync("import micropip\nawait micropip.install(wheel_url)\nfrom symbolica import E, S\nimport symbolica.community.spenso\nimport symbolica.community.idenso");
      // Keep display helpers separate from the user's persistent Python globals.
      const formatterGlobals = pyodide.runPython("dict()");
      try {
        formatResult = pyodide.runPython(`
def format_result(value):
    try:
        render = getattr(value, "_repr_html_", None)
        html = render() if callable(render) else None
        # IPython also permits (HTML, metadata); this page uses the HTML only.
        if isinstance(html, tuple) and len(html) == 2:
            html = html[0]
        if isinstance(html, str):
            return ("html", html)
    except Exception:
        pass  # A broken optional formatter should still allow plain output.
    return ("text", repr(value))

format_result
`, { globals: formatterGlobals });
      } finally {
        formatterGlobals.destroy();
      }
      pyodide.setStdout({ batched: text => send("output", { text: text + "\n" }) });
      pyodide.setStderr({ batched: text => send("output", { text: text + "\n" }) });
      send("ready");
    } else if (data.type === "run") {
      const start = performance.now();
      let globals;
      try {
        if (data.isolated) globals = pyodide.runPython("dict(__name__='__main__')");
        const result = await pyodide.runPythonAsync(data.code, globals ? { globals } : undefined);
        try {
          if (result !== undefined) {
            const formatted = formatResult(result);
            try {
              const [format, content] = formatted.toJs();
              if (format === "html") send("display", { html: content });
              else send("output", { text: content + "\n" });
            } finally {
              formatted.destroy();
            }
          }
        } finally {
          result?.destroy?.();
        }
        send("done", { elapsed: performance.now() - start });
      } catch (error) {
        send("error", { text: String(error), fatal: false });
      } finally {
        globals?.destroy();
      }
    }
  } catch (error) {
    send("error", { text: String(error), fatal: true });
  }
};
