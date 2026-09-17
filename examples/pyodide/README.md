# Local Pyodide playground

After building the PyEmscripten wheel, build the editor and ty once
(requires Node/npm, Rust stable 1.98 or newer, and wasm-pack):

```sh
cd examples/pyodide
npm ci
npm run build
python3 build_ty.py
cd ../..
```

Run from the repository root:

```sh
python3 examples/pyodide/serve.py
```

Open <http://localhost:8000>. The default paths use the build cache at
`~/.cache/symbolica-community-build`. To use other build outputs:

```sh
python3 examples/pyodide/serve.py \
  --wheel /path/to/symbolica-3.0.0-cp314-abi3-pyemscripten_2026_0_wasm32.whl \
  --runtime "$(pyodide config get dist_dir)" \
  --port 8000
```

The server binds to localhost and serves the existing files without copying them.
Pyodide, micropip, and the community wheel are loaded locally. Python executes
in a web worker, keeping the page responsive. Run with Ctrl/Command + Enter;
variables persist between runs. Stop terminates the worker; Reset starts a fresh
Python session. Examples cover algebra, integration, Spenso, and Idenso.

Hover over a Python name or method to see its inferred type, signature, and
documentation. A separate ty WASM worker analyzes the current editor contents.
The server extracts `.pyi` files, Python package wrappers, and `py.typed` directly
from the same wheel used by Pyodide, preserving the package's import structure.
Hover hints do not require executing the code. Definitions that exist only in
previous runs are not part of this static analysis. NumPy-specific annotations
may be incomplete because this example loads only Symbolica's package files.

ty is built from a pinned upstream revision and served locally with the editor;
no CDN is used at runtime. The status below the editor shows when type hints are
ready. Override `--ty-runtime` if the generated `ty_wasm.js` and
`ty_wasm_bg.wasm` are stored elsewhere.

Completions use that same ty worker and the wheel's stubs, without running Python.
Type a name or `.` to see suggestions, or press **Tab** after a name to request
them. **Tab** or **Enter** accepts the selected item; arrow keys select and
**Escape** closes the list. **Alt+/** opens suggestions explicitly, avoiding
browser shortcuts that intercept Ctrl+Space. Tab on indentation or a selection
still indents, and Shift+Tab unindents. Completion uses the current cell's source
and imports; it does not add imports or inspect variables from earlier runs.
The selected completion uses ty's hover Markdown for documentation, so parameter
lists and code examples have the same formatting as ordinary hover hints.

**Ctrl+S / Command+S** in an editor downloads its current contents as a UTF-8
Python file: `playground.py` in the playground, or the example's filename in
the quick start (for example, `integrate.py`). This also works before Python
finishes loading. Only the focused cell is saved; outputs are not included.

Vakint and compiled native evaluators are unavailable in this WASM build.
If needed, set your Symbolica license in the editor with
`symbolica.set_license_key(...)`; the playground does not persist the editor.

## Live quick start

Open <http://localhost:8000/quick-start.html> for an interactive adaptation of
the [Symbolica quick start](https://symbolica.io/docs/quick_start.html). Nine
editable examples cover expressions, expansion, patterns, differentiation,
integration, rational functions, series and solving, numerical evaluation, and
polynomial objects. Run examples individually or with **Run all**. Each run
uses fresh Python globals; the shared runtime still retains imported modules
and Symbolica's process-wide state. Stop terminates Python; Reset reloads it.

The numerical example uses `evaluate_with_prec`, which works without native
JIT compilation or NumPy. Both pages render ty hover documentation as sanitized
Markdown, with syntax-highlighted Python signatures and examples. All editors
share a single ty WASM worker. Hover results describe the current source in
the hovered editor and do not depend on running the examples.

## Rich output

On either page, leave an object as the final expression to display its
`_repr_html_()` result, as in a notebook:

```python
from symbolica import E
expression = E("(x + 1)^2 / 3")
expression
```

`print(expression)` still produces plain text. If the object has no usable HTML
representation, its Python `repr()` is displayed instead. `None` and cells
ending with an assignment produce no final result. HTML is sanitized before
insertion, preserving basic formatting, tables, and Symbolica's colors;
scripts, embedded frames, and arbitrary CSS are not executed. A trailing
semicolon suppresses the final result, following Pyodide's default behavior.
