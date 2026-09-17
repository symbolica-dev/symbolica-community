#!/usr/bin/env python3
"""Serve the playground, a local Pyodide distribution, and a community wheel."""

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile


def main():
    cache = Path.home() / ".cache/symbolica-community-build"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--ty-runtime", type=Path, default=cache / "ty-wasm")
    parser.add_argument(
        "--runtime", type=Path,
        default=cache / "pyodide-xbuildenv/314.0.7/xbuildenv/pyodide-root/dist",
        help="Pyodide distribution directory (pyodide config get dist_dir)",
    )
    args = parser.parse_args()
    wheels = sorted((cache / "wasm-dist").glob("*-pyemscripten_2026_0_wasm32.whl"))
    wheel = args.wheel or (wheels[-1] if wheels else None)
    if wheel is None or not wheel.is_file():
        parser.error("Build a PyEmscripten wheel first, or pass --wheel PATH.")
    runtime = args.runtime.resolve()
    if not (runtime / "pyodide.mjs").is_file():
        parser.error("Pass --runtime pointing to a Pyodide distribution.")
    wheel = wheel.resolve()
    site = Path(__file__).resolve().parent
    ty_runtime = args.ty_runtime.resolve()
    if not (ty_runtime / "ty_wasm.js").is_file():
        parser.error("Build ty's WASM module first; see examples/pyodide/README.md.")
    if not all((site / "assets" / name).is_file() for name in ("editor.js", "output.js")):
        parser.error("Run npm ci && npm run build in examples/pyodide first.")
    with ZipFile(wheel) as archive:
        stubs = json.dumps({"files": {
            name: archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.startswith("symbolica/") and name.endswith((".py", ".pyi", "py.typed"))
        }}).encode()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if urlsplit(self.path).path == "/type-stubs.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(stubs)))
                self.end_headers()
                self.wfile.write(stubs)
                return
            if urlsplit(self.path).path == "/config.json":
                body = json.dumps({
                    "wheel": f"/wheels/{wheel.name}",
                    "version": wheel.name.split("-")[1],
                    "bytes": wheel.stat().st_size,
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def translate_path(self, path):
            path = unquote(urlsplit(path).path)
            if path == f"/wheels/{wheel.name}":
                return str(wheel)
            if path.startswith("/runtime/"):
                root, relative = runtime, path[9:]
            elif path.startswith("/ty/"):
                root, relative = ty_runtime, path[4:]
            else:
                root, relative = site, path.lstrip("/")
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                return str(site / "__not_found__")
            return str(resolved)

        def list_directory(self, path):
            self.send_error(404)
            return None

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Symbolica playground: http://localhost:{args.port}", flush=True)
    print(f"Wheel: {wheel}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
