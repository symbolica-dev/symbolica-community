#!/usr/bin/env python3
"""Serve a bundle locally, using precompressed .br/.zst files for wheel URLs."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = Path(self.translate_path(self.path))
        accepted = {}
        for entry in self.headers.get("Accept-Encoding", "").split(","):
            encoding, *parameters = entry.strip().split(";")
            quality = 1.0
            for parameter in parameters:
                if parameter.strip().startswith("q="):
                    try:
                        quality = float(parameter.strip()[2:])
                    except ValueError:
                        quality = 0.0
            accepted[encoding] = quality
        candidates = []
        for encoding, suffix in (("br", ".br"), ("zstd", ".zst")):
            encoded = Path(str(path) + suffix)
            quality = accepted.get(encoding, accepted.get("*", 0.0))
            if path.suffix == ".whl" and encoded.is_file() and quality > 0:
                candidates.append((-quality, encoded.stat().st_size, encoding, encoded))
        if candidates:
            _, size, encoding, encoded = min(candidates)
            stream = encoded.open("rb")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Encoding", encoding)
            self.send_header("Content-Length", str(size))
            self.send_header("Vary", "Accept-Encoding")
            self.end_headers()
            return stream
        return super().send_head()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.bind, args.port), partial(Handler, directory=str(args.directory.resolve())))
    print(f"Serving at http://{args.bind}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
