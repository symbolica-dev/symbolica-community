#!/usr/bin/env python3
"""Measure a Pyodide wheel and its WASM payload (sizes exclude the runtime)."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def section_sizes(data):
    names = {0: "custom", 1: "type", 2: "import", 3: "function", 4: "table",
             5: "memory", 6: "global", 7: "export", 8: "start", 9: "element",
             10: "code", 11: "data", 12: "data_count", 13: "tag"}
    result = {}
    offset = 8
    while offset < len(data):
        kind = data[offset]
        offset += 1
        length = shift = 0
        while True:
            byte = data[offset]
            offset += 1
            length |= (byte & 127) << shift
            if not byte & 128:
                break
            shift += 7
        name = names.get(kind, f"section_{kind}")
        result[name] = result.get(name, 0) + length
        offset += length
    assert offset == len(data), "Invalid WASM section boundaries"
    return result


def sizes(data):
    result = {
        "bytes": len(data),
        "MiB": round(len(data) / 2**20, 3),
        "gzip_9_bytes": len(gzip.compress(data, compresslevel=9, mtime=0)),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    try:
        import brotli
    except ImportError:
        pass
    else:
        result["brotli_11_bytes"] = len(brotli.compress(data, quality=11))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with ZipFile(args.wheel) as archive:
        modules = [name for name in archive.namelist() if name.endswith(".so")]
        assert len(modules) == 1, modules
        module = archive.read(modules[0])
        assert module[:4] == b"\0asm", "Extension must contain WebAssembly"
        report = {
            "wheel": {"filename": args.wheel.name, **sizes(args.wheel.read_bytes())},
            "wasm": {"wheel_path": modules[0], **sizes(module),
                     "section_payload_bytes": section_sizes(module)},
            "installed_bytes": sum(info.file_size for info in archive.infolist()),
            "runtime_included": False,
        }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
