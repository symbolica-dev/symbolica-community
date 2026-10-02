#!/usr/bin/env python3
"""Prepare a stored ZIP wheel for HTTP Brotli delivery without changing code."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import brotli


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--quality", type=int, default=11, choices=range(12))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    stored = args.outdir / args.wheel.name
    if stored.resolve() == args.wheel.resolve() or stored.exists():
        parser.error("Choose a fresh output directory")
    with ZipFile(args.wheel) as source, ZipFile(stored, "w") as destination:
        for info in source.infolist():
            entry = copy.copy(info)
            entry.compress_type = ZIP_STORED
            destination.writestr(entry, source.read(info.filename))
    data = stored.read_bytes()
    encoded = brotli.compress(data, quality=args.quality)
    (args.outdir / (args.wheel.name + ".br")).write_bytes(encoded)
    report = {"wheel": stored.name, "stored_zip_bytes": len(data), "http_brotli_bytes": len(encoded), "brotli_quality": args.quality, "http_brotli_sha256": hashlib.sha256(encoded).hexdigest(), "package_contents_unchanged": True}
    (args.outdir / "transport-size.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
