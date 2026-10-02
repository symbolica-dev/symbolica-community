#!/usr/bin/env python3
"""Compress a stored wheel with zstd and verify each round trip (Python 3.14+)."""

import argparse
from compression import zstd
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Stored ZIP wheel for HTTP delivery")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 9, 19, 22])
    parser.add_argument("--window-log", type=int, default=23,
                        help="23 limits HTTP zstd frames to 8 MiB (RFC 9659)")
    parser.add_argument("--long-distance-matching", action="store_true",
                        help="Enable zstd's long-distance match finder")
    args = parser.parse_args()
    data = args.wheel.read_bytes()
    report = {"wheel": args.wheel.name, "input_bytes": len(data),
              "input_sha256": hashlib.sha256(data).hexdigest(),
              "zstd_version": zstd.zstd_version, "window_log": args.window_log,
              "http_window_compatible": args.window_log <= 23,
              "long_distance_matching": args.long_distance_matching, "levels": {}}
    suffix = "-ldm" if args.long_distance_matching else ""
    report_path = args.wheel.parent / f"zstd-window{args.window_log}{suffix}.json"
    for level in args.levels:
        start = time.perf_counter()
        options = {
            zstd.CompressionParameter.compression_level: level,
            zstd.CompressionParameter.window_log: args.window_log,
        }
        if args.long_distance_matching:
            options[zstd.CompressionParameter.enable_long_distance_matching] = 1
        encoded = zstd.compress(data, options=options)
        seconds = time.perf_counter() - start
        output = args.wheel.with_name(args.wheel.name + f".zstd{level}-w{args.window_log}{suffix}.zst")
        output.write_bytes(encoded)
        start = time.perf_counter()
        decoded = zstd.decompress(encoded, options={
            zstd.DecompressionParameter.window_log_max: args.window_log,
        })
        decode_seconds = time.perf_counter() - start
        assert decoded == data
        result = {"bytes": len(encoded), "compression_seconds": seconds,
                  "decompression_seconds": decode_seconds,
                  "sha256": hashlib.sha256(encoded).hexdigest(),
                  "file": output.name, "round_trip_verified": True}
        report["levels"][str(level)] = result
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"zstd {level}, window {args.window_log}: {len(encoded)} bytes, {seconds:.3f}s", flush=True)


if __name__ == "__main__":
    main()
