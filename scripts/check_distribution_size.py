"""Require release distributions to fit a conservative 100 MB upload budget."""

import argparse
from pathlib import Path
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", nargs="+", type=Path)
    parser.add_argument("--max-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be positive")
    oversized = []
    for path in args.distributions:
        size = path.stat().st_size
        print(f"{path.name}: {size:,} bytes ({size / 1_000_000:.2f} MB)")
        if size < args.max_bytes:
            continue
        oversized.append(path.name)
        if path.suffix == ".whl":
            with ZipFile(path) as wheel:
                for entry in sorted(wheel.infolist(), key=lambda item: item.compress_size, reverse=True)[:5]:
                    print(f"  {entry.filename}: {entry.compress_size:,} compressed bytes")
    if oversized:
        raise SystemExit(
            f"Distributions must be smaller than {args.max_bytes:,} bytes: "
            + ", ".join(oversized)
        )


if __name__ == "__main__":
    main()
