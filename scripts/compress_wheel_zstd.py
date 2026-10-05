#!/usr/bin/env python3
"""Recompress a wheel in place with zstd level 22 (requires Python 3.14+)."""

import argparse
import copy
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from zipfile import ZIP_ZSTANDARD, ZipFile


def compress_wheel(wheel: Path) -> None:
    original_size = wheel.stat().st_size
    with TemporaryDirectory(dir=wheel.parent) as temporary:
        output = Path(temporary) / wheel.name
        with ZipFile(wheel) as source, ZipFile(output, "w") as destination:
            destination.comment = source.comment
            for info in source.infolist():
                entry = copy.copy(info)
                entry.compress_type = ZIP_ZSTANDARD
                entry.compress_level = 22
                with source.open(info) as reader, destination.open(entry, "w") as writer:
                    shutil.copyfileobj(reader, writer)
        with ZipFile(output) as archive:
            if any(info.compress_type != ZIP_ZSTANDARD for info in archive.infolist()):
                raise ValueError("Compressed wheel contains non-zstd entries")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"Compressed wheel failed CRC verification: {corrupt}")
        output.replace(wheel)
    print(f"{wheel.name}: {original_size} -> {wheel.stat().st_size} bytes (zstd level 22)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    compress_wheel(args.wheel)


if __name__ == "__main__":
    main()
