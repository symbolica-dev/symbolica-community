#!/usr/bin/env python3
"""Remove Community Python packages from a wheel built with wasm-core.

The native module must already have been built without Community features.
Resource requirements and the core stub are preserved, including its local
reflected-arithmetic protocols, which do not import optional Community types.
"""

import argparse
import base64
import copy
import csv
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


def prepare_core_wheel(wheel: Path) -> None:
    with TemporaryDirectory(dir=wheel.parent) as temporary:
        output = Path(temporary) / wheel.name
        with ZipFile(wheel) as source:
            records = [name for name in source.namelist() if name.endswith(".dist-info/RECORD")]
            if len(records) != 1:
                raise ValueError("Expected one wheel RECORD")
            record_name = records[0]
            removed = 0
            rows = []
            with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as target:
                target.comment = source.comment
                for info in source.infolist():
                    if info.filename.startswith(("symbolica/community/", "examples/hep/")):
                        removed += 1
                        continue
                    if info.filename == record_name:
                        continue
                    payload = source.read(info)
                    entry = copy.copy(info)
                    target.writestr(entry, payload)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
                    rows.append((info.filename, "sha256=" + digest.decode(), str(len(payload))))
                record = io.StringIO(newline="")
                writer = csv.writer(record, lineterminator="\n")
                writer.writerows(sorted(rows))
                writer.writerow((record_name, "", ""))
                target.writestr(copy.copy(source.getinfo(record_name)), record.getvalue())
        with ZipFile(output) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"Core wheel failed CRC verification: {corrupt}")
        output.replace(wheel)
    print(f"Prepared core-only wheel {wheel.name}: removed {removed} Community/example files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    prepare_core_wheel(parser.parse_args().wheel)


if __name__ == "__main__":
    main()
