#!/usr/bin/env python3
"""Remove community Python packages from a wheel built with wasm-core."""

import argparse
import copy
import csv
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


def prepare_core_wheel(wheel: Path) -> None:
    with TemporaryDirectory(dir=wheel.parent) as temporary:
        output = Path(temporary) / wheel.name
        with ZipFile(wheel) as source, ZipFile(output, "w") as destination:
            removed = {name for name in source.namelist()
                       if name.startswith("symbolica/community/")}
            record = next(name for name in source.namelist()
                          if name.endswith(".dist-info/RECORD"))
            rows = csv.reader(io.StringIO(source.read(record).decode()))
            contents = io.StringIO(newline="")
            csv.writer(contents, lineterminator="\n").writerows(
                row for row in rows if row[0] not in removed
            )
            destination.comment = source.comment
            for info in source.infolist():
                if info.filename in removed:
                    continue
                data = contents.getvalue().encode() if info.filename == record else source.read(info)
                destination.writestr(copy.copy(info), data,
                                     compress_type=ZIP_DEFLATED, compresslevel=9)
        with ZipFile(output) as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ValueError(f"Core wheel failed CRC verification: {corrupt}")
        output.replace(wheel)
    print(f"{wheel.name}: removed {len(removed)} community package entries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    prepare_core_wheel(parser.parse_args().wheel)
