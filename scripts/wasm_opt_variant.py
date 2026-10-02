#!/usr/bin/env python3
"""Replay a captured Emscripten wasm-opt command and package a testable wheel."""

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path, help="Directory containing invocation.json")
    parser.add_argument("wheel", type=Path, help="Original baseline wheel")
    parser.add_argument("outdir", type=Path, help="Separate output directory for this variant")
    parser.add_argument("--opt", default="-O3", choices=["-O0", "-O1", "-O2", "-O3", "-O4", "-Os", "-Oz"])
    parser.add_argument("--wasm-opt", help="Override the captured Binaryen executable")
    parser.add_argument("--extra", action="append", default=[], help="Additional optimizer flag, e.g. --extra=--flatten")
    args = parser.parse_args()
    capture = args.capture.resolve()
    invocation = json.loads((capture / "invocation.json").read_text())
    executable = args.wasm_opt or invocation["executable"]
    if not Path(executable).exists():
        executable = shutil.which(executable) or shutil.which("wasm-opt")
    if not executable:
        parser.error("Pass --wasm-opt=/path/to/wasm-opt")
    args.outdir.mkdir(parents=True, exist_ok=True)
    output = (args.outdir / "core.wasm").resolve()
    wheel_output = (args.outdir / args.wheel.name).resolve()
    if wheel_output == args.wheel.resolve() or output.exists() or wheel_output.exists():
        parser.error("Choose a fresh output directory; baseline files are never overwritten")
    replacements = {
        name: str(capture / f"input-{i}.wasm")
        for i, name in enumerate(invocation["inputs"])
    }
    # Emscripten normally optimizes in place, so input and output have the
    # same original path. Distinguish them by their argument positions.
    flags = []
    output_next = False
    for flag in invocation["arguments"]:
        if output_next:
            flags.append(str(output))
            output_next = False
        elif flag in ("-o", "--output"):
            flags.append(flag)
            output_next = True
        elif flag in ("-O0", "-O1", "-O2", "-O3", "-O4", "-Os", "-Oz"):
            flags.append(args.opt)
        else:
            flags.append(replacements.get(flag, flag))
    command = [str(executable), *flags, *args.extra]
    subprocess.run(command, check=True)
    payload = output.read_bytes()
    assert payload[:4] == b"\0asm"
    with ZipFile(args.wheel) as source, ZipFile(wheel_output, "w") as destination:
        modules = [name for name in source.namelist() if name.endswith(".so")]
        assert len(modules) == 1, modules
        record = next(name for name in source.namelist() if name.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(source.read(record).decode())))
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        for row in rows:
            if row[0] == modules[0]:
                row[1:] = ["sha256=" + digest, str(len(payload))]
        record_data = io.StringIO(newline="")
        csv.writer(record_data, lineterminator="\n").writerows(rows)
        for info in source.infolist():
            data = payload if info.filename == modules[0] else source.read(info.filename)
            if info.filename == record:
                data = record_data.getvalue().encode()
            destination.writestr(info, data)
    (args.outdir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"{wheel_output}: {wheel_output.stat().st_size} bytes; WASM: {len(payload)} bytes")
    print("Validate this variant with .github/scripts/test_pyodide.mjs before using it.")


if __name__ == "__main__":
    main()
