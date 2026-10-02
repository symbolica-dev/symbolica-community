#!/usr/bin/env python3
"""Wrap wasm-opt to preserve the linked input before Emscripten transforms it.

Set REAL_WASM_OPT and WASM_OPT_CAPTURE_DIR, and install this wrapper as
bin/wasm-opt under a private EM_BINARYEN_ROOT. Other SDK binaries can be symlinks.
WASM_OPT_CAPTURE_INPUT_NAME selects the final module instead of dependencies.
WASM_OPT_FORCE_LEVEL changes only its Binaryen level, preserving linker defaults.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    real = os.environ["REAL_WASM_OPT"]
    arguments = sys.argv[1:]
    capture = None
    output = None
    if "--post-emscripten" in arguments:
        inputs = []
        output_next = False
        for argument in arguments:
            if output_next:
                output = argument
                output_next = False
            elif argument in ("-o", "--output"):
                output_next = True
            elif argument.endswith(".wasm") and Path(argument).is_file():
                inputs.append(argument)
        input_name = os.environ.get("WASM_OPT_CAPTURE_INPUT_NAME")
        if inputs and (not input_name or any(Path(p).name == input_name for p in inputs)):
            capture = Path(os.environ["WASM_OPT_CAPTURE_DIR"])
            capture.mkdir(parents=True, exist_ok=False)
            for index, name in enumerate(inputs):
                shutil.copy2(name, capture / f"input-{index}.wasm")
            (capture / "invocation.json").write_text(json.dumps({
                "executable": real, "arguments": arguments, "cwd": os.getcwd(),
                "inputs": inputs, "output": output,
            }, indent=2) + "\n")
            level = os.environ.get("WASM_OPT_FORCE_LEVEL")
            if level:
                levels = {"-O0", "-O1", "-O2", "-O3", "-O4", "-Os", "-Oz"}
                if level not in levels:
                    raise ValueError(f"Invalid WASM_OPT_FORCE_LEVEL: {level}")
                arguments = [level if flag in levels else flag for flag in arguments]
                (capture / "baseline-invocation.json").write_text(
                    json.dumps([real, *arguments], indent=2) + "\n"
                )
    result = subprocess.run([real, *arguments])
    if result.returncode == 0 and capture is not None and output:
        shutil.copy2(output, capture / "output.wasm")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
