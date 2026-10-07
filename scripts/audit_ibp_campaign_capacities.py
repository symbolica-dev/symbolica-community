#!/usr/bin/env python3
"""Check const-generic campaign widths in a production RustRed app archive.

Use the LLVM nm shipped with the build's Emscripten SDK: release archives may
contain LLVM bitcode. Unit-test binaries intentionally instantiate extra widths
and are not suitable evidence for the packaged runtime.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


CAMPAIGN = "rustred_app::application::routed_campaign::"
WIDTH = re.compile(re.escape(CAMPAIGN) + r"[\w:]+(?:::)?<(\d+)[,>]")
VERIFY = re.compile(re.escape(CAMPAIGN) + r"walking::verify_closure::verify(?:::)?<(\d+)[,>]")


def audit(archive: Path, llvm_nm: str, expected: set[int]) -> dict:
    result = subprocess.run(
        [llvm_nm, "--defined-only", "--demangle", "--format=posix", str(archive)],
        check=True, capture_output=True, text=True,
    )
    widths = Counter()
    verify_widths = set()
    total_bytes = campaign_bytes = campaign_functions = 0
    for line in result.stdout.splitlines():
        fields = line.rsplit(" ", 3)
        if len(fields) != 4 or fields[1] not in {"t", "T"}:
            continue
        name, _, _, size = fields
        size = int(size, 16)
        total_bytes += size
        if CAMPAIGN not in name:
            continue
        campaign_functions += 1
        campaign_bytes += size
        widths.update(set(map(int, WIDTH.findall(name))))
        verify_widths.update(map(int, VERIFY.findall(name)))
    report = {
        "archive": str(archive),
        "expected_widths": sorted(expected),
        "campaign_widths": sorted(widths),
        "closure_verifier_widths": sorted(verify_widths),
        "functions_by_width": dict(sorted(widths.items())),
        "campaign_functions": campaign_functions,
        "campaign_function_bytes": campaign_bytes,
        "total_function_bytes": total_bytes,
    }
    if set(widths) != expected or verify_widths != expected:
        raise SystemExit("Campaign capacity audit failed:\n" + json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--llvm-nm", default="llvm-nm")
    parser.add_argument("--expected-widths", default="4,8,16")
    args = parser.parse_args()
    expected = {int(width) for width in args.expected_widths.split(",")}
    if not expected or any(width < 1 for width in expected):
        parser.error("expected widths must be positive")
    print(json.dumps(audit(args.archive, args.llvm_nm, expected), indent=2))


if __name__ == "__main__":
    main()
