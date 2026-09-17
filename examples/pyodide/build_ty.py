#!/usr/bin/env python3
"""Build the pinned ty WASM API used by the playground's hover worker."""

import os
from pathlib import Path
import shutil
import subprocess

REVISION = "72b1fd3cefed68152f4219886210da425afa63a0"


def main():
    for executable in ("git", "wasm-pack", "rustup"):
        if shutil.which(executable) is None:
            raise SystemExit(f"Install {executable} before building ty.")
    cache = Path.home() / ".cache/symbolica-community-build"
    source = cache / "ty-source"
    if not source.exists():
        source.mkdir(parents=True)
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run([
            "git", "fetch", "--depth", "1", "https://github.com/astral-sh/ruff", REVISION,
        ], cwd=source, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if revision != REVISION:
        raise SystemExit(f"Expected ty source revision {REVISION}, found {revision} in {source}.")
    env = os.environ.copy()
    env.update({
        "RUSTUP_TOOLCHAIN": "stable",
        # Native target-cpu flags inherited from Cargo config break WASM output.
        "CARGO_ENCODED_RUSTFLAGS": "",
        "CARGO_TARGET_DIR": str(cache / "ty-target"),
        "TY_WASM_COMMIT_SHORT_HASH": REVISION[:12],
    })
    subprocess.run([
        "wasm-pack", "build", str(source / "crates/ty_wasm"), "--target", "web",
        "--release", "--no-opt", "--out-dir", str(cache / "ty-wasm"), "--", "--locked",
    ], cwd=source, env=env, check=True)


if __name__ == "__main__":
    main()
