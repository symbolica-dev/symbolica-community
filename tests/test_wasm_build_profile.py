"""Build-command checks with a fake Pyodide driver; no compilation is performed."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("profile, link_level, debug", [
    ("dev", "-O0", "0"),
    ("release-performance", "-O3", None),
    ("release-small", "-O3", None),
])
def test_wasm_profile_controls_linking_and_debug_sections(tmp_path, profile, link_level, debug):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    sdk = tmp_path / "sdk"
    (sdk / "upstream/bin").mkdir(parents=True)
    (sdk / "upstream/bin/wasm-opt").write_text("test-only binary placeholder")
    driver = binaries / "pyodide"
    driver.write_text(f"#!{sys.executable}\n" + """
import json
import os
from pathlib import Path
import sys
if sys.argv[1:] == ['config', 'get', 'emsdk_dir']:
    print(os.environ['TEST_PYODIDE_SDK'])
elif sys.argv[1] == 'build':
    Path(os.environ['TEST_PYODIDE_RECORD']).write_text(json.dumps({
        'arguments': sys.argv[1:],
        'debug': os.environ.get('CARGO_PROFILE_DEV_DEBUG'),
        'binaryen': os.environ['WASM_OPT_FORCE_LEVEL'],
    }))
else:
    raise AssertionError(sys.argv)
""")
    driver.chmod(0o755)
    record = tmp_path / "command.json"
    env = dict(os.environ)
    env.pop("CARGO_PROFILE_DEV_DEBUG", None)
    env.update({
        "PATH": str(binaries) + os.pathsep + env.get("PATH", ""),
        "TEST_PYODIDE_SDK": str(sdk), "TEST_PYODIDE_RECORD": str(record),
        "WASM_RUST_PROFILE": profile, "WASM_OPT_LEVEL": "-O0",
        "WASM_FEATURES": "wasm", "WASM_SKIP_TESTS": "1",
        "HOST_CC": sys.executable, "HOST_AR": sys.executable,
    })
    subprocess.run(
        ["bash", str(ROOT / "scripts/build_wasm_performance.sh"), str(tmp_path / "output")],
        check=True, capture_output=True, text=True, env=env, timeout=10,
    )
    build = json.loads(record.read_text())
    configuration = build["arguments"][build["arguments"].index("-C") + 1]
    assert f"--profile {profile}" in configuration
    assert f"link-arg={link_level}" in configuration
    assert "--locked" in configuration and "--no-default-features --features wasm" in configuration
    assert build["debug"] == debug
    assert build["binaryen"] == "-O0"
