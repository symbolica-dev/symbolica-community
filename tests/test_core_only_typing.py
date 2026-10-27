"""The shared arithmetic stub must not require optional Community modules."""

import shutil
import subprocess
import sys
from pathlib import Path


def test_core_stub_types_without_community(tmp_path):
    root = Path(__file__).parents[1]
    package = tmp_path / "symbolica"
    package.mkdir()
    (package / "__init__.pyi").write_text("from .core import *\n")
    shutil.copyfile(root / "python/symbolica/core.pyi", package / "core.pyi")
    probe = tmp_path / "core_arithmetic.py"
    shutil.copyfile(root / "tests/typing/core_arithmetic.py", probe)

    executable = Path(sys.executable).with_name("ty")
    if not executable.is_file():
        executable = shutil.which("ty")
    assert executable, "Install the project's ty development dependency."
    result = subprocess.run(
        [
            str(executable),
            "check",
            "--project",
            str(tmp_path),
            "--python",
            sys.executable,
            "--python-version",
            "3.9",
            str(probe),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
