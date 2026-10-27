"""Core packaging must remove optional modules without changing the kernel."""

import base64
import csv
import hashlib
import importlib.util
import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_core_wheel", Path(__file__).parents[1] / "scripts/prepare_core_wheel.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_core_wheel_preserves_kernel_metadata_and_valid_record(tmp_path):
    wheel = tmp_path / "symbolica-3.0.3-cp314-cp314-pyemscripten_2026_0_wasm32.whl"
    retained = {
        "symbolica/__init__.py": b"from .core import *\n",
        "symbolica/core.cpython-314-wasm32-emscripten.so": b"\x00asm kernel-only build",
        "symbolica/core.pyi": b"# optional operand protocols are local\n",
        "symbolica/data.py": b"# resource loader\n",
        "symbolica/py.typed": b"",
        "symbolica-3.0.3.dist-info/METADATA": b"Requires-Dist: symbolica-fonts==3.0.3\n",
        "symbolica-3.0.3.dist-info/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: false\n",
        "symbolica-3.0.3.dist-info/licenses/LICENSE": b"license text",
    }
    record_name = "symbolica-3.0.3.dist-info/RECORD"
    with ZipFile(wheel, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in retained.items():
            archive.writestr(name, payload)
        archive.writestr("symbolica/community/__init__.py", "")
        archive.writestr("symbolica/community/tensor/__init__.pyi", "optional stub")
        archive.writestr("examples/hep/massive_kite.py", "optional example")
        archive.writestr(record_name, "old manifest including Community packages")
    PREPARE.prepare_core_wheel(wheel)
    with ZipFile(wheel) as archive:
        assert set(archive.namelist()) == set(retained) | {record_name}
        for name, payload in retained.items():
            assert archive.read(name) == payload
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode())))
        assert len(rows) == len(archive.namelist())
        assert rows[-1] == [record_name, "", ""]
        for name, digest, size in rows[:-1]:
            payload = archive.read(name)
            expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
            assert digest == "sha256=" + expected.decode()
            assert int(size) == len(payload)
        first_record = archive.read(record_name)
    PREPARE.prepare_core_wheel(wheel)
    with ZipFile(wheel) as archive:
        assert archive.read(record_name) == first_record


def test_invalid_wheel_is_not_replaced(tmp_path):
    wheel = tmp_path / "incomplete.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("symbolica/core.pyi", "stub")
    original = wheel.read_bytes()
    with pytest.raises(ValueError, match="Expected one wheel RECORD"):
        PREPARE.prepare_core_wheel(wheel)
    assert wheel.read_bytes() == original
