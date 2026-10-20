"""Core-only packaging must remove wrappers without changing retained payloads."""

import base64
import csv
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile


def test_core_wheel_preserves_contents_and_record(tmp_path):
    wheel = tmp_path / "symbolica-3.0.0-cp314-abi3-pyemscripten_2026_0_wasm32.whl"
    record = "symbolica-3.0.0.dist-info/RECORD"
    files = {
        "symbolica/__init__.py": b"from symbolica.core import *\n",
        "symbolica/core.abi3.so": b"\0asm\x01\0\0\0",
        "symbolica/core.pyi": b"def get_citations(): ...\n",
        "symbolica/py.typed": b"",
        "symbolica/community/__init__.py": b"",
        "symbolica/community/tensor/__init__.pyi": b"class Tensor: ...\n",
        "symbolica-3.0.0.dist-info/METADATA": b"Name: symbolica\nVersion: 3.0.0\n",
        "symbolica-3.0.0.dist-info/WHEEL": b"Wheel-Version: 1.0\n",
    }
    contents = io.StringIO(newline="")
    writer = csv.writer(contents, lineterminator="\n")
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((record, "", ""))
    with ZipFile(wheel, "w") as archive:
        archive.comment = b"preserve archive metadata"
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(record, contents.getvalue())

    script = Path(__file__).resolve().parents[1] / "scripts/prepare_core_wheel.py"
    subprocess.run([sys.executable, script, wheel], check=True)
    with ZipFile(wheel) as archive:
        expected = {name: data for name, data in files.items()
                    if not name.startswith("symbolica/community/")}
        assert set(archive.namelist()) == {*expected, record}
        assert archive.comment == b"preserve archive metadata"
        assert all(info.compress_type == ZIP_DEFLATED for info in archive.infolist())
        for name, data in expected.items():
            assert archive.read(name) == data
        rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
        assert {row[0] for row in rows} == set(archive.namelist())
        for name, digest, size in rows:
            if name != record:
                data = archive.read(name)
                actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
                assert digest == "sha256=" + actual
                assert int(size) == len(data)
