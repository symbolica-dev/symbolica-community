"""Check the installed integration package layout and shared Cargo kernel."""
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--locked", "--format-version", "1"], text=True))
resolved = {node["id"] for node in metadata["resolve"]["nodes"]}
for name in ("symbolica", "numerica", "graphica", "pyo3"):
    matches = [p for p in metadata["packages"] if p["name"] == name and p["id"] in resolved]
    assert len(matches) == 1, (name, [p["id"] for p in matches])
for node in metadata["resolve"]["nodes"]:
    package = next(p for p in metadata["packages"] if p["id"] == node["id"])
    if package["name"] == "symbolica":
        assert "faster_alloc" not in node["features"], "host allocator policy changed"
for wheel in sys.argv[1:]:
    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        base = "symbolica/community/hepkit/integration/"
        assert {base+"__init__.py", base+"__init__.pyi", "symbolica/py.typed"} <= names
        source = archive.read(base+"__init__.pyi").decode()
        assert "class IntegrationOptions" in source and "class IntegrationError" in source
        assert "class Expression" not in source
        assert not any(n.startswith("hyperbolica/") for n in names)
print("Integration wheel and shared-kernel checks passed")
