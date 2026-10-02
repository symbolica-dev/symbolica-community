"""Keep canonical core declarations and the community citation API together."""

import runpy
from pathlib import Path
from zipfile import ZipFile

import pytest

check_core_stub = runpy.run_path(
    str(Path(__file__).parents[1] / ".github/scripts/check_core_stub.py")
)["check_core_stub"]

CANONICAL = """class Citation:
    @property
    def id(self) -> str: ...

class Expression:
    def __getitem__(self, index: int, /) -> Expression: ...
"""
COMMUNITY = "\ndef get_citations() -> list[Citation]: ...\n"


@pytest.mark.parametrize("wheel", [False, True])
def test_canonical_core_with_community_citations(tmp_path, wheel):
    canonical = tmp_path / "symbolica.pyi"
    canonical.write_text(CANONICAL)
    path = tmp_path / ("symbolica.whl" if wheel else "core.pyi")
    if wheel:
        with ZipFile(path, "w") as archive:
            archive.writestr("symbolica/core.pyi", CANONICAL + COMMUNITY)
    else:
        path.write_text(CANONICAL + COMMUNITY)
    check_core_stub(path, canonical)


@pytest.mark.parametrize(
    "source",
    [
        COMMUNITY,
        CANONICAL,
        CANONICAL + "\nclass Citation: ...\n" + COMMUNITY,
        CANONICAL + COMMUNITY + COMMUNITY,
        CANONICAL + "\ndef get_citations(reset: bool = False) -> list[Citation]: ...\n",
        CANONICAL + "\ndef get_citations() -> list[str]: ...\n",
    ],
)
def test_missing_duplicate_or_incorrect_citation_api_is_rejected(tmp_path, source):
    path = tmp_path / "core.pyi"
    path.write_text(source)
    with pytest.raises(SystemExit, match="Citation|get_citations"):
        check_core_stub(path)


@pytest.mark.parametrize(
    "source",
    [
        CANONICAL.replace("index: int", "index: str") + COMMUNITY,
        CANONICAL + COMMUNITY + "\ndef extra_core_api() -> None: ...\n",
    ],
)
def test_canonical_declaration_drift_is_rejected(tmp_path, source):
    canonical = tmp_path / "symbolica.pyi"
    canonical.write_text(CANONICAL)
    path = tmp_path / "core.pyi"
    path.write_text(source)
    with pytest.raises(SystemExit, match="differs from"):
        check_core_stub(path, canonical)
