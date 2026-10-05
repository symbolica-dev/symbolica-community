"""Process-wide citations activate on operations, never on module import."""

import os
import subprocess
import sys
import textwrap


def test_rustred_strategy_citation_is_usage_gated_and_unique():
    """Importing the API is not evidence that its IBP strategy was used."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import importlib
            from symbolica import S, get_citations
            from symbolica.community import hepkit as hep

            importlib.import_module("symbolica.community.hepkit.rustred")
            software_id = "https://github.com/alphal00p/rustred"
            paper_id = "arXiv:2604.25916"

            def citations_by_id():
                citations = get_citations()
                assert len(citations) == len({entry.id for entry in citations})
                return {entry.id: entry for entry in citations}

            assert not {software_id, paper_id} & citations_by_id().keys()
            d, k = S("citation_rustred_d", "citation_rustred_k")
            kin = hep.Kinematics(d, momenta=[k])
            family = hep.IntegralFamily(
                [k], [], [kin.scalar_product(k, k) + 1], kinematics=kin,
            )
            assert not {software_id, paper_id} & citations_by_id().keys()
            hep.IBPFamily(family)
            first = citations_by_id()
            assert {software_id, paper_id} <= first.keys()
            paper = first[paper_id]
            assert paper.reference and paper.reasons
            assert "parametric IBP" in " ".join(paper.reasons)
            bibtex = paper.to_bibtex()
            assert bibtex.startswith("@article{Dlapa:2026oyq,")
            assert "10.1103/wkp4-vy6g" in bibtex and "2604.25916" in bibtex
            hep.IBPFamily(family)
            second = citations_by_id()
            assert first.keys() == second.keys()
            assert second[paper_id].reasons == paper.reasons
            assert second[paper_id].to_bibtex() == bibtex
        """),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_citations_track_native_feature_usage():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'from symbolica import E, S, get_citations\nfrom symbolica.community import hepkit as hep\nfrom symbolica.community import tensor\nfrom symbolica.community.hepkit import oneloop\n\ndef ids():\n    citations = get_citations()\n    assert len(citations) == len({c.id for c in citations})\n    assert all(c.reference and c.reasons and c.to_bibtex().startswith("@") for c in citations)\n    return {c.id for c in citations}\n\nsymbolica = "doi:10.5281/zenodo.17054381"\nassert ids() == {symbolica}, ids()\nfor module in (hep, tensor, oneloop):\n    assert not hasattr(module, "get_citations")\nassert "https://github.com/alphal00p/oneloopmaster" not in ids()\nx = S("citation_x")\nE("citation_x").integrate(x)\nassert "https://github.com/symbolica-dev/symbolica-integrate" in ids()\nassert "https://rulebasedintegration.org" in ids()\nd, k, p, s = S("citation_d", "citation_k", "citation_p", "citation_s")\nkin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)\nassert "https://github.com/alphal00p/gammaloop#feynkit" in ids()\nassert "arXiv:2411.02233" not in ids()\nfamily = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k), kin.scalar_product(k-p, k-p)], kinematics=kin)\nassert "https://github.com/ecavan/one-loop-reduce" not in ids()\nreduction = oneloop.reduce(family, [1, 1])\nassert "https://github.com/ecavan/one-loop-reduce" in ids()\nreduction.to_expression()\nexpected = {"https://github.com/alphal00p/oneloopmaster", "arXiv:1007.4716", "arXiv:0903.4665"}\nassert expected <= ids(), ids()\nassert ids() == ids()\n\ntensor.TensorExpression(E("1"))\nassert {"10.5281/zenodo.18248388", "10.5281/zenodo.18248409"} <= ids()\nhep.IBPFamily(family)\nassert "https://github.com/alphal00p/rustred" in ids()\nimport importlib\nvakint = importlib.import_module("symbolica.community.hepkit.vakint")\nassert "https://github.com/alphal00p/vakint#vakint" not in ids()\nvakint.VakintExpression(E("0"))\nassert "https://github.com/alphal00p/vakint#vakint" in ids()\nassert not {"arXiv:1203.6543", "arXiv:hep-ph/0009029", "arXiv:1707.01710", "arXiv:1703.09692"} & ids()\n',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
