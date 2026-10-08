"""Process-wide citations activate on operations, never on module import."""

import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("operation", ["import", "ordinary", "kinematic", "higgs", "model", "projector", "automatic"])
def test_numerical_transport_citations_are_usage_gated_and_cumulative(operation):
    """Each case starts with a fresh native registry; no test-only reset exists."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            from symbolica import E, S, get_citations
            from symbolica.community import hepkit as hep
            from symbolica.community.hep import integration
            operation = sys.argv[1]
            method_ids = {'arXiv:2607.08477', 'arXiv:2006.05510'}
            amflow_id, higgs_id = 'arXiv:2201.11669', 'arXiv:2112.07578'
            all_ids = method_ids | {amflow_id, higgs_id}
            def citations():
                values = get_citations()
                assert len(values) == len({c.id for c in values})
                return {c.id:c for c in values}
            assert not all_ids & citations().keys()
            assert not hasattr(integration, 'get_citations')
            integration.EvaluationOptions()
            integration.BoundaryCache()
            try:
                integration.HiggsJetIntegralSystem('invalid')
                raise AssertionError('invalid Higgs topology accepted')
            except integration.InvalidInputError:
                pass
            assert not all_ids & citations().keys()
            expected = set()
            if operation == 'ordinary':
                integration.DifferentialSystem(S('citation_x'), [[E('0')]])
                expected = set(method_ids)
            elif operation == 'kinematic':
                integration.KinematicTransport(S('citation_eps'), {S('citation_x'):[[E('0')]]},
                    [S('citation_I')], E('1'), branch_domain='citation preparation')
                expected = set(method_ids)
            elif operation == 'higgs':
                integration.HiggsJetIntegralSystem('planar')
                expected = method_ids | {higgs_id}
            elif operation == 'model':
                integration.HiggsJetAmplitude.with_form_factor_vertices(hep.Model.standard_model())
                expected = {higgs_id}
            elif operation == 'projector':
                integration.HiggsJetFormFactorProjector()
                expected = {higgs_id}
            elif operation == 'automatic':
                if integration.automatic_boundary_generation_available:
                    integration.IntegralEvaluator()
                    expected = method_ids | {amflow_id}
            first = citations()
            assert all_ids & first.keys() == expected
            for identifier in expected:
                citation = first[identifier]
                assert citation.reference and citation.description and citation.reasons
                assert citation.to_bibtex().startswith('@article{')
                assert identifier.split(':')[1] in citation.to_bibtex()
            if method_ids <= expected:
                methods = [first[identifier] for identifier in sorted(method_ids)]
                assert methods[0].description != methods[1].description
                assert methods[0].reasons != methods[1].reasons
            second = citations()
            assert first.keys() == second.keys()
            assert all(second[k].reasons == first[k].reasons for k in expected)
            if operation != 'import':
                integration.HiggsJetAmplitude.with_form_factor_vertices(hep.Model.standard_model())
                accumulated = citations()
                assert all_ids & accumulated.keys() == expected | {higgs_id}
                for identifier in expected:
                    assert set(first[identifier].reasons) <= set(accumulated[identifier].reasons)
                integration.DifferentialSystem(S('citation_second_x'), [[E('0')]])
                expected |= method_ids | {higgs_id}
                assert all_ids & citations().keys() == expected
                if integration.automatic_boundary_generation_available:
                    integration.IntegralEvaluator()
                    expected |= {amflow_id}
                accumulated = citations()
                assert all_ids & accumulated.keys() == expected
                for identifier in first.keys() & all_ids:
                    assert set(first[identifier].reasons) <= set(accumulated[identifier].reasons)
        """), operation],
        capture_output=True,
        text=True,
        timeout=90,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
            litered_id = "https://github.com/rnlg/LiteRed2"
            rustred_ids = {software_id, paper_id, litered_id}

            def citations_by_id():
                citations = get_citations()
                assert len(citations) == len({entry.id for entry in citations})
                return {entry.id: entry for entry in citations}

            assert not rustred_ids & citations_by_id().keys()
            d, k = S("citation_rustred_d", "citation_rustred_k")
            kin = hep.Kinematics(d, momenta=[k])
            family = hep.IntegralFamily(
                [k], [], [kin.scalar_product(k, k) + 1], kinematics=kin,
            )
            assert not rustred_ids & citations_by_id().keys()
            hep.IBPFamily(family)
            first = citations_by_id()
            assert rustred_ids <= first.keys()
            litered = first[litered_id]
            assert "Roman N. Lee" in litered.reference and "LiteRed2" in litered.reference
            assert litered.description and litered.reasons
            assert "symbolic IBP rules" in " ".join(litered.reasons)
            assert "applicability conditions" in " ".join(litered.reasons)
            litered_bibtex = litered.to_bibtex()
            assert litered_bibtex.startswith("@software{Lee:LiteRed2,")
            assert litered_id in litered_bibtex
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
            assert second[litered_id].reasons == litered.reasons
            assert second[litered_id].to_bibtex() == litered_bibtex
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
            'from symbolica import E, S, get_citations\nfrom symbolica.community import hepkit as hep\nfrom symbolica.community import tensor\nfrom symbolica.community.hepkit import oneloop\n\ndef ids():\n    citations = get_citations()\n    assert len(citations) == len({c.id for c in citations})\n    assert all(c.reference and c.reasons and c.to_bibtex().startswith("@") for c in citations)\n    return {c.id for c in citations}\n\nsymbolica = "doi:10.5281/zenodo.17054381"\nassert ids() == {symbolica}, ids()\nfor module in (hep, tensor, oneloop):\n    assert not hasattr(module, "get_citations")\nassert "https://github.com/alphal00p/oneloopmaster" not in ids()\nx = S("citation_x")\nE("citation_x").integrate(x)\nassert "https://github.com/symbolica-dev/symbolica-integrate" in ids()\nassert "https://rulebasedintegration.org" in ids()\nd, k, p, s = S("citation_d", "citation_k", "citation_p", "citation_s")\nkin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)\nassert "https://github.com/alphal00p/gammaloop#feynkit" in ids()\nassert "arXiv:2411.02233" not in ids()\nfamily = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k), kin.scalar_product(k-p, k-p)], kinematics=kin)\nassert "https://github.com/ecavan/one-loop-reduce" not in ids()\nreduction = oneloop.reduce(family, [1, 1])\nassert "https://github.com/ecavan/one-loop-reduce" in ids()\nreduction.to_expression()\nexpected = {"https://github.com/alphal00p/oneloopmaster", "arXiv:1007.4716", "arXiv:0903.4665"}\nassert expected <= ids(), ids()\nassert ids() == ids()\n\ntensor_value = tensor.TensorExpression(E("1"))\nassert "10.5281/zenodo.18248388" in ids()\nassert "10.5281/zenodo.18248409" not in ids()\ntensor_value.contract()\nassert {"10.5281/zenodo.18248388", "10.5281/zenodo.18248409"} <= ids()\nhep.IBPFamily(family)\nassert "https://github.com/alphal00p/rustred" in ids()\nimport importlib\nvakint = importlib.import_module("symbolica.community.hepkit.vakint")\nassert "https://github.com/alphal00p/vakint#vakint" not in ids()\nvakint.VakintExpression(E("0"))\nassert "https://github.com/alphal00p/vakint#vakint" in ids()\nassert not {"arXiv:1203.6543", "arXiv:hep-ph/0009029", "arXiv:1707.01710", "arXiv:1703.09692"} & ids()\n',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
