"""Process-wide citations activate on operations, never on module import."""
import os
import subprocess
import sys


def test_citations_track_native_feature_usage():
    script = """
from tests._license import configure_license_key
configure_license_key()
from symbolica import E, S, get_citations
from symbolica.community import hepkit as hep
from symbolica.community import tensor
from symbolica.community.hepkit import oneloop

def ids():
    citations = get_citations()
    assert len(citations) == len({c.id for c in citations})
    assert all(c.reference and c.reasons and c.to_bibtex().startswith("@") for c in citations)
    return {c.id for c in citations}

symbolica = "doi:10.5281/zenodo.17054381"
assert ids() == {symbolica}, ids()
for module in (hep, tensor, oneloop):
    assert not hasattr(module, "get_citations")
assert "https://github.com/alphal00p/oneloopmaster" not in ids()
x = S("citation_x")
E("citation_x").integrate(x)
assert "https://github.com/symbolica-dev/symbolica-integrate" in ids()
assert "https://rulebasedintegration.org" in ids()
d, k, p, s = S("citation_d", "citation_k", "citation_p", "citation_s")
kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
assert "https://github.com/alphal00p/gammaloop#feynkit" in ids()
assert "arXiv:2411.02233" not in ids()
family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k), kin.scalar_product(k-p, k-p)], kinematics=kin)
assert "https://github.com/ecavan/one-loop-reduce" not in ids()
reduction = oneloop.reduce(family, [1, 1])
assert "https://github.com/ecavan/one-loop-reduce" in ids()
reduction.to_expression()
expected = {"https://github.com/alphal00p/oneloopmaster", "arXiv:1007.4716", "arXiv:0903.4665"}
assert expected <= ids(), ids()
assert ids() == ids()

tensor.TensorExpression(E("1"))
assert {"10.5281/zenodo.18248388", "10.5281/zenodo.18248409"} <= ids()
hep.IBPFamily(family)
assert "https://github.com/alphal00p/rustred" in ids()
import importlib
vakint = importlib.import_module("symbolica.community.hepkit.vakint")
assert "https://github.com/alphal00p/vakint#vakint" not in ids()
vakint.VakintExpression(E("0"))
assert "https://github.com/alphal00p/vakint#vakint" in ids()
assert not {"arXiv:1203.6543", "arXiv:hep-ph/0009029", "arXiv:1707.01710", "arXiv:1703.09692"} & ids()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
