"""The definite integrator shares HEPkit's kernel and preserves its IBP API."""
import subprocess
import sys


def test_shared_expression_contract():
    from .integration_contract import check_integration_contract
    check_integration_contract()


def test_hepkit_symanzik_integral():
    from .integration_contract import check_symanzik_example
    check_symanzik_example()


def test_existing_ibp_api():
    from symbolica.community.hepkit import integration, ibp, IBPFamily
    assert ibp.IBPFamily is IBPFamily
    assert not hasattr(integration, "ibp")


def test_citations_and_import_order():
    code = """
from symbolica import E, S, get_citations
before = {c.id for c in get_citations()}
from symbolica.community.hepkit import integration as api
assert {c.id for c in get_citations()} == before
x = S('citation_integral_x')
assert api.integrate(1/(x+1)**2, [x]) == E('1')
assert 'https://github.com/benruijl/hyperbolica' in {c.id for c in get_citations()}
assert api.integrate.__module__ == 'symbolica.community.hepkit.integration'
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_expression_save_load_fresh_process(tmp_path):
    payload = tmp_path / "expression.bin"
    writer = """
import sys
from symbolica import S
from symbolica.community.hepkit import integration
x = S('integration_pickle::x')
assert integration.integrate(1/(x+1)**2, [x]) == 1
((x+1)**3).save(sys.argv[1])
"""
    reader = """
import sys
from symbolica import S, Expression
for i in range(32): S(f'perturbed_symbol_{i}')
value = Expression.load(sys.argv[1])
assert type(value) is Expression
assert value == (S('integration_pickle::x')+1)**3
"""
    for code in (writer, reader):
        result = subprocess.run([sys.executable, '-c', code, str(payload)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
