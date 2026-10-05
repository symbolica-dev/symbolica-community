from symbolica import E, S, Expression
from symbolica.community.hepkit import integration
from symbolica.community.hepkit.integration import ibp

x = S("x")
options = integration.IntegrationOptions(parallel=False)
prepared: integration.PreparedIntegral = integration.prepare(1/(x+1)**2, [x], options)
answer: Expression = prepared.integrate()
detailed: integration.IntegrationResult = integration.integrate_detailed(1/(x+1)**2, [x])
finite: Expression = integration.integrate_over(E("1"), [x], [(E("0"), E("1"))])
family_type = ibp.IBPFamily
