"""D=2 unit-mass bubble at p²=0, with the loop-measure prefactor stripped.

HEPkit supplies U,F. We choose propagator powers (1,1) and gauge x2=1.
The convergent parameter integral is integral_0^infinity dx1/F(x1,1) = 1.
No regulator expansion or physical continuation is inferred.
"""
from symbolica import E, S
from symbolica.community import hepkit as hep
from symbolica.community.hepkit import integration

k, p, x1, x2 = S("k", "p", "x1", "x2")
kin = hep.Kinematics(E("2"), momenta=[k, p]).with_scalar_product(p, p, E("0"))
family = hep.IntegralFamily([k], [p], [kin.scalar_product(k,k)-1,
    kin.scalar_product(k-p,k-p)-1], kinematics=kin)
U, F = family.symanzik([x1, x2])
answer = integration.integrate(1/F.replace(x2, E("1")), [x1],
    integration.IntegrationOptions(check_divergences=True))
assert answer == E("1")
print(answer)
