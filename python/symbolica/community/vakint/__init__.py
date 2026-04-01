"""
# Vakint

This python library aims at the analytical and numerical evaluation of single-scale vacuum integrals in Quantum Field Theory.
It uses combines FORM scripts from existing works with Symbolica's numerical capabilities to directly deliver arbitrary precision evaluations of such integrals.

For questions or bug reports, visit https://github.com/alphal00p/vakint/.

# Citables
- vakint : https://github.com/alphal00p/gammaloop/tree/main/crates/vakint
- MATAD  : https://arxiv.org/pdf/hep-ph/0009029
- FMFT   : https://arxiv.org/pdf/1707.01710

# Contributors
- Valentin Hirschi valentin.hirschi@gmail.com

# Example:
```
from symbolica.community.vakint import Vakint, VakintEvaluationMethod, VakintExpression, VakintNumericalResult
from symbolica import E, S

masses = {"muvsq": 2., "mursq": 3.}
external_momenta = {
    1: (0.1, 0.2, 0.3, 0.4),
    2: (0.5, 0.6, 0.7, 0.8)
}

# Use vakint defaults with:
# vakint = Vakint()
# Or specify more options:
vakint = Vakint(
    integral_normalization_factor="MSbar",
    mu_r_sq_symbol=S("mursq"),
    # If you select 5 terms, then MATAD will be used, but for 4 and fewer, alphaLoop is will be used as
    # it is first in the evaluation_order supplied.
    number_of_terms_in_epsilon_expansion=4,
    evaluation_order=[
        VakintEvaluationMethod.new_alphaloop_method(),
        VakintEvaluationMethod.new_matad_method(),
        VakintEvaluationMethod.new_fmft_method(),
        # VakintEvaluationMethod.new_pysecdec_method(
        #     min_n_evals=10_000,
        #     max_n_evals=1000_000,
        #     numerical_masses=masses,
        #     numerical_external_momenta=external_momenta
        # ),
    ],
    form_exe_path="form",
    python_exe_path="python3",
)

integral = E(\"\"\"
        (
            k(1,11)*k(2,11)*k(1,22)*k(2,22)
          + p(1,11)*k(3,11)*k(3,22)*p(2,22)
          + p(1,11)*p(2,11)*(k(2,22)+k(1,22))*k(2,22)
        )*topo(
             prop(1,edge(1,2),k(1),muvsq,1)
            * prop(2,edge(2,3),k(2),muvsq,1)
            * prop(3,edge(3,1),k(3),muvsq,1)
            * prop(4,edge(1,4),k(3)-k(1),muvsq,1)
            * prop(5,edge(2,4),k(1)-k(2),muvsq,1)
            * prop(6,edge(3,4),k(2)-k(3),muvsq,1)
)\"\"\", default_namespace="vakint")
print(f"\nStarting integral:\n{VakintExpression(integral)}")

canonical_integral = vakint.to_canonical(integral, short_form=True)
print(f"\nCanonical integral:\n{VakintExpression(canonical_integral)}")

tensor_reduced_integral = vakint.tensor_reduce(canonical_integral)
print(f"\nTensor reduced integral:\n{
      VakintExpression(tensor_reduced_integral)}")

evaluated_integral = vakint.evaluate_integral(tensor_reduced_integral)
print(f"\nEvaluated integral:\n{evaluated_integral}")

# Direct evaluation all at once
direct_evaluation = vakint.evaluate(integral)

assert direct_evaluation == evaluated_integral

num_eval, num_error = vakint.numerical_evaluation(
    evaluated_integral, params=masses, externals=external_momenta)

print(f"\nNumerical evaluation:\n{num_eval}")

print(f"\nNumerical evaluation as list:\n{num_eval.to_list()}")

print(f"\nNumerical evaluation, as expression:\n{vakint.numerical_result_to_expression(num_eval)}")

benchmark = VakintNumericalResult([
    (-3, (0.0, -4230.112451731339)),
    (-2, (0.0,  22000.89349621258)),
    (-1, (0.0, -64562.68098612698)),
    (-0, (0.0,  41560.11791065207)),
])

match_res, match_msg = benchmark.compare_to(
    num_eval, relative_threshold=1.0e-10
)

print(f"\nMatch result: {match_res}, {match_msg}")

assert match_res
```

"""

from ..vakint_native import *

initialize_module()
