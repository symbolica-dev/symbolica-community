# One-loop gluon propagator and UV expansion

```sh
.venv/bin/python -m marimo edit examples/hep_showcase.py --watch
```

All symbols, contractions, projectors, routing functions, and UV helpers are
inlined in the notebook. It needs only the adjacent `hep_sm.json` model;
there is no `hep_gluon.py` import.

The numerator, contraction, routing, projector, scalar-product, and coupling
helpers return `TensorExpression`. Projection leaves a rank-zero
`TensorExpression`; `invariants` checks that it is scalar before explicitly
converting to `Expression` for the UV series. The indexed UV input is restored
as a `TensorExpression` for FeynKit's reducer.

The contraction helper uses Idenso's expression functions internally and
rebuilds the validated tensor interface afterward. It evaluates the existing
SU(3) factors before that conversion because this upstream revision misreads
symbolic color Casimirs as tensor ports. The projector contracts all four
explicitly named external slots; ordinary typed multiplication rejects the
multiple possible port pairings as ambiguous.

Generation uses the current keyword API:

```python
diagrams = model.generate_diagrams(
    [g], [g], loops=1,
    coupling_orders={"QCD": 2, "QED": 0},
    particle_veto=["c", "t", "s", "u", "d"],
    zero_snails=SnailFilterOptions(),
)
```

`GenerationOptions` has been removed. Both `Model.generate_diagrams` and
`Generator.generate` accept the same filter keywords; excluded processes
return an empty `GenerationResult`.

The default selection keeps the gluon, ghost, and bottom-quark bubbles.
Lorentz indices are promoted to symbolic `D` before Dirac/color/metric
contraction. The transverse, color-averaged projector is

```text
P_T = delta_ab * (g_mu_nu - p_mu*p_nu/p^2) / (8*(D-1)).
```

It extracts `p^2 Pi_T`. Keep the external momentum off shell, `p^2 != 0`.

## UV expansion on an expression copy

The main notebook constructs the full projected integrand, including both
propagators. `uv_expression_copy = copy(projected_integrand)` creates the
expression branch to expand. The original expression remains available.

The UV scaling acts directly on the loop vector:

```python
t = S("hep_gluon::t", is_scalar=True)
uv_scaled_expression = uv_expression_copy.replace(K(mink(D)), K(mink(D)) / t)
```

The scalar attribute makes dot normalization pull `1/t` out of each argument.
`k2` and `kp` are Python aliases for `dot(K,K)` and `dot(K,P)` expressions;
there are no scalar placeholder substitutions for either loop-momentum dot.
The `series(t, 0, 4)` cell keeps the quadratic, linear, and logarithmic UV
orders. It expands the numerator and denominators together, holding external
momenta and physical masses fixed. No graph copy or Feynman-parameter shift
is used in this workflow.

The resulting vacuum-denominator terms are passed directly to FeynKit:

```python
tensor_reducer = fk.TensorReducer(D).with_integrated_vector(UV_K(mink(D)))
uv_tensor_reduced = tensor_reducer.reduce(uv_indexed_series)
```

Only the loop momentum is integrated. `uv_tensor_input` converts powers of
the mixed dot `dot(K,P)` to distinct indexed vector contractions; it does
not compute reduction coefficients. This adapter is needed because this
revision treats compact mixed dots as scalar coefficients, and its
`expand_dots` API requires concrete dimensions. FeynKit computes all tensor
reduction coefficients and vanishing odd-rank terms. `uv_scalar_invariants`
then restores the original loop-vector name in `dot(K,K)` and replaces the
external dot with `p2`.

After angular averaging, write the UV integrand as `A/k2+B/k2^2`.
The infrared-regulated, angular-averaged subtraction is

```text
C_UV = -A/(k2-Muv2) - (B-A*Muv2)/(k2-Muv2)^2.
```

The compensation term preserves the large-k expansion and cancels the
auxiliary mass from the integrated pole. This is an angular-averaged
subtraction, not a pointwise tensor subtraction. The massless scaleless
expansion must not simply be integrated as zero.

With conventional dimensional regularization `D=4-2 eps`, `tr(1)=4`, Feynman
gauge, and the `d^D k/(2*pi)^D` measure, the integrated MS counterterm is
`-i*B(D=4)/(16*pi^2*eps)`. Finite parts are not calculated.

| Loop | Transverse residue / (G² p²) | Longitudinal residue / (G² p²) |
| --- | ---: | ---: |
| Ghost | 1/4 | 3/4 |
| Gluon | 19/4 | -3/4 |
| Bottom | -2/3 | 0 |
| Sum | 13/3 | 0 |

The bottom mass cancels from the residue. The result is the gluon-field
coefficient `5 C_A/3 - 4 T_F n_f/3` at `C_A=3`, `T_F=1/2`, `n_f=1`, not
the beta-function coefficient. See [hep-ph/9701375](https://cds.cern.ch/record/319039/files/9701375.pdf).

`hep_showcase_uv.py` remains an optional independent Feynman-parameter
cross-check; its code is also inlined. The main notebook contains the
requested expression-copy UV expansion.

## Current environment

The Linux `.venv` uses marimo 0.24.2, ty, and FeynKit branch revision
`0ea2140565027f7541293396e9053cd67c4c0a1e`. HEP type hints were regenerated.

This upstream revision includes the `FeynKit::Momentum` printer-registration
fix. The running build uses upstream crates directly, with no local override.
The previous checkout and patch remain in `.venv` as historical backups.

To rebuild this environment:

```sh
VIRTUAL_ENV="$PWD/.venv" .venv/bin/maturin develop --locked
```

Two other API details are explicit in the notebook: external momentum signs
are converted from physical in/out conventions to edge orientations, with
vertex conservation checked; and the ghost-loop Grassmann minus is supplied
because the generator's `is_fermion()` loop count excludes spin -1 ghosts.

## Verification

```sh
.venv/bin/python -m pytest tests/test_hep_gluon_example.py tests/test_hep.py -q
.venv/bin/python -m marimo check --strict examples/hep_showcase.py examples/hep_showcase_uv.py
.venv/bin/python examples/hep_showcase.py
.venv/bin/python examples/hep_showcase_uv.py
```

Tests run the actual inlined notebook code and check expression-copy
preservation, absence of external indices after projection, native tensor
reduction through rank six, per-loop UV residues, regulator independence, cancellation
of the asymptotic terms, transversality, and unsupported-topology rejection.
