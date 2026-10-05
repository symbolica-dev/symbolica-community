# HEP bindings

The GammaLoop `feynkit` branch is bundled through `feynkit-py`. Its public API
is flat under `symbolica.community.hepkit`, sharing the same Symbolica kernel as
Spenso and Idenso:

```python
from symbolica.community.hepkit import FeynmanDiagram, Model, TensorReducer
```

| Component | Examples of public classes |
| --- | --- |
| `feynkit-graph` | `FeynmanDiagram`, `DiagramEdge`, `DiagramVertex`, `LoopMomentumBasis` |
| `feynkit-generator` | `Process`, `GenerationResult`, `GenerationProgress` |
| `feynkit-cff` | `CffGenerator`, `CffResult`, `CffSurface`, `CffOrientation` |
| `feynkit-tensor` | `TensorReducer`; reduction methods also live on `FeynmanDiagram` |
| `feynkit-py` | All of the above, plus models, UFO loading, kinematics, and jet clustering |

Class identities and the exception hierarchy are preserved; their Python
`__module__` and the bundled stubs name `symbolica.community.hepkit`.

Start with [gamma algebra](gamma_simplification.py), then
[color algebra](color_algebra.py) and [tensor reduction](tensor_reduction.py).
[Higgs diphoton decay](Higgs_diphoton_decay.py) follows a generated fermion loop
through symbolic reduction, native master evaluation, and a complete CFF Monte
Carlo decay-width calculation, with separate numerator and mass-shift
derivations of the R₂ and R₁ rational terms and typeset physical results.
[Tensor benchmarks](tensor_benchmarks.py) keep timing experiments, component
oracles, and optional FORM runs separate from the introductory examples.

Two generated four-loop propagator examples reduce three-rung ladders to dot
products: the [all-gluon ladder](three_gluon_rung_ladder.py) and the
[outer quark-loop ladder](quark_loop_ladder.py). Both select their topology with
the graph generator, retain its Feynman rules and weights, and project with
`g_mu_nu delta_ab/8` in Feynman gauge with symbolic Lorentz dimension. The quark
example uses one massless flavor. Each notebook displays the generated graph,
the 15 scalar-product coordinates, and downloadable full numerators, keeping
propagator denominators separate; these examples do not integrate the loops.

The tensor examples use `symbolica.community.tensor` consistently:

1. Construct typed tensors and declare their representations and dimensions.
2. Use `simplify_algebra(AlgebraSettings(...))` for explicitly enabled identity
   families, such as `gamma=GammaSimplifySettings()` or
   `color=ColorSimplifySettings()`.
3. Use `contract(ContractSettings(...))` for structural contractions and
   requested chain or trace notation. Collecting a trace does not evaluate it.
4. Use `to_dots()` only to normalize existing scalar-product notation. Convert
   to a scalar `Expression` before scalar substitutions, series, or integration.
5. Check the result against an exact identity or an independent reference.

Reduction returns an alias-owning result; its `to_expression()` materializes a
`TensorExpression`, whose own `to_expression()` exposes the underlying scalar
expression once all tensor ports have been contracted. Arithmetic expansion is
an explicit step in examples that need polynomial coefficients. Neither settings
type implicitly enables unrelated algebraic identities.

The Marimo directory server (`marimo edit examples/` from the repository root)
provides separate worked IBP notebooks:

- [One-loop reduction and master evaluation](oneloop_reduce.py): pass a shared
  `hepkit.IntegralFamily` and explicit powers to `hepkit.oneloop.reduce` for primitive
  `oneloopmaster::` symbols, exported as `oneloop.A0`, `B0`, `C0`, and `D0`
  and evaluated through their native Rust hooks.
  Interactive examples cover a triangle numerator, an exact dilogarithmic C0
  expression with `get_expression` and `select_branch`, and a squared tadpole
  whose finite term includes epsilon-dependent prefactors.
- [Unequal-mass bubble](ibp_bubble.py): raised powers and numerical master evaluation.
- [Bubble differential equations](ibp_differential_equations.py): compatible equations
  in the invariant and masses, with numerical transport from a boundary value.
- [Massless form-factor triangle](ibp_triangle.py): raised powers and infrared poles.
- [Massive two-loop 2- and 3-point functions](ibp_two_loop_masses.py): symbolic-index
  recurrences with independent masses, application to raised powers and linear
  combinations, and exact checks of the defining IBP identities.
- [Two-loop phi4 self-energy](ibp_phi4.py) and [vertex](phi4_two_loop_vertex.py):
  generated diagrams, reductions and counterterms.

All these examples use the shared `hepkit.IntegralFamily` frontend with
`hepkit.Kinematics` and a symbolic dimension. The one-loop example chooses
`hepkit.oneloop.reduce` for reduction to OneLoopMaster symbols; `hepkit.IBPFamily`
provides native RustRed reductions of the same families. Links inside each
notebook stay on the same server.

## Numerical transport and supplied reductions

Native numerical loop evaluation lives under
`symbolica.community.hep.integration`. The [Higgs-jet notebook](gg_hg.py) uses
these classes with the same HEPKit model, families and exact kinematics as the
other examples. Its complete empty-cache two-loop boundary acceptance is still
pending; opening the notebook starts no boundary calculation.

`IntegralEvaluator(reductions=tables)` accepts a `ReductionTables` collection.
Use `with_family` to add exact rules and declared residual masters for a native
`hepkit.IntegralFamily`; it returns a new collection and rejects duplicate scopes
or uncovered rule leaves. Rules use the family's original epsilon symbol with
the dimension convention already substituted. Propagator order, routing,
parameter namespaces, dimension and numerator-slot roles belong to the scope.
Equivalent rational coefficient forms share a scope, while original denominator
restrictions remain distinct. Preserve uncancelled expressions when admitting
rules, or pass their exclusions explicitly as `nonzero_conditions`.

Auxiliary deformations, specialized kinematics and recursive boundary families
need their own admitted tables. Supplied identities remain the caller's
mathematical contract; admission checks scope and consistency of the reduction
graph. `PreparedIntegralFamily` exposes the retained basis, target reductions
and nonzero conditions for inspection. `BoundaryCache` separates entries for
different ordered bases, normalizations and conditions, including after binary
reload. A condition omitted from a simplified differential matrix still
restricts endpoint and path admissibility.

Focused native-object examples and regressions are executable without the
two-loop notebook:

```sh
python -m pytest tests/test_loop_integration_reductions.py -q
```

## Notebook tensor displays

Install the optional live-display dependencies alongside the community package:

```sh
pip install 'symbolica[notebook-display]'
```

Large `TensorExpression` outputs and tensor-backed `formatted()` results page
automatically in marimo and Jupyter. To choose the initial maximum page size:

```python
expression.paged(page_size=25, settings=None, notation_source=None)
```

Previous/Next replace the displayed page. The selector offers 25, 100, 250, or
500 terms, defaulting to 25; complex expressions may show fewer. Ellipses mark omitted portions.
When an inner sum is paged, a compact expression outline keeps its surrounding
factors and parentheses visible. Numbered ellipses identify the selected sum
and other omitted subexpressions; the selected sum's terms appear separately
below the outline. Open-subexpression buttons and Parent navigation inspect
other large parts without expanding the expression or introducing scalar-product
aliases. Wrapping adds no inner vertical scroll area.

Each request selects at most 2,000 expression nodes, depth 32, and 64 KiB of
expression data. Rendering limits its output to 256 KiB and 10,000 MathML
elements, and compilation to ten seconds in a separate process. The status line
explains when a budget reduces the selected maximum. The Horizontal scroll
checkbox switches between wrapped terms and a scrollable line (the default). A viewer caches at most three pages. Oversized individual terms become
bounded subexpression previews. These limits are independent of marimo's
8 MB output guard, which remains enabled.

Static exports and environments without Anywidget show a bounded preview.
Navigation requires a live notebook kernel. `to_html()` and `to_svg()` remain
explicit full-export methods; ordinary Symbolica styled text keeps its existing
truncation. Existing notebook kernels need to load the rebuilt package before
using the new display hooks.

After installing a community wheel built from this checkout:

```sh
python examples/hep/diagrams.py
```

The small `scalar_phi3.json` model is a one-particle, cubic-interaction subset
of GammaLoop's `scalars_2p_3p.json` test fixture (MIT / Apache-2.0). It requires
no external model downloads or integral backend.

The bindings are included in both native and Pyodide builds. Native-only
dependencies such as Vakint remain excluded from WASM. Diagram SVG/HTML
rendering may additionally require the upstream optional `typst-py` renderer;
generation, DOT/JSON export, and CFF algebra do not require it.

Regenerate only the HEP stubs, without rewriting the core or other modules:

```sh
cargo run --bin stub_gen --no-default-features --features python_stubgen -- --hep-only
```
