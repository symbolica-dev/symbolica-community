<h1 align="center">
  <br>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://symbolica.io/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://symbolica.io/logo.svg">
  <img src="https://symbolica.io/logo.svg" alt="logo" width="200">
</picture>
  <br>
</h1>

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://reform.zulipchat.com"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/symbolica-dev/symbolica-community"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
</p>

# Community-enhanced Symbolica

[Symbolica](https://symbolica.io) is a computer algebra system for Python and Rust.
This package adds community tools for symbolic integration, tensor algebra, and
high-energy physics.

The community extensions include:

- Symbolic integration with step-by-step explanations.
- Tensor networks, Dirac algebra, and color algebra with Spenso and Idenso.
- Feynman diagrams, process generation, models, and kinematics with FeynKit.
- Exact and numerical loop integration, integral reduction, and sector decomposition.
- Massive vacuum integrals with Vakint.

## Installation

Install with Python 3.9 or newer:

```sh
pip install symbolica
```

In Pyodide 314.x (Python 3.14), install with `micropip`:

```python
import micropip
await micropip.install("symbolica")
```

Vakint and compiled tensor evaluators require a native Python installation.

## Usage

Use Symbolica to manipulate expressions and compute integrals:

```python
from symbolica import E, S

x = S("x")
print(E("(x+1)^2").expand())
print(E("x^2").integrate(x))
```

Import community tools from `symbolica.community`:

```python
from symbolica.community import tensor
from symbolica.community.hepkit import FeynmanDiagram, Model, Generator, TensorReducer
```

For a complete introduction to Symbolica, see the
[documentation](https://symbolica.io/docs).

## Examples

- [HEP guide](examples/hep/README.md): diagrams, tensors, and loop calculations.
- [One-loop reduction](examples/hep/oneloop_reduce.py): reduce integrals and evaluate their coefficients.
- [Numerical CFF and LTD](examples/hep/numerical_cff_ltd.py): compare energy representations and numerical integration methods.
- [Higgs-pair production](examples/hep/gghh_complete.py): explore diagrams, sector decomposition, and numerical integration.
- [Pyodide playground](examples/pyodide/README.md): use Symbolica in the browser.

## Community

Ask questions and discuss Symbolica on [Zulip](https://reform.zulipchat.com).
For contributions, see the [contribution guidelines](Contributing.md).
For independent extension development, see the [extension author guide](docs/writing-extensions.md).
For local WASM builds, see the [build guide](docs/local-wasm-build.md).
