<h1 align="center">
  <br>
  <img src="https://symbolica.io/logo.svg" alt="logo" width="200">
  <br>
</h1>

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://zulip.symbolica.io"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/benruijl/symbolica"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
    <a href="https://app.codecov.io/gh/benruijl/symbolica"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/benruijl/symbolica?token=N43MATK5XJ&style=flat-square"></a>
</p>

# Symbolica ⊆ Modern Computer Algebra

Symbolica is a blazing fast computer algebra system for Python and Rust, born of a need to push the boundaries of computations in science and enterprise.
Check out the live [Jupyter Notebook demo](https://colab.research.google.com/drive/1VAtND2kddgBwNt1Tjsai8vnbVIbgg-7D?usp=sharing)!

For documentation and more, see [symbolica.io](https://symbolica.io).



## Quick Example

Symbolica allows you to build and manipulate mathematical expressions, for example from a Jupyter Notebook:

<img width="600" alt="A demo of Symbolica" src="https://symbolica.io/resources/demo.light.svg">

You are able to perform these operations from the comfort of a programming language that you (probably) already know, by using Symbolica's bindings to Python and Rust:

<img width="600" alt="A demo of Symbolica" src="https://symbolica.io/resources/completion.light.png">

# Installation

Symbolica can be installed for Python 3.7 or newer using `pip`:

```sh
pip install symbolica
```

Visit the [Get Started](https://symbolica.io/docs/get_started.html) page for detailed installation instructions.

For Pyodide 314.x (Python 3.14), install the PyEmscripten wheel with `micropip`:

```python
import micropip
await micropip.install("symbolica")
```

The WebAssembly build uses Symbolica's Rust numeric backends. Native code
generation is unavailable in this build.

# Examples

Below we list some examples of the features of Symbolica. Check the [guide](https://symbolica.io/docs/) for a complete overview.

### Pattern matching

Variables ending with a `_` are wildcards that match to any subexpression.
In the following example we try to match the pattern `f(w1_,w2_)`:

```python
from symbolica import *
x, y, w1_, w2_, f = S('x','y','w1_','w2_', 'f')
e = f(3,x)*y**2+5
r = e.replace_all(f(w1_,w2_), f(w1_ - 1, w2_**2))
print(r)
```
which yields `y^2*f(2,x^2)+5`.

### Solving a linear system

Solve a linear system in `x` and `y` with a parameter `c`:

```python
from symbolica import *

x, y, c, f = S('x', 'y', 'c', 'f')

x_r, y_r = Expression.solve_linear_system(
    [f(c)*x + y + c, y + c**2], [x, y])
print('x =', x_r, ', y =', y_r)
```
which yields `x = (-c+c^2)*f(c)^-1` and `y = -c^2`.

### Series expansion

Perform a series expansion in `x`:

```python
from symbolica import *
e = E('exp(5+x)/(1-x)').series(S('x'), 0, 3)

print(e)
```
which yields `(exp(5))+(2*exp(5))*x+(5/2*exp(5))*x^2+(8/3*exp(5))*x^3+𝒪(x^4)`.

### Rational arithmetic

Symbolica is world-class in rational arithmetic, outperforming Mathematica, Maple, Form, Fermat, and other computer algebra packages. Simply convert an expression to a rational polynomial:
```python
from symbolica import *
p = E('(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)').to_rational_polynomial()
print(p)
```
which yields `(45+13*x+50*x*y^2+152*x^2+25*x^2*y^4+300*x^3*y^2+150*x^4*y^4)/(5+2*x+30*x^2+12*x^3)`.


# Community packages

Version 3.0 ships core Symbolica and symbolic integration, including step-by-step
explanations, through `symbolica-integrate` 2.0. Integration rule metadata is
compressed to reduce download size.

Native wheels also include Spenso, Idenso, FeynKit HEP tools, Vakint, and the example extension.
PyEmscripten wheels include Idenso, Spenso, HEP, and the example extension.
Vakint and Spenso's compiled evaluators require a native installation.

The HEP classes are available directly under `symbolica.community.hep`, including
`FeynmanDiagram`, `Model`, `Generator`, `CffGenerator`, and `TensorReducer`.

## Development

Follow the development and discussions on [Zulip](https://reform.zulipchat.com)!
