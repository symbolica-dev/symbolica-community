"""One-loop reduction and native master evaluation in one Symbolica kernel.

Use reduce() on a hep.IntegralFamily to obtain scalar master integrals,
or evaluate a primitive directly with a0(), b0(), c0() or d0(). Coefficient
tuples are ordered (finite, 1/eps, 1/eps^2); all masses and scales are squared.
"""
from decimal import Decimal
from typing import Literal, Sequence, TypeVar
from symbolica.core import Expression, Replacement, Evaluator as SymbolicaEvaluator

class DecimalComplex:
    r"""
    A complex number with Python Decimal components.

    Use decimal strings or Decimal inputs to preserve digits beyond binary64.
    The constructor retains the supplied components without rounding to the
    ambient Decimal context. ``complex(z)`` explicitly converts to binary64 and
    can lose precision. Nonfinite components and booleans are rejected.

    Examples
    --------
    >>> from decimal import Decimal
    >>> from symbolica.community.hepkit import oneloop
    >>> z = oneloop.DecimalComplex("1.25", "-0.5")
    >>> assert z.real == Decimal("1.25")
    >>> assert z.imag == Decimal("-0.5")
    """
    def __init__(self, real: int | float | str | Decimal, imag: int | float | str | Decimal | None = None) -> None:
        r"""
        Construct finite real and imaginary Decimal components; omitted imaginary part is zero.

        Examples
        --------
        >>> from decimal import Decimal
        >>> from symbolica.community.hepkit import oneloop
        >>> z = oneloop.DecimalComplex("1.25", "-0.5")
        >>> assert complex(z) == 1.25 - 0.5j

        Parameters
        ----------
        real : int, float, str or Decimal
            Real component. A float preserves its binary approximation; use a
            string for an exact decimal input.
        imag : int, float, str, Decimal or None, optional
            Imaginary component; None means zero.
        """
    @property
    def real(self) -> Decimal:
        r"""
        Real component as a Decimal, without conversion to a Python float.

        Examples
        --------
        >>> from decimal import Decimal
        >>> from symbolica.community.hepkit import oneloop
        >>> z = oneloop.DecimalComplex("1.25", "-0.5")
        >>> assert z.real == Decimal("1.25")
        """
    @property
    def imag(self) -> Decimal:
        r"""
        Imaginary component as a Decimal, without conversion to a Python float.

        Examples
        --------
        >>> from decimal import Decimal
        >>> from symbolica.community.hepkit import oneloop
        >>> z = oneloop.DecimalComplex("1.25", "-0.5")
        >>> assert z.imag == Decimal("-0.5")
        """
    def __complex__(self) -> complex:
        r"""
        Convert both components to a built-in complex number; precision beyond binary64 is lost.

        Examples
        --------
        >>> from decimal import Decimal
        >>> from symbolica.community.hepkit import oneloop
        >>> z = oneloop.DecimalComplex("1.25", "-0.5")
        >>> assert complex(z) == complex(1.25, -0.5)
        """
    def __repr__(self) -> str:
        r"""
        Show both Decimal components for inspection.

        Examples
        --------
        >>> from decimal import Decimal
        >>> from symbolica.community.hepkit import oneloop
        >>> z = oneloop.DecimalComplex("1.25", "-0.5")
        >>> text = repr(z)
        """

Number = int | float | complex | Decimal | DecimalComplex | Expression
Coefficient = complex | DecimalComplex
Coefficients = tuple[Coefficient, Coefficient, Coefficient]
Backend = Literal["auto", "native", "symjit", "expression", "symbolica"]
COEFFICIENT_ORDER: tuple[Literal[0], Literal[-1], Literal[-2]]
DEFAULT_BACKEND: str
SYMBOLICA_REVISION: str
EXPRESSION_INTEROP: bool

def is_initialized() -> bool:
    r"""
    Report whether the native one-loop module has initialized its evaluation support.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> initialized = oneloop.is_initialized()
    """

class Evaluator:
    r"""
    Reuse a numerical evaluator for one scalar one-loop master family.

    Pass a bare primitive symbol: ``oneloop.A0``, ``B0``, ``dB0``, ``C0`` or
    ``D0``. Every evaluation takes physical arguments in that primitive's order,
    with the squared renormalization scale last. Unlike the lowercase convenience
    functions, ``evaluate`` requires the scale explicitly.

    Results are ordered ``(finite, 1/eps, 1/eps**2)``. ``prec`` counts decimal
    significant digits. Binary64 evaluation returns Python complex values;
    arbitrary-precision evaluation returns DecimalComplex components. Use decimal
    strings through Decimal to avoid rounding your inputs before evaluation.
    Per-call precision/backend overrides do not change the stored defaults.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> evaluator = oneloop.Evaluator(oneloop.A0)
    >>> finite, pole, double_pole = evaluator.evaluate([1.0, 1.0])
    >>> assert pole == 1+0j and double_pole == 0j
    >>> rows = evaluator.evaluate_batch([[1.0, 1.0], [2.0, 1.0]])
    >>> assert len(rows) == 2
    """
    def __init__(self, family: Expression, rebuild: bool = False, *, prec: int = 16, backend: Backend = "auto") -> None:
        r"""
        Prepare an evaluator and retain default precision and backend choices.

        ``auto`` selects a supported backend for the requested precision. ``native``
        uses native numerical evaluation; ``symjit`` supports binary64 only.
        ``expression`` (also named ``symbolica``) evaluates Symbolica formulas.
        Unsupported backend/precision combinations raise an error.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> assert evaluator.family == "A0" and evaluator.arity == 2

        Parameters
        ----------
        family : Expression
            Bare primitive symbol such as oneloop.B0, not a B0(...) call.
        rebuild : bool, optional
            Recreate the evaluator's workspace; default False.
        prec : int, optional
            Positive number of decimal significant digits; default 16.
        backend : str, optional
            "auto", "native", "symjit", "expression" or "symbolica"; default "auto".
        """
    @property
    def family(self) -> str:
        r"""
        Primitive family name, such as "A0" or "B0".

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> assert evaluator.family == "A0"
        """
    @property
    def arity(self) -> int:
        r"""
        Number of physical arguments including the final squared scale.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> assert evaluator.arity == 2
        >>> assert oneloop.Evaluator(oneloop.B0).arity == 4
        """
    @property
    def prec(self) -> int:
        r"""
        Default decimal significant-digit count; per-call overrides leave it unchanged.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> assert evaluator.prec == 16
        """
    @property
    def backend(self) -> str:
        r"""
        Requested default backend name; "auto" remains "auto" after backend selection.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> assert evaluator.backend == "auto"
        """
    def evaluate(self, arguments: Sequence[Number], *, prec: int | None = None, backend: Backend | None = None) -> Coefficients:
        r"""
        Evaluate one kinematic point and return (finite, simple pole, double pole).

        Supply exactly ``arity`` numeric arguments including the squared scale.
        Inputs must satisfy the selected primitive's kinematic domain. A per-call
        ``prec`` or ``backend`` override changes only this evaluation.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> finite, pole, double_pole = evaluator.evaluate([1.0, 1.0])
        >>> assert pole == 1+0j
        >>> from decimal import Decimal
        >>> high_precision = evaluator.evaluate([Decimal("1"), Decimal("1")], prec=40)
        >>> assert high_precision[1].real == Decimal("1")

        Parameters
        ----------
        arguments : sequence[Number]
            Ordered invariants, squared masses and squared scale for the primitive.
        prec : int or None, optional
            Decimal significant digits; None uses the constructor default.
        backend : str or None, optional
            Backend override; None uses the constructor default.
        """
    def evaluate_batch(self, rows: Sequence[Sequence[Number]], *, prec: int | None = None, backend: Backend | None = None) -> list[Coefficients]:
        r"""
        Evaluate a sequence of kinematic rows, returning one coefficient tuple per row.

        All rows have ``arity`` entries and use the same precision/backend choice.
        An empty batch returns an empty list. No NumPy array is required.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> rows = evaluator.evaluate_batch([[1.0, 1.0], [2.0, 1.0]])
        >>> assert rows[0] == evaluator.evaluate([1.0, 1.0])
        >>> assert evaluator.evaluate_batch([]) == []

        Parameters
        ----------
        rows : sequence[sequence[Number]]
            Ordered physical arguments, including the squared scale, for each point.
        prec : int or None, optional
            Decimal significant digits; None uses the constructor default.
        backend : str or None, optional
            Backend override; None uses the constructor default.
        """
    def rebuild(self) -> None:
        r"""
        Recreate cached evaluator workspaces while preserving the configured family and defaults.

        Native evaluation refreshes constants/workspaces; a SymJIT evaluator is
        recompiled. This is usually unnecessary between evaluations at new points.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> from symbolica.community.hepkit import oneloop
        >>> evaluator = oneloop.Evaluator(oneloop.A0)
        >>> before = evaluator.evaluate([1.0, 1.0])
        >>> evaluator.rebuild()
        >>> assert evaluator.evaluate([1.0, 1.0]) == before
        """
    def __repr__(self) -> str:
        r"""
        Summarize the selected family, precision and requested backend.

        Examples
        --------
        >>> from symbolica.community.hepkit import oneloop
        >>> summary = repr(oneloop.Evaluator(oneloop.A0))
        """

# Callable primitive symbols from oneloopmaster. Untagged calls include the
# squared scale last; prepend 0, -1 or -2 for native Laurent evaluation hooks.
# Tagged numeric calls containing a Float or ComplexFloat normalize through
# those hooks. Untagged and exact-only calls stay symbolic for inspection.
A0: Expression
B0: Expression
dB0: Expression
C0: Expression
D0: Expression

def a0(mass_squared: Number, mu_squared: Number | None = None, *, rebuild: bool = False, prec: int = 16, backend: Backend = "auto") -> Coefficients:
    r"""
    Evaluate a scalar tadpole A0(mass_squared, mu_squared).

    Results are ordered (finite, 1/eps, 1/eps**2). Invariants and masses are
    squared quantities; omitted mu_squared is exactly one. ``prec`` is the
    positive decimal significant-digit count (default 16). Decimal inputs or
    higher precision produce DecimalComplex values. ``backend`` selects "auto",
    "native", "symjit" (binary64 only), "expression", or "symbolica".
    ``rebuild=True`` refreshes the evaluator workspace.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> finite, pole, double_pole = oneloop.a0(1.0, 1.0)

    Parameters
    ----------
    mass_squared : Number
        Squared mass of the tadpole propagator.
    mu_squared : Number or None, optional
        Squared renormalization scale; None uses exactly one.
    rebuild : bool, optional
        Refresh the evaluator workspace before evaluation; default False.
    prec : int, optional
        Positive number of decimal significant digits; default 16. Use Decimal
        inputs to retain input digits beyond binary64 precision.
    backend : {"auto", "native", "symjit", "expression", "symbolica"}, optional
        Evaluation backend; default "auto" selects a supported backend for the
        requested precision. "symjit" supports binary64 only; "symbolica" is
        an alias for "expression".
    """
def b0(momentum_squared: Number, mass_0_squared: Number, mass_1_squared: Number, mu_squared: Number | None = None, *, rebuild: bool = False, prec: int = 16, backend: Backend = "auto") -> Coefficients:
    r"""
    Evaluate a scalar bubble B0(momentum_squared, mass_0_squared, mass_1_squared, mu_squared).

    Results are ordered (finite, 1/eps, 1/eps**2). Invariants and masses are
    squared quantities; omitted mu_squared is exactly one. ``prec`` is the
    positive decimal significant-digit count (default 16). Decimal inputs or
    higher precision produce DecimalComplex values. ``backend`` selects "auto",
    "native", "symjit" (binary64 only), "expression", or "symbolica".
    ``rebuild=True`` refreshes the evaluator workspace.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> finite, pole, double_pole = oneloop.b0(-1.0, 1.0, 1.0, 1.0)

    Parameters
    ----------
    momentum_squared : Number
        External momentum squared; negative values describe spacelike momentum.
    mass_0_squared, mass_1_squared : Number
        Squared masses of the two propagators, in B0 argument order.
    mu_squared : Number or None, optional
        Squared renormalization scale; None uses exactly one.
    rebuild : bool, optional
        Refresh the evaluator workspace before evaluation; default False.
    prec : int, optional
        Positive number of decimal significant digits; default 16. Use Decimal
        inputs to retain input digits beyond binary64 precision.
    backend : {"auto", "native", "symjit", "expression", "symbolica"}, optional
        Evaluation backend; default "auto" selects a supported backend for the
        requested precision. "symjit" supports binary64 only; "symbolica" is
        an alias for "expression".
    """
def db0(momentum_squared: Number, mass_0_squared: Number, mass_1_squared: Number, mu_squared: Number | None = None, *, rebuild: bool = False, prec: int = 16, backend: Backend = "auto") -> Coefficients:
    r"""
    Evaluate the derivative of B0 with respect to its external momentum squared.

    Results are ordered (finite, 1/eps, 1/eps**2). Invariants and masses are
    squared quantities; omitted mu_squared is exactly one. ``prec`` is the
    positive decimal significant-digit count (default 16). Decimal inputs or
    higher precision produce DecimalComplex values. ``backend`` selects "auto",
    "native", "symjit" (binary64 only), "expression", or "symbolica".
    ``rebuild=True`` refreshes the evaluator workspace.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> finite, pole, double_pole = oneloop.db0(-1.0, 1.0, 1.0, 1.0)

    Parameters
    ----------
    momentum_squared : Number
        External momentum squared at which the derivative of B0 is evaluated.
    mass_0_squared, mass_1_squared : Number
        Squared masses held fixed while differentiating B0.
    mu_squared : Number or None, optional
        Squared renormalization scale; None uses exactly one.
    rebuild : bool, optional
        Refresh the evaluator workspace before evaluation; default False.
    prec : int, optional
        Positive number of decimal significant digits; default 16. Use Decimal
        inputs to retain input digits beyond binary64 precision.
    backend : {"auto", "native", "symjit", "expression", "symbolica"}, optional
        Evaluation backend; default "auto" selects a supported backend for the
        requested precision. "symjit" supports binary64 only; "symbolica" is
        an alias for "expression".
    """
def c0(p1_squared: Number, p2_squared: Number, p3_squared: Number, mass_0_squared: Number, mass_1_squared: Number, mass_2_squared: Number, mu_squared: Number | None = None, *, rebuild: bool = False, prec: int = 16, backend: Backend = "auto") -> Coefficients:
    r"""
    Evaluate C0(p1_squared, p2_squared, p3_squared, mass_0_squared, mass_1_squared, mass_2_squared, mu_squared).

    Results are ordered (finite, 1/eps, 1/eps**2). Invariants and masses are
    squared quantities; omitted mu_squared is exactly one. ``prec`` is the
    positive decimal significant-digit count (default 16). Decimal inputs or
    higher precision produce DecimalComplex values. ``backend`` selects "auto",
    "native", "symjit" (binary64 only), "expression", or "symbolica".
    ``rebuild=True`` refreshes the evaluator workspace.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> finite, pole, double_pole = oneloop.c0(-1.0, -2.0, -3.0, 1.0, 1.0, 1.0, 1.0)

    Parameters
    ----------
    p1_squared, p2_squared, p3_squared : Number
        Three external momentum invariants, in C0 argument order.
    mass_0_squared, mass_1_squared, mass_2_squared : Number
        Squared masses of the three propagators, in C0 argument order.
    mu_squared : Number or None, optional
        Squared renormalization scale; None uses exactly one.
    rebuild : bool, optional
        Refresh the evaluator workspace before evaluation; default False.
    prec : int, optional
        Positive number of decimal significant digits; default 16. Use Decimal
        inputs to retain input digits beyond binary64 precision.
    backend : {"auto", "native", "symjit", "expression", "symbolica"}, optional
        Evaluation backend; default "auto" selects a supported backend for the
        requested precision. "symjit" supports binary64 only; "symbolica" is
        an alias for "expression".
    """
def d0(p1_squared: Number, p2_squared: Number, p3_squared: Number, p4_squared: Number, s12: Number, s23: Number, mass_0_squared: Number, mass_1_squared: Number, mass_2_squared: Number, mass_3_squared: Number, mu_squared: Number | None = None, *, rebuild: bool = False, prec: int = 16, backend: Backend = "auto") -> Coefficients:
    r"""
    Evaluate D0 with four external squared momenta, s12, s23, four squared masses, and the squared scale.

    Results are ordered (finite, 1/eps, 1/eps**2). Invariants and masses are
    squared quantities; omitted mu_squared is exactly one. ``prec`` is the
    positive decimal significant-digit count (default 16). Decimal inputs or
    higher precision produce DecimalComplex values. ``backend`` selects "auto",
    "native", "symjit" (binary64 only), "expression", or "symbolica".
    ``rebuild=True`` refreshes the evaluator workspace.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> finite, pole, double_pole = oneloop.d0(-1.0, -1.0, -1.0, -1.0, -3.0, -4.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    Parameters
    ----------
    p1_squared, p2_squared, p3_squared, p4_squared : Number
        Four external momenta squared, in D0 argument order.
    s12 : Number
        Channel invariant (p1 + p2)**2.
    s23 : Number
        Channel invariant (p2 + p3)**2.
    mass_0_squared, mass_1_squared, mass_2_squared, mass_3_squared : Number
        Squared masses of the four propagators, in D0 argument order.
    mu_squared : Number or None, optional
        Squared renormalization scale; None uses exactly one.
    rebuild : bool, optional
        Refresh the evaluator workspace before evaluation; default False.
    prec : int, optional
        Positive number of decimal significant digits; default 16. Use Decimal
        inputs to retain input digits beyond binary64 precision.
    backend : {"auto", "native", "symjit", "expression", "symbolica"}, optional
        Evaluation backend; default "auto" selects a supported backend for the
        requested precision. "symjit" supports binary64 only; "symbolica" is
        an alias for "expression".
    """

def master_coefficients(master: Expression) -> list[Expression]:
    r"""
    Return the three Laurent coefficients of a complete primitive master call.

    The input includes physical arguments and squared scale, without a Laurent
    tag. The returned symbolic expressions carry tags 0, -1 and -2 for finite,
    simple-pole and double-pole coefficients and retain native evaluation hooks.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> s = S("s")
    >>> coefficients = oneloop.master_coefficients(oneloop.B0(s, 0, 0, 1))
    >>> assert len(coefficients) == 3

    Parameters
    ----------
    master : Expression
        Complete untagged A0/B0/dB0/C0/D0 call, including squared scale.
    """
def reduction_coefficients(reduction: Reduction, mu_squared: Expression | None = None) -> list[Expression]:
    r"""
    Expand a reduction about d=4-2*eps and combine its master coefficients.

    Returns [finite, simple_pole, double_pole] with native evaluation hooks.
    Raises ValueError for coefficient poles at d=4 requiring unavailable
    positive-order master coefficients, fractional Taylor powers, or
    dimension-dependent kinematics or scale.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> d, k, p, s = S("d", "k", "p", "s")
    >>> kin = hep.Kinematics(d, momenta=[k, p]).with_scalar_product(p, p, s)
    >>> family = hep.IntegralFamily([k], [p], [kin.scalar_product(k, k),
    ...     kin.scalar_product(k-p, k-p)], kinematics=kin)
    >>> reduction = oneloop.reduce(family, [1, 1])
    >>> coefficients = oneloop.reduction_coefficients(reduction)
    >>> assert len(coefficients) == 3

    Parameters
    ----------
    reduction : Reduction
        Symbolic one-loop reduction with dimension dependence retained.
    mu_squared : Expression or None, optional
        Squared scale; None uses one.
    """
    ...
def compile_native(expressions: Sequence[Expression | int | float | complex], parameters: Sequence[Expression]) -> SymbolicaEvaluator:
    r"""
    Compile symbolic combinations of master coefficients for repeated evaluation.

    Returns a Symbolica evaluator retaining the primitive master definitions.
    Use its complex evaluation method and pass parameter values in the supplied
    order. This operation builds an evaluator; it does not evaluate a point.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> m2 = S("m2")
    >>> coefficients = oneloop.master_coefficients(oneloop.A0(m2, 1))
    >>> compiled = oneloop.compile_native(coefficients, [m2])
    >>> values = compiled.evaluate_complex([1+0j])

    Parameters
    ----------
    expressions : sequence[Expression | int | float | complex]
        Outputs to compile, usually master or reduction coefficients.
    parameters : sequence[Expression]
        Ordered independent symbols receiving numerical values.
    """
def get_expression(master: Expression, *, coefficient: Literal[0, -1, -2] | None = None, max_nodes: int = 1_000_000, max_depth: int = 512) -> Expression | tuple[Expression, Expression, Expression]:
    r"""
    Expand a complete master call into symbolic Laurent-coefficient formulas.

    Without ``coefficient`` return (finite, simple_pole, double_pole). With tag
    0, -1 or -2 return that coefficient only. Piecewise branches remain explicit
    until their conditions can be decided. ``max_nodes`` and ``max_depth`` bound
    formula expansion and raise an error when exceeded.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> m2 = S("m2")
    >>> pole = oneloop.get_expression(oneloop.A0(m2, 1), coefficient=-1)

    Parameters
    ----------
    master : Expression
        Untagged primitive master call with physical arguments and squared scale.
    coefficient : {0, -1, -2} or None, optional
        Laurent power to select; None returns all three coefficients.
    max_nodes : int, optional
        Expression expansion budget; default 1_000_000.
    max_depth : int, optional
        Expansion nesting limit; default 512.
    """
_Expressions = TypeVar("_Expressions", Expression, list[Expression], tuple[Expression, ...])
def select_branch(expression: _Expressions, replacement_rules: Sequence[Replacement]) -> _Expressions:
    r"""
    Resolve piecewise conditions using replacements, preserving symbolic branch values.

    Replacements act only in conditions, not in the selected result expressions.
    Unresolved predicates remain symbolic. Input may be one expression or a list
    or tuple; the returned container has the same shape.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> from symbolica.community.hepkit import oneloop
    >>> from symbolica import Replacement
    >>> m2 = S("m2")
    >>> expressions = oneloop.get_expression(oneloop.A0(m2, 1))
    >>> selected = oneloop.select_branch(expressions, [Replacement(m2, E("1"))])
    >>> assert len(selected) == 3

    Parameters
    ----------
    expression : Expression, list[Expression] or tuple[Expression, ...]
        Piecewise formula or collection to inspect.
    replacement_rules : sequence[Replacement]
        Kinematic assumptions used only to decide branch conditions.
    """
