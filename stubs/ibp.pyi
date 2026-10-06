"""Native RustRed IBP solving in the host Symbolica kernel."""

from typing import overload

from symbolica import Expression

class IBPFamily:
    r"""
    Find exact integration-by-parts (IBP) relations for a complete integral family.

    Wrap a ``hep.IntegralFamily`` after checking ``is_complete`` and
    ``is_independent``. The denominator order, signs, masses, dimension and
    external scalar products are preserved. Supply a symbolic dimension through
    ``Kinematics``. Missing scalar-product coordinates must be added explicitly
    with ``family.complete()``; auxiliary denominators normally have nonpositive powers.

    Choose ``solve_parametric`` for a reusable symbolic recurrence in a sector,
    or ``reduce_laporta`` for a finite set of integer-power targets. Inspect the
    returned rules and residual integrals before using a reduction. Searches are
    bounded: they do not certify a complete master-integral basis.

    Searches use the compiled runtime arity registry (by default 1–16 denominators)
    and integer powers from -64 through 63.
    Applying a discovered rule accepts signed 64-bit powers. Booleans are not powers.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> d, k, m2 = S("d", "k", "m2")
    >>> kin = hep.Kinematics(d, momenta=[k])
    >>> family = hep.IntegralFamily([k], [], [kin.scalar_product(k, k) - m2], kinematics=kin)
    >>> ibp = hep.IBPFamily(family, name="T")
    >>> solution = ibp.solve_parametric([True])
    >>> terms = solution.reduce([2])
    >>> assert terms[0][0] == [1]
    >>> assert (terms[0][1] - (d-2)/(2*m2)).together() == E("0")

    Parameters
    ----------
    family : IntegralFamily
        Complete, independent denominator basis with symbolic dimension.
    name : str, optional
        Family label used in diagnostic output; default "F".
    cut : list[bool] or None, optional
        Reverse-unitarity flags in denominator order; None leaves all lines uncut.
    """
    def __init__(self, family: IntegralFamily, *, name: str = "F", cut: list[bool] | None = None) -> None:
        r"""
        Prepare a complete denominator basis for exact IBP searches.

        Raises ``ValueError`` for an incomplete/dependent basis, unsupported
        kinematics, or too many denominators. Complete missing coordinates before
        constructing this object; dependent propagators require partial fractions.

        Examples
        --------
        >>> from symbolica import S, E
        >>> from symbolica.community import hepkit as hep
        >>> d, k, m2 = S("d", "k", "m2")
        >>> kin = hep.Kinematics(d, momenta=[k])
        >>> family = hep.IntegralFamily([k], [], [kin.scalar_product(k, k) - m2], kinematics=kin)
        >>> ibp = hep.IBPFamily(family, name="T")
        >>> assert ibp.denominator_count == 1

        Parameters
        ----------
        family : IntegralFamily
            Complete, independent family; its denominator order is retained.
        name : str, optional
            Diagnostic family label; default "F".
        cut : list[bool] or None, optional
            One reverse-unitarity flag per denominator. Nonpositive powers of a
            cut denominator vanish; None leaves all denominators uncut.
        """
    @staticmethod
    def compiled_runtime_arities() -> list[int]:
        """Solver entry points compiled into this host, not a mathematical bound.

        Examples
        --------
        >>> from symbolica.community import hepkit as hep
        >>> arities = hep.IBPFamily.compiled_runtime_arities()
        >>> assert arities == sorted(set(arities))
        """
    @property
    def cut(self) -> list[bool]:
        """Reverse-unitarity flags in denominator order; nonpositive cut powers vanish.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> assert ibp.cut == [False]
        """
    @property
    def index_symbols(self) -> list[Expression]:
        r"""
        Formal integral-index symbols, one per denominator in family order.

        These symbols occur in ``ibp_identities`` and parametric rules. Use the
        returned symbols for substitutions: freshly creating a symbol named ``n1``
        does not identify this family's privately scoped index.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> n = ibp.index_symbols[0]
        >>> assert len(ibp.index_symbols) == ibp.denominator_count
        """
    @property
    def denominator_count(self) -> int:
        r"""
        Number of entries required in every sector, target and fixed-power vector.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> assert ibp.denominator_count == len(family.denominators)
        """
    @property
    def parameter_bindings(self) -> list[tuple[Expression, Expression]]:
        """Internal polynomial parameters paired with original HEPKit atoms.

        A presentation legend, not a replacement of serialized artifact names.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> bindings = ibp.parameter_bindings
        >>> assert {original for internal, original in bindings} == {d, m2}
        """
        ...
    def start_generation(self, *, event_capacity: int = 256, **options: object) -> rustred.CandidateGenerationSession:
        """Start native generation from this exact routed family, without re-parsing.

        Poll the bounded stream for progress; cancellation drains at native safe
        boundaries. Output is a generated candidate, not a closure certificate.
        ``nonpositive_indices`` identifies auxiliary scalar-product slots.
        Cut families currently use the existing cut-aware finite/parametric APIs.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> session = ibp.start_generation(n_cores=1, numerical_depth=1)
        >>> assert session.wait(timeout=120)
        >>> artifact = session.result().artifact()
        >>> assert artifact.metadata()["arity"] == ibp.denominator_count

        Parameters
        ----------
        event_capacity : int, optional
            Maximum number of retained progress events; default 256.
        options : object
            Native candidate-generation keyword options, including ``n_cores``,
            ``numerical_depth``, and ``nonpositive_indices`` for auxiliary slots.
        """
        ...
    def normalize_candidate_terminals(
        self, artifact: rustred.CandidateArtifact, *, max_terminals: int = 1000000,
        max_supports: int = 100000, max_matrix_cells: int = 1000000,
        max_output_terms: int = 4000000,
    ) -> rustred.TerminalNormalization:
        """Explicit exact normalization of saved finite residuals in this family.

        Uses native unit aliases and weighted polynomial-numerator relations;
        it does not regenerate candidates, solve ordinary IBPs, certify closure,
        or prove master independence. Native family identity must match.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> session = ibp.start_generation(n_cores=1, numerical_depth=1)
        >>> assert session.wait(timeout=120)
        >>> artifact = session.result().artifact()
        >>> normalized = ibp.normalize_candidate_terminals(artifact)
        >>> assert normalized.metadata()["exact_within_family"]

        Parameters
        ----------
        artifact : rustred.CandidateArtifact
            Generated candidate artifact belonging to this exact native family.
        max_terminals : int, optional
            Maximum number of input or output terminal keys; default 1000000.
        max_supports : int, optional
            Maximum number of normalization supports; default 100000.
        max_matrix_cells : int, optional
            Maximum number of cells in normalization matrices; default 1000000.
        max_output_terms : int, optional
            Maximum number of terms in the normalized output; default 4000000.
        """
        ...
    def ibp_identities(self) -> list[list[tuple[list[Expression], Expression]]]:
        r"""
        Generate the L*(L+E) ordinary IBP equations with symbolic indices.

        Each row is a list of ``(powers, coefficient)`` pairs representing
        ``sum(coefficient * I(*powers)) = 0``. Powers contain index symbols and
        integer shifts, not just shifts. Coefficients retain the family's kinematics.
        This method generates equations without solving them.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> I = S("I")
        >>> equations = [sum((coefficient * I(*powers) for powers, coefficient in row), E("0"))
        ...              for row in ibp.ibp_identities()]
        >>> assert len(equations) == 1  # one loop, no external momenta
        """
    def solve_parametric(
        self,
        sector: list[bool],
        *,
        fixed: list[int | None] | None = None,
        max_depth: int = 2,
        include_lorentz: bool = False,
    ) -> IBPSolution:
        r"""
        Search for symbolic recurrences within one positive/nonpositive power sector.

        ``True`` means a strictly positive denominator power; ``False`` means zero
        or negative. The sector and optional ``fixed`` vector must each have one
        entry per denominator. ``None`` leaves an index symbolic; an integer fixes
        its absolute power and must agree with the sector.

        Returns an ``IBPSolution`` whose rules carry explicit nonzero conditions and
        exceptional branches. ``reduce`` performs one recurrence step per call.
        Increasing ``max_depth`` searches more seeds but does not guarantee closure.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> solution = ibp.solve_parametric([True], fixed=[None], max_depth=1)
        >>> assert solution.rules
        >>> lowered = solution.reduce([3])
        >>> assert lowered[0][0] == [2]  # one recurrence step

        Parameters
        ----------
        sector : list[bool]
            Positive/nonpositive support, in denominator order.
        fixed : list[int | None] or None, optional
            Fixed absolute powers in -64..63; default all symbolic.
        max_depth : int, optional
            Nonnegative seed-search depth; default 2.
        include_lorentz : bool, optional
            Add Lorentz-invariance identities; default False.
        """
    def reduce_laporta(
        self,
        targets: list[list[int]],
        *,
        max_depth: int = 2,
        include_lorentz: bool = False,
        max_targets: int = 1024,
        preferred_masters: list[list[int]] | None = None,
        until_stable: bool = False,
    ) -> IBPSolution:
        r"""
        Reduce requested integer-power integrals by exact finite-target elimination.

        The solver searches seed neighborhoods and back-substitutes solved targets.
        ``solution.residuals`` lists the basis left at this search depth, which is
        not a certified set of independent master integrals. Increase ``max_depth``
        when more relations are needed. Exceeding ``max_targets`` raises ``ValueError``
        rather than silently truncating the search.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> solution = ibp.reduce_laporta([[3]], max_depth=1)
        >>> terms = solution.reduce([3])
        >>> assert terms[0][0] == [1]  # solved intermediate powers are back-substituted

        Parameters
        ----------
        targets : list[list[int]]
            Power vectors, one integer in -64..63 per denominator.
        max_depth : int, optional
            Nonnegative signed L1 seed radius; default 2.
        include_lorentz : bool, optional
            Add Lorentz-invariance identities; default False.
        max_targets : int, optional
            Limit on distinct integrals searched, including discovered targets;
            default 1024.
        preferred_masters : list[list[int]] or None, optional
            Requested residual basis, used where the discovered equations permit
            replacement. None retains the solver's residual basis.
        until_stable : bool, optional
            Check two further depths for an unchanged residual set, within
            ``max_depth``. This heuristic does not certify a complete basis.
        """
    def __repr__(self) -> str:
        r"""
        Summarize this IBPFamily for interactive inspection.

        Examples
        --------
        Using the setup in the ``IBPFamily`` class example:

        >>> summary = repr(ibp)
        """

class IBPRule:
    r"""
    One solved IBP identity, with its domain of validity.

    Obtain rules from ``IBPSolution.rules``; they have no public constructor.
    ``target`` describes the left-hand integral. ``terms`` is the right-hand
    linear combination as ``(powers, coefficient)`` pairs. Symbolic powers use
    ``IBPFamily.index_symbols``.

    Every ``nonzero_conditions`` expression must remain nonzero. An exceptional
    branch is a list of polynomials that vanish simultaneously; any such branch
    forbids application. Integer-index checks are performed by ``apply``, but
    conditions still symbolic in masses, dimension or invariants remain the
    caller's responsibility when specializing kinematics.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> d, k, m2 = S("d", "k", "m2")
    >>> kin = hep.Kinematics(d, momenta=[k])
    >>> family = hep.IntegralFamily([k], [], [kin.scalar_product(k, k) - m2], kinematics=kin)
    >>> ibp = hep.IBPFamily(family, name="T")
    >>> solution = ibp.solve_parametric([True], max_depth=1)
    >>> rule = solution.rules[0]
    >>> terms = rule.apply([2])
    >>> assert terms[0][0] == [1]
    >>> assert (terms[0][1] - (d-2)/(2*m2)).together() == E("0")
    """
    @property
    def target(self) -> list[Expression]:
        r"""
        Left-hand powers as expressions in the family indices. Fixed coordinates are integers.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> target_powers = rule.target
        >>> assert len(target_powers) == ibp.denominator_count
        """
    @property
    def terms(self) -> list[tuple[list[Expression], Expression]]:
        r"""
        Right-hand terms as (power expressions, coefficient) pairs. Their order is not a reduction priority.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> I = S("I")
        >>> rhs = sum((coefficient * I(*powers) for powers, coefficient in rule.terms), E("0"))
        """
    @property
    def nonzero_conditions(self) -> list[Expression]:
        r"""
        Expressions required to stay nonzero for this rule. Symbolic kinematic conditions must be checked before specialization.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> conditions = rule.nonzero_conditions
        >>> for condition in conditions:
        ...     print("Required nonzero:", condition)
        """
    @property
    def exceptions(self) -> list[list[Expression]]:
        r"""
        Exceptional branches: a rule is forbidden if every polynomial in any one branch vanishes.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> for branch in rule.exceptions:
        ...     print("Excluded simultaneous zeroes:", branch)
        """
    @property
    def sector(self) -> list[bool]:
        r"""
        Positive-power support of the left-hand integral; False includes both zero and negative powers.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> assert rule.sector == [True]
        """
    @overload
    def apply(
        self, powers: list[int], *, integral: None = None
    ) -> list[tuple[list[int], Expression]]:
        r"""
        Substitute integer powers into this rule and return its right-hand side.

        Returns a list of ``(powers, coefficient)`` pairs, or a Symbolica expression
        when ``integral`` is a bare symbol. Raises ``ValueError`` for wrong arity,
        incompatible fixed powers/sector, or a detected exceptional locus. Conditions
        that remain symbolic in kinematic parameters must still be nonzero.
        This applies one rule once; it does not recursively reduce the result.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> terms = rule.apply([3])
        >>> assert terms[0][0] == [2]
        >>> I = S("I")
        >>> expression = rule.apply([3], integral=I)
        >>> assert (expression - terms[0][1]*I(2)).together() == E("0")

        Parameters
        ----------
        powers : list[int]
            One signed 64-bit integer per denominator; booleans are rejected.
        integral : Expression or None, optional
            Bare Symbolica symbol used as the integral function head; default
            None returns structured terms. Pass S("I"), not I(1).
        """
    @overload
    def apply(self, powers: list[int], *, integral: Expression) -> Expression:
        r"""
        Substitute integer powers into this rule and return its right-hand side.

        Returns a list of ``(powers, coefficient)`` pairs, or a Symbolica expression
        when ``integral`` is a bare symbol. Raises ``ValueError`` for wrong arity,
        incompatible fixed powers/sector, or a detected exceptional locus. Conditions
        that remain symbolic in kinematic parameters must still be nonzero.
        This applies one rule once; it does not recursively reduce the result.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> terms = rule.apply([3])
        >>> assert terms[0][0] == [2]
        >>> I = S("I")
        >>> expression = rule.apply([3], integral=I)
        >>> assert (expression - terms[0][1]*I(2)).together() == E("0")

        Parameters
        ----------
        powers : list[int]
            One signed 64-bit integer per denominator; booleans are rejected.
        integral : Expression or None, optional
            Bare Symbolica symbol used as the integral function head; default
            None returns structured terms. Pass S("I"), not I(1).
        """
    def __repr__(self) -> str:
        r"""
        Summarize this IBPRule for interactive inspection.

        Examples
        --------
        Using the setup in the ``IBPRule`` class example:

        >>> summary = repr(rule)
        """

class IBPSolution:
    r"""
    Rules, unresolved integrals and statistics returned by an IBP search.

    Obtain a solution from ``IBPFamily.solve_parametric`` or ``reduce_laporta``;
    there is no public constructor. ``reduce`` chooses the first applicable rule.
    For parametric solutions it takes one recurrence step. Laporta solutions
    already back-substitute solved targets. Unmatched integrals remain explicit.

    Residual integrals are the unresolved basis at the chosen search depth;
    they are not certified independent masters. Rule conditions still apply
    when masses, invariants or the dimension are specialized.

    Examples
    --------
    >>> from symbolica import S, E
    >>> from symbolica.community import hepkit as hep
    >>> d, k, m2 = S("d", "k", "m2")
    >>> kin = hep.Kinematics(d, momenta=[k])
    >>> family = hep.IntegralFamily([k], [], [kin.scalar_product(k, k) - m2], kinematics=kin)
    >>> ibp = hep.IBPFamily(family, name="T")
    >>> solution = ibp.reduce_laporta([[3]], max_depth=1)
    >>> I = S("I")
    >>> reduced = solution.reduce([3], integral=I)
    >>> expected = (d-2)*(d-4)/(8*m2**2)*I(1)
    >>> assert (reduced - expected).together() == E("0")
    >>> assert [1] in solution.residuals
    """
    @property
    def rules(self) -> list[IBPRule]:
        r"""
        Solved identities in application order.

        Inspect each rule's target, sector, nonzero conditions and exceptions before
        using it at specialized kinematics. An empty list means no solved rules were found.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> assert solution.rules
        >>> for rule in solution.rules:
        ...     identity = (rule.target, rule.terms)
        """
    @property
    def residuals(self) -> list[list[int]]:
        r"""
        Integer power vectors left unresolved by the finite-target search.

        The list describes this search's remaining basis, not a proof of master
        independence. Parametric searches may have no residual list even though a
        recurrence leaves boundary cases unsolved.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> assert [1] in solution.residuals
        >>> I = S("I")
        >>> remaining_integrals = [I(*powers) for powers in solution.residuals]
        """
    @property
    def stats(self) -> dict[str, int]:
        r"""
        Search counters: sectors, seeds, rows, and exact_trace_rows.

        Use these counts to compare search effort at different depths. They report
        work performed, not completeness or the number of independent masters.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> assert solution.stats["rows"] > 0
        >>> seed_count = solution.stats["seeds"]
        """
    @property
    def depth(self) -> int | None:
        """Seed depth of a Laporta search; None for parametric rules.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> assert solution.depth == 1
        """
    @property
    def stable_depth(self) -> int | None:
        """First depth reproduced by the next two depths; a stability heuristic.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> assert solution.stable_depth is None  # no stability search requested
        """
    @property
    def preferred_masters(self) -> list[tuple[list[int], str]]:
        """Requested powers and their replaced/residual status, in request order.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> assert solution.preferred_masters == []  # no preferred basis requested
        """
    @property
    def replaced(self) -> list[list[int]]:
        """Search residuals replaced by the preferred masters.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> assert solution.replaced == []  # the solver's residual basis is retained
        """
    def certify(self, *, count_masters: bool = True, replay: bool = True, seed: int = 0) -> IBPCertificate:
        """Replay exact Laporta identities and optionally check generic sector master counts.

        Parametric solutions cannot be certified. A count-consistent result is
        not a proof that all residual integrals are independent.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> certificate = solution.certify(count_masters=False)
        >>> assert certificate.reduction == "verified"
        >>> assert certificate.masters == "unchecked"

        Parameters
        ----------
        count_masters : bool, optional
            Request probabilistic generic sector master counts; default True.
        replay : bool, optional
            Reconstruct the reduction from original exact identities; default True.
        seed : int, optional
            Nonnegative seed for probabilistic master counting; default 0.
        """
    @overload
    def reduce(
        self, powers: list[int], *, integral: None = None
    ) -> list[tuple[list[int], Expression]]:
        r"""
        Apply the first valid rule, leaving unmatched integrals unchanged.

        Returns ``(powers, coefficient)`` terms by default. With a bare Symbolica
        symbol as ``integral``, returns their symbolic sum; an empty sum is zero.
        Laporta rules are back-substituted, whereas parametric rules perform one
        recurrence step. A rule's symbolic kinematic conditions still apply.

        Wrong power-vector length or a non-symbol ``integral`` raises ``ValueError``.
        Unmatched input returns ``[(powers, 1)]`` or ``integral(*powers)`` rather
        than an error; inspect the returned powers to see what remains.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> I = S("I")
        >>> terms = solution.reduce([3])
        >>> assert terms[0][0] == [1]
        >>> assert solution.reduce([1], integral=I) == I(1)

        Parameters
        ----------
        powers : list[int]
            One signed 64-bit integer per denominator; booleans are rejected.
        integral : Expression or None, optional
            Bare function-head symbol, e.g. S("I"); None returns structured terms.
        """
    @overload
    def reduce(self, powers: list[int], *, integral: Expression) -> Expression:
        r"""
        Apply the first valid rule, leaving unmatched integrals unchanged.

        Returns ``(powers, coefficient)`` terms by default. With a bare Symbolica
        symbol as ``integral``, returns their symbolic sum; an empty sum is zero.
        Laporta rules are back-substituted, whereas parametric rules perform one
        recurrence step. A rule's symbolic kinematic conditions still apply.

        Wrong power-vector length or a non-symbol ``integral`` raises ``ValueError``.
        Unmatched input returns ``[(powers, 1)]`` or ``integral(*powers)`` rather
        than an error; inspect the returned powers to see what remains.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> I = S("I")
        >>> terms = solution.reduce([3])
        >>> assert terms[0][0] == [1]
        >>> assert solution.reduce([1], integral=I) == I(1)

        Parameters
        ----------
        powers : list[int]
            One signed 64-bit integer per denominator; booleans are rejected.
        integral : Expression or None, optional
            Bare function-head symbol, e.g. S("I"); None returns structured terms.
        """
    def __repr__(self) -> str:
        r"""
        Summarize this IBPSolution for interactive inspection.

        Examples
        --------
        Using the setup in the ``IBPSolution`` class example:

        >>> summary = repr(solution)
        """


class IBPCertificate:
    """Evidence returned by IBPSolution.certify; no public constructor.

    Exact identity replay and probabilistic master counts are separate evidence.
    Neither a stable residual set nor a count-consistent result proves closure
    or independence of the residual integrals.

    Examples
    --------
    >>> from symbolica import S
    >>> from symbolica.community import hepkit as hep
    >>> d, k, m2 = S("d", "k", "m2")
    >>> kin = hep.Kinematics(d, momenta=[k])
    >>> family = hep.IntegralFamily([k], [], [kin.scalar_product(k, k) - m2], kinematics=kin)
    >>> ibp = hep.IBPFamily(family, name="T")
    >>> solution = ibp.reduce_laporta([[3]], max_depth=1)
    >>> certificate = solution.certify(seed=0)
    >>> assert certificate.reduction == "verified"
    """
    @property
    def reduction(self) -> str:
        """Exact identity replay status: verified or unchecked.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.reduction == "verified"
        """
    @property
    def masters(self) -> str:
        """unchecked, incomplete, no-verdict, or count-consistent.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.masters == "count-consistent"
        """
    @property
    def master_counts(self) -> dict[tuple[bool, ...], int | None] | None:
        """Generic count per residual sector; None where no verdict is available.

        The whole mapping is None when master counting was not requested.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.master_counts == {(True,): 1}
        """
    @property
    def residual_counts(self) -> dict[tuple[bool, ...], int] | None:
        """Number of search residuals in each checked sector.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.residual_counts == {(True,): 1}
        """
    @property
    def excess_sectors(self) -> list[list[bool]]:
        """Sectors with more residuals than the counted master number.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.excess_sectors == []
        """
    @property
    def no_verdict(self) -> dict[tuple[bool, ...], str] | None:
        """Explanation for sectors without a count verdict.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.no_verdict == {}
        """
    @property
    def stable_depth(self) -> int | None:
        """First depth whose residual set survived two further search depths.

        None means no stable depth was recorded; this is not a closure test.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.stable_depth is None
        """
    @property
    def replayed_rules(self) -> int | None:
        """Number of rules reconstructed from the original exact identities.

        None means replay was not requested.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.replayed_rules > 0
        """
    @property
    def identities(self) -> int | None:
        """Number of instantiated original identities used during replay.

        None means replay was not requested.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.identities > 0
        """
    @property
    def seed(self) -> int | None:
        """Probabilistic counting seed; None when counting was not requested.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> assert certificate.seed == 0
        """
    def __repr__(self) -> str:
        """Summarize replay, counting, and stability evidence for inspection.

        Examples
        --------
        Using the setup in the ``IBPCertificate`` class example:

        >>> summary = repr(certificate)
        """
