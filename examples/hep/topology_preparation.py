import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="B → ηc: two-loop topology preparation",
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # B → ηc: two-loop topology preparation

    [Browse notebooks](/) · [Integral families](/?file=hep/integral_families.py) ·
    [Two-loop IBP](/?file=hep/ibp_two_loop_masses.py)

    Run the full input from FeynCalc's
    [topology-minimization example](https://feyncalc.github.io/FeynCalcExamples/TopologyIdentification/TwoLoops/B-EtaC):
    **251 integrals in 248 distinct families**, built from 89 propagators.
    The denominators mix quadratic and eikonal terms, with light-cone vectors
    $n^2=\bar n^2=0$, $n\cdot\bar n=2$.

    Shared `IntegralFamily` operations decompose dependent denominators,
    check the resulting independent sectors, and complete each basis with
    propagators drawn from the full input. Every partial-fraction identity and
    every reconstructed scalar product is checked exactly.

    The surviving sectors are grouped with verified loop-momentum maps. Every
    input integral is expressed in completed representative families, retaining
    its coefficients and propagator powers. The grouping keeps external momenta
    fixed and treats different propagator counts separately. Cross-size
    subtopology minimization remains to be integrated into this workflow.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Setup and notebook helpers

    Imports and supporting routines are folded below. Expand a cell’s code to
    inspect or edit it; the calculation that follows shows the HEP operations.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    from functools import reduce
    from pathlib import Path

    import marimo as mo
    from symbolica import E, Replacement, S
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import IntegralFamily, Kinematics

    _set_namespace("etac")
    return (
        E,
        IntegralFamily,
        Kinematics,
        Path,
        Replacement,
        S,
        json,
        mo,
        reduce,
    )


@app.cell(hide_code=True)
def _(E, Replacement, S, k1, k2, kin, pool):
    def complete_families(families):
        """Certify completions, scaleless sectors and numerator relations for each family."""
        statistics = {
            "scaleless certificate": 0,
            "not detected": 0,
            "transverse certificate": 0,
        }
        completed_rows = []
        for _family in families:
            parameters = [S(f"x{_i}") for _i in range(len(_family.denominators))]
            U, F = _family.symanzik(parameters)
            if U == E("0"):
                direction = _family.scaleless_transverse_direction()
                assert direction is not None and any(_w != E("0") for _w in direction)
                assert all(_w.is_real() is True for _w in direction)
                # Independently shift each loop by w_i*r_perp. External products stay
                # fixed; z_i=r_perp.k_i and z2=r_perp^2 are independent formal symbols.
                loops = [k1, k2]
                z = S("etac_shift::z1", "etac_shift::z2")
                z_squared = S("etac_shift::squared")
                shifts = [
                    Replacement(
                        kin.scalar_product(_ki, _kj),
                        kin.scalar_product(_ki, _kj)
                        + direction[_i] * z[_j]
                        + direction[_j] * z[_i]
                        + direction[_i] * direction[_j] * z_squared,
                    )
                    for _i, _ki in enumerate(loops)
                    for _j, _kj in enumerate(loops[_i:], _i)
                ]
                for _denominator in _family.denominators:
                    assert (
                        _denominator.replace_multiple(shifts) - _denominator
                    ).together() == E("0")
                status = "transverse certificate"
            else:
                weights = _family.scaleless_scaling(parameters)
                status = "not detected" if weights is None else "scaleless certificate"
                if weights is not None:
                    G = U + F
                    assert (
                        sum(
                            (
                                _w * _x * G.derivative(_x)
                                for _w, _x in zip(weights, parameters)
                            ),
                            E("0"),
                        )
                        - G
                    ).expand() == E("0")
            statistics[status] += 1
            if status == "not detected":
                # All surviving families must pass automatic self-mapping before any
                # global minimization can rely on the shift search. Fractional
                # light-cone offsets previously broke candidate reconstruction.
                _mapping = _family.find_mapping(_family)
                assert _mapping is not None
                for _source, _target in enumerate(_mapping.denominator_map):
                    assert (
                        _mapping.apply(_family.denominators[_source])
                        - _family.denominators[_target]
                    ).together() == E("0")
            completed = _family.complete(candidates=pool)
            assert completed.is_complete and completed.is_independent
            assert (
                completed.denominators[: len(_family.denominators)]
                == _family.denominators
            )
            assert all(_d in pool for _d in completed.denominators)
            assert (
                completed.complete(candidates=pool).denominators
                == completed.denominators
            )
            # Reconstruct every independent loop scalar product from the selected basis.
            names = [S(f"d{_i}") for _i in range(len(completed.denominators))]
            rules = completed.scalar_product_rules(names)
            for _scalar_product, _expression in rules:
                for _name, _denominator in zip(names, completed.denominators):
                    _expression = _expression.replace(_name, _denominator)
                assert (_expression - _scalar_product).together() == E("0")
            completed_rows.append((_family, completed, status))
        return statistics, completed_rows

    return (complete_families,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
def _(Path, json):
    fixture = json.loads(
        (Path(__file__).parent / "data/feyncalc_etac_topologies.json").read_text()
    )
    return (fixture,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Declare the scalar-product basis
    """)
    return


@app.cell
def _(E, Kinematics, S, fixture):
    D, k1, k2, n, nb, gkin, meta, u0b = S(
        "D",
        "k1",
        "k2",
        "n",
        "nb",
        "gkin",
        "meta",
        "u0b",
        is_real=True,
    )
    kin = (
        Kinematics(D, momenta=[k1, k2, n, nb])
        .with_scalar_product(n, n, E("0"))
        .with_scalar_product(nb, nb, E("0"))
        .with_scalar_product(n, nb, E("2"))
    )
    momenta = dict(zip(("k1", "k2", "n", "nb"), (k1, k2, n, nb)))
    labels = S(*fixture["labels"], is_real=True)
    products = [
        kin.scalar_product(momenta[_a], momenta[_b])
        for _a, _b in fixture["scalar_products"]
    ]
    return gkin, k1, k2, kin, labels, meta, n, nb, products, u0b


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Read the denominator pool

    Verify the captured input before preparing any families.
    """)
    return


@app.cell
def _(E, fixture, gkin, k1, k2, kin, labels, meta, n, nb, products, u0b):
    pool = []
    for _text in fixture["denominators"]:
        _denominator = E(_text)
        for _label, _product in zip(labels, products):
            _denominator = _denominator.replace(_label, _product)
        pool.append(_denominator.expand())
    assert len(pool) == len(set(pool)) == 89
    assert len(fixture["integrals"]) == 251
    assert (
        len({tuple(sorted(_row["propagators"])) for _row in fixture["integrals"]})
        == 248
    )
    # Two direct source checks fix the SFAD mass-term sign and the mixed linear form.
    assert pool[0] == kin.scalar_product(k1, k1)
    assert (
        pool[2]
        - kin.scalar_product(k1 + k2, k1 + k2)
        - 2 * gkin * meta * u0b * kin.scalar_product(k1 + k2, n)
        + meta * u0b * kin.scalar_product(k1 + k2, nb)
        + 2 * gkin * meta**2 * u0b**2
    ).expand() == E("0")
    return (pool,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build and partial-fraction the sectors
    """)
    return


@app.cell
def _(E, IntegralFamily, fixture, k1, k2, kin, n, nb, pool, reduce):
    sectors = {}
    input_rows = []
    for _number, _row in enumerate(fixture["integrals"], 1):
        denominators = [pool[_i] for _i in _row["propagators"]]
        _family = IntegralFamily([k1, k2], [n, nb], denominators, kinematics=kin)
        _fractions = _family.partial_fraction(_row["powers"])
        original = reduce(
            lambda a, b: a * b,
            (_d ** (-_p) for _d, _p in zip(denominators, _row["powers"])),
            E("1"),
        )
        reconstructed = E("0")
        for _coefficient, _powers in _fractions:
            reconstructed += _coefficient * reduce(
                lambda a, b: a * b,
                (_d ** (-_p) for _d, _p in zip(denominators, _powers)),
                E("1"),
            )
            _sector = _family.sector(_powers)
            assert _sector.is_independent
            _key = tuple(
                sorted(_d.to_canonical_string() for _d in _sector.denominators)
            )
            sectors.setdefault(_key, _sector)
        assert (original - reconstructed).together() == E("0"), _number
        input_rows.append((_family, _fractions))
    assert len(sectors) == 677
    return input_rows, sectors


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Complete and classify each family

    The folded routine checks every completion identity and scaleless-sector certificate. The expected inventory stays visible here.
    """)
    return


@app.cell
def _(complete_families, sectors):
    statistics, completed_rows = complete_families(sectors.values())
    assert statistics == {
        "scaleless certificate": 112,
        "not detected": 434,
        "transverse certificate": 131,
    }

    # Group physical sectors before adding auxiliary denominators. Complete only
    # the retained representatives when constructing final integral coordinates.
    return completed_rows, statistics


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Find exact mappings among survivors

    Native mappings preserve numerator relations and denominator powers.
    """)
    return


@app.cell
def _(E, IntegralFamily, completed_rows, k1, k2, kin):
    surviving_rows = [_row for _row in completed_rows if _row[2] == "not detected"]
    surviving = [_row[0] for _row in surviving_rows]
    mappings = IntegralFamily.find_mappings(surviving)
    representatives = sorted({_target for _target, _ in mappings})
    for _source, (_target, _mapping) in enumerate(mappings):
        assert mappings[_target][0] == _target
        for _i, _j in enumerate(_mapping.denominator_map):
            assert (
                _mapping.apply(surviving[_source].denominators[_i])
                - surviving[_target].denominators[_j]
            ).together() == E("0")
        # Independently check the two-loop Jacobian in the returned coordinates.
        images = [_image for _, _image in _mapping.momentum_rules]
        coefficients = [dict(_image.coefficient_list(k1, k2)) for _image in images]
        determinant = (
            coefficients[0].get(k1, E("0")) * coefficients[1].get(k2, E("0"))
            - coefficients[0].get(k2, E("0")) * coefficients[1].get(k1, E("0"))
        ).together()
        assert determinant in (E("1"), E("-1"))
        assert (
            _mapping.apply(kin.scalar_product(k1, k2)) - kin.scalar_product(*images)
        ).together() == E("0")
    assert [
        _target
        for _target, _ in IntegralFamily.find_mappings(
            [surviving[_i] for _i in representatives]
        )
    ] == list(range(len(representatives)))

    # Map every nonzero partial-fraction term from every original input to the
    # completed representative. Preserve coefficients and propagator powers.
    return mappings, representatives, surviving, surviving_rows


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Reconstruct the original integrals

    Apply the certified maps to each partial-fraction term and keep explicit checks for all input integrals.
    """)
    return


@app.cell
def _(
    E,
    input_rows,
    k1,
    k2,
    mappings,
    representatives,
    surviving,
    surviving_rows,
):
    survivor_indices = {
        tuple(sorted(_d.to_canonical_string() for _d in _family.denominators)): _i
        for _i, _family in enumerate(surviving)
    }
    final_integrals = []
    for _family, _fractions in input_rows:
        _terms = {}
        for _coefficient, _powers in _fractions:
            assert all(_power >= 0 for _power in _powers)
            _sector = _family.sector(_powers)
            _key = tuple(
                sorted(_d.to_canonical_string() for _d in _sector.denominators)
            )
            if _key not in survivor_indices:
                continue  # Removed only by one of the independently checked certificates.
            _source = survivor_indices[_key]
            alignment = _sector.mapping_to(surviving[_source], [k1, k2])
            assert alignment is not None
            source_powers = alignment.map_powers(
                [_power for _power in _powers if _power > 0]
            )
            _target, _mapping = mappings[_source]
            mapped_powers = _mapping.map_powers(source_powers)
            completed = surviving_rows[_target][1]
            mapped_powers += [0] * (len(completed.denominators) - len(mapped_powers))
            _term = (_target, tuple(mapped_powers))
            _terms[_term] = _terms.get(_term, E("0")) + _coefficient
        final_integrals.append(
            {
                _term: _coefficient.together()
                for _term, _coefficient in _terms.items()
                if _coefficient.together() != E("0")
            }
        )
    assert len(final_integrals) == 251
    assert len(representatives) == 223
    assert sum(len(_terms) for _terms in final_integrals) == 551
    assert sum(not _terms for _terms in final_integrals) == 59
    return (final_integrals,)


@app.cell(hide_code=True)
def _(fixture, mo, statistics):
    mo.vstack(
        [
            mo.md("## Verified preparation"),
            mo.ui.table(
                [
                    {"Classification": name, "Independent sectors": count}
                    for name, count in statistics.items()
                ],
                selection=None,
            ),
            mo.md(r"""
        This partial-fraction ordering produces **677 distinct sectors**.
        All have independent denominators, and all can be completed to the
        seven-dimensional loop scalar-product basis using the supplied pool.
        There are **112 parametric scaling certificates** and **131 transverse
        certificates**. For the latter, an explicit loop shift leaves every
        denominator unchanged, isolating a polynomial transverse integral that
        vanishes in dimensional regularization. The other 434 sectors remain
        unclassified. A singular quadratic form alone does not prove zero.
        Different decomposition orders can produce different intermediate counts.

        Use `family.complete(candidates=pool)` to prefer existing propagators;
        `family.complete()` uses bare scalar products. Original propagators
        retain their positions, dependent candidates are skipped, and bare scalar
        products fill any missing directions. Completion does not add physical
        propagators to an integral: auxiliary powers initially remain zero.
        """),
            mo.md(
                f"[Pinned mathematical input]({fixture['source']}) · Source SHA-256: `{fixture['source_sha256']}`"
            ),
            mo.md(
                "The fixture omits the common +iη prescription for these algebraic checks. No contour equivalence is inferred."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    input_index = mo.ui.slider(1, 251, value=1, step=1, label="Input integral")
    sector_index = mo.ui.slider(1, 677, value=1, step=1, label="Independent sector")
    mo.hstack([input_index, sector_index])
    return input_index, sector_index


@app.cell
def _(final_integrals, fixture, input_index, input_rows, mo, topology_names):
    selected_family, selected_fractions = input_rows[input_index.value - 1]
    _input = fixture["integrals"][input_index.value - 1]
    mo.vstack(
        [
            mo.md("## Input and partial fractions"),
            selected_family,
            mo.md(f"**Original powers:** {_input['powers']}"),
            mo.ui.table(
                [
                    {
                        "Coefficient": str(coefficient),
                        "Powers in original order": str(powers),
                    }
                    for coefficient, powers in selected_fractions
                ],
                selection=None,
            ),
            mo.md(
                "The exact rational difference between the input and this sum is zero."
            ),
            mo.md("### In completed representative families"),
            mo.ui.table(
                [
                    {
                        "Topology": topology_names[target],
                        "Coefficient": str(coefficient),
                        "Powers": str(powers),
                    }
                    for (target, powers), coefficient in final_integrals[
                        input_index.value - 1
                    ].items()
                ],
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(completed_rows, mo, sector_index):
    original_sector, completed_sector, sector_status = completed_rows[
        sector_index.value - 1
    ]
    added_count = len(completed_sector.denominators) - len(original_sector.denominators)
    mo.vstack(
        [
            mo.md("## Complete the selected sector"),
            mo.md(
                f"**Classification:** {sector_status}. **Auxiliary propagators added:** {added_count}."
            ),
            original_sector,
            completed_sector,
            mo.md(
                "All seven loop scalar products reconstruct exactly from the completed basis. Every auxiliary entry comes from the 89-propagator pool."
            ),
        ]
    )
    return (original_sector,)


@app.cell
def _(mo, original_sector):
    transverse_direction = original_sector.scaleless_transverse_direction()
    if transverse_direction is not None:
        mo.output.append(
            mo.md(r"""
        ### Transverse certificate
        The entries below define $k_i \mapsto k_i+w_i r_\perp$, with
        $r_\perp\cdot n=r_\perp\cdot\bar n=0$. Every denominator is
        unchanged. The unconstrained transverse integral vanishes for any
        polynomial numerator in dimensional regularization.
        """)
        )
        mo.output.append(
            mo.ui.table(
                [
                    {"Loop": str(loop), "Weight": str(weight)}
                    for loop, weight in zip(
                        original_sector.loop_momenta, transverse_direction
                    )
                ],
                selection=None,
            )
        )
    return


@app.cell
def _(final_integrals, mappings, mo, representatives):
    topology_names = {target: f"T{i + 1}" for i, target in enumerate(representatives)}
    mo.md(f"""
    ## Verified topology mappings

    **{len(mappings)} surviving families → {len(representatives)} representatives.**
    Every map preserves the loop measure and matches every denominator exactly.
    All representatives have complete seven-propagator bases; auxiliary powers
    remain zero. {sum(not terms for terms in final_integrals)} of the 251 input
    integrals vanish after certified scaleless terms are removed and equal
    mapped terms are combined.

    `IntegralFamily.find_mappings(families)` returns `(target_index, mapping)`
    for each input, with indices into the original list. Earlier families are
    preferred, and every map goes directly to a representative.
    """)
    return (topology_names,)


@app.cell
def _(mappings, mo):
    mapping_index = mo.ui.slider(
        1, len(mappings), value=1, step=1, label="Surviving family"
    )
    mo.output.append(mapping_index)
    return (mapping_index,)


@app.cell
def _(mapping_index, mappings, mo, surviving_rows, topology_names):
    mapped_target, selected_mapping = mappings[mapping_index.value - 1]
    mapped_source = surviving_rows[mapping_index.value - 1][0]
    mapped_family = surviving_rows[mapped_target][1]
    mo.vstack(
        [
            mo.md(
                f"### Family {mapping_index.value} → {topology_names[mapped_target]}"
            ),
            mo.ui.table(
                [
                    {"Source momentum": str(momentum), "Target coordinates": str(image)}
                    for momentum, image in selected_mapping.momentum_rules
                ],
                selection=None,
            ),
            mo.md(
                f"**Propagator map (zero-based):** {selected_mapping.denominator_map}"
            ),
            mapped_source,
            mapped_family,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
