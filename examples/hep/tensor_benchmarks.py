# /// script
# requires-python = ">=3.11"
# dependencies = ["symbolica==3.0.1", "marimo==0.24.0", "typst==0.15.0"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Tensor reduction benchmarks")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Tensor reduction benchmarks

    [Browse all notebooks](/) · [Gamma algebra tutorial](/?file=hep/gamma_simplification.py)

    Compare complete reductions with exact tensor identities and independent
    component oracles. Native FORM and ladder timing runs are opt-in.
    Construction and rendering are excluded where stated. Historical snapshots
    are preserved separately in `data/tensor_benchmark_history.md`.
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
    from symbolica.community import tensor as sp
    from math import prod
    from time import perf_counter
    import marimo as mo
    from gamma_simplification import app as gamma_examples

    return gamma_examples, mo, perf_counter, prod, sp


@app.cell(hide_code=True)
def _(
    AUTO,
    E,
    S,
    TensorName,
    gamma,
    lorentz,
    slash,
    slash_length,
    slash_pattern,
    sp,
    tr,
):
    from functools import reduce
    from operator import mul

    _metric = sp.TensorPattern.dot
    _p, _q = (
        TensorName.vector(f"gamma_benchmark::{name}").to_expression()
        for name in ("p", "q")
    )
    _n = slash_length.value
    _m = _n // 2
    momentum_names = (
        ["p", "p"] + ["q"] * (_n - 2)
        if slash_pattern.value == "paired"
        else ["p", "q"] * _m
    )
    _momenta = [{"p": _p, "q": _q}[name] for name in momentum_names]
    _indices = [lorentz(S(f"gamma_benchmark::mu{i}")) for i in range(_n)]
    bare_trace = tr(*(gamma(AUTO, AUTO, i) for i in _indices))
    momentum_factors = reduce(
        mul, (p(i.to_expression()) for p, i in zip(_momenta, _indices))
    )
    indexed_trace = momentum_factors * bare_trace
    compact_trace = tr(*(slash(p(lorentz.to_expression())) for p in _momenta))

    _pp, _qq, _pq = S(
        "gamma_benchmark::pp", "gamma_benchmark::qq", "gamma_benchmark::pq"
    )
    if slash_pattern.value == "paired":
        scalar_oracle = 4 * _pp * _qq ** (_m - 1)
    else:
        _previous, scalar_oracle = E("4"), 4 * _pq
        for _ in range(2, _m + 1):
            _previous, scalar_oracle = (
                scalar_oracle,
                (2 * _pq * scalar_oracle - _pp * _qq * _previous).expand(),
            )
    _p_compact, _q_compact = _p(lorentz.to_expression()), _q(lorentz.to_expression())
    scalar_expected = (
        scalar_oracle.replace(_pp, _metric(_p_compact, _p_compact))
        .replace(_qq, _metric(_q_compact, _q_compact))
        .replace(_pq, _metric(_p_compact, _q_compact))
    )
    return (
        bare_trace,
        compact_trace,
        indexed_trace,
        momentum_factors,
        momentum_names,
        scalar_expected,
        scalar_oracle,
    )


@app.cell(hide_code=True)
def _(
    E,
    bare_trace,
    compact_trace,
    indexed_trace,
    momentum_factors,
    perf_counter,
    scalar_expected,
):
    from statistics import median

    # Inputs are built outside the timed region. Every timed route computes
    # the complete scalar result; no precomputed trace is reused in a route.
    _routes = {
        "Trace first": lambda: (
            (
                momentum_factors
                * bare_trace.simplify_algebra(contract="dots", gamma=True, epsilon=True)
            )
            .contract()
            .to_dots()
            .expand()
        ),
        "Contract first": lambda: (
            indexed_trace.contract()
            .simplify_algebra(contract="dots", gamma=True, epsilon=True)
            .expand()
        ),
        "Compact input": lambda: compact_trace.simplify_algebra(
            contract="dots", gamma=True, epsilon=True
        ).expand(),
    }
    _samples = {name: [] for name in _routes}
    benchmark_results = {}
    for _name, _operation in _routes.items():
        _result = _operation()  # one untimed warm-up per route
        assert (_result.to_expression() - scalar_expected).expand() == E("0"), _name
    _names = list(_routes)
    for _round in range(3):
        # Rotate order to reduce systematic first/last-route bias.
        for _name in _names[_round:] + _names[:_round]:
            _start = perf_counter()
            _result = _routes[_name]()
            _samples[_name].append(perf_counter() - _start)
            assert (_result.to_expression() - scalar_expected).expand() == E("0"), _name
            benchmark_results[_name] = _result
    benchmark_seconds = {name: median(samples) for name, samples in _samples.items()}
    benchmark_speedup = (
        benchmark_seconds["Trace first"] / benchmark_seconds["Contract first"]
    )
    return benchmark_results, benchmark_seconds, benchmark_speedup, median


@app.cell(hide_code=True)
def _(
    AUTO,
    S,
    compact_form_source,
    gamma,
    generic_form_source,
    lorentz,
    median,
    perf_counter,
    time_large_python_trace,
    tr,
):
    import os
    import shutil
    import sys

    form_records = []
    form_version = None
    idenso_native_record = None
    # Browser notebooks cannot launch native subprocesses. Native users can
    # select an installed binary explicitly without changing the notebook.
    form_executable = (
        None
        if sys.platform == "emscripten"
        else os.environ.get("FORM_EXECUTABLE") or shutil.which("form")
    )
    if form_executable:
        import re
        import subprocess
        import tempfile
        from pathlib import Path

        form_version = subprocess.run(
            [form_executable, "-v"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        with tempfile.TemporaryDirectory(prefix="gamma-trace-form-") as _directory:
            _path = Path(_directory) / "trace.frm"
            for _label, _source in (
                ("Compact scalar / trace4", compact_form_source),
                (
                    "Compact scalar / tracen",
                    compact_form_source.replace("trace4,1;", "tracen,1;"),
                ),
                ("Fourteen free indices / trace4", generic_form_source),
                (
                    "Fourteen free indices / tracen",
                    generic_form_source.replace("trace4,1;", "tracen,1;"),
                ),
            ):
                _path.write_text(_source)
                _samples = []
                for _round in range(4):
                    _start = perf_counter()
                    _run = subprocess.run(
                        [form_executable, "-q", str(_path)],
                        cwd=_directory,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    _seconds = perf_counter() - _start
                    if _run.returncode:
                        raise RuntimeError(
                            f"FORM failed for {_label}:\n{_run.stdout}\n{_run.stderr}"
                        )
                    assert re.search(
                        r"CHECK=\s*0\s*(?:;|$)", _run.stdout, re.MULTILINE
                    ), _run.stdout
                    _terms = re.search(r"NTERMS=(\d+)", _run.stdout)
                    assert _terms, _run.stdout
                    if _round:
                        _samples.append(_seconds)
                form_records.append(
                    {
                        "case": _label,
                        "terms": int(_terms.group(1)),
                        "process median (ms)": round(1000 * median(_samples), 3),
                        "source": _source,
                        "stdout": _run.stdout,
                    }
                )
        if time_large_python_trace.value:
            _idenso_input = tr(
                *(
                    gamma(AUTO, AUTO, lorentz(S(f"gamma_form::mu{i}")))
                    for i in range(14)
                )
            )
            _start = perf_counter()
            _idenso_result = _idenso_input.simplify_algebra(
                contract="dots", gamma=True, epsilon=True
            )
            _first = perf_counter() - _start
            _samples = []
            for _ in range(3):
                _start = perf_counter()
                _idenso_result = _idenso_input.simplify_algebra(
                    contract="dots", gamma=True, epsilon=True
                )
                _samples.append(perf_counter() - _start)
            _count = len(list(_idenso_result.expand().to_expression().terms()))
            assert _count == 26931
            idenso_native_record = {
                "case": "Idenso / fourteen free indices",
                "terms": _count,
                "first call (ms)": round(1000 * _first, 3),
                "warm in-process median (ms)": round(1000 * median(_samples), 3),
            }
    return form_executable, form_records, form_version, idenso_native_record


@app.cell(hide_code=True)
def _(
    E,
    S,
    TensorExpression,
    TensorName,
    form_executable,
    g5,
    lorentz,
    slash,
    sp,
    tr,
):
    from itertools import permutations

    from symbolica import Expression
    from symbolica.community.tensor import (
        Tensor,
        TensorLibrary,
        TensorNetwork,
    )

    _names = [TensorName.vector(f"gamma_hep::p{i}") for i in range(10)]
    _metric = sp.TensorPattern.dot
    _momenta = [name.to_expression()(lorentz.to_expression()) for name in _names]
    _base = [
        [2, 1, 0, 1],
        [1, 2, 1, -1],
        [3, -1, 2, 1],
        [1, 0, -1, 2],
        [2, -1, -2, 1],
        [-1, 2, 1, 2],
        [3, 1, -1, 0],
        [1, 1, 2, -2],
        [-2, 1, 3, -1],
        [2, 2, -1, 3],
    ]
    _epsilon_name = TensorName(sp.TensorName.levi_civita().to_expression().get_name())
    _epsilon = Tensor.sparse(
        _epsilon_name(lorentz, lorentz, lorentz, lorentz), Expression
    )
    for _perm in permutations(range(4)):
        _inversions = sum(
            _perm[i] > _perm[j] for i in range(4) for j in range(i + 1, 4)
        )
        _epsilon[_perm] = E("-1𝑖" if _inversions % 2 == 0 else "1𝑖")

    _expressions = {}
    for _length in (4, 8, 10):
        for _axial in (False, True):
            _factors = [slash(p) for p in _momenta[:_length]]
            if _axial:
                _factors.insert(0, g5)
            _original = tr(*_factors).to_expression()
            _expressions[_length, _axial] = {
                "Original gamma network": _original,
                "Idenso metric/epsilon network": TensorExpression(_original)
                .simplify_algebra(contract="dots", gamma=True, epsilon=True)
                .to_expression(),
            }

    hep_component_form_sources = {}
    if form_executable:
        import re as _re
        import subprocess as _subprocess
        import tempfile as _tempfile
        from pathlib import Path as _Path

        with _tempfile.TemporaryDirectory(prefix="gamma-hep-form-") as _directory:
            _path = _Path(_directory) / "components.frm"
            for _length in (4, 8, 10):
                _word = ",".join(f"p{i}" for i in range(_length))
                for _mode in ("trace4", "tracen"):
                    _source = (
                        f"Off Statistics;\nVectors {_word};\nLocal F=g_(1,{_word});\n"
                        f'{_mode},1;\n.sort\n#write "RESULT=%E",F\n.end\n'
                    )
                    _path.write_text(_source)
                    _run = _subprocess.run(
                        [form_executable, "-q", str(_path)],
                        cwd=_directory,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    _polynomial = (
                        _run.stdout.split("RESULT=", 1)[1].strip().removesuffix(";")
                    )
                    # Import each FORM scalar product as a Spenso metric tensor.
                    # Symbolica parses the polynomial; no Python eval is used.
                    _polynomial = _re.sub(
                        r"p(\d+)\.p(\d+)",
                        lambda match: f"gamma_hep::dot{match[1]}x{match[2]}",
                        _polynomial,
                    )
                    _expression = E(_polynomial)
                    for _i in range(_length):
                        for _j in range(_i, _length):
                            _expression = _expression.replace(
                                S(f"gamma_hep::dot{_i}x{_j}"),
                                _metric(_momenta[_i], _momenta[_j]),
                            )
                    _expressions[_length, False][f"FORM {_mode} metric network"] = (
                        _expression
                    )
                    hep_component_form_sources[f"{_length} / {_mode}"] = _source

    hep_component_checks = []
    for _sample in range(2):
        _library = TensorLibrary.hep_lib_atom()
        _library.register(_epsilon)
        for _name, _components in zip(_names, _base, strict=True):
            # Sparse Expression storage forces exact components; automatic
            # dense input conversion can otherwise choose floating-point data.
            _tensor = Tensor.sparse(_name(lorentz), Expression)
            for _axis, _value in enumerate(_components):
                _tensor[_axis] = E(
                    str(_value if _sample == 0 else _value * (1, -1, 2, 1)[_axis])
                )
            _library.register(_tensor)

        def _evaluate(_expression, _library=_library):
            _network = TensorNetwork(_expression, library=_library)
            _network.execute(library=_library)
            return _network.result_scalar()

        assert _evaluate(_epsilon_name.to_expression()(*_momenta[:4])) != E("0")
        for (_length, _axial), _routes in _expressions.items():
            _expected = _evaluate(_routes["Original gamma network"])
            for _route, _expression in _routes.items():
                _value = _evaluate(_expression)
                assert _value == _expected, (_sample, _length, _axial, _route)
                hep_component_checks.append(
                    {
                        "sample": _sample + 1,
                        "gammas": _length,
                        "gamma5": _axial,
                        "route": _route,
                        "exact HEP value": _value.format_plain(),
                        "agrees": True,
                    }
                )
    return hep_component_checks, hep_component_form_sources


@app.cell(hide_code=True)
def _(E, S, TensorName, lorentz, prod):
    from tensor_benchmark_cases import HistoricalLadder as _HistoricalLadder

    (
        ladder_expected_counts,
        ladder_input,
        ladder_indices,
        ladder_metric,
        ladder_momenta,
        ladder_patterns,
        ladder_rhs,
        ladder_vertices,
        ladder_vertex_rule,
    ) = _HistoricalLadder.geometry(E, S, TensorName, lorentz, prod)
    return (
        ladder_expected_counts,
        ladder_indices,
        ladder_input,
        ladder_metric,
        ladder_momenta,
        ladder_patterns,
        ladder_rhs,
        ladder_vertex_rule,
        ladder_vertices,
    )


@app.cell(hide_code=True)
def _(
    S,
    ladder_indices,
    ladder_input,
    ladder_metric,
    ladder_momenta,
    ladder_vertex_rule,
):
    from tensor_benchmark_cases import HistoricalLadder as _HistoricalLadder

    (
        ladder_outside_absorb,
        ladder_outside_d,
        ladder_outside_input,
        ladder_outside_k,
        ladder_outside_rhs,
    ) = _HistoricalLadder.scalar_fixture(
        S,
        ladder_indices,
        ladder_input,
        ladder_metric,
        ladder_momenta,
        ladder_vertex_rule,
    )
    return (
        ladder_outside_absorb,
        ladder_outside_d,
        ladder_outside_input,
        ladder_outside_k,
        ladder_outside_rhs,
    )


@app.cell(hide_code=True)
def _(
    S,
    TensorName,
    ladder_patterns,
    ladder_vertex_rule,
    ladder_vertices,
    prod,
):
    from tensor_benchmark_cases import HistoricalLadder as _HistoricalLadder

    (
        ladder_native_input,
        _ladder_native_patterns,
        _ladder_native_rhs,
        ladder_native_rules,
    ) = _HistoricalLadder.native_fixture(
        S, TensorName, ladder_patterns, ladder_vertex_rule, ladder_vertices, prod
    )
    return ladder_native_input, ladder_native_rules


@app.cell(hide_code=True)
def _(ladder_order):
    # Preserve the supplied rules and routing; select the same order as Python.
    from tensor_benchmark_cases import HistoricalLadder as _HistoricalLadder

    (ladder_form_source,) = _HistoricalLadder.form_fixture(ladder_order)
    return (ladder_form_source,)


@app.cell(hide_code=True)
def _(
    form_executable,
    ladder_form_source,
    ladder_order,
    ladder_run_counts,
    median,
    perf_counter,
):
    import re as _re
    import subprocess as _subprocess
    import tempfile as _tempfile
    from pathlib import Path as _Path

    ladder_form_stages = []
    ladder_form_process_samples = []
    ladder_form_polynomial = None
    ladder_form_stdout = None
    if form_executable:
        with _tempfile.TemporaryDirectory(prefix="gluon-ladder-form-") as _directory:
            _path = _Path(_directory) / "gluon.frm"
            _path.write_text(ladder_form_source)
            for _round in range(4):
                _start = perf_counter()
                _run = _subprocess.run(
                    [form_executable, "-q", str(_path)],
                    cwd=_directory,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                _seconds = perf_counter() - _start
                if _run.returncode:
                    raise RuntimeError(
                        f"FORM ladder failed:\n{_run.stdout}\n{_run.stderr}"
                    )
                _stages = _re.findall(
                    r"Time =\s*([\d.]+) sec\s+Generated terms =\s*(\d+)"
                    r"\s+F\s+Terms in output =\s*(\d+)\s+gluon-(\d+)"
                    r"\s+Bytes used\s*=\s*(\d+)",
                    _run.stdout,
                )
                assert tuple(int(stage[2]) for stage in _stages) == ladder_run_counts, (
                    _run.stdout
                )
                assert [int(stage[3]) for stage in _stages] == list(ladder_order)
                if _round:
                    ladder_form_process_samples.append(_seconds)
                    ladder_form_stages.append(_stages)
            ladder_form_stdout = _run.stdout
            ladder_form_stages = [
                {
                    "vertex": ladder_order[_i],
                    "FORM terms": ladder_run_counts[_i],
                    "FORM generated terms": int(ladder_form_stages[0][_i][1]),
                    "FORM cumulative CPU (s)": median(
                        float(run[_i][0]) for run in ladder_form_stages
                    ),
                }
                for _i in range(8)
            ]
            # A separate, untimed run exports the exact polynomial for validation.
            _path.write_text(
                ladder_form_source.replace(
                    ".end", '#write <polynomial.txt> "%E",F\n.end'
                )
            )
            _check = _subprocess.run(
                [form_executable, "-q", str(_path)],
                cwd=_directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if _check.returncode:
                raise RuntimeError(
                    f"FORM polynomial export failed:\n{_check.stdout}\n{_check.stderr}"
                )
            ladder_form_polynomial = (_Path(_directory) / "polynomial.txt").read_text()
    return (
        ladder_form_polynomial,
        ladder_form_process_samples,
        ladder_form_stages,
        ladder_form_stdout,
    )


@app.cell(hide_code=True)
def _(
    E,
    S,
    ladder_form_polynomial,
    ladder_metric,
    ladder_momenta,
    ladder_result,
):
    import re as _re

    ladder_exact_match = None
    if ladder_result is not None and ladder_form_polynomial is not None:
        # Compare complete polynomials in independent scalar products. Term
        # counts alone cannot certify signs, coefficients or momentum routing.
        _scalar = ladder_result
        for _i, _p in ladder_momenta.items():
            for _j, _q in ladder_momenta.items():
                if _i <= _j:
                    _scalar = _scalar.replace(
                        ladder_metric(_p, _q), S(f"gluon_ladder::s{_i}x{_j}")
                    )
        _reference = E(
            _re.sub(
                r"k(\d+)\.k(\d+)",
                lambda match: "gluon_ladder::s{}x{}".format(
                    *sorted(map(int, match.groups()))
                ),
                ladder_form_polynomial.strip().rstrip(";"),
            )
        )
        ladder_exact_match = _scalar == _reference
        assert ladder_exact_match, "Spenso and FORM ladder polynomials differ"
    return (ladder_exact_match,)


@app.cell(hide_code=True)
def _(fermion_ladder_case, form_executable, mo, run_fermion_ladder):
    import sys as _sys
    from pathlib import Path as _Path

    fermion_ladder_report = None
    if run_fermion_ladder.value:
        mo.stop(
            _sys.platform == "emscripten" or not form_executable,
            mo.md(
                "Run this comparison locally with FORM on PATH or FORM_EXECUTABLE set."
            ),
        )
        _directory = str(_Path(__file__).resolve().parent)
        if _directory not in _sys.path:
            _sys.path.insert(0, _directory)
        from fermion_ladder import benchmark as _benchmark

        _options = {
            "loops": 3 if fermion_ladder_case.value == "Three-loop fermion" else 4
        }
        if fermion_ladder_case.value == "Four-loop gluon":
            from gluon_ladder import GluonLadder as _GluonLadder

            _options.update(
                case_factory=_GluonLadder,
                form_script=_Path(_directory) / "gluon_propagator_ladder.frm",
            )
        with mo.status.spinner(
            title="Comparing complete 4D and D-dimensional numerators"
        ):
            fermion_ladder_report = _benchmark(form_executable, **_options)
    return (fermion_ladder_report,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell
async def _(gamma_examples):
    _examples = await gamma_examples.embed()
    AUTO = _examples.defs["AUTO"]
    E = _examples.defs["E"]
    S = _examples.defs["S"]
    TensorExpression = _examples.defs["TensorExpression"]
    TensorName = _examples.defs["TensorName"]
    display_settings = _examples.defs["display_settings"]
    g5 = _examples.defs["g5"]
    gamma = _examples.defs["gamma"]
    lorentz = _examples.defs["lorentz"]
    slash = _examples.defs["slash"]
    tr = _examples.defs["tr"]
    return (
        AUTO,
        E,
        S,
        TensorExpression,
        TensorName,
        display_settings,
        g5,
        gamma,
        lorentz,
        slash,
        tr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Trace growth: measure a small example

    Generic-dimensional pairing recursion produces $(2n-1)!!$ terms for
    $2n$ distinct Lorentz indices. Four-dimensional traces now use generated
    kernels through length 14, derived from the three-gamma epsilon identity.
    At lengths 10, 12 and 14 they give **693, 4,383 and 26,931** metric terms,
    matching FORM 5.0.0 `trace4` for these inputs. Repeated indices and momenta
    can simplify earlier through the shared chain contractions.

    Each even arity has a lazily generated ordinary and gamma5 table; odd
    traces return zero. The first call includes initialization if this table
    has not already been used in the session. Warm timing excludes that cost.

    Change the length to time the installed simplifier. The measurement excludes
    rendering and counts expanded terms only for this small diagnostic. It is
    a local observation, not a large-amplitude performance guarantee.
    """)
    return


@app.cell
def _(mo):
    trace_length = mo.ui.slider(
        start=2,
        stop=10,
        step=2,
        value=6,
        label="Distinct gamma factors",
        show_value=True,
    )
    trace_length
    return (trace_length,)


@app.cell
def _(AUTO, S, gamma, lorentz, mo, perf_counter, prod, tr, trace_length):
    _length = trace_length.value
    _input = tr(
        *(
            gamma(AUTO, AUTO, lorentz(S(f"gamma_tutorial::ell_{index}")))
            for index in range(_length)
        )
    )
    _start = perf_counter()
    _result = _input.simplify_algebra(contract="dots", gamma=True, epsilon=True)
    trace_seconds = perf_counter() - _start
    trace_terms = len(list(_result.expand().to_expression().terms()))
    assert trace_terms == (1, 3, 15, 105, 693, 4383, 26931)[_length // 2 - 1]
    _samples = []
    for _ in range(3):
        _start = perf_counter()
        _input.simplify_algebra(contract="dots", gamma=True, epsilon=True)
        _samples.append(perf_counter() - _start)
    mo.ui.table(
        [
            {
                "gamma factors": _length,
                "expanded metric terms": trace_terms,
                "generic pairing terms": prod(range(1, _length, 2)),
                "first call (ms)": round(1000 * trace_seconds, 3),
                "warm median (ms)": round(1000 * sorted(_samples)[1], 3),
            }
        ],
        selection=None,
        pagination=False,
        show_download=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Schoonschip notation changes the work, not just the display

    Contracting $p_\mu\gamma^\mu$ into $\not p$ exposes repeated momenta
    before the trace generates metric pairings. Compare **the same scalar**
    through three routes:

    | Route | Ordered stages inside the timer |
    |:--|:--|
    | Trace first | Free-index trace → attach momenta → `contract()` → `expand()` |
    | Contract first | Indexed input → `contract()` → `simplify_algebra(gamma=True, epsilon=True)` → `expand()` |
    | Compact input | Constructed slash trace → `simplify_algebra(gamma=True, epsilon=True)` → `expand()` |

    The first two start from the same indexed problem; their ratio includes
    the early contraction cost. Compact input starts after that conversion,
    so its time is reported separately. The gamma engine is shared. The first
    difference is whether it sees distinct indices or repeated momenta.
    `DisplaySettings(tensor_layout="schoonschip")` only changes rendering and
    cannot produce this speedup. Here the network pass changes the expression.

    Paired slashes give
    $\mathrm{tr}(\not p\not p\not q^{\,2m-2})=4p^2(q^2)^{m-1}$.
    For alternating slashes, the independent oracle is
    $T_m=2(p\cdot q)T_{m-1}-p^2q^2T_{m-2}$ with
    $T_0=4$ and $T_1=4p\cdot q$.
    These relations do not require on-shell momenta.
    """)
    return


@app.cell
def _(mo):
    slash_length = mo.ui.dropdown([4, 6, 8, 10], value=8, label="Gamma factors")
    slash_pattern = mo.ui.radio(
        {"Adjacent pairs": "paired", "Alternating p, q": "alternating"},
        value="Adjacent pairs",
        inline=True,
        label="Momentum pattern",
    )
    mo.hstack([slash_length, slash_pattern], justify="start")
    return slash_length, slash_pattern


@app.cell
def _(compact_trace, display_settings, indexed_trace, mo):
    contracted_trace = indexed_trace.contract()
    assert contracted_trace.to_expression() == compact_trace.to_expression()
    mo.vstack(
        [
            mo.md(
                "**Same input, after contracting the momentum indices** — exact compact expression checked."
            ),
            mo.Html(indexed_trace.to_html(settings=display_settings)),
            mo.Html(contracted_trace.to_html(settings=display_settings)),
        ]
    )
    return


@app.cell
def _(
    bare_trace,
    benchmark_results,
    benchmark_seconds,
    benchmark_speedup,
    display_settings,
    mo,
    slash_length,
):
    metric_pairings = len(
        list(
            bare_trace.simplify_algebra(contract="dots", gamma=True, epsilon=True)
            .expand()
            .to_expression()
            .terms()
        )
    )
    assert (
        metric_pairings
        == (1, 3, 15, 105, 693, 4383, 26931)[slash_length.value // 2 - 1]
    )
    mo.vstack(
        [
            mo.ui.table(
                [
                    {
                        "route": name,
                        "median (ms)": round(1000 * seconds, 3),
                        "final scalar terms": len(
                            list(
                                benchmark_results[name].to_expression().expand().terms()
                            )
                        ),
                    }
                    for name, seconds in benchmark_seconds.items()
                ],
                selection=None,
                pagination=False,
                show_download=False,
            ),
            mo.md(
                f"**Measured early-contraction speedup: {benchmark_speedup:.2f}×.** The trace-first route builds **{metric_pairings}** free-index metric terms before seeing the repeated momenta. All three final scalar results agree exactly. Timings are medians of three warm runs, excluding construction, assertions and rendering; they depend on the installed build and hardware."
            ),
            mo.Html(
                benchmark_results["Contract first"].to_html(settings=display_settings)
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3. Are we at FORM `trace4` parity?

    [Recorded inputs, measurements, and analysis](data/tensor_benchmark_history.md#10-are-we-at-form-trace4-parity). These are historical results; use the controls below for the installed build.
    """)
    return


@app.cell
def _(mo):
    time_large_python_trace = mo.ui.checkbox(
        value=False,
        label="Also time fourteen free indices in Python (metadata inference can take minutes)",
    )
    time_large_python_trace
    return (time_large_python_trace,)


@app.cell
def _(momentum_names, scalar_oracle):
    _word = ",".join(momentum_names)
    # The oracle is a short rational polynomial in the declared pp, qq, pq.
    _oracle = str(scalar_oracle)
    compact_form_source = f"""Off Statistics;
    Vectors p,q;
    Symbols pp,qq,pq;
    Local F = g_(1,{_word});
    trace4,1;
    .sort
    id p.p=pp;
    id q.q=qq;
    id p.q=pq;
    .sort
    Local Check = F - ({_oracle});
    .sort
    #$nterms=termsin_(F);
    #write "NTERMS=%$",$nterms
    #write "CHECK=%E",Check
    .end
    """
    _indices = ",".join(f"mu{i}" for i in range(1, 15))
    _projection = "*".join(f"{'p' if i <= 2 else 'q'}(mu{i})" for i in range(1, 15))
    generic_form_source = f"""Off Statistics;
    Vectors p,q;
    Indices {_indices};
    Local F = g_(1,{_indices});
    trace4,1;
    .sort
    #$nterms=termsin_(F);
    #write "NTERMS=%$",$nterms
    Multiply {_projection};
    .sort
    Local Check = F - 4*p.p*(q.q)^6;
    .sort
    #write "CHECK=%E",Check
    .end
    """
    return compact_form_source, generic_form_source


@app.cell(hide_code=True)
def _(
    compact_form_source,
    form_executable,
    form_records,
    form_version,
    idenso_native_record,
    mo,
):
    if form_executable:
        mo.output.append(
            mo.md(
                f"**Native comparison:** `{form_version}`. Every FORM check reduced the scalar difference to zero. Process timings include startup, parsing, tracing, sorting and verification; they are not isolated trace-kernel times."
            )
        )
        mo.output.append(
            mo.ui.table(
                [
                    {
                        key: value
                        for key, value in record.items()
                        if key not in {"source", "stdout"}
                    }
                    for record in form_records
                ],
                selection=None,
                pagination=False,
                show_download=False,
            )
        )
        if idenso_native_record is not None:
            mo.output.append(
                mo.ui.table(
                    [idenso_native_record],
                    selection=None,
                    pagination=False,
                    show_download=False,
                )
            )
        _native_times = {
            record["case"]: record["process median (ms)"] for record in form_records
        }
        _trace4_speedup = (
            _native_times["Fourteen free indices / tracen"]
            / _native_times["Fourteen free indices / trace4"]
        )
        mo.output.append(
            mo.md(
                f"**Within FORM, the fourteen-index pipeline is {_trace4_speedup:.2f}× faster with `trace4` than `tracen`.** This compares the same native executable and projection check. Tiny compact traces may instead be dominated by startup time."
            )
        )
        mo.output.append(
            mo.accordion(
                {
                    record["case"]: mo.md(
                        f"```form\n{record['source']}\n```\n\n```text\n{record['stdout']}\n```"
                    )
                    for record in form_records
                }
            )
        )
    else:
        mo.output.append(
            mo.md(
                "**Native FORM comparison not run.** Browser notebooks cannot launch FORM. Locally, put `form` on `PATH` or set `FORM_EXECUTABLE` to its absolute path before starting Marimo. The program below reproduces the selected scalar check; no FORM timing is invented when it is unavailable."
            )
        )
        mo.output.append(mo.md(f"```form\n{compact_form_source}\n```"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Compare tensor values, allowing different simplifications

    The original trace contracts the HEP library's explicit 4×4 gamma
    matrices. The simplified expression contracts its metric and epsilon
    tensors against the **same exact integer four-vectors**. Neither route
    uses the other's symbolic answer as its oracle. Both samples span four
    dimensions, so gamma5 checks cannot pass merely because epsilon vanishes.
    We use the repository's (+−−−) metric and epsilon convention
    $\epsilon_{0123}=-i$, with exact Symbolica components throughout.

    When FORM is available, its ordinary `trace4` and `tracen` outputs are
    imported as metric networks and evaluated by that same HEP library.
    Length 10 is particularly useful: `trace4` has 693 terms and `tracen`
    has 945, yet their four-dimensional component values must agree.
    No comparison requires equal term counts or identical symbolic forms.

    These finite assignments are regression checks, not a proof for all
    tensors. The Rust HEP tests additionally cover every length 1–14,
    gamma5 positions, repeated momenta and cyclic contracted indices.

    ### Recorded comparison: gamma5 and twelve ordinary gammas

    For $\operatorname{Tr}(\gamma_5\gamma^{\mu_0}\cdots\gamma^{\mu_{11}})$,
    both Idenso's **first gamma rewrite** and FORM 5.0.0 `trace4` produce
    **1,029 terms**. The expressions are different: **741 terms coincide**,
    with **288 unique to each**. Their raw symbolic difference has 576 terms.
    The first rewrite equals Idenso's full pipeline output exactly.

    Three separate full-rank integer momentum assignments give these exact
    HEP component contractions of all twelve Lorentz indices:

    | Assignment | Original gamma network | Idenso first rewrite | FORM `trace4` |
    |--:|--:|--:|--:|
    | 1 | −96,536 i | −96,536 i | −96,536 i |
    | 2 | 468,485,024 i | 468,485,024 i | 468,485,024 i |
    | 3 | 114,798,012 i | 114,798,012 i | 114,798,012 i |

    FORM's `d_` and `e_` map to the same HEP metric and epsilon convention.
    The four-gamma axial trace is `4*e_(mu0,mu1,mu2,mu3)` in FORM and
    `4*epsilon(mu0,mu1,mu2,mu3)` in Idenso; no fitted sign is introduced.
    The [FORM metric conventions](https://form-dev.github.io/form-docs/master/manual/#a-few-notes-on-the-use-of-a-metric)
    specify this trace normalization.

    A fresh optimized run measures **2.42 ms** for the first rewrite versus
    **0.654 ms** amortized FORM wall time, about **3.7×** in FORM's favor.
    Full FORM process latency is 11.97 ms; its internal trace-and-sort CPU time
    is 0.624 ms. These are different timing boundaries. The first rewrite is
    already complete on this input, while Idenso's public pipeline still pays
    the previously measured 2.56 s for its extra passes.

    The controls above reproduce the component checks and FORM comparisons;
    the historical measurements are summarized in `data/tensor_benchmark_history.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Contraction order and intermediate terms

    [Recorded inputs, measurements, and analysis](data/tensor_benchmark_history.md#contraction-order-and-intermediate-terms). These are historical results; use the controls below for the installed build.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Generic-dimensional traces and contraction progress

    [Recorded inputs, measurements, and analysis](data/tensor_benchmark_history.md#generic-dimensional-traces-and-contraction-progress). These are historical results; use the controls below for the installed build.
    """)
    return


@app.cell(hide_code=True)
def _(hep_component_checks, hep_component_form_sources, mo):
    mo.vstack(
        [
            mo.ui.table(
                hep_component_checks,
                selection=None,
                pagination=False,
                show_download=False,
            ),
            mo.md(
                "FORM rows are included only when the executable is available; the HEP checks always run."
            ),
            mo.accordion(
                {
                    f"FORM component check: {name}": mo.md(f"```form\n{source}\n```")
                    for name, source in hep_component_form_sources.items()
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Ordered gluonic-ladder Feynman rules

    [Recorded inputs, measurements, and analysis](data/tensor_benchmark_history.md#ordered-gluonic-ladder-feynman-rules). These are historical results; use the controls below for the installed build.
    """)
    return


@app.cell
def _(ladder_input, ladder_patterns, ladder_rhs, median, perf_counter):
    ladder_substitution_samples = []
    for _round in range(6):
        _result = ladder_input
        _start = perf_counter()
        for _pattern in ladder_patterns:
            _result = _result.replace(_pattern, ladder_rhs, rhs_cache_size=1000)
        ladder_substitution_samples.append(perf_counter() - _start)
        if _round == 0:
            ladder_factored = _result
        else:
            assert _result == ladder_factored
    ladder_substitution_seconds = median(ladder_substitution_samples[1:])
    return ladder_substitution_samples, ladder_substitution_seconds


@app.cell
def _(mo):
    ladder_order_selector = mo.ui.dropdown(
        options=["Close rungs early", "Original vertex order"],
        value="Close rungs early",
        label="Contraction order (used by both engines)",
    )
    ladder_order_selector
    return (ladder_order_selector,)


@app.cell
def _(ladder_expected_counts, ladder_order_selector):
    if ladder_order_selector.value == "Close rungs early":
        ladder_order = (5, 4, 6, 3, 7, 2, 8, 1)
        ladder_run_counts = (6, 34, 63, 352, 829, 4586, 10516, 9652)
    else:
        ladder_order = tuple(range(1, 9))
        ladder_run_counts = ladder_expected_counts
    return ladder_order, ladder_run_counts


@app.cell
def _(mo):
    run_ladder_reduction = mo.ui.run_button(label="Run ladder comparisons")
    run_ladder_reduction
    return (run_ladder_reduction,)


@app.cell
def _(
    E,
    TensorExpression,
    ladder_order,
    ladder_patterns,
    ladder_rhs,
    ladder_run_counts,
    ladder_vertices,
    mo,
    perf_counter,
    run_ladder_reduction,
):
    ladder_stages = []
    ladder_result = None
    if run_ladder_reduction.value:
        _result = TensorExpression(E("1"))
        _cumulative = 0.0
        with mo.status.progress_bar(
            total=8, title="Applying ladder vertices"
        ) as _progress:
            for _i, _vertex_number in enumerate(ladder_order):
                _vertex = ladder_vertices[_vertex_number - 1]
                _pattern = ladder_patterns[_vertex_number - 1]
                _start = perf_counter()
                _factor = _vertex.replace(_pattern, ladder_rhs, rhs_cache_size=1000)
                # Carry the known interface between stages. Each newly applied
                # vertex is expanded against the accumulator before contraction.
                _result = (_result * TensorExpression(_factor)).contract().expand()
                _seconds = perf_counter() - _start
                _cumulative += _seconds
                _terms = sum(1 for _ in _result.to_expression().terms())
                assert _terms == ladder_run_counts[_i], (_vertex_number, _terms)
                ladder_stages.append(
                    {
                        "vertex": _vertex_number,
                        "Spenso terms": _terms,
                        "Spenso step wall (s)": _seconds,
                        "Spenso cumulative wall (s)": _cumulative,
                    }
                )
                _progress.update()
        ladder_result = _result.to_expression()
    return ladder_result, ladder_stages


@app.cell
def _(
    ladder_order,
    ladder_outside_absorb,
    ladder_outside_input,
    ladder_outside_rhs,
    ladder_patterns,
    ladder_result,
    mo,
    perf_counter,
    run_ladder_reduction,
):
    ladder_outside_stages = []
    ladder_outside_result = None
    if run_ladder_reduction.value and ladder_result is not None:
        _result = ladder_outside_input
        _cumulative = 0.0
        with mo.status.progress_bar(
            total=8, title="Outside-in Symbolica diagnostic"
        ) as _progress:
            for _vertex_number in ladder_order:
                _start = perf_counter()
                # Future vertices remain opaque. Their already-bound ports enter
                # the held RHS reduction and its existing match-result cache.
                _result = _result.replace(
                    ladder_patterns[_vertex_number - 1],
                    ladder_outside_rhs,
                    rhs_cache_size=1000,
                )
                _result = _result.expand().replace(
                    *ladder_outside_absorb, min_level=0, max_level=0, repeat=True
                )
                _seconds = perf_counter() - _start
                _cumulative += _seconds
                _terms = sum(1 for _ in _result.to_expression().terms())
                ladder_outside_stages.append(
                    {
                        "vertex": _vertex_number,
                        "outside-in terms": _terms,
                        "diagnostic step wall (s)": _seconds,
                        "diagnostic cumulative wall (s)": _cumulative,
                    }
                )
                _progress.update()
        ladder_outside_result = _result
    return ladder_outside_result, ladder_outside_stages


@app.cell
def _(
    ladder_exact_match,
    ladder_metric,
    ladder_momenta,
    ladder_outside_d,
    ladder_outside_k,
    ladder_outside_result,
    ladder_result,
):
    ladder_outside_typed_match = None
    ladder_outside_form_match = None
    if ladder_outside_result is not None:
        _converted = ladder_outside_result
        for _i, _p in ladder_momenta.items():
            for _j, _q in ladder_momenta.items():
                if _i <= _j:
                    _converted = _converted.replace(
                        ladder_outside_d(ladder_outside_k(_i), ladder_outside_k(_j)),
                        ladder_metric(_p, _q),
                    )
        ladder_outside_typed_match = _converted == ladder_result
        assert ladder_outside_typed_match, (
            "Outside-in and typed ladder polynomials differ"
        )
        if ladder_exact_match is not None:
            # Reuse the existing complete typed-to-FORM polynomial certificate.
            ladder_outside_form_match = (
                ladder_outside_typed_match and ladder_exact_match
            )
            assert ladder_outside_form_match, (
                "Outside-in and FORM ladder polynomials differ"
            )
    return ladder_outside_form_match, ladder_outside_typed_match


@app.cell(hide_code=True)
def _(
    ladder_outside_form_match,
    ladder_outside_stages,
    ladder_outside_typed_match,
    mo,
):
    mo.vstack(
        [
            mo.md("""
        ### Explicit outside-in Symbolica route

        This alternative keeps all unapplied vertices opaque. After applying one
        vertex, it contracts exposed indices into future vertex arguments; the
        next held rule is reduced locally using those bound arguments. It reuses
        the routing, six-term rule, order selector and run button above.

        This recipe uses a plain scalar `d` function and tagged index symbols.
        It exercises the supplied reduction rules without the full tensor-interface
        API; its final result is converted back to Spenso and checked exactly.

        These are **single-run diagnostic clocks**, not a controlled benchmark.
        Each step excludes term counting, notation conversion and equality checks.
        Intermediate term counts include the remaining opaque vertices, so they
        describe a different schedule from the typed accumulator above.
        """),
            mo.ui.table(ladder_outside_stages, selection=None)
            if ladder_outside_stages
            else mo.md("Use the shared ladder run button to execute the comparisons."),
            mo.md(
                f"**Exact final polynomial:** typed result `{ladder_outside_typed_match}`; "
                f"FORM `{ladder_outside_form_match}`. The FORM check reuses the full-polynomial "
                "certificate above after converting the outside-in notation back to Spenso."
            )
            if ladder_outside_typed_match is not None
            else mo.md(""),
        ]
    )
    return


@app.cell
def _(
    TensorExpression,
    ladder_native_input,
    ladder_native_rules,
    ladder_order,
    ladder_result,
    mo,
    perf_counter,
    run_ladder_reduction,
):
    ladder_native_stages = []
    ladder_native_match = None
    if run_ladder_reduction.value and ladder_result is not None:
        _result = TensorExpression(ladder_native_input)
        _cumulative = 0.0
        with mo.status.progress_bar(
            total=8, title="Binding future tensor vertices"
        ) as _progress:
            for _step, _vertex_number in enumerate(ladder_order, 1):
                _start = perf_counter()
                _result = _result.replace(
                    ladder_native_rules[_vertex_number - 1]
                ).contract()
                if _step == len(ladder_order):
                    # Include the complete final scalar polynomial in the clock.
                    _result = _result.expand()
                _seconds = perf_counter() - _start
                _cumulative += _seconds
                ladder_native_stages.append(
                    {
                        "vertex": _vertex_number,
                        "typed outside-in root terms": sum(
                            1 for _ in _result.to_expression().terms()
                        ),
                        "step wall (s)": _seconds,
                        "cumulative wall (s)": _cumulative,
                    }
                )
                _progress.update()
        ladder_native_match = _result.to_expression() == ladder_result
        assert ladder_native_match, (
            "Typed outside-in and accumulator polynomials differ"
        )
        assert _result.is_scalar
    return ladder_native_match, ladder_native_stages


@app.cell(hide_code=True)
def _(ladder_exact_match, ladder_native_match, ladder_native_stages, mo):
    mo.vstack(
        [
            mo.md("""
        ### Outside-in reduction with tensor interfaces

        This applies the same schedule using Spenso's metrics, tagged vectors,
        and typed tensor expressions throughout. Routing momenta are scalar
        metadata; each explicit index or bound compact vector occupies a vertex
        port. A reusable `TensorRule` checks and substitutes each vertex rule
        while retaining its factorization. Shared `contract()` selects the
        connected factors and keeps generated scalar sums factored.
        It binds exposed indices into the remaining vertices, so later replacements
        start with fewer free ports. Intermediate counts include those opaque
        vertices and count the factored expression before explicit expansion.

        `replace(rule)` replaces whole tensor factors and checks each substituted
        right-hand side locally, reusing those checks with repeated matches.
        It preserves the tensor interface without rescanning the complete sum
        after every vertex replacement.

        Single-run clocks include replacement, typed result validation and
        contraction. The last step also includes expansion of the
        complete final scalar polynomial. Setup, counting and exact checks are
        outside the clock.
        """),
            mo.ui.table(ladder_native_stages, selection=None)
            if ladder_native_stages
            else mo.md("Use the shared ladder run button to execute the comparisons."),
            mo.md(
                f"**Exact final polynomial:** accumulator `{ladder_native_match}`; "
                f"FORM `{ladder_native_match and ladder_exact_match}`."
            )
            if ladder_native_match is not None
            else mo.md(""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(
    form_executable,
    form_version,
    ladder_exact_match,
    ladder_form_process_samples,
    ladder_form_source,
    ladder_form_stages,
    ladder_form_stdout,
    ladder_order,
    ladder_run_counts,
    ladder_stages,
    ladder_substitution_samples,
    ladder_substitution_seconds,
    median,
    mo,
):
    _rows = []
    _reference_cpu = (0.00, 0.00, 0.00, 0.00, 0.02, 0.17, 0.36, 0.49)
    for _i, _terms in enumerate(ladder_run_counts):
        _row = {
            "vertex": ladder_order[_i],
            "reference terms": _terms,
        }
        if ladder_order == tuple(range(1, 9)):
            _row["supplied FORM cumulative CPU (s)"] = _reference_cpu[_i]
        if ladder_form_stages:
            _row.update(ladder_form_stages[_i])
        if ladder_stages:
            _row.update(ladder_stages[_i])
        _rows.append(_row)
    _notes = [
        (
            f"**Vertex order:** {' → '.join(map(str, ladder_order))}. "
            "Python carries typed interfaces between stages; FORM uses the same vertex order."
        ),
        (
            f"**Rule substitution only:** first call {1000 * ladder_substitution_samples[0]:.3f} ms; "
            f"five-run warm median {1000 * ladder_substitution_seconds:.3f} ms. "
            "This result remains factored and is not the 9,652-term scalar polynomial."
        ),
        (
            "Stage times exclude input construction, term counting, validation and display. "
            "Spenso reports a single full run's wall time; FORM statistics report cumulative "
            "CPU time rounded to 0.01 s. The supplied original-order 0.49 s is a reference measurement, "
            "not a measurement of this notebook's machine. Use an optimized community build "
            "for performance comparisons."
        ),
    ]
    if form_executable:
        _notes.append(
            f"**Local FORM:** `{form_version}`. Warm process median "
            f"{median(ladder_form_process_samples):.3f} s over three measured runs, "
            "including startup, parsing and sorting; polynomial export is untimed."
        )
    else:
        _notes.append(
            "Native FORM is unavailable here. The table shows the supplied reference; "
            "set `FORM_EXECUTABLE` in a native notebook session to measure it locally."
        )
    if ladder_stages:
        _seconds = ladder_stages[-1]["Spenso cumulative wall (s)"]
        _notes.append(
            f"**Full Spenso reduction:** {_seconds:.3f} s, with all eight term counts checked."
        )
        if ladder_form_process_samples:
            _notes.append(
                f"Spenso in-process wall / FORM process wall = "
                f"{_seconds / median(ladder_form_process_samples):.2f}×."
            )
    else:
        _notes.append(
            "Run the full reduction above to fill in the Spenso stage timings."
        )
    if ladder_exact_match:
        _notes.append(
            "**Exact check passed:** all coefficients of the 9,652-term scalar polynomial agree with FORM."
        )
    mo.vstack(
        [
            *(mo.md(note) for note in _notes),
            mo.ui.table(_rows, selection=None, pagination=False, show_download=True),
            mo.accordion(
                {
                    "Reference FORM source": mo.md(
                        f"```form\n{ladder_form_source}\n```"
                    ),
                    "Local FORM output": mo.md(
                        f"```text\n{ladder_form_stdout or 'Not run'}\n```"
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Fermionic outer ring: three-loop propagator ladder

    [Recorded inputs, measurements, and analysis](data/tensor_benchmark_history.md#fermionic-outer-ring-three-loop-propagator-ladder). These are historical results; use the controls below for the installed build.
    """)
    return


@app.cell
def _(mo):
    fermion_ladder_case = mo.ui.dropdown(
        options=["Three-loop fermion", "Four-loop fermion", "Four-loop gluon"],
        value="Three-loop fermion",
        label="Propagator numerator",
    )
    run_fermion_ladder = mo.ui.run_button(label="Compare selected ladder: 4D and D")
    mo.hstack([fermion_ladder_case, run_fermion_ladder])
    return fermion_ladder_case, run_fermion_ladder


@app.cell(hide_code=True)
def _(fermion_ladder_report, mo):
    mo.stop(fermion_ladder_report is None)
    _rows = [
        {
            "dimension": _case["dimension"],
            "terms": _case["terms"],
            "Idenso CPU (ms)": _case["idenso_median_cpu_ms"],
            "Idenso wall (ms)": _case["idenso_median_wall_ms"],
            "FORM CPU (ms)": _case["form_median_cpu_ms"],
            "Idenso / FORM": _case["cpu_ratio"],
            "exact polynomial match": _case["exact_form_match"],
        }
        for _case in fermion_ladder_report["cases"]
    ]
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None, pagination=False),
            mo.md(
                "**Passed:** exact FORM polynomials, D → 4, and independent exact HEP component checks."
            ),
            mo.accordion(
                {"Raw samples and validation": mo.json(fermion_ladder_report)}
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html(
        '<p data-notebook-ready="gamma_simplification">All identity, boundary and HEP component checks passed.</p>'
    )
    return


if __name__ == "__main__":
    app.run()
