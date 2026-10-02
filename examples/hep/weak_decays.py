import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Charged-current W and top decays")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Charged-current W and top decays
    [Browse notebooks](/) · [W-pair production](/?file=hep/ww_production.py) · [Muon decay](/?file=hep/muon_decay.py) · [Z decays](/?file=hep/z_decay.py) · [Higgs decays](/?file=hep/higgs_decay.py)

    Generate eight channels from `Model.standard_model()`: charged W decays to leptons, unequal-mass quarks or massless light quarks, together with top and antitop decay. Keep the chiral vertex and complex CKM factor, sum physical final polarizations and average the initial spin. The full massive-vector spin sum includes the longitudinal polarization.

    The symbolic comparisons reproduce the [leptonic W](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/W-ElAnel), [quark W](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/W-QiQjbar) and [top](https://feyncalc.github.io/FeynCalcExamples/EW/Tree/Qt-QbW) references. Massive W quark channels use the model's charm/bottom fields to retain two nonzero, unequal masses. Their formula is independent of the chosen generations.

    Write $H$ for the parent mass, $x,y$ for the daughter masses and $V=V_R+iV_I$. All widths use the shared two-body phase-space and decay-flux methods, with $e^2=4\sqrt2 G_Fm_W^2\sin^2\theta_W$. The physical domain is $H>x+y$; the numerical checks also include the threshold limit.
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
    from symbolica.community import hepkit as hep
    from symbolica.community import tensor as sp
    import math

    import marimo as mo
    from symbolica import E, Replacement, S, Symbol
    from symbolica import set_namespace as _set_namespace
    from symbolica.community.hepkit import Kinematics, Model
    from symbolica.community.tensor import TensorExpression

    _set_namespace("weak_decay")
    return E, Kinematics, Model, Replacement, S, Symbol, hep, math, mo, sp


@app.cell(hide_code=True)
def _():
    from symbolica.community.hepkit import Symbols

    return (Symbols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Calculation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Define the channels and conventions
    """)
    return


@app.cell
def _(E, Model, S, hep):
    model = Model.standard_model()
    P = hep.Kinematics.external_momentum
    charge = -model.particle("e-").electric_charge
    sw = model.parameter("sw").symbol
    mw = model.particle("W+").mass
    Vr, Vi, GF = S("Vr", "Vi", "GF")
    H, x, y = (S(name, is_positive=True) for name in ("H", "x", "y"))
    zero = E("0")
    channels = {
        "W- leptons": (("W-", "e-", "ve~"), None),
        "W+ leptons": (("W+", "e+", "ve"), None),
        "W- quarks": (("W-", "c~", "b"), "CKM2x3"),
        "W+ quarks": (("W+", "c", "b~"), "CKM2x3"),
        "W- light quarks": (("W-", "u~", "d"), "CKM1x1"),
        "W+ light quarks": (("W+", "u", "d~"), "CKM1x1"),
        "Top": (("t", "b", "W+"), "CKM3x3"),
        "Antitop": (("t~", "b~", "W-"), "CKM3x3"),
    }
    return GF, H, P, Vi, Vr, channels, charge, model, mw, sw, x, y, zero


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Generate the decay amplitudes

    Each channel has one tree diagram. The native amplitude handles its weights, external ports and complex conjugation.
    """)
    return


@app.cell
def _(E, channels, model):
    amplitudes = {
        label: model.process([names[0]], list(names[1:])).generate_amplitude(
            max_vertices=1, progress=None
        )
        for label, (names, _) in channels.items()
    }
    diagrams = {}
    for _label, _amplitude in amplitudes.items():
        assert len(_amplitude.diagrams) == 1
        diagrams[_label] = _amplitude.diagrams[0]
        assert diagrams[_label].denominator_expression(
            dimension=4
        ).to_expression() == E("1")
    return amplitudes, diagrams


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Sum spins and colors

    Average the parent spin in both conventions. A physical top width also averages its initial color; the gallery convention sums it. Complex CKM parameters remain complex during conjugation.
    """)
    return


@app.cell
def _(amplitudes):
    settings = dict(gamma=True, color=True, epsilon=True)
    model_squares = {}
    for _label, _amplitude in amplitudes.items():
        for _convention in ("Physical", "Gallery color sum"):
            _summed = (
                _amplitude.squared()
                .sum_spins(average_initial=True)
                .sum_colors(average_initial=_convention == "Physical")
            )
            _scalar = _summed.expression().simplify_algebra(contract="dots", **settings)
            assert _scalar.is_scalar
            model_squares[_label, _convention] = _scalar.to_expression().together()
    return (model_squares,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Apply the two-body kinematics

    Use positive physical masses $H,x,y$, keeping exactly massless daughter states at zero. Write the complex mixing factor as $V_R+iV_I$ only after forming the physical square.
    """)
    return


@app.cell
def _(
    E,
    GF,
    H,
    Kinematics,
    P,
    Replacement,
    Symbol,
    Symbols,
    Vi,
    Vr,
    channels,
    charge,
    model,
    model_squares,
    mw,
    sp,
    sw,
    x,
    y,
    zero,
):
    physical_squares, decay_data = {}, {}
    for _label, (_names, _ckm_name) in channels.items():
        _parent, _first, _second = [model.particle(name).mass for name in _names]
        _kin = (
            Kinematics()
            .with_scalar_product(P(0), P(0), _parent**2)
            .with_scalar_product(P(1), P(1), _first**2)
            .with_scalar_product(P(2), P(2), _second**2)
            .with_scalar_product(P(1), P(2), (_parent**2 - _first**2 - _second**2) / 2)
            .with_scalar_product(P(0), P(1), (_parent**2 + _first**2 - _second**2) / 2)
            .with_scalar_product(P(0), P(2), (_parent**2 - _first**2 + _second**2) / 2)
        )
        _mass_rules = [Replacement(_parent, H)]
        if _first != zero:
            _mass_rules.append(Replacement(_first, x))
        if _second != zero:
            _mass_rules.append(Replacement(_second, y))
        decay_data[_label] = (
            x if _first != zero else zero,
            y if _second != zero else zero,
            Vr**2 + Vi**2 if _ckm_name else E("1"),
            _names[0] in ("t", "t~"),
        )
        for _convention in ("Physical", "Gallery color sum"):
            _square = _kin.apply(model_squares[_label, _convention]).together()
            if _ckm_name:
                _ckm = model.parameter(_ckm_name).symbol
                _conj = sp.BroadcastFunction.conj().to_expression()
                _square = _square.replace(
                    _conj(Symbols.model_conjugate(_ckm)), Vr + Symbol.I * Vi
                )
                _square = _square.replace(_conj(_ckm), Vr - Symbol.I * Vi)
                _square = _square.replace(
                    Symbols.model_conjugate(_ckm), Vr - Symbol.I * Vi
                ).replace(_ckm, Vr + Symbol.I * Vi)
            physical_squares[_label, _convention] = (
                _square.replace(charge**2, 4 * E("2").sqrt() * GF * mw**2 * sw**2)
                .replace_multiple(_mass_rules)
                .together()
            )
    return decay_data, physical_squares


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Integrate the physical phase space

    `two_body_phase_space()` and `flux()` provide the normalization. Check the measure against the independent Källén formula.
    """)
    return


@app.cell
def _(H, Kinematics, P, S, Symbol, decay_data, zero):
    phase_factors, kallen_polynomials = {}, {}
    for _label, (_first, _second, _, _) in decay_data.items():
        _kin = (
            Kinematics()
            .with_scalar_product(P(0), P(0), H**2)
            .with_scalar_product(P(1), P(1), _first**2)
            .with_scalar_product(P(2), P(2), _second**2)
            .with_scalar_product(P(1), P(2), (H**2 - _first**2 - _second**2) / 2)
        )
        _measure = _kin.two_body_phase_space(P(1), P(2))
        _radicand = S("radicand_")
        for _match in list(_measure.match(_radicand.sqrt())):
            _value = dict(_match)[_radicand]
            _measure = _measure.replace(_value.sqrt(), _value.expand().sqrt())
        _flux = _kin.flux(P(0))
        assert _flux == 2 * H
        _kallen = (
            H**4
            + _first**4
            + _second**4
            - 2 * H**2 * _first**2
            - 2 * H**2 * _second**2
            - 2 * _first**2 * _second**2
        ).expand()
        assert (
            _measure - _kallen.sqrt() / (32 * Symbol.PI**2 * H**2)
        ).together() == zero
        phase_factors[_label] = 4 * Symbol.PI * _measure / _flux
        kallen_polynomials[_label] = _kallen
    return kallen_polynomials, phase_factors


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check the massive reference formulas

    Verify both the squared matrix element and the width, including the initial-color convention. These independent formulas retain unequal daughter masses.
    """)
    return


@app.cell
def _(
    E,
    GF,
    H,
    Symbol,
    channels,
    decay_data,
    kallen_polynomials,
    phase_factors,
    physical_squares,
    x,
    y,
    zero,
):
    results = {}
    for _label, (_first, _second, _mixing, _is_top) in decay_data.items():
        for _convention in ("Physical", "Gallery color sum"):
            _color = (
                (3 if _convention == "Gallery color sum" else 1)
                if _is_top
                else 3
                if channels[_label][1] is not None
                else 1
            )
            if _is_top:
                _polynomial = (H**2 - x**2) ** 2 + y**2 * (H**2 + x**2) - 2 * y**4
                _reference = E("2").sqrt() * _color * GF * _mixing * _polynomial
            else:
                _a, _b = _first**2, _second**2
                _polynomial = 2 * H**4 - H**2 * (_a + _b) - (_a - _b) ** 2
                _reference = 2 * E("2").sqrt() * _color * GF * _mixing * _polynomial / 3
            _squared = physical_squares[_label, _convention]
            assert (_squared - _reference).together() == zero, (
                _label,
                _convention,
                _squared,
            )
            _width = (phase_factors[_label] * _squared).together()
            _reference_width = (
                _color
                * GF
                * _mixing
                * kallen_polynomials[_label].sqrt()
                * _polynomial
                * E("2").sqrt()
                / (Symbol.PI * H**3 * (16 if _is_top else 24))
            )
            assert (_width - _reference_width).together() == zero
            results[_label, _convention] = {
                "squared": _squared,
                "width": _width,
                "mixing": _mixing,
                "first": _first,
                "second": _second,
            }
        assert (
            results[_label, "Gallery color sum"]["width"]
            - (3 if _is_top else 1) * results[_label, "Physical"]["width"]
        ).together() == zero
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Check charge conjugation
    """)
    return


@app.cell
def _(results, zero):
    for _negative, _positive in [
        ("W- leptons", "W+ leptons"),
        ("W- quarks", "W+ quarks"),
        ("W- light quarks", "W+ light quarks"),
        ("Top", "Antitop"),
    ]:
        for _convention in ("Physical", "Gallery color sum"):
            assert (
                results[_negative, _convention]["width"]
                - results[_positive, _convention]["width"]
            ).together() == zero
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initial color convention
    The physical top width averages its initial color, so the final bottom-color sum gives an overall factor one. The gallery sums initial colors and therefore has an extra factor three. Both choices use `Particle.color_sum`; the calculation proves their ratio exactly. W decays have no initial color, so both choices coincide.

    The squared amplitudes and widths of each charge-conjugate pair agree for generic $V_R,V_I$, with dependence only on $|V|^2=V_R^2+V_I^2$.
    """)
    return


@app.cell
def _(E, GF, H, Replacement, Vi, Vr, math, results, x, y, zero):
    numeric_checks = []
    for (_label, _convention), _result in results.items():
        _is_top = _label in ("Top", "Antitop")
        _ratios = (
            [(0.0, 0.46), (0.03, 0.46), (0.3, 0.69), (0.3, 0.7)]
            if _is_top
            else [(0.0, 0.0), (0.01, 0.02), (0.2, 0.3), (0.45, 0.54)]
        )
        for _r1, _r2 in _ratios:
            _r1 = _r1 if _result["first"] != zero else 0.0
            _r2 = _r2 if _result["second"] != zero else 0.0
            _substitutions = [
                Replacement(H, E("100")),
                Replacement(x, E(str(100 * _r1))),
                Replacement(y, E(str(100 * _r2))),
                Replacement(GF, E("0.00001")),
                Replacement(Vr, E("0.8")),
                Replacement(Vi, E("0.3")),
            ]
            _actual = complex(
                _result["width"].replace_multiple(_substitutions).evaluate({})
            )
            _phase = math.sqrt(
                max(0.0, (1 - (_r1 + _r2) ** 2) * (1 - (_r1 - _r2) ** 2))
            )
            _mixing = 1.0 if "leptons" in _label else 0.8**2 + 0.3**2
            _colors = (
                (3 if _convention == "Gallery color sum" else 1)
                if _is_top
                else 1
                if "leptons" in _label
                else 3
            )
            _shape = (
                (1 - _r1**2) ** 2 + _r2**2 * (1 + _r1**2) - 2 * _r2**4
                if _is_top
                else 1 - (_r1**2 + _r2**2) / 2 - (_r1**2 - _r2**2) ** 2 / 2
            )
            _expected = (
                _colors
                * 1e-05
                * 100**3
                * _mixing
                * _phase
                * _shape
                / ((8 if _is_top else 6) * math.sqrt(2) * math.pi)
            )
            assert abs(_actual - _expected) < 2e-09, (
                _label,
                _convention,
                _r1,
                _r2,
                _actual,
                _expected,
            )
            numeric_checks.append((_label, _convention, _r1, _r2, _actual, _expected))
    return (numeric_checks,)


@app.cell
def _(mo):
    channel = mo.ui.dropdown(
        [
            "W- leptons",
            "W+ leptons",
            "W- quarks",
            "W+ quarks",
            "W- light quarks",
            "W+ light quarks",
            "Top",
            "Antitop",
        ],
        value="Top",
        label="Decay channel",
    )
    color_convention = mo.ui.dropdown(
        ["Physical", "Gallery color sum"],
        value="Physical",
        label="Initial color convention",
    )
    kinematic_point = mo.ui.dropdown(
        {
            "Light daughter limit": 0,
            "Separated masses": 1,
            "Heavy daughters": 2,
            "Closest to threshold": 3,
        },
        value="Separated masses",
        label="Kinematics",
    )
    mo.vstack([channel, color_convention, kinematic_point])
    return channel, color_convention, kinematic_point


@app.cell
def _(
    channel,
    color_convention,
    diagrams,
    kinematic_point,
    mo,
    numeric_checks,
    results,
):
    selected_result = results[channel.value, color_convention.value]
    _checks = [
        _row
        for _row in numeric_checks
        if _row[:2] == (channel.value, color_convention.value)
    ]
    selected_point = _checks[kinematic_point.value]
    _, _, _r1, _r2, selected_width, reference_width = selected_point
    assert abs(selected_width - reference_width) < 2e-09
    assert selected_width.real >= -1e-12 and abs(selected_width.imag) < 2e-09
    mo.vstack(
        [
            diagrams[channel.value],
            mo.md("## Spin-averaged squared amplitude"),
            selected_result["squared"],
            mo.md("## Total width"),
            selected_result["width"],
            mo.md("## Numerical comparison"),
            mo.ui.table(
                [
                    {
                        "Parent H (GeV)": 100,
                        "First daughter (GeV)": 100 * _r1,
                        "Second daughter (GeV)": 100 * _r2,
                        "GF (GeV⁻²)": 1e-05,
                        "CKM V": ".8 + .3i" if "leptons" not in channel.value else "1",
                    }
                ],
                selection=None,
            ),
            mo.md(f"**Generated width:** `{selected_width.real:.12g} GeV`"),
            mo.md(f"**Reference width:** `{reference_width:.12g} GeV`"),
            mo.md(
                f"**Absolute difference:** `{abs(selected_width - reference_width):.3g} GeV`"
            ),
            mo.md(
                "These are illustrative kinematic inputs; the symbolic formula above retains arbitrary masses, GF and CKM element. All eight channels and both color conventions pass the exact and numerical comparisons."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
