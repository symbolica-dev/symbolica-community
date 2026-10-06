"""Formatting only; native values and uncertainty are never recomputed."""

from html import escape
import math

EXAMPLES = {
    "Massive triangle": "triangle",
    "Massless box": "box",
    "Rank-two box numerator": "rank_two_box",
    "Coupled two-loop sunset": "sunset",
}

# User-selected allocations; these never run or change an existing native owner.
QMC_PRESETS = {
    "quick": {"points": 1024, "shifts": 8, "seed": 20261005,
              "package_points": 1024, "rule": "kuo_33002", "periodization": "korobov3"},
    "gghh_accuracy": {"points": 32768, "shifts": 16, "seed": 20261007,
                      "package_points": 1024, "rule": "hkkn_alpha3", "periodization": "korobov3"},
}

def validate_configuration(value):
    if value is None:
        return None
    # marimo form validation receives raw frontend dropdown selections; the
    # submitted form.value is converted by the original UI elements afterwards.
    kind = value["example"]
    if isinstance(kind, list):
        kind = EXAMPLES.get(kind[0]) if kind else None
    if not math.isfinite(value["s"]) or value["s"] >= 0:
        return "Choose a finite negative s for this Euclidean example."
    if kind in {"box", "rank_two_box"} and (not math.isfinite(value["t"]) or value["t"] >= 0):
        return "Choose a finite negative t for the box."
    if kind == "triangle" and (not math.isfinite(value["mass"]) or value["mass"] <= 0):
        return "The triangle mass must be finite and positive."
    return None

def epsilon_label(order):
    return "ε" + str(order).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))

def table(mo, rows):
    if not rows:
        return mo.md("No native observations yet.")
    formats = {key: (lambda value: "—" if value is None else f"{value:.8g}")
               for key in rows[0] if any(isinstance(row.get(key), float) for row in rows)}
    if "relative error" in rows[0]:
        formats["relative error"] = lambda value: "—" if value is None else f"{100 * value:.4g}%"
    return mo.ui.table(rows, selection=None, show_column_summaries=False,
                       show_data_types=False, pagination=len(rows) > 12,
                       page_size=12, format_mapping=formats)
