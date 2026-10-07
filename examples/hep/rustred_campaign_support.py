"""Notebook orchestration only: native HEPKit owns graphs, algebra and generation.

No simulator, CLI parser, checkpoint decoder, or precomputed rules live here.
The optional injected clock/output path make lifecycle tests inexpensive.
"""

from collections import deque
from html import escape
import json
from pathlib import Path
from tempfile import mkdtemp
from time import monotonic


FAMILY_NAMES = ("H", "X", "BMW", "FG")
DATA_DIRECTORY = Path(__file__).resolve().parent / "data" / "rustred_four_loop"


def dot_sources():
    """Read packaged graph inputs, independent of the current working directory."""
    return {
        name: (DATA_DIRECTORY / f"{name.lower()}.dot").read_text()
        for name in FAMILY_NAMES
    }


def preferred_auxiliaries(name, routed_family):
    """Choose the reference ISP coordinates using native routed loop symbols.

    These are auxiliary inverse propagators, not additional graph edges.
    HEPKit still checks independence/completion; their integral powers stay
    nonpositive. Physical denominators come exclusively from the DOT graph.
    """
    k1, k2, k3, k4 = routed_family.loop_momenta
    momenta = {
        "H": (k1 - k2,),
        "X": (k3 + k4,),
        "BMW": (k1 - k3, k2 - k4),
        "FG": (k2 - k4, k3 - k4),
    }[name]
    return [routed_family.kinematics.scalar_product(q, q) - 1 for q in momenta]


class FourLoopCampaign:
    """Run one native family session at a time, advancing on nonblocking polls.

    The notebook owns this object for its lifetime. Cancel is cooperative: the
    queue stops immediately, but an active family is not declared stopped until
    native ``done`` is true. Keeping polling enabled advances the next family.
    Each invocation writes only to a fresh directory; no automatic resume.
    """

    def __init__(self, families, nonpositive_indices, *, output_directory=None,
                 clock=monotonic, capabilities=None):
        if tuple(families) != FAMILY_NAMES:
            raise ValueError("The demonstration must retain H, X, BMW and FG in order")
        self.families = families
        self.nonpositive_indices = nonpositive_indices
        self.output_directory = output_directory
        self.clock = clock
        self.capabilities = capabilities or {
            "background_sessions": True, "live_event_polling": True,
            "cancellation_in_flight": True,
        }
        self.state = "ready"
        self.started = None
        self.finished = None
        self.session = None
        self.active_name = None
        self.results = {}
        self.artifacts = {}
        self.rows = {name: {"family": name, "state": "queued"} for name in families}
        self.events = deque(maxlen=80)
        self.last_error = None
        self.evidence_error = None
        self.cancel_requested = False
        self.options = {}
        self.normalizations = {}
        self.normalization_rows = {}
        self.normalization_error = None

    def start(self, **options):
        if self.state != "ready":
            return False
        self.options = dict(options)
        self.output_directory = Path(self.output_directory or mkdtemp(
            prefix="rustred-four-loop-"))
        self.output_directory.mkdir(parents=True, exist_ok=True)
        # Do not overwrite or silently resume an existing run.
        if any(self.output_directory.iterdir()):
            raise ValueError("Choose an empty output directory for a fresh campaign")
        self.started = self.clock()
        self.state = "running"
        self._write_evidence("input-controls.json", {
            "schema": "hepkit.four-loop-notebook-controls.v1",
            "families": list(self.families), "options": self.options,
            "nonpositive_indices": self.nonpositive_indices,
            "scope": "Candidate generation only; no traversal or closure proof",
            "execution_capabilities": self.capabilities,
        })
        self._start_next()
        if not self.capabilities["background_sessions"]:
            # Single-thread browser calls return completed sessions. Run the
            # explicit queue to completion; polling records completed evidence.
            while self.session is not None:
                self.poll()
        self._record_snapshot()
        return True

    def _start_next(self):
        if self.cancel_requested:
            return
        pending = [name for name, row in self.rows.items() if row["state"] == "queued"]
        if not pending:
            self.state = "completed"
            self.finished = self.clock()
            return
        name = pending[0]
        directory = self.output_directory / name.lower()
        self.active_name = name
        self.rows[name]["state"] = "starting"
        try:
            directory.mkdir(exist_ok=False)
            self.session = self.families[name].start_generation(
                **self.options,
                nonpositive_indices=list(self.nonpositive_indices[name]),
                checkpoint_dir=str(directory / "checkpoint"),
                resume=False,
            )
        except Exception as error:
            self._fail(name, error)

    def _fail(self, name, error):
        self.last_error = str(error)
        self.rows[name].update(state="failed", last_error=self.last_error)
        self.state = "failed"
        self.finished = self.clock()
        self.session = None

    def cancel(self):
        if (not self.capabilities["cancellation_in_flight"]
                or self.state not in {"running", "cancelling"}):
            return False
        self.cancel_requested = True
        self.state = "cancelling"
        for row in self.rows.values():
            if row["state"] == "queued":
                row["state"] = "not started"
        if self.session is not None:
            self.session.cancel()
        self._record_snapshot()
        return True

    def poll(self):
        """Drain a bounded event batch; no wait or coefficient rendering."""
        if self.session is None:
            return self._record_snapshot()
        session, name = self.session, self.active_name
        batch = session.poll_events(max_events=128, timeout=0.0)
        self._write_evidence("progress.jsonl", {"family": name, "batch": batch}, append=True)
        native = batch["snapshot"]
        self.rows[name].update(
            state=native["state"], elapsed_seconds=native["elapsed_seconds"],
            **native["counts"], dropped_events=batch["dropped_events"],
        )
        self.rows[name]["active_jobs"] = native["active_jobs"]
        self.rows[name]["details_truncated"] = native.get("details_truncated", False)
        self.events.extend(dict(event, family=name) for event in batch["events"])
        if not native["done"]:
            return self._record_snapshot()
        if native["state"] == "completed":
            try:
                collected_at = self.clock()
                result = session.result()
                collected_seconds = self.clock() - collected_at
                opened_at = self.clock()
                artifact = result.artifact()
                opened_seconds = self.clock() - opened_at
                written_at = self.clock()
                path = self.output_directory / name.lower() / "candidate.rrbin"
                with path.open("xb") as output:
                    output.write(result.bundle)
                (path.parent / "generation-report.toml").write_text(result.to_toml())
                written_seconds = self.clock() - written_at
                self.results[name] = result
                self.artifacts[name] = artifact
                self.rows[name]["artifact_path"] = str(path)
                self.rows[name]["artifact_status"] = result.status
                self.rows[name].update(result_collection_seconds=collected_seconds,
                                       artifact_open_seconds=opened_seconds,
                                       artifact_write_seconds=written_seconds)
            except Exception as error:
                self._fail(name, error)
                return self._record_snapshot()
        elif native["state"] == "failed":
            self._fail(name, native.get("last_error") or "Native generation failed")
            return self._record_snapshot()
        elif native["state"] != "cancelled":
            self._fail(name, f"Unexpected completed native state: {native['state']}")
            return self._record_snapshot()
        self.session = None
        if self.cancel_requested or native["state"] == "cancelled":
            self.state = "cancelled"
            self.finished = self.clock()
        else:
            self._start_next()
        return self._record_snapshot()

    def snapshot(self):
        end = self.finished if self.finished is not None else self.clock()
        return {
            "state": self.state,
            "elapsed_seconds": 0 if self.started is None else end - self.started,
            "completed_families": len(self.results),
            "families": [dict(row) for row in self.rows.values()],
            "events": list(self.events),
            "last_error": self.last_error,
            "evidence_error": self.evidence_error,
            "output_directory": str(self.output_directory or "not created"),
        }

    def _write_evidence(self, name, value, *, append=False):
        """Small bounded poll batches only; no fsync, algebra, or artifact decode."""
        if self.started is None or self.evidence_error:
            return
        try:
            path = self.output_directory / name
            text = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
            if append:
                with path.open("a") as stream:
                    stream.write(text)
            else:
                temporary = path.with_name(path.name + ".tmp")
                temporary.write_text(text)
                temporary.replace(path)
        except (OSError, TypeError, ValueError) as error:
            # Reporting failure is not native completion or cancellation. Keep
            # the live handle so Cancel/poll can still drain the native job.
            self.evidence_error = str(error)

    def _record_snapshot(self):
        value = self.snapshot()
        self._write_evidence("snapshot.json", value)
        value["evidence_error"] = self.evidence_error
        return value

    def observe_view(self, family, operation, callback):
        started = self.clock()
        value = callback()
        self._write_evidence("views.jsonl", {
            "family": family, "operation": operation,
            "observed_seconds": self.clock() - started,
            "scope": "Caller-observed wait + native view operation; not isolated CAS time",
            "metadata": value if operation == "metadata" else None,
        }, append=True)
        return value

    def close(self):
        """Request cancellation without blocking the notebook shutdown path."""
        self.cancel()

    def normalize_completed(self):
        """Explicit post-generation native algebra; never called by polling."""
        if self.state in {"running", "cancelling"}:
            self.normalization_error = "Wait for native generation to drain before normalizing."
            return False
        if not self.artifacts:
            self.normalization_error = "No completed artifact is available yet."
            return False
        self.normalization_error = None
        for name, artifact in self.artifacts.items():
            if name in self.normalizations:
                continue
            started = self.clock()
            try:
                normalized = self.families[name].normalize_candidate_terminals(artifact)
                metadata = normalized.metadata()
                self.normalizations[name] = normalized
                self.normalization_rows[name] = {
                    "family": name, "state": "completed", "metadata": metadata,
                    "observed_seconds": self.clock() - started,
                }
            except Exception as error:
                self.normalization_rows[name] = {
                    "family": name, "state": "refused", "error": str(error),
                    "observed_seconds": self.clock() - started,
                }
            self._write_evidence("terminal-normalization.json", self.normalization_rows)
        return True


def normalization_summary_rows(campaign):
    rows = []
    for name in FAMILY_NAMES:
        row = campaign.normalization_rows.get(name, {})
        data = row.get("metadata", {})
        rows.append({"Family": name, "State": row.get("state", "not requested"),
            "Raw records": data.get("raw_terminal_records", "—"),
            "Distinct raw keys": data.get("unique_raw_terminals", "—"),
            "After unit aliases": data.get("after_unit_aliases", "—"),
            "Weighted outputs": data.get("canonical_terminals", "—"),
            "Observed seconds": round(row["observed_seconds"], 3) if row else "—"})
    return rows


def summary_rows(snapshot):
    """Compact scalar-only rows; unknown totals are shown as unknown, not zero."""
    return [
        {
            "Family": row["family"], "State": row["state"],
            "Sectors": f"{row.get('generated', '—')} / {row.get('sectors_total', '—')}",
            "Rules": row.get("rules", "—"),
            "Residuals": row.get("finite_residuals", "—"),
            "Failed sectors": row.get("failures", "—"),
            "Time (s)": round(row["elapsed_seconds"], 2)
            if "elapsed_seconds" in row else "—",
        }
        for row in snapshot["families"]
    ]


def integral_notation(key):
    """Format native integer structure only; never parse coefficient text."""
    if isinstance(key, dict):
        values, symbolic = key["values"], key["symbolic"]
        if len(values) != len(symbolic):
            raise ValueError("Integral flags and values have different arities")
    else:
        values, symbolic = key, [False] * len(key)
    powers = []
    for axis, (value, variable) in enumerate(zip(values, symbolic)):
        if type(value) is not int or type(variable) is not bool:
            raise ValueError("Expected exact native integer powers and boolean flags")
        if not variable:
            powers.append(str(value))
        else:
            suffix = f" + {value}" if value > 0 else f" - {-value}" if value < 0 else ""
            powers.append(f"n_{axis}{suffix}")
    return "I(" + ", ".join(powers) + ")"


def fixed_condition(case):
    return ", ".join(f"n_{item['axis']} = {item['value']}" for item in case["fixed"])


def rule_summary_rows(items):
    return [
        {"ordinal": row["ordinal"], "Target": integral_notation(row["target"]),
         "Case": fixed_condition(row["case"]) or row["case"]["kind"],
         "Affine equations": row["case"]["affine_equation_count"],
         "RHS terms": row["rhs_terms"], "Sources": row["retained_source_count"],
         "Excluded equations": row["guard_count"]}
        for row in items
    ]


def terminal_rows(page):
    return [{"ordinal": page["start"] + index, "Integral": integral_notation(key)}
            for index, key in enumerate(page["items"])]


def _equation_preview(ids, limit=8):
    text = " AND ".join(f"c_{cid} = 0" for cid in ids[:limit])
    if len(ids) > limit:
        text += f" AND … ({len(ids) - limit} more required equations; preview only)"
    return text


def rule_condition_rows(rule, sector, *, start=0, limit=10):
    """Keep native AND-within/OR-between exclusion semantics explicit."""
    rows = [
        {"Condition": "Sector", "Meaning": ", ".join(
            f"n_{axis} {'> 0' if active else '≤ 0'}"
            for axis, active in enumerate(sector))},
        {"Condition": "Fixed powers", "Meaning": fixed_condition(rule["case"]) or "None"},
        {"Condition": "Affine case equations (all required)", "Meaning":
         _equation_preview(rule["case"]["affine_zero_equations"]) or "None"},
    ]
    for index, branch in enumerate(rule["excluded_all_zero_conjunctions"][start:start + limit], start):
        rows.append({"Condition": f"Excluded branch {index + 1}",
                     "Meaning": _equation_preview(branch)
                     or "Always true (empty conjunction)"})
    return rows


def rule_coefficient_ids(rule, *, rhs_start=0, condition_start=0, limit=10):
    """IDs on the current bounded previews only, never an artifact-wide dropdown."""
    ids = {term["coefficient_id"] for term in rule["rhs"][rhs_start:rhs_start + limit]}
    ids.update(rule["case"]["affine_zero_equations"][:8])
    ids.update(cid for branch in rule["excluded_all_zero_conjunctions"][condition_start:condition_start + limit]
               for cid in branch[:8])
    return sorted(ids)


def rule_page_payload(rule, *, rhs_start=0, condition_start=0, limit=10):
    """Small structural preview; a collapsed accordion is not a lazy boundary."""
    return {
        "ordinal": rule["ordinal"], "target": rule["target"],
        "rhs_start": rhs_start, "rhs_total": len(rule["rhs"]),
        "rhs": rule["rhs"][rhs_start:rhs_start + limit],
        "case": {**rule["case"], "affine_zero_equations": rule["case"]["affine_zero_equations"][:8]},
        "affine_equations_total": len(rule["case"]["affine_zero_equations"]),
        "excluded_branch_start": condition_start,
        "excluded_branches_total": len(rule["excluded_all_zero_conjunctions"]),
        "excluded_branches_preview": [
            {"equations": branch[:8], "equations_total": len(branch)}
            for branch in rule["excluded_all_zero_conjunctions"][condition_start:condition_start + limit]
        ],
        "scope": "Bounded display preview, not a complete applicability certificate",
    }


def parameter_rows(bindings):
    """A legend of existing native Atom pairs, not a substitution engine."""
    return [{"Artifact variable": str(internal), "Original HEPKit expression": str(original)}
            for internal, original in bindings]


def _table(mo, rows, *, pagination=False):
    return mo.ui.table(rows, selection=None, pagination=pagination,
                       page_size=10, show_column_summaries=False, show_download=False)


def _code(mo, text):
    return mo.Html(
        '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;max-height:20rem;'
        'overflow:auto;padding:0.8rem;border:1px solid var(--gray-6);border-radius:0.4rem">'
        + escape(text) + '</pre>'
    )


def rule_view(mo, rule, sector, *, rhs_start=0, condition_start=0, limit=10):
    """Readable structure from one bounded native detail, with no coefficient call."""
    rhs = [{"Term": term["ordinal"], "Coefficient": f"c_{term['coefficient_id']}",
            "Integral": integral_notation(term["integral"])}
           for term in rule["rhs"][rhs_start:rhs_start + limit]]
    return mo.vstack([
        mo.md(f"**Rule {rule['ordinal']} · {len(rule['rhs'])} RHS terms · "
              f"{rule['retained_source_count']} retained sources**"),
        _code(mo, integral_notation(rule["target"]) + " = sum of the terms below"),
        _table(mo, rhs) if rhs else mo.md("**RHS: 0**" if not rule["rhs"] else "Empty RHS page."),
        mo.md(f"RHS preview: {rhs_start}–{rhs_start + len(rhs)} of {len(rule['rhs'])}; "
              "change the offset for another page. No coefficients are decoded here."),
        mo.md("Each row denotes `c_ID × I(…)`. Index labels are zero-based native "
              "coordinates; a literal integer is a fixed replacement, not a shift."),
        mo.accordion({
            "Case and excluded branches": mo.vstack([
                _table(mo, rule_condition_rows(rule, sector, start=condition_start, limit=limit)),
                mo.md(f"Excluded branches: page starting {condition_start} of "
                      f"{len(rule['excluded_all_zero_conjunctions'])}; at most 8 equation IDs per branch."),
                mo.md("All affine case equations must vanish. The rule is excluded if "
                      "**any** excluded branch holds; equations within a branch are "
                      "joined by **AND**. These are saved conditions, not a dispatch "
                      "guarantee: earlier rules, source conditions and denominator "
                      "poles still govern native applicability."),
            ]),
            "Raw native structure · current bounded preview only": mo.json(rule_page_payload(
                rule, rhs_start=rhs_start, condition_start=condition_start, limit=limit)),
        }),
    ])


def coefficient_view(mo, coefficient):
    """Native printer text, displayed directly without a text-to-algebra round trip."""
    return mo.vstack([
        mo.md(f"**c_{coefficient['id']} = numerator / denominator** · "
              f"{coefficient['numerator_terms']} / {coefficient['denominator_terms']} "
              "native polynomial terms"),
        mo.md("**Numerator**"), _code(mo, coefficient["numerator"]),
        mo.md("**Denominator**"), _code(mo, coefficient["denominator"]),
        mo.md("Native Symbolica printer output is display-only. Use the family's "
              "parameter legend for original HEPKit names; no string substitution "
              "or algebraic re-parsing is performed here."),
        mo.accordion({"Coefficient metadata · no duplicate expression payload": mo.json({
            key: value for key, value in coefficient.items() if key not in {"numerator", "denominator"}
        })}),
    ])


def normalization_relation_view(mo, relation, *, start=0, limit=10):
    """A bounded structural relation; native coefficients remain separate."""
    terms = relation["rhs"][start:start + limit]
    return mo.vstack([
        _code(mo, integral_notation(relation["integral"]) + " = sum of the terms below"),
        _table(mo, [{"Term": start + i, "Coefficient": f"c_{term['coefficient_id']}",
                     "Integral": integral_notation(term["integral"])}
                    for i, term in enumerate(terms)]) if terms else mo.md("**RHS: 0**"),
        mo.md(f"Terms {start}–{start + len(terms)} of {len(relation['rhs'])}. "
              "These coefficient IDs belong to this normalization result, not the candidate artifact."),
        mo.accordion({"Raw relation · current page only": mo.json({
            "ordinal": relation["ordinal"], "integral": relation["integral"],
            "rhs_start": start, "rhs_total": len(relation["rhs"]), "rhs": terms,
        })}),
    ])


def evaluate_h_numerator(integral, temporary_directory):
    """Explicit native Vakint operation; no generated candidate is substituted.

    Decimal references are the original Vakint H rank-four regression values.
    They are comparison inputs, never used to construct the computed result.
    """
    from symbolica import Float, N, S
    from symbolica.community.hepkit import vakint

    engine = vakint.Vakint(
        run_time_decimal_precision=32, number_of_terms_in_epsilon_expansion=5,
        integral_normalization_factor="MSbar", mu_r_sq_symbol=S("vakint::mursq"),
        tensor_reduction_method="feynkit",
        evaluation_order=[vakint.VakintEvaluationMethod.new_rustred_method()],
        form_exe_path="/this/path/must/not/be/invoked/by-rustred-notebook",
        temporary_directory=str(temporary_directory),
    )
    started = monotonic()
    evaluated = engine.evaluate(integral.to_expression())
    symbolic_seconds = monotonic() - started
    # Preserve the original reference's f64 external-vector input boundary.
    vectors = {i: (0.17 * (i + 1), 0.4 * (i + 2), 0.3 * (i + 3), 0.12 * (i + 4))
               for i in (1, 2)}
    numerical_started = monotonic()
    result, error = engine.numerical_evaluation(
        evaluated, params={"vakint::muvsq": 3.0, "vakint::mursq": 5.0}, externals=vectors)
    numerical_seconds = monotonic() - numerical_started
    values = (
        (-4, "1.809145974886785501452557650622e-9"),
        (-3, "1.862208677723707446525921998109e-8"),
        (-2, "8.865577059648962609058045604434e-8"),
        (-1, "4.064224364096375531168559391726e-7"),
        (0, "7.705260630861442737917312763495e-6"),
    )
    epsilon = S("vakint::ε")
    expected = sum((N(Float(value, decimal_digits=32)) * epsilon ** power
                    for power, value in values), N(0))
    reference = engine.numerical_result_from_expression(expected)
    matches, detail = result.compare_to(
        reference, relative_threshold=1e-30, error=error, max_pull=1.0)
    return {"evaluated": evaluated, "result": result, "error": error,
            "reference": reference, "metrics": {
                "reference_matches": matches, "comparison": detail,
                "symbolic_seconds": symbolic_seconds, "numerical_seconds": numerical_seconds,
                "decimal_digits": 32, "relative_threshold": 1e-30,
                "mass_squared": 3, "mu_squared": 5, "normalization": "MSbar",
                "form_requested": False, "generated_candidate_files_used": False,
                "backend": "Vakint shipped RustRed assets + FeynKit tensor reduction",
                "external_vectors": vectors}}


class ExplicitNumeratorEvaluation:
    """One explicit, cached post-generation operation; polling cannot invoke it."""

    def __init__(self, operation=evaluate_h_numerator):
        self.operation = operation
        self.state = "ready"
        self.result = None
        self.error = None

    def run(self, campaign, integral):
        if self.state != "ready":
            return False
        if campaign.state != "completed":
            self.error = "Finish and drain all four generation sessions before evaluating."
            return False
        self.state = "running"
        self.error = None
        started = monotonic()
        try:
            directory = campaign.output_directory / "vakint-h-numerator"
            directory.mkdir(exist_ok=True)
            self.result = self.operation(integral, directory)
            self.state = "completed"
        except Exception as exc:
            self.state = "failed"
            self.error = str(exc)
        evidence = {"schema": "hepkit.four-loop-numerator-evaluation.v1",
                    "state": self.state, "elapsed_seconds": monotonic() - started,
                    "error": self.error, "metrics": self.result["metrics"] if self.result else None}
        try:
            campaign._write_evidence("vakint-h-numerator.json", evidence)
            if campaign.evidence_error:
                self.error = f"Evaluation state retained; evidence issue: {campaign.evidence_error}"
        except Exception as exc:
            self.error = f"Evaluation state retained, but evidence write failed: {exc}"
        return self.state == "completed"
