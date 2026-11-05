"""Deterministic selection from explicitly normalized full-volume Dice rows.

Input JSON is a mapping of method names to lists of objects, for example::

    {"PromptGen": [{"source": "AMOS", "case_id": "case-001",
                    "class_id": "liver", "dice": 0.93,
                    "metric_scope": "full_volume"}],
     "SAM2": [{"source": "AMOS", "case_id": "case-001",
               "class_id": "liver", "dice": 0.6,
               "metric_scope": "full_volume"}]}

All methods in the mapping participate. Production callers must include every
intended comparator; this module cannot infer omitted methods. Numeric Dice must
be finite in [0,1], never percentages or strings. IDs are exact, nonempty strings:
normalizers must explicitly resolve global class semantics across evaluators
before constructing class_id, never match raw local label indices implicitly.
Only source names use the explicit alias table below. Extra row fields are ignored;
freeze their CT/prompt/checkpoint/protocol references separately before inference.
Metric scope is an explicit upstream assertion, not independently verified here.
Different native method prompt budgets remain upstream protocol responsibilities.

Missing keys across methods are excluded and enumerated, not imputed. Duplicate
keys, unsupported sources, invalid IDs/metrics and non-full-volume scopes fail.
By default, select only Dice>=0.85 and gap>=0.05 against the strongest comparator.
Rank by (Dice+gap) descending, Dice descending, gap descending, exact case/class
ascending. Eligible rows precede ineligible rows. Thresholds are configurable and
are recorded, not fitted. No fallback selection unless explicitly requested.
These are post-hoc illustrative examples, never representative average results.

Pure standard library; does not load CT, masks, models, or raw evaluator schemas.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


DEFAULT_SOURCES = ("AMOS", "MAGIC", "KiTS23", "MSD10", "MSD07")
SOURCE_ALIASES = {
    "amos": "AMOS", "magic": "MAGIC", "kits23": "KiTS23",
    "msd10": "MSD10", "msd_task10": "MSD10", "task10_colon": "MSD10",
    "msd07": "MSD07", "msd_task07": "MSD07", "task07_pancreas": "MSD07",
}


def normalize_source(source):
    """Resolve only enumerated, case-insensitive source aliases."""
    if not isinstance(source, str) or source.casefold() not in SOURCE_ALIASES:
        raise ValueError("unsupported source: {!r}".format(source))
    return SOURCE_ALIASES[source.casefold()]


def _identifier(value, label):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("{} must be a nonempty string without outer whitespace".format(label))
    return value


def _fraction(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("{} must be finite numeric in [0,1]".format(label))
    return float(value)


def select_cases(method_rows, sources=DEFAULT_SOURCES, promptgen_method="PromptGen",
                 min_dice=0.85, min_gap=0.05, top_k=10, allow_best_available=False):
    """Return auditable selections and exclusions, without changing input rows."""
    min_dice = _fraction(min_dice, "min_dice")
    min_gap = _fraction(min_gap, "min_gap")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(allow_best_available, bool):
        raise ValueError("allow_best_available must be boolean")
    _identifier(promptgen_method, "promptgen_method")
    if not isinstance(method_rows, dict) or promptgen_method not in method_rows:
        raise ValueError("method mapping must contain {}".format(promptgen_method))
    if len(method_rows) < 2:
        raise ValueError("at least one comparator is required")
    for method in method_rows:
        _identifier(method, "method")
    methods = sorted(method_rows)
    comparators = [method for method in methods if method != promptgen_method]
    if isinstance(sources, str) or not sources:
        raise ValueError("sources must be a nonempty sequence")
    requested = [normalize_source(source) for source in sources]
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate requested source")

    indexed = {}
    all_keys = set()
    for method in methods:
        rows = method_rows[method]
        if not isinstance(rows, list):
            raise ValueError("{} rows must be a list".format(method))
        indexed[method] = {}
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("{} row {} must be an object".format(method, index))
            required = ("source", "case_id", "class_id", "dice", "metric_scope")
            if any(field not in row for field in required):
                raise ValueError("{} row {} missing required fields".format(method, index))
            source = normalize_source(row["source"])
            if source not in requested:
                raise ValueError("source {} outside requested sources".format(source))
            case_id = _identifier(row["case_id"], "case_id")
            class_id = _identifier(row["class_id"], "class_id")
            if row["metric_scope"] != "full_volume":
                raise ValueError("metric_scope must be full_volume, got {!r}".format(row["metric_scope"]))
            dice = _fraction(row["dice"], "dice")
            key = (source, case_id, class_id)
            if key in indexed[method]:
                raise ValueError("duplicate key for {}: {!r}".format(method, key))
            indexed[method][key] = dice
            all_keys.add(key)

    result = {
        "schema_version": 1,
        "interpretation": "post-hoc illustrative examples; not representative averages",
        "methods": methods,
        "promptgen_method": promptgen_method,
        "rule": {
            "min_promptgen_dice": min_dice,
            "min_gap_to_strongest_comparator": min_gap,
            "inclusive_threshold_roundoff_tolerance": 1e-12,
            "score": "promptgen_dice + gap",
            "ordering": ["meets_criteria descending", "score descending",
                         "promptgen_dice descending", "gap descending",
                         "case_id ascending", "class_id ascending"],
            "join": "exact canonical source + case_id + class_id across every supplied method",
            "metric_scope": "full_volume",
            "allow_best_available": allow_best_available,
            "top_k": top_k,
        },
        "sources": {},
    }
    for source in requested:
        candidates = []
        excluded = []
        for key in sorted(all_keys):
            if key[0] != source:
                continue
            missing = [method for method in methods if key not in indexed[method]]
            identifiers = dict(source=source, case_id=key[1], class_id=key[2])
            if missing:
                excluded.append(dict(identifiers, missing_methods=missing))
                continue
            metrics = {method: indexed[method][key] for method in methods}
            promptgen_dice = metrics[promptgen_method]
            strongest = min(comparators, key=lambda method: (-metrics[method], method))
            strongest_dice = metrics[strongest]
            gap = promptgen_dice - strongest_dice
            candidates.append(dict(
                identifiers, metrics=metrics, promptgen_dice=promptgen_dice,
                strongest_comparator=strongest, strongest_comparator_dice=strongest_dice,
                gap=gap, score=promptgen_dice + gap, all_method_win=gap > 0,
                meets_criteria=(promptgen_dice + 1e-12 >= min_dice and gap + 1e-12 >= min_gap and gap > 0)))
        candidates.sort(key=lambda item: (not item["meets_criteria"], -item["score"],
                                         -item["promptgen_dice"], -item["gap"],
                                         item["case_id"], item["class_id"]))
        selected = None
        status = "no_common_tasks"
        if candidates:
            status = "criteria_unmet"
            if candidates[0]["meets_criteria"]:
                selected, status = candidates[0], "selected"
            elif allow_best_available:
                selected, status = candidates[0], "best_available_criteria_unmet"
        result["sources"][source] = dict(
            status=status, selected=selected, common_count=len(candidates),
            eligible_count=sum(item["meets_criteria"] for item in candidates),
            candidates=candidates[:top_k], excluded_missing_methods=excluded)
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {!r}".format(key))
        result[key] = value
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="normalized method->rows JSON")
    parser.add_argument("--output", required=True, type=Path, help="new selection JSON (never overwrites)")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--promptgen-method", default="PromptGen")
    parser.add_argument("--min-dice", type=float, default=.85)
    parser.add_argument("--min-gap", type=float, default=.05)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--allow-best-available", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        result = select_cases(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object), sources=args.sources,
                              promptgen_method=args.promptgen_method, min_dice=args.min_dice,
                              min_gap=args.min_gap, top_k=args.top_k,
                              allow_best_available=args.allow_best_available)
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        with args.output.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, "error: {}\n".format(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
