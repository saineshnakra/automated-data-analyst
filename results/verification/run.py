"""Run ADA on the evaluation cases and save what it returned, with automatic numeric checks.

    python -I run.py ADA_CHECKOUT DATA_ROOT HOLDOUT_ROOT OUT.json [--split dev|holdout|all] [--only ID,ID]
                     [--cases CASES.yml]

Numbers are compared with the expected values in cases.yml. Criteria written as `must:` are
left for a written verdict, with ADA's full output saved beside them.
"""

import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
checkout, data_root, holdout_root, out_path = (Path(a) for a in sys.argv[1:5])
args = sys.argv[5:]
split = args[args.index("--split") + 1] if "--split" in args else "dev"
only = set(args[args.index("--only") + 1].split(",")) if "--only" in args else None
cases_path = Path(args[args.index("--cases") + 1]) if "--cases" in args else HERE / "cases.yml"
sys.path.insert(0, str(checkout))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import ada  # noqa: E402

spec = yaml.safe_load(cases_path.read_text())
datasets = spec["datasets"]


def folder_of(name):
    root = data_root if datasets[name]["split"] == "dev" else holdout_root
    return root / name


def close(got, want, rel=0.005, absolute=None):
    if got is None or want is None:
        return False
    try:
        got, want = float(got), float(want)
    except (TypeError, ValueError):
        return False
    if math.isnan(got):
        return False
    tolerance = absolute if absolute is not None else max(abs(want) * rel, 1e-9)
    return abs(got - want) <= tolerance


def numbers_in(text):
    return [float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*\.?\d*(?:e-?\d+)?", str(text))]


def label_key(label):
    """A group label as text, with a whole number written without its ".0": 4.0 is "4"."""
    try:
        number = float(label)
    except (TypeError, ValueError):
        return str(label)
    return str(int(number)) if number.is_integer() else str(label)


def share_forms(want):
    """The forms an expected share may take in an answer: 0.323, or 32.3 as a percentage."""
    return [want, want * 100] if isinstance(want, int | float) and 0 < want < 1 else [want]


def table_values(table):
    """Label -> every number on that label's row of an answer table."""
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return {}
    numeric = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    # The groups are the first text column, or the first column when they are numbers (cyl 4, 6, 8).
    labels = [c for c in table.columns if c not in numeric] or list(table.columns[:1])
    if not numeric:
        return {}
    keys = table[labels[0]]
    rows = zip(keys, table.iterrows(), strict=True)
    return {label_key(k): [row[c] for c in numeric] for k, (_, row) in rows}


def run_profile(case):
    path = folder_of(case["dataset"]) / case["file"]
    report = ada.analyze(path)
    frame = report.frame
    got = {
        "rows": int(report.rows_analyzed),
        "columns": int(frame.shape[1]),
        "missing": {c: int(frame[c].isna().sum()) for c in case["expected"].get("missing", {}) if c in frame},
        "cleaning_notes": list(report.cleaning.notes),
        "headline": report.headline,
        "data_checks": report.quality().to_dict(orient="records"),
    }
    expected = case["expected"]
    checks = {}
    if "rows" in expected:
        checks["rows"] = got["rows"] == expected["rows"]
    if "columns" in expected:
        checks["columns"] = got["columns"] == expected["columns"]
    for column, count in expected.get("missing", {}).items():
        checks[f"missing:{column}"] = got["missing"].get(column) == count
    return got, checks


def run_compute(case):
    path = folder_of(case["dataset"]) / case["file"]
    report = ada.analyze(path)
    answer = report.ask(case["question"])
    if isinstance(answer, ada.Refusal):
        return {"refused": answer.code, "message": answer.message}, {"answered": False}
    table = answer.table if isinstance(getattr(answer, "table", None), pd.DataFrame) else None
    got = {
        "answer": answer.answer,
        "calculation": answer.calculation,
        "table": None if table is None else table.head(20).to_dict(orient="records"),
    }
    expected = case["expected"]
    checks = {"answered": True}
    if "values" in expected:
        values = table_values(table)
        for label, want in expected["values"].items():
            checks[f"value:{label}"] = any(
                close(v, form) for v in values.get(label_key(label), []) for form in share_forms(want)
            )
    if "value" in expected:
        found = numbers_in(answer.answer) + [v for row in table_values(table).values() for v in row]
        checks["value"] = any(close(n, form) for n in found for form in share_forms(expected["value"]))
        if "wrong_value" in expected:
            # The answer's own value when it has one: "1,681 cases in 72 rows" does not answer 72.
            given = [answer.value] if getattr(answer, "value", None) is not None else found
            checks["avoids_wrong_value"] = not any(
                close(n, form) for n in given for form in share_forms(expected["wrong_value"])
            )
    return got, checks


def run_claim(case):
    checker = getattr(ada, "check_claim", None)
    if checker is None:
        return {"unsupported": "no claim-checking interface"}, {"supported": False}
    if case.get("folder"):
        source = folder_of(case["dataset"])
    else:
        source = folder_of(case["dataset"]) / case["file"]
    result = checker(source, case["claim"])
    got = result.to_dict()
    expected = case["expected"]
    checks = {"status": got.get("status") == expected["status"]}
    if "reason" in expected:
        checks["reason"] = got.get("reason") == expected["reason"]
    if "value" in expected:
        computed = (got.get("result") or {}).get("value")
        checks["value"] = any(close(computed, form) for form in share_forms(expected["value"]))
    evidence = got.get("evidence") or {}
    if "rows_used" in expected:
        checks["rows_used"] = evidence.get("rows_used") == expected["rows_used"]
    if "rows_excluded" in expected:
        left = sum(item.get("count", 0) for item in evidence.get("left_out", []))
        checks["rows_excluded"] = left == expected["rows_excluded"]
    if "candidates" in expected:
        named = json.dumps([got.get("alternatives"), got.get("clarification")])
        checks["candidates"] = all(c in named for c in expected["candidates"])
    return got, checks


def run_folder(case):
    scanner = getattr(ada, "scan", None)
    if scanner is None:
        return {"unsupported": "no folder interface"}, {"supported": False}
    result = scanner(folder_of(case["dataset"]))
    got = result.to_dict()
    links = got.get("links", [])
    checks = {}

    def same(first, second):
        # A composite key names the same columns in any order: "t.(a, b)" is "t.(b, a)".
        def parts(label):
            table, _, columns = str(label).partition(".")
            return table, frozenset(column.strip() for column in columns.strip("()").split(","))
        return parts(first) == parts(second)

    wanted = case["expected"].get("links", [])
    matched = []
    for want in wanted:
        def found(start, end):
            return [link for link in links if same(link.get("from"), start) and same(link.get("to"), end)]
        hit = found(want["from"], want["to"]) or found(want["to"], want["from"])
        matched.extend(hit)
        checks[f"link:{want['from']}->{want['to']}"] = bool(hit)
    if not wanted:
        checks["no_links_proposed"] = not links
    # Links nobody expected are listed for review, not scored: an expected list
    # names the links the dataset's documentation gives, which may not be all.
    got["extra_links"] = [f"{link['from']} -> {link['to']} ({link['kind']})" for link in links
                          if link not in matched]
    return got, checks


RUNNERS = {"profile": run_profile, "compute": run_compute, "claim": run_claim, "folder": run_folder}

results = []
for case in spec["cases"]:
    case_split = datasets[case["dataset"]]["split"]
    if split != "all" and case_split != split:
        continue
    if only and case["id"] not in only:
        continue
    started = time.time()
    try:
        got, checks = RUNNERS[case["kind"]](case)
        error = None
    except Exception as exc:  # a crash is a result too
        got, checks, error = {}, {"ran": False}, "".join(traceback.format_exception_only(exc)).strip()
    results.append({
        "id": case["id"], "dataset": case["dataset"], "split": case_split, "kind": case["kind"],
        "tests": case["tests"], "expected": case["expected"], "checks": checks,
        "passed_automatic": bool(checks) and all(checks.values()), "error": error,
        "seconds": round(time.time() - started, 1), "got": got,
    })
    mark = "ok  " if results[-1]["passed_automatic"] else "FAIL"
    print(f"{mark} {case['id']:40s} {json.dumps(checks)[:150]}{'  ERROR ' + error[:120] if error else ''}",
          flush=True)
    for extra in got.get("extra_links", []) if isinstance(got, dict) else []:
        print(f"       extra link {extra}", flush=True)

out_path.write_text(json.dumps({"ada": ada.__version__, "split": split, "results": results}, indent=1,
                               default=str))
print(f"{sum(r['passed_automatic'] for r in results)}/{len(results)} pass the automatic checks")
