"""Run ADA on the evaluation cases and save what it returned, with automatic numeric checks.

    python -I run.py ADA_CHECKOUT DATA_ROOT HOLDOUT_ROOT OUT.json [--split dev|holdout|all] [--only ID,ID]

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
sys.path.insert(0, str(checkout))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import ada  # noqa: E402

spec = yaml.safe_load((HERE / "cases.yml").read_text())
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


def table_values(table):
    """Label -> every number on that label's row of an answer table."""
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return {}
    numeric = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    labels = [c for c in table.columns if c not in numeric]
    if not numeric:
        return {}
    keys = table[labels[0]] if labels else table.index
    return {str(k): [row[c] for c in numeric] for k, (_, row) in zip(keys, table.iterrows(), strict=True)}


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
            checks[f"value:{label}"] = any(close(v, want) for v in values.get(str(label), []))
    if "value" in expected:
        found = numbers_in(answer.answer) + [v for row in table_values(table).values() for v in row]
        checks["value"] = any(close(n, expected["value"]) for n in found)
        if "wrong_value" in expected:
            checks["avoids_wrong_value"] = not any(close(n, expected["wrong_value"]) for n in found)
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
        checks["value"] = close(computed, expected["value"])
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
    for want in wanted:
        def found(start, end):
            return [link for link in links if same(link.get("from"), start) and same(link.get("to"), end)]
        hit = found(want["from"], want["to"]) or found(want["to"], want["from"])
        checks[f"link:{want['from']}->{want['to']}"] = bool(hit)
    if not wanted:
        checks["no_links_proposed"] = not links
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

out_path.write_text(json.dumps({"ada": ada.__version__, "split": split, "results": results}, indent=1,
                               default=str))
print(f"{sum(r['passed_automatic'] for r in results)}/{len(results)} pass the automatic checks")
