"""Accuracy per condition in a results file from gsm8k.py, on the problems every condition finished.

    python benchmarks/summary.py results.jsonl
"""

import json
import sys
from collections import defaultdict

CONDITIONS = ("none", "tool", "check")


def main(path: str) -> int:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by = defaultdict(dict)
    for row in rows:
        by[row["index"]][row["condition"]] = row
    complete = sorted(index for index in by if all(condition in by[index] for condition in CONDITIONS))
    if not complete:
        print("No problem has all three conditions yet.")
        return 1
    print(f"{len(complete)} problems")
    for condition in CONDITIONS:
        right = sum(by[index][condition]["correct"] for index in complete)
        print(f"  {condition:6} {right}/{len(complete)} = {100 * right / len(complete):.1f}%")
    flagged = [index for index in complete
               if any(not item["correct"] for item in by[index]["check"].get("checks", []))]
    alone = {index: by[index]["none"]["correct"] for index in complete}
    checked = {index: by[index]["check"]["correct"] for index in complete}
    fixed = [index for index in complete if checked[index] and not alone[index]]
    broke = [index for index in complete if alone[index] and not checked[index]]
    print(f"  check: {len(flagged)} answers had a wrong equation; fixed {len(fixed)}, broke {len(broke)}")
    used = [index for index in complete if by[index]["tool"]["tool_calls"]]
    print(f"  tool: called on {len(used)} problems")
    stuck = sum(1 for row in rows if row.get("error"))
    print(f"  replies that never ended or failed: {stuck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
