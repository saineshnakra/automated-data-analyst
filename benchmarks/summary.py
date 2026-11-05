"""Accuracy per condition in a results file from gsm8k.py, on the problems every condition in it finished.

    python benchmarks/summary.py results.jsonl
"""

import json
import sys
from collections import Counter, defaultdict

CONDITIONS = ("none", "tool", "check", "solve")


def main(path: str) -> int:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by = defaultdict(dict)
    for row in rows:
        by[row["index"]][row["condition"]] = row
    conditions = [condition for condition in CONDITIONS if any(condition in found for found in by.values())]
    complete = sorted(index for index in by if all(condition in by[index] for condition in conditions))
    if not complete:
        print("No problem has every condition yet.")
        return 1
    print(f"{len(complete)} problems")
    for condition in conditions:
        right = sum(by[index][condition]["correct"] for index in complete)
        print(f"  {condition:6} {right}/{len(complete)} = {100 * right / len(complete):.1f}%")
    alone = {index: by[index]["none"]["correct"] for index in complete} if "none" in conditions else None
    if "check" in conditions:
        flagged = [index for index in complete
                   if any(not item["correct"] for item in by[index]["check"].get("checks", []))]
        print(f"  check: {len(flagged)} answers had a wrong equation", end="")
        if alone is not None:
            checked = {index: by[index]["check"]["correct"] for index in complete}
            fixed = sum(checked[index] and not alone[index] for index in complete)
            broke = sum(alone[index] and not checked[index] for index in complete)
            print(f"; fixed {fixed}, broke {broke}", end="")
        print()
        equations = [item for index in complete for item in by[index]["check"].get("checks", [])]
        slips = sum(not item["correct"] for item in equations)
        print(f"  check: {len(equations)} equations written in the model's own answers, {slips} wrong")
    if "tool" in conditions:
        used = [index for index in complete if by[index]["tool"]["tool_calls"]]
        print(f"  tool: called on {len(used)} problems")
    if "solve" in conditions:
        statuses = Counter(by[index]["solve"].get("ada", {}).get("status", "failed") for index in complete)
        print("  solve: " + ", ".join(f"{status} {count}" for status, count in statuses.most_common()))
        if alone is not None:
            solved = {index: by[index]["solve"] for index in complete}
            either = {index: solved[index]["correct"] if solved[index]["given"] is not None else alone[index]
                      for index in complete}
            right = sum(either.values())
            gained = sum(either[index] and not alone[index] for index in complete)
            lost = sum(alone[index] and not either[index] for index in complete)
            print(f"  solve, else the model alone when ADA has no value: {right}/{len(complete)} = "
                  f"{100 * right / len(complete):.1f}%; right only with it {gained}, only alone {lost}")
    stuck = sum(1 for row in rows if row.get("error"))
    print(f"  replies that never ended or failed: {stuck}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
