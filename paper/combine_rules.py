"""Ways of combining the model's own answer with ADA's, scored on a run that has none, check and solve.

    python paper/combine_rules.py paper/runs/RUN --gsm8k test.jsonl --gsmhard gsmhardv2.jsonl

Each rule takes the answer both give when the model's own answer (``none``) and ADA's (``solve``) agree, and
``check``'s when ADA has none. They differ in what they take when the two answers differ:

- ADA's: ADA's answer.
- check decides: ADA's when ADA found a wrong equation in the model's own answer, the model's otherwise.
- check or large numbers: as "check decides", and ADA's too when the question has a number of five or more
  digits. This is benchmarks/gsm8k.py's ``combine`` condition.
- either right: right when either answer is right, the most any rule could get.

The output is Markdown and the same each time.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "benchmarks")]

from analysis import grouped, load, slipped  # noqa: E402
from gsm8k import agree, large_numbers  # noqa: E402

RULES = ("ADA's", "check decides", "check or large numbers", "either right")
TOLERANCE = {"gsm8k": Fraction(0), "gsmhard": Fraction(1, 1000)}


def questions(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row.get("question", row.get("input")) for row in rows]


def scored(found: dict, question: str, tolerance: Fraction) -> dict[str, bool]:
    """Whether each rule's answer is right for one problem."""
    alone, checked, solved = found["none"], found["check"], found["solve"]
    best = alone["correct"] or solved["correct"]
    if solved.get("given") is None:
        return {**{rule: checked["correct"] for rule in RULES[:-1]}, "either right": best}
    if agree(alone["given"], solved["given"], tolerance):
        return {**{rule: alone["correct"] for rule in RULES[:-1]}, "either right": best}
    slip = slipped(checked)
    large = large_numbers(question)
    return {"ADA's": solved["correct"],
            "check decides": solved["correct"] if slip else alone["correct"],
            "check or large numbers": solved["correct"] if slip or large else alone["correct"],
            "either right": best}


def report(records: list[dict], texts: dict[str, list[str]]) -> str:
    groups = grouped(records)
    lines = []
    for dataset in sorted({dataset for dataset, _ in groups}):
        lines += [f"## {dataset}", "", "| Model | Problems | Disagree | none | check | solve | "
                  + " | ".join(RULES) + " |", "|---|---|---|---|---|---|" + "---|" * len(RULES)]
        totals = dict.fromkeys(("none", "check", "solve", *RULES), 0)
        for model in sorted(model for data, model in groups if data == dataset):
            problems = groups[(dataset, model)]
            done = [p for p, found in problems.items() if all(c in found for c in ("none", "check", "solve"))]
            counts = dict.fromkeys(totals, 0)
            disagree = 0
            for problem in done:
                found = problems[problem]
                for condition in ("none", "check", "solve"):
                    counts[condition] += found[condition]["correct"]
                for rule, right in scored(found, texts[dataset][int(problem)], TOLERANCE[dataset]).items():
                    counts[rule] += right
                given = found["solve"].get("given")
                disagree += given is not None and not agree(found["none"]["given"], given, TOLERANCE[dataset])
            for name in totals:
                totals[name] += counts[name]
            cells = " | ".join(str(counts[name]) for name in totals)
            lines.append(f"| {model} | {len(done)} | {disagree} | {cells} |")
        lines += ["| All | | | " + " | ".join(str(value) for value in totals.values()) + " |", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--gsm8k", type=Path, required=True, help="GSM8K test.jsonl")
    parser.add_argument("--gsmhard", type=Path, required=True, help="GSM-Hard gsmhardv2.jsonl")
    args = parser.parse_args()
    print(report(load(args.run), {"gsm8k": questions(args.gsm8k), "gsmhard": questions(args.gsmhard)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
