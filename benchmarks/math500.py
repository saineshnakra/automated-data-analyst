"""MATH-500 with and without ADA: does a small local model get more competition problems right when it can
call ada_compute for exact arithmetic and algebra?

    curl -L -o math500.jsonl https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
    pip install "math-verify[antlr4_13_2]==0.9.0"
    ollama pull qwen3:1.7b
    python benchmarks/math500.py math500.jsonl --model qwen3:1.7b --condition none --out none.jsonl
    python benchmarks/math500.py --summary none.jsonl tool.jsonl

Each problem is solved with the model alone (``none``) or with ``ada_compute`` offered (``tool``), with the
same model and settings. One run does one condition, and ``--shard 0/4`` takes every fourth problem from the
first, so conditions and shards can run side by side on separate machines. ``--subjects`` keeps the problems
of the subjects named (by default Algebra and Intermediate Algebra, the subjects ADA's algebra covers).

The model ends with its answer in \\boxed{}, as MATH's solutions do. Math-Verify, the grader made for MATH,
compares it with the answer in the data as mathematics, so 14/3 and \\frac{14}{3} agree. Math-Verify grades
the benchmark only; ADA does not depend on it. Temperature is 0 and thinking is off, so a run repeats.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gsm8k import solve  # noqa: E402

PROMPT = "Solve the problem. Think step by step. Put the final answer in \\boxed{}."
TOOL_PROMPT = (
    PROMPT + " You can call ada_compute to work out any step exactly: arithmetic (2^10 + 3/4, sqrt(72)), "
    "simplifying an expression with letters ((x + 1)^2 - x^2), or solving equations written in plain text, "
    "one or several separated by ; (x^2 - 5x + 6 = 0, x*y = 6; x + y = 5, solve a*x^2 + b*x + c = 0 for x). "
    "Use the values it returns."
)
CONDITIONS = ("none", "tool")
SUBJECTS = ("Algebra", "Intermediate Algebra")
#: The longest reply, in tokens: MATH's worked solutions run longer than GSM8K's.
MAX_TOKENS = 3072


def grade(gold: str, reply: str) -> tuple[str | None, bool]:
    """The answer Math-Verify reads from the reply, and whether it equals the gold answer."""
    from math_verify import parse, verify  # noqa: PLC0415

    expected = parse(f"${gold}$")
    given = parse(reply)
    shown = str(given[-1]) if given else None
    try:
        return shown, bool(given) and bool(verify(expected, given))
    except Exception:  # noqa: BLE001 - a reply Math-Verify cannot compare counts as wrong
        return shown, False


def chosen(rows: list[dict], subjects: list[str], problems: int | None, shard: str) -> list[dict]:
    rows = [row for row in rows if row["subject"] in subjects][:problems]
    part, parts = (int(value) for value in shard.split("/"))
    return rows[part::parts]


def run(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    rows = chosen(rows, args.subjects, args.problems, args.shard)
    out = Path(args.out)
    done = set()
    if out.exists():
        # A run that stopped picks up where it left off.
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line}
    prompts = (PROMPT, TOOL_PROMPT)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["unique_id"] in done:
                continue
            started = time.monotonic()
            try:
                solved = solve(args.base_url, args.model, row["problem"], args.condition == "tool", prompts,
                               MAX_TOKENS)
            except (OSError, http.client.HTTPException) as error:
                # A reply that never ends or a dropped connection counts as no answer.
                solved = {"reply": "", "calls": [], "error": str(error)}
            given, correct = grade(row["answer"], solved["reply"])
            record = {"id": row["unique_id"], "subject": row["subject"], "level": row["level"],
                      "condition": args.condition, "model": args.model, "gold": row["answer"], "given": given,
                      "correct": correct, "seconds": round(time.monotonic() - started, 1),
                      "tool_calls": len(solved["calls"]), "reply": solved["reply"], "calls": solved["calls"],
                      **({"error": solved["error"]} if "error" in solved else {})}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
    return 0


def summary(paths: list[str]) -> int:
    """Accuracy per model and condition, by subject and level, on the problems both conditions finished, with
    an exact McNemar test on the problems where the two disagree."""
    from scipy.stats import binomtest  # noqa: PLC0415

    by = defaultdict(dict)
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                by[(record["model"], record["id"])][record["condition"]] = record
    for model in sorted({model for model, _ in by}):
        pairs = [conditions for (name, _), conditions in sorted(by.items())
                 if name == model and all(condition in conditions for condition in CONDITIONS)]
        if not pairs:
            print(f"{model}: no problem has both conditions yet.")
            continue
        print(f"{model}: {len(pairs)} problems")
        for condition in CONDITIONS:
            right = sum(pair[condition]["correct"] for pair in pairs)
            print(f"  {condition:5} {right}/{len(pairs)} = {100 * right / len(pairs):.1f}%")
        groups = defaultdict(list)
        for pair in pairs:
            groups[(pair["none"]["subject"], pair["none"]["level"])].append(pair)
        for (subject, level), items in sorted(groups.items()):
            counts = ", ".join(f"{condition} {sum(item[condition]['correct'] for item in items)}"
                               for condition in CONDITIONS)
            print(f"    {subject}, level {level}: {len(items)} problems; {counts}")
        gained = sum(pair["tool"]["correct"] and not pair["none"]["correct"] for pair in pairs)
        lost = sum(pair["none"]["correct"] and not pair["tool"]["correct"] for pair in pairs)
        p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
        print(f"  tool against none: right only with the tool {gained}, only without it {lost}; "
              f"exact McNemar p = {p:.3g}")
        used = sum(1 for pair in pairs if pair["tool"]["tool_calls"])
        calls = sum(pair["tool"]["tool_calls"] for pair in pairs)
        print(f"  tool: called on {used} problems, {calls} calls")
        failed = sum(1 for pair in pairs for condition in CONDITIONS if pair[condition].get("error"))
        print(f"  replies that never ended or failed: {failed}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", nargs="?", help="MATH-500 test.jsonl")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--condition", choices=CONDITIONS, default="none")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--problems", type=int, help="the first N problems of those subjects (default: all)")
    parser.add_argument("--shard", default="0/1", help="part/parts: every parts-th problem from part")
    parser.add_argument("--out", help="write one JSON line per problem here")
    parser.add_argument("--summary", nargs="+", metavar="RESULTS", help="summarise these results files")
    args = parser.parse_args()
    if args.summary:
        return summary(args.summary)
    if not (args.data and args.out):
        parser.error("give the data file and --out, or --summary with results files")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
