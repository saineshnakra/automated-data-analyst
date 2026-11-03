"""MATH-500 with and without ADA: does a small local model get more competition problems right when it can
call ada_compute for exact arithmetic and algebra?

    curl -L -o math500.jsonl https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
    pip install "math-verify[antlr4_13_2]==0.9.0"
    ollama pull qwen3:1.7b
    python benchmarks/math500.py math500.jsonl --model qwen3:1.7b --condition none --out none.jsonl
    python benchmarks/math500.py --summary none.jsonl tool.jsonl

Each problem is solved under one condition, with the same model and settings:

- ``none``: the model alone.
- ``tool``: ``ada_compute`` offered, with a system prompt that says what it does.
- ``check``: the model alone, then ADA checks every equation it wrote (``ada.compute.check_arithmetic``). When
  one is wrong, the model is told which, with the right value, and answers again; otherwise the answer stands.
- ``prompt`` and ``schema``: the two halves of ``tool``, to tell which one changes the answers: the tool's
  system prompt with no tool offered, and the tool offered with the plain system prompt.

One run does one condition, or several separated by commas (``none,check`` asks once and checks that reply),
and ``--shard 0/4`` takes every fourth problem from the first, so conditions and
shards can run side by side on separate machines. ``--think`` turns on a thinking model's thinking (Qwen3's),
with the sampling its makers recommend for it (temperature 0.6, top-p 0.95, top-k 20; seed 0, so a run
repeats) and room for a long trace. ``--subjects`` keeps the problems
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

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from gsm8k import chat, solve  # noqa: E402

from ada.compute import check_arithmetic  # noqa: E402

PROMPT = "Solve the problem. Think step by step. Put the final answer in \\boxed{}."
TOOL_PROMPT = (
    PROMPT + " You can call ada_compute to work out any step exactly: arithmetic (2^10 + 3/4, sqrt(72)), "
    "simplifying an expression with letters ((x + 1)^2 - x^2), or solving equations written in plain text, "
    "one or several separated by ; (x^2 - 5x + 6 = 0, x*y = 6; x + y = 5, solve a*x^2 + b*x + c = 0 for x). "
    "Use the values it returns."
)
CONDITIONS = ("none", "tool", "check", "prompt", "schema")
#: For each condition: whether ada_compute is offered, and the system prompt.
SETTINGS = {"none": (False, PROMPT), "tool": (True, TOOL_PROMPT), "check": (False, PROMPT),
            "prompt": (False, TOOL_PROMPT), "schema": (True, PROMPT)}
SUBJECTS = ("Algebra", "Intermediate Algebra")
#: The longest reply, in tokens: MATH's worked solutions run longer than GSM8K's.
MAX_TOKENS = 3072
#: With thinking on: the longest reply, thinking included, and the settings Qwen3 recommends for it.
THINK_TOKENS = 14336
THINK_OPTIONS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "num_ctx": 16384}


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


def solved_under(condition: str, base: str, model: str, question: str, think: bool = False,
                 alone: dict | None = None) -> dict:
    """The reply under one condition. ``check`` starts from ``alone``, the model's reply on its own, when it
    is given, rather than asking again for the same reply."""
    offered, system = SETTINGS[condition]
    tokens, options = (THINK_TOKENS, THINK_OPTIONS) if think else (MAX_TOKENS, None)
    if alone is None or condition != "check":
        alone = solve(base, model, question, offered, (system, system), tokens, think, options)
    if condition != "check":
        return alone
    checks = check_arithmetic(alone["reply"])
    # A model that repeats itself writes the same slip many times; it is named once.
    wrong = list({(item.expression, item.stated): item for item in checks if not item.correct}.values())
    if not wrong:
        return {**alone, "checks": [item.to_dict() for item in checks]}
    errors = "; ".join(f"{item.expression} is {item.computed}, not {item.stated}" for item in wrong)
    messages = alone["messages"] + [{"role": "user", "content": (
        f"ADA checked the arithmetic in your answer and found errors: {errors}. Correct every step that uses "
        "these numbers, then put the final answer in \\boxed{}.")}]
    message = chat(base, model, messages, [], tokens, think, options)
    return {"reply": message.get("content", ""), "calls": [], "checks": [item.to_dict() for item in checks],
            "first_reply": alone["reply"]}


def chosen(rows: list[dict], subjects: list[str], problems: int | None, shard: str) -> list[dict]:
    rows = [row for row in rows if row["subject"] in subjects][:problems]
    part, parts = (int(value) for value in shard.split("/"))
    return rows[part::parts]


def run(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    rows = chosen(rows, args.subjects, args.problems, args.shard)
    conditions = args.condition.split(",")
    out = Path(args.out)
    done = set()
    if out.exists():
        # A run that stopped picks up where it left off.
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                done.add((record["id"], record["condition"]))
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            alone = None
            for condition in conditions:
                if (row["unique_id"], condition) in done:
                    continue
                started = time.monotonic()
                try:
                    solved = solved_under(condition, args.base_url, args.model, row["problem"], args.think,
                                          alone)
                except (OSError, http.client.HTTPException) as error:
                    # A reply that never ends or a dropped connection counts as no answer.
                    solved = {"reply": "", "calls": [], "error": str(error)}
                if condition == "none" and "error" not in solved:
                    alone = solved
                given, correct = grade(row["answer"], solved["reply"])
                record = {"id": row["unique_id"], "subject": row["subject"], "level": row["level"],
                          "condition": condition, "model": args.model, "think": args.think,
                          "gold": row["answer"], "given": given, "correct": correct,
                          "seconds": round(time.monotonic() - started, 1),
                          "thinking_chars": len(solved.get("thinking", "")),
                          "tool_calls": len(solved["calls"]), "reply": solved["reply"],
                          "calls": solved["calls"],
                          **{key: solved[key] for key in ("checks", "first_reply", "error") if key in solved}}
                handle.write(json.dumps(record) + "\n")
                handle.flush()
    return 0


def summary(paths: list[str]) -> int:
    """Accuracy per model and condition, each condition against the model alone on the problems both finished,
    by subject and level, with an exact McNemar test on the problems where the two disagree."""
    from scipy.stats import binomtest  # noqa: PLC0415

    by = defaultdict(dict)
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                model = record["model"] + (" (thinking)" if record.get("think") else "")
                by[(model, record["id"])][record["condition"]] = record
    for model in sorted({model for model, _ in by}):
        problems = [conditions for (name, _), conditions in sorted(by.items()) if name == model]
        alone = [conditions for conditions in problems if "none" in conditions]
        right = sum(conditions["none"]["correct"] for conditions in alone)
        print(f"{model}: alone {right}/{len(alone)} = {100 * right / max(len(alone), 1):.1f}%")
        for condition in CONDITIONS[1:]:
            pairs = [conditions for conditions in alone if condition in conditions]
            if not pairs:
                continue
            base = sum(pair["none"]["correct"] for pair in pairs)
            with_it = sum(pair[condition]["correct"] for pair in pairs)
            gained = sum(pair[condition]["correct"] and not pair["none"]["correct"] for pair in pairs)
            lost = sum(pair["none"]["correct"] and not pair[condition]["correct"] for pair in pairs)
            p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
            print(f"  {condition:6} {with_it}/{len(pairs)} = {100 * with_it / len(pairs):.1f}% "
                  f"(alone {base}); right only with it {gained}, only alone {lost}; "
                  f"exact McNemar p = {p:.3g}")
            groups = defaultdict(list)
            for pair in pairs:
                groups[(pair["none"]["subject"], pair["none"]["level"])].append(pair)
            for (subject, level), items in sorted(groups.items()):
                print(f"    {subject}, level {level}: {len(items)} problems; alone "
                      f"{sum(item['none']['correct'] for item in items)}, {condition} "
                      f"{sum(item[condition]['correct'] for item in items)}")
            calls = [pair[condition]["tool_calls"] for pair in pairs]
            if any(calls):
                print(f"    tool called on {sum(1 for value in calls if value)} problems, {sum(calls)} calls")
            if condition == "check":
                flagged = sum(1 for pair in pairs
                              if any(not item["correct"] for item in pair["check"].get("checks", [])))
                print(f"    answers with a wrong equation: {flagged}")
        failed = sum(1 for conditions in problems for record in conditions.values() if record.get("error"))
        print(f"  replies that never ended or failed: {failed}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", nargs="?", help="MATH-500 test.jsonl")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--condition", default="none",
                        help="one of " + ", ".join(CONDITIONS) + ", or several separated by commas; with "
                             "none and check together, check starts from the reply none got")
    parser.add_argument("--think", action="store_true", help="turn on a thinking model's thinking")
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
    unknown = [name for name in args.condition.split(",") if name not in CONDITIONS]
    if unknown:
        parser.error(f"unknown condition {unknown[0]}; the conditions are {', '.join(CONDITIONS)}")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
