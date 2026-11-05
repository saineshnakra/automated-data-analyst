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
- ``solve``: the model writes the problem as equations without solving it, names the quantity asked for,
  and ADA solves the equations exactly (``formalize.py``). ADA's value is the answer; when ADA cannot solve
  them, there is none.

One run does one condition, or several separated by commas (``none,check`` asks once and checks that reply),
and ``--shard 0/4`` takes every fourth problem from the first, so conditions and
shards can run side by side on separate machines. ``--think`` turns on a thinking model's thinking (Qwen3's),
with the sampling its makers recommend for it (temperature 0.6, top-p 0.95, top-k 20; seed 0, so a run
repeats) and room for a long trace. ``--subjects`` keeps the problems
of the subjects named (by default Algebra and Intermediate Algebra, the subjects ADA's algebra covers).

``--samples 5`` asks up to five times, with seeds 0 to 4, and takes the answer most replies give, two ways:
every reply's answer, and only the answers of replies ADA finds no wrong equation in. Asking stops as soon
as more asks cannot change either vote, so the votes are those of all five replies. ``--resample-above
20000`` asks again only when the first reply gave no answer or its thinking ran over 20,000 characters;
otherwise the first answer stands. ``--think-tokens 30720`` gives each reply more room than the 14,336 tokens
it has by default; a reply in the results that gave no answer in fewer tokens is asked again with the larger
room, and so is the check that started from it. A reply that did not end in time, which takes over two hours
on a slow runner, is asked again too. With thinking on, each record keeps the thinking.

The model ends with its answer in \\boxed{}, as MATH's solutions do. Math-Verify, the grader made for MATH,
compares it with the answer in the data as mathematics, so 14/3 and \\frac{14}{3} agree. Math-Verify grades
the benchmark only; ADA does not depend on it. Temperature is 0 and thinking is off, so a run repeats.
Two replies give the same answer when Math-Verify finds them equal, so the votes use no gold answer.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

from formalize import FORMAL_PROMPT, formal_messages, latex_value, solved_by_ada  # noqa: E402
from gsm8k import chat, solve  # noqa: E402

from ada.compute import check_arithmetic  # noqa: E402

PROMPT = "Solve the problem. Think step by step. Put the final answer in \\boxed{}."
TOOL_PROMPT = (
    PROMPT + " You can call ada_compute to work out any step exactly: arithmetic (2^10 + 3/4, sqrt(72)), "
    "simplifying an expression with letters ((x + 1)^2 - x^2), or solving equations written in plain text, "
    "one or several separated by ; (x^2 - 5x + 6 = 0, x*y = 6; x + y = 5, solve a*x^2 + b*x + c = 0 for x). "
    "Use the values it returns."
)
CONDITIONS = ("none", "tool", "check", "prompt", "schema", "solve")
#: For each condition: whether ada_compute is offered, and the system prompt.
SETTINGS = {"none": (False, PROMPT), "tool": (True, TOOL_PROMPT), "check": (False, PROMPT),
            "prompt": (False, TOOL_PROMPT), "schema": (True, PROMPT), "solve": (False, FORMAL_PROMPT)}
SUBJECTS = ("Algebra", "Intermediate Algebra")
#: The longest reply, in tokens: MATH's worked solutions run longer than GSM8K's.
MAX_TOKENS = 3072
#: With thinking on: the longest reply by default, thinking included, and the settings Qwen3 recommends.
THINK_TOKENS = 14336
THINK_OPTIONS = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
#: Room in the context for the system prompt and the problem, beside the longest reply.
PROMPT_TOKENS = 2048


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


def wrong_equations(checks: list) -> list:
    """The equations ADA found wrong, each once: a model that repeats itself writes the same slip many
    times."""
    return list({(item.expression, item.stated): item for item in checks if not item.correct}.values())


def answer_of(reply: str, parsed: dict) -> list:
    """The answer Math-Verify reads from a reply, parsed once."""
    from math_verify import parse  # noqa: PLC0415

    if reply not in parsed:
        parsed[reply] = parse(reply)
    return parsed[reply]


def agreeing(record: dict, earlier: list[dict], parsed: dict) -> int | None:
    """The sample whose answer this reply gives: the first earlier sample, among those that start a group of
    equal answers, that Math-Verify finds equal to it, else its own; None when it gives no answer."""
    given = answer_of(record["reply"], parsed)
    if not given:
        return None
    from math_verify import verify  # noqa: PLC0415

    for other in earlier:
        if other.get("agrees_with") == other["sample"]:
            try:
                if verify(answer_of(other["reply"], parsed), given):
                    return other["sample"]
            except Exception:  # noqa: BLE001, S112 - answers Math-Verify cannot compare count as different
                continue
    return record["sample"]


def counted(records: list[dict], ada: bool) -> list[dict]:
    """The replies that vote: those with an answer and, with ``ada``, no equation ADA found wrong."""
    answered = [record for record in records if record.get("agrees_with") is not None]
    return [record for record in answered if not record.get("wrong_equations")] if ada else answered


def vote(records: list[dict], ada: bool = False) -> dict | None:
    """The first reply to give the answer most replies give; a tie goes to the answer given first. With
    ``ada``, only replies ADA found no wrong equation in vote, unless it found one in every reply."""
    voters = counted(records, ada) or counted(records, False)
    counts = Counter(record["agrees_with"] for record in voters)
    if not counts:
        return None
    first = min(counts, key=lambda sample: (-counts[sample], sample))
    return next(record for record in records if record["sample"] == first)


def decided(records: list[dict], remaining: int, ada: bool = False) -> bool:
    """Whether ``remaining`` more replies cannot change the vote."""
    counts = sorted(Counter(record["agrees_with"] for record in counted(records, ada)).values(), reverse=True)
    counts += [0, 0]
    return counts[0] > counts[1] + remaining


def solved_under(condition: str, base: str, model: str, question: str, think: bool = False,
                 alone: dict | None = None, seed: int = 0, budget: int = THINK_TOKENS) -> dict:
    """The reply under one condition, in up to ``budget`` tokens with thinking on. ``check`` starts from
    ``alone``, the model's reply on its own, when it is given, rather than asking again for the same reply."""
    offered, system = SETTINGS[condition]
    tokens, options = ((budget, {**THINK_OPTIONS, "num_ctx": budget + PROMPT_TOKENS, "seed": seed}) if think
                       else (MAX_TOKENS, None))
    if condition == "solve":
        message = chat(base, model, formal_messages(question), [], tokens, think, options)
        ada = solved_by_ada(message.get("content", ""))
        reply = "" if ada["value"] is None else f"\\boxed{{{latex_value(ada['value'])}}}"
        return {"reply": reply, "calls": [], "formal": message.get("content", ""), "ada": ada,
                "thinking": message.get("thinking", "")}
    if alone is None or condition != "check":
        alone = solve(base, model, question, offered, (system, system), tokens, think, options)
    if condition != "check":
        return alone
    checks = check_arithmetic(alone["reply"])
    wrong = wrong_equations(checks)
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


def asked(args: argparse.Namespace, row: dict, condition: str, sample: int, alone: dict | None
          ) -> tuple[dict, dict]:
    """One reply under one condition with one seed, graded and checked, and the reply as ``solve`` gave it."""
    started = time.monotonic()
    try:
        solved = solved_under(condition, args.base_url, args.model, row["problem"], args.think, alone, sample,
                              args.think_tokens)
    except (OSError, http.client.HTTPException) as error:
        # A reply that never ends or a dropped connection counts as no answer, and is asked again when a run
        # picks up the results.
        solved = {"reply": "", "calls": [], "error": str(error)}
    given, correct = grade(row["answer"], solved["reply"])
    record = {"id": row["unique_id"], "subject": row["subject"], "level": row["level"],
              "condition": condition, "model": args.model, "think": args.think, "sample": sample,
              "samples": args.samples, "gold": row["answer"], "given": given, "correct": correct,
              "seconds": round(time.monotonic() - started, 1), "max_tokens": budget_of(args),
              "thinking_chars": len(solved.get("thinking", "")), "tool_calls": len(solved["calls"]),
              "wrong_equations": len(wrong_equations(check_arithmetic(solved["reply"]))),
              "reply": solved["reply"], "calls": solved["calls"],
              **({"thinking": solved["thinking"]} if solved.get("thinking") else {}),
              **{key: solved[key] for key in ("checks", "first_reply", "formal", "ada", "error")
                 if key in solved}}
    return record, solved


def budget_of(args: argparse.Namespace) -> int:
    """The most tokens a reply may use in this run."""
    return args.think_tokens if args.think else MAX_TOKENS


def outgrown(record: dict, budget: int) -> bool:
    """Whether a reply gave no answer in fewer tokens than ``budget``: then it is asked again with more."""
    used = record.get("max_tokens", THINK_TOKENS if record.get("think") else MAX_TOKENS)
    return record.get("agrees_with") is None and used < budget


def uncertain(earlier: dict[str, list[dict]], limit: int | None) -> bool:
    """Whether the first reply leaves the problem open: it gave no answer, or its thinking ran over ``limit``
    characters. With no limit, every problem is open."""
    if limit is None:
        return True
    records = earlier.get("none") or next(iter(earlier.values()), [])
    first = next((record for record in records if record.get("sample", 0) == 0), None)
    return first is None or first.get("agrees_with", first.get("given")) is None or \
        first.get("thinking_chars", 0) > limit


def settled(earlier: dict[str, list[dict]], conditions: list[str], remaining: int) -> bool:
    """Whether ``remaining`` more samples cannot change any vote: each condition's, and for the model alone
    also the vote of the replies ADA found no wrong equation in."""
    votes = [(condition, False) for condition in conditions] + [("none", True)] * ("none" in conditions)
    return all(decided(earlier[condition], remaining, ada) for condition, ada in votes)


def completed(records: list[dict], parsed: dict) -> bool:
    """Gives records written before samples were voted on their sample number, the count of equations ADA
    finds wrong and the sample they agree with; whether any record lacked them."""
    groups = defaultdict(list)
    for record in records:
        groups[record["id"], record["condition"]].append(record)
    missing = False
    for group in groups.values():
        before = []
        for record in sorted(group, key=lambda item: item.get("sample", 0)):
            if "agrees_with" not in record:
                missing = True
                record.setdefault("sample", 0)
                record.setdefault("wrong_equations", len(wrong_equations(check_arithmetic(record["reply"]))))
                record["agrees_with"] = agreeing(record, before, parsed)
            before.append(record)
    return missing


def replied(row: dict, record: dict) -> dict:
    """The model's reply alone, as ``solve`` gives it, rebuilt from its record for the check to start from."""
    messages = [{"role": "system", "content": SETTINGS["none"][1]},
                {"role": "user", "content": row["problem"]},
                {"role": "assistant", "content": record["reply"]}]
    return {"reply": record["reply"], "calls": [], "messages": messages}


def stale(check: dict, alone: dict | None) -> bool:
    """Whether ADA, as it checks now, finds other wrong equations in the reply a check record started from
    than it found when the record was made: then the record is made again."""
    if alone is None or "checks" not in check:
        return False
    now = {(item.expression, item.stated) for item in wrong_equations(check_arithmetic(alone["reply"]))}
    then = {(item["expression"], item["stated"]) for item in check["checks"] if not item["correct"]}
    return now != then


def run(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    rows = chosen(rows, args.subjects, args.problems, args.shard)
    conditions = args.condition.split(",")
    out = Path(args.out)
    parsed = {}
    # A run that stopped, or an earlier run's results, picks up where it left off. Results for other problems,
    # from an earlier run split another way, are left out.
    ids = {row["unique_id"] for row in rows}
    found = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
             if line.strip()] if out.exists() else []
    records = [record for record in found if record["id"] in ids]
    if completed(records, parsed) or len(records) < len(found):
        out.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    # A record made again comes later in the file and takes the place of the one before.
    latest = {(record["id"], record["condition"], record["sample"]): record for record in records}
    previous = defaultdict(list)
    for record in latest.values():
        previous[record["id"], record["condition"]].append(record)
    with out.open("a", encoding="utf-8") as handle:
        for row in rows:
            earlier = {condition: sorted(previous[row["unique_id"], condition],
                                         key=lambda item: item["sample"]) for condition in conditions}
            for sample in range(args.samples):
                if sample == 1 and not uncertain(earlier, args.resample_above):
                    break
                if args.samples > 1 and settled(earlier, conditions, args.samples - sample):
                    break
                alone, renewed = None, False
                for condition in conditions:
                    done = [record for record in earlier[condition] if record["sample"] == sample]
                    if condition == "check" and alone is None:
                        # The check starts from the reply the model gave alone, without asking for it again.
                        source = next((record for record in earlier.get("none", [])
                                       if record["sample"] == sample and not record.get("error")), None)
                        alone = replied(row, source) if source else None
                    # A reply with no answer in fewer tokens than this run allows is asked again, and so is
                    # the check that started from it; so is a reply that did not end in time.
                    again = done and (outgrown(done[0], budget_of(args)) or bool(done[0].get("error")) or
                                      condition == "check" and (renewed or stale(done[0], alone)))
                    if done and not again:
                        continue
                    if done:
                        earlier[condition].remove(done[0])
                    record, solved = asked(args, row, condition, sample, alone)
                    if condition == "none":
                        renewed = True
                        if "error" not in solved:
                            alone = solved
                    record["agrees_with"] = agreeing(record, earlier[condition], parsed)
                    earlier[condition].append(record)
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    # One line a reply, so a job's log shows how far it has got.
                    print(f"{record['id']} {condition} sample {sample}: "
                          f"{'right' if record['correct'] else 'wrong'}, {record['seconds']} s", flush=True)
    return 0


def summary(paths: list[str], resample_above: int | None = None) -> int:
    """Accuracy per model and condition, each condition against the model alone on the problems both finished,
    by subject and level, with an exact McNemar test on the problems where the two disagree."""
    from scipy.stats import binomtest  # noqa: PLC0415

    by = defaultdict(dict)
    sampled = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                model = record["model"] + (" (thinking)" if record.get("think") else "")
                sampled[(model, record["id"])][record["condition"]][record.get("sample", 0)] = record
                if record.get("sample", 0) == 0:
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
            if condition == "solve":
                statuses = Counter(pair["solve"].get("ada", {}).get("status", "failed") for pair in pairs)
                print("    ADA: " + ", ".join(f"{status} {count}"
                                             for status, count in statuses.most_common()))
                either = [pair["solve"]["correct"] if pair["solve"].get("ada", {}).get("value") is not None
                          else pair["none"]["correct"] for pair in pairs]
                print(f"    solve, else the model alone when ADA has no value: {sum(either)}/{len(pairs)} = "
                      f"{100 * sum(either) / len(pairs):.1f}%")
        failed = sum(1 for conditions in problems for record in conditions.values() if record.get("error"))
        print(f"  replies that never ended or failed: {failed}")
        budgets = Counter(conditions["none"].get("max_tokens") or 0 for conditions in alone)
        if len(budgets) > 1:
            print("  first replies alone, by the most tokens they could use: " + ", ".join(
                f"{budget or 'not recorded'}: {count}" for budget, count in sorted(budgets.items())))
        voted([{condition: [records[sample] for sample in sorted(records)]
                for condition, records in conditions.items()}
               for (name, _), conditions in sorted(sampled.items()) if name == model], binomtest,
              resample_above)
    return 0


def voted(problems: list[dict], binomtest, resample_above: int | None = None) -> None:
    """For runs with several samples, on the problems whose votes are settled: the vote of every answer and
    the vote of the answers ADA found no wrong equation in, against the first sample alone. With
    ``resample_above``, a problem whose first reply gave an answer after a short thought is settled by it."""
    size = max([record.get("samples", 1) for conditions in problems for records in conditions.values()
                for record in records] or [1])
    if size < 2:
        return
    finished = []
    for conditions in problems:
        if "none" not in conditions:
            continue
        asked = len(conditions["none"])
        if asked >= size or settled(conditions, list(conditions), size - asked) or \
                asked == 1 and not uncertain(conditions, resample_above):
            finished.append(conditions)
    if not finished:
        return
    first = [conditions["none"][0]["correct"] for conditions in finished]

    def right(chosen: dict | None) -> bool:
        return bool(chosen and chosen["correct"])

    def line(label: str, values: list[bool], against: list[bool], against_label: str) -> None:
        gained = sum(value and not other for value, other in zip(values, against, strict=True))
        lost = sum(other and not value for value, other in zip(values, against, strict=True))
        p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
        share = 100 * sum(values) / len(values)
        print(f"    {label}: {sum(values)}/{len(values)} = {share:.1f}%; right only with it {gained}, "
              f"only with {against_label} {lost}; exact McNemar p = {p:.3g}")

    asked = sum(len(conditions["none"]) for conditions in finished) / len(finished)
    print(f"  votes of up to {size} samples, on {len(finished)} problems ({asked:.2f} samples each):")
    if resample_above is not None:
        opened = sum(1 for conditions in finished if uncertain(conditions, resample_above))
        print(f"    asked again on the {opened} whose first reply gave no answer or thought for more than "
              f"{resample_above:,} characters")
    print(f"    first sample: {sum(first)}/{len(first)} = {100 * sum(first) / len(first):.1f}%")
    every = [right(vote(conditions["none"])) for conditions in finished]
    line("vote of every answer", every, first, "the first sample")
    line("vote of the answers ADA found no wrong equation in", [right(vote(conditions["none"], True))
                                                                for conditions in finished],
         every, "the vote of every answer")
    if all("check" in conditions for conditions in finished):
        line("vote of the checked answers", [right(vote(conditions["check"])) for conditions in finished],
             every, "the vote of every answer")
    some = sum(any(record["correct"] for record in conditions["none"]) for conditions in finished)
    print(f"    some sample right: {some}/{len(finished)} = {100 * some / len(finished):.1f}%")
    failed = sum(1 for conditions in finished for records in conditions.values() for record in records
                 if record.get("error"))
    print(f"    replies that never ended or failed, in every sample: {failed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data", nargs="?", help="MATH-500 test.jsonl")
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--condition", default="none",
                        help="one of " + ", ".join(CONDITIONS) + ", or several separated by commas; with "
                             "none and check together, check starts from the reply none got")
    parser.add_argument("--think", action="store_true", help="turn on a thinking model's thinking")
    parser.add_argument("--samples", type=int, default=1,
                        help="ask up to this many times, with seeds 0, 1, ..., and vote (needs --think)")
    parser.add_argument("--think-tokens", type=int, default=THINK_TOKENS, metavar="TOKENS",
                        help=f"with --think, the most tokens a reply may use, thinking included (default "
                             f"{THINK_TOKENS}); a reply in the results that gave no answer in fewer is asked "
                             "again")
    parser.add_argument("--resample-above", type=int, metavar="CHARACTERS",
                        help="with --samples, ask again only when the first reply gave no answer or thought "
                             "for more than this many characters")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--problems", type=int, help="the first N problems of those subjects (default: all)")
    parser.add_argument("--shard", default="0/1", help="part/parts: every parts-th problem from part")
    parser.add_argument("--out", help="write one JSON line per problem here")
    parser.add_argument("--summary", nargs="+", metavar="RESULTS", help="summarise these results files")
    args = parser.parse_args()
    if args.summary:
        return summary(args.summary, args.resample_above)
    if not (args.data and args.out):
        parser.error("give the data file and --out, or --summary with results files")
    unknown = [name for name in args.condition.split(",") if name not in CONDITIONS]
    if unknown:
        parser.error(f"unknown condition {unknown[0]}; the conditions are {', '.join(CONDITIONS)}")
    if args.think_tokens < 1024:
        parser.error("--think-tokens is at least 1024: a thinking reply needs room to think and answer")
    if args.samples < 1 or (args.samples > 1 and not args.think):
        parser.error("--samples is 1 or more, and above 1 needs --think: with thinking off, temperature is 0 "
                     "and every sample gives the same reply")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
