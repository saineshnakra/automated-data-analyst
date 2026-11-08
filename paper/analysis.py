"""The tables of a run in paper/runs, from its results.jsonl.gz.

    python paper/analysis.py paper/runs/RUN

For each data set and model: problems right under each condition, on the problems every condition of the run
finished; each condition against the model alone (right only with it / right only alone, and the exact McNemar
test, SciPy's binomtest on the problems where the two differ), or against the run's first condition when the
run did not ask the model alone; ADA's statuses under ``solve``, ``json``, ``plan``, ``retry`` and ``pair``,
and how the programs under ``program`` ended; for ``pair``, what asking again about numbers left out did;
``retry`` against ``solve``, split by whether retry's first reply is solve's reply; and "ADA's answer, else
the model's": ADA's value when it solved the equations to an answer, the model's own answer when it did not;
and, for MATH-500, the problems right by subject and level. Only each problem's first sample counts in those
tables, so a run that voted over several samples is read by its first reply there. For GSM8K and GSM-Hard, the
last tables give each condition asked more than once by its first reply and by vote, and each one against each
one before it. A condition this script does not know is named, not counted. The output is Markdown and the
same each time.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ORDER = ("none", "shots", "tool", "prompt", "schema", "check", "solve", "json", "plan", "retry", "pair",
         "program", "combine", "judge")


def load(run: Path) -> list[dict]:
    with gzip.open(run / "results.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def grouped(records: list[dict]) -> dict[tuple[str, str], dict]:
    """For each data set and model, each problem's record under each condition (first sample only)."""
    groups = defaultdict(lambda: defaultdict(dict))
    for record in records:
        if record.get("sample", 0) == 0:
            problem = record.get("index", record.get("id"))
            groups[(record["dataset"], record["model"])][problem][record["condition"]] = record
    return groups


def complete(problems: dict, conditions: list[str]) -> list:
    """The problems every one of the conditions finished."""
    return sorted((problem for problem, found in problems.items() if all(c in found for c in conditions)),
                  key=str)


def rank(condition: str) -> tuple[int, str]:
    """Where a condition comes in ORDER; those ORDER does not name come last, by name."""
    return (ORDER.index(condition) if condition in ORDER else len(ORDER), condition)


def exact(gained: int, lost: int) -> float:
    """The exact McNemar test: SciPy's binomtest on the problems where two answers differ."""
    from scipy.stats import binomtest  # noqa: PLC0415

    return binomtest(gained, gained + lost).pvalue if gained + lost else 1.0


def differing(gained: int, lost: int) -> str:
    """Right only with one answer / right only with the other, and the exact McNemar p."""
    return f"{gained} / {lost}, p = {p_value(exact(gained, lost))}"


def mcnemar(problems: dict, done: list, condition: str, against: str = "none") -> tuple[int, int, float]:
    """How many problems only ``condition`` got right, how many only ``against`` did, and the exact
    two-sided McNemar p."""
    gained = sum(problems[p][condition]["correct"] and not problems[p][against]["correct"] for p in done)
    lost = sum(problems[p][against]["correct"] and not problems[p][condition]["correct"] for p in done)
    return gained, lost, exact(gained, lost)


def either(problems: dict, done: list) -> int:
    """Right answers taking ADA's value under ``solve`` when it has one and the model's own otherwise."""
    return sum(problems[p]["solve" if solved(problems[p]["solve"]) else "none"]["correct"] for p in done)


def solved(record: dict) -> bool:
    """Whether ``solve`` gave an answer: ADA solved the model's equations to a value that can be written as
    the answer (not, say, a complex number for a word problem)."""
    return record.get("given") is not None


def p_value(p: float) -> str:
    return f"{p:.2g}" if p >= 0.001 else f"{p:.0e}".replace("e-0", "e-")


def report(records: list[dict]) -> str:
    groups = grouped(records)
    lines = []
    for dataset in sorted({dataset for dataset, _ in groups}):
        models = [model for data, model in groups if data == dataset]
        conditions = [c for c in ORDER if any(c in found for model in models
                                              for found in groups[(dataset, model)].values())]
        unknown = sorted({c for model in models for found in groups[(dataset, model)].values()
                          for c in found} - set(ORDER))
        lines += [f"## {dataset}", ""]
        if unknown:
            lines += ["Not in these tables, since this script does not know them: " + ", ".join(unknown), ""]
        lines += [ "Problems right, on the problems every condition finished:", "",
                  "| Model | Problems | " + " | ".join(conditions)
                  + (" | ADA's answer, else the model's |" if {"none", "solve"} <= set(conditions) else " |"),
                  "|---|---|" + "---|" * (len(conditions) + ({"none", "solve"} <= set(conditions)))]
        for model in models:
            problems = groups[(dataset, model)]
            done = complete(problems, conditions)
            cells = [str(sum(problems[p][c]["correct"] for p in done)) for c in conditions]
            if {"none", "solve"} <= set(conditions):
                cells.append(str(either(problems, done)))
            lines.append(f"| {model} | {len(done)} | " + " | ".join(cells) + " |")
        base = "none" if "none" in conditions else conditions[0] if conditions else None
        others = [c for c in conditions if c != base]
        if others:
            heading = (f"Against {base}: right only with the condition / right only under {base}, and p:"
                       if base != "none" else
                       "Against the model alone: right only with the condition / right only alone, and p:")
            lines += ["", heading,
                      "", "| Model | " + " | ".join(others) + " |", "|---|" + "---|" * len(others)]
            totals = {c: [0, 0] for c in others}
            for model in models:
                problems = groups[(dataset, model)]
                done = complete(problems, conditions)
                cells = []
                for condition in others:
                    gained, lost, p = mcnemar(problems, done, condition, base)
                    totals[condition][0] += gained
                    totals[condition][1] += lost
                    cells.append(f"{gained} / {lost}, p = {p_value(p)}")
                lines.append(f"| {model} | " + " | ".join(cells) + " |")
            if len(models) > 1:
                cells = [f"{g} / {lost}, p = {p_value(exact(g, lost))}" for g, lost in totals.values()]
                lines.append("| All | " + " | ".join(cells) + " |")
        for condition in [c for c in ("solve", "json", "plan", "retry", "pair") if c in conditions]:
            lines += ["", f"ADA's statuses under {condition}:", ""]
            for model in models:
                problems = groups[(dataset, model)]
                statuses = Counter(found[condition].get("ada", {}).get("status", "failed")
                                   for found in problems.values() if condition in found)
                lines.append(f"- {model}: " + ", ".join(f"{status} {count}"
                                                        for status, count in statuses.most_common()))
        if "program" in conditions:
            lines += ["", "How the programs under program ended:", ""]
            for model in models:
                statuses = Counter(found["program"].get("program", {}).get("status", "failed")
                                   for found in groups[(dataset, model)].values() if "program" in found)
                lines.append(f"- {model}: " + ", ".join(f"{status} {count}"
                                                        for status, count in statuses.most_common()))
        if {"solve", "retry"} <= set(conditions):
            lines += ["", "retry against solve. retry's first message is solve's, so where its first reply "
                      "is solve's reply, the problems asked again show what asking again does; where it is "
                      "not, the two replies differ only as two replies to the same message can:", "",
                      "| Model | Asked again | First reply as under solve, asked again | Right after asking "
                      "again | First reply not as under solve | Right only under retry / only under solve |",
                      "|---|---|---|---|---|---|"]
            totals = [0] * 6
            for model in models:
                problems = groups[(dataset, model)]
                counts = retried(problems, complete(problems, conditions))
                totals = [total + count for total, count in zip(totals, counts, strict=True)]
                lines.append(f"| {model} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} "
                             f"| {counts[4]} / {counts[5]} |")
            if len(models) > 1:
                lines.append(f"| All | {totals[0]} | {totals[1]} | {totals[2]} | {totals[3]} "
                             f"| {totals[4]} / {totals[5]} |")
        if "pair" in conditions and dataset in CLOSE:
            lines += ["", "pair: asking again when ADA solved equations that leave out a number of the "
                      "question. The second answer is taken when ADA solves it and it leaves out fewer "
                      "numbers:", "",
                      "| Model | Asked again | Second answer taken | Right before | Right after |",
                      "|---|---|---|---|---|"]
            totals = [0] * 4
            for model in models:
                problems = groups[(dataset, model)]
                counts = reread(problems, complete(problems, conditions), CLOSE[dataset])
                totals = [total + count for total, count in zip(totals, counts, strict=True)]
                lines.append("| " + " | ".join([model, *map(str, counts)]) + " |")
            if len(models) > 1:
                lines.append("| All | " + " | ".join(map(str, totals)) + " |")
        if {"none", "check"} <= set(conditions):
            lines += ["", "Wrong answers alone, and how many of them have an arithmetic slip ADA finds:", ""]
            for model in models:
                problems = groups[(dataset, model)]
                wrong = [found for found in problems.values()
                         if {"none", "check"} <= set(found) and not found["none"]["correct"]]
                slips = sum(1 for found in wrong if slipped(found["check"]))
                lines.append(f"- {model}: {slips} of {len(wrong)}")
        failed = sum(1 for model in models for found in groups[(dataset, model)].values()
                     for record in found.values() if record.get("error"))
        lines += ["", f"Replies that failed or never ended: {failed}", ""]
        lines += parts(groups, dataset, models, conditions)
    lines += paired(groups)
    lines += vote_table(records)
    return "\n".join(lines)


def retried(problems: dict, done: list) -> tuple[int, ...]:
    """For ``retry`` against ``solve``: the problems asked again; of them, those whose first reply is solve's
    reply, and how many of those are right after asking again; the problems whose first reply is not solve's
    reply, and of them how many are right only under retry and only under solve."""
    again = same = fixed = other = gained = lost = 0
    for problem in done:
        first, retry = problems[problem]["solve"], problems[problem]["retry"]
        asked = "first_ada" in retry
        again += asked
        if retry.get("first_formal", retry.get("formal")) == first.get("formal"):
            same += asked
            fixed += asked and retry["correct"]
        else:
            other += 1
            gained += retry["correct"] and not first["correct"]
            lost += first["correct"] and not retry["correct"]
    return again, same, fixed, other, gained, lost


def reread(problems: dict, done: list, tolerance: Fraction) -> tuple[int, int, int, int]:
    """For ``pair``: the problems asked again because ADA solved equations that leave out a number of the
    question; of them, those whose second answer was taken, those right with the first answer and those right
    in the end."""
    asked = taken = before = after = 0
    for problem in done:
        record = problems[problem]["pair"]
        first = record.get("first_ada") or {}
        if first.get("status") != "solved" or not record.get("left_out"):
            continue
        asked += 1
        taken += not record.get("kept_first")
        before += first.get("value") is not None and _same(first["value"], record["gold"], tolerance)
        after += bool(record["correct"])
    return asked, taken, before, after


def parts(groups: dict, dataset: str, models: list[str], conditions: list[str]) -> list[str]:
    """For a data set whose problems have a subject and a level (MATH-500), the problems right under each
    condition by subject and by level, over all the models, on the problems every condition finished."""
    counts = defaultdict(Counter)
    for model in models:
        problems = groups[(dataset, model)]
        for problem in complete(problems, conditions):
            first = problems[problem][conditions[0]]
            if "subject" not in first or "level" not in first:
                return []
            for part in (first["subject"], f"Level {first['level']}"):
                counts[part]["problems"] += 1
                for condition in conditions:
                    counts[part][condition] += problems[problem][condition]["correct"]
    if not counts:
        return []
    names = sorted(name for name in counts if not name.startswith("Level ")) + sorted(
        name for name in counts if name.startswith("Level "))
    lines = ["Problems right by subject and level, over all the models:", "",
             "| Part | Problems | " + " | ".join(conditions) + " |", "|---|---|" + "---|" * len(conditions)]
    for name in names:
        lines.append(f"| {name} | {counts[name]['problems']} | "
                     + " | ".join(str(counts[name][condition]) for condition in conditions) + " |")
    return lines + [""]


#: How close two answers must be to count as the same in a vote, by data set. MATH-500's answers are LaTeX
#: that needs Math-Verify to compare, so its runs are voted by math500.py and have no vote table here.
CLOSE = {"gsm8k": Fraction(0), "gsmhard": Fraction(1, 1000)}


def voted(samples: list[dict], tolerance: Fraction) -> dict | None:
    """The record whose answer most samples give, the earliest of them when answers tie, or None when no
    sample has an answer. Answers within ``tolerance`` count as the same."""
    groups = []  # [answer, count, first record], in the order the samples came
    for record in sorted(samples, key=lambda item: item.get("sample", 0)):
        given = record.get("given")
        if given is None:
            continue
        for group in groups:
            if _same(group[0], given, tolerance):
                group[1] += 1
                break
        else:
            groups.append([given, 1, record])
    return max(groups, key=lambda group: group[1])[2] if groups else None


def _same(first: str, second: str, tolerance: Fraction) -> bool:
    try:
        return abs(Fraction(first) - Fraction(second)) <= tolerance
    except (ValueError, ZeroDivisionError):
        return first == second


def _gains(mine: dict, theirs: dict, which: int) -> tuple[int, int]:
    """On the problems both finished, how many only ``mine`` gets right and how many only ``theirs`` does, by
    first reply (``which`` 0) or by vote (1)."""
    both = mine.keys() & theirs.keys()
    return (sum(mine[p][which] and not theirs[p][which] for p in both),
            sum(theirs[p][which] and not mine[p][which] for p in both))


def vote_table(records: list[dict]) -> list[str]:
    """For each GSM8K or GSM-Hard condition asked more than once: problems right with the first reply and by
    vote of every sample, per model and pooled, on the problems every sample of the condition finished, with
    the exact McNemar test of the vote against the first reply; then each such condition against each one
    before it in ORDER, by first reply and by vote, on the problems every sample of both finished. The data
    set's answer never picks the sample: it only grades the vote."""
    asked = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record["dataset"] in CLOSE:
            problem = record.get("index", record.get("id"))
            asked[(record["dataset"], record["model"], record["condition"])][problem].append(record)
    counts = {key: max(len(samples) for samples in problems.values()) for key, problems in asked.items()}
    lines = []
    for dataset in sorted({key[0] for key, count in counts.items() if count > 1}):
        keys = [key for key, count in counts.items() if key[0] == dataset and count > 1]
        models = sorted({model for _, model, _ in keys})
        conditions = sorted({condition for _, _, condition in keys}, key=rank)
        # For each model and condition, each finished problem's first reply and vote: right or not.
        right = {}
        for key in keys:
            right[key[1:]] = {
                problem: (any(record["correct"] for record in samples if record.get("sample", 0) == 0),
                          bool((winner := voted(samples, CLOSE[dataset])) and winner["correct"]))
                for problem, samples in asked[key].items() if len(samples) == counts[key]}
        lines += [f"## {dataset}: votes", "",
                  "Problems right with the first reply, and by vote of every sample (the answer most "
                  "samples give), with the vote against the first reply: right only by vote / right only by "
                  "the first reply, and p:", "",
                  "| Model | Condition | Samples | Problems | First reply | Vote | Vote against first "
                  "reply |", "|---|---|---|---|---|---|---|"]
        rows = [(model, condition) for model in models for condition in conditions
                if (model, condition) in right]
        if len(models) > 1:
            rows += [("All", condition) for condition in conditions]
        for model, condition in rows:
            chosen = [name for name in models if model in ("All", name) and (name, condition) in right]
            found = [answers for name in chosen for answers in right[(name, condition)].values()]
            samples = ", ".join(str(n) for n in sorted({counts[(dataset, name, condition)]
                                                         for name in chosen}))
            first = sum(answers[0] for answers in found)
            vote = sum(answers[1] for answers in found)
            gained = sum(answers[1] and not answers[0] for answers in found)
            lost = sum(answers[0] and not answers[1] for answers in found)
            lines.append(f"| {model} | {condition} | {samples} | {len(found)} | {first} | {vote} "
                         f"| {differing(gained, lost)} |")
        pairs = [(later, earlier) for place, earlier in enumerate(conditions)
                 for later in conditions[place + 1:]]
        if pairs:
            lines += ["", "Each condition against each one before it: right only with the condition / "
                      "right only with the other, and p, by first reply and by vote, on the problems every "
                      "sample of both finished:", "", "| Model | Condition | Against | First reply | Vote |",
                      "|---|---|---|---|---|"]
            for condition, other in pairs:
                totals = [0, 0, 0, 0]
                for model in models:
                    if (model, condition) not in right or (model, other) not in right:
                        continue
                    found = (_gains(right[(model, condition)], right[(model, other)], 0)
                             + _gains(right[(model, condition)], right[(model, other)], 1))
                    totals = [total + count for total, count in zip(totals, found, strict=True)]
                    lines.append(f"| {model} | {condition} | {other} | {differing(*found[:2])} "
                                 f"| {differing(*found[2:])} |")
                if len(models) > 1:
                    lines.append(f"| All | {condition} | {other} | {differing(*totals[:2])} "
                                 f"| {differing(*totals[2:])} |")
        lines.append("")
    return lines


def slipped(check: dict) -> bool:
    """Whether ADA found a wrong equation in the model's own answer."""
    return any(not item["correct"] for item in check.get("checks", []))


def paired(groups: dict) -> list[str]:
    """GSM-Hard's problem i is GSM8K's problem i with larger numbers: of the GSM8K problems a model gets right
    alone, how many of their GSM-Hard versions it gets right under each condition. A run that did not ask the
    model alone has no such table."""
    models = [model for dataset, model in groups if dataset == "gsm8k" and ("gsmhard", model) in groups
              and any("none" in found for found in groups[(dataset, model)].values())]
    if not models:
        return []
    conditions = [c for c in ORDER if all(c in found for model in models
                                          for found in groups[("gsmhard", model)].values())]
    lines = ["## The same problems with larger numbers", "",
             "Of the GSM8K problems a model gets right alone, how many GSM-Hard versions it gets right:",
             "", "| Model | GSM8K right alone | " + " | ".join(f"GSM-Hard, {c}" for c in conditions) + " |",
             "|---|---|" + "---|" * len(conditions)]
    for model in models:
        small, large = groups[("gsm8k", model)], groups[("gsmhard", model)]
        known = sorted((p for p, found in small.items() if "none" in found and found["none"]["correct"]
                        and p in large), key=str)
        cells = [str(sum(large[p][c]["correct"] for p in known if c in large[p])) for c in conditions]
        lines.append(f"| {model} | {len(known)} | " + " | ".join(cells) + " |")
    return lines + [""]


def main() -> int:
    print(report(load(Path(sys.argv[1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
