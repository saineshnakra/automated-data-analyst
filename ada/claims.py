"""Check a number someone states about the data: verified, or unverified with the reason.

    import ada
    check = ada.check_claim("flights.csv", "The average departure delay was 12.64 minutes.")
    check.status        # "verified" or "unverified"
    check.reason        # None, or why it is unverified
    check.to_dict()     # the claim, how it was read, the result, the calculation and the rows behind it

A claim is a sentence with one number in it. ADA takes the number out, reads
the rest as a question (see ``Analysis.ask``), computes the answer and
compares. The number stands for the interval its writing implies: "12.6" is
anything from 12.55 to 12.65, "about 13" is 12 to 14, "about 100" is 90 to
110, "more than 30%" is above 30. A number with a different scale or unit is
not taken to agree, except that a unit named in the column ("body_mass_g") is
converted to the claim's ("5.08 kg").

``status`` is "verified" only when the computed answer falls in that interval
and no check on the analysis failed. Otherwise it is "unverified", and
``reason`` says why:

    missing_context        the claim leaves out something the check needs: a number, a
                           column, a group, or the unit that decides the scale
    ambiguous_intent       the words fit more than one calculation and their answers differ
    inaccessible_data      the data could not be read
    unsupported_analysis   ADA has no way to compute what the claim describes
    conflicting_evidence   the claim's unit measures something other than the unit in the
                           column's name: minutes for a column in mm
    failed_check           the computed answer is outside the claimed interval, or a check
                           on the analysis failed

Unverified does not mean false: only ``failed_check`` with ``agreement ==
"mismatch"`` says the data give a different number. A verified claim is
verified as read: ``reading``, ``assumptions`` and ``warnings`` say what was
checked, and nothing beyond it.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from .file_io import SUPPORTED_SUFFIXES as READABLE
from .nlq import QueryAnswer
from .sources import Source

#: Version of the ``ClaimCheck.to_dict`` layout. Bumped when a key changes meaning.
CLAIM_SCHEMA_VERSION = 1

REASONS = (
    "missing_context",
    "ambiguous_intent",
    "inaccessible_data",
    "unsupported_analysis",
    "conflicting_evidence",
    "failed_check",
)

_CURRENCY = r"(?:R\$|US\$|A\$|C\$|\$|€|£|¥|₹)"
_DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|\.\d+"
NUMBER_PATTERN = re.compile(
    rf"(?P<currency>{_CURRENCY})?\s?(?P<sign>[-−])?(?P<number>{_DIGITS})"
    r"(?:\s?(?P<scale>thousand|million|billion|trillion|bn|mn)\b|(?P<suffix>[kKM])\b)?"
    r"(?:\s?(?P<percent>%|percent\b|per\s?cent\b))?",
    re.IGNORECASE,
)
SCALES = {
    "thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "trillion": 1e12,
}
# Words before a number that bound or widen it, longest first so "just over" wins over "over".
QUALIFIERS = (
    ("no less than", "at least"), ("not less than", "at least"), ("at least", "at least"),
    ("no more than", "at most"), ("not more than", "at most"), ("at most", "at most"), ("up to", "at most"),
    ("in excess of", "more than"), ("more than", "more than"), ("greater than", "more than"),
    ("higher than", "more than"), ("exceeding", "more than"), ("above", "more than"),
    ("just over", "just over"), ("just under", "just under"), ("over", "more than"),
    ("fewer than", "less than"), ("less than", "less than"), ("lower than", "less than"),
    ("below", "less than"), ("under", "less than"),
    ("close to", "about"), ("approximately", "about"), ("approx.", "about"), ("approx", "about"),
    ("around", "about"), ("roughly", "about"), ("about", "about"), ("some", "about"), ("circa", "about"),
    ("~", "about"), ("≈", "about"),
    ("nearly", "nearly"), ("almost", "nearly"), ("exactly", "exactly"), ("precisely", "exactly"),
)
# Words right before the number that link it to what it describes: "is 12.6", "averaged 12.6".
PREDICATE_CUES = frozenset({
    "is", "was", "were", "are", "be", "been", "equals", "equalled", "equaled", "averaged", "averages",
    "averaging", "totalled", "totaled", "totals", "totalling", "totaling", "reached", "hit", "by",
    "at", "of", "stood", "came", "amounted", "numbered", "weighed", "weighs", "weigh", "measured",
    "rose", "fell", "grew", "dropped", "increased", "decreased", "paid", "made", "had", "has", "have",
})
# Cue words that say how rows combine, put back into the question as the aggregation.
AGGREGATION_CUES = {
    "averaged": "average", "averages": "average", "averaging": "average",
    "totalled": "total", "totaled": "total", "totals": "total", "totalling": "total", "totaling": "total",
}
# Words that end a unit after the number ("12.6 minutes in 2013"): grammar, not part of the unit.
UNIT_STOPS = frozenset({
    "in", "on", "for", "of", "per", "from", "by", "at", "during", "than", "and", "or", "with", "among",
    "across", "when", "where", "while", "since", "between", "after", "before", "to", "a", "an", "the",
    "each", "every", "over", "under", "as", "if", "is", "was", "were", "are",
})
YEAR_RANGE = (1500, 2100)
# Verbs that stand for a column when a claim names none: "weigh 4000 g" is a mass. "measure" takes
# its kind from the claim's unit.
UNIT_VERBS = {
    "weigh": "mass", "weighs": "mass", "weighed": "mass", "weighing": "mass",
    "lasted": "time", "lasts": "time", "measure": None, "measures": None, "measured": None,
}
# Units a claim or a column name can state plainly, as (dimension, size in the dimension's base unit).
# Short forms that also mean something else ("m" for million, "t", "tons") are left out on purpose.
UNITS = {
    "mg": ("mass", 1e-3), "milligram": ("mass", 1e-3), "milligrams": ("mass", 1e-3),
    "g": ("mass", 1.0), "gram": ("mass", 1.0), "grams": ("mass", 1.0),
    "kg": ("mass", 1e3), "kilogram": ("mass", 1e3), "kilograms": ("mass", 1e3),
    "mm": ("length", 1e-3), "millimetre": ("length", 1e-3), "millimetres": ("length", 1e-3),
    "millimeter": ("length", 1e-3), "millimeters": ("length", 1e-3),
    "cm": ("length", 1e-2), "centimetre": ("length", 1e-2), "centimetres": ("length", 1e-2),
    "centimeter": ("length", 1e-2), "centimeters": ("length", 1e-2),
    "metre": ("length", 1.0), "metres": ("length", 1.0), "meter": ("length", 1.0), "meters": ("length", 1.0),
    "km": ("length", 1e3), "kilometre": ("length", 1e3), "kilometres": ("length", 1e3),
    "kilometer": ("length", 1e3), "kilometers": ("length", 1e3),
    "sec": ("time", 1.0), "secs": ("time", 1.0), "second": ("time", 1.0), "seconds": ("time", 1.0),
    "min": ("time", 60.0), "mins": ("time", 60.0), "minute": ("time", 60.0), "minutes": ("time", 60.0),
    "hr": ("time", 3600.0), "hrs": ("time", 3600.0), "hour": ("time", 3600.0), "hours": ("time", 3600.0),
    "day": ("time", 86400.0), "days": ("time", 86400.0),
}


@dataclass(frozen=True)
class ClaimedNumber:
    """The number a claim states, and the interval of values its writing stands for.

    ``value`` is in the claim's terms: 32 for "32%", 16008872 for "R$16,008,872",
    3.75e10 for "37.5 billion". ``low`` and ``high`` bound the interval;
    ``low_open`` and ``high_open`` say whether a bound itself is excluded.
    """

    text: str
    value: float
    low: float
    high: float
    low_open: bool = False
    high_open: bool = False
    qualifier: str | None = None
    percent: bool = False
    scale: float = 1.0
    scale_word: str | None = None
    unit: str | None = None
    decimals: int = 0

    def holds(self, number: float | None) -> bool:
        """Whether ``number`` lies in the claimed interval."""
        if number is None or not math.isfinite(number):
            return False
        above = number > self.low if self.low_open else number >= self.low - _slack(self.low)
        below = number < self.high if self.high_open else number <= self.high + _slack(self.high)
        return above and below

    def interval(self) -> str:
        """The interval in words: "12.635 to 12.645", "more than 30"."""
        if math.isinf(self.low) and math.isinf(self.high):
            return "any value"
        if math.isinf(self.high):
            return f"{'more than' if self.low_open else 'at least'} {_plain(self.low)}"
        if math.isinf(self.low):
            return f"{'less than' if self.high_open else 'at most'} {_plain(self.high)}"
        if self.low == self.high:
            return f"exactly {_plain(self.low)}"
        return f"{_plain(self.low)} to {_plain(self.high)}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "value": self.value,
            "interval": [_json_number(self.low), _json_number(self.high)],
            "interval_text": self.interval(),
            "qualifier": self.qualifier,
            "percent": self.percent,
            "scale": self.scale_word,
            "unit": self.unit,
        }


def _slack(bound: float) -> float:
    """Room for floating-point error at an interval's edge, far below any written precision."""
    return abs(bound) * 1e-12 + 1e-12


def _json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _plain(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.10g}" if abs(value) >= 1 else f"{value:.10g}"


@dataclass(frozen=True)
class Check:
    """One check on a claim: its name, whether it passed and what it found."""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class Alternative:
    """Another reading of the claim's words, with the number it gives."""

    reading: str
    question: str
    value: float | None
    agrees: bool
    calculation: str

    def to_dict(self) -> dict:
        return {
            "reading": self.reading, "question": self.question, "value": self.value,
            "agrees": self.agrees, "calculation": self.calculation,
        }


@dataclass(frozen=True)
class ClaimCheck:
    """The result of checking one claim. See the module docstring for ``status`` and ``reason``."""

    claim: str
    status: Literal["verified", "unverified"]
    reason: str | None
    message: str
    claimed: ClaimedNumber | None = None
    question: str | None = None
    answer: QueryAnswer | None = None
    #: The computed number that was compared with the claim, in the claim's terms.
    compared: float | None = None
    agreement: Literal["match", "mismatch", "not_compared"] = "not_compared"
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    checks: tuple[Check, ...] = ()
    alternatives: tuple[Alternative, ...] = ()
    #: The smallest question that would let the check go ahead, for the person who made the claim.
    clarification: str | None = None
    source: Source | None = None

    @property
    def verified(self) -> bool:
        return self.status == "verified"

    def to_dict(self) -> dict:
        """The check as JSON-safe data, in the layout ``CLAIM_SCHEMA_VERSION`` names."""
        answer = self.answer.to_dict() if self.answer is not None else None
        named = None if self.source is None else self.source.to_dict()
        evidence = answer["evidence"] if answer else {"source": named}
        return {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "claim": self.claim,
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
            "claimed": None if self.claimed is None else self.claimed.to_dict(),
            "question": self.question,
            "agreement": self.agreement,
            "result": {
                "value": self.compared,
                **({key: answer["result"][key] for key in ("values", "figures")} if answer else {}),
            },
            "reading": answer["reading"] if answer else None,
            "calculation": answer["calculation"] if answer else None,
            "evidence": evidence,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
            "checks": [check.to_dict() for check in self.checks],
            "alternatives": [alternative.to_dict() for alternative in self.alternatives],
            "clarification": self.clarification,
        }

    def __str__(self) -> str:
        lines = [f"{self.status}: {self.claim}", f"  {self.message}"]
        if self.answer is not None:
            lines.append(f"  calc: {self.answer.calculation}")
        for label, items in (("assumes", self.assumptions), ("warning", self.warnings)):
            lines.extend(f"  {label}: {item}" for item in items)
        if self.clarification:
            lines.append(f"  to check it: {self.clarification}")
        return "\n".join(lines)


def parse_number(match: re.Match) -> tuple[float, int, str | None, float, bool]:
    """The value, its decimals, its scale word, the scale and whether it is a percent, for one match."""
    digits = match.group("number").replace(",", "")
    value = float(digits)
    if match.group("sign"):
        value = -value
    mantissa = digits.lower().split("e")[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    if "e" in digits.lower():
        decimals -= int(digits.lower().split("e")[1])
    word = (match.group("scale") or match.group("suffix") or "").lower() or None
    scale = SCALES.get(word, 1.0) if word else 1.0
    return value, decimals, word, scale, bool(match.group("percent"))


def _significant_unit(value: float, decimals: int) -> float:
    """The place of the last non-zero digit: 1 for 13, 1000 for 37,000, 0.1 for 10.5."""
    if decimals > 0 or value == 0:
        return 10.0 ** -decimals
    magnitude = abs(value)
    unit = 1.0
    while magnitude >= unit * 10 and (magnitude / (unit * 10)) == int(magnitude / (unit * 10)):
        unit *= 10
    return unit


def _approximate_width(value: float, decimals: int) -> float:
    """How far "about" reaches either side: one significant unit, at most a tenth of the number,
    and never less than the number's own rounding. "about 13" is 12 to 14, "about 100" is 90 to 110.
    """
    rounding = 10.0 ** -decimals / 2
    return max(rounding, min(_significant_unit(value, decimals), abs(value) / 10))


def claimed_number(text: str, match: re.Match, qualifier: str | None, unit: str | None) -> ClaimedNumber:
    """The interval a number stands for, from its decimals and the qualifier before it.

    The bounds are worked out in decimal arithmetic, so "12.64" stands for 12.635
    to 12.645 exactly rather than to 12.645000000000001.
    """
    value, decimals, word, scale, percent = parse_number(match)
    exact = Decimal(match.group("number").replace(",", "")) * (-1 if match.group("sign") else 1)
    half = Decimal(1).scaleb(-decimals) / 2
    widened = _approximate_width(value, decimals)
    low, high, low_open, high_open = float(exact - half), float(exact + half), False, False
    if qualifier == "about":
        low, high = value - widened, value + widened
    elif qualifier == "nearly":
        low, high = value - widened, value
    elif qualifier == "just over":
        low, high, low_open = value, value + widened, True
    elif qualifier == "just under":
        low, high, high_open = value - widened, value, True
    elif qualifier == "more than":
        low, high, low_open = value, math.inf, True
    elif qualifier == "at least":
        low, high = value, math.inf
    elif qualifier == "less than":
        low, high, high_open = -math.inf, value, True
    elif qualifier == "at most":
        low, high = -math.inf, value
    elif qualifier == "exactly":
        low = high = value
    return ClaimedNumber(
        text=text.strip(), value=value * scale, low=low * scale, high=high * scale,
        low_open=low_open, high_open=high_open, qualifier=qualifier, percent=percent,
        scale=scale, scale_word=word, unit=unit, decimals=decimals,
    )


@dataclass(frozen=True)
class _Candidate:
    match: re.Match
    qualifier: str | None
    qualifier_start: int
    cue: str | None
    cue_start: int
    year_like: bool
    filter_like: bool
    ordinal: bool


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _column_words(columns: list[str]) -> set[str]:
    words: set[str] = set()
    for column in columns:
        words.update(word for word in re.findall(r"[a-z]+", str(column).lower()) if len(word) >= 2)
    return words


def _candidates(claim: str, columns: list[str]) -> list[_Candidate]:
    """Every number in the claim, with the qualifier and cue before it, and whether it is part of the
    question instead: a year, a value after a column's name, or an ordinal ("25th percentile").
    """
    found = []
    column_words = _column_words(columns)
    for match in NUMBER_PATTERN.finditer(claim):
        start = match.start()
        before = claim[:start].rstrip()
        lowered = before.lower()
        qualifier, qualifier_start = None, start
        for phrase, kind in QUALIFIERS:
            if not lowered.endswith(phrase):
                continue
            alone = len(lowered) == len(phrase) or not lowered[-len(phrase) - 1].isalnum()
            if alone or not phrase[0].isalnum():
                qualifier, qualifier_start = kind, len(before) - len(phrase)
                before = before[: -len(phrase)].rstrip()
                lowered = before.lower()
                break
        previous = re.search(r"([A-Za-z]+)\W*$", before)
        cue = previous.group(1).lower() if previous and previous.group(1).lower() in PREDICATE_CUES else None
        cue_start = previous.start(1) if cue and previous else qualifier_start
        marked = any(match.group(name) for name in ("currency", "scale", "suffix", "percent"))
        plain = not marked and "," not in match.group("number") and "." not in match.group("number")
        number = float(match.group("number").replace(",", ""))
        year_like = plain and YEAR_RANGE[0] <= number <= YEAR_RANGE[1] and number == int(number)
        filter_like = bool(previous) and previous.group(1).lower() in column_words and cue is None
        ordinal = plain and re.match(r"(?:st|nd|rd|th)\b", claim[match.end():], re.IGNORECASE) is not None
        found.append(
            _Candidate(match, qualifier, qualifier_start, cue, cue_start, year_like, filter_like, ordinal)
        )
    return found


def _unit_after(claim: str, end: int, columns: list[str], row_noun: str | None) -> tuple[str | None, int]:
    """The unit words right after a number ("minutes", "million tonnes"), and where they end.

    Words that name a column or the rows are part of the question, not a unit,
    unless the first word is a unit of its own ("43.9 mm" for bill_length_mm).
    """
    column_words = _column_words(columns) | ({row_noun} if row_noun else set())
    words = []
    position = end
    for match in re.finditer(r"\s*([A-Za-z][A-Za-z.]*)", claim[end:]):
        if match.start() != position - end:
            break
        word = match.group(1).rstrip(".")
        known_unit = word.lower() in UNITS and not words
        if word.lower() in UNIT_STOPS or (word.lower() in column_words and not known_unit) or len(words) == 3:
            break
        words.append(word)
        position = end + match.end()
        if match.group(1).endswith("."):
            # A full stop ends the sentence, not the unit: "5076 g."
            position -= len(match.group(1)) - len(word)
            break
    return (" ".join(words) or None), position


def read_claim(
    claim: str, columns: list[str], row_noun: str | None = None
) -> tuple[str | None, ClaimedNumber | None, str]:
    """Split a claim into the question it asks and the number it states.

    Returns (question, number, problem). ``problem`` is "" when the split
    worked, or why it didn't: no number, or several numbers that each look
    like the one being claimed.
    """
    candidates = _candidates(claim, columns)
    if not candidates:
        return None, None, "The claim states no number to check."
    plain = [item for item in candidates if not (item.year_like or item.filter_like or item.ordinal)]
    usable = plain or [item for item in candidates if not (item.year_like or item.ordinal)] or candidates
    cued = [item for item in usable if item.cue or item.qualifier]
    chosen = cued[0] if cued else usable[-1]
    # Any other number left in the question would be read as part of it ("... and 3700 g" as a filter).
    others = cued if len(cued) > 1 else plain
    rivals = [item for item in others if item.match.group(0) != chosen.match.group(0)]
    if rivals:
        listed = ", ".join(dict.fromkeys(item.match.group(0).strip() for item in [chosen, *rivals]))
        return None, None, (
            f"The claim states more than one number that could be the one claimed ({listed}); "
            "check them one at a time."
        )
    unit, unit_end = _unit_after(claim, chosen.match.end(), columns, row_noun)
    number = claimed_number(claim[chosen.qualifier_start:unit_end], chosen.match, chosen.qualifier, unit)
    # "There are 344 penguins", "152 penguins are Adelie": a number before the rows' noun counts them.
    lead = claim[:chosen.qualifier_start].strip()
    following = re.match(r"\s*([A-Za-z]+)", claim[chosen.match.end():])
    counted = following is not None and unit is None and (
        following.group(1).lower() == row_noun or following.group(1).lower().endswith("s")
    )
    if counted and re.fullmatch(r"(?:there\s+(?:are|were|is|was))?", lead, flags=re.IGNORECASE):
        rest = re.sub(r"\s+", " ", claim[chosen.match.end():]).strip(" .,;:!?")
        return f"how many {rest}", number, ""
    start = chosen.cue_start if chosen.cue in AGGREGATION_CUES or chosen.cue in {
        "is", "was", "were", "are", "be", "been", "equals", "equalled", "equaled", "stood", "amounted",
        "came", "numbered", "at", "by", "of",
    } else chosen.qualifier_start
    question = f"{claim[:start]} {claim[unit_end:]}"
    if chosen.cue in AGGREGATION_CUES:
        question = f"{AGGREGATION_CUES[chosen.cue]} {question}"
    question = re.sub(r"\bon average\b", "", question, flags=re.IGNORECASE)
    if re.search(r"\bon average\b", claim, re.IGNORECASE):
        question = f"average {question}"
    question = re.sub(r"\s+", " ", question).strip(" .,;:!?")
    question = re.sub(r"^(?:the|a|an)\s+", "", question, flags=re.IGNORECASE)
    question = re.sub(r"\s+(?:is|was|were|are|of|at|by)$", "", question, flags=re.IGNORECASE)
    if not question:
        return None, number, "The claim says nothing about what the number measures."
    return question, number, ""


def _abbreviates(word: str, part: str) -> bool:
    """Whether a question word and a column-name word are the same word, or one shortens the other."""
    if word == part:
        return True
    return (len(part) >= 3 and word.startswith(part)) or (len(word) >= 3 and part.startswith(word))


def _substitute(question: str, column: str) -> str | None:
    """The question with its longest run of words that all match ``column``'s name replaced by the name."""
    parts = re.findall(r"[a-z]+", str(column).lower())
    if not parts:
        return None
    tokens = list(re.finditer(r"[A-Za-z]+", question))
    best: tuple[int, int] | None = None
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and any(_abbreviates(tokens[j].group(0).lower(), part) for part in parts):
            j += 1
        if j > i and (best is None or j - i > best[1] - best[0]):
            best = (i, j)
        i = max(j, i + 1)
    if best is None:
        return None
    start, end = tokens[best[0]].start(), tokens[best[1] - 1].end()
    return f"{question[:start]}{column}{question[end:]}"


def _counted(reason: str, count: int) -> str:
    """"2 rows with x missing", or "1 row with x missing" for one."""
    if count == 1:
        reason = re.sub(r"^(rows|values|repeats)\b", lambda m: m.group(1)[:-1], reason)
        reason = reason.replace("exact repeats", "exact repeat")
    return f"{count:,} {reason}"


def _named(question: str, column: str) -> bool:
    """Whether the question names ``column``: one of its words, or a word that shortens or lengthens one."""
    parts = re.findall(r"[a-z]+", str(column).lower())
    words = re.findall(r"[a-z]+", question.lower())
    if str(column).lower() in question.lower():
        return True
    return any(_abbreviates(word, part) for word in words for part in parts if len(part) >= 2)


def _candidate_columns(message: str, columns: list[str]) -> list[str]:
    """Columns named in a refusal's "Did you mean A, B or C?"."""
    asked = re.search(r"Did you mean (.+?)\?", message)
    if not asked:
        return []
    named = re.split(r",\s*|\s+or\s+", asked.group(1))
    known = {str(column): column for column in columns}
    return [known[name.strip()] for name in named if name.strip() in known]


def _comparable(answer: QueryAnswer, number: ClaimedNumber) -> tuple[float | None, list[str]]:
    """The computed number to compare with the claim, in the claim's terms, and how it was taken."""
    figures = dict(answer.figures)
    notes: list[str] = []
    if number.percent:
        if "share_percent" in figures:
            notes.append("compared with the share of rows, as a percentage")
            return figures["share_percent"], notes
        if "change_percent" in figures:
            return figures["change_percent"], notes
        if answer.value is not None and abs(answer.value) <= 1 and not number.holds(answer.value):
            notes.append(f"compared as a percentage: {answer.value:.6g} is {answer.value * 100:.6g}%")
            return answer.value * 100, notes
        return answer.value, notes
    if answer.value is not None:
        return answer.value, notes
    if len(answer.values) == 1:
        return answer.values[0][1], notes
    return None, notes


def _scale_note(number: ClaimedNumber, computed: float, column: str | None) -> tuple[str, str] | None:
    """When the claim agrees only if a scale word is read as part of the unit, or the data are in other units.

    Returns (kind, sentence): "unit" when "37,528 million tonnes" agrees with
    37,528 read as the unit's own count; "scale" when it would agree only if the
    column were recorded in thousands, millions or billions.
    """
    name = column or "the value"
    if number.scale_word and number.scale != 1 and number.holds(computed * number.scale):
        unit = " ".join(part for part in (number.scale_word, number.unit) if part)
        return "unit", (
            f'Read "{unit}" as the unit {name} is recorded in, so {_plain(number.value / number.scale)} '
            f"is compared with {name} as recorded."
        )
    for factor, word in ((1e3, "thousands"), (1e6, "millions"), (1e9, "billions")):
        readings = ((computed * factor, f"in {word}"), (computed / factor, f"in units {word} smaller"))
        for scaled, recorded in readings:
            if number.holds(scaled):
                return "scale", (
                    f"The claim agrees with {name} only if {name} is recorded {recorded}, and the data "
                    "do not say what unit it is in."
                )
    return None


def column_unit(column: str | None) -> str | None:
    """The unit a column's name ends in: "g" for body_mass_g, "kg" for "Weight (kg)", None for co2.

    A name that is only a unit word ("Days") is a measure of its own, not an annotation.
    """
    if not column:
        return None
    bracketed = re.search(r"[(\[]\s*([A-Za-z]+)\s*[)\]]\s*$", str(column))
    if bracketed and bracketed.group(1).lower() in UNITS:
        return bracketed.group(1).lower()
    parts = [part for part in re.split(r"[^A-Za-z]+", str(column)) if part]
    if len(parts) >= 2 and parts[-1].lower() in UNITS:
        return parts[-1].lower()
    return None


def _claim_unit(number: ClaimedNumber) -> str | None:
    if number.percent or number.scale_word or not number.unit:
        return None
    first = number.unit.split()[0].lower()
    return first if first in UNITS else None


def _unit_columns(analysis: Any, asked: str, number: ClaimedNumber) -> list[tuple[str, str]]:
    """Columns whose unit is the kind of quantity the claim states, with the question naming each.

    The kind comes from the claim's unit ("4000 g" is a mass) or from a verb
    ("weigh"). The verb, if any, is replaced by the column's name; otherwise
    the name is added to the question.
    """
    verbs = [word for word in re.findall(r"[A-Za-z]+", asked) if word.lower() in UNIT_VERBS]
    claimed = _claim_unit(number)
    kind = UNITS[claimed][0] if claimed else UNIT_VERBS.get(verbs[0].lower()) if verbs else None
    if kind is None:
        return []
    found = []
    for column in analysis.roles.numeric:
        unit = column_unit(str(column))
        if unit and UNITS[unit][0] == kind:
            rewritten = (
                re.sub(rf"\b{re.escape(verbs[0])}\b", str(column), asked, count=1) if verbs
                else f"{asked} {column}"
            )
            found.append((str(column), rewritten))
    return found


def _repeated_units(analysis: Any, answer: QueryAnswer) -> Check | None:
    """Whether the answer treats rows as independent when a unit (a subject, a site) has several rows."""
    roles = analysis.roles
    unit = roles.identifier
    plan = answer.plan
    if not unit or unit not in analysis.frame.columns or plan.intent not in (
        "aggregate", "compare", "correlation", "correlates", "breakdown", "rank", "trend",
    ):
        return None
    counts = analysis.frame[unit].dropna().value_counts()
    if counts.empty or counts.max() < 2:
        return None
    per = plan.per or ()
    if unit in per:
        return Check("repeated_units", True, f"{unit} has several rows each; the answer combines them first.")
    units, rows = len(counts), int(counts.sum())
    if plan.intent in ("compare", "correlation", "correlates"):
        detail = (
            f"{rows:,} rows come from {units:,} {unit} values, so they are not independent; a test or "
            "correlation over rows overstates how much evidence there is."
        )
        return Check("repeated_units", False, detail)
    detail = (
        f"{rows:,} rows come from {units:,} {unit} values; this figure weights each {unit} by its number "
        "of rows."
    )
    return Check("repeated_units", True, detail)


def _source_of(data: Any) -> tuple[Any | None, Source | None, str | None]:
    """An Analysis for ``data``, its Source, and why it could not be read."""
    from .api import AdaError, Analysis, analyze  # noqa: PLC0415

    if isinstance(data, Analysis):
        return data, data.source, None
    if isinstance(data, str | Path) and Path(data).is_dir():
        files = sorted(item.name for item in Path(data).iterdir() if item.suffix.lower() in READABLE)
        listed = ", ".join(files[:12]) + (" and more" if len(files) > 12 else "")
        return None, Source(name=str(data), path=str(data)), (
            f"{data} is a folder. Name the file the claim is about" + (f": {listed}." if files else ".")
        )
    try:
        return analyze(data, row_limit=sys.maxsize), None, None
    except AdaError as error:
        name = str(data) if isinstance(data, str | Path) else type(data).__name__
        return None, Source(name=name, path=name if isinstance(data, str | Path) else None), str(error)


def check_claim(
    data: Any,
    claim: str | None = None,
    *,
    question: str | None = None,
    value: str | float | None = None,
    sheet: str | None = None,
) -> ClaimCheck:
    """Check a claim about ``data`` (a path, bytes, a DataFrame or an ``Analysis``).

    Give the claim as a sentence (``claim``), or as ``question`` and ``value``
    when the reading is already known: ``question="average dep_delay",
    value="12.64"``. A string ``value`` keeps its decimals, which set the
    interval; a float is taken at its shortest written form. Every row of the
    file is read: no row limit applies to a check.
    """
    text = claim if claim is not None else f"{question} is {value}"
    if sheet is not None and isinstance(data, str | Path):
        from .api import analyze  # noqa: PLC0415

        try:
            data = analyze(data, sheet=sheet, row_limit=sys.maxsize)
        except Exception as error:  # noqa: BLE001 - reported as unreadable
            return ClaimCheck(text, "unverified", "inaccessible_data", f"The data could not be read: {error}")
    if isinstance(data, str | Path) and Path(data).is_dir():
        return _check_folder(Path(data), text, claim, question=question, value=value)
    analysis, source, problem = _source_of(data)
    if analysis is None:
        reason = "missing_context" if problem and "is a folder" in problem else "inaccessible_data"
        message = problem if reason == "missing_context" else f"The data could not be read: {problem}"
        return ClaimCheck(text, "unverified", reason, message, source=source,
                          clarification=problem if reason == "missing_context" else None)
    source = analysis.source
    columns = [str(column) for column in analysis.frame.columns]
    if question is not None and value is not None:
        written = value if isinstance(value, str) else repr(float(value)).removesuffix(".0")
        match = NUMBER_PATTERN.search(str(written))
        if not match:
            problem = f"{written!r} is not a number."
            return ClaimCheck(text, "unverified", "missing_context", problem, source=source)
        number = claimed_number(str(written), match, None, None)
        asked = question
    else:
        asked, number, problem = read_claim(text, columns, analysis.roles.row_noun)
        if problem:
            reason = "unsupported_analysis" if "more than one number" in problem else "missing_context"
            return ClaimCheck(
                text, "unverified", reason, problem, claimed=number, question=asked, source=source,
                clarification="Restate the claim with one number, what it measures and over what rows."
                if reason == "missing_context" else "Send each number as its own claim.",
            )
    assert number is not None and asked is not None
    return _check(analysis, text, asked, number, source)


def _check_folder(
    folder: Path, text: str, claim: str | None, *, question: str | None, value: str | float | None
) -> ClaimCheck:
    """Check a claim about a folder against each of its tables, and say which one fits.

    The claim is checked against every readable file. When exactly one gives a
    number for it, that file's check is the result, and its assumptions say the
    file was chosen this way. When several do and they disagree, the claim is
    ambiguous. When none does, the claim may need two files joined, and the
    links a scan proposes are named; ADA does not join files itself.
    """
    files = sorted(
        item for item in folder.iterdir()
        if item.is_file() and not item.name.startswith(".") and item.suffix.lower() in READABLE
    )
    listed = ", ".join(item.name for item in files[:12]) + (" and more" if len(files) > 12 else "")
    folder_source = Source(name=str(folder), path=str(folder))
    if not files:
        return ClaimCheck(text, "unverified", "inaccessible_data", f"{folder} holds no table ADA can read.",
                          source=folder_source)
    fitted: list[tuple[Path, ClaimCheck]] = []
    for item in files:
        check = check_claim(item, claim, question=question, value=value)
        if check.answer is not None and check.compared is not None:
            fitted.append((item, check))
    if len(fitted) == 1:
        item, check = fitted[0]
        chosen = f"Checked against {item.name}, the only file in {folder.name} whose columns fit the claim."
        return replace(check, assumptions=(chosen, *check.assumptions))
    if fitted:
        values = {round(check.compared, 12) for _, check in fitted}
        if len(values) == 1 and all(check.verified for _, check in fitted):
            item, check = fitted[0]
            names = [found.name for found, _ in fitted]
            each = (
                f"{names[0]} and {names[1]} both give" if len(names) == 2 else f"{', '.join(names)} all give"
            )
            same = f"{each} {_plain(check.compared)}."
            return replace(check, assumptions=(same, *check.assumptions))
        alternatives = tuple(
            Alternative(item.name, check.question or "", check.compared, check.agreement == "match",
                        check.answer.calculation if check.answer else "")
            for item, check in fitted
        )
        shown = "; ".join(f"{item.reading} gives {_plain(item.value)}" for item in alternatives
                          if item.value is not None)
        names = [item.reading for item in alternatives]
        return ClaimCheck(
            text, "unverified", "ambiguous_intent",
            f"The claim fits more than one file in {folder.name}, and they give different answers: {shown}.",
            claimed=fitted[0][1].claimed, question=fitted[0][1].question, alternatives=alternatives,
            source=folder_source,
            clarification=f"Which file is the claim about: {', '.join(names[:-1])} or {names[-1]}?",
        )
    from .folder import scan  # noqa: PLC0415

    links = [link for link in scan(folder).links if link.kind in ("many-to-one", "one-to-one")]
    joins = "; ".join(f"{link.label('from')} to {link.label('to')}" for link in links[:6])
    message = f"No single file in {folder.name} holds what the claim names ({listed})."
    if joins:
        message += (
            f" It may need two files joined; the keys that join them are {joins}. ADA does not join files."
        )
    return ClaimCheck(
        text, "unverified", "missing_context" if joins else "unsupported_analysis", message,
        source=folder_source,
        clarification="Which file is the claim about, or which files hold the numbers it combines?",
    )


def _check(analysis: Any, text: str, asked: str, number: ClaimedNumber, source: Source | None) -> ClaimCheck:
    from .api import Refusal  # noqa: PLC0415

    answer = analysis.ask(asked)
    alternatives: list[Alternative] = []
    assumptions: list[str] = []
    if isinstance(answer, Refusal):
        readings = []
        columns = _candidate_columns(answer.message, list(analysis.frame.columns))
        named = [(column, _substitute(asked, column)) for column in columns]
        by_unit = not named
        for column, rewritten in named or _unit_columns(analysis, asked, number):
            if not rewritten:
                continue
            other = analysis.ask(rewritten)
            if isinstance(other, Refusal):
                continue
            computed, _ = _comparable(other, number)
            readings.append((column, rewritten, other, computed))
        if len(readings) == 1:
            column, asked, answer, _ = readings[0]
            assumptions.append(
                f"Read the claim as being about {column}, the only column recorded in a unit of that kind."
                if by_unit else f"Read the claim's words as {column}, the only column the question fits."
            )
        elif readings:
            alternatives = [
                Alternative(column, rewritten, computed, number.holds(computed), other.calculation)
                for column, rewritten, other, computed in readings
            ]
            shown = "; ".join(
                f"{item.reading} gives {_plain(item.value) if item.value is not None else 'no single number'}"
                for item in alternatives
            )
            fits = "The claim's unit fits" if by_unit else "The claim's words fit"
            differ = "they give different answers" if len({a.agrees for a in alternatives}) > 1 else (
                "each agrees with the claim" if alternatives[0].agrees else "none agrees with the claim"
            )
            names = [a.reading for a in alternatives]
            listed = f"{', '.join(names[:-1])} or {names[-1]}"
            message = f"{fits} more than one column, and {differ}: {shown}."
            return ClaimCheck(
                text, "unverified", "ambiguous_intent", message,
                claimed=number, question=asked, alternatives=tuple(alternatives), source=source,
                clarification=f"Which column is meant: {listed}?",
            )
        else:
            reason = "unsupported_analysis" if answer.code == "unsupported_phrasing" else "missing_context"
            return ClaimCheck(
                text, "unverified", reason,
                f"ADA could not read the claim as a calculation on this table: {answer.message}",
                claimed=number, question=asked, source=source,
                clarification="Say which column the number is about, and over which rows (a group, a year).",
            )
    unnamed = [
        column for column in (answer.plan.measure, answer.plan.partner)
        if column and not _named(asked, column)
    ]
    if unnamed:
        return ClaimCheck(
            text, "unverified", "missing_context",
            f"The claim does not say what the number measures; ADA would have used {unnamed[0]}, the column "
            "it picks when none is named, and a check should not rest on that guess.",
            claimed=number, question=asked, source=source,
            clarification="Which column is the number about?",
        )
    computed, notes = _comparable(answer, number)
    assumptions.extend(notes)
    scope = answer.scope
    if scope is not None:
        assumptions.extend(scope.readings)
        assumptions.extend(f"Left out {_counted(reason, count)}." for reason, count in scope.left_out)
    if number.qualifier:
        assumptions.append(f'Read "{number.qualifier}" as {number.interval()}.')
    measure = answer.plan.measure
    claimed_unit, recorded_unit = _claim_unit(number), column_unit(measure)
    if claimed_unit and recorded_unit and UNITS[claimed_unit][0] != UNITS[recorded_unit][0]:
        return ClaimCheck(
            text, "unverified", "conflicting_evidence",
            f'The claim gives the number in {number.unit}, but {measure} is recorded in {recorded_unit}, '
            f"which measures {UNITS[recorded_unit][0]}, not {UNITS[claimed_unit][0]}.",
            claimed=number, question=asked, answer=answer, assumptions=tuple(assumptions), source=source,
            clarification=f"Is the number about {measure}, or about another column?",
        )
    if claimed_unit and recorded_unit and computed is not None and claimed_unit != recorded_unit:
        converted = computed * UNITS[recorded_unit][1] / UNITS[claimed_unit][1]
        assumptions.append(
            f"Converted {measure} from {recorded_unit}, the unit in its name, to {claimed_unit}: "
            f"{_plain(computed)} {recorded_unit} is {_plain(converted)} {claimed_unit}."
        )
        computed = converted
    elif number.unit and not number.scale_word and not recorded_unit:
        assumptions.append(f'The unit "{number.unit}" comes from the claim; the check compares numbers only.')
    warnings = list(scope.cautions) if scope is not None else []
    checks: list[Check] = []
    repeated = _repeated_units(analysis, answer)
    if repeated is not None:
        checks.append(repeated)
        if not repeated.passed or "weights each" in repeated.detail:
            warnings.append(repeated.detail)
    if computed is None:
        groups = len(answer.values)
        dated = answer.plan.year is not None or answer.plan.month is not None
        if groups > 1:
            reason, clarification = "missing_context", "Which group is the number about?"
            message = f"The answer gives {groups} numbers, one per group, and the claim does not say which."
        elif dated and not groups:
            reason, clarification = "missing_context", "Which column holds the date or year?"
            message = f"ADA computed no number for it: {answer.answer}"
        else:
            reason, clarification = "unsupported_analysis", "Say which single number the claim is about."
            message = f"The calculation the claim describes gives no single number: {answer.answer}"
        return ClaimCheck(
            text, "unverified", reason, message, claimed=number, question=asked, answer=answer,
            assumptions=tuple(assumptions), warnings=tuple(warnings), checks=tuple(checks),
            clarification=clarification, source=source,
        )
    agreement: Literal["match", "mismatch"] = "match" if number.holds(computed) else "mismatch"
    reason = None
    if agreement == "mismatch" and not recorded_unit:
        scaled = _scale_note(number, computed, measure)
        if scaled and scaled[0] == "unit":
            agreement = "match"
            assumptions.append(scaled[1])
        elif scaled:
            return ClaimCheck(
                text, "unverified", "missing_context", scaled[1], claimed=number, question=asked,
                answer=answer, compared=computed, agreement="mismatch", assumptions=tuple(assumptions),
                warnings=tuple(warnings),
                checks=tuple(checks), source=source,
                clarification=f"What unit is {measure or 'the value'} recorded in?",
            )
    checks.insert(0, Check(
        "numeric_agreement", agreement == "match",
        f"computed {_plain(computed)}; the claim stands for {number.interval()}",
    ))
    failed = [check for check in checks if not check.passed]
    described = answer.calculation
    rows = f" ({scope.rows_used:,} rows)" if scope is not None and scope.rows_used is not None else ""
    if agreement == "mismatch":
        reason = "failed_check"
        message = (
            f"The data give {_plain(computed)}{rows} for {described}; the claim's {number.text} stands for "
            f"{number.interval()}."
        )
        per_row = dict(answer.figures).get("per_row")
        if per_row is not None and number.holds(per_row) and answer.plan.per:
            message += (
                f" The claim matches the average taken row by row ({_plain(per_row)}), which counts each row "
                f"rather than each {answer.plan.per[0]}."
            )
    elif failed:
        reason = "failed_check"
        message = (
            f"The number agrees ({_plain(computed)}{rows}, {described}), but a check on the analysis failed: "
            f"{failed[0].detail}"
        )
    else:
        message = (
            f"{_plain(computed)}{rows} from {described} lies in {number.interval()}, "
            f"which {number.text} stands for."
        )
    return ClaimCheck(
        text, "verified" if reason is None else "unverified", reason, message, claimed=number, question=asked,
        answer=answer, compared=computed, agreement=agreement, assumptions=tuple(dict.fromkeys(assumptions)),
        warnings=tuple(dict.fromkeys(warnings)), checks=tuple(checks), source=source,
    )


__all__ = [
    "CLAIM_SCHEMA_VERSION",
    "REASONS",
    "Alternative",
    "Check",
    "ClaimCheck",
    "ClaimedNumber",
    "check_claim",
    "read_claim",
]
