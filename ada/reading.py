"""Which numbers of a word problem a model's working leaves out: a check of how the model read the problem,
not of its arithmetic.

    from ada.reading import check_reading
    reading = check_reading(
        "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the "
        "house by 150%. How much profit did he make?",
        "cost = 80000; repairs = 50000; value = cost + repairs + repairs / 2; "
        "profit = value - cost - repairs")
    reading.left_out        # ("150",)
    reading.warning         # "The working leaves out 150 from the problem. ..."

    ada compute "x = 80000 + 50000" --problem "Josh buys a house for $80,000 ..."

A model that leaves a number of the problem out of its working has usually read the problem wrong. In the
GSM8K runs of paper/runs/2026-11-05-gsm-ladder, problems 0 to 299, the equations ADA solved gave the right
answer 66% of the time when they used every number of the problem and 23% of the time when they left one
out, by the rules below.

- The problem's numbers are those written in digits, with thousands separators (80,000) and decimals
  (2.5). Ordinals (the 3rd day) and clock times (8:30) are not counted, nor are 0 and 1, which working
  often leaves implicit (1 glass costs $5 gives price = 5). Numbers written in words are not read.
- A number is used when it, or for a percentage the number divided by 100 (25 as 0.25), appears as a
  number in the working.

A number left out is a reason to read the problem again, not proof of a mistake: working may combine two
numbers into one (16 glasses as 8 pairs) or leave out a number the question does not need.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

#: A number in digits, with thousands separators and a decimal part; not part of a word, a clock time or an
#: ordinal.
_PROBLEM_NUMBER = re.compile(r"(?<![\w.,:])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![\d:])(?!(?:st|nd|rd|th)\b)")
_CLOCK = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)")
#: Numbers in working: each run of digits, with or without thousands separators, so that 80,000 and
#: gcd(12,180) are both read.
_WORKING_GROUPED = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?")
_WORKING_PLAIN = re.compile(r"(?<![\w])\d*\.?\d+")
#: Numbers working often leaves implicit.
_IMPLICIT = frozenset({Fraction(0), Fraction(1)})


@dataclass(frozen=True)
class Reading:
    """The numbers of a problem, as written, and those the working leaves out."""

    numbers: tuple[str, ...]
    left_out: tuple[str, ...]

    @property
    def warning(self) -> str | None:
        """A sentence for the model when the working leaves out a number, else None."""
        if not self.left_out:
            return None
        named = _listed(self.left_out)
        each = "it stands" if len(self.left_out) == 1 else "each stands"
        return (f"The working leaves out {named} from the problem. A number left out often means the problem "
                f"was read wrong: check what {each} for, and whether the question needs it.")

    def to_dict(self) -> dict:
        return {"problem_numbers": list(self.numbers), "left_out": list(self.left_out)}


def check_reading(problem: str, working: str) -> Reading:
    """The numbers of ``problem`` and those ``working`` (equations or an expression) leaves out, each once,
    in the order the problem gives them."""
    numbers: dict[Fraction, str] = {}
    for whole, part in _PROBLEM_NUMBER.findall(_CLOCK.sub(" ", problem or "")):
        value = Fraction((whole + part).replace(",", ""))
        if value not in _IMPLICIT:
            numbers.setdefault(value, whole + part)
    used = _working_numbers(working or "")
    left = [written for value, written in numbers.items() if value not in used and value / 100 not in used]
    return Reading(tuple(numbers.values()), tuple(left))


def _working_numbers(working: str) -> set[Fraction]:
    found = {Fraction(text.replace(",", "")) for text in _WORKING_GROUPED.findall(working)}
    found |= {Fraction(text) for text in _WORKING_PLAIN.findall(working)}
    return found


def _listed(items: tuple[str, ...]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
