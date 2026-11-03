"""Arithmetic written in LaTeX, as worked answers to competition problems write it.

``check_arithmetic`` reads a line with a fraction, a root, a power in braces, a binomial coefficient or a
boxed answer here (``\\frac{14}{3}``, ``\\sqrt{27} = 3\\sqrt{3}``, ``2^{10}``, ``\\binom{6}{2}``,
``\\boxed{7}``). Each side of each ``=`` is read into an expression ``ada.compute`` takes, and every step of
a chain whose two sides are made of numbers is checked, the stated side too when it is an expression
rather than a number. A side with a letter, a ``\\pm``, a mixed number such as ``3\\frac{1}{2}`` or a sign
ADA does not read is skipped, and so is a step written as a condition or a contradiction.
"""

from __future__ import annotations

import re
from decimal import Decimal
from fractions import Fraction

from .api import Refusal
from .compute import (
    _BOLD,
    _BULLET,
    _CONDITION_WORDS,
    _CONTRADICTION,
    _SCALE_WORDS,
    Checked,
    Computation,
    _exact_value,
    _written_as,
    compute,
)

#: A line with any of these is read as LaTeX.
LATEX = re.compile(r"\\(?:[dtc]?frac|sqrt|[dt]?binom|boxed)(?![A-Za-z])|\^\{")
#: Digits worked out to compare two values that are not both rational, and how many of them must agree.
DIGITS = 45
AGREE = 30
#: Braces nested deeper than this are not read.
MAX_DEPTH = 40

_FRACTIONS = ("frac", "dfrac", "tfrac", "cfrac")
_BINOMIALS = ("binom", "dbinom", "tbinom")
_SIZES = ("left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr")
_SIGNS = {"cdot": "*", "times": "*", "ast": "*", "div": "/", "pi": "pi", "%": "%", "{": "(", "}": ")",
          ",": " ", ";": " ", ":": " ", "!": "", " ": " ", "displaystyle": "", "textstyle": ""}
#: The brackets \left and \right take; \left. is no bracket.
_BRACKETS = (("\\lbrace", "("), ("\\rbrace", ")"), ("\\{", "("), ("\\}", ")"), ("(", "("), ("[", "("),
             (")", ")"), ("]", ")"), (".", ""))
#: Where one equation ends and the next begins, at the top level of a line: spaces, arrows, relations other
#: than =, text, and a comma or semicolon followed by a space.
_BREAK = re.compile(r"\\(?:q?quad|Rightarrow|Longrightarrow|implies|iff|Leftrightarrow|rightarrow|"
                    r"longrightarrow|to|therefore|because|le|leq|ge|geq|lt|gt|neq|ne|approx|simeq|sim|"
                    r"equiv|pmod|bmod|mod|in|mid)(?![A-Za-z])|\\(?:text|textbf|mathrm|mbox)\s*\{[^{}]*\}|"
                    r"[<>≈≤≥≠⇒→]|[,;:](?=\s)")
_TEXT = re.compile(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}")
_DELIMITERS = re.compile(r"\$\$?|\\[()\[\]]")
_LEADING = re.compile(r"\s*(?:[-+*/×÷−]|\\(?:cdot|times|div)(?![A-Za-z]))")
#: Read two ways: "3\frac{1}{2}" (3½ or 3 × ½) and "2\sqrt{2}^2" ((2√2)² or 2 × (√2)²).
_TWO_WAYS = re.compile(r"\d\s*\\[dtc]?frac(?![A-Za-z])|\d\s*\\sqrt\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}\s*\^")
_NAMES = re.compile(r"(?<![A-Za-z])(?:sqrt|root|comb|pi)(?![A-Za-z])")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?%?")
_OPERATION = re.compile(r"[-+*/^!%]|(?<![A-Za-z])(?:sqrt|root|comb)\(")
_WORD = re.compile(r"(?<![\\A-Za-z])[A-Za-z]+")


class _Unreadable(Exception):
    """A sign ADA does not read as arithmetic."""


def to_plain(latex: str) -> str | None:
    """The expression ``ada.compute`` takes for a side of an equation written in LaTeX, or None when the
    side has a letter or a sign ADA does not read: ``\\frac{1}{2}\\sqrt{8}`` gives ``((1)/(2))sqrt(8)``."""
    text = latex.replace("{,}", "").replace(",\\!", "").replace("−", "-").replace("·", "*")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    if not text.strip() or _TWO_WAYS.search(text):
        return None
    try:
        plain = _Reader(text).read()
    except _Unreadable:
        return None
    if re.search(r"[A-Za-z_&#$]", _NAMES.sub("", plain)):
        return None
    return " ".join(plain.split())


class _Reader:
    """Reads LaTeX from left to right into ADA's notation."""

    def __init__(self, text: str):
        self.text = text
        self.at = 0
        self.depth = 0

    def read(self, stop: str | None = None) -> str:
        parts = []
        while self.at < len(self.text):
            char = self.text[self.at]
            if char == stop:
                return "".join(parts)
            if char == "\\":
                parts.append(self.command())
            elif char == "{":
                parts.append(f"({self.group()})")
            elif char in "}]" and stop is not None:
                raise _Unreadable
            elif char == "^":
                self.at += 1
                parts.append(f"^({self.argument()})")
            else:
                parts.append({"[": "(", "]": ")"}.get(char, char))
                self.at += 1
        if stop is not None:
            raise _Unreadable
        return "".join(parts)

    def group(self) -> str:
        self.at += 1
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise _Unreadable
        inner = self.read("}")
        self.depth -= 1
        self.at += 1
        return inner

    def skip_spaces(self) -> None:
        while self.at < len(self.text) and self.text[self.at] == " ":
            self.at += 1

    def argument(self) -> str:
        self.skip_spaces()
        if self.at >= len(self.text):
            raise _Unreadable
        char = self.text[self.at]
        if char == "{":
            return self.group()
        if char == "\\":
            return self.command()
        if char in "}^_":
            raise _Unreadable
        self.at += 1
        return char

    def command(self) -> str:
        match = re.compile(r"\\([A-Za-z]+|.)").match(self.text, self.at)
        if match is None:
            raise _Unreadable
        name = match.group(1)
        self.at = match.end()
        if name in _FRACTIONS:
            top, bottom = self.argument(), self.argument()
            return f"(({top})/({bottom}))"
        if name == "sqrt":
            self.skip_spaces()
            index = None
            if self.text.startswith("[", self.at):
                self.at += 1
                index = self.read("]")
                self.at += 1
            radicand = self.argument()
            return f"root(({radicand}), ({index}))" if index else f"sqrt({radicand})"
        if name in _BINOMIALS:
            n, k = self.argument(), self.argument()
            return f"comb(({n}), ({k}))"
        if name == "boxed":
            return f"({self.argument()})"
        if name in _SIZES:
            return self.delimiter()
        if name in _SIGNS:
            return _SIGNS[name]
        raise _Unreadable

    def delimiter(self) -> str:
        self.skip_spaces()
        for written, plain in _BRACKETS:
            if self.text.startswith(written, self.at):
                self.at += len(written)
                return plain
        raise _Unreadable


def check_line(line: str) -> list[Checked]:
    """Check every step written in LaTeX on one line of a worked answer."""
    line = _BOLD.sub("", line)
    displayed = not _DELIMITERS.search(line)
    # A line of a displayed block that starts with an operator goes on from the line before: "+ \frac{3}{10}".
    # Elsewhere a "- " or "+ " at the start is a list marker.
    continued = displayed and bool(_LEADING.match(line))
    if not displayed:
        line = _BULLET.sub("", line)
    pieces = []
    for index, piece in enumerate(_DELIMITERS.split(line)):
        # Outside $...$ is prose, unless the line has no $ at all: a line of a displayed block.
        if index % 2 or displayed:
            pieces += _clauses(piece)
        else:
            pieces.append(("prose", piece))
    checked = []
    for index, (kind, text) in enumerate(pieces):
        if kind == "math":
            before = " ".join(item[1] for item in pieces[:index])
            after = " ".join(item[1] for item in pieces[index + 1:])
            checked += _checked_clause(text, before, after, continued and index == 0)
    return checked


def _clauses(math: str) -> list[tuple[str, str]]:
    """Math split where one equation ends and the next begins, at the top level; text is prose."""
    pieces, depth, start, at = [], 0, 0, 0
    while at < len(math):
        char = math[at]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and (found := _BREAK.match(math, at)):
            pieces.append(("math", math[start:at]))
            text = _TEXT.match(found.group())
            pieces.append(("prose", text.group(1) if text else found.group()))
            start = at = found.end()
            continue
        elif char == "\\" and at + 1 < len(math):
            at += 1
        at += 1
    pieces.append(("math", math[start:]))
    return [(kind, text) for kind, text in pieces if text.strip()]


def _sides(math: str) -> list[str]:
    """The sides of a chain of equations, split at each = at the top level."""
    sides, depth, start = [], 0, 0
    for at, char in enumerate(math):
        depth += {"{": 1, "}": -1}.get(char, 0)
        if char == "=" and depth == 0:
            sides.append(math[start:at])
            start = at + 1
    return sides + [math[start:]]


def _checked_clause(math: str, before: str, after: str, continued: bool) -> list[Checked]:
    """Check each step of one chain of equations; ``continued`` when its first side goes on from the line
    before."""
    sides = [side.replace("&", " ") for side in _sides(math)]
    if len(sides) < 2:
        return []
    # "if 2 + 2 = 5" and "3 + 4 = 8, a contradiction": the writer does not claim them.
    if re.search(rf"\b(?:{'|'.join(_CONDITION_WORDS)})\W*$", before, re.IGNORECASE):
        return []
    if _CONTRADICTION.search(after.lower()):
        return []
    checked = []
    for index in range(len(sides) - 1):
        left, stated = sides[index], sides[index + 1]
        if index == 0:
            left = None if continued else _end_of(left)
            # "\text{so } + \frac{3}{10} = 1.4667" goes on from what came before too.
            if left is not None and re.match(r"\s*(?:[+*/^×÷]|\\(?:cdot|times|div)(?![A-Za-z]))", left):
                left = None
        if index == len(sides) - 2:
            stated = _start_of(stated, after)
        item = _checked_step(left, stated, first=index == 0, last=index == len(sides) - 2)
        if item is not None:
            checked.append(item)
    return checked


def _words(side: str) -> list[re.Match]:
    """The words at the top level of a side, not inside braces."""
    return [word for word in _WORD.finditer(side) if not _depth_at(side, word.start())]


def _end_of(side: str) -> str | None:
    """The first side of a chain without the words before it: "the area is \\frac{1}{2} \\cdot 6", but not
    "B + 2 + 2", whose side goes on to the left, or "x2 + 3"."""
    if to_plain(side) is not None:
        return side
    words = _words(side)
    if not words or words[-1].group().lower() in _CONDITION_WORDS:
        return None
    rest = side[words[-1].end():]
    if not rest[:1].isspace() or re.match(r"\s*(?:[-+*/^=)×÷]|\\(?:cdot|times|div)(?![A-Za-z]))", rest):
        return None
    return rest


def _start_of(side: str, after: str) -> str | None:
    """The last side of a chain without the words after it: "6 square units", but not "2x", "10 million" or
    "16 friends \\cdot 3", whose stated value goes on."""
    words = _words(side)
    if words:
        head = side[:words[0].start()]
        if not head[-1:].isspace():
            return None
        following, side = side[words[0].start():] + " " + after, head
    else:
        following = after
    first = re.match(r"\s*([A-Za-z]+)\s*([-+*/×÷]|\\(?:cdot|times|div)(?![A-Za-z]))?", following)
    if first and (first.group(1).lower() in _SCALE_WORDS or first.group(2)):
        return None
    return side


def _depth_at(text: str, at: int) -> int:
    return text[:at].count("{") - text[:at].count("}")


def _checked_step(left: str | None, stated: str | None, first: bool, last: bool) -> Checked | None:
    """Check one step, ``left = stated``, when both sides are made of numbers and one has an operation."""
    if left is None or stated is None:
        return None
    plain_left, plain_stated = to_plain(left), to_plain(stated)
    if plain_left is None or plain_stated is None:
        return None
    operated = _OPERATION.search(plain_left.lstrip("-"))
    # "1 = \sqrt{11 - 2(1)} + 4" puts a value into an equation to test it, rather than claiming it holds.
    if not operated and (first or not _OPERATION.search(plain_stated.lstrip("-"))):
        return None
    value, claimed = _value(plain_left), _value(plain_stated)
    if value is None or claimed is None:
        return None
    if _NUMBER.fullmatch(plain_stated.replace(" ", "")):
        number = plain_stated.replace(" ", "").rstrip("%")
        places = len(number.split(".")[1]) if "." in number else 0
        exact = _exact(value)
        readings = [(Fraction(Decimal(number)), places)]
        if plain_stated.endswith("%"):
            readings.append((Fraction(Decimal(number)) / 100, places + 2))
        correct = any(_written_as(exact, reading, shown) for reading, shown in readings)
    else:
        # "= 0.5 + 0.666 + 2": a stated side with rounded decimals agrees to as many places as they have.
        # Otherwise the sides agree exactly when both are rational, and to 30 digits when one is not.
        places = [len(found) for found in re.findall(r"\d\.(\d+)", plain_stated)]
        if value.exact and claimed.exact and not places:
            correct = _exact_value(value) == _exact_value(claimed)
        else:
            one, other = Decimal(value.decimal), Decimal(claimed.decimal)
            agree = min(places) if places else AGREE
            correct = abs(one - other) <= Decimal(10) ** -agree * max(Decimal(1), abs(one))
    if not correct and not last:
        # "3 \times 4 = 12 + 1 = 13": a running total, not a claim that 3 × 4 is 13.
        total = re.match(r"\(?(-?\d+(?:\.\d+)?)\)?\s*[-+*/]", plain_stated)
        if total and Fraction(Decimal(total.group(1))) == _exact(value):
            return None
    shown = value.value if value.exact else compute(plain_left).value
    return Checked(" ".join(left.split()), " ".join(stated.split()), shown, correct)


def _value(plain: str) -> Computation | None:
    result = compute(plain, digits=DIGITS)
    return None if isinstance(result, Refusal) else result


def _exact(value: Computation) -> Fraction:
    return _exact_value(value) if value.exact else Fraction(Decimal(value.decimal))
