"""Exact arithmetic for people and language models: an expression in, its value and every step out.

    from ada.compute import compute
    result = compute("2^10 + 3/4")
    result.value        # "4099/4"
    result.decimal      # "1024.75"
    result.steps        # ["2^10 = 1024", "1024 + 3/4 = 4099/4"]

    ada compute "0.1 + 0.2"          # 3/10, exactly 0.3

A language model guesses at arithmetic: it carries a digit wrong, or rounds
too early, and cannot tell that it has. This module does the arithmetic from
first principles instead:

- Numbers stay exact. Integers have no size limit short of ``MAX_DIGITS``, a
  decimal such as 0.1 is read as the fraction 1/10, and division gives a
  fraction, so 0.1 + 0.2 is exactly 3/10. Only a result that is not a
  rational number (the square root of 2, pi, most logarithms) is approximate.
  It is computed with Python's decimal module to the significant digits asked
  for, and marked approximate.
- Every operation is a step, written with the values it combined.
- An expression that reads two ways with two values is refused with both:
  6/2(1+2) is 9 or 1. Where the readings agree, or a convention settles it
  (-2^2 is -4, 2^3^2 is 2^9), the reading taken is listed in ``assumptions``.
- Nothing is run as Python code. A small grammar reads numbers, the operators
  + - * / // ^ ** mod ! % and the functions in ``FUNCTIONS``, and nothing else.

A result that cannot be computed honestly is a ``Refusal`` with the reason:
division by zero, the square root of a negative number, a name ADA does not
know, ``log`` without a base, or a number too large to keep exactly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal, localcontext
from decimal import Overflow as DecimalOverflow
from fractions import Fraction
from typing import Any

from .api import Refusal

#: Significant digits of an approximate result, unless asked otherwise.
DIGITS = 20
#: The most significant digits that may be asked for.
MAX_DIGITS_ASKED = 1000
#: The most digits an exact integer, numerator or denominator may have.
MAX_DIGITS = 10_000
#: The longest expression read, in characters.
MAX_LENGTH = 2000
#: The deepest nesting of parentheses and operators read.
MAX_DEPTH = 100
#: Steps listed before the rest are counted instead.
MAX_STEPS = 100
#: The longest repeating block written out in a repeating decimal.
MAX_PERIOD = 60
#: Digits of an exact integer shown whole; a longer one is shown by its ends and its length.
MAX_SHOWN = 400
#: Extra digits carried while computing an approximate result, then rounded away.
GUARD_DIGITS = 10

Number = Fraction | Decimal


class _Refused(Exception):
    """A refusal raised from deep inside parsing or evaluation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.refusal = Refusal(code, message)


def _refuse(code: str, message: str):
    raise _Refused(code, message)


# --- Reading the expression ---------------------------------------------------------------------------

#: Characters that mean the same as an ASCII operator.
SYMBOLS = {"×": "*", "·": "*", "⋅": "*", "∙": "*", "÷": "/", "−": "-", "–": "-", "—": "-", "²": "^2",
           "³": "^3", "π": "pi", "∗": "*"}
CURRENCY = "$€£¥₹"
TOKEN = re.compile(
    r"\s*(?:"
    r"(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>\*\*|//|[-+*/^%!(),])"
    r")"
)


@dataclass(frozen=True)
class _Token:
    kind: str
    text: str
    start: int
    #: No space before it: "1,234" has a tight comma, "gcd(1, 234)" does not.
    tight: bool


def _tokens(text: str) -> tuple[list[_Token], list[str]]:
    assumptions = []
    # √16 is sqrt(16), √(2+2) is sqrt(2+2).
    text = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", text).replace("√", "sqrt")
    for symbol, plain in SYMBOLS.items():
        text = text.replace(symbol, plain)
    if any(sign in text for sign in CURRENCY):
        assumptions.append("Currency signs are dropped; the numbers are read as written.")
        text = "".join(" " if character in CURRENCY else character for character in text)
    tokens, position = [], 0
    while position < len(text):
        if text[position:].strip() == "":
            break
        match = TOKEN.match(text, position)
        if not match or match.end() == position:
            character = text[position + len(text[position:]) - len(text[position:].lstrip())]
            _refuse("invalid_expression", f"ADA cannot read {character!r} in the expression. "
                                          f"It reads numbers, + - * / // ^ mod ! %, parentheses and "
                                          f"the functions {', '.join(sorted(FUNCTIONS))}.")
        kind = match.lastgroup
        tokens.append(_Token(kind, match.group(kind), match.start(kind), match.start(kind) == position))
        position = match.end()
    if not tokens:
        _refuse("invalid_expression", "The expression is empty.")
    return tokens, assumptions


@dataclass(frozen=True)
class _Node:
    kind: str                       # number, constant, negative, binary, call, factorial, percent, group
    value: Any = None               # the Fraction of a number, the name of a constant or call, the operator
    children: tuple = ()
    text: str = ""                  # how a number was written


#: Binding strength of the binary operators, weakest first.
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "//": 2, "mod": 2, "implicit": 2, "of": 2, "^": 4}


class _Parser:
    """Precedence climbing over the tokens.

    ``implicit`` says how to read two operands with no operator between them,
    as in 2(3+4): as an ordinary multiplication ("loose", so 6/2(1+2) is
    (6/2)*(1+2)) or binding tighter than * and / ("tight", 6/(2*(1+2))).
    """

    def __init__(self, tokens: list[_Token], implicit: str):
        self.tokens, self.position, self.implicit = tokens, 0, implicit
        self.assumptions: list[str] = []
        self.used_implicit = False
        self.depth = 0
        # How many function calls are open, so "1,234" in gcd(1,234) is two numbers.
        self.calls = 0

    def peek(self, offset: int = 0) -> _Token | None:
        index = self.position + offset
        return self.tokens[index] if index < len(self.tokens) else None

    def take(self) -> _Token:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def expect(self, text: str) -> None:
        token = self.peek()
        if token is None or token.text != text:
            found = "the end" if token is None else repr(token.text)
            _refuse("invalid_expression", f"Expected {text!r} but found {found}.")
        self.take()

    def parse(self) -> _Node:
        node = self.additive()
        if self.peek() is not None:
            if self.peek().text == ",":
                _refuse("invalid_expression", "ADA reads a comma only between a function's arguments, as in "
                                              "gcd(12, 18), or between thousands, as in 1,234.")
            _refuse("invalid_expression", f"ADA cannot read the expression from {self.peek().text!r} on.")
        return node

    def nested(self):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            _refuse("too_large", f"The expression nests more than {MAX_DEPTH} levels deep.")

    def additive(self) -> _Node:
        node = self.multiplicative()
        while (token := self.peek()) is not None and token.text in "+-" and token.kind == "op":
            self.take()
            node = _Node("binary", token.text, (node, self.multiplicative()))
        return node

    def starts_operand(self, token: _Token | None) -> bool:
        if token is None:
            return False
        return token.kind == "number" or token.text == "(" or (
            token.kind == "name" and token.text.lower() not in ("mod", "of"))

    def multiplicative(self) -> _Node:
        node = self.implicit_product() if self.implicit == "tight" else self.unary()
        while True:
            token = self.peek()
            if token is not None and (token.text in ("*", "/", "//") or token.text.lower() == "mod"):
                self.take()
                operator = "mod" if token.text.lower() == "mod" else token.text
                right = self.implicit_product() if self.implicit == "tight" else self.unary()
                node = _Node("binary", operator, (node, right))
            elif token is not None and token.text.lower() == "of" and node.kind == "percent":
                self.take()
                node = _Node("binary", "of", (node, self.unary()))
            elif self.implicit == "loose" and self.starts_operand(token):
                self.check_implicit(node, token)
                node = _Node("binary", "implicit", (node, self.unary()))
            else:
                return node

    def implicit_product(self) -> _Node:
        node = self.unary()
        while self.starts_operand(self.peek()):
            self.check_implicit(node, self.peek())
            node = _Node("binary", "implicit", (node, self.unary()))
        return node

    def check_implicit(self, left: _Node, token: _Token) -> None:
        if left.kind == "number" and token.kind == "number":
            _refuse("invalid_expression", f"There is no operator between {left.text} and {token.text}. "
                                          "Write the operator, such as * or +.")
        self.used_implicit = True

    def unary(self) -> _Node:
        token = self.peek()
        if token is not None and token.kind == "op" and token.text in "+-":
            self.take()
            self.nested()
            operand = self.unary()
            self.depth -= 1
            if token.text == "+":
                return operand
            if operand.kind == "binary" and operand.value == "^":
                base = operand.children[0]
                self.assumptions.append(
                    f"-{show(base)}^{show(operand.children[1])} is read as -({show(operand)}), "
                    "the power first, as in algebra.")
            return _Node("negative", None, (operand,))
        return self.power()

    def power(self) -> _Node:
        base = self.postfix()
        token = self.peek()
        if token is not None and token.text in ("^", "**"):
            self.take()
            self.nested()
            exponent = self.unary()
            self.depth -= 1
            if exponent.kind == "binary" and exponent.value == "^":
                self.assumptions.append(f"{show(base)}^{show(exponent)} is read as "
                                        f"{show(base)}^({show(exponent)}), from the right, as in algebra.")
            return _Node("binary", "^", (base, exponent))
        return base

    def postfix(self) -> _Node:
        node = self.primary()
        while (token := self.peek()) is not None and token.text in ("!", "%"):
            if token.text == "!":
                self.take()
                node = _Node("factorial", None, (node,))
                continue
            after = self.peek(1)
            if self.starts_operand(after):
                _refuse("ambiguous_expression",
                        f"{show(node)} % {after.text} can mean the remainder of {show(node)} divided by "
                        f"{after.text}, or {show(node)} percent of {after.text}. Write {show(node)} mod "
                        f"{after.text} for the remainder, or {show(node)}% of {after.text} for the percent.")
            self.take()
            self.assumptions.append(f"{show(node)}% is read as {show(node)}/100.")
            node = _Node("percent", None, (node,))
        return node

    def primary(self) -> _Node:
        token = self.peek()
        if token is None:
            _refuse("invalid_expression", "The expression ends where a number was expected.")
        if token.kind == "number":
            self.take()
            return self.number(token)
        if token.text == "(":
            self.take()
            self.nested()
            outer, self.calls = self.calls, 0
            inner = self.additive()
            self.calls = outer
            self.expect(")")
            self.depth -= 1
            return _Node("group", None, (inner,))
        if token.kind == "name":
            self.take()
            name = token.text
            key = name.lower()
            if self.peek() is not None and self.peek().text == "(" and key in FUNCTION_NAMES:
                return self.call(name, key)
            if key in CONSTANTS:
                return _Node("constant", key)
            if key in FUNCTION_NAMES:
                if self.starts_operand(self.peek()):
                    # sqrt 16: a function written without its parentheses takes the operand after it.
                    operand = self.power()
                    self.assumptions.append(f"{name} {show(operand)} is read as {name}({show(operand)}).")
                    return _Node("call", ALIASES.get(key, key), (operand,))
                _refuse("invalid_expression",
                        f"{name} needs its arguments in parentheses, such as {name}(9).")
            _refuse("invalid_expression",
                    f"ADA does not know {name!r}. Write the expression with numbers only, or use one of "
                    f"the functions {', '.join(sorted(FUNCTIONS))} and the constants pi and e. "
                    "Work out what each name stands for first, then compute.")
        _refuse("invalid_expression", f"ADA cannot read {token.text!r} where a number was expected.")

    def number(self, token: _Token) -> _Node:
        text = token.text
        if self.calls == 0 and "." not in text and "e" not in text.lower() and len(text) <= 3:
            # 1,234,567 outside a function's parentheses is one number written with thousands separators.
            groups = [text]
            while ((comma := self.peek()) is not None and comma.text == "," and comma.tight
                   and (digits := self.peek(1)) is not None and digits.tight and digits.kind == "number"
                   and re.fullmatch(r"\d{3}(\.\d+)?", digits.text)):
                self.take()
                groups.append(self.take().text)
            if len(groups) > 1:
                written = ",".join(groups)
                text = "".join(groups)
                self.assumptions.append(f"{written} is read as {text}, with commas between thousands.")
        written = Decimal(text)
        if written and abs(written.adjusted()) > MAX_DIGITS:
            _refuse("too_large", f"{text} has more than {MAX_DIGITS:,} digits; ADA keeps exact numbers up "
                                 f"to {MAX_DIGITS:,} digits.")
        return _Node("number", Fraction(written), text=text)

    def call(self, name: str, key: str) -> _Node:
        if key not in FUNCTION_NAMES:
            _refuse("invalid_expression",
                    f"ADA does not know the function {name!r}. It knows "
                    f"{', '.join(sorted(FUNCTIONS))}.")
        self.take()                                 # "("
        self.nested()
        self.calls += 1
        arguments = []
        if self.peek() is not None and self.peek().text != ")":
            arguments.append(self.additive())
            while self.peek() is not None and self.peek().text == ",":
                self.take()
                arguments.append(self.additive())
        self.calls -= 1
        self.expect(")")
        self.depth -= 1
        return _Node("call", ALIASES.get(key, key), tuple(arguments))


def show(node: _Node) -> str:
    """The expression as ADA read it, with its operators written out."""
    if node.kind == "number":
        return node.text
    if node.kind == "constant":
        return node.value
    if node.kind == "group":
        return f"({show(node.children[0])})"
    if node.kind == "negative":
        return f"-{show(node.children[0])}"
    if node.kind == "factorial":
        return f"{show(node.children[0])}!"
    if node.kind == "percent":
        return f"{show(node.children[0])}%"
    if node.kind == "call":
        return f"{node.value}({', '.join(show(child) for child in node.children)})"
    if node.value == "^":
        return f"{show(node.children[0])}^{show(node.children[1])}"
    spine = []
    while node.kind == "binary" and node.value != "^":
        spine.append(node)
        node = node.children[0]
    text = show(node)
    for link in reversed(spine):
        right = link.children[1]
        written = show(right)
        if right.kind == "binary" and right.value != "^" and \
                PRECEDENCE[right.value] <= PRECEDENCE[link.value]:
            written = f"({written})"
        if link.value == "implicit":
            text += f"*{written}"
        elif link.value in ("+", "-", "mod", "of"):
            text += f" {link.value} {written}"
        else:
            text += f"{link.value}{written}"
    return text


# --- Values ---------------------------------------------------------------------------------------------


def _digits_of(value: int) -> int:
    return len(str(abs(value))) if abs(value) < 10 ** 20 else int(abs(value).bit_length() * math.log10(2)) + 1


def _check_size(value: Number) -> Number:
    if isinstance(value, Fraction):
        largest = max(_digits_of(value.numerator), _digits_of(value.denominator))
        if largest > MAX_DIGITS:
            _refuse("too_large", f"The result has about {largest:,} digits; ADA keeps exact numbers up to "
                                 f"{MAX_DIGITS:,} digits.")
    elif not value.is_finite():
        _refuse("too_large", "The result is too large to compute.")
    return value


def _approx(value: Number) -> Decimal:
    if isinstance(value, Decimal):
        return +value
    return Decimal(value.numerator) / Decimal(value.denominator)


def _integer(value: Number, what: str) -> int:
    if isinstance(value, Fraction) and value.denominator == 1:
        return value.numerator
    _refuse("invalid_expression", f"{what} must be a whole number, and {shown(value)} is not.")


def _iroot(value: int, n: int) -> int | None:
    """The exact n-th root of a non-negative integer, or None when it has none."""
    if value < 2 or n == 1:
        return value
    if n == 2:
        root = math.isqrt(value)
        return root if root * root == value else None
    # Newton's method on whole numbers, from above: it stops at the floor of the root.
    root = 1 << -(-value.bit_length() // n)
    while True:
        better = ((n - 1) * root + value // root ** (n - 1)) // n
        if better >= root:
            break
        root = better
    return root if root ** n == value else None


def _exact_root(value: Fraction, n: int) -> Fraction | None:
    sign = -1 if value < 0 else 1
    top, bottom = _iroot(abs(value.numerator), n), _iroot(value.denominator, n)
    if top is None or bottom is None:
        return None
    return sign * Fraction(top, bottom)


def _pi() -> Decimal:
    """Pi to the context's precision, by the series in the decimal module's documentation."""
    with localcontext() as context:
        context.prec += 2
        three = Decimal(3)
        last, total, term, n, na, d, da = 0, three, three, 1, 0, 0, 24
        while total != last:
            last = total
            n, na = n + na, na + 8
            d, da = d + da, da + 32
            term = term * n / d
            total += term
    return +total


@dataclass(frozen=True)
class _Function:
    least: int
    most: int | None
    about: str


FUNCTIONS = {
    "sqrt": _Function(1, 1, "square root"),
    "cbrt": _Function(1, 1, "cube root"),
    "root": _Function(2, 2, "root(x, n): the n-th root of x"),
    "abs": _Function(1, 1, "absolute value"),
    "floor": _Function(1, 1, "the largest whole number not above x"),
    "ceil": _Function(1, 1, "the smallest whole number not below x"),
    "round": _Function(1, 2, "round(x) or round(x, places); halves round away from zero"),
    "gcd": _Function(2, None, "greatest common divisor"),
    "lcm": _Function(2, None, "least common multiple"),
    "factorial": _Function(1, 1, "n!, the product 1 × 2 × … × n"),
    "comb": _Function(2, 2, "comb(n, k): ways to choose k of n, order not counted"),
    "perm": _Function(2, 2, "perm(n, k): ways to arrange k of n, order counted"),
    "mod": _Function(2, 2, "mod(a, n): the remainder of a divided by n, from 0 to n - 1"),
    "min": _Function(1, None, "the smallest of the values"),
    "max": _Function(1, None, "the largest of the values"),
    "exp": _Function(1, 1, "e to the power x"),
    "ln": _Function(1, 1, "natural logarithm (base e)"),
    "log10": _Function(1, 1, "logarithm base 10"),
    "log2": _Function(1, 1, "logarithm base 2"),
    "log": _Function(2, 2, "log(x, base): logarithm in the base given"),
}
ALIASES = {"ncr": "comb", "choose": "comb", "binomial": "comb", "npr": "perm", "fact": "factorial",
           "ceiling": "ceil", "logn": "ln"}
FUNCTION_NAMES = frozenset(FUNCTIONS) | frozenset(ALIASES)
CONSTANTS = ("pi", "e")


class _Evaluator:
    def __init__(self, digits: int):
        self.digits = digits
        self.steps: list[str] = []
        self.extra_steps = 0
        self.assumptions: list[str] = []

    def step(self, text: str) -> None:
        if len(self.steps) < MAX_STEPS:
            self.steps.append(text)
        else:
            self.extra_steps += 1

    def record(self, written: str, value: Number) -> Number:
        value = _check_size(value)
        sign = "=" if isinstance(value, Fraction) else "≈"
        self.step(f"{written} {sign} {shown(value, self.digits)}")
        return value

    def value(self, node: _Node) -> Number:
        kind = node.kind
        if kind == "number":
            return node.value
        if kind == "group":
            return self.value(node.children[0])
        if kind == "constant":
            constant = _pi() if node.value == "pi" else Decimal(1).exp()
            return self.record(node.value, constant)
        if kind == "negative":
            return -self.value(node.children[0])
        if kind == "percent":
            inner = self.value(node.children[0])
            return self.record(f"{shown(inner, self.digits)}%", inner / 100)
        if kind == "factorial":
            inner = self.value(node.children[0])
            n = _integer(inner, "A factorial's number")
            if n < 0:
                _refuse("invalid_expression", f"{n}! is not defined: a factorial needs a number from 0 up.")
            self.limit_factorial(n)
            return self.record(f"{n}!", Fraction(math.factorial(n)))
        if kind == "call":
            return self.call(node.value, [self.value(child) for child in node.children])
        if node.value == "^":
            return self.binary("^", self.value(node.children[0]), self.value(node.children[1]))
        # 1 + 2 + ... + 999 is a chain down its left side: walk it rather than recurse once per term.
        spine = []
        while node.kind == "binary" and node.value != "^":
            spine.append(node)
            node = node.children[0]
        total = self.value(node)
        for link in reversed(spine):
            right = link.children[1]
            if link.value in ("+", "-") and right.kind == "percent":
                self.percent_change(total, link.value, right)
            total = self.binary(link.value, total, self.value(right))
        return total

    def percent_change(self, base: Number, sign: str, percent: _Node) -> None:
        """100 + 10% is 100.1 to a programming language and 110 to most people: refuse with both."""
        inner = self.value(percent.children[0])
        rate = inner / 100 if isinstance(inner, Fraction) else _approx(inner) / 100
        factor = 1 + rate if sign == "+" else 1 - rate
        added = base + rate if sign == "+" else base - rate
        number, word = shown(base, self.digits), "plus" if sign == "+" else "minus"
        _refuse("ambiguous_expression",
                f"{number} {sign} {show(percent)} reads two ways: "
                f"{number} {sign} {shown(rate, self.digits)} = {shown(added, self.digits)}, "
                f"or {number} {word} {show(percent)} of itself = {shown(base * factor, self.digits)}. "
                f"Write {number} * {shown(factor, self.digits)} for the second.")

    def limit_factorial(self, n: int) -> None:
        digits = math.lgamma(n + 1) / math.log(10) if n > 1 else 1
        if digits > MAX_DIGITS:
            _refuse("too_large", f"{n}! has about {int(digits):,} digits; ADA keeps exact numbers up to "
                                 f"{MAX_DIGITS:,} digits.")

    def operand(self, value: Number, wrap_fractions: bool = False) -> str:
        text = shown(value, self.digits)
        if text.startswith("-") or (wrap_fractions and "/" in text):
            return f"({text})"
        return text

    def binary(self, operator: str, left: Number, right: Number) -> Number:
        symbol = {"implicit": "*", "of": "*"}.get(operator, operator)
        written = f"{shown(left, self.digits)} {symbol} {self.operand(right, operator in ('/', '//', 'mod'))}"
        if operator == "^":
            return self.power(left, right)
        exact = isinstance(left, Fraction) and isinstance(right, Fraction)
        if operator in ("/", "//", "mod") and right == 0:
            _refuse("division_by_zero", f"{written} divides by zero, which has no value.")
        if operator == "+":
            result = left + right if exact else _approx(left) + _approx(right)
        elif operator == "-":
            result = left - right if exact else _approx(left) - _approx(right)
        elif operator in ("*", "implicit", "of"):
            result = left * right if exact else _approx(left) * _approx(right)
        elif operator == "/":
            result = left / right if exact else _approx(left) / _approx(right)
        elif operator == "//":
            if not exact:
                _refuse("invalid_expression", f"{written}: whole-number division needs exact numbers.")
            result = Fraction(math.floor(left / right))
            written = f"floor({shown(left)} / {shown(right)})"
        else:
            return self.call("mod", [left, right])
        return self.record(written, result)

    def power(self, base: Number, exponent: Number) -> Number:
        written = f"{self.operand(base, True)}^{self.operand(exponent, True)}"
        if base == 0 and exponent == 0:
            self.assumptions.append("0^0 is taken as 1, as in counting and in power series.")
            return self.record(written, Fraction(1))
        if base == 0 and exponent < 0:
            _refuse("division_by_zero", f"{written} divides by zero, which has no value.")
        if isinstance(exponent, Fraction) and exponent.denominator == 1:
            n = exponent.numerator
            if isinstance(base, Fraction):
                largest = max(abs(base.numerator), base.denominator)
                if largest > 1 and abs(n) * math.log10(largest) > MAX_DIGITS:
                    _refuse("too_large", f"{written} has about {int(abs(n) * math.log10(largest)):,} digits; "
                                         f"ADA keeps exact numbers up to {MAX_DIGITS:,} digits.")
                return self.record(written, base ** n)
            return self.record(written, _approx(base) ** n)
        if base < 0:
            _refuse("not_a_real_number",
                    f"{written}: a negative number to a fractional power has no single real value. For a "
                    f"real odd root write root(x, n), such as root(-8, 3).")
        if isinstance(base, Fraction) and isinstance(exponent, Fraction):
            root = _exact_root(base, exponent.denominator)
            if root is not None:
                return self.record(written, self.power_exact(root, exponent.numerator, written))
        return self.record(written, _approx(base) ** _approx(exponent))

    def power_exact(self, base: Fraction, n: int, written: str) -> Fraction:
        largest = max(abs(base.numerator), base.denominator)
        if largest > 1 and abs(n) * math.log10(largest) > MAX_DIGITS:
            _refuse("too_large", f"{written} has more than {MAX_DIGITS:,} digits.")
        return base ** n

    def call(self, name: str, arguments: list[Number]) -> Number:
        function = FUNCTIONS[name]
        if len(arguments) < function.least or (function.most is not None and len(arguments) > function.most):
            wanted = (str(function.least) if function.most == function.least
                      else f"{function.least} or more" if function.most is None
                      else f"{function.least} to {function.most}")
            if name == "log" and len(arguments) == 1:
                _refuse("ambiguous_expression",
                        f"log({shown(arguments[0])}) can mean base 10 or base e. Write log10(x), ln(x), "
                        "or log(x, base).")
            _refuse("invalid_expression", f"{name} takes {wanted} argument(s), and was given "
                                          f"{len(arguments)}: {function.about}.")
        written = f"{name}({', '.join(shown(argument, self.digits) for argument in arguments)})"
        handler = getattr(self, f"call_{name}")
        return self.record(written, handler(arguments, written))

    # Each call_ method takes the evaluated arguments and returns the value.

    def call_sqrt(self, arguments, written):
        return self.nth_root(arguments[0], 2, written)

    def call_cbrt(self, arguments, written):
        return self.nth_root(arguments[0], 3, written)

    def call_root(self, arguments, written):
        n = _integer(arguments[1], "A root's degree")
        if n < 1:
            _refuse("invalid_expression", f"{written}: a root's degree must be 1 or more.")
        return self.nth_root(arguments[0], n, written)

    def nth_root(self, value: Number, n: int, written: str) -> Number:
        if value < 0 and n % 2 == 0:
            _refuse("not_a_real_number",
                    f"{written}: an even root of a negative number is not a real number.")
        if isinstance(value, Fraction):
            exact = _exact_root(value, n)
            if exact is not None:
                return exact
        approximate = _approx(value)
        if n == 2:
            return approximate.sqrt()
        magnitude = abs(approximate) ** (Decimal(1) / Decimal(n))
        return -magnitude if approximate < 0 else magnitude

    def call_abs(self, arguments, written):
        return abs(arguments[0])

    def settled(self, value: Decimal, boundary: Decimal, written: str) -> None:
        """An approximate value carries about ``digits`` + ``GUARD_DIGITS`` correct digits; one that close
        to a whole number (or a halfway point) cannot be rounded honestly."""
        tolerance = Decimal(1).scaleb(value.adjusted() - (self.digits + GUARD_DIGITS) + 3)
        if abs(value - boundary) < tolerance:
            _refuse("needs_more_digits",
                    f"{written}: the value is too close to {_decimal_text(boundary, self.digits)} to tell "
                    f"which side it falls on at {self.digits} significant digits. Ask for more digits.")

    def call_floor(self, arguments, written):
        value = arguments[0]
        if isinstance(value, Fraction):
            return Fraction(math.floor(value))
        self.settled(value, value.to_integral_value(), written)
        return Fraction(int(value.to_integral_value(rounding=ROUND_FLOOR)))

    def call_ceil(self, arguments, written):
        value = arguments[0]
        if isinstance(value, Fraction):
            return Fraction(math.ceil(value))
        self.settled(value, value.to_integral_value(), written)
        return Fraction(int(value.to_integral_value(rounding=ROUND_CEILING)))

    def call_round(self, arguments, written):
        places = _integer(arguments[1], "The number of decimal places") if len(arguments) > 1 else 0
        if abs(places) > MAX_DIGITS_ASKED:
            _refuse("too_large", f"{written}: round to at most {MAX_DIGITS_ASKED} places.")
        value = arguments[0]
        if isinstance(value, Decimal):
            if places > self.digits + GUARD_DIGITS - 3 - max(value.adjusted(), 0):
                _refuse("needs_more_digits",
                        f"{written}: an approximate value is known to about {self.digits} significant "
                        "digits, too few for that many places. Ask for more digits.")
            step = Decimal(1).scaleb(-places)
            halfway = (value / step).to_integral_value(rounding=ROUND_FLOOR) * step + step / 2
            self.settled(value, halfway, written)
            with localcontext() as context:
                context.prec = max(context.prec, value.adjusted() + places + 3)
                return Fraction(value.quantize(step, rounding=ROUND_HALF_UP))
        # The exact fraction is compared with the halfway point, so 2.675 rounds to 2.68, not 2.67.
        scale = Fraction(10) ** places
        scaled = abs(value) * scale
        whole = math.floor(scaled)
        if scaled - whole >= Fraction(1, 2):
            whole += 1
        return (whole if value >= 0 else -whole) / scale

    def integers(self, arguments, name) -> list[int]:
        return [_integer(argument, f"Each number given to {name}") for argument in arguments]

    def call_gcd(self, arguments, written):
        return Fraction(math.gcd(*self.integers(arguments, "gcd")))

    def call_lcm(self, arguments, written):
        return Fraction(math.lcm(*self.integers(arguments, "lcm")))

    def call_factorial(self, arguments, written):
        n = _integer(arguments[0], "A factorial's number")
        if n < 0:
            _refuse("invalid_expression", f"{written} is not defined: a factorial needs a number from 0 up.")
        self.limit_factorial(n)
        return Fraction(math.factorial(n))

    def counts(self, arguments, name) -> tuple[int, int]:
        n, k = self.integers(arguments, name)
        if n < 0 or k < 0:
            _refuse("invalid_expression", f"{name}(n, k) needs n and k from 0 up.")
        if k <= n and n > 1:
            digits = (math.lgamma(n + 1) - math.lgamma(n - k + 1)) / math.log(10)
            if digits > MAX_DIGITS:
                _refuse("too_large", f"{name}({n}, {k}) has more than {MAX_DIGITS:,} digits.")
        return n, k

    def call_comb(self, arguments, written):
        return Fraction(math.comb(*self.counts(arguments, "comb")))

    def call_perm(self, arguments, written):
        return Fraction(math.perm(*self.counts(arguments, "perm")))

    def call_mod(self, arguments, written):
        a, n = arguments
        if not (isinstance(a, Fraction) and isinstance(n, Fraction)):
            _refuse("invalid_expression", f"{written}: a remainder needs exact numbers.")
        if n == 0:
            _refuse("division_by_zero", f"{written} divides by zero, which has no value.")
        if n < 0:
            _refuse("invalid_expression", f"{written}: the number divided by must be positive.")
        return a - n * math.floor(a / n)

    def call_min(self, arguments, written):
        return min(arguments, key=_approx)

    def call_max(self, arguments, written):
        return max(arguments, key=_approx)

    def call_exp(self, arguments, written):
        value = arguments[0]
        if value == 0:
            return Fraction(1)
        return _approx(value).exp()

    def logarithm(self, value: Number, base: Number, written: str) -> Number:
        if value <= 0:
            _refuse("not_a_real_number",
                    f"{written}: the logarithm of zero or a negative number is not a real number.")
        if base <= 0 or base == 1:
            _refuse("invalid_expression", f"{written}: a logarithm's base must be positive and not 1.")
        if isinstance(value, Fraction) and isinstance(base, Fraction):
            # log(8, 2) is exactly 3: try the whole power the approximation points to.
            guess = round(float(_approx(value).ln() / _approx(base).ln()))
            largest = max(abs(base.numerator), base.denominator)
            if abs(guess) * math.log10(largest) <= MAX_DIGITS and base ** guess == value:
                return Fraction(guess)
        return _approx(value).ln() / _approx(base).ln()

    def call_ln(self, arguments, written):
        if arguments[0] == 1:
            return Fraction(0)
        return self.logarithm(arguments[0], Decimal(1).exp(), written)

    def call_log10(self, arguments, written):
        return self.logarithm(arguments[0], Fraction(10), written)

    def call_log2(self, arguments, written):
        return self.logarithm(arguments[0], Fraction(2), written)

    def call_log(self, arguments, written):
        return self.logarithm(arguments[0], arguments[1], written)


# --- Showing values -------------------------------------------------------------------------------------


def _terminates(value: Fraction) -> bool:
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    return denominator == 1


def _decimal_text(value: Decimal, digits: int) -> str:
    """A decimal rounded to ``digits`` significant digits, in plain notation unless very large or small."""
    if value == 0:
        return "0"
    with localcontext() as context:
        context.prec = digits
        rounded = +value
    exponent = rounded.adjusted()
    if -7 <= exponent < digits + 5:
        text = format(rounded, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text
    return format(rounded.normalize(), "E").replace("E+", "e+").replace("E-", "e-")


#: Significant digits up to which a step writes a decimal that ends as a decimal rather than a fraction.
SHORT_DECIMAL = 20


def shown(value: Number, digits: int = DIGITS) -> str:
    """A value as a step writes it: a whole number, a short decimal that ends (0.15), a fraction (1/3), or a
    rounded decimal for a value that is not exact."""
    if isinstance(value, Decimal):
        return _decimal_text(value, digits)
    if value.denominator != 1 and _terminates(value):
        exact = _exact_decimal(value)
        if len(exact.replace("-", "").replace(".", "").lstrip("0")) <= SHORT_DECIMAL:
            return exact
    return _fraction_text(value, digits)


def _exact_decimal(value: Fraction) -> str:
    """The decimal of a fraction whose decimal ends, written out in full."""
    with localcontext() as context:
        context.prec = MAX_DIGITS * 2
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def _fraction_text(value: Fraction, digits: int = DIGITS) -> str:
    """A whole number, or a fraction in lowest terms; one too long to read is shown by its size."""
    if value.denominator == 1:
        text = str(value.numerator)
        if len(text.lstrip("-")) > MAX_SHOWN:
            sign = "-" if value < 0 else ""
            body = text.lstrip("-")
            return f"{sign}{body[:20]}…{body[-20:]} ({len(body):,} digits)"
        return text
    text = f"{value.numerator}/{value.denominator}"
    return text if len(text) <= MAX_SHOWN else _decimal_text(_approx_wide(value, digits), digits)


def _approx_wide(value: Number, digits: int) -> Decimal:
    with localcontext() as context:
        context.prec = digits + GUARD_DIGITS
        if isinstance(value, Decimal):
            return +value
        return Decimal(value.numerator) / Decimal(value.denominator)


def repeating(value: Fraction) -> str | None:
    """A fraction whose decimal does not end, with its repeating block in parentheses: 1/7 is 0.(142857).

    None when the decimal ends, or when the repeating block is longer than ``MAX_PERIOD`` digits.
    """
    if _terminates(value):
        return None
    sign = "-" if value < 0 else ""
    numerator, denominator = abs(value.numerator), value.denominator
    whole, remainder = divmod(numerator, denominator)
    seen: dict[int, int] = {}
    digits = []
    while remainder and remainder not in seen:
        if len(digits) > MAX_PERIOD + 40:
            return None
        seen[remainder] = len(digits)
        remainder *= 10
        digit, remainder = divmod(remainder, denominator)
        digits.append(str(digit))
    start = seen[remainder]
    if len(digits) - start > MAX_PERIOD:
        return None
    return f"{sign}{whole}.{''.join(digits[:start])}({''.join(digits[start:])})"


# --- The result -----------------------------------------------------------------------------------------


@dataclass
class Computation:
    """An expression's value with the steps that made it.

    When ``exact`` is true, ``value`` is exact: a whole number, a decimal that
    ends (1024.75), or else a fraction in lowest terms (1/3); ``fraction`` is
    the value as a fraction either way. Otherwise ``value`` is rounded to
    ``digits`` significant digits. ``decimal`` is the value as a decimal, exact
    when ``decimal_exact`` is true and otherwise rounded to ``digits``
    significant digits; ``repeating`` is the repeating decimal of a fraction,
    such as 0.(3) for 1/3.
    """

    expression: str
    reading: str
    value: str
    exact: bool
    decimal: str
    decimal_exact: bool
    digits: int
    repeating: str | None = None
    fraction: str | None = None
    steps: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        """One line: the expression as read, its value, and its decimal when that differs."""
        shown_already = {self.reading}
        line = self.reading
        if self.value not in shown_already:
            line += f" {'=' if self.exact else '≈'} {self.value}"
            shown_already.add(self.value)
        if self.fraction and self.fraction not in shown_already:
            line += f" = {self.fraction}"
            shown_already.add(self.fraction)
        if self.exact and self.decimal not in shown_already:
            line += f" = {self.repeating}" if self.repeating else \
                f" {'=' if self.decimal_exact else '≈'} {self.decimal}"
        return line

    def __str__(self) -> str:
        lines = [self.answer]
        if not self.exact:
            lines.append(f"Approximate: rounded to {self.digits} significant digits.")
        if self.steps:
            lines.append("Steps:")
            lines += [f"  {step}" for step in self.steps]
        lines += [f"Assumption: {assumption}" for assumption in self.assumptions]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "status": "computed",
            "expression": self.expression,
            "reading": self.reading,
            "answer": self.answer,
            "value": self.value,
            "exact": self.exact,
            "decimal": self.decimal,
            "decimal_exact": self.decimal_exact,
            "fraction": self.fraction,
            "repeating": self.repeating,
            "digits": self.digits,
            "steps": list(self.steps),
            "assumptions": list(self.assumptions),
        }


def compute(expression: str, digits: int = DIGITS) -> Computation | Refusal:
    """Evaluate an arithmetic expression, exactly wherever the value is rational, and show each step.

    ``digits`` sets the significant digits of an approximate result and of the
    decimal shown for a fraction whose decimal does not end. Returns a
    :class:`Computation`, or a ``Refusal`` with the reason when the expression
    cannot be computed honestly.
    """
    if not isinstance(expression, str) or not expression.strip():
        return Refusal("invalid_expression", "The expression is empty.")
    if len(expression) > MAX_LENGTH:
        return Refusal("too_large", f"The expression is longer than {MAX_LENGTH:,} characters.")
    if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= MAX_DIGITS_ASKED:
        return Refusal("invalid_expression", f"digits must be a whole number from 1 to {MAX_DIGITS_ASKED}.")
    written = expression.strip()
    # "2 + 2 =" and "2 + 2?" are the expression 2 + 2.
    text = written.rstrip("=?").strip() or written
    try:
        tokens, assumptions = _tokens(text)
        loose = _Parser(tokens, "loose")
        tree = loose.parse()
        assumptions += loose.assumptions
        number, evaluator = _evaluate(tree, digits)
        if loose.used_implicit:
            tight = _Parser(tokens, "tight")
            other = tight.parse()
            second, _ = _evaluate(other, digits)
            if not _same(number, second, digits):
                _refuse("ambiguous_expression",
                        f"{text} has a multiplication with no sign, so it reads two ways: {show(tree)} = "
                        f"{shown(number, digits)}, or {show(other)} = {shown(second, digits)}. Write the * "
                        "and the parentheses meant.")
            assumptions.append(f"A multiplication sign is read where none is written: {show(tree)}.")
    except _Refused as refused:
        return refused.refusal
    except (OverflowError, DecimalOverflow):
        return Refusal("too_large", f"The value of {text} is too large to compute.")
    except (ArithmeticError, ValueError) as error:
        return Refusal("invalid_expression", f"ADA could not compute {text}: {error}.")
    assumptions += [item for item in evaluator.assumptions if item not in assumptions]
    steps = list(evaluator.steps)
    if evaluator.extra_steps:
        steps.append(f"… and {evaluator.extra_steps} more steps.")
    return _result(written, show(tree), number, digits, steps, assumptions)


def _evaluate(tree: _Node, digits: int) -> tuple[Number, _Evaluator]:
    evaluator = _Evaluator(digits)
    with localcontext() as context:
        context.prec = digits + GUARD_DIGITS
        context.Emax, context.Emin = 10 ** 8, -(10 ** 8)
        number = _check_size(evaluator.value(tree))
    return number, evaluator


def _same(first: Number, second: Number, digits: int) -> bool:
    if isinstance(first, Fraction) and isinstance(second, Fraction):
        return first == second
    return (_decimal_text(_approx_wide(first, digits), digits)
            == _decimal_text(_approx_wide(second, digits), digits))


def _result(expression: str, reading: str, number: Number, digits: int, steps: list[str],
            assumptions: list[str]) -> Computation:
    if isinstance(number, Decimal):
        text = _decimal_text(number, digits)
        return Computation(expression, reading, text, False, text, False, digits, steps=steps,
                           assumptions=assumptions)
    fraction = _fraction_text(number, digits)
    if number.denominator == 1:
        decimal = fraction if len(str(abs(number.numerator))) <= MAX_SHOWN else \
            _decimal_text(_approx_wide(number, digits), digits)
        return Computation(expression, reading, fraction, True, decimal, decimal == fraction, digits,
                           steps=steps, assumptions=assumptions)
    if _terminates(number):
        decimal = _exact_decimal(number)
        if len(decimal.replace("-", "").replace(".", "").lstrip("0")) <= 60:
            return Computation(expression, reading, decimal, True, decimal, True, digits, fraction=fraction,
                               steps=steps, assumptions=assumptions)
    decimal = _decimal_text(_approx_wide(number, digits), digits)
    return Computation(expression, reading, fraction, True, decimal, False, digits, repeating(number),
                       fraction, steps, assumptions)


# --- Checking written arithmetic ------------------------------------------------------------------------

#: A number as people write it in a worked answer: 1,234.5 or 0.25. Commas only between groups of three.
_WRITTEN = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d|,\d)|\d+(?:\.\d+)?"
#: One piece of a line of a worked answer. Words and other signs end an expression.
_TOKEN = re.compile(rf"(?P<number>{_WRITTEN})|(?P<space>[ \t]+)|(?P<op>\*\*|[-+*/×÷^−])|(?P<open>\()"
                    r"|(?P<close>\))|(?P<percent>%)|(?P<equals>=)|(?P<word>[^\W\d_]+)|(?P<other>.)")
_ARITHMETIC = {"number", "space", "op", "open", "close", "percent", "equals"}
#: How worked answers write operators and numbers: LaTeX, signs, currency.
_WRITTEN_OPERATORS = (("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"), ("\\%", "%"), ("\\left(", "("),
                      ("\\right)", ")"), ("\\(", ""), ("\\)", ""), ("$", ""), ("€", ""), ("£", ""),
                      ("·", "*"), ("²", "^2"), ("³", "^3"))
#: Markdown bold (**15**), kept as a power only between numbers (2**3).
_BOLD = re.compile(r"(?<![\d)])\*\*|\*\*(?![\d(])")
#: A list marker at the start of a line: "- ", "* ", "1. ", "2) ".
_BULLET = re.compile(r"^[ \t]*(?:[-*+•]|\d+[.)])[ \t]+")
#: Words after a number that scale it: "10 million".
_SCALE_WORDS = {"hundred", "thousand", "million", "billion", "trillion", "hundreds", "thousands", "millions",
                "billions"}
#: Words before an equation that make it a condition rather than a claim.
_CONDITION_WORDS = {"if", "unless", "whether"}


@dataclass(frozen=True)
class Checked:
    """One equation written in a text, with ADA's value for its left side.

    ``correct`` is true when the stated value is the computed one, or the computed one rounded or cut off to
    as many decimal places as the stated value has: for 22/3, 7.33, 7.3 and 7 are correct; 7.34 is not.
    """

    expression: str
    stated: str
    computed: str
    correct: bool

    def to_dict(self) -> dict:
        return {"expression": self.expression, "stated": self.stated, "computed": self.computed,
                "correct": self.correct}


def check_arithmetic(text: str) -> list[Checked]:
    """Find each equation written in ``text`` ("204 + 160 + 330 = 700") and check it with exact arithmetic.

    Only operations on written numbers are checked, and only where the whole operation is written out:

    - An equation is read within one line.
    - One whose left side goes on to the left (``S + B + 2 + 2 = 28``, ``16 friends * 3/4 = 12``) is
      skipped, as is one whose stated value goes on into a longer expression (``15 - 5 = 2x``).
    - In a chain (``5,000 * (1 + 0.025) = 5,000 * 1.025 = 5,125``) the last step is checked, since a running
      total (``3 * 4 = 12 + 1 = 13``) is written the same way.
    - A number with a misplaced comma (``373, 846.92``) and an expression ADA refuses or cannot read are
      skipped.

    Use it to check a worked answer a language model wrote: an equation that comes back with ``correct``
    false is an arithmetic slip, with the right value beside it.
    """
    plain = text
    for written, operator in _WRITTEN_OPERATORS:
        plain = plain.replace(written, operator)
    plain = _BOLD.sub("", plain)
    checked = []
    for line in plain.splitlines():
        tokens = [[match.lastgroup, match.group()] for match in _TOKEN.finditer(_BULLET.sub("", line))]
        _read_times_signs(tokens)
        start = 0
        while start < len(tokens):
            if tokens[start][0] not in _ARITHMETIC:
                start += 1
                continue
            end = start
            while end < len(tokens) and tokens[end][0] in _ARITHMETIC:
                end += 1
            item = _checked_run(tokens, start, end)
            if item is not None:
                checked.append(item)
            start = end
    return checked


def _read_times_signs(tokens: list[list[str]]) -> None:
    """The letter x between two numbers (3 x 4) is a multiplication sign."""
    for index, (kind, value) in enumerate(tokens):
        if kind != "word" or value not in ("x", "X"):
            continue
        before = [token for token in tokens[:index] if token[0] != "space"]
        after = [token for token in tokens[index + 1:] if token[0] != "space"]
        if before and after and before[-1][0] in ("number", "close") and after[0][0] in ("number", "open"):
            tokens[index][0] = "op"


def _checked_run(tokens: list[list[str]], start: int, end: int) -> Checked | None:
    """Check the last step of the equation in ``tokens[start:end]``, a run of numbers and operators."""
    run = tokens[start:end]
    following = tokens[end:]
    # "$1,500/month": a rate, not a division by what follows.
    rate = (run[-1][0] == "op" and run[-1][1] == "/" and bool(following) and following[0][0] == "word"
            and len(following[0][1]) > 1)
    if rate:
        run = run[:-1]
    # "= 2x" or "= 12km": the stated value runs into a name.
    if not rate and following and following[0][0] == "word" and run[-1][0] == "number":
        return None
    # "= 373, 846.92" or "= 583,0694": a number with a misplaced comma.
    if following and following[0][1] == "," and len(following) > 1:
        attached = following[1][0] == "number"
        spaced = (following[1][0] == "space" and len(following) > 2 and following[2][0] == "number"
                  and re.fullmatch(r"\d{3}(?:\.\d+)?", following[2][1]))
        if attached or spaced:
            return None
    # "= 16 friends * 3/4": the stated value starts a longer expression.
    rest = [token for token in following if token[0] != "space"]
    if len(rest) > 1 and rest[0][0] == "word" and rest[1][0] == "op":
        return None
    # "= 10 million": the value is scaled by a word.
    if rest and rest[0][1].lower() in _SCALE_WORDS:
        return None
    # "only possible if 240 + 3,516,571 = 0": a condition, not a claim.
    if start > 0 and tokens[start - 1][0] == "word" and tokens[start - 1][1].lower() in _CONDITION_WORDS:
        return None
    terms = [[]]
    for token in run:
        if token[0] == "equals":
            terms.append([])
        else:
            terms[-1].append(token)
    terms = [_strip_spaces(term) for term in terms]
    if len(terms) < 2:
        return None
    left, stated = terms[-2], terms[-1]
    # "= 12 (the apples)": an aside after the value.
    while stated and stated[-1][0] == "open":
        stated = _strip_spaces(stated[:-1])
    stated = _drop_unmatched(stated, "close")
    if not _is_number(stated):
        return None
    if len(terms) == 2:
        # The first expression of a run must start it: not "B + 2 + 2" or "x2 + 3".
        before = tokens[start - 1] if start > 0 else None
        if before is not None and before[0] == "word" and run[0][0] != "space":
            return None
        if before is not None and before[1] in ".,_\\" and run[0][0] != "space":
            return None
        if left and left[0][0] == "op":
            unary = (left[0][1] in "-−" and len(left) > 1 and left[1][0] in ("number", "open")
                     and (before is None or before[0] not in ("word", "number")))
            if not unary:
                return None
        left = _drop_unmatched(left, "open")
    if not any(token[0] == "op" for token in left):
        return None
    written = "".join(token[1] for token in left)
    if written.count("(") != written.count(")"):
        return None
    expression = "".join("*" if token[0] == "op" and token[1] in "xX" else token[1] for token in left)
    result = compute(expression.replace(",", ""))
    if isinstance(result, Refusal):
        return None
    shown = "".join(token[1] for token in stated)
    number = next(token[1] for token in stated if token[0] == "number").replace(",", "")
    claimed = Fraction(Decimal(number)) * (-1 if stated[0][1] in "-−" else 1)
    places = len(number.split(".")[1]) if "." in number else 0
    value = _exact_value(result)
    # "1/4 = 25%" and "0.25 * 100 = 25%" are both written; either reading is accepted.
    readings = [(claimed, places)] + ([(claimed / 100, places + 2)] if stated[-1][0] == "percent" else [])
    correct = any(_written_as(value, reading, shown_places) for reading, shown_places in readings)
    return Checked(written.strip(), shown, result.value, correct)


def _written_as(value: Fraction, claimed: Fraction, places: int) -> bool:
    """Whether ``claimed`` is ``value`` rounded or cut off to ``places`` decimal places."""
    if abs(value - claimed) <= Fraction(1, 2) / 10 ** places:
        return True
    scaled = value * 10 ** places
    cut = math.floor(scaled) if scaled >= 0 else math.ceil(scaled)
    return Fraction(cut, 10 ** places) == claimed


def _strip_spaces(term: list[list[str]]) -> list[list[str]]:
    while term and term[0][0] == "space":
        term = term[1:]
    while term and term[-1][0] == "space":
        term = term[:-1]
    return term


def _drop_unmatched(term: list[list[str]], kind: str) -> list[list[str]]:
    """Drop a parenthesis opened before the equation or closed after it: "(so 3 * 4 = 12)"."""
    opened = sum(token[0] == "open" for token in term)
    closed = sum(token[0] == "close" for token in term)
    if kind == "open":
        while term and term[0][0] in ("open", "space") and opened > closed:
            opened -= term[0][0] == "open"
            term = term[1:]
    else:
        while term and term[-1][0] in ("close", "space") and closed > opened:
            closed -= term[-1][0] == "close"
            term = term[:-1]
    return term


def _is_number(term: list[list[str]]) -> bool:
    """A written value: a number, with a minus sign before it or a percent sign after it."""
    kinds = [token[0] for token in term]
    if kinds and kinds[0] == "op":
        if term[0][1] not in "-−":
            return False
        kinds = kinds[1:]
    if kinds and kinds[-1] == "percent":
        kinds = kinds[:-1]
    return kinds == ["number"]


def _exact_value(result: Computation) -> Fraction:
    if result.fraction:
        return Fraction(result.fraction)
    return Fraction(Decimal(result.decimal))
