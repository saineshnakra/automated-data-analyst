"""Algebra with exact fractions: expressions with letters simplified, and equations solved step by step.

    import ada
    ada.simplify("3(x + 2) - x")                  # 2x + 6
    ada.solve("2x + 3 = 11")                      # x = 4
    ada.solve("x + y = 10; x - y = 2")            # x = 6, y = 4
    ada.solve("d = r*t", unknowns="t")            # t = d/r, if r is not 0
    ada.solve("x*y = 6; x + y = 5")               # (x = 2, y = 3) or (x = 3, y = 2)
    ada.check_solution("2x + 3 = 11", {"x": 4})   # 2(4) + 3 = 11 holds

A letter is a variable: one letter, such as x or y, with e kept as the constant. Coefficients are fractions,
so nothing is rounded. Every solution is put back into the equations it came from, and that check is the last
step. Equations linear in the unknowns (each to the first power, no two multiplied together) may have other
letters in their coefficients, as in a formula solved for one of its letters; equations with powers of the
unknowns go to ``roots.py`` (one unknown) or ``systems.py`` (several).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .api import Refusal
from .compute import (
    CONSTANTS,
    DIGITS,
    FUNCTION_NAMES,
    MAX_DEPTH,
    MAX_LENGTH,
    MAX_STEPS,
    _approx_wide,
    _check_size,
    _decimal_text,
    _Evaluator,
    _exact_decimal,
    _Node,
    _Parser,
    _refuse,
    _Refused,
    _terminates,
    _tokens,
    compute,
    show,
)

#: The most terms an expression may have once multiplied out.
MAX_TERMS = 2000
#: The highest power of the letters in an expression.
MAX_DEGREE = 200
#: The most equations ``solve`` takes at once.
MAX_EQUATIONS = 50
#: The highest power at which a common factor of a quotient in one letter is looked for; above it the
#: fractions in Euclid's algorithm grow too long to be worth it.
GCD_DEGREE = 60

Monomial = tuple  # ((letter, power), ...) sorted by letter; () for a number


# --- Polynomials and quotients of them ------------------------------------------------------------------


class Polynomial:
    """A sum of terms, each a fraction times a product of letters to whole powers."""

    __slots__ = ("terms",)

    def __init__(self, terms: dict | None = None):
        self.terms = {monomial: value for monomial, value in (terms or {}).items() if value != 0}

    @staticmethod
    def number(value: Fraction) -> Polynomial:
        return Polynomial({(): Fraction(value)})

    @staticmethod
    def letter(name: str) -> Polynomial:
        return Polynomial({((name, 1),): Fraction(1)})

    def is_zero(self) -> bool:
        return not self.terms

    def is_number(self) -> bool:
        return all(monomial == () for monomial in self.terms)

    def value(self) -> Fraction:
        """The number this polynomial is, when it has no letters."""
        return self.terms.get((), Fraction(0))

    def letters(self) -> tuple[str, ...]:
        return tuple(sorted({name for monomial in self.terms for name, _ in monomial}))

    def degree(self, letters: tuple[str, ...] | None = None) -> int:
        """The highest total power, counting only ``letters`` when given."""
        return max((sum(power for name, power in monomial if letters is None or name in letters)
                    for monomial in self.terms), default=0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Polynomial) and self.terms == other.terms

    def __hash__(self) -> int:
        return hash(frozenset(self.terms.items()))

    def __neg__(self) -> Polynomial:
        return Polynomial({monomial: -value for monomial, value in self.terms.items()})

    def __add__(self, other: Polynomial) -> Polynomial:
        terms = dict(self.terms)
        for monomial, value in other.terms.items():
            terms[monomial] = terms.get(monomial, 0) + value
        return _checked(Polynomial(terms))

    def __sub__(self, other: Polynomial) -> Polynomial:
        return self + (-other)

    def __mul__(self, other: Polynomial) -> Polynomial:
        return _checked(self.times(other))

    def times(self, other: Polynomial) -> Polynomial:
        """The product, without the limits on its size: for a comparison that is thrown away."""
        if len(self.terms) * len(other.terms) > MAX_TERMS * 50:
            _refuse("too_large", f"Multiplying out would make more than {MAX_TERMS:,} terms.")
        terms: dict = {}
        for first, a in self.terms.items():
            for second, b in other.terms.items():
                monomial = _times(first, second)
                terms[monomial] = terms.get(monomial, 0) + a * b
        return Polynomial(terms)

    def scaled(self, factor: Fraction) -> Polynomial:
        return _checked(Polynomial({monomial: value * factor for monomial, value in self.terms.items()}))

    def power(self, n: int) -> Polynomial:
        if n == 0:
            return Polynomial.number(Fraction(1))
        if self.degree() * n > MAX_DEGREE:
            _refuse("too_large", f"The power would reach {self.degree() * n}; ADA keeps powers up to "
                                 f"{MAX_DEGREE}.")
        count = len(self.terms)
        if count > 1 and math.comb(n + count - 1, count - 1) > MAX_TERMS * 10:
            _refuse("too_large", f"Multiplying out would make more than {MAX_TERMS:,} terms.")
        result, base = Polynomial.number(Fraction(1)), self
        while n:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def substituted(self, values: dict[str, Quotient]) -> Quotient:
        """This polynomial with each letter in ``values`` replaced by its value."""
        total = Quotient.number(Fraction(0))
        for monomial, coefficient in self.terms.items():
            term = Quotient.number(coefficient)
            for name, power in monomial:
                factor = values.get(name) or Quotient(Polynomial.letter(name))
                term = term * factor.power(power)
            total = total + term
        return total

    def __str__(self) -> str:
        return _polynomial_text(self)


def _times(first: Monomial, second: Monomial) -> Monomial:
    powers = dict(first)
    for name, power in second:
        powers[name] = powers.get(name, 0) + power
    return tuple(sorted(powers.items()))


def _checked(polynomial: Polynomial) -> Polynomial:
    if len(polynomial.terms) > MAX_TERMS:
        _refuse("too_large", f"The expression has more than {MAX_TERMS:,} terms once multiplied out.")
    if polynomial.degree() > MAX_DEGREE:
        _refuse("too_large", f"The expression reaches a power of {polynomial.degree()}; ADA keeps powers "
                             f"up to {MAX_DEGREE}.")
    for value in polynomial.terms.values():
        _check_size(value)
    return polynomial


class Quotient:
    """A polynomial divided by a polynomial, kept in lowest terms where ADA can tell them.

    Whole-number factors and powers of letters common to the top and bottom are
    cancelled, and so is any common factor when only one letter appears. A
    common factor in two or more letters, as in (x^2 - y^2)/(x - y), or above the
    power ``GCD_DEGREE``, is left.
    """

    __slots__ = ("numerator", "denominator")

    def __init__(self, numerator: Polynomial, denominator: Polynomial | None = None):
        denominator = denominator if denominator is not None else Polynomial.number(Fraction(1))
        if denominator.is_zero():
            _refuse("division_by_zero", "The expression divides by zero, which has no value.")
        self.numerator, self.denominator = _lowest_terms(numerator, denominator)

    @staticmethod
    def number(value: Fraction) -> Quotient:
        return Quotient(Polynomial.number(value))

    def is_polynomial(self) -> bool:
        return self.denominator.is_number()

    def is_number(self) -> bool:
        return self.numerator.is_number() and self.denominator.is_number()

    def value(self) -> Fraction:
        return self.numerator.value() / self.denominator.value()

    def is_zero(self) -> bool:
        return self.numerator.is_zero()

    def letters(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.numerator.letters()) | set(self.denominator.letters())))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quotient):
            return False
        if self.numerator == other.numerator and self.denominator == other.denominator:
            return True
        return self.numerator.times(other.denominator) == other.numerator.times(self.denominator)

    def __hash__(self) -> int:
        return hash(str(self))

    def __neg__(self) -> Quotient:
        return Quotient(-self.numerator, self.denominator)

    def __add__(self, other: Quotient) -> Quotient:
        if self.denominator == other.denominator:
            return Quotient(self.numerator + other.numerator, self.denominator)
        return Quotient(self.numerator * other.denominator + other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __sub__(self, other: Quotient) -> Quotient:
        return self + (-other)

    def __mul__(self, other: Quotient) -> Quotient:
        return Quotient(self.numerator * other.numerator, self.denominator * other.denominator)

    def __truediv__(self, other: Quotient) -> Quotient:
        if other.is_zero():
            _refuse("division_by_zero", f"Dividing by {other} divides by zero, which has no value.")
        return Quotient(self.numerator * other.denominator, self.denominator * other.numerator)

    def power(self, n: int) -> Quotient:
        if n >= 0:
            return Quotient(self.numerator.power(n), self.denominator.power(n))
        if self.is_zero():
            _refuse("division_by_zero", f"({self})^{n} divides by zero, which has no value.")
        return Quotient(self.denominator.power(-n), self.numerator.power(-n))

    def __str__(self) -> str:
        top = str(self.numerator)
        if self.is_polynomial():
            return top
        bottom = str(self.denominator)
        if len(self.numerator.terms) > 1:
            top = f"({top})"
        # 1/x^2 reads back as written; 1/x*y would read as y/x, so it is written 1/(x*y).
        if len(self.denominator.terms) > 1 or not re.fullmatch(r"[A-Za-z](\^\d+)?|\d+", bottom):
            bottom = f"({bottom})"
        return f"{top}/{bottom}"


def _lowest_terms(numerator: Polynomial, denominator: Polynomial) -> tuple[Polynomial, Polynomial]:
    if numerator.is_zero():
        return numerator, Polynomial.number(Fraction(1))
    if denominator.is_number():
        return numerator.scaled(1 / denominator.value()), Polynomial.number(Fraction(1))
    # Powers of letters common to every term, top and bottom.
    common = None
    for monomial in list(numerator.terms) + list(denominator.terms):
        powers = dict(monomial)
        common = powers if common is None else {name: min(power, powers.get(name, 0))
                                                 for name, power in common.items() if powers.get(name)}
    if common:
        numerator, denominator = _divide_monomial(numerator, common), _divide_monomial(denominator, common)
    letters = set(numerator.letters()) | set(denominator.letters())
    if len(letters) == 1 and max(numerator.degree(), denominator.degree()) <= GCD_DEGREE:
        divisor = _gcd_one_letter(numerator, denominator, next(iter(letters)))
        if divisor.degree() > 0:
            numerator = _divide_exactly(numerator, divisor, next(iter(letters)))
            denominator = _divide_exactly(denominator, divisor, next(iter(letters)))
    if denominator.is_number():
        return numerator.scaled(1 / denominator.value()), Polynomial.number(Fraction(1))
    # Whole numbers on top and bottom, with no common factor and a positive first term below.
    scale = _content(numerator) / _content(denominator)
    numerator = numerator.scaled(1 / _content(numerator))
    denominator = denominator.scaled(1 / _content(denominator))
    if _leading(denominator)[1] < 0:
        numerator, denominator = -numerator, -denominator
    numerator = numerator.scaled(scale.numerator)
    denominator = denominator.scaled(scale.denominator)
    return numerator, denominator


def _divide_monomial(polynomial: Polynomial, powers: dict) -> Polynomial:
    terms = {}
    for monomial, value in polynomial.terms.items():
        left = tuple((name, power - powers.get(name, 0)) for name, power in monomial
                     if power - powers.get(name, 0))
        terms[left] = value
    return Polynomial(terms)


def _content(polynomial: Polynomial) -> Fraction:
    """The positive fraction that leaves whole-number coefficients with no common factor."""
    values = list(polynomial.terms.values())
    top = math.gcd(*(value.numerator for value in values))
    bottom = math.lcm(*(value.denominator for value in values))
    return Fraction(top, bottom)


def _order(polynomial: Polynomial):
    letters = polynomial.letters()
    return lambda monomial: (-sum(power for _, power in monomial),
                             tuple(-dict(monomial).get(name, 0) for name in letters))


def _leading(polynomial: Polynomial) -> tuple[Monomial, Fraction]:
    monomial = min(polynomial.terms, key=_order(polynomial))
    return monomial, polynomial.terms[monomial]


def _one_letter_coefficients(polynomial: Polynomial, letter: str) -> dict[int, Fraction]:
    return {dict(monomial).get(letter, 0): value for monomial, value in polynomial.terms.items()}


def _from_coefficients(coefficients: dict[int, Fraction], letter: str) -> Polynomial:
    return Polynomial({((letter, power),) if power else (): value for power, value in coefficients.items()})


def _divmod_one_letter(top: Polynomial, bottom: Polynomial, letter: str) -> tuple[Polynomial, Polynomial]:
    remainder = _one_letter_coefficients(top, letter)
    divisor = _one_letter_coefficients(bottom, letter)
    high = max(divisor)
    quotient: dict[int, Fraction] = {}
    while remainder and max(remainder) >= high:
        power = max(remainder)
        factor = remainder[power] / divisor[high]
        quotient[power - high] = factor
        for degree, value in divisor.items():
            remainder[degree + power - high] = remainder.get(degree + power - high, 0) - factor * value
        remainder = {degree: value for degree, value in remainder.items() if value != 0}
    return _from_coefficients(quotient, letter), _from_coefficients(remainder, letter)


def _gcd_one_letter(first: Polynomial, second: Polynomial, letter: str) -> Polynomial:
    while not second.is_zero():
        _, remainder = _divmod_one_letter(first, second, letter)
        first, second = second, remainder
    return first


def _divide_exactly(top: Polynomial, bottom: Polynomial, letter: str) -> Polynomial:
    quotient, _ = _divmod_one_letter(top, bottom, letter)
    return quotient


def _number_text(value: Fraction) -> str:
    """A number as ADA writes it: whole, a decimal that ends in a few digits, else a fraction."""
    if value.denominator == 1:
        return str(value.numerator)
    if _terminates(value):
        decimal = _exact_decimal(value)
        if len(decimal.replace("-", "").replace(".", "").lstrip("0")) <= 12:
            return decimal
    return f"{value.numerator}/{value.denominator}"


def _polynomial_text(polynomial: Polynomial) -> str:
    if polynomial.is_zero():
        return "0"
    text = ""
    for monomial in sorted(polynomial.terms, key=_order(polynomial)):
        value = polynomial.terms[monomial]
        factors = "*".join(name if power == 1 else f"{name}^{power}" for name, power in monomial)
        size = abs(value)
        if not monomial:
            body = _number_text(size)
        elif size == 1:
            body = factors
        else:
            number = _number_text(size)
            body = f"({number}){factors}" if "/" in number else f"{number}{factors}"
        if not text:
            text = f"-{body}" if value < 0 else body
        else:
            text += f" {'-' if value < 0 else '+'} {body}"
    return text


# --- Reading expressions with letters -------------------------------------------------------------------


class _Algebra:
    """Turns a parsed expression into a quotient of polynomials, writing down each step that does work."""

    def __init__(self, digits: int, bindings: dict[str, Fraction] | None = None):
        self.digits = digits
        self.bindings = bindings or {}
        self.steps: list[str] = []
        self.extra_steps = 0
        self.assumptions: list[str] = []
        #: Each expression with a letter that the expression divides by, which must not be 0.
        self.denominators: list[Polynomial] = []
        self.arithmetic = _Evaluator(digits)

    def step(self, text: str) -> None:
        if len(self.steps) < MAX_STEPS:
            self.steps.append(text)
        else:
            self.extra_steps += 1

    def value(self, node: _Node) -> Quotient:
        kind = node.kind
        if kind == "number":
            return Quotient.number(node.value)
        if kind == "variable":
            if node.value in self.bindings:
                return Quotient.number(self.bindings[node.value])
            return Quotient(Polynomial.letter(node.value))
        if kind == "group":
            return self.value(node.children[0])
        if kind == "negative":
            return -self.value(node.children[0])
        if kind == "percent":
            return self.value(node.children[0]) / Quotient.number(Fraction(100))
        if kind == "constant":
            _refuse("invalid_expression",
                    f"{node.value} is not a fraction, and ADA's algebra keeps every coefficient exact. Put "
                    "in its value to the digits needed, or compute the rest without it.")
        if kind in ("factorial", "call"):
            arguments = [self.value(child) for child in node.children]
            if not all(argument.is_number() for argument in arguments):
                _refuse("unsupported_expression",
                        f"{show(node)} applies a function to a letter. ADA's algebra adds, subtracts, "
                        "multiplies and divides letters, and raises them to whole powers.")
            numbers = [argument.value() for argument in arguments]
            if kind == "factorial":
                factorial = _Node("factorial", None, (_number_node(numbers[0]),))
                return self.exact(self.arithmetic.value(factorial), show(node))
            return self.exact(self.arithmetic.call(node.value, numbers), show(node))
        if node.value == "^":
            return self.power(node, self.value(node.children[0]), self.value(node.children[1]))
        spine = []
        while node.kind == "binary" and node.value != "^":
            spine.append(node)
            node = node.children[0]
        total = self.value(node)
        for link in reversed(spine):
            right = link.children[1]
            if link.value in ("+", "-") and right.kind == "percent":
                rate = show(right)
                _refuse("ambiguous_expression",
                        f"{show(link)} reads two ways: {rate} as the number {show(right.children[0])}/100, "
                        f"or {rate} of what comes before it. Write /100 for the first, or multiply by 1 "
                        f"{link.value} {show(right.children[0])}/100 for the second.")
            total = self.binary(link.value, total, self.value(right), link)
        return total

    def exact(self, number, written: str) -> Quotient:
        if not isinstance(number, Fraction):
            _refuse("invalid_expression",
                    f"{written} is not a fraction, and ADA's algebra keeps every coefficient exact.")
        return Quotient.number(number)

    def binary(self, operator: str, left: Quotient, right: Quotient, node: _Node) -> Quotient:
        if operator in ("//", "mod"):
            if not (left.is_number() and right.is_number()):
                _refuse("unsupported_expression",
                        f"{show(node)} uses {operator} with a letter. ADA's algebra adds, subtracts, "
                        "multiplies and divides letters, and raises them to whole powers.")
            return self.exact(self.arithmetic.binary(operator, left.value(), right.value()), show(node))
        if operator in ("+", "-"):
            result = left + right if operator == "+" else left - right
        elif operator in ("*", "implicit", "of"):
            result = left * right
        else:
            if right.is_zero():
                _refuse("division_by_zero", f"{show(node)} divides by zero, which has no value.")
            if not right.is_number():
                self.denominators.append(right.numerator)
            result = left / right
        symbol = {"implicit": "*", "of": "*"}.get(operator, operator)
        if _worth_a_step(operator, left, right, result):
            self.step(f"{_operand(left)} {symbol} {_operand(right)} = {result}")
        return result

    def power(self, node: _Node, base: Quotient, exponent: Quotient) -> Quotient:
        if not exponent.is_number():
            _refuse("unsupported_expression",
                    f"{show(node)} has a letter in the power. ADA's algebra raises letters to whole powers "
                    "only.")
        n = exponent.value()
        if base.is_number():
            return self.exact(self.arithmetic.power(base.value(), n), show(node))
        if n.denominator != 1:
            _refuse("unsupported_expression",
                    f"{show(node)} raises a letter to the power {n}. ADA's algebra raises letters to whole "
                    "powers only.")
        if n < 0:
            self.denominators.append(base.numerator)
        result = base.power(n.numerator)
        if len(base.numerator.terms) > 1 or len(base.denominator.terms) > 1:
            self.step(f"{_operand(base)}^{n.numerator} = {result}")
        return result


def _number_node(value: Fraction) -> _Node:
    return _Node("number", value, text=str(value))


def _operand(value: Quotient) -> str:
    text = str(value)
    if (len(value.numerator.terms) > 1 and value.is_polynomial()) or text.startswith("-") or \
            (not value.is_polynomial() and not text.startswith("(")):
        return f"({text})"
    return text


def _worth_a_step(operator: str, left: Quotient, right: Quotient, result: Quotient) -> bool:
    """Whether an operation does work worth writing down: multiplying out, collecting like terms, or
    arithmetic on two numbers. Writing 3 next to x, or adding unlike terms, is not."""
    if left.is_number() and right.is_number():
        return True
    if operator in ("+", "-"):
        sizes = len(left.numerator.terms) + len(right.numerator.terms)
        return not (left.is_polynomial() and right.is_polynomial() and len(result.numerator.terms) == sizes)
    if operator in ("*", "implicit", "of"):
        return len(left.numerator.terms) > 1 or len(right.numerator.terms) > 1 or \
            not (left.is_polynomial() and right.is_polynomial())
    if right.is_number():
        return len(left.numerator.terms) > 1
    # x/(x + 1) is already as simple as it gets; (x^2 - 1)/(x - 1) = x + 1 is not.
    return str(result) != f"{_wrapped(left)}/{_wrapped(right)}"


def _wrapped(value: Quotient) -> str:
    text = str(value)
    return text if len(value.numerator.terms) == 1 and value.is_polynomial() else f"({text})"


def _read(text: str, digits: int,
          bindings: dict[str, Fraction] | None = None) -> tuple[Quotient, _Algebra, str]:
    """Read one expression with letters; refuse it when a missing multiplication sign changes its value."""
    if not text.strip():
        _refuse("invalid_expression", "An expression is empty.")
    tokens, assumptions = _tokens(text)
    loose = _Parser(tokens, "loose", variables=True)
    tree = loose.parse()
    algebra = _Algebra(digits, bindings)
    algebra.assumptions += assumptions + loose.assumptions
    value = algebra.value(tree)
    if loose.used_implicit:
        tight = _Parser(tokens, "tight", variables=True)
        other = tight.parse()
        second = _Algebra(digits, bindings).value(other)
        if not value == second:
            _refuse("ambiguous_expression",
                    f"{text.strip()} has a multiplication with no sign, so it reads two ways: {show(tree)} = "
                    f"{value}, or {show(other)} = {second}. Write the * and the parentheses meant.")
    return value, algebra, show(tree)


def _letters_in(text: str) -> set[str]:
    try:
        tokens, _ = _tokens(text)
    except _Refused:
        return set()
    return {token.text for token in tokens if token.kind == "name" and len(token.text) == 1
            and token.text.lower() not in CONSTANTS and token.text.lower() not in FUNCTION_NAMES}


# --- Simplifying ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Simplified:
    """An expression with letters, multiplied out with like terms collected.

    ``conditions`` lists each expression it divides by, which must not be 0
    for the expression to have a value.
    """

    expression: str
    reading: str
    result: str
    letters: list[str]
    conditions: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def answer(self) -> str:
        line = self.expression if self.expression == self.result else f"{self.expression} = {self.result}"
        return line + (f", when {', '.join(self.conditions)}" if self.conditions else "")

    def __str__(self) -> str:
        lines = [self.answer]
        if self.steps:
            lines.append("Steps:")
            lines += [f"  {step}" for step in self.steps]
        lines += [f"Assumption: {assumption}" for assumption in self.assumptions]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "status": "simplified", "expression": self.expression,
                "reading": self.reading, "answer": self.answer, "result": self.result,
                "letters": list(self.letters), "conditions": list(self.conditions), "steps": list(self.steps),
                "assumptions": list(self.assumptions)}


def simplify(expression: str, digits: int = DIGITS) -> Simplified | Refusal:
    """Multiply out an expression with letters and collect like terms, exactly, showing each step.

    ``ada.simplify("3(x + 2) - x")`` is 2x + 6. A quotient is kept in lowest
    terms where ADA can tell them. Returns a ``Refusal`` for what it cannot
    read or what reads two ways, as ``compute`` does.
    """
    refusal = _check_text(expression)
    if refusal:
        return refusal
    text = expression.strip()
    try:
        value, algebra, reading = _read(text, digits)
    except _Refused as refused:
        return refused.refusal
    except (OverflowError, RecursionError):
        return Refusal("too_large", f"{text} is too large to work out.")
    steps = list(algebra.steps)
    if algebra.extra_steps:
        steps.append(f"… and {algebra.extra_steps} more steps.")
    return Simplified(text, reading, str(value), list(value.letters()),
                      _conditions(algebra.denominators), steps, _unique(algebra.assumptions))


def _check_text(text: Any) -> Refusal | None:
    if not isinstance(text, str) or not text.strip():
        return Refusal("invalid_expression", "The expression is empty.")
    if len(text) > MAX_LENGTH:
        return Refusal("too_large", f"The expression is longer than {MAX_LENGTH:,} characters.")
    return None


def _conditions(denominators: list[Polynomial]) -> list[str]:
    conditions: list[str] = []
    for denominator in denominators:
        conditions += _not_zero_all(denominator)
    return _unique(conditions)


def _not_zero_all(polynomial: Polynomial) -> list[str]:
    """What must not be 0 for a divisor in one letter: each rational root ruled out ("x ≠ 1, x ≠ -3"), and
    what is left of it; a divisor in more letters as it is."""
    letters = polynomial.letters()
    if len(letters) != 1 or polynomial.degree() == 1:
        return [_not_zero(polynomial)]
    from . import roots as rooting  # noqa: PLC0415

    letter = letters[0]
    coefficients = [Fraction(0)] * (polynomial.degree() + 1)
    for power, value in _one_letter_coefficients(polynomial, letter).items():
        coefficients[power] = value
    ruled_out, rest = [], []
    for factor, _ in rooting.factor_over_rationals(coefficients):
        if rooting._degree(factor) == 1:
            ruled_out.append(-factor[0] / factor[1])
        else:
            rest.append(f"{rooting.polynomial_text(factor, letter)} ≠ 0")
    return [f"{letter} ≠ {_number_text(value)}" for value in sorted(ruled_out)] + rest


def _not_zero(polynomial: Polynomial) -> str:
    """"x ≠ 1" for x - 1, else "x^2 + 1 ≠ 0"."""
    letters = polynomial.letters()
    if len(letters) == 1 and polynomial.degree() == 1:
        coefficients = _one_letter_coefficients(polynomial, letters[0])
        return f"{letters[0]} ≠ {_number_text(-coefficients.get(0, Fraction(0)) / coefficients[1])}"
    return f"{polynomial} ≠ 0"


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


# --- Solving --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Solution:
    """What a set of equations says about its unknowns.

    ``kind`` is "one" (a single value for each unknown, in ``values``),
    "several" (one unknown with more than one value, in ``roots``), "none" (no
    values work; ``reason`` says why) or "many" (``free`` lists the unknowns
    that can take any value, and ``values`` gives the others in terms of
    them). ``values`` are exact; ``decimals`` gives a decimal for each value
    that is a fraction. For an equation in one unknown with a power above 1,
    ``roots`` lists each real root smallest first, with whether it is exact,
    its decimal, how many times it is a root and, for a root with no exact
    form, an interval where the equation changes sign. For several unknowns
    with equations that are not linear, ``solutions`` lists each real
    solution, ordered by the unknowns' values: its ``values``, whether each is
    ``exact``, its ``decimals`` and, for a value with no exact form, the
    interval it lies in (``brackets``). ``conditions`` lists what must not be
    0, such as a denominator or a letter divided by. The last steps check the
    values in the equations.
    """

    equations: list[str]
    reading: list[str]
    unknowns: list[str]
    kind: str
    values: dict[str, str] = field(default_factory=dict)
    decimals: dict[str, str] = field(default_factory=dict)
    free: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    reason: str | None = None
    steps: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    roots: list[dict] = field(default_factory=list)
    solutions: list[dict] = field(default_factory=list)

    @property
    def answer(self) -> str:
        if self.kind == "none":
            return f"No solution: {self.reason}"
        if self.solutions:
            texts = [_solution_text(solution) for solution in self.solutions]
            line = texts[0] if len(texts) == 1 else " or ".join(f"({text})" for text in texts)
            return line + (f" (when {', '.join(self.conditions)})" if self.conditions else "")
        if self.roots:
            name = self.unknowns[0]
            line = " or ".join(_root_text(name, root) for root in self.roots)
            return line + (f" (when {', '.join(self.conditions)})" if self.conditions else "")
        values = ", ".join(f"{name} = {value}" + (f" ≈ {self.decimals[name]}" if name in self.decimals
                                                  else "") for name, value in self.values.items())
        if self.kind == "one":
            line = values
        elif values:
            line = f"{values}, for any {' and '.join(self.free)}"
        else:
            line = f"Every value of {' and '.join(self.free)} works"
        return line + (f" (when {', '.join(self.conditions)})" if self.conditions else "")

    def __str__(self) -> str:
        lines = [self.answer]
        if self.steps:
            lines.append("Steps:")
            lines += [f"  {step}" for step in self.steps]
        lines += [f"Assumption: {assumption}" for assumption in self.assumptions]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"schema_version": 1, "status": "solved", "equations": list(self.equations),
                "reading": list(self.reading), "unknowns": list(self.unknowns), "kind": self.kind,
                "answer": self.answer, "values": dict(self.values), "decimals": dict(self.decimals),
                "free": list(self.free), "conditions": list(self.conditions), "reason": self.reason,
                "roots": [dict(root) for root in self.roots],
                "solutions": [{key: dict(value) for key, value in solution.items()}
                              for solution in self.solutions],
                "steps": list(self.steps), "assumptions": list(self.assumptions)}


def _solution_text(solution: dict) -> str:
    parts = []
    for name, value in solution["values"].items():
        if not solution["exact"][name]:
            parts.append(f"{name} ≈ {value}")
        elif solution["decimals"][name] != value:
            parts.append(f"{name} = {value} ≈ {solution['decimals'][name]}")
        else:
            parts.append(f"{name} = {value}")
    return ", ".join(parts)


def _root_text(name: str, root: dict) -> str:
    if not root["exact"]:
        line = f"{name} ≈ {root['value']}"
    elif root["decimal"] != root["value"]:
        line = f"{name} = {root['value']} ≈ {root['decimal']}"
    else:
        line = f"{name} = {root['value']}"
    times = root["multiplicity"]
    return line + ({1: "", 2: " (twice)"}.get(times, f" ({times} times)"))


@dataclass
class _Equation:
    written: str
    left: Quotient
    right: Quotient
    sides: tuple[str, str]
    denominators: list[Polynomial] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def reading(self) -> str:
        return f"{self.sides[0]} = {self.sides[1]}"


def solve(equations: str | list[str], unknowns: str | list[str] | None = None,
          digits: int = DIGITS) -> Solution | Refusal:
    """Solve equations exactly, step by step.

    ``equations`` is one equation, several separated by ``;`` or new lines, or a
    list. The unknowns are every letter, unless ``unknowns`` names some ("t",
    or ["x", "y"]); the other letters are then treated as known, so
    ``solve("d = r*t", unknowns="t")`` gives t = d/r. Equations linear in the
    unknowns are solved by elimination; one equation in one unknown with
    powers gives its real roots (``roots.py``); a system of polynomial
    equations with finitely many solutions, and no other letters, gives its
    real solutions (``systems.py``). Returns a :class:`Solution`, or a
    ``Refusal`` when the equations cannot be read or solved.
    """
    try:
        parsed = _equations(equations, digits)
        letters = sorted({name for equation in parsed for name in _equation_letters(equation)})
        wanted = _unknowns(unknowns, letters)
        return _solve(parsed, wanted, digits)
    except _Refused as refused:
        return refused.refusal
    except (OverflowError, RecursionError):
        return Refusal("too_large", "The equations are too large to work out.")


def _unknowns(unknowns: str | list[str] | None, letters: list[str]) -> list[str]:
    if unknowns is None:
        if not letters:
            _refuse("invalid_expression", "The equation has no letter to solve for. To check arithmetic, "
                                          "compute each side.")
        return letters
    names = re.split(r"[,\s]+", unknowns.strip()) if isinstance(unknowns, str) else list(unknowns)
    names = [name for name in names if name]
    for name in names:
        if not (isinstance(name, str) and re.fullmatch(r"[A-Za-z]", name)) or name == "e":
            _refuse("invalid_expression",
                    f"{name!r} is not a letter to solve for; give one letter, such as x.")
        if name not in letters:
            _refuse("invalid_expression", f"{name} does not appear in the equations.")
    return list(dict.fromkeys(names))


def _split(text: str) -> list[str]:
    """The equations in a text: separated by ; or new lines, by "and", or by commas between equations."""
    pieces = [piece for piece in re.split(r";|\n|\band\b", text) if piece.strip()]
    found = []
    for piece in pieces:
        parts, depth, start = [], 0, 0
        for index, character in enumerate(piece):
            depth += {"(": 1, ")": -1}.get(character, 0)
            if character == "," and depth == 0 and piece[index + 1:index + 2] == " ":
                parts.append(piece[start:index])
                start = index + 1
        parts.append(piece[start:])
        found += parts if len(parts) > 1 and all("=" in part for part in parts) else [piece]
    return [piece.strip() for piece in found]


def _equations(equations: str | list[str], digits: int) -> list[_Equation]:
    if isinstance(equations, str):
        if (refusal := _check_text(equations)) is not None:
            raise _Refused(refusal.code, refusal.message)
        texts = _split(equations)
    elif isinstance(equations, (list, tuple)) and all(isinstance(item, str) for item in equations):
        texts = [piece for item in equations for piece in _split(item)]
    else:
        _refuse("invalid_expression", "Give the equations as text, or as a list of texts.")
    if not texts:
        _refuse("invalid_expression", "There is no equation.")
    if len(texts) > MAX_EQUATIONS:
        _refuse("too_large", f"There are more than {MAX_EQUATIONS} equations.")
    parsed = []
    for text in texts:
        if re.search(r"[<>≤≥≠]|!=", text):
            _refuse("unsupported_expression",
                    f"{text} is an inequality. ADA solves equations, written with =.")
        sides = text.replace("==", "=").split("=")
        if len(sides) < 2:
            _refuse("invalid_expression", f"{text} has no = sign, so it is not an equation.")
        if len(sides) > MAX_DEPTH:
            _refuse("too_large", f"{text} has too many = signs.")
        read = [_read(side, digits) for side in sides]
        for (left, first, left_text), (right, second, right_text) in zip(read, read[1:], strict=False):
            parsed.append(_Equation(text, left, right, (left_text, right_text),
                                    first.denominators + second.denominators, first.steps + second.steps,
                                    first.assumptions + second.assumptions))
    return parsed


def _equation_letters(equation: _Equation) -> set[str]:
    return set(equation.left.letters()) | set(equation.right.letters())


def _cleared(equation: _Equation) -> Polynomial:
    """left - right with every denominator multiplied away: the polynomial that must be 0."""
    difference = equation.left - equation.right
    return difference.numerator


def _solve(parsed: list[_Equation], unknowns: list[str], digits: int) -> Solution:
    steps: list[str] = []
    numbered = len(parsed) > 1
    known = tuple(unknowns)
    zeros = [_cleared(equation) for equation in parsed]
    if any(zero.degree(known) > 1 for zero in zeros):
        for index, equation in enumerate(parsed, 1):
            steps.append(f"({index}) {equation.written}" if numbered else equation.written)
            steps += equation.steps
        if len(parsed) == 1 and len(unknowns) == 1:
            return _solve_polynomial(parsed[0], unknowns[0], zeros[0], digits, steps)
        from .systems import solve_system  # noqa: PLC0415

        return solve_system(parsed, unknowns, zeros, digits, steps)
    rows = []
    for index, equation in enumerate(parsed, 1):
        label = f"({index}) " if numbered else ""
        steps.append(f"{label}{equation.written}")
        steps += equation.steps
        zero = zeros[index - 1]
        row = [*(_coefficient(zero, name, known) for name in unknowns), -_coefficient(zero, None, known)]
        first = next((entry for entry in row[:-1] if not entry.is_zero()), None)
        if first is not None and _leading(first.numerator)[1] < 0:
            # 8x = 120 rather than -8x = -120.
            row = [-entry for entry in row]
        rows.append(row)
        simplified = _row_text(row, unknowns)
        if simplified != equation.written:
            below = list(dict.fromkeys(side.denominator for side in (equation.left, equation.right)
                                       if not side.is_polynomial()))
            how = (f"Multiply both sides by {' and '.join(_operand(Quotient(item)) for item in below)} and "
                   "collect terms") if below else "Collect terms"
            steps.append(f"{how}: {simplified}")
    conditions_polynomials = [denominator for equation in parsed for denominator in equation.denominators]
    rows, pivots, divisors, operations = _eliminate(rows, unknowns, numbered)
    steps += operations
    # Dividing by an expression in the known letters holds only while it is not 0.
    conditions_polynomials += [divisor.numerator for divisor in divisors if not divisor.is_number()]
    for row in rows:
        if all(entry.is_zero() for entry in row[:-1]) and not row[-1].is_zero():
            if row[-1].is_number():
                reason = f"the equations reduce to 0 = {row[-1]}, which is false."
                return _finish(parsed, unknowns, "none", {}, [], conditions_polynomials, reason, steps,
                               digits)
            _refuse("unsupported_expression",
                    f"The equations reduce to 0 = {row[-1]}, so whether they have a solution depends on the "
                    "other letters.")
    pivot_columns = {column: row for row, column in pivots.items()}
    free = [name for column, name in enumerate(unknowns) if column not in pivot_columns]
    values: dict[str, Quotient] = {}
    for column, name in enumerate(unknowns):
        if column in pivot_columns:
            row = rows[pivot_columns[column]]
            value = row[-1]
            for other, entry in enumerate(row[:-1]):
                if other != column and not entry.is_zero():
                    value = value - entry * Quotient(Polynomial.letter(unknowns[other]))
            values[name] = value
    kind = "many" if free else "one"
    return _finish(parsed, unknowns, kind, values, free, conditions_polynomials, None, steps, digits)


def _solve_polynomial(equation: _Equation, unknown: str, zero: Polynomial, digits: int,
                      steps: list[str]) -> Solution:
    """One equation in one unknown with a power above 1: its real roots, exact where they can be written."""
    from . import roots as rooting  # noqa: PLC0415

    if set(zero.letters()) - {unknown}:
        others = " and ".join(sorted(set(zero.letters()) - {unknown}))
        _refuse("unsupported_expression",
                f"{equation.written} has {unknown} to a power above 1 and also {others}. ADA solves such an "
                f"equation when {unknown} is its only letter.")
    coefficients = [Fraction(0)] * (zero.degree() + 1)
    for power, value in _one_letter_coefficients(zero, unknown).items():
        coefficients[power] = value
    below = list(dict.fromkeys(side.denominator for side in (equation.left, equation.right)
                               if not side.is_polynomial()))
    how = (f"Multiply both sides by {' and '.join(_operand(Quotient(item)) for item in below)} and move "
           "everything to one side") if below else "Move everything to one side"
    collected = f"{rooting.polynomial_text(coefficients, unknown)} = 0"
    if collected != equation.written:
        steps.append(f"{how}: {collected}")
    # A value that makes a denominator zero is not a solution, so its factor is divided out first.
    for denominator in dict.fromkeys(equation.denominators):
        divisor = [Fraction(0)] * (denominator.degree() + 1)
        for power, value in _one_letter_coefficients(denominator, unknown).items():
            divisor[power] = value
        common = rooting._gcd(coefficients, divisor)
        while rooting._degree(common) > 0:
            coefficients, _ = rooting._divmod(coefficients, common)
            steps.append(f"The values of {unknown} that make {denominator} zero are not solutions, since the "
                         f"equation divides by it; dividing out {rooting.polynomial_text(common, unknown)} "
                         f"leaves {rooting.polynomial_text(coefficients, unknown)} = 0.")
            common = rooting._gcd(coefficients, divisor)
    conditions = _conditions(equation.denominators)
    reading = [equation.reading]
    if rooting._degree(coefficients) < 1:
        reason = ("every candidate value makes a denominator zero." if coefficients else
                  "the equation holds for every value except those that make a denominator zero.")
        if not coefficients:
            return Solution([equation.written], reading, [unknown], "many", {}, {}, [unknown], conditions,
                            reason, _capped(steps), _unique(equation.assumptions))
        return Solution([equation.written], reading, [unknown], "none", reason=reason, steps=_capped(steps),
                        assumptions=_unique(equation.assumptions))
    found, root_steps = rooting.real_roots(coefficients, unknown, digits)
    steps += root_steps
    for root in found:
        if root.exact and "sqrt" not in root.value and "root" not in root.value:
            steps += _checks([equation], {unknown: Quotient.number(Fraction(root.value))})
    if not found:
        return Solution([equation.written], reading, [unknown], "none",
                        reason=f"no real number solves {equation.written}.", steps=_capped(steps),
                        assumptions=_unique(equation.assumptions))
    listed = [root.to_dict() for root in found]
    if len(found) == 1:
        root = found[0]
        decimals = {unknown: root.decimal} if root.exact and root.decimal != root.value else {}
        return Solution([equation.written], reading, [unknown], "one", {unknown: root.value}, decimals,
                        conditions=conditions, steps=_capped(steps),
                        assumptions=_unique(equation.assumptions), roots=listed)
    return Solution([equation.written], reading, [unknown], "several", conditions=conditions,
                    steps=_capped(steps), assumptions=_unique(equation.assumptions), roots=listed)


def _coefficient(polynomial: Polynomial, name: str | None, unknowns: tuple[str, ...]) -> Quotient:
    """The part of a polynomial, linear in ``unknowns``, that multiplies ``name`` (or has no unknown)."""
    terms = {}
    for monomial, value in polynomial.terms.items():
        inside = [(letter, power) for letter, power in monomial if letter in unknowns]
        rest = tuple((letter, power) for letter, power in monomial if letter not in unknowns)
        if (name is None and not inside) or (name is not None and inside == [(name, 1)]):
            terms[rest] = terms.get(rest, 0) + value
    return Quotient(Polynomial(terms))


def _row_text(row: list[Quotient], unknowns: list[str]) -> str:
    left = Quotient.number(Fraction(0))
    for entry, name in zip(row, unknowns, strict=False):
        left = left + entry * Quotient(Polynomial.letter(name))
    return f"{left} = {row[-1]}"


def _eliminate(rows: list[list[Quotient]], unknowns: list[str], numbered: bool):
    """Gauss-Jordan elimination, writing each row operation as the equation it gives."""
    rows = [list(row) for row in rows]
    labels = list(range(1, len(rows) + 1))
    operations: list[str] = []
    pivots: dict[int, int] = {}
    divisors: list[Quotient] = []
    top = 0
    for column in range(len(unknowns)):
        candidates = [index for index in range(top, len(rows)) if not rows[index][column].is_zero()]
        if not candidates:
            continue
        # A number to divide by is better than an expression, which must then not be 0.
        chosen = next((index for index in candidates if rows[index][column].is_number()), candidates[0])
        rows[top], rows[chosen] = rows[chosen], rows[top]
        labels[top], labels[chosen] = labels[chosen], labels[top]
        pivot = rows[top][column]
        name = f"({labels[top]})" if numbered else "both sides"
        if not (pivot.is_number() and pivot.value() == 1):
            divisors.append(pivot)
            rows[top] = [entry / pivot for entry in rows[top]]
            operations.append(f"Divide {name} by {_operand(pivot)}: {_row_text(rows[top], unknowns)}")
        for index in range(len(rows)):
            if index == top or rows[index][column].is_zero():
                continue
            factor = rows[index][column]
            rows[index] = [entry - factor * pivot_entry
                           for entry, pivot_entry in zip(rows[index], rows[top], strict=True)]
            operations.append(f"{_combination(labels[index], labels[top], factor)}: "
                              f"{_row_text(rows[index], unknowns)}")
        pivots[top] = column
        top += 1
    return rows, pivots, divisors, operations


def _combination(target: int, source: int, factor: Quotient) -> str:
    if factor.is_number():
        value = factor.value()
        if value == 1:
            return f"({target}) − ({source})"
        if value == -1:
            return f"({target}) + ({source})"
        return f"({target}) − {_number_text(value)}×({source})" if value > 0 else \
            f"({target}) + {_number_text(-value)}×({source})"
    return f"({target}) − ({factor})×({source})"


def _finish(parsed: list[_Equation], unknowns: list[str], kind: str, values: dict[str, Quotient],
            free: list[str], conditions: list[Polynomial], reason: str | None, steps: list[str],
            digits: int) -> Solution:
    shown_conditions = _conditions(conditions)
    assumptions = _unique([item for equation in parsed for item in equation.assumptions])
    if kind == "one":
        bound = {name: value for name, value in values.items()}
        for condition in conditions:
            at = condition.substituted(bound)
            if at.is_zero():
                kind = "none"
                reason = (f"{', '.join(f'{name} = {value}' for name, value in values.items())} makes "
                          f"{condition} zero, and the equations divide by it.")
                steps.append(f"{', '.join(f'{name} = {value}' for name, value in values.items())} makes "
                             f"{condition} = 0, which the equations divide by, so it is not a solution.")
                return Solution([equation.written for equation in parsed],
                                [equation.reading for equation in parsed], unknowns, kind, reason=reason,
                                steps=_capped(steps), assumptions=assumptions)
        steps += _checks(parsed, bound)
    elif kind == "many" and values:
        steps += _checks(parsed, values)
    texts = {name: str(value) for name, value in values.items()}
    decimals = {}
    for name, value in values.items():
        if value.is_number() and not _terminates(value.value()):
            decimals[name] = _decimal_text(_approx_wide(value.value(), digits), digits)
        elif value.is_number() and "/" in texts[name]:
            decimals[name] = _exact_decimal(value.value())
    if kind == "many" and not values:
        reason = "every equation holds for any value."
    elif kind == "many":
        reason = (f"there are fewer independent equations than unknowns, so {' and '.join(free)} can be "
                  "anything.")
    return Solution([equation.written for equation in parsed], [equation.reading for equation in parsed],
                    unknowns, kind, texts, decimals, free, shown_conditions, reason, _capped(steps),
                    assumptions)


def _checks(parsed: list[_Equation], values: dict[str, Quotient]) -> list[str]:
    """Put the values back into each equation, as the last steps."""
    lines = []
    given = ", ".join(f"{name} = {value}" for name, value in values.items())
    for equation in parsed:
        left = _substitute(equation.left, values)
        right = _substitute(equation.right, values)
        verdict = "holds" if left == right else "does NOT hold"
        lines.append(f"Check {equation.written} with {given}: {left} = {right}, which {verdict}.")
    return lines


def _substitute(value: Quotient, values: dict[str, Quotient]) -> Quotient:
    return value.numerator.substituted(values) / value.denominator.substituted(values)


def _capped(steps: list[str]) -> list[str]:
    if len(steps) <= MAX_STEPS:
        return steps
    return steps[:MAX_STEPS] + [f"… and {len(steps) - MAX_STEPS} more steps."]


# --- Checking a solution --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Substitution:
    """One equation with values put in for its letters: both sides, and whether they are equal."""

    equation: str
    values: dict[str, str]
    left: str
    right: str
    holds: bool

    def to_dict(self) -> dict:
        return {"equation": self.equation, "values": dict(self.values), "left": self.left,
                "right": self.right, "holds": self.holds}


def check_solution(equations: str | list[str], values: dict | str,
                   digits: int = DIGITS) -> list[Substitution] | Refusal:
    """Put ``values`` into each equation and say whether both sides come out equal, exactly.

    ``values`` maps each letter to a number or an expression to compute, such
    as {"x": 4} or {"x": "7/2"}, or is written out: "x = 4, y = 2". Use it to
    check a solution a language model gives.
    """
    try:
        bindings = _values(values, digits)
        parsed = _equations(equations, digits)
        results = []
        for equation in parsed:
            sides = []
            for text in equation.sides:
                value, _, _ = _read(text, digits, bindings)
                if not value.is_number():
                    missing = ", ".join(value.letters())
                    _refuse("missing_value", f"No value is given for {missing} in {equation.written}.")
                sides.append(value.value())
            shown_values = {name: _number_text(value) for name, value in bindings.items()
                            if name in _equation_letters(equation)}
            results.append(Substitution(equation.written, shown_values, _number_text(sides[0]),
                                        _number_text(sides[1]), sides[0] == sides[1]))
        return results
    except _Refused as refused:
        return refused.refusal


def _values(values: dict | str, digits: int) -> dict[str, Fraction]:
    if isinstance(values, str):
        pairs = [piece.split("=") for piece in re.split(r"[;,]\s*", values) if piece.strip()]
        if any(len(pair) != 2 for pair in pairs):
            _refuse("invalid_expression", "Write the values as x = 4, y = 2.")
        values = {name.strip(): value.strip() for name, value in pairs}
    if not isinstance(values, dict) or not values:
        _refuse("invalid_expression", "Give a value for each letter, such as {\"x\": 4}.")
    bindings = {}
    for name, value in values.items():
        if not (isinstance(name, str) and re.fullmatch(r"[A-Za-z]", name)):
            _refuse("invalid_expression", f"{name!r} is not a letter; give one letter, such as x.")
        if isinstance(value, bool):
            _refuse("invalid_expression", f"The value of {name} is not a number.")
        if isinstance(value, (int, Fraction)):
            bindings[name] = Fraction(value)
            continue
        computed = compute(str(value), digits=digits)
        if isinstance(computed, Refusal):
            raise _Refused(computed.code, f"The value of {name}: {computed.message}")
        if not computed.exact:
            _refuse("invalid_expression", f"The value of {name}, {value}, is not exact; give it as a "
                                          "fraction or a decimal.")
        bindings[name] = Fraction(computed.fraction or computed.value)
    return bindings


# --- One entry for the tool and the command line ---------------------------------------------------------


#: "solve 2x + 3 = 11 for x": the words around an equation that name it and its unknowns.
_SOLVE_FOR = re.compile(r"(?is)\s*(?:solve\b\s*:?)?(.*?)"
                        r"(?:\s+for\s+([A-Za-z](?:\s*(?:,|and)\s*[A-Za-z])*))?\s*")


def work_out(text: str, digits: int = DIGITS):
    """Compute, simplify or solve, whichever ``text`` is: an equation (with =) is solved, an expression with
    letters is simplified, and anything else is computed. "solve 2x + 3 = 11 for x" names the unknown."""
    if not isinstance(text, str):
        return Refusal("invalid_expression", "The expression is empty.")
    written = text.strip()
    unknowns = None
    match = _SOLVE_FOR.fullmatch(written)
    if match and ("=" in match.group(1) or written.lower().startswith("solve")):
        body = match.group(1)
        if match.group(2):
            unknowns = re.findall(r"\b[A-Za-z]\b", match.group(2))
    else:
        body = written
    body = re.sub(r"(?i)^\s*(?:simplify|expand)\b\s*:?", "", body)
    stripped = body.rstrip("=?").strip() or body
    if "=" in stripped:
        return solve(stripped, unknowns=unknowns, digits=digits)
    if _letters_in(stripped):
        return simplify(stripped, digits=digits)
    return compute(written if body == written else stripped, digits=digits)
