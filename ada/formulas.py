"""Polynomial equations in one unknown with other letters in the coefficients, solved for the unknown.

    import ada
    ada.solve("a*x^2 + b*x + c = 0", unknowns="x")
    # x = (-b - sqrt(b^2 - 4a*c))/(2a) or x = (-b + sqrt(b^2 - 4a*c))/(2a) (when a ≠ 0, b^2 - 4a*c ≥ 0)
    ada.solve("x^2 - (a + b)x + a*b = 0", unknowns="x")      # x = a or x = b

SymPy factors the polynomial over the rationals, the other letters included, and ADA multiplies the factors
back to check them. Each factor then gives its roots: one of degree 1 in the unknown gives one root, a
quadratic gives two by the quadratic formula with square factors taken out of the root, A*x^n + C gives
root(-C/A, n), and a factor with no other letters gives its real roots as in ``roots.py``. What each answer
needs is stated as a condition: a leading coefficient that is not 0, a number under a root that is not
negative, and denominators that are not 0.

Every root is checked with ADA's own arithmetic: put into its factor, with sqrt(R)^2 = R or root(t, n)^n = t,
it gives exactly 0.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from . import roots as rooting
from .compute import _refuse

#: The most letters, the unknown included.
MAX_LETTERS = 6
#: The highest total power of the polynomial.
MAX_POWER = 30
#: The letter that stands for a square root while a root is checked; never one the equation can use.
_ROOT = "√"


def solve_with_letters(equation, unknown: str, zero, digits: int, steps: list[str]):
    """The roots of ``zero`` = 0 in ``unknown``, with the other letters treated as known."""
    from .algebra import (  # noqa: PLC0415
        Polynomial,
        Quotient,
        Solution,
        _capped,
        _operand,
        _unique,
    )

    letters = [unknown, *sorted(set(zero.letters()) - {unknown})]
    others = letters[1:]
    if len(letters) > MAX_LETTERS:
        _refuse("too_large", f"{equation.written} has {len(letters)} letters; ADA solves an equation with "
                             f"powers of the unknown and up to {MAX_LETTERS} letters.")
    if zero.degree() > MAX_POWER:
        _refuse("too_large", f"{equation.written} reaches a power of {zero.degree()}; ADA solves an equation "
                             f"with other letters up to a power of {MAX_POWER}.")
    below = list(dict.fromkeys(side.denominator for side in (equation.left, equation.right)
                               if not side.is_polynomial()))
    how = (f"Multiply both sides by {' and '.join(_operand(Quotient(item)) for item in below)} and move "
           "everything to one side") if below else "Move everything to one side"
    if f"{_shown(zero, unknown)} = 0" != equation.written:
        steps.append(f"{how}: {_shown(zero, unknown)} = 0")

    content, factors = _factored(zero, letters)
    if len(factors) > 1 or factors[0][1] > 1 or content != 1:
        lead = "" if content == 1 else "-" if content == -1 else f"{rooting._number_text(content)} * "
        written = " * ".join(_power_of(_shown(factor, unknown), times) for factor, times in factors)
        steps.append(f"Factor over the rationals (SymPy): {_shown(zero, unknown)} = {lead}{written}.")

    denominators = list(dict.fromkeys(equation.denominators))
    conditions = [_not_zero_in(denominator, unknown) for denominator in denominators]
    numeric: list[tuple[Fraction, dict]] = []
    symbolic: list[dict] = []
    for factor, times in factors:
        by_power = _in_powers(factor, unknown)
        degree = max(by_power)
        text = _shown(factor, unknown)
        if degree == 0:
            conditions.append(_not_zero_in(factor, unknown))
            steps.append(f"When {text} = 0, every value of {unknown} works, so the answer takes {text} ≠ 0.")
            continue
        if any(_divides(factor, denominator, letters) for denominator in denominators):
            steps.append(f"The values of {unknown} that make {text} zero also make a denominator zero, and "
                         "the equation divides by it, so they are not solutions.")
            continue
        if not set(factor.letters()) - {unknown}:
            coefficients = [by_power.get(power, Polynomial()).value() for power in range(degree + 1)]
            whole = rooting._primitive(coefficients)
            for key, root in rooting._roots_of_factor(whole, unknown, digits, steps):
                numeric.append((key, {**root.to_dict(), "multiplicity": times}))
            continue
        lead = by_power[degree]
        if not lead.is_number():
            conditions.append(_not_zero_in(lead, unknown))
            steps.append(f"This takes {lead} ≠ 0: when {lead} = 0, {text} has a lower power of {unknown}, "
                         "so solve it again with that value.")
        if degree == 1:
            value = Quotient(-by_power.get(0, Polynomial())) / Quotient(lead)
            held = factor.substituted({unknown: value}).is_zero()
            steps.append(f"{text} = 0 gives {unknown} = {value}.")
            steps.append(_check_text(unknown, str(value), text, held))
            symbolic.append(_symbolic(str(value), times))
        elif degree == 2:
            found, needs = _quadratic(by_power, unknown, text, others, steps)
            conditions += needs
            symbolic += [_symbolic(value, times) for value in found]
        elif all(power in (0, degree) for power in by_power):
            found, needs = _binomial(by_power, degree, unknown, text, steps)
            conditions += needs
            symbolic += [_symbolic(value, times) for value in found]
        else:
            _refuse("unsupported_expression",
                    f"{text} = 0 is of degree {degree} in {unknown} with {' and '.join(others)} in it. ADA "
                    f"solves such an equation when each factor is of degree 1 or 2 in {unknown}, or a number "
                    f"times a power of {unknown} plus a part without it, such as a*{unknown}^{degree} - b.")
    roots = [root for _, root in sorted(numeric, key=lambda item: item[0])] + symbolic
    conditions = _merged(_unique(conditions))
    written, reading = [equation.written], [equation.reading]
    assumptions = _unique(equation.assumptions)
    if not roots:
        return Solution(written, reading, [unknown], "none", conditions=conditions,
                        reason=f"no real value of {unknown} solves {equation.written}"
                               + (f" (when {', '.join(conditions)})." if conditions else "."),
                        steps=_capped(steps), assumptions=assumptions)
    if len(roots) == 1:
        root = roots[0]
        decimals = {unknown: root["decimal"]} if root["decimal"] not in (None, root["value"]) else {}
        return Solution(written, reading, [unknown], "one", {unknown: root["value"]}, decimals,
                        conditions=conditions, steps=_capped(steps), assumptions=assumptions, roots=roots)
    return Solution(written, reading, [unknown], "several", conditions=conditions, steps=_capped(steps),
                    assumptions=assumptions, roots=roots)


def _shown(polynomial, unknown: str) -> str:
    """The polynomial written by powers of the unknown, highest first, each with its coefficient in the other
    letters: x^2 - (a + b)x + a*b rather than a*b - a*x - b*x + x^2."""
    from .algebra import _leading  # noqa: PLC0415

    by_power = _in_powers(polynomial, unknown)
    if set(by_power) == {0}:
        return str(polynomial)
    text = ""
    for power in sorted(by_power, reverse=True):
        coefficient = by_power[power]
        letter = "" if power == 0 else unknown if power == 1 else f"{unknown}^{power}"
        negative = _leading(coefficient)[1] < 0
        size = -coefficient if negative else coefficient
        if power == 0:
            body = str(size)
        elif size.is_number():
            body = letter if size.value() == 1 else f"{rooting._number_text(size.value())}{letter}"
        elif len(size.terms) == 1:
            body = f"{size}*{letter}"
        else:
            body = f"({size}){letter}"
        if not text:
            text = f"-{body}" if negative else body
        else:
            text += f" {'-' if negative else '+'} {body}"
    return text


def _symbolic(value: str, times: int) -> dict:
    return {"value": value, "exact": True, "decimal": None, "multiplicity": times, "bracket": None}


def _check_text(unknown: str, value: str, text: str, held: bool, using: str = "") -> str:
    """The check as a step; a root that does not give 0 is refused rather than given."""
    if not held:
        _refuse("unsupported_expression", f"Putting {unknown} = {value} into {text}{using} does not give "
                                          "0, so ADA gives no answer.")
    return f"Check {unknown} = {value} in {text}{using}: it gives exactly 0, which holds."


def _power_of(text: str, times: int) -> str:
    wrapped = f"({text})" if " " in text else text
    return f"{wrapped}^{times}" if times > 1 else wrapped


def _not_zero_in(polynomial, unknown: str) -> str:
    """"x ≠ a" for x - a, "a ≠ 0" for 2a, and otherwise the polynomial ≠ 0, with no whole-number factor."""
    from .algebra import Quotient, _not_zero  # noqa: PLC0415

    by_power = _in_powers(polynomial, unknown)
    if max(by_power) == 1 and by_power[1].is_number() and set(polynomial.letters()) - {unknown}:
        value = Quotient(-by_power.get(0, type(polynomial)())) / Quotient(by_power[1])
        return f"{unknown} ≠ {value}"
    return _not_zero(_without_content(polynomial))


def _without_content(polynomial):
    """The polynomial divided by the positive whole number or fraction common to its coefficients."""
    from .algebra import _content  # noqa: PLC0415

    if polynomial.is_number():
        return polynomial
    return polynomial.scaled(1 / abs(_content(polynomial)))


def _merged(conditions: list[str]) -> list[str]:
    """"a ≠ 0" and "a ≥ 0" become "a > 0"."""
    merged = []
    for condition in conditions:
        if condition.endswith(" ≥ 0") and condition.replace(" ≥ 0", " ≠ 0") in conditions:
            continue
        if condition.endswith(" ≠ 0") and condition.replace(" ≠ 0", " ≥ 0") in conditions:
            merged.append(condition.replace(" ≠ 0", " > 0"))
            continue
        merged.append(condition)
    return merged


# --- Factoring -----------------------------------------------------------------------------------------


def _factored(polynomial, letters: list[str]):
    """(content, [(factor, times)]) from SymPy's factor_list, refused unless the factors multiply back."""
    from sympy import factor_list  # noqa: PLC0415

    from .algebra import Polynomial  # noqa: PLC0415

    symbols = _symbols(letters)
    content, factors = factor_list(_to_sympy(polynomial, symbols), *symbols.values())
    content = Fraction(int(content.p), int(content.q))
    found = [(_from_sympy(factor, letters), int(times)) for factor, times in factors]
    product = Polynomial.number(content)
    for factor, times in found:
        product = product * factor.power(times)
    if product != polynomial:
        _refuse("unsupported_expression", "The factors SymPy gives do not multiply back to the polynomial, "
                                          "so ADA gives no answer.")
    return content, found


def _divides(factor, denominator, letters: list[str]) -> bool:
    """Whether every value of the unknown that makes ``factor`` zero makes ``denominator`` zero too."""
    from sympy import prem  # noqa: PLC0415

    if not set(denominator.letters()) & {letters[0]}:
        return False
    symbols = _symbols(letters)
    remainder = prem(_to_sympy(denominator, symbols), _to_sympy(factor, symbols), symbols[letters[0]])
    return remainder == 0


def _symbols(letters: list[str]) -> dict:
    from sympy import Symbol  # noqa: PLC0415

    return {name: Symbol(name) for name in letters}


def _to_sympy(polynomial, symbols: dict):
    from .systems import _to_sympy as convert  # noqa: PLC0415

    return convert(polynomial, symbols)


def _from_sympy(expression, letters: list[str]):
    from sympy import Poly  # noqa: PLC0415

    from .algebra import Polynomial  # noqa: PLC0415

    symbols = _symbols(letters)
    terms = {}
    for exponents, value in Poly(expression, *symbols.values()).terms():
        monomial = tuple(sorted((name, power) for name, power in zip(letters, exponents, strict=True)
                                if power))
        terms[monomial] = Fraction(int(value.p), int(value.q))
    return Polynomial(terms)


def _in_powers(polynomial, letter: str) -> dict:
    """The polynomial as a sum of powers of ``letter``, each with a coefficient in the other letters."""
    from .algebra import Polynomial  # noqa: PLC0415

    groups: dict[int, dict] = {}
    for monomial, value in polynomial.terms.items():
        powers = dict(monomial)
        power = powers.pop(letter, 0)
        groups.setdefault(power, {})[tuple(sorted(powers.items()))] = value
    return {power: Polynomial(terms) for power, terms in groups.items()}


# --- A quadratic factor --------------------------------------------------------------------------------


def _quadratic(by_power: dict, unknown: str, text: str, others: list[str], steps: list[str]):
    """The two roots of A*x^2 + B*x + C by the quadratic formula, with what they need."""
    from .algebra import Polynomial  # noqa: PLC0415

    a, b, c = (by_power.get(power, Polynomial()) for power in (2, 1, 0))
    discriminant = b * b - a * c.scaled(Fraction(4))
    square, radicand = _square_root_parts(discriminant, others)
    top, half, bottom = _lowest(-b, square, a.scaled(Fraction(2)), others)
    needs = []
    sign = _sign_everywhere(radicand)
    if radicand.is_number() and radicand.value() == 1:
        _refuse("unsupported_expression", f"{text} should have factored, since b^2 - 4ac = {discriminant} "
                                          "is a square; ADA gives no answer.")
    if sign < 0 and square.is_number():
        steps.append(f"{text} = 0: the discriminant b^2 - 4ac = {discriminant} is negative for every value "
                     f"of {' and '.join(others)}, so no real number is a root.")
        return [], needs
    if sign < 0:
        _refuse("unsupported_expression",
                f"{text} = 0 has a real root only when {_without_content(square)} = 0, since b^2 - 4ac = "
                f"{discriminant} is "
                f"negative otherwise. ADA gives no general answer: give {' and '.join(others)} a value.")
    if sign == 0:
        needs.append(f"{_without_content(radicand)} ≥ 0")
    both = _surd_text(top, half, radicand, bottom, None)
    steps.append(f"{text} = 0 by the quadratic formula: the discriminant b^2 - 4ac = {discriminant}, so "
                 f"{unknown} = (-b ± sqrt(b^2 - 4ac))/(2a) = {both}" + (f", when {', '.join(needs)}" if needs
                                                                        else "") + ".")
    found = []
    for sign_of_root in (-1, 1):
        value = _surd_text(top, half, radicand, bottom, sign_of_root)
        held = _surd_holds(a, b, c, top, half.scaled(Fraction(sign_of_root)), radicand, bottom)
        steps.append(_check_text(unknown, value, text, held, f", with sqrt({radicand})^2 = {radicand}"))
        found.append(value)
    return found, needs


def _square_root_parts(discriminant, others: list[str]):
    """sqrt(discriminant) as square * sqrt(radicand), square factors taken out (SymPy's factor_list)."""
    from sympy import factor_list  # noqa: PLC0415

    from .algebra import Polynomial  # noqa: PLC0415

    symbols = _symbols(others)
    content, factors = factor_list(_to_sympy(discriminant, symbols), *symbols.values())
    numerator, denominator = int(content.p), int(content.q)
    k, m = rooting._square_part(abs(numerator * denominator))
    square = Polynomial.number(Fraction(k, denominator))
    radicand = Polynomial.number(Fraction(m if numerator > 0 else -m))
    for factor, times in factors:
        polynomial = _from_sympy(factor, others)
        square = square * polynomial.power(int(times) // 2)
        if int(times) % 2:
            radicand = radicand * polynomial
    return square, radicand


def _lowest(top, half, bottom, others: list[str]):
    """top, half and bottom with whole-number coefficients, no common factor, and bottom's first term
    positive: the roots (top ± half*sqrt(R))/bottom are unchanged."""
    from sympy import gcd  # noqa: PLC0415

    from .algebra import _leading  # noqa: PLC0415

    scale = math.lcm(*(value.denominator for part in (top, half, bottom) for value in part.terms.values()))
    top, half, bottom = (part.scaled(Fraction(scale)) for part in (top, half, bottom))
    if others:
        symbols = _symbols(others)
        common = gcd(gcd(_to_sympy(top, symbols), _to_sympy(half, symbols)), _to_sympy(bottom, symbols))
        common = _from_sympy(common, others)
        if not common.is_number():
            top, half, bottom = (_exactly(part, common, others) for part in (top, half, bottom))
    whole = math.gcd(*(int(value) for part in (top, half, bottom) for value in part.terms.values()))
    scale = Fraction(1, whole) * (1 if _leading(bottom)[1] > 0 else -1)
    top, half, bottom = (part.scaled(scale) for part in (top, half, bottom))
    if _leading(half)[1] < 0:
        half = -half
    return top, half, bottom


def _exactly(polynomial, divisor, others: list[str]):
    from sympy import div  # noqa: PLC0415

    if polynomial.is_zero():
        return polynomial
    symbols = _symbols(others)
    quotient, remainder = div(_to_sympy(polynomial, symbols), _to_sympy(divisor, symbols), *symbols.values())
    if remainder != 0:
        _refuse("unsupported_expression", "A common factor did not divide exactly, so ADA gives no answer.")
    return _from_sympy(quotient, others)


def _surd_text(top, half, radicand, bottom, sign: int | None) -> str:
    """(top ± half*sqrt(radicand))/bottom, written so it reads back; sign None writes ±."""
    from .algebra import Quotient, _wrapped  # noqa: PLC0415

    radical = f"sqrt({radicand})"
    if half.is_number():
        size = half.value()
        body = radical if size == 1 else f"{rooting._number_text(size)}*{radical}"
    else:
        body = f"{_wrapped(Quotient(half))}*{radical}"
    mark = "±" if sign is None else "+" if sign > 0 else "-"
    if top.is_zero():
        numerator, single = (f"±{body}" if sign is None else body if sign > 0 else f"-{body}"), True
    else:
        numerator, single = f"{top} {mark} {body}", False
    if bottom.is_number() and bottom.value() == 1:
        return numerator
    below = str(bottom)
    if len(bottom.terms) > 1 or not re.fullmatch(r"[A-Za-z](\^\d+)?|\d+", below):
        below = f"({below})"
    return f"{numerator}/{below}" if single else f"({numerator})/{below}"


def _surd_holds(a, b, c, top, half, radicand, bottom) -> bool:
    """Whether a*x^2 + b*x + c is exactly 0 at x = (top + half*sqrt(radicand))/bottom, with sqrt(radicand)^2
    = radicand: bottom^2 times the value, multiplied out, must be 0."""
    from .algebra import Polynomial  # noqa: PLC0415

    root = Polynomial.letter(_ROOT)
    x = top + half * root
    total = a * x * x + b * x * bottom + c * bottom * bottom
    reduced = Polynomial()
    for monomial, value in total.terms.items():
        powers = dict(monomial)
        k = powers.pop(_ROOT, 0)
        term = Polynomial({tuple(sorted(powers.items())): value}) * radicand.power(k // 2)
        reduced = reduced + (term * root if k % 2 else term)
    return reduced.is_zero()


def _sign_everywhere(polynomial) -> int:
    """1 when the polynomial is at least 0 for every value of its letters (every power even, every
    coefficient positive), -1 when it is below 0 for every value (the same with negative coefficients and a
    constant term), else 0."""
    if not all(power % 2 == 0 for monomial in polynomial.terms for _, power in monomial):
        return 0
    values = list(polynomial.terms.values())
    if all(value > 0 for value in values):
        return 1
    if all(value < 0 for value in values) and () in polynomial.terms:
        return -1
    return 0


# --- A*x^n + C ------------------------------------------------------------------------------------------


def _binomial(by_power: dict, degree: int, unknown: str, text: str, steps: list[str]):
    """A*x^n + C = 0: x^n = -C/A, so x = root(-C/A, n), and its negative when n is even, with t ≥ 0."""
    from .algebra import Polynomial, Quotient  # noqa: PLC0415

    a, c = by_power[degree], by_power.get(0, Polynomial())
    t = Quotient(-c) / Quotient(a)
    held = (Quotient(a) * t + Quotient(c)).is_zero()
    inside = str(t)
    sign = _sign_everywhere(t.numerator) if t.is_polynomial() else 0
    if degree % 2 == 0 and sign < 0:
        steps.append(f"{text} = 0 means {unknown}^{degree} = {inside}, which is negative for every value of "
                     "the other letters, and no real number to an even power is negative, so it has no real "
                     "root.")
        return [], []
    shown = _without_content(t.numerator) if t.is_polynomial() else inside
    needs = [] if degree % 2 or sign > 0 else [f"{shown} ≥ 0"]
    rule = f"by definition, root(t, {degree})^{degree} = t"
    values = [f"root({inside}, {degree})"] if degree % 2 else [f"-root({inside}, {degree})",
                                                               f"root({inside}, {degree})"]
    steps.append(f"{text} = 0 means {unknown}^{degree} = {inside}, so {unknown} = " + " or ".join(values)
                 + (f", when {inside} ≥ 0" if needs else "") + f": {rule}.")
    if not held:
        _refuse("unsupported_expression", f"With {unknown}^{degree} = {inside}, {text} does not give 0, "
                                          "so ADA gives no answer.")
    steps.append(f"Check: with {unknown}^{degree} = {inside}, {text} gives exactly 0, which holds.")
    return values, needs
