"""The real roots of a polynomial in one letter with fraction coefficients, exact wherever they can be
written.

    from fractions import Fraction
    from ada.roots import real_roots
    roots, steps = real_roots([Fraction(1), Fraction(-3), Fraction(1)], "x")   # x^2 - 3x + 1
    [root.value for root in roots]        # ["(3 - sqrt(5))/2", "(3 + sqrt(5))/2"]

Coefficients run from the constant term up. A root comes back as a whole number or fraction, as
a + b*sqrt(m) from the quadratic formula, as root(t, n) when the polynomial is a*x^n + c, and otherwise as a
decimal to the digits asked for, with an interval where the polynomial changes sign: a root lies inside it.
Every exact root is put back into the polynomial with exact arithmetic, and the check is one of the steps.
Each value is written so that ``ada.compute`` reads it back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .compute import DIGITS, _decimal_text, _exact_decimal, _refuse, _terminates, compute

#: The highest power of a polynomial whose roots are looked for.
MAX_ROOT_DEGREE = 30
#: Whole numbers up to this size are split into prime factors to list rational candidates and simplify roots.
FACTOR_LIMIT = 10 ** 12
#: Halvings of an interval before ADA stops narrowing a root.
MAX_HALVINGS = 2000
#: Above this many rational candidates, only those near a real root are tried.
MAX_CANDIDATES = 25_000

Coefficients = list  # list[Fraction], constant term first


@dataclass(frozen=True)
class Root:
    """One real root: its value as ADA writes it, whether that value is exact, a decimal, how many times it is
    a root, and for an approximate root the interval where the polynomial changes sign."""

    value: str
    exact: bool
    decimal: str
    multiplicity: int = 1
    bracket: tuple[str, str] | None = None

    def to_dict(self) -> dict:
        return {"value": self.value, "exact": self.exact, "decimal": self.decimal,
                "multiplicity": self.multiplicity, "bracket": list(self.bracket) if self.bracket else None}


# --- Polynomials as lists ----------------------------------------------------------------------------------


def _trim(p: Coefficients) -> Coefficients:
    p = list(p)
    while p and p[-1] == 0:
        p.pop()
    return p


def _degree(p: Coefficients) -> int:
    return len(_trim(p)) - 1


def _evaluate(p: Coefficients, x: Fraction) -> Fraction:
    total = Fraction(0)
    for value in reversed(p):
        total = total * x + value
    return total


def _derivative(p: Coefficients) -> Coefficients:
    return _trim([power * value for power, value in enumerate(p)][1:])


def _divmod(top: Coefficients, bottom: Coefficients) -> tuple[Coefficients, Coefficients]:
    top, bottom = _trim(top), _trim(bottom)
    quotient = [Fraction(0)] * max(len(top) - len(bottom) + 1, 1)
    remainder = list(top)
    while len(_trim(remainder)) >= len(bottom) and _trim(remainder):
        remainder = _trim(remainder)
        shift = len(remainder) - len(bottom)
        factor = remainder[-1] / bottom[-1]
        quotient[shift] = factor
        for index, value in enumerate(bottom):
            remainder[index + shift] -= factor * value
    return _trim(quotient), _trim(remainder)


def _monic(p: Coefficients) -> Coefficients:
    p = _trim(p)
    return [value / p[-1] for value in p]


def _gcd(first: Coefficients, second: Coefficients) -> Coefficients:
    first, second = _trim(first), _trim(second)
    while second:
        _, remainder = _divmod(first, second)
        first, second = second, remainder
    return _monic(first) if first else [Fraction(1)]


def _primitive(p: Coefficients) -> Coefficients:
    """Whole-number coefficients with no common factor and a positive leading one."""
    p = _trim(p)
    scale = math.lcm(*(value.denominator for value in p))
    whole = [int(value * scale) for value in p]
    common = math.gcd(*whole) * (1 if whole[-1] > 0 else -1)
    return [Fraction(value // common) for value in whole]


def _minus(first: Coefficients, second: Coefficients) -> Coefficients:
    length = max(len(first), len(second))
    return _trim([(first[i] if i < len(first) else 0) - (second[i] if i < len(second) else 0)
                  for i in range(length)])


def _square_free(p: Coefficients) -> list[tuple[Coefficients, int]]:
    """Yun's method: p as a product of factors with no repeated root, each with how many times it appears."""
    p = _monic(p)
    derivative = _derivative(p)
    common = _gcd(p, derivative)
    rest, _ = _divmod(p, common)
    other, _ = _divmod(derivative, common)
    difference = _minus(other, _derivative(rest))
    factors, times = [], 1
    while _degree(rest) > 0:
        factor = _gcd(rest, difference)
        if _degree(factor) > 0:
            factors.append((factor, times))
        rest, _ = _divmod(rest, factor)
        other, _ = _divmod(difference, factor)
        difference = _minus(other, _derivative(rest))
        times += 1
    return factors


def _factor_text(text: str, times: int) -> str:
    wrapped = f"({text})" if " " in text else text
    return f"{wrapped}^{times}" if times > 1 else wrapped


def polynomial_text(p: Coefficients, letter: str) -> str:
    """p written highest power first, as ADA's algebra writes it."""
    from .algebra import Polynomial  # noqa: PLC0415

    return str(Polynomial({((letter, power),) if power else (): value for power, value in enumerate(p)}))


# --- Whole numbers --------------------------------------------------------------------------------------


def _factor(n: int) -> dict[int, int] | None:
    """The prime factors of |n| by trial division, or None when |n| is too large to factor this way."""
    n = abs(n)
    if n > FACTOR_LIMIT:
        return None
    factors: dict[int, int] = {}
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            n //= divisor
        divisor += 1 if divisor == 2 else 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def _divisors(n: int) -> list[int] | None:
    factors = _factor(n)
    if factors is None:
        return None
    divisors = [1]
    for prime, power in factors.items():
        divisors = [d * prime ** k for d in divisors for k in range(power + 1)]
    return sorted(divisors)


def _square_part(n: int) -> tuple[int, int]:
    """n = k^2 * m with m square-free, as (k, m), when n is small enough to factor; else (1, n)."""
    factors = _factor(n)
    if factors is None:
        return 1, n
    k, m = 1, 1
    for prime, power in factors.items():
        k *= prime ** (power // 2)
        m *= prime ** (power % 2)
    return k, m


# --- Numbers of the form a + b*sqrt(m) -------------------------------------------------------------------


@dataclass(frozen=True)
class _Surd:
    rational: Fraction
    radical: Fraction
    m: int

    def __mul__(self, other: _Surd) -> _Surd:
        return _Surd(self.rational * other.rational + self.radical * other.radical * self.m,
                     self.rational * other.radical + self.radical * other.rational, self.m)

    def __add__(self, other: _Surd) -> _Surd:
        return _Surd(self.rational + other.rational, self.radical + other.radical, self.m)


def _evaluate_surd(p: Coefficients, x: _Surd) -> _Surd:
    total = _Surd(Fraction(0), Fraction(0), x.m)
    for value in reversed(p):
        total = total * x + _Surd(value, Fraction(0), x.m)
    return total


def _surd_text(rational: Fraction, radical: Fraction, m: int) -> str:
    """a + b*sqrt(m) as one fraction ADA reads back: (3 + sqrt(5))/2, -2*sqrt(3)/3."""
    denominator = math.lcm(rational.denominator, radical.denominator)
    top_rational, top_radical = int(rational * denominator), int(radical * denominator)
    size = abs(top_radical)
    radical_text = f"sqrt({m})" if size == 1 else f"{size}*sqrt({m})"
    if top_rational == 0:
        numerator = radical_text if top_radical > 0 else f"-{radical_text}"
        return numerator if denominator == 1 else f"{numerator}/{denominator}"
    numerator = f"{top_rational} {'+' if top_radical > 0 else '-'} {radical_text}"
    return numerator if denominator == 1 else f"({numerator})/{denominator}"


def _number_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    if _terminates(value):
        decimal = _exact_decimal(value)
        if len(decimal.replace("-", "").replace(".", "").lstrip("0")) <= 12:
            return decimal
    return f"{value.numerator}/{value.denominator}"


def _decimal_of(text: str, digits: int) -> str:
    """The decimal of an exact root, worked out by step 1's arithmetic from the text ADA wrote."""
    result = compute(text, digits=digits)
    if hasattr(result, "decimal"):
        return result.decimal
    return text


# --- Real roots ------------------------------------------------------------------------------------------


def real_roots(coefficients: Coefficients, letter: str = "x",
               digits: int = DIGITS) -> tuple[list[Root], list[str]]:
    """The real roots of a polynomial, smallest first, with the steps that found and checked them.

    A root that comes from a repeated factor carries its multiplicity. A
    polynomial of degree above ``MAX_ROOT_DEGREE`` is refused as too large.
    """
    p = _trim([Fraction(value) for value in coefficients])
    if _degree(p) < 1:
        _refuse("invalid_expression", "The polynomial has no letter, so it has no roots to find.")
    if _degree(p) > MAX_ROOT_DEGREE:
        _refuse("too_large", f"{polynomial_text(p, letter)} = 0 has degree {_degree(p)}; ADA looks for the "
                             f"roots of polynomials up to degree {MAX_ROOT_DEGREE}.")
    steps: list[str] = []
    found: list[tuple[Fraction, Root]] = []
    factors = _square_free(p)
    if len(factors) > 1 or factors[0][1] > 1:
        written = " * ".join(_factor_text(polynomial_text(_primitive(factor), letter), times)
                             for factor, times in factors)
        steps.append(f"Repeated factors: {polynomial_text(_primitive(p), letter)} = {written}, up to a whole "
                     "number.")
    for factor, times in factors:
        for sort_key, root in _roots_of_square_free(_primitive(factor), letter, digits, steps):
            found.append((sort_key, Root(root.value, root.exact, root.decimal, times, root.bracket)))
    found.sort(key=lambda item: item[0])
    return [root for _, root in found], steps


def _roots_of_square_free(p: Coefficients, letter: str, digits: int, steps: list[str]):
    """(sort key, Root) pairs for a whole-number polynomial with no repeated root."""
    text = polynomial_text(p, letter)
    results = []
    # A root of 0 first: x^k divides out.
    while p and p[0] == 0:
        results.append((Fraction(0), Root("0", True, "0")))
        steps.append(f"{letter} = 0 is a root." if text == letter else
                     f"{letter} = 0 is a root of {text}, since it has no constant term.")
        p = p[1:]
    if _degree(p) < 1:
        return results
    rational, p = _rational_roots(p, letter, steps)
    results += [(value, Root(_number_text(value), True, _decimal_of(_number_text(value), digits)))
                for value in rational]
    degree = _degree(p)
    if degree == 1:
        value = -p[0] / p[1]
        results.append((value, Root(_number_text(value), True, _decimal_of(_number_text(value), digits))))
    elif degree == 2:
        results += _quadratic(p, letter, digits, steps)
    elif degree >= 3:
        results += _binomial(p, letter, digits, steps) if all(value == 0 for value in p[1:-1]) else \
            _isolated(p, letter, digits, steps)
    return results


def _rational_roots(p: Coefficients, letter: str, steps: list[str]) -> tuple[list[Fraction], Coefficients]:
    """Rational roots p/q, with p dividing the constant term and q the leading coefficient, divided out."""
    tops, bottoms = _divisors(int(p[0])), _divisors(int(p[-1]))
    text = polynomial_text(p, letter)
    if tops is None or bottoms is None:
        steps.append(f"The coefficients of {text} are too large to list its rational candidates, so none is "
                     "looked for.")
        return [], p
    if len(tops) * len(bottoms) * 2 > MAX_CANDIDATES:
        candidates = _candidates_near_roots(p, bottoms)
        steps.append(f"Rational roots of {text} = 0 are among ±(divisors of {int(abs(p[0]))})/(divisors of "
                     f"{int(p[-1])}); there are too many to try each, so only those inside a narrow interval "
                     "around each real root are tried.")
        listed = False
    else:
        candidates = sorted({sign * Fraction(top, bottom)
                             for top in tops for bottom in bottoms for sign in (1, -1)})
        listed = True
    roots = [value for value in candidates if _evaluate(p, value) == 0]
    if listed and len(candidates) <= 24:
        shown = ", ".join(_number_text(value) for value in candidates)
        steps.append(f"Rational roots of {text} = 0 are among ±(divisors of {int(abs(p[0]))})/(divisors of "
                     f"{int(p[-1])}): {shown}.")
    elif listed:
        steps.append(f"Rational roots of {text} = 0 are among the {len(candidates)} fractions ±(divisors of "
                     f"{int(abs(p[0]))})/(divisors of {int(p[-1])}).")
    if roots:
        steps.append(", ".join(f"{letter} = {_number_text(value)}" for value in roots)
                     + f" make{'s' if len(roots) == 1 else ''} it 0.")
        for value in roots:
            p, _ = _divmod(p, [-value, Fraction(1)])
        if _degree(p) > 0:
            steps.append(f"Dividing those out leaves {polynomial_text(_primitive(p), letter)} = 0.")
        p = _primitive(p) if _degree(p) > 0 else p
    elif _degree(p) > 1:
        steps.append("None of them makes it 0.")
    return roots, p


def _candidates_near_roots(p: Coefficients, bottoms: list[int]) -> list[Fraction]:
    """The fractions top/bottom, bottom dividing the leading coefficient, inside a narrow interval around each
    real root: two such fractions with a given bottom are at least 1/bottom apart."""
    sequence = _sturm(_trim(p))
    bound = 1 + max(abs(value / p[-1]) for value in p[:-1])
    width = Fraction(1, 2 * max(bottoms))
    candidates = set()
    pending = [(-bound, bound)]
    while pending:
        low, high = pending.pop()
        inside = _sign_changes(sequence, low) - _sign_changes(sequence, high)
        if inside == 0:
            continue
        if high - low > width:
            middle = (low + high) / 2
            pending += [(low, middle), (middle, high)]
            continue
        for bottom in bottoms:
            for top in range(math.floor(low * bottom), math.ceil(high * bottom) + 1):
                candidates.add(Fraction(top, bottom))
    return sorted(candidates)


def _quadratic(p: Coefficients, letter: str, digits: int, steps: list[str]):
    c, b, a = p
    discriminant = b * b - 4 * a * c
    text = polynomial_text(p, letter)
    if discriminant < 0:
        steps.append(f"{text} = 0: the discriminant b^2 - 4ac = {_number_text(discriminant)} is negative, so "
                     "no real number is a root. Its roots are the complex numbers "
                     f"{_complex(a, b, discriminant)}.")
        return []
    k, m = _square_part(discriminant.numerator * discriminant.denominator)
    radical = Fraction(k, discriminant.denominator) / (2 * a)
    rational = -b / (2 * a)
    both = _surd_text(rational, radical, m)
    both = both.replace(" + ", " ± ", 1) if rational else f"±{both}"
    steps.append(f"{text} = 0 by the quadratic formula: the discriminant b^2 - 4ac = "
                 f"{_number_text(discriminant)}, so {letter} = (-b ± sqrt(b^2 - 4ac))/(2a) = {both}.")
    results = []
    for sign in (-1, 1):
        surd = _Surd(rational, sign * radical, m)
        written = _surd_text(surd.rational, surd.radical, m)
        check = _evaluate_surd(p, surd)
        holds = check.rational == 0 and check.radical == 0
        steps.append(f"Check {letter} = {written} in {text}: it gives exactly 0, which holds." if holds else
                     f"Check {letter} = {written} in {text}: it does NOT give 0.")
        value = Fraction(_decimal_of(written, digits + 5))
        results.append((value, Root(written, True, _decimal_of(written, digits))))
    return results


def _complex(a: Fraction, b: Fraction, discriminant: Fraction) -> str:
    k, m = _square_part((-discriminant).numerator * (-discriminant).denominator)
    radical = Fraction(k, (-discriminant).denominator) / (2 * a)
    rational = -b / (2 * a)
    imaginary = (f"{_number_text(radical)}*" if radical != 1 else "") + (f"sqrt({m})*i" if m != 1 else "i")
    return f"{_number_text(rational)} ± {imaginary}" if rational else f"±{imaginary}"


def _binomial(p: Coefficients, letter: str, digits: int, steps: list[str]):
    """a*x^n + c = 0 with no rational root: x^n = t, so x = root(t, n), and -root(t, n) when n is even."""
    n = _degree(p)
    t = -p[0] / p[-1]
    text = polynomial_text(p, letter)
    if t < 0 and n % 2 == 0:
        steps.append(f"{text} = 0 means {letter}^{n} = {_number_text(t)}, and no real number to an even "
                     "power is negative, so it has no real root.")
        return []
    size = _number_text(abs(t))
    written = f"root({size}, {n})" if t > 0 else f"-root({size}, {n})"
    values = [written] if n % 2 else [f"-root({size}, {n})", f"root({size}, {n})"]
    steps.append(f"{text} = 0 means {letter}^{n} = {_number_text(t)}, so {letter} = "
                 + (" or ".join(values)) + f": by definition, root(t, {n})^{n} = t.")
    results = []
    for value in values:
        decimal = _decimal_of(value, digits)
        results.append((Fraction(_decimal_of(value, digits + 5)), Root(value, True, decimal)))
    return results


# --- Roots that have no simpler form ------------------------------------------------------------------------


def _sturm(p: Coefficients) -> list[Coefficients]:
    sequence = [_trim(p), _derivative(p)]
    while _degree(sequence[-1]) > 0:
        _, remainder = _divmod(sequence[-2], sequence[-1])
        if not remainder:
            break
        # A positive multiple keeps the signs and stops the fractions from growing.
        remainder = [-value for value in remainder]
        scale = math.lcm(*(value.denominator for value in remainder))
        whole = [value * scale for value in remainder]
        common = math.gcd(*(int(value) for value in whole)) or 1
        sequence.append([value / common for value in whole])
    return sequence


def _sign_changes(sequence: list[Coefficients], x: Fraction) -> int:
    signs = [value for value in (_evaluate(p, x) for p in sequence) if value != 0]
    return sum(1 for first, second in zip(signs, signs[1:], strict=False) if (first > 0) != (second > 0))


def _isolated(p: Coefficients, letter: str, digits: int, steps: list[str]):
    """Real roots counted by Sturm's theorem, each found by halving an interval where p changes sign."""
    sequence = _sturm(p)
    bound = 1 + max(abs(value / p[-1]) for value in p[:-1])
    count = _sign_changes(sequence, -bound) - _sign_changes(sequence, bound)
    text = polynomial_text(p, letter)
    if count == 0:
        steps.append(f"{text} = 0 has no rational root, and Sturm's theorem counts no real root between "
                     f"-{_number_text(bound)} and {_number_text(bound)}, where every root lies.")
        return []
    intervals, pending = [], [(-bound, bound)]
    while pending:
        low, high = pending.pop()
        inside = _sign_changes(sequence, low) - _sign_changes(sequence, high)
        if inside == 1:
            intervals.append((low, high))
        elif inside > 1:
            middle = (low + high) / 2
            pending += [(low, middle), (middle, high)]
    results = []
    for low, high in sorted(intervals):
        low, high = _narrowed(p, low, high, digits)
        middle = (low + high) / 2
        decimal = _decimal_text(_as_decimal(middle, digits + 5), digits)
        # The ends are rounded outwards, so the interval written still holds the sign change.
        shown_low, shown_high = _outward(low, digits + 2, "down"), _outward(high, digits + 2, "up")
        if not (_evaluate(p, Fraction(shown_low)) * _evaluate(p, Fraction(shown_high)) < 0):
            _refuse("needs_more_digits", f"A root of {text} = 0 could not be bracketed at {digits} digits.")
        results.append((middle, Root(decimal, False, decimal, bracket=(shown_low, shown_high))))
    steps.append(f"{text} = 0 has no rational root, and Sturm's theorem counts {count} real root"
                 f"{'' if count == 1 else 's'}; each is found by halving an interval where {text} changes "
                 f"sign, to {digits} significant digits.")
    for _, root in results:
        steps.append(f"Check {letter} ≈ {root.value}: {text} changes sign between {root.bracket[0]} and "
                     f"{root.bracket[1]}, so a root lies between them.")
    return results


def _narrowed(p: Coefficients, low: Fraction, high: Fraction, digits: int) -> tuple[Fraction, Fraction]:
    """Halve [low, high], keeping a sign change of p inside, until both ends round to the same digits."""
    low_value = _evaluate(p, low)
    if low_value == 0:
        return low, low
    for _ in range(MAX_HALVINGS):
        if _rounded(low, digits) == _rounded(high, digits):
            break
        middle = (low + high) / 2
        value = _evaluate(p, middle)
        if value == 0:
            return middle, middle
        if (value > 0) == (low_value > 0):
            low, low_value = middle, value
        else:
            high = middle
    return low, high


def _rounded(value: Fraction, digits: int) -> str:
    return _decimal_text(_as_decimal(value, digits + 5), digits)


def _as_decimal(value: Fraction, digits: int):
    from decimal import Decimal, localcontext  # noqa: PLC0415

    with localcontext() as context:
        context.prec = digits
        return Decimal(value.numerator) / Decimal(value.denominator)


def _outward(value: Fraction, digits: int, direction: str) -> str:
    """value to ``digits`` significant digits, rounded down or up."""
    from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext  # noqa: PLC0415

    with localcontext() as context:
        context.prec = digits
        context.rounding = ROUND_FLOOR if direction == "down" else ROUND_CEILING
        rounded = Decimal(value.numerator) / Decimal(value.denominator)
    return format(rounded, "f")
