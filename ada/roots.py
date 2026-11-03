"""The real roots of a polynomial in one letter with fraction coefficients, exact wherever they can be
written.

    from fractions import Fraction
    from ada.roots import real_roots
    roots, steps = real_roots([Fraction(1), Fraction(-3), Fraction(1)], "x")   # x^2 - 3x + 1
    [root.value for root in roots]        # ["(3 - sqrt(5))/2", "(3 + sqrt(5))/2"]

Coefficients run from the constant term up. The polynomial is factored over the rationals with SymPy, and
each factor's roots come back as a whole number or fraction, as a + b*sqrt(m) from the quadratic formula, as
root(t, n) when the factor is a*x^n + c, and otherwise as a decimal to the digits asked for, with an interval
where the polynomial changes sign: a root lies inside it.
Every exact root is put back into the polynomial with exact arithmetic, and the check is one of the steps.
Each value is written so that ``ada.compute`` reads it back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from .compute import DIGITS, _decimal_text, _exact_decimal, _refuse, _terminates, compute

#: The highest power of a polynomial whose roots are looked for.
MAX_ROOT_DEGREE = 100
#: Whole numbers up to this size are split into primes to simplify a square root.
FACTOR_LIMIT = 10 ** 30
#: Halvings of an interval before ADA stops narrowing a root.
MAX_HALVINGS = 2000
#: Up to this degree, Sturm's theorem counts the real roots too, as a check on SymPy's count.
STURM_DEGREE = 30

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


def _times(first: Coefficients, second: Coefficients) -> Coefficients:
    product = [Fraction(0)] * (len(first) + len(second) - 1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            product[i + j] += a * b
    return product


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


def _factor_text(text: str, times: int) -> str:
    wrapped = f"({text})" if " " in text else text
    return f"{wrapped}^{times}" if times > 1 else wrapped


def polynomial_text(p: Coefficients, letter: str) -> str:
    """p written highest power first, as ADA's algebra writes it."""
    from .algebra import Polynomial  # noqa: PLC0415

    return str(Polynomial({((letter, power),) if power else (): value for power, value in enumerate(p)}))


# --- Whole numbers --------------------------------------------------------------------------------------


def _factor(n: int) -> dict[int, int] | None:
    """The prime factors of |n|, from SymPy, or None when |n| is too large to factor quickly."""
    n = abs(n)
    if n > FACTOR_LIMIT:
        return None
    from sympy import factorint  # noqa: PLC0415

    return {int(prime): int(power) for prime, power in factorint(n).items()}


def factor_over_rationals(p: Coefficients) -> list[tuple[Coefficients, int]]:
    """p as a product of polynomials that do not factor further over the rationals, each with whole-number
    coefficients and how many times it appears, from SymPy's factor_list; lowest degree first.

    The factors are multiplied back together with exact arithmetic and must give p's whole-number form
    (``_primitive``) exactly, or the factoring is refused.
    """
    from sympy import Poly, Rational, Symbol  # noqa: PLC0415

    symbol = Symbol("x")
    poly = Poly([Rational(value.numerator, value.denominator) for value in reversed(_trim(p))], symbol)
    factors = []
    for factor, times in poly.factor_list()[1]:
        coefficients = [Fraction(int(value.p), int(value.q)) for value in reversed(factor.all_coeffs())]
        factors.append((_primitive(coefficients), int(times)))
    product = [Fraction(1)]
    for factor, times in factors:
        for _ in range(times):
            product = _times(product, factor)
    if _trim(product) != _primitive(p):
        _refuse("unsupported_expression", "The factors SymPy gives do not multiply back to the polynomial, "
                                          "so ADA gives no answer.")
    return sorted(factors, key=lambda item: (len(item[0]), [abs(value) for value in reversed(item[0])],
                                             list(reversed(item[0]))))


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

    The polynomial is factored over the rationals first (SymPy's factor_list), so a root that comes from a
    repeated factor carries its multiplicity, and a quadratic factor gives exact square roots wherever it
    sits. A polynomial of degree above ``MAX_ROOT_DEGREE`` is refused as too large.
    """
    p = _trim([Fraction(value) for value in coefficients])
    if _degree(p) < 1:
        _refuse("invalid_expression", "The polynomial has no letter, so it has no roots to find.")
    if _degree(p) > MAX_ROOT_DEGREE:
        _refuse("too_large", f"{polynomial_text(p, letter)} = 0 has degree {_degree(p)}; ADA looks for the "
                             f"roots of polynomials up to degree {MAX_ROOT_DEGREE}.")
    steps: list[str] = []
    found: list[tuple[Fraction, Root]] = []
    factors = factor_over_rationals(p)
    text = polynomial_text(_primitive(p), letter)
    if len(factors) > 1 or factors[0][1] > 1:
        written = " * ".join(_factor_text(polynomial_text(factor, letter), times)
                             for factor, times in factors)
        steps.append(f"Factor over the rationals: {text} = {written}.")
    elif _degree(p) > 2:
        steps.append(f"{text} does not factor over the rationals.")
    for factor, times in factors:
        for sort_key, root in _roots_of_factor(factor, letter, digits, steps):
            found.append((sort_key, Root(root.value, root.exact, root.decimal, times, root.bracket)))
    found.sort(key=lambda item: item[0])
    return [root for _, root in found], steps


def _roots_of_factor(p: Coefficients, letter: str, digits: int, steps: list[str]):
    """(sort key, Root) pairs for a whole-number polynomial that does not factor over the rationals."""
    degree = _degree(p)
    if degree == 1:
        value = -p[0] / p[1]
        return [(value, Root(_number_text(value), True, _decimal_of(_number_text(value), digits)))]
    if degree == 2:
        return _quadratic(p, letter, digits, steps)
    if all(value == 0 for value in p[1:-1]):
        return _binomial(p, letter, digits, steps)
    return _isolated(p, letter, digits, steps)


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
    """Real roots isolated by SymPy (and counted by Sturm's theorem as a check, up to ``STURM_DEGREE``), each
    narrowed by halving an interval where p changes sign."""
    from sympy import Poly, Rational, Symbol  # noqa: PLC0415

    text = polynomial_text(p, letter)
    poly = Poly([Rational(int(value)) for value in reversed(p)], Symbol("x"))
    intervals = [(Fraction(int(low.p), int(low.q)), Fraction(int(high.p), int(high.q)))
                 for (low, high), _ in poly.intervals()]
    how = "isolated by SymPy"
    if _degree(p) <= STURM_DEGREE:
        sequence = _sturm(p)
        bound = 1 + max(abs(value / p[-1]) for value in p[:-1])
        count = _sign_changes(sequence, -bound) - _sign_changes(sequence, bound)
        if count != len(intervals):
            _refuse("unsupported_expression",
                    f"Sturm's theorem counts {count} real roots of {text} = 0 and SymPy isolates "
                    f"{len(intervals)}; since the two disagree, ADA gives no answer.")
        how = "counted by Sturm's theorem and isolated by SymPy"
    if not intervals:
        steps.append(f"{text} = 0 has no real root ({how}).")
        return []
    results = []
    for low, high in sorted(intervals):
        if _evaluate(p, low) * _evaluate(p, high) > 0:
            _refuse("needs_more_digits", f"A root of {text} = 0 could not be bracketed at {digits} digits.")
        low, high = _narrowed(p, low, high, digits)
        middle = (low + high) / 2
        decimal = _decimal_text(_as_decimal(middle, digits + 5), digits)
        # The ends are rounded outwards, so the interval written still holds the sign change.
        shown_low, shown_high = _outward(low, digits + 2, "down"), _outward(high, digits + 2, "up")
        if not (_evaluate(p, Fraction(shown_low)) * _evaluate(p, Fraction(shown_high)) < 0):
            _refuse("needs_more_digits", f"A root of {text} = 0 could not be bracketed at {digits} digits.")
        results.append((middle, Root(decimal, False, decimal, bracket=(shown_low, shown_high))))
    count = len(results)
    steps.append(f"{text} = 0 has {count} real root{'' if count == 1 else 's'}, {how}; each is found by "
                 f"halving an interval where {text} changes sign, to {digits} significant digits.")
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
