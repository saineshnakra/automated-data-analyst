"""Systems of polynomial equations with finitely many solutions, solved for their real solutions.

    import ada
    ada.solve("x*y = 6; x + y = 5")            # (x = 2, y = 3) or (x = 3, y = 2)
    ada.solve("x^2 + y^2 = 25; x + y = 7")     # (x = 3, y = 4) or (x = 4, y = 3)

SymPy eliminates the unknowns with a Gröbner basis in lex order. When the system has finitely many solutions
and one letter tells them apart, the basis is one equation in that letter, g = 0, and each other unknown as a
polynomial in it. When no unknown tells them apart, the letter is a combination such as u = x + 2y. ADA finds
the real roots of g (``roots.factored_roots``) and works out each unknown from them: exactly when the root
is a fraction, a + b*sqrt(m) or root(t, n), and otherwise to the digits asked for, inside an interval worked
out with exact fractions.

Every solution is checked with ADA's own arithmetic. Each equation, with the unknowns written as polynomials
in the letter, must leave no remainder when divided by the factor of g the root comes from, so it is 0 at
every root of that factor. Rational solutions are also put back into the equations as numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

from . import roots as rooting
from .compute import _refuse

#: The most unknowns in a system that is not linear.
MAX_UNKNOWNS = 6
#: The largest product of the equations' powers (Bezout's bound on the number of solutions). Lex Gröbner
#: bases grow fast with it; at 36 the slowest systems tried take a few seconds.
MAX_SOLUTIONS = 36
#: Up to this bound every unknown is tried as the letter; above it, the first that works is kept.
TRY_EVERY_LETTER = 16
#: Combinations u = w1*x + w2*y + ... tried in turn when no single unknown tells the solutions apart.
_WEIGHTS = ((1, 2, 3, 5, 7, 11), (1, 3, 7, 13, 19, 29), (1, -2, 5, -9, 14, -20), (2, 5, 11, 17, 23, 31))
#: Letters for the combination, the first one the equations do not use.
_SPARE_LETTERS = "uvwtsrqpnmkjhgfdcba"


@dataclass(frozen=True)
class _Value:
    text: str
    exact: bool
    decimal: str
    bracket: tuple[str, str] | None = None


@dataclass(frozen=True)
class _Shape:
    """g(letter) = 0 and each unknown as a polynomial in the letter (constant term first)."""

    letter: str
    combination: str | None
    g: list
    formulas: dict


def solve_system(parsed: list, unknowns: list[str], zeros: list, digits: int, steps: list[str]):
    """The real solutions of equations that are not all linear, with finitely many solutions; ``zeros`` are
    the equations with denominators multiplied away and everything moved to one side."""
    from .algebra import (  # noqa: PLC0415
        Quotient,
        Solution,
        _capped,
        _checks,
        _conditions,
        _equation_letters,
        _unique,
    )

    written = [equation.written for equation in parsed]
    reading = [equation.reading for equation in parsed]
    assumptions = _unique([item for equation in parsed for item in equation.assumptions])
    others = sorted({name for equation in parsed for name in _equation_letters(equation)} - set(unknowns))
    if others:
        names = " and ".join(others)
        _refuse("unsupported_expression",
                f"The equations are not linear in {' and '.join(unknowns)}, and they also have {names}. "
                f"ADA solves such a system when its only letters are the unknowns: give {names} a value, or "
                "solve for every letter.")
    if len(unknowns) > MAX_UNKNOWNS:
        _refuse("too_large", f"The system has {len(unknowns)} unknowns; ADA solves systems that are not "
                             f"linear in up to {MAX_UNKNOWNS}.")
    nonzero = [zero for zero in zeros if not zero.is_zero()]
    powers = sorted((max(zero.degree(), 1) for zero in nonzero), reverse=True)[:len(unknowns)]
    if math.prod(powers) > MAX_SOLUTIONS:
        _refuse("too_large", f"The equations could have up to {math.prod(powers)} solutions (the product of "
                             f"their powers); ADA solves systems that are not linear with up to "
                             f"{MAX_SOLUTIONS}.")
    numbered = len(parsed) > 1
    moved = ", ".join(f"({index}) {zero} = 0" if numbered else f"{zero} = 0"
                      for index, zero in enumerate(zeros, 1))
    how = ("Multiply by the denominators and move everything to one side"
           if any(equation.denominators for equation in parsed) else "Move everything to one side")
    steps.append(f"{how}: {moved}.")
    denominators = list(dict.fromkeys(item for equation in parsed for item in equation.denominators))
    conditions = _conditions(denominators)

    def finish(kind: str, reason: str | None = None, **fields) -> Solution:
        return Solution(written, reading, list(unknowns), kind, reason=reason, steps=_capped(steps),
                        assumptions=assumptions, **fields)

    if not nonzero:
        return finish("many", "every equation holds for any value.", free=list(unknowns),
                      conditions=conditions)
    shape = _eliminate(nonzero, unknowns, math.prod(powers), steps)
    if shape is None:
        return finish("none", "eliminating the unknowns leaves 1 = 0, so no values, real or complex, satisfy "
                              "every equation.")
    found: list[tuple[tuple, dict[str, _Value]]] = []
    dropped = False
    g_text = rooting.polynomial_text(shape.g, shape.letter)
    for factor, _, pairs in rooting.factored_roots(shape.g, shape.letter, digits, steps):
        if not pairs:
            continue
        factor_text = rooting.polynomial_text(factor, shape.letter)
        residues = {name: _reduced(formula, factor) for name, formula in shape.formulas.items()}
        for equation, zero in zip(parsed, zeros, strict=True):
            if _composed(zero, residues, factor):
                _refuse("unsupported_expression",
                        f"Putting the solutions from {factor_text} = 0 into {equation.written} does not give "
                        "0, so ADA gives no answer.")
        if any(not _composed(denominator, residues, factor) for denominator in denominators):
            steps.append(f"The values from {factor_text} = 0 make a denominator zero, and the equations "
                         "divide by it, so they are not solutions.")
            dropped = True
            continue
        if rooting._degree(factor) > 1:
            given = ", ".join(f"{name} = {rooting.polynomial_text(formula, shape.letter)}"
                              for name, formula in shape.formulas.items() if name != shape.letter)
            steps.append(f"Check: {f'with {given}, ' if given else ''}each equation leaves no remainder when "
                         f"divided by {factor_text}, so each holds at every root of {factor_text} = 0.")
        for sort_key, root in pairs:
            values = _values_at(factor, root, residues, unknowns, digits)
            if len(unknowns) > 1:
                shown = ", ".join(f"{name} {'=' if values[name].exact else '≈'} {values[name].text}"
                                  for name in unknowns if name != shape.letter)
                steps.append(f"{shape.letter} {'=' if root.exact else '≈'} {root.value} gives {shown}.")
            if all(value.exact and _is_fraction(value.text) for value in values.values()):
                steps += _checks(parsed, {name: Quotient.number(Fraction(values[name].text))
                                          for name in unknowns})
            order = tuple(Fraction(values[name].decimal) for name in unknowns) + (sort_key,)
            found.append((order, values))
    found.sort(key=lambda item: item[0])
    solutions = [values for _, values in found]
    if not solutions:
        reason = ("every real solution makes a denominator zero, and the equations divide by it." if dropped
                  else f"{g_text} = 0 has no real root, so no real values satisfy every equation.")
        return finish("none", reason)
    kind = "one" if len(solutions) == 1 else "several"
    if len(unknowns) == 1:
        name = unknowns[0]
        listed = [{"value": values[name].text, "exact": values[name].exact, "decimal": values[name].decimal,
                   "multiplicity": 1, "bracket": list(values[name].bracket) if values[name].bracket else None}
                  for values in solutions]
        single = {"values": {name: listed[0]["value"]},
                  "decimals": _decimals(solutions[0])} if kind == "one" else {}
        return finish(kind, conditions=conditions, roots=listed, **single)
    entries = [_entry(values, unknowns) for values in solutions]
    single = {"values": entries[0]["values"], "decimals": _decimals(solutions[0])} if kind == "one" else {}
    return finish(kind, conditions=conditions, solutions=entries, **single)


def _entry(values: dict[str, _Value], unknowns: list[str]) -> dict:
    return {"values": {name: values[name].text for name in unknowns},
            "exact": {name: values[name].exact for name in unknowns},
            "decimals": {name: values[name].decimal for name in unknowns},
            "brackets": {name: list(values[name].bracket) for name in unknowns if values[name].bracket}}


def _decimals(values: dict[str, _Value]) -> dict[str, str]:
    return {name: value.decimal for name, value in values.items()
            if value.exact and value.decimal != value.text}


def _is_fraction(text: str) -> bool:
    try:
        Fraction(text)
    except ValueError:
        return False
    return True


# --- Elimination ----------------------------------------------------------------------------------------


def _eliminate(polynomials: list, unknowns: list[str], bound: int, steps: list[str]) -> _Shape | None:
    """The system as g(letter) = 0 and each unknown a polynomial in the letter, from SymPy's lex Gröbner
    basis; None when the basis is 1 (no solution at all).

    Up to ``TRY_EVERY_LETTER`` solutions, each unknown is tried as the letter and the one whose g leaves the
    fewest factors without an exact form is kept. When no unknown tells the solutions apart, a solution
    counted more than once is first counted once (the square-free part of the equation each unknown
    satisfies is added, which leaves the solutions as they are), and then a combination such as u = x + 2y is
    the letter.
    """
    from sympy import Poly, Symbol, groebner, sqf_part  # noqa: PLC0415

    symbols = {name: Symbol(name) for name in unknowns}
    expressions = [_to_sympy(polynomial, symbols) for polynomial in polynomials]
    orders = [unknowns[:index] + unknowns[index + 1:] + [unknowns[index]]
              for index in reversed(range(len(unknowns)))]

    def basis_for(order: list[str], given: list):
        return groebner(given, *(symbols[name] for name in order), order="lex")

    def by_unknown(given: list) -> tuple[list, list]:
        candidates, bases = [], []
        for number, order in enumerate(orders):
            if candidates and (candidates[0][0] == 0 or bound > TRY_EVERY_LETTER):
                break
            basis = basis_for(order, given)
            bases.append(basis)
            if number == 0 and list(basis.exprs) == [1]:
                return [], [basis]
            if number == 0 and not basis.is_zero_dimensional:
                _refuse("unsupported_expression",
                        "The equations leave infinitely many solutions, counting complex ones, so there is "
                        "no list of solutions to give. ADA solves a system that is not linear when it has "
                        f"finitely many; with {len(unknowns)} unknowns that usually takes {len(unknowns)} "
                        "equations.")
            found = _shape(basis, [symbols[name] for name in order])
            if found is not None:
                g, formulas = found
                letter = order[-1]
                formulas = {name: formulas[name] if name != letter else [Fraction(0), Fraction(1)]
                            for name in unknowns}
                candidates.append((_inexact_factors(g), number, _Shape(letter, None, g, formulas)))
        return candidates, bases

    candidates, bases = by_unknown(expressions)
    if list(bases[0].exprs) == [1]:
        return None
    if not candidates:
        bases += [basis_for(order, expressions) for order in orders[len(bases):]]
        added, shown = [], []
        for order, basis in zip(orders, bases, strict=True):
            eliminant = Poly(basis.exprs[-1], symbols[order[-1]])
            squarefree = Poly(sqf_part(eliminant), symbols[order[-1]])
            if squarefree.degree() < eliminant.degree():
                added.append(squarefree.as_expr())
                shown.append(f"{rooting.polynomial_text(_coefficients(squarefree), order[-1])} = 0")
        if added:
            expressions = expressions + added
            steps.append(f"Some solutions are counted more than once. Adding {', '.join(shown)} (the "
                         "equations the unknowns satisfy, with each root once) counts each once and changes "
                         "no solution.")
            candidates, _ = by_unknown(expressions)
    if candidates:
        shape = min(candidates, key=lambda item: item[:2])[2]
        _say(shape, unknowns, steps)
        return shape
    letter = next(name for name in _SPARE_LETTERS if name not in unknowns)
    u = Symbol(letter)
    for weights in _WEIGHTS:
        combination = sum(weight * symbols[name] for weight, name in zip(weights, unknowns, strict=False))
        basis = groebner([*expressions, u - combination], *(symbols[name] for name in unknowns), u,
                         order="lex")
        found = _shape(basis, [*(symbols[name] for name in unknowns), u])
        if found is not None:
            g, formulas = found
            shape = _Shape(letter, str(combination).replace("*", ""), g,
                           {name: formulas[name] for name in unknowns})
            _say(shape, unknowns, steps)
            return shape
    _refuse("unsupported_expression",
            "ADA could not bring the equations down to one equation in one letter with the other unknowns "
            "given by it, so it gives no answer.")


def _inexact_factors(g: list[Fraction]) -> int:
    """How many of g's factors over the rationals have roots with no exact form ADA writes."""
    return sum(1 for factor, _ in rooting.factor_over_rationals(g)
               if rooting._degree(factor) > 2 and any(value != 0 for value in factor[1:-1]))


def _say(shape: _Shape, unknowns: list[str], steps: list[str]) -> None:
    g_text = f"{rooting.polynomial_text(shape.g, shape.letter)} = 0"
    others = [f"{name} = {rooting.polynomial_text(formula, shape.letter)}"
              for name, formula in shape.formulas.items() if name != shape.letter]
    if shape.combination:
        steps.append(f"No single unknown tells the solutions apart, so let {shape.letter} = "
                     f"{shape.combination}. Eliminating with a Gröbner basis (SymPy, lex order) gives "
                     f"{g_text}, with {', '.join(others)}.")
    elif others:
        steps.append(f"Eliminate {' and '.join(name for name in unknowns if name != shape.letter)} with a "
                     f"Gröbner basis (SymPy, lex order): {g_text}, with {', '.join(others)}.")
    else:
        steps.append(f"The equations have the same solutions as {g_text} (a Gröbner basis, SymPy).")


def _shape(basis, gens: list) -> tuple[list[Fraction], dict[str, list[Fraction]]] | None:
    """g and the formulas when the reduced basis is g(t) and x - h(t) for each other unknown x, else None."""
    from sympy import Poly  # noqa: PLC0415

    last = gens[-1]
    expressions = list(basis.exprs)
    if len(expressions) != len(gens):
        return None
    g, formulas = None, {}
    for expression in expressions:
        free = expression.free_symbols
        if free <= {last}:
            if g is not None:
                return None
            g = _coefficients(Poly(expression, last))
            continue
        others = free - {last}
        if len(others) != 1:
            return None
        (symbol,) = others
        poly = Poly(expression, symbol)
        if poly.degree() != 1:
            return None
        lead, rest = poly.all_coeffs()
        if not lead.is_Number:
            return None
        formulas[str(symbol)] = _coefficients(Poly(-rest / lead, last))
    if g is None or len(formulas) != len(gens) - 1 or rooting._degree(g) < 1:
        return None
    return rooting._primitive(g), formulas


def _coefficients(poly) -> list[Fraction]:
    return rooting._trim([Fraction(int(value.p), int(value.q)) for value in reversed(poly.all_coeffs())])


def _to_sympy(polynomial, symbols: dict):
    from sympy import Poly, Rational  # noqa: PLC0415

    names = list(symbols)
    terms = {}
    for monomial, value in polynomial.terms.items():
        powers = dict(monomial)
        terms[tuple(powers.get(name, 0) for name in names)] = Rational(value.numerator, value.denominator)
    return Poly.from_dict(terms, *symbols.values()).as_expr()


# --- Arithmetic modulo a factor -----------------------------------------------------------------------


def _reduced(p: list[Fraction], modulus: list[Fraction]) -> list[Fraction]:
    return rooting._divmod(p, modulus)[1] if rooting._trim(p) else []


def _composed(polynomial, residues: dict[str, list[Fraction]], modulus: list[Fraction]) -> list[Fraction]:
    """The polynomial with each unknown replaced by its formula, as the remainder on division by modulus."""
    total: list[Fraction] = []
    for monomial, coefficient in polynomial.terms.items():
        term = [Fraction(coefficient)]
        for name, power in monomial:
            for _ in range(power):
                if not term or not residues[name]:
                    term = []
                    break
                term = _reduced(rooting._times(term, residues[name]), modulus)
        total = _plus(total, term)
    return _reduced(total, modulus)


def _plus(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    size = max(len(first), len(second))
    return rooting._trim([(first[i] if i < len(first) else 0) + (second[i] if i < len(second) else 0)
                          for i in range(size)])


# --- Values at a root -------------------------------------------------------------------------------------


def _values_at(factor: list[Fraction], root, residues: dict[str, list[Fraction]], unknowns: list[str],
               digits: int) -> dict[str, _Value]:
    """Each unknown's value at one root of factor: exact when the root has an exact form, else approximate."""
    values: dict[str, _Value] = {}
    for name in unknowns:
        residue = residues[name]
        if len(residue) <= 1:
            values[name] = _rational(residue[0] if residue else Fraction(0), digits)
    pending = {name: residues[name] for name in unknowns if name not in values}
    if not pending:
        return values
    degree = rooting._degree(factor)
    if degree == 1:
        t = -factor[0] / factor[1]
        values.update({name: _rational(rooting._evaluate(residue, t), digits)
                       for name, residue in pending.items()})
    elif degree == 2:
        lower, upper = rooting.quadratic_surds(factor)
        surd = lower if rooting._surd_text(lower.rational, lower.radical, lower.m) == root.value else upper
        for name, residue in pending.items():
            at = rooting._evaluate_surd(residue, surd)
            if at.radical == 0:
                values[name] = _rational(at.rational, digits)
            else:
                text = rooting._surd_text(at.rational, at.radical, at.m)
                values[name] = _Value(text, True, rooting._decimal_of(text, digits))
    elif root.exact:
        for name, residue in pending.items():
            text = _power_text(residue, root.value)
            values[name] = _Value(text, True, rooting._decimal_of(text, digits))
    else:
        values.update(_approximate(factor, root, pending, digits))
    return values


def _rational(value: Fraction, digits: int) -> _Value:
    text = rooting._number_text(value)
    return _Value(text, True, rooting._decimal_of(text, digits))


def _power_text(residue: list[Fraction], root_text: str) -> str:
    """c0 + c1*t + c2*t^2 + ... written with t as the root's text, so compute reads it back."""
    base = f"({root_text})" if root_text.startswith("-") else root_text
    terms = []
    for power in reversed(range(len(residue))):
        value = residue[power]
        if value == 0:
            continue
        size = rooting._number_text(abs(value))
        if power == 0:
            body = size
        else:
            raised = base if power == 1 else f"{base}^{power}"
            body = raised if abs(value) == 1 else f"{size}*{raised}"
        terms.append(("-" if value < 0 else "+", body))
    first_sign, first_body = terms[0]
    rest = "".join(f" {sign} {body}" for sign, body in terms[1:])
    return ("-" if first_sign == "-" else "") + first_body + rest


def _approximate(factor: list[Fraction], root, pending: dict[str, list[Fraction]],
                 digits: int) -> dict[str, _Value]:
    """Each value to ``digits`` significant digits: the root's interval is narrowed until the interval each
    value lies in, worked out with exact fractions, rounds to one decimal."""
    low, high = (Fraction(end) for end in root.bracket)
    values: dict[str, _Value] = {}
    extra = 4
    while pending:
        if extra > 256:
            _refuse("needs_more_digits", f"A solution from {rooting.polynomial_text(factor, 'x')} = 0 could "
                                         f"not be pinned down to {digits} digits.")
        low, high = rooting._narrowed(factor, low, high, digits + extra)
        for name, residue in list(pending.items()):
            bottom, top = _interval(residue, low, high)
            shown = rooting._rounded(bottom, digits)
            if shown == rooting._rounded(top, digits):
                bracket = (rooting._outward(bottom, digits + 2, "down"),
                           rooting._outward(top, digits + 2, "up"))
                values[name] = _Value(shown, False, shown, bracket)
                del pending[name]
        extra *= 2
    return values


def _interval(p: list[Fraction], low: Fraction, high: Fraction) -> tuple[Fraction, Fraction]:
    """An interval holding p(t) for every t in [low, high], by Horner's rule on intervals."""
    bottom = top = Fraction(0)
    for value in reversed(p):
        products = (bottom * low, bottom * high, top * low, top * high)
        bottom, top = min(products) + value, max(products) + value
    return bottom, top
