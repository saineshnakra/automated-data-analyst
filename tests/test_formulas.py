"""Equations in one unknown with other letters in them: ada.solve with unknowns named, for formulas."""

import random
import unittest
from decimal import Decimal
from unittest import mock

import numpy as np
from sympy import Integer, Symbol

from ada import compute, solve
from ada.algebra import work_out
from ada.api import REFUSAL_CODES, Refusal
from ada.formulas import MAX_LETTERS, MAX_POWER

x_symbol, a_symbol = Symbol("x"), Symbol("a")


def refused(result, code):
    assert isinstance(result, Refusal), result
    assert result.code == code and code in REFUSAL_CODES, (result.code, result.message)
    return result.message


def values(result):
    return [root["value"] for root in result.roots]


class QuadraticTests(unittest.TestCase):
    def test_the_quadratic_formula(self):
        result = solve("a*x^2 + b*x + c = 0", unknowns="x")
        self.assertEqual(values(result), ["(-b - sqrt(-4a*c + b^2))/(2a)", "(-b + sqrt(-4a*c + b^2))/(2a)"])
        self.assertEqual(result.conditions, ["a ≠ 0", "-4a*c + b^2 ≥ 0"])
        self.assertEqual(result.kind, "several")
        self.assertIn("This takes a ≠ 0: when a = 0, a*x^2 + b*x + c has a lower power of x, so solve it "
                      "again with that value.", result.steps)
        self.assertEqual(sum("with sqrt(-4a*c + b^2)^2 = -4a*c + b^2: it gives exactly 0, which holds" in step
                             for step in result.steps), 2)

    def test_square_factors_come_out_of_the_root(self):
        self.assertEqual(values(solve("x^2 - 2a*x + a^2 - b = 0", unknowns="x")),
                         ["a - sqrt(b)", "a + sqrt(b)"])
        self.assertEqual(values(solve("x^2 = 2a^2", unknowns="x")), ["-a*sqrt(2)", "a*sqrt(2)"])
        self.assertEqual(values(solve("x^2/a = 4", unknowns="x")), ["-2*sqrt(a)", "2*sqrt(a)"])

    def test_conditions_are_as_simple_as_they_can_be(self):
        self.assertEqual(solve("a*x^2 = 1", unknowns="x").conditions, ["a > 0"])
        self.assertEqual(solve("2x^2 = 4a", unknowns="x").conditions, ["a ≥ 0"])
        # a^2 + 1 is never negative, so it needs no condition.
        self.assertEqual(solve("x^2 - a^2 - 1 = 0", unknowns="x").conditions, [])

    def test_no_real_root(self):
        result = solve("x^2 + a^2 + 1 = 0", unknowns="x")
        self.assertEqual(result.kind, "none")
        self.assertTrue(any("negative for every value of a" in step for step in result.steps))

    def test_a_formula(self):
        result = solve("h = v*t - g*t^2/2", unknowns="t")
        self.assertEqual(values(result), ["(v - sqrt(-2g*h + v^2))/g", "(v + sqrt(-2g*h + v^2))/g"])
        self.assertEqual(result.conditions, ["g ≠ 0", "-2g*h + v^2 ≥ 0"])
        self.assertEqual(work_out("solve x^2 + y^2 = r^2 for y").answer,
                         "y = -sqrt(r^2 - x^2) or y = sqrt(r^2 - x^2) (when r^2 - x^2 ≥ 0)")


class FactorTests(unittest.TestCase):
    def test_factors_give_their_roots(self):
        result = solve("x^2 - (a + b)x + a*b = 0", unknowns="x")
        self.assertEqual((values(result), result.conditions), (["a", "b"], []))
        self.assertIn("Factor over the rationals (SymPy): x^2 - (a + b)x + a*b = (x - a) * (x - b).",
                      result.steps)
        self.assertEqual(values(solve("x^3 - a*x = 0", unknowns="x")), ["0", "-sqrt(a)", "sqrt(a)"])

    def test_numbers_come_first_in_order(self):
        result = solve("(x^2 - 2)(x - a) = 0", unknowns="x")
        self.assertEqual(values(result), ["-sqrt(2)", "sqrt(2)", "a"])
        self.assertEqual([root["decimal"] for root in result.roots][:2],
                         ["-1.4142135623730950488", "1.4142135623730950488"])

    def test_a_factor_in_the_other_letters_only(self):
        result = solve("a(x^2 - 1) = 0", unknowns="x")
        self.assertEqual((values(result), result.conditions), (["-1", "1"], ["a ≠ 0"]))
        self.assertIn("When a = 0, every value of x works, so the answer takes a ≠ 0.", result.steps)

    def test_repeated_factor(self):
        result = solve("(x - a)^2 = 0", unknowns="x")
        self.assertEqual((result.kind, result.values, result.roots[0]["multiplicity"]),
                         ("one", {"x": "a"}, 2))
        self.assertEqual(result.answer, "x = a (twice)")


class PowerTests(unittest.TestCase):
    def test_odd_and_even_powers(self):
        self.assertEqual(solve("x^3 = a", unknowns="x").answer, "x = root(a, 3)")
        even = solve("x^4 = 3a", unknowns="x")
        self.assertEqual((values(even), even.conditions), (["-root(3a, 4)", "root(3a, 4)"], ["a ≥ 0"]))
        self.assertEqual(solve("x^4 = a^2 + 1", unknowns="x").conditions, [])
        self.assertEqual(solve("x^4 = -a^2 - 1", unknowns="x").kind, "none")


class DenominatorTests(unittest.TestCase):
    def test_a_root_that_divides_by_zero_is_dropped(self):
        result = solve("(x^2 - a^2)/(x - a) = 0", unknowns="x")
        self.assertEqual((result.values, result.conditions), ({"x": "-a"}, ["x ≠ a"]))
        self.assertTrue(any("also make a denominator zero" in step for step in result.steps))

    def test_conditions_from_denominators(self):
        result = solve("1/x + 1/(x - a) = 1", unknowns="x")
        self.assertEqual(values(result), ["(a + 2 - sqrt(a^2 + 4))/2", "(a + 2 + sqrt(a^2 + 4))/2"])
        self.assertEqual(result.conditions, ["x ≠ 0", "x ≠ a"])


class RefusalTests(unittest.TestCase):
    def test_a_factor_of_degree_three_with_letters(self):
        message = refused(solve("x^3 + a*x + 1 = 0", unknowns="x"), "unsupported_expression")
        self.assertIn("of degree 3 in x with a in it", message)

    def test_a_real_root_only_for_some_values(self):
        message = refused(solve("x^2 + a^2 = 0", unknowns="x"), "unsupported_expression")
        self.assertIn("only when a = 0", message)

    def test_limits(self):
        letters = "abcdfgh"[:MAX_LETTERS]
        refused(solve("x^2 = " + " + ".join(letters), unknowns="x"), "too_large")
        refused(solve(f"x^{MAX_POWER + 1} = a", unknowns="x"), "too_large")

    def test_a_root_whose_check_fails_is_refused(self):
        with mock.patch("ada.formulas._surd_holds", return_value=False):
            message = refused(solve("a*x^2 + b*x + c = 0", unknowns="x"), "unsupported_expression")
        self.assertIn("does not give 0", message)
        wrong = (Integer(1), [(x_symbol - 2, Integer(1)), (a_symbol, Integer(1))])
        with mock.patch("sympy.factor_list", return_value=wrong):
            message = refused(solve("a*x^2 - 4a = 0", unknowns="x"), "unsupported_expression")
        self.assertIn("do not multiply back", message)

    def test_one_equation_in_several_letters_says_how_to_name_the_unknown(self):
        message = refused(solve("a*x^2 + b*x + c = 0"), "unsupported_expression")
        self.assertIn('"solve ... for x"', message)


class AgreementTests(unittest.TestCase):
    """The formulas with numbers put in for the letters, against the same equation solved with numbers, and
    against numpy."""

    def test_quadratics_with_a_letter_in_them(self):
        draw = random.Random(17)
        checked = 0
        while checked < 40:
            p, q, r, s = (draw.randint(-5, 5) for _ in range(4))
            if p == 0 and q == 0:
                continue
            equation = f"x^2 + ({p}a + {q})x + ({r}a + {s}) = 0"
            result = solve(equation, unknowns="x")
            if isinstance(result, Refusal) or len(result.roots) != 2 or \
                    "sqrt" not in result.roots[0]["value"]:
                continue
            for a in (-3, 1, 4):
                b, c = p * a + q, r * a + s
                reference = sorted(value.real for value in np.roots([1, b, c]) if abs(value.imag) < 1e-9)
                if b * b - 4 * c < 0:
                    self.assertEqual(reference, [])
                    continue
                numeric = solve(f"x^2 + ({b})x + ({c}) = 0")
                expected = [Decimal(root["decimal"]) for root in numeric.roots]
                found = sorted(Decimal(compute(root["value"].replace("a", f"({a})")).decimal)
                               for root in result.roots)
                if len(expected) == 1:
                    expected = expected * 2
                self.assertEqual(len(found), len(expected), (equation, a))
                for value, wanted, close in zip(found, expected, sorted(reference * (2 // len(reference))),
                                                strict=True):
                    self.assertLess(abs(value - wanted), Decimal("1e-18"), (equation, a))
                    self.assertAlmostEqual(float(value), close, places=6, msg=(equation, a))
            checked += 1


if __name__ == "__main__":
    unittest.main()
