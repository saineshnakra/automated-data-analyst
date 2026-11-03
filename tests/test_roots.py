"""The real roots of a polynomial in one letter, and the equations ada.solve sends to them."""

import math
import random
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from ada import compute, solve
from ada.api import REFUSAL_CODES, Refusal
from ada.roots import MAX_ROOT_DEGREE, real_roots


def roots_of(*coefficients):
    """The roots of the polynomial with these coefficients, constant term first."""
    found, steps = real_roots([Fraction(value) for value in coefficients], "x")
    return found, steps


def values(found):
    return [root.value for root in found]


def refused(result, code):
    assert isinstance(result, Refusal), result
    assert result.code == code and code in REFUSAL_CODES, (result.code, result.message)
    return result.message


class ExactRootTests(unittest.TestCase):
    def test_rational_roots(self):
        found, steps = roots_of(6, -5, 1)                        # x^2 - 5x + 6
        self.assertEqual(values(found), ["2", "3"])
        self.assertIn("x = 2, x = 3 make it 0.", steps)
        self.assertEqual(values(roots_of(-2, 3, 2)[0]), ["-2", "0.5"])   # 2x^2 + 3x - 2
        self.assertEqual(values(roots_of(-6, 11, -6, 1)[0]), ["1", "2", "3"])

    def test_the_quadratic_formula_with_a_square_root(self):
        found, steps = roots_of(1, -3, 1)                        # x^2 - 3x + 1
        self.assertEqual(values(found), ["(3 - sqrt(5))/2", "(3 + sqrt(5))/2"])
        self.assertEqual([root.decimal for root in found], ["0.3819660112501051518", "2.6180339887498948482"])
        self.assertTrue(all(root.exact for root in found))
        self.assertEqual(sum("gives exactly 0, which holds" in step for step in steps), 2)
        # Square factors come out of the root.
        self.assertEqual(values(roots_of(-8, 0, 1)[0]), ["-2*sqrt(2)", "2*sqrt(2)"])
        self.assertEqual(values(roots_of(-1, 0, 3)[0]), ["-sqrt(3)/3", "sqrt(3)/3"])

    def test_the_values_read_back_in_compute(self):
        for root in roots_of(1, -3, 1)[0] + roots_of(-2, 0, 0, 1)[0]:
            self.assertEqual(compute(root.value).decimal, root.decimal)

    def test_no_real_root(self):
        found, steps = roots_of(1, 0, 1)                         # x^2 + 1
        self.assertEqual(found, [])
        self.assertIn("complex numbers ±i", steps[-1])
        self.assertIn("1 ± sqrt(2)*i", roots_of(3, -2, 1)[1][-1])      # x^2 - 2x + 3

    def test_repeated_roots_carry_their_multiplicity(self):
        found, steps = roots_of(4, -4, 1)                        # (x - 2)^2
        self.assertEqual([(root.value, root.multiplicity) for root in found], [("2", 2)])
        self.assertTrue(steps[0].startswith("Repeated factors"))
        found, _ = roots_of(-2, 5, -3, -1, 1)                   # (x - 1)^3 (x + 2)
        self.assertEqual([(root.value, root.multiplicity) for root in found], [("-2", 1), ("1", 3)])
        found, _ = roots_of(0, 0, -2, 1)                         # x^2 (x - 2)
        self.assertEqual([(root.value, root.multiplicity) for root in found], [("0", 2), ("2", 1)])

    def test_powers_equal_to_a_number(self):
        self.assertEqual(values(roots_of(-2, 0, 0, 1)[0]), ["root(2, 3)"])
        self.assertEqual(values(roots_of(-16, 0, 0, 0, 1)[0]), ["-2", "2"])
        self.assertEqual(values(roots_of(-2, 0, 0, 0, 1)[0]), ["-root(2, 4)", "root(2, 4)"])
        self.assertEqual(values(roots_of(32, 0, 0, 0, 0, 1)[0]), ["-2"])      # x^5 = -32
        self.assertEqual(values(roots_of(3, 0, 0, 0, 1)[0]), [])             # x^4 = -3


    def test_too_many_candidates_to_try_each(self):
        result = solve("(720720x - 963761198400)(963761198400x + 1) = 0")
        self.assertEqual([root["value"] for root in result.roots], ["-1/963761198400", "1337220"])
        self.assertTrue(any("too many to try each" in step for step in result.steps))


class ApproximateRootTests(unittest.TestCase):
    def test_a_root_with_no_exact_form_is_bracketed(self):
        found, steps = roots_of(-1, -1, 0, 1)                   # x^3 - x - 1, the plastic number
        self.assertEqual(len(found), 1)
        root = found[0]
        self.assertFalse(root.exact)
        self.assertEqual(root.decimal, "1.324717957244746026")
        low, high = (Fraction(end) for end in root.bracket)
        cubic = [Fraction(-1), Fraction(-1), Fraction(0), Fraction(1)]

        def at(x):
            return sum(value * x ** power for power, value in enumerate(cubic))

        self.assertLess(at(low) * at(high), 0)
        self.assertIn("changes sign between", steps[-1])

    def test_rational_roots_come_out_first(self):
        found, _ = roots_of(-2, 3, 3, -5, 1)                    # (x - 1)(x^3 - 4x^2 - x + 2)
        self.assertEqual([root.exact for root in found], [False, False, True, False])
        self.assertEqual(found[2].value, "1")

    def test_digits(self):
        found, _ = real_roots([Fraction(-1), Fraction(-1), Fraction(0), Fraction(1)], "x", digits=40)
        self.assertEqual(found[0].decimal, "1.324717957244746025960908854478097340734")


class RootRefusalTests(unittest.TestCase):
    def test_limits(self):
        refused(solve(f"x^{MAX_ROOT_DEGREE + 1} = x + 1"), "too_large")
        refused(solve("x^2 + x*y = 1"), "unsupported_expression")


class SolveTests(unittest.TestCase):
    """ada.solve with powers above 1."""

    def test_answers(self):
        self.assertEqual(solve("x^2 = 4").answer, "x = -2 or x = 2")
        self.assertEqual(solve("(x - 2)^2 = 0").answer, "x = 2 (twice)")
        self.assertEqual(solve("x^3 - x - 1 = 0").answer, "x ≈ 1.324717957244746026")
        self.assertEqual(solve("x^3 = 2").answer, "x = root(2, 3) ≈ 1.2599210498948731648")
        none = solve("x^2 + 1 = 0")
        self.assertEqual((none.kind, none.reason), ("none", "no real number solves x^2 + 1 = 0."))

    def test_kinds_and_roots(self):
        several = solve("x(x - 3) = 0")
        self.assertEqual((several.kind, [root["value"] for root in several.roots]), ("several", ["0", "3"]))
        one = solve("x^2 - 6x + 9 = 0")
        self.assertEqual((one.kind, one.values, one.roots[0]["multiplicity"]), ("one", {"x": "3"}, 2))

    def test_rational_roots_are_checked_in_the_equation(self):
        result = solve("x^2 + 6 = 5x")
        self.assertIn("Check x^2 + 6 = 5x with x = 2: 10 = 10, which holds.", result.steps)

    def test_a_value_that_divides_by_zero_is_dropped(self):
        result = solve("1/x + 1/x^2 = 1")
        self.assertEqual([root["value"] for root in result.roots], ["(1 - sqrt(5))/2", "(1 + sqrt(5))/2"])
        self.assertEqual(result.conditions, ["x ≠ 0"])
        dropped = solve("(x^2 - 1)/((x - 1)(x + 3)) = 0")
        self.assertEqual(dropped.values, {"x": "-1"})
        self.assertEqual(solve("x/(x^2 - 1) = 1/(x^2 - 1)").kind, "none")


class AgreementTests(unittest.TestCase):
    """Roots checked against what is known without ADA: roots built in, the cubic's discriminant, numpy."""

    def test_polynomials_built_from_their_roots(self):
        draw = random.Random(3)
        for _ in range(60):
            chosen = sorted({Fraction(draw.randint(-12, 12), draw.choice([1, 2, 3, 5])) for _ in range(3)})
            b, c = draw.randint(-9, 9), draw.randint(-9, 9)
            polynomial = [Fraction(c), Fraction(b), Fraction(1)]          # x^2 + bx + c
            for value in chosen:
                polynomial = [Fraction(0)] + polynomial                    # multiply by (x - value)
                polynomial = [polynomial[i] - value * (polynomial[i + 1] if i + 1 < len(polynomial) else 0)
                              for i in range(len(polynomial))]
            found, _ = real_roots(polynomial, "x")
            expected = [float(value) for value in chosen]
            discriminant = b * b - 4 * c
            if discriminant >= 0:
                with localcontext() as context:
                    context.prec = 40
                    for sign in (-1, 1):
                        expected.append(float((-b + sign * Decimal(discriminant).sqrt()) / 2))
            # A root of the quadratic part may equal a chosen one, making it a double root.
            distinct = sorted(set(round(value, 9) for value in expected))
            self.assertEqual([round(float(Decimal(root.decimal)), 9) for root in found], distinct,
                             (polynomial, chosen, b, c))
            rational = list(chosen)
            if discriminant >= 0 and math.isqrt(discriminant) ** 2 == discriminant:
                rational += [Fraction(-b + sign * math.isqrt(discriminant), 2) for sign in (-1, 1)]
            for root in found:
                if "sqrt" not in root.value and "root" not in root.value:
                    self.assertTrue(root.exact)
                    self.assertIn(Fraction(root.value), rational)

    def test_cubics_against_the_discriminant_and_numpy(self):
        draw = random.Random(5)
        checked = 0
        while checked < 80:
            a, b, c = (draw.randint(-9, 9) for _ in range(3))
            discriminant = 18 * a * b * c - 4 * a ** 3 * c + a * a * b * b - 4 * b ** 3 - 27 * c * c
            if discriminant == 0:
                continue
            found, _ = real_roots([Fraction(c), Fraction(b), Fraction(a), Fraction(1)], "x")
            self.assertEqual(len(found), 3 if discriminant > 0 else 1, (a, b, c))
            reference = sorted(value.real for value in np.roots([1, a, b, c]) if abs(value.imag) < 1e-7)
            self.assertEqual(len(reference), len(found), (a, b, c))
            for root, value in zip(found, reference, strict=True):
                self.assertAlmostEqual(float(Decimal(root.decimal)), value, places=6, msg=(a, b, c))
            checked += 1


if __name__ == "__main__":
    unittest.main()
