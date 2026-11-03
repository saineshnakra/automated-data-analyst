"""Systems of polynomial equations: ada.solve with several equations or unknowns that are not linear."""

import itertools
import math
import random
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

from ada import compute, solve
from ada.api import REFUSAL_CODES, Refusal
from ada.systems import MAX_SOLUTIONS, MAX_UNKNOWNS


def refused(result, code):
    assert isinstance(result, Refusal), result
    assert result.code == code and code in REFUSAL_CODES, (result.code, result.message)
    return result.message


def values_of(result):
    return [solution["values"] for solution in result.solutions]


def residual(equation, values):
    """|left - right| with the values' decimals put in, from step 1's arithmetic, to 40 digits."""
    left, right = equation.split("=")
    for name, value in values.items():
        left, right = (side.replace(name, f"({value})") for side in (left, right))
    result = compute(f"abs(({left}) - ({right}))", digits=40)
    return Decimal(result.decimal)


class AnswerTests(unittest.TestCase):
    def test_rational_solutions(self):
        self.assertEqual(solve("x*y = 6; x + y = 5").answer, "(x = 2, y = 3) or (x = 3, y = 2)")
        self.assertEqual(solve("x^2 + y^2 = 25; x + y = 7").answer, "(x = 3, y = 4) or (x = 4, y = 3)")
        self.assertEqual(solve("x^2 + y = 1; x - y = 1").answer, "(x = -2, y = -3) or (x = 1, y = 0)")
        self.assertEqual(values_of(solve("x^2 = 4; y^2 = 9")),
                         [{"x": "-2", "y": "-3"}, {"x": "-2", "y": "3"}, {"x": "2", "y": "-3"},
                          {"x": "2", "y": "3"}])

    def test_one_solution_fills_values(self):
        result = solve("x + y = 2; x*y = 1")
        self.assertEqual((result.kind, result.values, result.answer),
                         ("one", {"x": "1", "y": "1"}, "x = 1, y = 1"))
        self.assertEqual(solve("x^3 + y = 2; y^3 + x = 2").values, {"x": "1", "y": "1"})

    def test_square_roots(self):
        result = solve("x^2 + y^2 = 1; x + y = 1/2")
        self.assertEqual(values_of(result), [{"x": "(1 - sqrt(7))/4", "y": "(1 + sqrt(7))/4"},
                                             {"x": "(1 + sqrt(7))/4", "y": "(1 - sqrt(7))/4"}])
        self.assertTrue(all(all(solution["exact"].values()) for solution in result.solutions))
        self.assertEqual(result.solutions[0]["decimals"]["x"], "-0.41143782776614764763")

    def test_a_root_of_a_number(self):
        result = solve("x^3 = 2; y = x^2 + 1")
        self.assertEqual(result.values, {"x": "root(2, 3)", "y": "root(2, 3)^2 + 1"})
        self.assertEqual(result.decimals, {"x": "1.2599210498948731648", "y": "2.5874010519681994748"})
        self.assertIn("Eliminate y with a Gröbner basis (SymPy, lex order): x^3 - 2 = 0, with y = x^2 + 1.",
                      result.steps)

    def test_values_with_no_exact_form(self):
        result = solve("x^2 + y^2 = 1; y = x^3")
        self.assertEqual(result.answer, "(x ≈ -0.82603135765418695597, y ≈ -0.56362416216125854857) or "
                                        "(x ≈ 0.82603135765418695597, y ≈ 0.56362416216125854857)")
        # y is a root of y^6 - 3y^4 + 4y^2 - 1 and x its cube root; mpmath gives both to 50 digits.
        known = {"y": Fraction("0.56362416216125854856849797436694447727643026421"),
                 "x": Fraction("0.82603135765418695596898700199772132767189630265")}
        for solution, sign in zip(result.solutions, (-1, 1), strict=True):
            self.assertEqual(solution["exact"], {"x": False, "y": False})
            for name in ("x", "y"):
                low, high = (Fraction(end) for end in solution["brackets"][name])
                self.assertLess(low, sign * known[name])
                self.assertLess(sign * known[name], high)
                self.assertLess(high - low, Fraction(1, 10 ** 18))

    def test_three_unknowns(self):
        result = solve("x + y + z = 6; x*y + y*z + z*x = 11; x*y*z = 6")
        self.assertEqual([tuple(values.values()) for values in values_of(result)],
                         sorted(itertools.permutations(("1", "2", "3"))))
        self.assertTrue(any(step.startswith("No single unknown tells the solutions apart, so let u = ")
                            for step in result.steps))

    def test_one_unknown_in_several_equations(self):
        result = solve("x^2 = 4; x^3 = 8")
        self.assertEqual((result.kind, result.values, result.answer), ("one", {"x": "2"}, "x = 2"))
        self.assertEqual([root["value"] for root in solve("x^2 = 2; x^4 = 4").roots], ["-sqrt(2)", "sqrt(2)"])

    def test_digits(self):
        result = solve("x^2 + y^2 = 1; y = x^3", digits=30)
        self.assertEqual(result.solutions[1]["values"]["y"], "0.563624162161258548568497974367")


class RepeatedSolutionTests(unittest.TestCase):
    def test_a_solution_counted_more_than_once_is_counted_once(self):
        result = solve("x^2 = 0; y^2 = 0")
        self.assertEqual((result.kind, result.values), ("one", {"x": "0", "y": "0"}))
        self.assertTrue(any(step.startswith("Some solutions are counted more than once. Adding y = 0, x = 0")
                            for step in result.steps))
        self.assertEqual(solve("(x - 1)^2 = 0; (y - 2)^2 = 0").values, {"x": "1", "y": "2"})
        self.assertEqual(solve("x^2 + y^2 = 0; x^2 = y^2").values, {"x": "0", "y": "0"})

    def test_a_double_point_where_one_unknown_tells_them_apart(self):
        self.assertEqual(solve("x + y = 2; x*y = 1").values, {"x": "1", "y": "1"})


class NoSolutionTests(unittest.TestCase):
    def test_only_complex_solutions(self):
        result = solve("x^2 + y^2 = -1; x + y = 0")
        self.assertEqual(result.kind, "none")
        self.assertEqual(result.reason,
                         "2y^2 + 1 = 0 has no real root, so no real values satisfy every equation.")

    def test_equations_that_contradict_each_other(self):
        result = solve("x*y = 1; x*y = 2")
        self.assertEqual(result.kind, "none")
        self.assertIn("leaves 1 = 0", result.reason)


class StepTests(unittest.TestCase):
    def test_the_steps_of_a_system(self):
        steps = solve("x*y = 6; x + y = 5").steps
        self.assertEqual(steps[:4], ["(1) x*y = 6", "(2) x + y = 5",
                                     "Move everything to one side: (1) x*y - 6 = 0, (2) x + y - 5 = 0.",
                                     "Eliminate x with a Gröbner basis (SymPy, lex order): y^2 - 5y + 6 = 0, "
                                     "with x = -y + 5."])
        self.assertIn("Factor over the rationals: y^2 - 5y + 6 = (y - 2) * (y - 3).", steps)
        self.assertIn("y = 2 gives x = 3.", steps)
        self.assertIn("Check x*y = 6 with x = 3, y = 2: 6 = 6, which holds.", steps)

    def test_a_factor_with_roots_that_are_not_fractions_is_checked_by_division(self):
        steps = solve("x^2 + y^2 = 1; x + y = 1/2").steps
        self.assertIn("Check: with x = -y + 0.5, each equation leaves no remainder when divided by "
                      "8y^2 - 4y - 3, so each holds at every root of 8y^2 - 4y - 3 = 0.", steps)


class DenominatorTests(unittest.TestCase):
    def test_conditions(self):
        result = solve("1/x + 1/y = 1; x + y = 4")
        self.assertEqual((result.answer, result.conditions), ("x = 2, y = 2 (when x ≠ 0, y ≠ 0)",
                                                               ["x ≠ 0", "y ≠ 0"]))

    def test_a_solution_that_divides_by_zero_is_dropped(self):
        result = solve("y/x = 1; x^2 = x")
        self.assertEqual(result.values, {"x": "1", "y": "1"})
        self.assertTrue(any("make a denominator zero" in step for step in result.steps))
        none = solve("y/x = 1; x^2 = 0")
        self.assertEqual(none.kind, "none")
        self.assertEqual(none.reason,
                         "every real solution makes a denominator zero, and the equations divide by it.")
        complex_only = solve("1/x = y; x^2 = -1")
        self.assertEqual(complex_only.reason, "y^2 + 1 = 0 has no real root, so no real values satisfy every "
                                              "equation.")


class RefusalTests(unittest.TestCase):
    def test_infinitely_many_solutions(self):
        message = refused(solve("x^2 + y^2 = 1"), "unsupported_expression")
        self.assertIn("infinitely many solutions", message)
        refused(solve("x*y = 0; x*y*z = 0"), "unsupported_expression")

    def test_other_letters(self):
        message = refused(solve("x*y = a; x + y = 1", unknowns="x y"), "unsupported_expression")
        self.assertIn("give a a value", message)

    def test_limits(self):
        names = "abcdfghjk"[:MAX_UNKNOWNS + 1]
        refused(solve("; ".join(f"{name}^2 = 1" for name in names)), "too_large")
        refused(solve("x^7 = 2; y^6 = 3"), "too_large")
        self.assertGreater(7 * 6, MAX_SOLUTIONS)


class AgreementTests(unittest.TestCase):
    """Systems whose solutions are known without ADA: from the roots of t^2 - st + p, a line and a circle, and
    three numbers from the sums of their products."""

    def assert_close(self, found, expected, message):
        self.assertEqual(len(found), len(expected), message)
        for values, wanted in zip(found, expected, strict=True):
            for name, value in wanted.items():
                self.assertAlmostEqual(float(values[name]), value, places=9, msg=message)

    def test_sum_and_product(self):
        draw = random.Random(7)
        for _ in range(40):
            s, p = draw.randint(-12, 12), draw.randint(-30, 30)
            result = solve(f"x + y = {s}; x*y = {p}")
            discriminant = s * s - 4 * p
            if discriminant < 0:
                self.assertEqual(result.kind, "none", (s, p))
                continue
            with localcontext() as context:
                context.prec = 40
                root = Decimal(discriminant).sqrt()
            low, high = float((s - root) / 2), float((s + root) / 2)
            expected = ([{"x": low, "y": high}, {"x": high, "y": low}] if discriminant
                        else [{"x": low, "y": low}])
            decimals = [solution["decimals"] for solution in result.solutions] or [result.values]
            self.assert_close(decimals, expected, (s, p))
            for solution in result.solutions:
                for equation in (f"x + y = {s}", f"x*y = {p}"):
                    self.assertLess(residual(equation, solution["values"]), Decimal("1e-30"),
                                    (s, p, solution))

    def test_a_line_and_a_circle(self):
        draw = random.Random(11)
        for _ in range(40):
            a, b = draw.choice([-3, -2, -1, 1, 2, 3]), draw.choice([-3, -2, -1, 1, 2, 3])
            c, r2 = draw.randint(-8, 8), draw.randint(1, 20)
            result = solve(f"x^2 + y^2 = {r2}; {a}x + {b}y = {c}".replace("+ -", "- "))
            # The line is c/sqrt(a^2 + b^2) from the centre: two points, one or none.
            gap = Fraction(c * c, a * a + b * b) - r2
            count = 2 if gap < 0 else 1 if gap == 0 else 0
            found = result.solutions or ([{"decimals": result.values}] if result.kind == "one" else [])
            self.assertEqual(len(found), count, (a, b, c, r2))
            for solution in result.solutions:
                x, y = (float(solution["decimals"][name]) for name in ("x", "y"))
                self.assertAlmostEqual(x * x + y * y, r2, places=9)
                self.assertAlmostEqual(a * x + b * y, c, places=9)

    def test_three_numbers_from_their_symmetric_sums(self):
        draw = random.Random(13)
        for _ in range(12):
            chosen = [draw.randint(-4, 4) for _ in range(3)]
            e1 = sum(chosen)
            e2 = chosen[0] * chosen[1] + chosen[1] * chosen[2] + chosen[0] * chosen[2]
            e3 = math.prod(chosen)
            result = solve(f"x + y + z = {e1}; x*y + y*z + z*x = {e2}; x*y*z = {e3}")
            expected = sorted(set(itertools.permutations(chosen)))
            found = [tuple(int(values[name]) for name in "xyz") for values in values_of(result)] \
                or [tuple(int(result.values[name]) for name in "xyz")]
            self.assertEqual(found, expected, chosen)


if __name__ == "__main__":
    unittest.main()
