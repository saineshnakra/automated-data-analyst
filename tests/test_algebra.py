"""Algebra with exact fractions: simplifying, solving linear equations and systems, and checking a
solution."""

import contextlib
import io
import json
import random
import re
import unittest
from fractions import Fraction

import ada
from ada import compute, tools
from ada.algebra import MAX_EQUATIONS, Polynomial, Quotient, check_solution, simplify, solve, work_out
from ada.api import REFUSAL_CODES, Refusal
from ada.cli import main


def at(text: str, point: Fraction) -> str:
    """The expression with a number in place of x, for step 1's arithmetic: 3x is 3*(2), x^2 is (2)^2."""
    return re.sub(r"(?<=[\d)])x", f"*({point})", text).replace("x", f"({point})")


def refused(result, code):
    """The result is a refusal with this code, and the code is one ADA lists."""
    assert isinstance(result, Refusal), result
    assert result.code == code, (result.code, result.message)
    assert result.code in REFUSAL_CODES
    return result.message


class PolynomialTests(unittest.TestCase):
    def test_terms_are_ordered_by_power_then_letter(self):
        x, y = Polynomial.letter("x"), Polynomial.letter("y")
        one = Polynomial.number(Fraction(1))
        self.assertEqual(str((x + y + one).power(2)), "x^2 + 2x*y + y^2 + 2x + 2y + 1")
        self.assertEqual(str(x.scaled(Fraction(1, 3)) - one.scaled(Fraction(5, 2))), "(1/3)x - 2.5")
        self.assertEqual(str(Polynomial()), "0")

    def test_a_quotient_is_kept_in_lowest_terms(self):
        x = Polynomial.letter("x")
        one = Polynomial.number(Fraction(1))
        self.assertEqual(str(Quotient(x * x - one, x - one)), "x + 1")
        top, bottom = x.scaled(Fraction(4)), x.scaled(Fraction(6)) + one.scaled(Fraction(2))
        self.assertEqual(str(Quotient(top, bottom)), "2x/(3x + 1)")
        self.assertEqual(str(Quotient(x, -x - one)), "-x/(x + 1)")

    def test_output_reads_back_as_the_same_expression(self):
        for text in ("(x + 1/3)^2 - y/2", "(2x + 3y)^3", "x*y - 0.25x^2"):
            first = simplify(text)
            self.assertEqual(simplify(first.result).result, first.result)


class SimplifyTests(unittest.TestCase):
    def test_multiplying_out_and_collecting(self):
        result = simplify("3(x + 2) - x")
        self.assertEqual(result.result, "2x + 6")
        self.assertEqual(result.steps, ["3 * (x + 2) = 3x + 6", "(3x + 6) - x = 2x + 6"])
        self.assertEqual(simplify("(x + 1)^2 - (x - 1)^2").result, "4x")
        self.assertEqual(simplify("x/2 + x/3").result, "(5/6)x")
        self.assertEqual(simplify("20% of x + x").result, "1.2x")

    def test_a_quotient_lists_what_must_not_be_zero(self):
        result = simplify("(x^2 - 1)/(x - 1)")
        self.assertEqual((result.result, result.conditions), ("x + 1", ["x ≠ 1"]))
        self.assertEqual(result.answer, "(x^2 - 1)/(x - 1) = x + 1, when x ≠ 1")
        self.assertEqual(simplify("x^-2").result, "1/x^2")

    def test_high_powers_in_a_quotient(self):
        self.assertEqual(simplify("(x^60 - 1)/(x^30 - 1)").result, "x^30 + 1")
        # Above GCD_DEGREE no common factor is looked for, and the quotient is kept as written.
        self.assertEqual(simplify("(x^150 + 3x^7 - 2)/(x^149 - x + 5)").result,
                         "(x^150 + 3x^7 - 2)/(x^149 - x + 5)")

    def test_numbers_inside_are_worked_out_exactly(self):
        self.assertEqual(simplify("sqrt(16) x + 3!").result, "4x + 6")
        self.assertEqual(simplify("x * 2^10").result, "1024x")

    def test_refusals(self):
        refused(simplify("6/2x"), "ambiguous_expression")
        refused(simplify("x + 10%"), "ambiguous_expression")
        refused(simplify("3 x 4"), "ambiguous_expression")
        self.assertIn("x*y", refused(simplify("2xy"), "invalid_expression"))
        refused(simplify("cost + 1"), "invalid_expression")
        refused(simplify("sqrt(x)"), "unsupported_expression")
        refused(simplify("2^x"), "unsupported_expression")
        refused(simplify("x^(1/2)"), "unsupported_expression")
        refused(simplify("pi*x"), "invalid_expression")
        refused(simplify("x/(x - x)"), "division_by_zero")
        refused(simplify(""), "invalid_expression")
        refused(simplify("(x + y + z + 1)^60"), "too_large")
        refused(simplify("x^500"), "too_large")


class SolveOneUnknownTests(unittest.TestCase):
    def test_steps_and_the_check(self):
        result = solve("2x + 3 = 11")
        self.assertEqual((result.kind, result.values), ("one", {"x": "4"}))
        self.assertEqual(result.steps, ["2x + 3 = 11", "Collect terms: 2x = 8",
                                        "Divide both sides by 2: x = 4",
                                        "Check 2x + 3 = 11 with x = 4: 11 = 11, which holds."])
        self.assertEqual(solve("5x + 2 = 3x + 10").values, {"x": "4"})

    def test_fractions_stay_exact(self):
        result = solve("3x = 1")
        self.assertEqual((result.values, result.decimals["x"]), ({"x": "1/3"}, "0.33333333333333333333"))
        self.assertEqual(result.answer, "x = 1/3 ≈ 0.33333333333333333333")
        self.assertEqual(solve("0.1x + 0.2 = 0.5").values, {"x": "3"})

    def test_no_solution_and_every_value(self):
        none = solve("x + 1 = x + 2")
        self.assertEqual((none.kind, none.reason), ("none", "the equations reduce to 0 = 1, which is false."))
        every = solve("2(x + 1) = 2x + 2")
        self.assertEqual((every.kind, every.free, every.answer), ("many", ["x"], "Every value of x works"))

    def test_a_letter_in_a_denominator(self):
        result = solve("120/x = 8")
        self.assertEqual((result.values, result.conditions), ({"x": "15"}, ["x ≠ 0"]))
        self.assertIn("Multiply both sides by x and collect terms: 8x = 120", result.steps)
        self.assertEqual(solve("x/(x+1) + 1 = 2/(x+1)").values, {"x": "0.5"})

    def test_a_value_that_makes_a_denominator_zero_is_not_a_solution(self):
        result = solve("(x^2 - 1)/(x - 1) = 2")
        self.assertEqual(result.kind, "none")
        self.assertIn("x = 1 makes x - 1 zero", result.reason)
        self.assertEqual(solve("x/(x - 1) = 1").kind, "none")


class SolveSystemTests(unittest.TestCase):
    def test_two_unknowns(self):
        result = solve("x + y = 10; x - y = 2")
        self.assertEqual(result.values, {"x": "6", "y": "4"})
        self.assertEqual(result.steps[2:5], ["(2) − (1): -2y = -8", "Divide (2) by (-2): y = 4",
                                             "(1) − (2): x = 6"])
        self.assertTrue(all("which holds" in step for step in result.steps[-2:]))

    def test_ways_to_separate_equations(self):
        for text in ("x + y = 10, x - y = 2", "x + y = 10 and x - y = 2", "x + y = 10\nx - y = 2"):
            self.assertEqual(solve(text).values, {"x": "6", "y": "4"}, text)
        self.assertEqual(solve(["x + y = 10", "x - y = 2"]).values, {"x": "6", "y": "4"})
        # A chain is two equations.
        self.assertEqual(solve("x + 1 = 2y = 6").values, {"x": "5", "y": "3"})

    def test_three_unknowns(self):
        result = solve("x + 2y + 3z = 14; 2x - y + z = 3; 3x + y - z = 2")
        self.assertEqual(result.values, {"x": "1", "y": "2", "z": "3"})

    def test_dependent_and_inconsistent_systems(self):
        many = solve("2x + 3y = 12")
        self.assertEqual((many.kind, many.values, many.free), ("many", {"x": "-1.5y + 6"}, ["y"]))
        self.assertIn("which holds", many.steps[-1])
        self.assertEqual(solve("x + y = 2; 2x + 2y = 4").kind, "many")
        self.assertEqual(solve("x + y = 2; x + y = 3").kind, "none")

    def test_more_equations_than_unknowns(self):
        self.assertEqual(solve("x + y = 3; x - y = 1; 2x = 4").values, {"x": "2", "y": "1"})
        self.assertEqual(solve("x + y = 3; x - y = 1; 2x = 5").kind, "none")


class SolveForALetterTests(unittest.TestCase):
    def test_a_formula_solved_for_one_letter(self):
        result = solve("d = r*t", unknowns="t")
        self.assertEqual((result.values, result.conditions), ({"t": "d/r"}, ["r ≠ 0"]))
        self.assertIn("Check d = r*t with t = d/r: d = d, which holds.", result.steps)
        self.assertEqual(solve("y = m*x + b", unknowns=["b"]).values, {"b": "-m*x + y"})
        self.assertEqual(solve("a*x + b = c", unknowns="x").values, {"x": "(-b + c)/a"})

    def test_unknowns_must_be_letters_in_the_equations(self):
        refused(solve("x + 1 = 2", unknowns="z"), "invalid_expression")
        refused(solve("x + 1 = 2", unknowns="xy"), "invalid_expression")


class SolveRefusalTests(unittest.TestCase):
    def test_not_linear_with_infinitely_many_solutions(self):
        message = refused(solve("x^2 + y = 1"), "unsupported_expression")
        self.assertIn("infinitely many solutions", message)
        # A power above 1 with another letter known is not solved for the unknown.
        refused(solve("a*x^2 = 1", unknowns="x"), "unsupported_expression")

    def test_what_is_not_an_equation(self):
        refused(solve("2x + 3"), "invalid_expression")
        refused(solve("x > 3"), "unsupported_expression")
        refused(solve("2 + 2 = 4"), "invalid_expression")
        refused(solve("= 4"), "invalid_expression")
        refused(solve(""), "invalid_expression")
        refused(solve(42), "invalid_expression")
        refused(solve("; ".join(["x = 1"] * (MAX_EQUATIONS + 1))), "too_large")

    def test_ambiguity_is_refused_inside_an_equation(self):
        refused(solve("6/2x = 3"), "ambiguous_expression")


class CheckSolutionTests(unittest.TestCase):
    def test_a_right_and_a_wrong_answer(self):
        right = check_solution("2x + 3 = 11", {"x": 4})
        self.assertEqual([(item.left, item.right, item.holds) for item in right], [("11", "11", True)])
        wrong = check_solution("x + y = 10; x - y = 2", "x = 6, y = 3")
        self.assertEqual([item.holds for item in wrong], [False, False])
        self.assertEqual(wrong[0].to_dict(), {"equation": "x + y = 10", "values": {"x": "6", "y": "3"},
                                              "left": "9", "right": "10", "holds": False})

    def test_values_as_fractions_and_expressions(self):
        self.assertTrue(check_solution("3x = 1", {"x": "1/3"})[0].holds)
        self.assertTrue(check_solution("3x = 1", {"x": Fraction(1, 3)})[0].holds)
        self.assertFalse(check_solution("3x = 1", {"x": "0.333"})[0].holds)

    def test_refusals(self):
        refused(check_solution("x + y = 3", {"x": 1}), "missing_value")
        refused(check_solution("2x = 2", {"x": "sqrt(2)"}), "invalid_expression")
        refused(check_solution("2x = 2", {}), "invalid_expression")
        refused(check_solution("2x = 2", {"xx": 1}), "invalid_expression")


class AgreementTests(unittest.TestCase):
    """Results checked against something independent of the algebra: a solution built in, and step 1's
    arithmetic at chosen points."""

    def test_systems_built_around_a_solution_give_it_back(self):
        draw = random.Random(7)
        letters = "wxyz"
        for _ in range(120):
            size = draw.randint(1, 4)
            names = letters[:size]
            answer = [Fraction(draw.randint(-20, 20), draw.choice([1, 1, 2, 3])) for _ in names]
            # A = L*U, with ones on L's diagonal and no zero on U's, can always be solved.
            lower = [[1 if i == j else (draw.randint(-3, 3) if j < i else 0) for j in range(size)]
                     for i in range(size)]
            upper = [[draw.choice([-3, -2, -1, 1, 2, 3]) if i == j else (draw.randint(-4, 4) if j > i else 0)
                      for j in range(size)] for i in range(size)]
            matrix = [[sum(lower[i][k] * upper[k][j] for k in range(size)) for j in range(size)]
                      for i in range(size)]
            equations = []
            for row in matrix:
                right = sum(coefficient * value for coefficient, value in zip(row, answer, strict=True))
                left = " + ".join(f"({coefficient})*{name}"
                                  for coefficient, name in zip(row, names, strict=True))
                equations.append(f"{left} = {right}")
            result = solve("; ".join(equations))
            self.assertEqual(result.kind, "one", equations)
            self.assertEqual({name: Fraction(result.values[name]) for name in names},
                             dict(zip(names, answer, strict=True)), equations)
            self.assertTrue(all("which holds" in step for step in result.steps if step.startswith("Check")))

    def test_simplified_expressions_agree_with_arithmetic_at_points(self):
        draw = random.Random(11)
        for _ in range(80):
            a, b, c, d, k = (draw.randint(-6, 6) for _ in range(5))
            text = f"({a}x + {b})*({c}x - {d})^{abs(k) % 4} - ({b}x - {a})/{draw.randint(1, 5)} + {c}x^2"
            result = simplify(text)
            for point in (Fraction(-3), Fraction(1, 2), Fraction(5, 3)):
                written, simplified = compute(at(text, point)), compute(at(result.result, point))
                self.assertEqual(written.fraction or written.value, simplified.fraction or simplified.value,
                                 (text, result.result, point))


class WorkOutTests(unittest.TestCase):
    """The tool and the command line send each text to compute, simplify or solve."""

    def test_routing(self):
        self.assertEqual(work_out("2 + 2").value, "4")
        self.assertEqual(work_out("2 + 2 =").value, "4")
        self.assertEqual(work_out("3(x + 2) - x").result, "2x + 6")
        self.assertEqual(work_out("2x + 3 = 11").values, {"x": "4"})
        self.assertEqual(work_out("solve 2x + 3 = 11").values, {"x": "4"})
        self.assertEqual(work_out("solve d = r*t for t").values, {"t": "d/r"})
        self.assertEqual(work_out("solve x + y = 10, x - y = 2 for x and y").values, {"x": "6", "y": "4"})
        self.assertEqual(work_out("simplify (x + 1)^2").result, "x^2 + 2x + 1")
        refused(work_out("cost + 1"), "invalid_expression")

    def test_the_tool(self):
        result = tools.run("ada_compute", {"expression": "x + y = 10; x - y = 2"}, brief=True)
        self.assertEqual((result["status"], result["values"], result["kind"]),
                         ("solved", {"x": "6", "y": "4"}, "one"))
        self.assertEqual(result["answer"], "x = 6, y = 4")
        system = tools.run("ada_compute", {"expression": "x*y = 6; x + y = 5"}, brief=True)
        self.assertEqual((system["kind"], system["answer"]), ("several", "(x = 2, y = 3) or (x = 3, y = 2)"))
        curve = tools.run("ada_compute", {"expression": "x^2 + y^2 = 1"})
        self.assertEqual(curve["code"], "unsupported_expression")
        simplified = tools.run("ada_compute", {"expression": "(x + 1)^2"}, brief=True)
        self.assertEqual((simplified["status"], simplified["result"]), ("simplified", "x^2 + 2x + 1"))

    def test_the_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["compute", "x + y = 10; x - y = 2", "--json"])
        self.assertEqual((code, json.loads(out.getvalue())["values"]), (0, {"x": "6", "y": "4"}))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = main(["compute", "x^2 + y^2 = 1"])
        self.assertEqual(code, 3)
        self.assertIn("unsupported_expression", err.getvalue())

    def test_the_package_names_them(self):
        self.assertEqual(ada.solve("2x = 6").values, {"x": "3"})
        self.assertIs(ada.simplify, simplify)
        self.assertIs(ada.check_solution, check_solution)

    def test_compute_alone_still_refuses_letters(self):
        refused(ada.compute("2x + 3"), "invalid_expression")


if __name__ == "__main__":
    unittest.main()
