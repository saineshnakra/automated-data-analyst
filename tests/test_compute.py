"""Exact arithmetic: values, steps, conventions stated as assumptions, and refusals with their reasons."""

import contextlib
import io
import json
import unittest
from decimal import Decimal, localcontext

from ada import tools
from ada.api import REFUSAL_CODES, Refusal
from ada.cli import main
from ada.compute import MAX_STEPS, check_arithmetic, compute, repeating


def value(expression: str, **options) -> str:
    result = compute(expression, **options)
    if isinstance(result, Refusal):
        raise AssertionError(f"{expression} was refused: {result.message}")
    return result.value


def refused(test: unittest.TestCase, expression: str, code: str) -> Refusal:
    result = compute(expression)
    test.assertIsInstance(result, Refusal, f"{expression} gave {getattr(result, 'answer', result)}")
    test.assertEqual(result.code, code, result.message)
    test.assertIn(result.code, REFUSAL_CODES)
    return result


class ExactTests(unittest.TestCase):
    def test_decimals_are_read_as_fractions(self):
        result = compute("0.1 + 0.2")
        self.assertEqual((result.value, result.fraction, result.exact), ("0.3", "3/10", True))

    def test_division_gives_a_fraction_in_lowest_terms(self):
        self.assertEqual(value("1/3 + 1/6"), "0.5")
        self.assertEqual(compute("2/6").fraction, "1/3")
        self.assertEqual(value("(1/3)*3"), "1")
        self.assertEqual(value("1/(2/3)"), "1.5")

    def test_large_integers_stay_exact(self):
        self.assertEqual(value("123456789 * 987654321"), "121932631112635269")
        self.assertEqual(value("2^64"), "18446744073709551616")
        self.assertEqual(value("100!"), str(__import__("math").factorial(100)))

    def test_a_decimal_that_ends_is_the_value_and_one_that_does_not_is_a_fraction(self):
        self.assertEqual(compute("2^10 + 3/4").answer, "2^10 + 3/4 = 1024.75 = 4099/4")
        third = compute("1/3")
        self.assertEqual((third.value, third.decimal_exact, third.repeating), ("1/3", False, "0.(3)"))
        self.assertEqual(third.answer, "1/3 = 0.(3)")

    def test_repeating_decimals(self):
        from fractions import Fraction

        self.assertEqual(repeating(Fraction(1, 7)), "0.(142857)")
        self.assertEqual(repeating(Fraction(1, 6)), "0.1(6)")
        self.assertEqual(repeating(Fraction(-22, 7)), "-3.(142857)")
        block = repeating(Fraction(1, 61))                               # a block of 60 digits is written out
        self.assertEqual(len(block[block.index("(") + 1:-1]), 60)
        self.assertIsNone(repeating(Fraction(1, 97)))                    # 96 digits: too long to write
        self.assertIsNone(repeating(Fraction(3, 4)))                     # the decimal ends

    def test_a_single_number_is_its_own_value_with_no_steps(self):
        result = compute("5")
        self.assertEqual((result.value, result.steps, result.answer), ("5", [], "5"))


class ApproximateTests(unittest.TestCase):
    def test_irrational_results_are_marked_and_rounded_to_the_digits_asked(self):
        result = compute("sqrt(2)")
        self.assertFalse(result.exact)
        with localcontext() as context:
            context.prec = 20
            self.assertEqual(result.value, str(+Decimal(2).sqrt()))
        self.assertEqual(compute("sqrt(2)", digits=5).value, "1.4142")
        self.assertIn("Approximate", str(result))

    def test_pi_and_e(self):
        self.assertEqual(compute("pi", digits=30).value, "3.14159265358979323846264338328")
        self.assertEqual(compute("e", digits=20).value, "2.7182818284590452354")
        self.assertEqual(compute("2*pi", digits=10).value, "6.283185307")

    def test_roots_are_exact_where_they_can_be(self):
        self.assertEqual(value("sqrt(16/9)"), "4/3")
        self.assertEqual(value("8^(2/3)"), "4")
        self.assertEqual(value("cbrt(27)"), "3")
        self.assertEqual(value("root(-8, 3)"), "-2")
        self.assertEqual(value("sqrt(10^40)"), "1" + "0" * 20)
        self.assertFalse(compute("2^0.5").exact)

    def test_logarithms_are_exact_where_they_can_be(self):
        self.assertEqual(value("log(8, 2)"), "3")
        self.assertEqual(value("log10(1000)"), "3")
        self.assertEqual(value("log2(1/8)"), "-3")
        self.assertEqual(value("ln(1)"), "0")
        self.assertEqual(value("exp(0)"), "1")
        self.assertFalse(compute("ln(2)").exact)


class ConventionTests(unittest.TestCase):
    def test_a_minus_sign_before_a_power_applies_after_it(self):
        result = compute("-2^2")
        self.assertEqual(result.value, "-4")
        self.assertIn("-(2^2)", result.assumptions[0])
        self.assertEqual(value("(-2)^2"), "4")

    def test_powers_read_from_the_right(self):
        result = compute("2^3^2")
        self.assertEqual(result.value, "512")
        self.assertIn("2^(3^2)", result.assumptions[0])

    def test_percent_and_percent_of(self):
        self.assertEqual(value("50%"), "0.5")
        self.assertEqual(value("20% of 50"), "10")
        self.assertEqual(value("20% of 50 + 5"), "15")
        self.assertEqual(value("1200 * 15%"), "180")

    def test_a_missing_multiplication_sign_is_read_when_every_reading_agrees(self):
        result = compute("2(3+4)")
        self.assertEqual(result.value, "14")
        self.assertTrue(any("multiplication sign" in item for item in result.assumptions))

    def test_thousands_separators_and_currency(self):
        result = compute("$1,200 * 3")
        self.assertEqual(result.value, "3600")
        self.assertEqual(len(result.assumptions), 2)
        self.assertEqual(value("1,234,567.5 * 2"), "2469135")
        self.assertEqual(value("gcd(120,180)"), "60")     # inside a call the comma separates arguments

    def test_zero_to_the_zero_is_one_and_says_so(self):
        result = compute("0^0")
        self.assertEqual(result.value, "1")
        self.assertIn("0^0", result.assumptions[0])

    def test_written_symbols(self):
        self.assertEqual(value("3 × 4 ÷ 6"), "2")
        self.assertEqual(value("2² + 2³"), "12")
        self.assertEqual(value("√16"), "4")
        self.assertEqual(value("10 − 3"), "7")
        self.assertEqual(value("2 + 2 ="), "4")

    def test_a_function_written_without_parentheses_takes_the_number_after_it(self):
        result = compute("sqrt 16 + 9")
        self.assertEqual(result.value, "13")
        self.assertTrue(result.assumptions)


class FunctionTests(unittest.TestCase):
    def test_whole_number_functions(self):
        self.assertEqual(value("gcd(12, 18, 30)"), "6")
        self.assertEqual(value("lcm(4, 6, 10)"), "60")
        self.assertEqual(value("comb(52, 5)"), "2598960")
        self.assertEqual(value("nCr(10, 3)"), "120")
        self.assertEqual(value("perm(5, 2)"), "20")
        self.assertEqual(value("comb(3, 5)"), "0")
        self.assertEqual(value("5! / 3!"), "20")
        self.assertEqual(value("0!"), "1")

    def test_remainders_and_whole_division(self):
        self.assertEqual(value("17 mod 5"), "2")
        self.assertEqual(value("-7 mod 3"), "2")
        self.assertEqual(value("mod(2^100, 7)"), "2")
        self.assertEqual(value("17 // 5"), "3")
        self.assertEqual(value("-17 // 5"), "-4")

    def test_rounding_halves_away_from_zero_on_the_exact_value(self):
        self.assertEqual(value("round(2.675, 2)"), "2.68")      # binary floating point gives 2.67
        self.assertEqual(value("round(2.5)"), "3")
        self.assertEqual(value("round(-2.5)"), "-3")
        self.assertEqual(value("round(1/3, 3)"), "0.333")
        self.assertEqual(value("round(1234.5, -2)"), "1200")

    def test_floor_ceil_abs_min_max(self):
        self.assertEqual(value("floor(-2.5)"), "-3")
        self.assertEqual(value("ceil(2.1)"), "3")
        self.assertEqual(value("floor(sqrt(2) * 100)"), "141")
        self.assertEqual(value("abs(-3/4)"), "0.75")
        self.assertEqual(value("min(3, 1/2, 2)"), "0.5")
        self.assertEqual(value("max(3, 7, 2)"), "7")


class RefusalTests(unittest.TestCase):
    def test_division_by_zero(self):
        refused(self, "1/0", "division_by_zero")
        refused(self, "5 mod 0", "division_by_zero")
        refused(self, "0^-1", "division_by_zero")

    def test_values_that_are_not_real_numbers(self):
        refused(self, "sqrt(-4)", "not_a_real_number")
        refused(self, "(-8)^(1/3)", "not_a_real_number")
        refused(self, "ln(0)", "not_a_real_number")

    def test_an_expression_that_reads_two_ways_gives_both_values(self):
        message = refused(self, "6/2(1+2)", "ambiguous_expression").message
        self.assertIn("= 9", message)
        self.assertIn("= 1", message)
        self.assertIn("6/(2*(1 + 2))", message)

    def test_a_percent_sign_between_two_numbers_is_ambiguous(self):
        message = refused(self, "17 % 5", "ambiguous_expression").message
        self.assertIn("17 mod 5", message)

    def test_adding_a_percent_is_ambiguous(self):
        message = refused(self, "100 + 10%", "ambiguous_expression").message
        self.assertIn("= 100.1", message)
        self.assertIn("= 110", message)
        self.assertIn("150", refused(self, "200 - 25%", "ambiguous_expression").message)
        self.assertEqual(value("100 * 10%"), "10")           # a percent of a number reads one way
        self.assertEqual(value("10% + 5"), "5.1")

    def test_rounding_an_approximate_value_too_close_to_tell(self):
        # e^(pi*sqrt(163)) is 262537412640768743.99999999999925...: 20 digits cannot say which side of the
        # whole number it falls, 40 can.
        refused(self, "floor(exp(pi*sqrt(163)))", "needs_more_digits")
        self.assertEqual(compute("floor(exp(pi*sqrt(163)))", digits=40).value, "262537412640768743")
        refused(self, "round(sqrt(2), 40)", "needs_more_digits")
        self.assertEqual(value("round(sqrt(2), 3)"), "1.414")
        self.assertEqual(value("floor(sqrt(2) * 100)"), "141")

    def test_log_without_a_base_is_ambiguous(self):
        message = refused(self, "log(8)", "ambiguous_expression").message
        self.assertIn("log10", message)

    def test_names_and_text_are_refused_with_what_to_do(self):
        message = refused(self, "x + 1", "invalid_expression").message
        self.assertIn("numbers only", message)
        refused(self, "what is 2 + 2", "invalid_expression")
        refused(self, "2 3", "invalid_expression")
        refused(self, "2 +", "invalid_expression")
        refused(self, "(2 + 3", "invalid_expression")
        refused(self, "2 # 3", "invalid_expression")
        refused(self, "", "invalid_expression")
        refused(self, "1,2", "invalid_expression")

    def test_wrong_arguments(self):
        refused(self, "sqrt(1, 2)", "invalid_expression")
        refused(self, "factorial(2.5)", "invalid_expression")
        refused(self, "(-3)!", "invalid_expression")
        refused(self, "gcd(1.5, 3)", "invalid_expression")
        refused(self, "comb(-1, 2)", "invalid_expression")
        refused(self, "log(8, 1)", "invalid_expression")

    def test_numbers_too_large_to_keep_exactly(self):
        self.assertIn("digits", refused(self, "2^100000", "too_large").message)
        refused(self, "10000!", "too_large")
        refused(self, "2^2^2^2^2", "too_large")
        refused(self, "comb(100000, 50000)", "too_large")
        refused(self, "1e10000000", "too_large")
        refused(self, "1" * 3000, "too_large")
        refused(self, "-" * 150 + "1", "too_large")

    def test_digits_out_of_range(self):
        self.assertEqual(compute("1/3", digits=0).code, "invalid_expression")
        self.assertEqual(compute("1/3", digits=True).code, "invalid_expression")


class StepTests(unittest.TestCase):
    def test_each_operation_is_a_step_with_its_values(self):
        result = compute("(2 + 3) * (4 - 1) / 5")
        self.assertEqual(result.steps, ["2 + 3 = 5", "4 - 1 = 3", "5 * 3 = 15", "15 / 5 = 3"])

    def test_approximate_steps_say_so(self):
        self.assertEqual(compute("sqrt(2)", digits=5).steps, ["sqrt(2) ≈ 1.4142"])

    def test_a_long_chain_is_computed_and_its_steps_are_cut_short(self):
        result = compute("+".join(["1"] * 600))
        self.assertEqual(result.value, "600")
        self.assertEqual(len(result.steps), MAX_STEPS + 1)
        self.assertIn("more steps", result.steps[-1])

    def test_the_result_as_data(self):
        data = compute("1/7").to_dict()
        self.assertEqual(data["status"], "computed")
        self.assertEqual((data["value"], data["repeating"], data["exact"]), ("1/7", "0.(142857)", True))
        json.dumps(data)


class CheckArithmeticTests(unittest.TestCase):
    """The equations in a worked answer, checked one by one."""

    def test_a_slip_is_found_with_the_right_value(self):
        found = check_arithmetic("Points: 204 + 160 + 330 = 700. Then 726 × 1.10 = 800.")
        self.assertEqual([(item.expression, item.computed, item.correct) for item in found],
                         [("204 + 160 + 330", "694", False), ("726 × 1.10", "798.6", False)])

    def test_right_equations_pass_including_rounded_ones(self):
        found = check_arithmetic("2 / 2 = 1 and 2 + 1 = 3; water (2/3) * 10 = 6.67 litres; 22/7 = 3.14; "
                                 "1,200 * 3 = 3,600; average (60 + 57 + 54 + 51) / 4 = 55.5.")
        self.assertEqual(len(found), 6)
        self.assertTrue(all(item.correct for item in found))
        self.assertFalse(check_arithmetic("22/7 = 3.15")[0].correct)

    def test_written_operators(self):
        found = check_arithmetic("3 x 4 = 12, $15 \\times 4 = $60, 2^10 = 1000.")
        self.assertEqual([item.correct for item in found], [True, True, False])

    def test_what_is_not_an_operation_on_numbers_is_skipped(self):
        self.assertEqual(check_arithmetic("x = 5, so y = 2x = 10. No equations here."), [])
        self.assertEqual(check_arithmetic(""), [])
        # 6/2(1+2) reads two ways, so it is not judged.
        self.assertEqual(check_arithmetic("6/2(1+2) = 9"), [])

    def test_the_result_as_data(self):
        item = check_arithmetic("7 * 8 = 54")[0]
        self.assertEqual(item.to_dict(), {"expression": "7 * 8", "stated": "54", "computed": "56",
                                          "correct": False})

    # The cases below are from small models' worked answers to GSM8K problems.

    def test_a_chain_checks_its_last_step(self):
        found = check_arithmetic("5,000 * (1 + 0.025) = 5,000 * 1.025 = 5,125")
        self.assertEqual([(item.expression, item.stated, item.correct) for item in found],
                         [("5,000 * 1.025", "5,125", True)])
        # A running total is written as a chain too.
        self.assertTrue(check_arithmetic("3 * 4 = 12 + 1 = 13")[0].correct)
        found = check_arithmetic("\\text{Salary} = 40,000 \\times (1 + 0.30) = 40,000 \\times 1.30 = 52,000")
        self.assertEqual([(item.expression, item.correct) for item in found], [("40,000 * 1.30", True)])

    def test_part_of_a_longer_expression_is_skipped(self):
        self.assertEqual(check_arithmetic("16 friends * (1 - 1/4) = 16 friends * 3/4 = 12 guests"), [])
        self.assertEqual(check_arithmetic("30 bees * 1/2 = 15 bees."), [])
        self.assertEqual(check_arithmetic("We know that S = 2B and S + B + 2 + 2 = 28."), [])
        self.assertEqual(check_arithmetic("2x + 15 - 5 = 2x + 10"), [])
        self.assertEqual(check_arithmetic("15 - 5 = 2x"), [])
        self.assertEqual(check_arithmetic("$50,000/year * 20 years = $1,000,000"), [])
        self.assertEqual(check_arithmetic("7/13 = 120/x"), [])

    def test_lists_and_lines(self):
        # A list marker is not a minus sign, and an equation does not run across lines.
        found = check_arithmetic("Half of 80 is 40\n   - 40 + 20 = 60\n"
                                 "   - 25% of 80 is 20\n   - 80 + 20 = 100")
        self.assertEqual([(item.expression, item.correct) for item in found],
                         [("40 + 20", True), ("80 + 20", True)])
        self.assertEqual(check_arithmetic("5\n- 3 = 12"), [])
        found = check_arithmetic("So the total is **3 * 4 = 12** apples (that is 5 * 6 = 30) "
                                 "and 2 * 3 = 6 (cups)")
        self.assertEqual([item.correct for item in found], [True, True, True])

    def test_rates_currency_and_misplaced_commas(self):
        found = check_arithmetic("- Total monthly payment: $450 + $300 + $750 = $1,500/month")
        self.assertEqual([(item.stated, item.correct) for item in found], [("1,500", True)])
        found = check_arithmetic("Increase = 10\\% \\times 726 = 0.10 \\times 726 = 72.60")
        self.assertEqual([(item.expression, item.correct) for item in found], [("0.10 * 726", True)])
        self.assertEqual(check_arithmetic("she sells 4933828 / 13 = 373, 846.92 eggs"), [])
        self.assertEqual(check_arithmetic("581,3040 + 1,764,912 + 12 = 583,0694"), [])

    def test_percent_and_cut_off_values(self):
        found = check_arithmetic("1/4 = 25% and 0.25 * 100 = 25%, but 1/4 = 30%.")
        self.assertEqual([item.correct for item in found], [True, True, False])
        # A value cut off to the places written is accepted, as a rounded one is.
        self.assertTrue(check_arithmetic("160 / 6 = 26.666")[0].correct)
        self.assertTrue(check_arithmetic("726 + 72.6 = 798")[0].correct)
        self.assertFalse(check_arithmetic("160 / 6 = 26.665")[0].correct)

    def test_conditions_and_scale_words_are_skipped(self):
        text = "This is only possible if 240 + 3,516,571 = 0, which is not true."
        self.assertEqual(check_arithmetic(text), [])
        self.assertEqual(check_arithmetic("2 * 5,000,000 = 10 million"), [])
        # A short unit after the value is not a scale word.
        self.assertFalse(check_arithmetic("7930843 / 2 = 3975421.5 M&Ms.")[0].correct)
        self.assertTrue(check_arithmetic("So 3 * 4 = 12, and if that holds")[0].correct)

    def test_suppositions_and_contradictions_drawn_on_purpose_are_skipped(self):
        # From MATH-500 replies: the writer knows these are false.
        self.assertEqual(check_arithmetic("0 + 0 = 2 \\Rightarrow \\text{No solution}"), [])
        self.assertEqual(check_arithmetic("5 + 3 = 9, which is a contradiction."), [])
        self.assertEqual(check_arithmetic("so 3 + 4 = 8, which is impossible"), [])
        self.assertEqual(check_arithmetic("Suppose 2 + 2 = 5. Then"), [])
        self.assertEqual(check_arithmetic("Assume 3 * 4 = 13 and see what follows."), [])
        # A slip with no such word after it is still found.
        self.assertFalse(check_arithmetic("2 + 2 = 5, so the answer is 5")[0].correct)
        self.assertFalse(check_arithmetic("(4)^2 + (1)^2 = 25,")[0].correct)

    def test_signs(self):
        self.assertTrue(check_arithmetic("Answer: -3 + 5 = 2")[0].correct)
        self.assertTrue(check_arithmetic("3 - 5 = -2")[0].correct)
        self.assertTrue(check_arithmetic("2**3 = 8")[0].correct)
        self.assertFalse(check_arithmetic("13 - 4933828 = 0 eggs")[0].correct)

    def test_slips_from_model_answers_are_found(self):
        text = ("Total time for Team 2 = $60 + 57 + 54 + 51 = 232$ seconds.\n"
                "New salary = 726 + 72.60 = 800.60\n"
                "Net earnings per year: 2256855 x $1.5 = $337.40625")
        self.assertEqual([(item.computed, item.correct) for item in check_arithmetic(text)],
                         [("222", False), ("798.6", False), ("3385282.5", False)])


class ToolTests(unittest.TestCase):
    def test_the_tool_returns_the_value_and_steps(self):
        result = tools.run("ada_compute", {"expression": "0.15 * (1200 - 300)"}, brief=True)
        self.assertEqual((result["status"], result["value"], result["exact"]), ("computed", "135", True))
        self.assertEqual(result["steps"], ["1200 - 300 = 900", "0.15 * 900 = 135"])

    def test_other_names_for_the_expression_and_digits_as_text(self):
        self.assertEqual(tools.run("ada_compute", {"expr": "2 + 2"})["value"], "4")
        given = '{"expression": "sqrt(2)", "digits": "5"}'
        self.assertEqual(tools.run("ada_compute", given)["value"], "1.4142")

    def test_refusals_and_errors(self):
        self.assertEqual(tools.run("ada_compute", {"expression": "6/2(1+2)"})["code"], "ambiguous_expression")
        self.assertEqual(tools.run("ada_compute", {})["status"], "error")

    def test_the_instructions_send_arithmetic_to_the_tool(self):
        self.assertIn("ada_compute", tools.SYSTEM_PROMPT)


class CommandTests(unittest.TestCase):
    def run_command(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["compute", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_the_value_and_exit_codes(self):
        code, out, _ = self.run_command("2^10", "+", "3/4")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("2^10 + 3/4 = 1024.75"))
        code, _, err = self.run_command("1/0")
        self.assertEqual(code, 3)
        self.assertIn("division_by_zero", err)

    def test_an_expression_that_starts_with_a_minus_sign(self):
        code, out, _ = self.run_command("-2^2", "--json")
        self.assertEqual((code, json.loads(out)["value"]), (0, "-4"))
        code, out, _ = self.run_command("--digits", "5", "sqrt(2)")
        self.assertIn("1.4142", out)


if __name__ == "__main__":
    unittest.main()
