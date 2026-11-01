"""Exact arithmetic: values, steps, conventions stated as assumptions, and refusals with their reasons."""

import contextlib
import io
import json
import unittest
from decimal import Decimal, localcontext

from ada import tools
from ada.api import REFUSAL_CODES, Refusal
from ada.cli import main
from ada.compute import MAX_STEPS, compute, repeating


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
