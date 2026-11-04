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
from ada.latex import to_plain


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

    def test_latex_names_beside_a_step_are_skipped(self):
        # A Greek letter or a function name next to a step multiplies it or takes it.
        self.assertEqual(check_arithmetic("S = 4 \\pi (3)^2 = 4 \\pi \\cdot 9 = 36 \\pi"), [])
        self.assertEqual(check_arithmetic("Since $\\log_{10}(10000) = \\log_{10}(10^4) = 4$, x = 4."), [])
        self.assertEqual(check_arithmetic("(3)^2 = 4 \\pi"), [])
        # Other LaTeX commands do not hide a slip.
        found = check_arithmetic("- $ x = -3 $: $ 81 - 135 + 81 + 3 - 14 = 28 \\ne 0 $")
        self.assertEqual([(item.computed, item.correct) for item in found], [("16", False)])
        self.assertFalse(check_arithmetic("\\quad 3 + 4 = 8")[0].correct)
        for text in ("\\pi", "= \\pi", "\\", "$\\log$", "\\pi ="):
            self.assertEqual(check_arithmetic(text), [])

    def test_congruences_and_other_bases(self):
        # On a line about congruences, = can mean "leaves the same remainder": a step is a slip only when its
        # sides leave different remainders for the line's modulus.
        self.assertEqual(check_arithmetic("3^3 = 6 \\mod 7 \\\\\n3^4 = 18 \\equiv 4 \\mod 7"), [])
        self.assertEqual(check_arithmetic("17 mod 5 = 2, and modulo 7, 3 + 4 = 0"), [])
        found = check_arithmetic("\\[ 2 + 2 \\cdot 21 = 42 \\equiv 2 \\pmod{5} \\]")
        self.assertEqual([(item.expression, item.computed, item.correct) for item in found],
                         [("2 + 2 * 21", "44", False)])
        found = check_arithmetic("17^{82} = 17^{64} \\cdot 17^{16} \\equiv 68 \\cdot 48 \\pmod{83}")
        self.assertEqual([item.correct for item in found], [False])
        self.assertTrue(check_arithmetic("2 + 2 \\cdot 16 = 34 \\equiv 4 \\pmod{5}")[0].correct)
        # With no modulus, or two, the remainder cannot be worked out, and the step is dropped.
        self.assertEqual(check_arithmetic("2 + 2 = 5 \\equiv 1"), [])
        self.assertEqual(check_arithmetic("2 + 2 = 9 (mod 5) and (mod 7)"), [])
        # A value with a base or an index is not the number written.
        self.assertEqual(check_arithmetic("2 \\cdot 5^2 + 0 \\cdot 5^1 + 4 \\cdot 5^0 = 204_5"), [])
        found = check_arithmetic("204_5 = 2*25 + 4 = 55")
        self.assertEqual([(item.expression, item.correct) for item in found], [("2*25 + 4", False)])
        # Words that start with mod do not change anything.
        self.assertFalse(check_arithmetic("The model gives 2 + 2 = 5")[0].correct)
        for text in ("mod", "\\equiv", "= 204_", "_5 = 5", "\\pmod{0} 3 + 4 = 8"):
            self.assertEqual(check_arithmetic(text), [])

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


class LatexTests(unittest.TestCase):
    """Arithmetic written in LaTeX, read by ada.latex. The lines are from MATH-500 and GSM-Hard replies."""

    def checked(self, text):
        return [(item.expression, item.stated, item.correct) for item in check_arithmetic(text)]

    def test_each_side_is_read_into_an_expression(self):
        self.assertEqual(to_plain("\\frac{1}{2}\\sqrt{8}"), "((1)/(2))sqrt(8)")
        self.assertEqual(to_plain("\\sqrt[3]{8} + 2^{10} + \\binom{6}{2}"),
                         "root((8), (3)) + 2^(10) + comb((6), (2))")
        self.assertEqual(to_plain("\\left(-\\frac{7}{4}\\right) \\cdot 1{,}000"), "(-((7)/(4))) * 1000")
        self.assertEqual(to_plain("\\boxed{7}"), "(7)")
        for unread in ("x + 1", "3\\frac{1}{2}", "2\\sqrt{2}^2", "\\pm 3", "90^\\circ", "\\frac{1}{2", ""):
            self.assertIsNone(to_plain(unread), unread)

    def test_every_step_of_a_chain_is_checked(self):
        found = self.checked("\\sqrt{27} = \\sqrt{9 \\cdot 3} = \\sqrt{9} \\cdot \\sqrt{3} = 3\\sqrt{3}")
        self.assertEqual(len(found), 3)
        self.assertTrue(all(correct for _, _, correct in found))
        self.assertEqual(self.checked("= 3 + \\frac{5}{3} = \\frac{9}{3} + \\frac{5}{3} = \\frac{14}{3}"),
                         [("3 + \\frac{5}{3}", "\\frac{9}{3} + \\frac{5}{3}", True),
                          ("\\frac{9}{3} + \\frac{5}{3}", "\\frac{14}{3}", True)])
        self.assertEqual(self.checked("x = \\frac{40}{0.02} = 2000"), [("\\frac{40}{0.02}", "2000", True)])

    def test_slips_from_model_answers_are_found(self):
        self.assertEqual(self.checked("11 + 18 + 18 + 18 + 10 + 2 = \\boxed{87}"),
                         [("11 + 18 + 18 + 18 + 10 + 2", "\\boxed{87}", False)])
        self.assertEqual(self.checked("= \\frac{1}{\\frac{4}{169}} = 169"),
                         [("\\frac{1}{\\frac{4}{169}}", "169", False)])
        self.assertFalse(check_arithmetic("\\sqrt{27} = 3\\sqrt{2}")[0].correct)
        self.assertFalse(check_arithmetic("$\\binom{6}{2} = 12$")[0].correct)
        self.assertEqual(check_arithmetic("\\frac{4}{12} \\cdot 10 = \\frac{5}{3}")[0].computed, "10/3")
        found = check_arithmetic("So $\\frac{3}{4} \\times 20 = 15$ apples and "
                                 "$\\frac{1}{4} \\times 20 = 6$.")
        self.assertEqual([item.correct for item in found], [True, False])

    def test_values_that_are_not_rational_agree_to_30_digits(self):
        self.assertTrue(check_arithmetic("(\\sqrt{7} + \\sqrt{5})^2 = 12 + 2\\sqrt{35}")[0].correct)
        self.assertTrue(check_arithmetic("\\frac{2\\pi}{4} = \\frac{\\pi}{2}")[0].correct)
        self.assertFalse(check_arithmetic("\\sqrt{2} = 1.5")[0].correct)
        self.assertTrue(check_arithmetic("\\sqrt{2} = 1.41")[0].correct)

    def test_rounded_decimals_agree_to_their_places(self):
        line = "4 + \\sqrt{4} + \\sqrt[3]{4} + \\sqrt[4]{4} = 4 + 2 + 1.587 + 1.414 = 10"
        self.assertEqual([correct for _, _, correct in self.checked(line)], [True, False])
        self.assertTrue(check_arithmetic("0.5 + \\frac{1}{1.5} + 2 = 0.5 + 0.666 + 2")[0].correct)
        self.assertTrue(check_arithmetic("\\frac{1}{3} = 0.33")[0].correct)

    def test_what_reads_two_ways_or_is_not_claimed_is_skipped(self):
        # A value put into an equation to test it, a mixed number, (2√2)² written without parentheses.
        self.assertEqual(check_arithmetic("1 = \\sqrt{11 - 2(1)} + 4"), [])
        self.assertEqual(check_arithmetic("3\\frac{1}{2} = 3.5"), [])
        self.assertEqual(check_arithmetic("2\\sqrt{2}^2 - 8 = 0"), [])
        # Conditions, contradictions and running totals.
        self.assertEqual(check_arithmetic("If \\frac{1}{2} + \\frac{1}{2} = 2, then"), [])
        self.assertEqual(check_arithmetic("\\frac{1}{2} + \\frac{1}{2} = 2, which is impossible"), [])
        self.assertEqual(self.checked("3 \\times 4 = 12 + \\frac{1}{2} = 12.5"),
                         [("12 + \\frac{1}{2}", "12.5", True)])
        # A line of a displayed block that goes on from the line before.
        self.assertEqual(self.checked("+ \\frac{3}{10} = 1.1667 + 0.3 = 1.4667"),
                         [("1.1667 + 0.3", "1.4667", True)])
        # Letters, signs ADA does not read, and words that scale the value.
        self.assertEqual(check_arithmetic("\\frac{x}{2} = 3, \\quad 90^\\circ = \\frac{\\pi}{2}"), [])
        self.assertEqual(check_arithmetic("2^{10} = 1024 \\text{ million}"), [])
        self.assertEqual(check_arithmetic("x = \\frac{-5 \\pm \\sqrt{25 - 24}}{2}"), [])

    def test_words_around_a_chain(self):
        found = self.checked("The area is \\frac{1}{2} \\cdot 6 \\cdot 4 = 12 square units.")
        self.assertEqual(found, [("\\frac{1}{2} \\cdot 6 \\cdot 4", "12", True)])
        self.assertEqual(check_arithmetic("\\frac{14}{2} = 2x"), [])
        self.assertEqual(check_arithmetic("\\frac{1}{2} \\cdot S + 2 = 28"), [])

    def test_steps_with_letters_are_checked_as_identities(self):
        self.assertEqual(self.checked("S = (p - 1) - (q - 1) = p - q"),
                         [("(p - 1) - (q - 1)", "p - q", True)])
        self.assertEqual(self.checked("S = x^2 + y^2 = (x+y)^2 - 2xy"),
                         [("x^2 + y^2", "(x+y)^2 - 2xy", True)])
        # Slips from MATH-500 replies, with LaTeX and without.
        self.assertEqual(self.checked("T = 4c + 4c + 4c + 4c + 4c = 5c"),
                         [("4c + 4c + 4c + 4c + 4c", "5c", False)])
        self.assertEqual(self.checked("P = (-1)(-2)(-3)(-r) = -6r"), [("(-1)(-2)(-3)(-r)", "-6r", False)])
        found = check_arithmetic("f(x) = \\frac{x^2}{2} + \\frac{x}{2} - 1 = \\frac{x^2 + x - 2}{2} = "
                                 "\\frac{(x+1)(x-2)}{2}")
        self.assertEqual([item.correct for item in found], [True, False])
        self.assertEqual(found[1].computed, "0.5x^2 + 0.5x - 1")
        # A line of a displayed block that starts with =.
        self.assertEqual(self.checked("= x + x = 2x"), [("x + x", "2x", True)])

    def test_what_may_not_be_an_identity_is_skipped(self):
        # The first step may be an equation to solve; the next rewrites one side.
        line = "b + 5 = \\frac{7}{3} \\cdot \\frac{3b - 12}{4} = \\frac{7(3b - 12)}{12}"
        self.assertEqual(self.checked(line),
                         [("\\frac{7}{3} \\cdot \\frac{3b - 12}{4}", "\\frac{7(3b - 12)}{12}", True)])
        self.assertEqual(check_arithmetic("3x + 2 = x + 10"), [])
        # A step that lowers a power puts in what an equation says, here x^2 = 3x - 1.
        self.assertEqual(self.checked("x^3 = x \\cdot x^2 = x(3x - 1) = 3x^2 - x"),
                         [("x(3x - 1)", "3x^2 - x", True)])
        # Other letters on each side, a function at a value, a derivative, and words.
        self.assertEqual(self.checked("S = (x+y)^2 - 2xy = 25 - 12"), [])
        self.assertEqual(check_arithmetic("f(3) = 2 - 2f(2) = 2 - 2(-1 - 2f(1)) = 4 + 4f(1)"), [])
        self.assertEqual(self.checked("y = \\frac{d}{dx}(x^3) = 3x^2 = x^2 + 2x^2"),
                         [("3x^2", "x^2 + 2x^2", True)])
        self.assertEqual(check_arithmetic("2 bolts + 1 bolt = 3 bolts = 3 bolts"), [])
        self.assertEqual(self.checked("= x + x = 2xy"), [])

    def test_thin_and_unusual_input(self):
        self.assertEqual(check_arithmetic("\\frac{1}{2}"), [])
        self.assertEqual(check_arithmetic("$$\\frac{}{}$$ = "), [])
        self.assertEqual(check_arithmetic("\\frac{" * 60 + "1" + "}{2}" * 60 + " = 1"), [])
        self.assertEqual(check_arithmetic("\\boxed{}"), [])


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
