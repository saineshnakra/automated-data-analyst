"""The benchmarks' solve condition: the model writes equations, ADA solves them (benchmarks/formalize.py)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import formalize  # noqa: E402
from formalize import (  # noqa: E402
    FORMAL_ASK,
    formal_messages,
    latex_value,
    lettered,
    plain_number,
    solved_by_ada,
    worked_turns,
    written,
)


class WrittenTests(unittest.TestCase):
    def test_equations_and_the_answer_line(self):
        equations, target = written("- boxes = 3 * 12\n- total = boxes + 5\n**Answer:** total")
        self.assertEqual((equations, target), (["boxes = 3 * 12", "total = boxes + 5"], "total"))

    def test_what_is_not_an_equation_is_left_out(self):
        equations, _ = written("Let x be the cost.\nx > 2\nx == 3\nx = 5 (dollars).\ny <= 4")
        self.assertEqual(equations, ["x = 5"])

    def test_answer_lines_written_other_ways(self):
        self.assertEqual(written("Answer: \\boxed{x}")[1], "x")
        self.assertEqual(written("Answer: cost = price * count")[1], "cost")
        self.assertEqual(written("Answer: 2 * 7 = 14")[1], "14")
        self.assertIsNone(written("x = 2")[1])

    def test_thin_input(self):
        self.assertEqual(written(""), ([], None))
        self.assertEqual(written("Answer:"), ([], None))

    def test_latex_names_units_and_lets(self):
        equations, _ = written("- let $ \\text{eggs\\_per\\_day} = 16 $\n"
                               "$$ \\text{morning feed} = 15 \\text{ cups} $$\n"
                               "$$ \\text{left} = \\text{eggs\\_per\\_day} \\times 2 "
                               "\\quad \\text{(all of them)} $$")
        self.assertEqual(equations, ["eggs_per_day = 16", "morning_feed = 15", "left = eggs_per_day * 2"])

    def test_plain_units_labels_and_thousands(self):
        equations, _ = written("Step 1: distance = 3 * 60 miles\ncost = 80,000\nlow = gcd(12,180)\n"
                               "area = 3 pi\nsheep = 3 (3 a day)")
        self.assertEqual(equations, ["distance = 3 * 60", "cost = 80000", "low = gcd(12,180)", "area = 3 pi",
                                     "sheep = 3"])

    def test_the_models_own_working_is_not_read(self):
        # A name defined again, the result at the end of a chain and arithmetic with no name in it.
        equations, _ = written("total = boxes + 4\ntotal = 36 + 4\ntotal = 40\nwhite = 2 / 2 = 1\n"
                               "Total: 10 + 15 = 25 liters\n= 48")
        self.assertEqual(equations, ["total = boxes + 4", "white = 2 / 2"])

    def test_the_last_equations_after_thinking_aloud(self):
        reply = ("Let me think. The jacket example had price = 80, so maybe total = 3 * 12.\n"
                 "Wait, I should check. a = ?\n"
                 "So the equations are:\n\ndays = 4 * 7\neggs = 3 * days\ndozens = eggs / 12\n\n"
                 "Answer: dozens\nThat is the number of dozens.")
        equations = ["days = 4 * 7", "eggs = 3 * days", "dozens = eggs / 12"]
        self.assertEqual(written(reply, last=True), (equations, "dozens"))
        self.assertEqual(solved_by_ada(reply)["value"], "7")

    def test_every_equation_when_the_last_ones_have_no_value(self):
        # A sentence between the equations does not cut off the ones before it when ADA needs them.
        reply = ("t = 2 * c\nCharleston has four times as many as Seattle:\nc = 4 * s\ns = 20\n"
                 "Answer: t + c + s")
        self.assertEqual(written(reply, last=True)[0], ["c = 4 * s", "s = 20"])
        found = solved_by_ada(reply)
        self.assertEqual((found["status"], found["value"]), ("solved", "260"))

    def test_a_worked_example_repeated_has_no_value(self):
        reply = ("Like the example:\nprice = 80\ndiscount = price * 25 / 100\npaid = price - discount - 5\n"
                 "Answer: paid")
        self.assertEqual(solved_by_ada(reply)["status"], "example repeated")
        self.assertEqual(solved_by_ada(reply.replace("price = 80", "price = 90"))["value"], "62.5")

    def test_sentences_are_not_read(self):
        equations, _ = written("Toulouse has 2 * 4 = 8 sheep.\nSo the cost is x = 5\nThen y = x + 1")
        self.assertEqual(equations, ["y = x + 1"])


class LetteredTests(unittest.TestCase):
    def test_names_get_letters_of_their_own(self):
        equations, target, mapping = lettered(["eggs_sold = 16 - 7", "money = eggs_sold * 2"], "money")
        self.assertEqual(mapping, {"eggs_sold": "a", "money": "b"})
        self.assertEqual((equations, target), (["a = 16 - 7", "b = a * 2"], "b"))

    def test_letters_already_used_and_functions_stay(self):
        equations, _, mapping = lettered(["a = sqrt(16) + pi", "total = a * comb(5, 2)"], "total")
        self.assertEqual(mapping, {"total": "b"})
        self.assertEqual(equations, ["a = sqrt(16) + pi", "b = a * comb(5, 2)"])

    def test_a_name_with_a_capital_is_the_same_name(self):
        equations, target, _ = lettered(["Total = t + c"], "total")
        self.assertEqual((equations, target), (["a = t + c"], "a"))

    def test_e_is_a_quantity_not_the_constant(self):
        equations, target, mapping = lettered(["e = 16 - 7"], "e")
        self.assertEqual((equations, target), (["a = 16 - 7"], "a"))


class SolvedTests(unittest.TestCase):
    def test_a_chain_of_steps(self):
        found = solved_by_ada("eggs_laid = 16\neggs_used = 3 + 4\neggs_sold = eggs_laid - eggs_used\n"
                              "money = eggs_sold * 2\nAnswer: money")
        self.assertEqual((found["status"], found["value"]), ("solved", "18"))

    def test_a_system_and_an_expression_to_report(self):
        found = solved_by_ada("x + y = 10\nx - y = 2\nAnswer: x * y")
        self.assertEqual((found["status"], found["value"]), ("solved", "24"))

    def test_an_expression_with_no_unknowns(self):
        self.assertEqual(solved_by_ada("Answer: (16 - 3 - 4) * 2")["value"], "18")
        # A bare number is the model's own answer, and is marked as such.
        self.assertEqual(solved_by_ada("Answer: 18")["status"], "number given")

    def test_several_values(self):
        self.assertEqual(solved_by_ada("x^2 = 4\nAnswer: x")["status"], "several values")
        # Counts and amounts are above zero: the one positive value is taken when asked.
        found = solved_by_ada("n * (n + 1) / 2 = 55\nAnswer: n", positive=True)
        self.assertEqual(found["value"], "10")

    def test_roots_and_pi_stay_exact(self):
        self.assertEqual(solved_by_ada("$$y = 2 \\sqrt{2}$$\n$$z = y^2$$\nAnswer: \\boxed{z}")["value"], "8")
        self.assertEqual(solved_by_ada("a = 3\nb = a*sqrt(3)\nAnswer: b")["value"], "3*sqrt(3)")
        self.assertEqual(solved_by_ada("area = pi * 3^2\nAnswer: area")["value"], "9*pi")

    def test_what_has_no_value(self):
        self.assertEqual(solved_by_ada("x = 2*y\nAnswer: x")["status"], "no value")
        self.assertEqual(solved_by_ada("x = 5\ny = x/0\nAnswer: y")["status"], "refused")
        self.assertEqual(solved_by_ada("x = 5")["status"], "no answer line")
        self.assertEqual(solved_by_ada("")["value"], None)


class MessagesTests(unittest.TestCase):
    def test_worked_examples_come_before_the_question(self):
        messages = formal_messages("How many?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual([message["role"] for message in messages[1:]],
                         ["user", "assistant"] * ((len(messages) - 2) // 2) + ["user"])
        self.assertEqual(messages[-1]["content"], f"How many?\n\n{FORMAL_ASK}")
        # Each worked example is read by ADA to the value its question asks for; a reply that repeats one has
        # no value.
        answers = [message["content"] for message in messages if message["role"] == "assistant"]
        self.assertEqual([formalize._solved(*written(answer), True)["value"] for answer in answers],
                         ["55", "24", "10", "9"])
        self.assertEqual({solved_by_ada(answer)["status"] for answer in answers}, {"example repeated"})

    def test_the_worked_answers_agree_with_ada(self):
        from ada.compute import check_arithmetic  # noqa: PLC0415

        answers = [turn["content"] for turn in worked_turns() if turn["role"] == "assistant"]
        self.assertEqual([answer.rsplit("Answer: ", 1)[1] for answer in answers], ["55", "24", "10", "9"])
        # Every step written in them is right.
        self.assertTrue(all(item.correct for answer in answers for item in check_arithmetic(answer)))


class ValueTests(unittest.TestCase):
    def test_plain_numbers_and_latex(self):
        self.assertEqual([plain_number(value) for value in ("18", "-2.25", "16/57", "3*sqrt(3)")],
                         ["18", "-2.25", "0.280701754386", "5.19615242271"])
        self.assertIsNone(plain_number("I"))
        self.assertEqual(latex_value("3*sqrt(3)"), "3 \\sqrt{3}")
        self.assertEqual(latex_value("16/57"), "\\frac{16}{57}")


if __name__ == "__main__":
    unittest.main()
