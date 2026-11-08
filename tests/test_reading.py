"""How a model read a word problem: the problem's numbers its working leaves out (ada.reading), through
`ada compute --problem` and the ada_compute tool's problem argument."""

import contextlib
import io
import json
import unittest

import ada
from ada import tools
from ada.cli import main
from ada.reading import check_reading

HOUSE = ("Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in "
         "repairs. This increased the value of the house by 150%. How much profit did he make?")


class ReadingTests(unittest.TestCase):
    def test_a_number_left_out(self):
        reading = check_reading(HOUSE, "cost = 80000; repairs = 50000; value = cost + repairs + repairs / 2; "
                                       "profit = value - cost - repairs")
        self.assertEqual(reading.numbers, ("80,000", "50,000", "150"))
        self.assertEqual(reading.left_out, ("150",))
        self.assertIn("leaves out 150 from the problem", reading.warning)
        self.assertEqual(reading.to_dict(),
                         {"problem_numbers": ["80,000", "50,000", "150"], "left_out": ["150"]})

    def test_every_number_used(self):
        reading = check_reading(HOUSE, "value = 130000 * (1 + 150/100); profit = value - 80000 - 50000")
        self.assertEqual(reading.left_out, ())
        self.assertIsNone(reading.warning)

    def test_a_percentage_may_be_used_divided_by_100(self):
        self.assertEqual(check_reading("A $25 shirt is 20% off.", "25 * (1 - 0.2)").left_out, ())
        self.assertEqual(check_reading("A $25 shirt is 20% off.", "25 * (1 - 20%)").left_out, ())
        self.assertEqual(check_reading("A $25 shirt is 20% off.", "25 * 0.8").left_out, ("20",))

    def test_what_is_not_counted(self):
        # Ordinals, clock times, 0 and 1, and numbers written in words.
        reading = check_reading("On the 3rd day, at 8:30, 1 of the 0 boxes held seven pens and 4 pencils.",
                                "")
        self.assertEqual(reading.numbers, ("4",))

    def test_numbers_as_written_once_each_in_order(self):
        reading = check_reading("She buys 12 eggs, then 2.5 kg of flour, then 12 more eggs.", "")
        self.assertEqual(reading.numbers, ("12", "2.5"))
        self.assertEqual(reading.left_out, ("12", "2.5"))
        self.assertIn("leaves out 12 and 2.5 from the problem", reading.warning)
        self.assertIn("what each stands for", reading.warning)

    def test_numbers_in_the_working_with_and_without_separators(self):
        self.assertEqual(check_reading("12 boxes of 180 pens", "gcd(12,180)").left_out, ())
        self.assertEqual(check_reading("It costs $80,000.", "x = 80,000 / 2").left_out, ())
        reading = check_reading("It is 0.5 km away, or 2 laps of .25 km.", "x = .5 / 2 + .25")
        self.assertEqual(reading.left_out, ())

    def test_a_number_inside_a_word_is_not_read(self):
        self.assertEqual(check_reading("Room B12 has 30 seats.", "seats = 30").numbers, ("30",))
        self.assertEqual(check_reading("It has 30 seats.", "x2 = 3").left_out, ("30",))

    def test_thin_input(self):
        for problem, working in (("", ""), ("", "x = 5"), (None, None), ("No numbers here.", "x = 2")):
            reading = check_reading(problem, working)
            self.assertEqual((reading.numbers, reading.left_out, reading.warning), ((), (), None))
        self.assertEqual(check_reading("He has 4 and 7.", "").left_out, ("4", "7"))

    def test_the_same_answer_each_time(self):
        first = check_reading(HOUSE, "x = 80000")
        self.assertEqual([check_reading(HOUSE, "x = 80000") for _ in range(3)], [first] * 3)

    def test_it_is_public(self):
        self.assertIn("check_reading", ada.__all__)
        self.assertIs(ada.check_reading, check_reading)


class CommandTests(unittest.TestCase):
    def run_command(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["compute", *argv])
        return code, out.getvalue(), err.getvalue()

    def test_the_problem_names_what_is_left_out(self):
        code, out, _ = self.run_command("p = 80000 + 50000", "--problem", HOUSE)
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("p = 130000"))
        self.assertIn("The working leaves out 150 from the problem.", out)
        code, out, _ = self.run_command("--problem=He has 4 apples and gets 3 more.", "4 + 3")
        self.assertEqual(code, 0)
        self.assertIn("4 + 3 = 7", out)
        self.assertIn("The working uses every number of the problem.", out)

    def test_json(self):
        code, out, _ = self.run_command("p = 80000 + 50000", "--json", "--problem", HOUSE)
        found = json.loads(out)
        self.assertEqual((code, found["values"]), (0, {"p": "130000"}))
        self.assertEqual(found["left_out"], ["150"])
        self.assertEqual(found["problem_numbers"], ["80,000", "50,000", "150"])
        self.assertEqual(len(found["warnings"]), 1)
        # Without --problem the result is as before.
        code, out, _ = self.run_command("4 + 3", "--json")
        self.assertFalse({"left_out", "problem_numbers", "warnings"} & set(json.loads(out)))

    def test_a_refusal_still_names_what_is_left_out(self):
        code, out, _ = self.run_command("6/0", "--json", "--problem", "He splits 6 pies among 3 people.")
        found = json.loads(out)
        self.assertEqual((code, found["status"], found["left_out"]), (3, "refused", ["3"]))
        code, _, err = self.run_command("6/0", "--problem", "He splits 6 pies among 3 people.")
        self.assertIn("division_by_zero", err)
        self.assertIn("leaves out 3", err)


class ToolTests(unittest.TestCase):
    def test_the_problem_argument(self):
        found = tools.run("ada_compute", {"expression": "p = 80000 + 50000", "problem": HOUSE})
        self.assertEqual((found["status"], found["left_out"]), ("solved", ["150"]))
        self.assertIn("leaves out 150", found["warnings"][0])
        # A small model's brief result keeps the warning.
        brief = tools.run("ada_compute", {"expression": "p = 80000 + 50000", "problem": HOUSE}, brief=True)
        self.assertIn("leaves out 150", brief["warnings"][0])

    def test_without_a_problem_nothing_is_added(self):
        found = tools.run("ada_compute", {"expression": "4 + 3"})
        self.assertFalse({"left_out", "problem_numbers", "warnings"} & set(found))
        self.assertEqual(tools.run("ada_compute", {"expression": "4 + 3", "problem": "  "}), found)

    def test_a_refusal_with_a_problem(self):
        found = tools.run("ada_compute", {"expression": "6/0", "problem": "He splits 6 pies among 3 people."})
        self.assertEqual((found["status"], found["code"], found["left_out"]),
                         ("refused", "division_by_zero", ["3"]))

    def test_the_definition_offers_the_problem(self):
        definition = next(tool for tool in tools.definitions() if tool["function"]["name"] == "ada_compute")
        properties = definition["function"]["parameters"]["properties"]
        self.assertIn("problem", properties)
        self.assertEqual(definition["function"]["parameters"]["required"], ["expression"])


if __name__ == "__main__":
    unittest.main()
