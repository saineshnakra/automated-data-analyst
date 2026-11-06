"""The program condition of the GSM8K benchmark (benchmarks/program.py): the model's program is read, refused
when it does more than arithmetic, and run in its own process."""

import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import program  # noqa: E402
from program import PROGRAMS, program_in, program_messages, refusal, rounded, run_program  # noqa: E402


def fenced(code):
    return f"Here is the program.\n```python\n{code}\n```"


class RunTests(unittest.TestCase):
    def test_the_examples_give_their_answers(self):
        self.assertEqual([run_program(fenced(code))["value"] for code in PROGRAMS], ["55", "24", "10", "9"])

    def test_math_and_floats(self):
        found = run_program(fenced("import math\ndef solution():\n    return math.sqrt(2)"))
        self.assertEqual((found["status"], found["value"]), ("ran", "1.414214"))
        # A float's rounding error does not change the answer.
        self.assertEqual(run_program(fenced("def solution():\n    return 0.1 + 0.2"))["value"], "0.3")
        floor = "from math import floor\ndef solution():\n    return floor(7 / 2)"
        self.assertEqual(run_program(fenced(floor))["value"], "3")

    def test_a_program_that_fails_has_no_answer(self):
        for code, message in (("def solution():\n    return 1 / 0", "ZeroDivisionError"),
                              ("def solution():\n    return 'seven'", "returned str"),
                              ("def solution():\n    return float('inf')", "returned inf"),
                              ("def answer():\n    return 3", "KeyError")):
            found = run_program(fenced(code))
            self.assertEqual((found["status"], found["value"]), ("failed", None), code)
            self.assertIn(message, found["message"])

    def test_a_program_that_runs_too_long_is_stopped(self):
        with unittest.mock.patch.object(program, "CPU_SECONDS", 1):
            found = run_program(fenced("def solution():\n    while True:\n        pass"))
        self.assertEqual(found["status"], "failed")
        self.assertIn("was stopped", found["message"])


class RefusalTests(unittest.TestCase):
    def test_only_arithmetic_is_run(self):
        for code in ("import os\ndef solution():\n    return 1",
                     "from os import path\ndef solution():\n    return 1",
                     "def solution():\n    return ().__class__",
                     "def solution():\n    return __import__('os')",
                     "def solution():\n    return (lambda: 1)()",
                     "class A:\n    pass\ndef solution():\n    return 1",
                     "def solution():\n    with x:\n        return 1",
                     "def solution():\n    return math.__dict__"):
            self.assertIsNotNone(refusal(code), code)
            self.assertEqual(run_program(fenced(code))["status"], "refused", code)
        self.assertIsNone(refusal(PROGRAMS[0]))

    def test_built_ins_beyond_arithmetic_are_missing(self):
        # open, eval and print are not refused when read, but the program cannot call them.
        for code in ("def solution():\n    open('/etc/passwd')\n    return 1",
                     "def solution():\n    return eval('1 + 1')",
                     "def solution():\n    print(3)\n    return 2"):
            found = run_program(fenced(code))
            self.assertEqual(found["status"], "failed", code)
            self.assertIn("NameError", found["message"])


class PartsTests(unittest.TestCase):
    def test_the_program_in_a_reply(self):
        self.assertEqual(program_in("```python\na = 1\n```\nthen\n```python\nb = 2\n```"), "b = 2")
        self.assertEqual(program_in("def solution():\n    return 4"), "def solution():\n    return 4")
        self.assertIsNone(program_in("The answer is 4."))

    def test_rounded(self):
        self.assertEqual(rounded("12"), "12")
        self.assertEqual(rounded("2.5"), "2.5")
        self.assertEqual(rounded("-1.0000004"), "-1")
        self.assertEqual(rounded("1e+20"), "100000000000000000000")
        self.assertEqual(rounded("123456789123456789123"), "123456789123456789123")

    def test_the_chat_has_the_examples_of_solve(self):
        messages = program_messages("How many?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(len(messages), 10)
        self.assertEqual(messages[-1], {"role": "user", "content": "How many?"})

    def test_thin_input(self):
        self.assertEqual(run_program("")["status"], "no program")
        self.assertEqual(run_program("```python\n```")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
