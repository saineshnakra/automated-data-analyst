"""How the GSM8K benchmark combines the model's own answer with ADA's (benchmarks/gsm8k.py)."""

import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import gsm8k  # noqa: E402
from gsm8k import agree, chat, combined, formalized, large_numbers, programmed, structured  # noqa: E402


def made(own, ada, slip=False, checked=None):
    """The records of none, check and solve for one problem."""
    return {"none": {"given": own, "reply": f"Answer: {own}"},
            "check": {"given": checked or own, "reply": f"Answer: {checked or own}",
                      "checks": [{"correct": not slip}]},
            "solve": {"given": ada, "reply": "" if ada is None else f"Answer: {ada}",
                      "ada": {"equations": ["a = 2 + 3"], "target": "a", "value": ada}}}


class CombinedTests(unittest.TestCase):
    def decided(self, records, question="Tom has 3 apples.", tolerance=Fraction(0)):
        found = combined("", "", question, records, tolerance, ask=False)
        return found["reply"], found["decided_by"]

    def test_the_answer_both_give(self):
        self.assertEqual(self.decided(made("5", "5")), ("Answer: 5", "both agree"))
        # Within the data set's tolerance is the same answer.
        self.assertEqual(self.decided(made("5", "5.0004"), tolerance=Fraction(1, 1000))[1], "both agree")

    def test_no_answer_from_ada_leaves_the_checked_answer(self):
        self.assertEqual(self.decided(made("6", None, slip=True, checked="5")),
                         ("Answer: 5", "no answer from ADA"))

    def test_a_wrong_equation_in_the_models_answer_takes_adas(self):
        self.assertEqual(self.decided(made("6", "5", slip=True)),
                         ("Answer: 5", "a wrong equation in its answer"))

    def test_large_numbers_take_adas(self):
        self.assertEqual(self.decided(made("6", "5"), "Tom has 12,345 apples."),
                         ("Answer: 5", "large numbers"))

    def test_otherwise_the_models_own(self):
        self.assertEqual(self.decided(made("6", "5"), "Tom has 1,234 apples."),
                         ("Answer: 6", "the model's answer"))


class RetryTests(unittest.TestCase):
    def asked(self, *replies):
        """formalized under retry, with the model's replies in turn; the result and how often it was asked."""
        with unittest.mock.patch.object(gsm8k, "chat", side_effect=[{"content": r} for r in replies]) as chat:
            return formalized("", "", "Tom has 3 apples.", retry=True), chat.call_count

    def test_equations_with_a_value_are_not_asked_again(self):
        found, calls = self.asked("apples = 3 + 2\nAnswer: apples")
        self.assertEqual((found["reply"], calls), ("Answer: 5", 1))
        self.assertNotIn("first_ada", found)

    def test_a_number_instead_of_equations_is_asked_again(self):
        found, calls = self.asked("Answer: 6", "apples = 3 + 2\nAnswer: apples")
        self.assertEqual((found["reply"], calls), ("Answer: 5", 2))
        self.assertEqual(found["first_ada"]["status"], "number given")
        self.assertIn("a number you worked out", found["feedback"])

    def test_the_first_number_stands_when_the_second_reply_has_no_value(self):
        found, _ = self.asked("Answer: 6", "apples = 3 + pears\nAnswer: apples")
        self.assertEqual(found["reply"], "Answer: 6")
        self.assertTrue(found["kept_first"])
        self.assertEqual(found["second_ada"]["status"], "no value")

    def test_no_value_is_asked_again_as_before(self):
        found, _ = self.asked("apples = 3 + pears\nAnswer: apples", "apples = 3 + 2\nAnswer: apples")
        self.assertEqual(found["reply"], "Answer: 5")
        self.assertNotIn("kept_first", found)

    def test_thin_input(self):
        found, calls = self.asked("", "")
        self.assertEqual((found["reply"], calls), ("", 2))


class PairTests(unittest.TestCase):
    QUESTION = "Tom has 3 apples and buys 4 bags of 6 pears. How many pieces of fruit does he have?"

    def asked(self, *replies):
        """formalized under pair, with the model's replies in turn; the result and how often it was asked."""
        with unittest.mock.patch.object(gsm8k, "chat", side_effect=[{"content": r} for r in replies]) as chat:
            return formalized("", "", self.QUESTION, pair=True), chat.call_count

    def test_equations_that_use_every_number_are_not_asked_again(self):
        found, calls = self.asked("apples = 3\npears = 4 * 6\nfruit = apples + pears\nAnswer: fruit")
        self.assertEqual((found["reply"], calls, found["left_out"]), ("Answer: 27", 1, []))
        self.assertNotIn("feedback", found)

    def test_a_number_left_out_is_named_and_asked_again(self):
        found, calls = self.asked("apples = 3\npears = 6\nfruit = apples + pears\nAnswer: fruit",
                                  "apples = 3\npears = 4 * 6\nfruit = apples + pears\nAnswer: fruit")
        self.assertEqual((found["reply"], calls), ("Answer: 27", 2))
        self.assertEqual((found["left_out"], found["second_left_out"]), (["4"], []))
        self.assertIn("leave out 4 from the problem", found["feedback"])
        self.assertEqual(found["first_ada"]["value"], "9")
        self.assertNotIn("kept_first", found)

    def test_the_first_answer_stands_unless_the_second_leaves_out_fewer(self):
        # The second reply leaves out as many numbers: the first stands.
        found, _ = self.asked("apples = 3\npears = 6\nfruit = apples + pears\nAnswer: fruit",
                              "apples = 3\nbags = 4\nfruit = apples + bags\nAnswer: fruit")
        self.assertEqual(found["reply"], "Answer: 9")
        self.assertTrue(found["kept_first"])
        self.assertEqual(found["second_left_out"], ["6"])
        # The second reply has no value: the first stands.
        found, _ = self.asked("apples = 3\npears = 6\nfruit = apples + pears\nAnswer: fruit",
                              "pears = 4 * 6\nfruit = apples + pears\nAnswer: fruit")
        self.assertEqual(found["reply"], "Answer: 9")
        self.assertTrue(found["kept_first"])

    def test_no_value_is_asked_again_as_under_retry_with_the_numbers_left_out(self):
        found, calls = self.asked("fruit = apples + pears\nAnswer: fruit",
                                  "apples = 3\npears = 4 * 6\nfruit = apples + pears\nAnswer: fruit")
        self.assertEqual((found["reply"], calls), ("Answer: 27", 2))
        self.assertIn("leave out 3, 4 and 6 from the problem", found["feedback"])
        self.assertIn("ADA found no value for fruit", found["feedback"])
        # A number given, then no value: the first number stands, as under retry.
        found, _ = self.asked("Answer: 27", "fruit = apples + pears\nAnswer: fruit")
        self.assertEqual(found["reply"], "Answer: 27")
        self.assertTrue(found["kept_first"])
        self.assertIn("a number you worked out", found["feedback"])

    def test_retry_is_as_before(self):
        # Under retry, equations ADA solves are not asked about, whatever they leave out.
        with unittest.mock.patch.object(gsm8k, "chat", return_value={
                "content": "apples = 3\npears = 6\nfruit = apples + pears\nAnswer: fruit"}) as chat:
            found = formalized("", "", self.QUESTION, retry=True)
        self.assertEqual((found["reply"], chat.call_count), ("Answer: 9", 1))
        self.assertNotIn("left_out", found)

    def test_thin_input(self):
        found, calls = self.asked("", "")
        self.assertEqual((found["reply"], calls), ("", 2))
        self.assertEqual(found["left_out"], ["3", "4", "6"])


class ProgramTests(unittest.TestCase):
    def test_the_program_answers(self):
        reply = "```python\ndef solution():\n    apples = 3 + 2\n    return apples\n```"
        with unittest.mock.patch.object(gsm8k, "chat", return_value={"content": reply}):
            found = programmed("", "", "Tom has 3 apples.")
        self.assertEqual(found["reply"], "Answer: 5")
        self.assertEqual(found["program"]["status"], "ran")
        with unittest.mock.patch.object(gsm8k, "chat", return_value={"content": "It is 5."}):
            self.assertEqual(programmed("", "", "Tom has 3 apples.")["reply"], "")


class JsonTests(unittest.TestCase):
    def test_the_reply_is_held_to_the_schema_and_read_as_equations(self):
        reply = json.dumps({"equations": ["apples = 3 + 2"], "answer": "apples"})
        with unittest.mock.patch.object(gsm8k, "chat", return_value={"content": reply}) as asked:
            found = structured("", "", "Tom has 3 apples.")
        self.assertEqual(found["reply"], "Answer: 5")
        self.assertEqual(asked.call_args.kwargs["form"], gsm8k.JSON_FORMAT)
        # A reply that is not such JSON is read as it is.
        plain = {"content": "apples = 4\nAnswer: apples"}
        with unittest.mock.patch.object(gsm8k, "chat", return_value=plain):
            self.assertEqual(structured("", "", "Tom has 3 apples.")["reply"], "Answer: 4")
        with unittest.mock.patch.object(gsm8k, "chat", return_value={"content": ""}):
            self.assertEqual(structured("", "", "Tom has 3 apples.")["reply"], "")


class ChatTests(unittest.TestCase):
    def sent(self, **kwargs):
        """The request body chat() sends to the model server."""
        response = unittest.mock.MagicMock()
        answered = json.dumps({"message": {"content": "ok"}}).encode()
        response.__enter__.return_value.read.return_value = answered
        opener = unittest.mock.MagicMock()
        opener.open.return_value = response
        with unittest.mock.patch.object(gsm8k.urllib.request, "build_opener", return_value=opener):
            chat("http://x", "m", [{"role": "system", "content": "Solve."}, {"role": "user", "content": "Q"}],
                 [], **kwargs)
        return json.loads(opener.open.call_args.args[0].data)

    def test_settings_schema_and_system_suffix(self):
        body = self.sent()
        self.assertEqual((body["options"]["temperature"], body["options"]["seed"]), (0, 0))
        self.assertNotIn("format", body)
        body = self.sent(options={"temperature": 0.7, "seed": 3}, form={"type": "object"})
        self.assertEqual((body["options"]["temperature"], body["options"]["seed"]), (0.7, 3))
        self.assertEqual(body["format"], {"type": "object"})
        with unittest.mock.patch.object(gsm8k, "SYSTEM_SUFFIX", "/no_think"):
            body = self.sent()
        self.assertEqual(body["messages"][0]["content"], "Solve. /no_think")
        self.assertEqual(body["messages"][1]["content"], "Q")


class SampleTests(unittest.TestCase):
    def run_benchmark(self, *flags):
        """gsm8k.py on two problems with a model that always answers 5, and the records it writes."""
        replies = []

        def model(base, name, messages, offered, max_tokens=0, think=False, options=None, form=None):
            replies.append(options)
            if form:
                return {"content": json.dumps({"equations": ["apples = 3 + 2"], "answer": "apples"})}
            if "exact solver" in messages[0]["content"] or "equations" in messages[0]["content"]:
                return {"content": "apples = 3 + 2\nAnswer: apples"}
            return {"content": "3 + 2 = 5\nAnswer: 5"}

        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "test.jsonl"
            rows = [{"question": f"Tom has {n} apples.", "answer": "#### 5"} for n in (3, 4)]
            data.write_text("".join(json.dumps(row) + "\n" for row in rows))
            out = Path(folder) / "out.jsonl"
            argv = ["gsm8k.py", str(data), "--out", str(out), "--problems", "2", *flags]
            with unittest.mock.patch.object(gsm8k, "chat", side_effect=model), \
                    unittest.mock.patch.object(sys, "argv", argv), \
                    unittest.mock.patch("sys.stdout", io.StringIO()):
                gsm8k.main()
            return [json.loads(line) for line in out.read_text().splitlines()], replies

    def test_conditions_are_asked_as_many_times_as_asked(self):
        records, replies = self.run_benchmark("--condition", "none,check,solve,json,combine",
                                              "--samples", "3", "--sampled", "none,json")
        counts = {}
        for record in records:
            counts[record["condition"]] = counts.get(record["condition"], 0) + 1
        # Two problems: none and json three times each, check, solve and combine once.
        self.assertEqual(counts, {"none": 6, "check": 2, "solve": 2, "json": 6, "combine": 2})
        self.assertEqual(sorted({r["sample"] for r in records if r["condition"] == "json"}), [0, 1, 2])
        # The first sample is at temperature 0; the others at SAMPLE_TEMPERATURE with their own seed.
        seeds = sorted({(o or {}).get("seed", 0) for o in replies})
        self.assertEqual(seeds, [0, 1, 2])
        self.assertTrue(all(o["temperature"] == gsm8k.SAMPLE_TEMPERATURE for o in replies if o))

    def test_pair_is_a_condition_that_may_be_sampled(self):
        records, _ = self.run_benchmark("--condition", "pair", "--samples", "2", "--sampled", "pair")
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["correct"] for record in records))
        # "Tom has 4 apples." with apples = 3 + 2 leaves out 4: asked again, the same reply leaves out as
        # many, and the first answer stands.
        fours = [record for record in records if record["index"] == 1]
        self.assertTrue(all(r["left_out"] == ["4"] and r["kept_first"] for r in fours))
        self.assertTrue(all(r["left_out"] == [] for r in records if r["index"] == 0))

    def test_a_run_of_one_sample_is_as_before(self):
        records, replies = self.run_benchmark("--condition", "none,solve")
        self.assertEqual([(r["condition"], r["sample"]) for r in records],
                         [("none", 0), ("solve", 0), ("none", 0), ("solve", 0)])
        self.assertTrue(all(o is None for o in replies))

    def test_check_and_combine_are_not_sampled(self):
        with self.assertRaises(SystemExit), unittest.mock.patch("sys.stderr", io.StringIO()):
            self.run_benchmark("--condition", "none,check", "--samples", "2", "--sampled", "check")


class PartsTests(unittest.TestCase):
    def test_large_numbers(self):
        self.assertTrue(large_numbers("A farm sells 48,000 eggs."))
        self.assertTrue(large_numbers("It costs 12345.5 dollars."))
        self.assertFalse(large_numbers("It costs 1234.56789 dollars."))
        self.assertFalse(large_numbers("No numbers here."))
        self.assertFalse(large_numbers(""))

    def test_agree(self):
        self.assertTrue(agree("18", "18", Fraction(0)))
        self.assertFalse(agree("18", "19", Fraction(0)))
        self.assertFalse(agree(None, "18", Fraction(0)))


if __name__ == "__main__":
    unittest.main()
