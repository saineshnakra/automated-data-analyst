"""The paper's run files and tables (paper/collect.py and paper/analysis.py)."""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))

from analysis import grouped, mcnemar, paired, report, retried  # noqa: E402
from collect import collect, named  # noqa: E402
from combine_rules import replaced  # noqa: E402


def record(dataset, model, index, condition, correct, **extra):
    return {"dataset": dataset, "model": model, "index": index, "condition": condition, "correct": correct,
            **extra}


class CollectTests(unittest.TestCase):
    def test_the_data_set_and_model_from_a_file_name(self):
        self.assertEqual(named(Path("gsm8k-qwen2.5-1.5b-3.jsonl")), ("gsm8k", "qwen2.5:1.5b"))
        self.assertEqual(named(Path("gsmhard-qwen3-4b-instruct-0.jsonl")), ("gsmhard", "qwen3:4b-instruct"))
        self.assertEqual(named(Path("gsm8k-qwen3-4b-0.jsonl")), ("gsm8k", "qwen3:4b"))
        with self.assertRaises(ValueError):
            named(Path("results.jsonl"))
        with self.assertRaises(ValueError):
            named(Path("gsm8k-unknown-model-0.jsonl"))

    def test_the_same_records_give_the_same_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "results"
            source.mkdir()
            (source / "gsm8k-llama3.2-1b-1.jsonl").write_text(
                json.dumps({"index": 1, "condition": "none", "correct": True}) + "\n")
            (source / "gsm8k-llama3.2-1b-0.jsonl").write_text(
                json.dumps({"index": 0, "condition": "none", "correct": False}) + "\n"
                # The same problem and condition asked again: the later record stands.
                + json.dumps({"index": 0, "condition": "none", "correct": True}) + "\n\n")
            first, second = Path(folder) / "a", Path(folder) / "b"
            self.assertEqual(collect(source, first, True), 2)
            collect(source, second, True)
            data = (first / "results.jsonl.gz").read_bytes()
            self.assertEqual(data, (second / "results.jsonl.gz").read_bytes())
            rows = [json.loads(line) for line in gzip.decompress(data).decode().splitlines()]
            self.assertEqual([(row["index"], row["correct"], row["model"]) for row in rows],
                             [(0, True, "llama3.2:1b"), (1, True, "llama3.2:1b")])

    def test_records_without_a_model_need_the_file_name(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "results"
            source.mkdir()
            (source / "gsm8k-llama3.2-1b-0.jsonl").write_text(json.dumps({"index": 0, "condition": "none"}))
            with self.assertRaises(ValueError):
                collect(source, Path(folder) / "out", False)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("gsm8k", "m", 0, "none", True), record("gsm8k", "m", 0, "solve", False, given=None),
            record("gsm8k", "m", 1, "none", False), record("gsm8k", "m", 1, "solve", True, given="5"),
            record("gsm8k", "m", 2, "none", False), record("gsm8k", "m", 2, "solve", True, given="7"),
            # A second sample does not count, and a problem without every condition is left out.
            record("gsm8k", "m", 2, "none", True, sample=1), record("gsm8k", "m", 3, "none", True),
            record("gsmhard", "m", 0, "none", False), record("gsmhard", "m", 0, "solve", True, given="9"),
        ]

    def test_against_the_model_alone(self):
        problems = grouped(self.records)[("gsm8k", "m")]
        gained, lost, p = mcnemar(problems, [0, 1, 2], "solve")
        self.assertEqual((gained, lost), (2, 1))
        self.assertAlmostEqual(p, 1.0)

    def test_the_tables(self):
        text = report(self.records)
        self.assertIn("| m | 3 | 1 | 2 | 3 |", text)
        self.assertIn("| m | 2 / 1, p = 1 |", text)
        # Problem 0 has no answer under solve, so the model's own right answer stands.
        self.assertIn("ADA's answer, else the model's", text)

    def test_the_same_problems_with_larger_numbers(self):
        lines = paired(grouped(self.records))
        # Problem 0 is right alone in GSM8K; problem 3 has no GSM-Hard version.
        self.assertIn("| m | 1 | 0 | 1 |", lines)

    def test_a_run_without_the_model_alone(self):
        records = [record("gsm8k", "m", 0, "solve", False, formal="a = 1"),
                   record("gsm8k", "m", 0, "retry", True, formal="a = 2", first_formal="a = 1",
                          first_ada={"status": "no value"}),
                   record("gsm8k", "m", 1, "solve", True, formal="b = 1"),
                   record("gsm8k", "m", 1, "retry", False, formal="b = 2"),
                   record("gsmhard", "m", 0, "solve", True), record("gsmhard", "m", 0, "retry", True)]
        text = report(records)
        # Compared with the run's first condition, and no table of the same problems with larger numbers.
        self.assertIn("Against solve: right only with the condition / right only under solve", text)
        self.assertIn("| m | 1 / 1, p = 1 |", text)
        self.assertNotIn("larger numbers", text)
        # Problem 0 was asked again after the reply solve gave; problem 1's first reply differs from solve's.
        self.assertEqual(retried(grouped(records)[("gsm8k", "m")], [0, 1]), (1, 1, 1, 1, 0, 1))
        self.assertIn("| m | 1 | 1 | 1 | 1 | 0 / 1 |", text)

    def test_adas_answers_from_another_run(self):
        records = [record("gsm8k", "m", 0, "none", True), record("gsm8k", "m", 0, "solve", False)]
        others = [record("gsm8k", "m", 0, "retry", True), record("gsm8k", "m", 0, "plan", False)]
        self.assertEqual([(r["condition"], r["correct"]) for r in replaced(records, others, "retry")],
                         [("none", True), ("solve", True)])
        self.assertEqual([r["condition"] for r in replaced(records, [], "retry")], ["none"])

    def test_by_subject_and_level(self):
        records = [record("math500", "m", "a", "none", True, subject="Algebra", level=1),
                   record("math500", "m", "a", "solve", False, subject="Algebra", level=1),
                   record("math500", "m", "b", "none", False, subject="Prealgebra", level=1),
                   record("math500", "m", "b", "solve", True, subject="Prealgebra", level=1),
                   # A problem without every condition is left out.
                   record("math500", "m", "c", "none", True, subject="Algebra", level=2)]
        text = report(records)
        self.assertIn("| Algebra | 1 | 1 | 0 |", text)
        self.assertIn("| Prealgebra | 1 | 0 | 1 |", text)
        self.assertIn("| Level 1 | 2 | 1 | 1 |", text)
        self.assertNotIn("Level 2", text)
        # GSM8K's problems have neither, so there is no such table.
        self.assertNotIn("by subject", report(self.records))

    def test_how_the_programs_ended(self):
        records = [record("gsm8k", "m", 0, "none", True), record("gsm8k", "m", 0, "program", True,
                                                                  program={"status": "ran"}),
                   record("gsm8k", "m", 1, "none", True), record("gsm8k", "m", 1, "program", False,
                                                                  program={"status": "refused"})]
        text = report(records)
        self.assertIn("| m | 2 | 2 | 1 |", text)
        self.assertIn("- m: ran 1, refused 1", text)

    def test_thin_input(self):
        self.assertEqual(report([]), "")
        self.assertEqual(paired(grouped([])), [])


if __name__ == "__main__":
    unittest.main()
