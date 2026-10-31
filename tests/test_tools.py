import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ada import tools
from ada.cli import main

try:
    import lightgbm  # noqa: F401
    import sklearn  # noqa: F401
    HAS_EXTRA = True
except ImportError:
    HAS_EXTRA = False


def sales(rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"city": rng.choice(["Oslo", "Lima"], rows),
                         "price": rng.normal(100, 10, rows).round(2)})


class DefinitionTests(unittest.TestCase):
    """The tools are described the way function-calling APIs take them."""

    def test_openai_style(self):
        found = tools.definitions()
        self.assertEqual([item["function"]["name"] for item in found], list(tools.NAMES))
        for item in found:
            self.assertEqual(item["type"], "function")
            schema = item["function"]["parameters"]
            self.assertEqual(schema["type"], "object")
            self.assertTrue(set(schema["required"]) <= set(schema["properties"]))
            self.assertTrue(item["function"]["description"])

    def test_anthropic_style(self):
        found = tools.definitions("anthropic")
        self.assertEqual([item["name"] for item in found], list(tools.NAMES))
        self.assertTrue(all("input_schema" in item for item in found))

    def test_each_call_gives_a_copy_and_unknown_styles_are_refused(self):
        tools.definitions()[0]["function"]["name"] = "changed"
        self.assertEqual(tools.definitions()[0]["function"]["name"], "ada_ask")
        with self.assertRaises(ValueError):
            tools.definitions("other")

    def test_the_command_prints_them(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["tools", "--style", "anthropic"]), 0)
        self.assertEqual([item["name"] for item in json.loads(out.getvalue())], list(tools.NAMES))


class RunTests(unittest.TestCase):
    """A model's tool call runs ADA and gets JSON back, or a reason it could not."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.frame = sales()
        self.frame.to_csv(self.root / "sales.csv", index=False)

    def tearDown(self):
        self.directory.cleanup()

    def test_ask_computes_the_number_with_its_calculation(self):
        result = tools.run("ada_ask", {"path": "sales.csv", "question": "average price"}, root=self.root)
        self.assertEqual(result["status"], "answered")
        self.assertAlmostEqual(result["result"]["value"], self.frame["price"].mean(), places=3)
        json.dumps(result)

    def test_a_claim_sent_as_json_text_is_checked(self):
        mean = self.frame["price"].mean()
        true = tools.run("ada_check_claim", json.dumps({"path": "sales.csv",
                                                         "claim": f"The average price was {mean:.2f}."}),
                         root=self.root)
        false = tools.run("ada_check_claim", {"path": "sales.csv", "claim": "The average price was 150."},
                          root=self.root)
        self.assertEqual(true["status"], "verified")
        self.assertEqual((false["status"], false["reason"]), ("unverified", "failed_check"))

    def test_a_question_ada_cannot_answer_is_refused_with_the_reason(self):
        result = tools.run("ada_ask", {"path": "sales.csv", "question": "what is the meaning of life"},
                           root=self.root)
        self.assertEqual(result["status"], "refused")
        self.assertTrue(result["message"])

    def test_scan_describes_the_folder(self):
        sales(50, seed=1).to_csv(self.root / "more.csv", index=False)
        result = tools.run("ada_scan", {"path": "."}, root=self.root)
        self.assertEqual(len(result["files"]), 2)

    def test_bad_calls_come_back_as_errors_to_read(self):
        outside = tools.run("ada_ask", {"path": "/etc/hosts", "question": "count rows"}, root=self.root)
        self.assertIn("outside the folder", outside["message"])
        def message(name, arguments):
            return tools.run(name, arguments, root=self.root)["message"]

        self.assertIn("outside the folder", message("ada_scan", {"path": "../"}))
        self.assertIn("does not exist", message("ada_scan", {"path": "none.csv"}))
        self.assertIn("question is required", message("ada_ask", {"path": "sales.csv"}))
        self.assertIn("no tool called", tools.run("ada_sum", {})["message"])
        self.assertEqual(tools.run("ada_ask", "not json")["status"], "error")

    @unittest.skipUnless(HAS_EXTRA, "needs the model extra")
    def test_model_writes_its_predictions_inside_the_folder(self):
        rng = np.random.default_rng(2)
        rows = 400
        x = rng.normal(0, 1, rows)
        frame = pd.DataFrame({"id": np.arange(rows), "x": x.round(3), "y": 2 * x + rng.normal(0, 0.5, rows)})
        frame[:300].to_csv(self.root / "train.csv", index=False)
        frame[300:].drop(columns="y").to_csv(self.root / "test.csv", index=False)
        result = tools.run("ada_model", {"train": "train.csv", "test": "test.csv", "target": "y",
                                         "metric": "rmse", "out": "predictions.csv"}, root=self.root)
        self.assertLess(result["score"], 1.0)
        self.assertEqual(len(pd.read_csv(self.root / "predictions.csv")), 100)
        refused = tools.run("ada_model", {"train": "sales.csv", "target": "price"}, root=self.root)
        self.assertEqual(refused["status"], "refused")


if __name__ == "__main__":
    unittest.main()
