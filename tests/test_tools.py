import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from ada import tools
from ada.cli import main
from ada.sources import Source, without_file_name


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

    def test_mcp_style(self):
        found = tools.definitions("mcp")
        self.assertEqual([item["name"] for item in found], list(tools.NAMES))
        self.assertTrue(all("inputSchema" in item and item["title"] for item in found))
        hints = {item["name"]: item["annotations"]["readOnlyHint"] for item in found}
        self.assertEqual(hints, {"ada_ask": True, "ada_check_claim": True, "ada_scan": True})

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

    def test_a_missing_or_wrong_path_names_the_files_to_pick_from(self):
        (self.root / "notes").mkdir()
        sales(20, seed=3).to_csv(self.root / "notes" / "old.csv", index=False)
        (self.root / "readme.md").write_text("not data")
        missing = tools.run("ada_check_claim", {"claim": "The average price was 100."}, root=self.root)
        self.assertEqual(missing["message"], "path is required: the path of a data file or folder. "
                                             "The data files are sales.csv and notes/old.csv.")
        wrong = tools.run("ada_ask", {"path": "sale.csv", "question": "average price"}, root=self.root)
        self.assertTrue(wrong["message"].startswith("sale.csv does not exist. The data files are sales.csv"))
        # Without a root the tools may read anywhere, so there is no folder to list.
        self.assertEqual(tools.run("ada_scan", {"path": str(self.root / "none.csv")})["message"],
                         f"{self.root / 'none.csv'} does not exist.")

    def test_thin_and_crowded_folders_in_the_message(self):
        with tempfile.TemporaryDirectory() as empty:
            message = tools.run("ada_scan", {"path": "x.csv"}, root=empty)["message"]
            self.assertEqual(message, "x.csv does not exist. There are no data files in the folder the tools "
                                      "may read.")
        for index in range(tools.LISTED_FILES + 5):
            sales(5).to_csv(self.root / f"part{index:02d}.csv", index=False)
        message = tools.run("ada_scan", {}, root=self.root)["message"]
        self.assertTrue(message.endswith(", among others."))
        self.assertEqual(message.count(".csv"), tools.LISTED_FILES)


class FileNameTests(unittest.TestCase):
    """A question or claim that names its own file is read without the file's name."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        rng = np.random.default_rng(4)
        self.frame = pd.DataFrame({"species": rng.choice(["Adelie", "Gentoo"], 120),
                                   "mass": rng.normal(4000, 300, 120).round(1)})
        (self.root / "data").mkdir()
        self.frame.to_csv(self.root / "data" / "penguins.csv", index=False)

    def tearDown(self):
        self.directory.cleanup()

    def test_the_file_name_is_dropped_with_its_folder_and_preposition(self):
        source = Source("penguins.csv", "data/penguins.csv")
        self.assertEqual(without_file_name("In data/penguins.csv, the average mass is 4000.", source),
                         "the average mass is 4000.")
        self.assertEqual(without_file_name("The average mass in the file 'penguins.csv' is 4000.", source),
                         "The average mass is 4000.")
        self.assertEqual(without_file_name("how many rows in ./data/Penguins.CSV?", source), "how many rows?")

    def test_other_names_and_tables_without_a_file_are_left_alone(self):
        source = Source("penguins.csv", "data/penguins.csv")
        for text in ("average mass in mypenguins.csv", "average mass in penguins.csvx", "average mass"):
            self.assertEqual(without_file_name(text, source), text)
        named = "average mass in penguins.csv"
        self.assertEqual(without_file_name(named, None), named)
        self.assertEqual(without_file_name("average mass in frame", Source("DataFrame")),
                         "average mass in frame")

    def test_a_claim_that_names_its_file_is_checked(self):
        mean = self.frame["mass"].mean()
        def check(claim):
            return tools.run("ada_check_claim", {"path": "data/penguins.csv", "claim": claim}, root=self.root)

        true = check(f"In data/penguins.csv the average mass is {mean:.1f}.")
        false = check("The average mass in penguins.csv is 5000.")
        self.assertEqual(true["status"], "verified")
        self.assertEqual(true["claim"], f"In data/penguins.csv the average mass is {mean:.1f}.")
        self.assertEqual((false["status"], false["reason"]), ("unverified", "failed_check"))

    def test_a_question_that_names_its_file_or_one_row_is_answered(self):
        gentoo = self.frame.loc[self.frame["species"] == "Gentoo", "mass"].mean()
        for question in ("average mass in data/penguins.csv for Gentoo", "average mass of a Gentoo penguin",
                         "average mass of Gentoo penguins"):
            result = tools.run("ada_ask", {"path": "data/penguins.csv", "question": question}, root=self.root)
            self.assertEqual(result["status"], "answered", question)
            self.assertAlmostEqual(result["result"]["value"], gentoo, places=3)
        counted = tools.run("ada_ask", {"path": "data/penguins.csv",
                                        "question": "how many penguin per species"}, root=self.root)
        self.assertIn("one of the penguins", counted["answer"])


class SmallModelSlipTests(unittest.TestCase):
    """The slips small models make are taken in stride where only one reading is possible."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        self.frame = pd.DataFrame({"species": ["Adelie", "Gentoo", "Chinstrap"] * 20,
                                   "mass": np.arange(60, dtype=float) * 10 + 3000})
        self.frame.to_csv(self.root / "data" / "penguins.csv", index=False)

    def tearDown(self):
        self.directory.cleanup()

    def run_tool(self, name, arguments):
        return tools.run(name, arguments, root=self.root, brief=True)

    def test_a_file_named_without_its_folder_is_found(self):
        result = self.run_tool("ada_ask", {"path": "penguins.csv", "question": "average mass"})
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["file"], str(self.root / "data" / "penguins.csv"))

    def test_a_path_left_out_is_taken_from_the_question_or_the_only_file(self):
        named = self.run_tool("ada_ask", {"question": "in data/penguins.csv, what is the average mass?"})
        self.assertEqual(named["status"], "answered")
        mean = self.frame["mass"].mean()
        only = self.run_tool("ada_check_claim", {"claim": f"The average mass is {mean}."})
        self.assertEqual(only["verdict"], "true")

    def test_a_path_from_the_top_of_the_disk_is_the_file_of_that_name_under_the_root(self):
        result = self.run_tool("ada_ask", {"path": "/penguins.csv", "question": "average mass"})
        self.assertEqual(result["status"], "answered")
        outside = self.run_tool("ada_ask", {"path": "/etc/hosts", "question": "average mass"})
        self.assertIn("outside the folder", outside["message"])

    def test_a_brief_answer_names_every_group(self):
        result = self.run_tool("ada_ask", {"path": "data/penguins.csv",
                                           "question": "average mass by species"})
        self.assertTrue(result["answer"].endswith("Each: Chinstrap 3,305; Gentoo 3,295; Adelie 3,285."),
                        result["answer"])
        # A sentence that names them all already is left as it is.
        tied = self.run_tool("ada_ask", {"path": "data/penguins.csv", "question": "how many rows by species"})
        self.assertNotIn("Each:", tied["answer"])

    def test_another_name_for_the_path_and_a_path_without_its_extension(self):
        aliased = self.run_tool("ada_ask", {"file": "data/penguins.csv", "question": "average mass"})
        self.assertEqual(aliased["status"], "answered")
        bare = self.run_tool("ada_ask", {"path": "/data/penguins", "question": "average mass"})
        self.assertEqual(bare["status"], "answered")

    def test_arguments_wrapped_as_the_call_are_unwrapped(self):
        wrapped = {"type": "function", "function": "ada_ask",
                   "parameters": {"path": "data/penguins.csv", "question": "average mass"}}
        self.assertEqual(self.run_tool("ada_ask", wrapped)["status"], "answered")

    def test_two_files_of_one_name_are_not_guessed_between(self):
        (self.root / "old").mkdir()
        self.frame.to_csv(self.root / "old" / "penguins.csv", index=False)
        for arguments in ({"path": "penguins.csv", "question": "average mass"}, {"question": "average mass"}):
            result = self.run_tool("ada_ask", arguments)
            self.assertEqual(result["status"], "error")
            self.assertIn("data/penguins.csv and old/penguins.csv", result["message"])

    def test_a_statement_sent_as_a_question_is_checked_as_a_claim(self):
        result = self.run_tool("ada_ask", {"path": "data/penguins.csv",
                                           "question": "The average mass is 5000."})
        self.assertEqual((result["verdict"], result["reason"]), ("false", "failed_check"))
        self.assertIn("checked it as a claim", result["note"])
        # A question ADA cannot answer stays refused.
        refused = self.run_tool("ada_ask", {"path": "data/penguins.csv",
                                            "question": "what is the meaning of life"})
        self.assertEqual(refused["status"], "refused")

    def test_a_question_sent_as_a_claim_is_answered(self):
        result = self.run_tool("ada_check_claim", {"path": "data/penguins.csv",
                                                   "claim": "Which species has the highest average mass?"})
        self.assertEqual(result["status"], "answered")
        self.assertIn("answered it", result["note"])

    def test_a_brief_result_keeps_what_a_model_needs(self):
        claim = self.run_tool("ada_check_claim", {"path": "data/penguins.csv",
                                                  "claim": "Gentoo has the most rows."})
        self.assertEqual(set(claim) - {"assumptions", "warnings"},
                         {"status", "verdict", "claim", "question", "message", "calculation", "file",
                          "rows_used"})
        unsure = self.run_tool("ada_check_claim", {"path": "data/penguins.csv",
                                                   "claim": "Sales were great."})
        self.assertEqual(unsure["verdict"], "cannot tell")
        scanned = self.run_tool("ada_scan", {"path": "data"})
        self.assertEqual(scanned["files"][0]["columns"], ["species", "mass"])
        full = tools.run("ada_ask", {"path": "data/penguins.csv", "question": "average mass"}, root=self.root)
        self.assertIn("evidence", full)

    def test_a_home_folder_path_is_read(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.root)}):
            result = tools.run("ada_ask", {"path": "~/data/penguins.csv", "question": "average mass"},
                               root="~")
        self.assertEqual(result["status"], "answered")


if __name__ == "__main__":
    unittest.main()
