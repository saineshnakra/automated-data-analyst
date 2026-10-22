import json
import tempfile
import unittest
from pathlib import Path

import yaml
from research_helpers import assay, assay_study, study_folder

from ada.research import build, init, load


class BuildTests(unittest.TestCase):
    def test_a_build_writes_the_records_and_tables(self):
        with study_folder(assay(), assay_study()) as path:
            built = build(path)
            out = built.out
            for name in (
                "audit/data-audit.md",
                "dictionary/data-dictionary.csv",
                "dictionary/data-dictionary.docx",
                "processing/processing-record.md",
                "tables/descriptives.csv",
                "tables/descriptives.tex",
                "tables/descriptives.docx",
                "provenance.json",
                "CHANGES.md",
                "README.md",
            ):
                self.assertTrue((out / name).is_file(), name)
            record = json.loads((out / "provenance.json").read_text())
            self.assertEqual(record["inputs"][0]["path"], "assay.csv")
            self.assertIn("descriptives", record["assets"])
            self.assertEqual(record["assets"]["descriptives"]["number"], "Table 1")
            self.assertIn("pandas", record["packages"])

    def test_a_second_build_reports_what_a_new_exclusion_changed(self):
        with study_folder(assay(), assay_study()) as path:
            build(path)
            study = yaml.safe_load(path.read_text())
            study["processing"] = [
                {"exclude": {"where": {"column": "Viability (%)", "below": 0}}, "reason": "Instrument error."}
            ]
            path.write_text(yaml.safe_dump(study, sort_keys=False))
            built = build(path)
            changed = {change.asset: change for change in built.changes}
            self.assertIn("descriptives", changed)
            self.assertTrue(any("processing" in reason for reason in changed["descriptives"].reasons))
            text = (built.out / "CHANGES.md").read_text()
            self.assertIn("**descriptives** changed", text)
            self.assertIn("First build", text)

    def test_an_unchanged_rebuild_reports_nothing(self):
        with study_folder(assay(), assay_study()) as path:
            build(path)
            built = build(path)
            self.assertEqual(built.changes, [])


class InitTests(unittest.TestCase):
    def test_init_without_questions_writes_a_study_that_builds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assay().to_csv(root / "assay.csv", index=False)
            path = init(root / "assay.csv", study_path=root / "study.yml", interactive=False)
            study = load(path)
            self.assertEqual(study.missing_codes, (-99,))
            self.assertEqual(study.with_role("outcome"), ["Viability (%)"])
            self.assertEqual(study.with_role("group"), ["Treatment"])
            self.assertEqual(study.variable("Viability (%)").unit, "%")
            self.assertTrue(study.analyses)
            self.assertFalse(any(analysis.planned for analysis in study.analyses))
            build(path)

    def test_answers_replace_the_guesses(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assay().to_csv(root / "assay.csv", index=False)
            answers = {
                "Study title": "My assay",
                "one row": "one well",
                "Unit of Viability": "percent",
                "Which were planned": "1",
                "Keep the others": "n",
                "made public": "none",
                "Authors": "Ada Lovelace",
                "Licence": "CC0-1.0",
                "Journal": "plos",
            }

            def ask(question: str) -> str:
                return next((answer for key, answer in answers.items() if key in question), "")

            path = init(root / "assay.csv", study_path=root / "study.yml", ask=ask, say=lambda _: None)
            study = load(path)
        self.assertEqual(study.title, "My assay")
        self.assertEqual(study.variable("Viability (%)").unit, "percent")
        self.assertEqual(study.sharing.data, "none")
        self.assertEqual(study.output.journal, "plos")
        self.assertEqual([analysis.planned for analysis in study.analyses], [True])

    def test_init_writes_a_relative_path_to_data_outside_the_study_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "data").mkdir()
            (root / "analysis").mkdir()
            assay().to_csv(root / "data" / "assay.csv", index=False)
            path = init(
                root / "data" / "assay.csv", study_path=root / "analysis" / "study.yml", interactive=False
            )
            self.assertEqual(load(path).data_path, "../data/assay.csv")

    def test_init_will_not_overwrite_a_study(self):
        with study_folder(assay(), assay_study()) as path, self.assertRaises(FileExistsError):
            init(path.parent / "assay.csv", study_path=path, interactive=False)


if __name__ == "__main__":
    unittest.main()
