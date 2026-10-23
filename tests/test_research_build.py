import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml
from research_helpers import assay, assay_study, study_folder

from ada.cli import main
from ada.research import StudyError, build, init, load
from ada.research.study import sha256_of


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


BROKEN_SCRIPT = """
from ada import research


@research.table("broken", title="A table that cannot be made")
def broken(data, study):
    raise ValueError("this table cannot be made")
"""


def files(out: Path) -> dict[str, bytes]:
    """Every file under ``out``, by its path from ``out``, with its bytes."""
    return {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*") if path.is_file()}


def edit_study(path: Path, change) -> None:
    study = yaml.safe_load(path.read_text(encoding="utf-8"))
    change(study)
    path.write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")


class FailedBuildTests(unittest.TestCase):
    def assertNoScratchLeft(self, path: Path) -> None:
        self.assertEqual([item.name for item in path.parent.iterdir() if "-building-" in item.name], [])

    def test_a_changed_data_file_leaves_the_previous_build_as_it_was(self):
        with study_folder(assay(), assay_study()) as path:
            data = path.parent / "assay.csv"
            edit_study(path, lambda study: study["data"].update(sha256=sha256_of(data)))
            out = build(path).out
            (out / "notes.txt").write_text("mine", encoding="utf-8")
            before = files(out)
            with data.open("a", encoding="utf-8") as handle:
                handle.write("Z99,Control,5,88.0\n")

            with self.assertRaisesRegex(StudyError, "has changed since study.yml recorded it"):
                build(path)

            self.assertEqual(files(out), before)
            self.assertGreater(len(before), 40)
            self.assertNoScratchLeft(path)

    def test_a_step_that_fails_leaves_the_previous_build_as_it_was(self):
        with study_folder(assay(), assay_study()) as path:
            out = build(path).out
            before = files(out)
            original = path.read_text(encoding="utf-8")

            with self.subTest("a custom table that raises"):
                (path.parent / "broken.py").write_text(BROKEN_SCRIPT, encoding="utf-8")
                edit_study(path, lambda study: study.update(custom=["broken.py"]))
                with self.assertRaisesRegex(ValueError, "cannot be made"):
                    build(path)
                self.assertEqual(files(out), before)
            path.write_text(original, encoding="utf-8")

            with self.subTest("a figure that cannot be saved"), mock.patch(
                "matplotlib.figure.Figure.savefig", side_effect=OSError("No space left on device")
            ):
                with self.assertRaisesRegex(OSError, "No space left"):
                    build(path)
                self.assertEqual(files(out), before)
            self.assertNoScratchLeft(path)

    def test_a_rebuild_replaces_what_the_build_owns_and_nothing_else(self):
        with study_folder(assay(), assay_study()) as path:
            out = build(path).out
            first = files(out)
            (out / "tables" / "stale.csv").write_text("left from before", encoding="utf-8")
            (out / "drafts").mkdir()
            (out / "drafts" / "letter.md").write_text("mine", encoding="utf-8")
            (out / "notes.txt").write_text("mine", encoding="utf-8")
            exclusion = {"exclude": {"where": {"column": "Viability (%)", "below": 0}}, "reason": "Error."}
            edit_study(path, lambda study: study.update(processing=[exclusion]))

            built = build(path)

            self.assertEqual(built.out, out)
            self.assertFalse((out / "tables" / "stale.csv").exists())
            self.assertNotEqual(files(out)["tables/descriptives.csv"], first["tables/descriptives.csv"])
            self.assertEqual((out / "drafts" / "letter.md").read_text(encoding="utf-8"), "mine")
            self.assertEqual((out / "notes.txt").read_text(encoding="utf-8"), "mine")
            changes = (out / "CHANGES.md").read_text(encoding="utf-8")
            self.assertLess(changes.index("## Build 2"), changes.index("## Build 1"))
            record = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
            self.assertIn("tables/descriptives.csv", record["assets"]["descriptives"]["files"])
            written = [file for asset in built.assets for file in asset.files]
            self.assertTrue(all(file.is_file() and file.is_relative_to(out) for file in written))
            self.assertNoScratchLeft(path)


class BuildSummaryTests(unittest.TestCase):
    def run_build(self, path: Path, *options: str) -> tuple[int, str, str]:
        printed, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(printed), contextlib.redirect_stderr(errors):
            code = main(["build", "--study", str(path), *options])
        return code, printed.getvalue(), errors.getvalue()

    def test_the_summary_counts_the_analyses_and_names_each_refused_one(self):
        study = assay_study()
        study["analyses"].append(
            {"id": "by-passage", "outcome": "Viability (%)", "compare": "Passage", "planned": False}
        )
        with study_folder(assay(), study) as path:
            code, printed, _ = self.run_build(path)
            self.assertEqual(code, 0)
            self.assertIn("Analyses: 2 requested, 1 fitted, 1 refused.", printed)
            self.assertIn("  refused  by-passage: `compare:` needs a categorical or binary group", printed)
            self.assertNotIn("note: Analysis by-passage was not fitted", printed)
            self.assertEqual(list(build(path).refused), ["by-passage"])

            code, _, errors = self.run_build(path, "--strict")
            self.assertEqual(code, 1)
            self.assertIn("--strict: 1 of 2 requested analyses was refused (by-passage)", errors)

    def test_strict_passes_when_every_analysis_is_fitted(self):
        with study_folder(assay(), assay_study()) as path:
            code, printed, errors = self.run_build(path, "--strict")
            self.assertEqual(code, 0)
            self.assertIn("Analyses: 1 requested, 1 fitted, 0 refused.", printed)
            self.assertEqual(errors, "")


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

    def test_init_takes_the_hourly_timestamp_not_its_year_as_the_time(self):
        stamps = pd.date_range("2016-12-25", periods=10 * 24, freq="h")
        for stations in (["Aotizhongxin"], ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan"]):
            with self.subTest(stations=len(stations)), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                frame = pd.DataFrame({
                    "year": stamps.year, "month": stamps.month, "day": stamps.day, "hour": stamps.hour,
                    "PM2.5": [float(10 + stamp.hour % 6) for stamp in stamps],
                })
                frame = pd.concat([frame.assign(station=station) for station in stations])
                frame.to_csv(root / "air.csv", index=False)
                study = load(init(root / "air.csv", study_path=root / "study.yml", interactive=False))
                self.assertEqual(study.variable("year-month-day-hour").type, "date")
                self.assertEqual(study.variable("hour").type, "numeric")
                # One station's timestamps each appear once, so they are not time points it repeats.
                self.assertEqual(study.time, None if len(stations) == 1 else "year-month-day-hour")

    def test_init_will_not_overwrite_a_study(self):
        with study_folder(assay(), assay_study()) as path, self.assertRaises(FileExistsError):
            init(path.parent / "assay.csv", study_path=path, interactive=False)


if __name__ == "__main__":
    unittest.main()
