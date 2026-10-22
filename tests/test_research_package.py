import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import unittest
from importlib import metadata
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml
from research_helpers import assay, assay_study, study_folder, weights

from ada import __version__
from ada.cli import main
from ada.research import StudyError, build, load, package
from ada.research.example import write_example
from ada.research.package import orcid_id, preview, review, share, split_name

EXTRA = """
from ada import research


@research.table("passage-counts", title="Wells per passage")
def passage_counts(data, study):
    return data.groupby("Passage").size().rename("Wells").reset_index()
"""


def with_operator(frame: pd.DataFrame) -> pd.DataFrame:
    """The assay with who read each well: the column these tests withhold."""
    frame = frame.copy()
    frame["Operator"] = ["Okafor" if index % 3 else "Lindqvist" for index in range(len(frame))]
    return frame


def operator_study(**sharing) -> dict:
    study = assay_study(
        processing=[
            {"exclude": {"where": {"column": "Viability (%)", "below": 0}}, "reason": "Instrument error."}
        ]
    )
    study["variables"]["Operator"] = {"role": "other", "type": "categorical"}
    study["sharing"] = {"data": "processed", "withhold": ["Operator"], "row_level": True, **sharing}
    return study


def handler_weights() -> pd.DataFrame:
    frame = weights()
    frame["Handler"] = ["Okafor" if index % 2 else "Lindqvist" for index in range(len(frame))]
    frame.loc[3, "Weight (g)"] = None
    return frame


def weights_study(**sharing) -> dict:
    return {
        "study": {"title": "Mouse weights", "authors": [{"name": "A. Researcher"}]},
        "data": {"path": "weights.csv", "observation": "one mouse on one day"},
        "design": {"type": "repeated", "subject": "Mouse", "time": "Day"},
        "variables": {
            "Mouse": {"role": "identifier", "type": "text"},
            "Diet": {"role": "group", "type": "categorical", "levels": ["Chow", "High fat"]},
            "Day": {"role": "time", "type": "numeric"},
            "Weight (g)": {"role": "outcome", "type": "numeric", "unit": "g"},
            "Handler": {"role": "other", "type": "categorical"},
        },
        "processing": [{"exclude": {"ids": ["M03"]}, "reason": "Given the wrong diet."}],
        "analyses": [
            {"id": "gain", "outcome": "Weight (g)", "change": "Day", "by": "Diet", "planned": True},
            {"id": "by-handler", "outcome": "Weight (g)", "compare": "Handler", "planned": False},
        ],
        "sharing": {"data": "demo", "withhold": ["Handler"], "row_level": True, **sharing},
    }


def csv_headers(folder: Path) -> dict[str, list[str]]:
    return {
        path.relative_to(folder).as_posix(): list(pd.read_csv(path, nrows=0).columns)
        for path in folder.rglob("*.csv")
    }


def quietly(function, *args, **kwargs):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = function(*args, **kwargs)
    return result, output.getvalue()


class ReproduceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        assay().to_csv(root / "assay.csv", index=False)
        (root / "analysis").mkdir()
        (root / "analysis" / "extra.py").write_text(EXTRA, encoding="utf-8")
        study = assay_study(custom=["analysis/extra.py"])
        study["study"]["authors"] = [
            {
                "name": "Ludwig van Beethoven",
                "orcid": "0000-0002-1825-0097",
                "affiliation": "Example Institute",
            },
            {"name": "Ada Lovelace", "orcid": "1234"},
        ]
        study["study"]["keywords"] = ["viability", "assay"]
        study["study"]["related"] = ["https://doi.org/10.1234/abcd.5678"]
        cls.path = root / "study.yml"
        cls.path.write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")
        cls.root = root
        cls.built = build(cls.path)
        cls.reproduce = cls.built.out / "reproduce"
        cls.repository = cls.built.out / "repository"

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_the_checksums_match_hashlib_and_sha256sum_accepts_them(self):
        text = (self.reproduce / "checksums.sha256").read_text()
        paths = []
        for line in text.splitlines():
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            self.assertIsNotNone(match, line)
            digest, path = match.groups()
            self.assertEqual(digest, hashlib.sha256((self.root / path).read_bytes()).hexdigest())
            paths.append(path)
        self.assertEqual(paths, ["assay.csv", "study.yml", "analysis/extra.py"])
        if shutil.which("sha256sum"):
            checked = subprocess.run(
                ["sha256sum", "-c", str(self.reproduce / "checksums.sha256")],
                cwd=self.root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)

    def test_study_yml_and_the_custom_scripts_are_copied_with_their_paths(self):
        self.assertEqual((self.reproduce / "study.yml").read_bytes(), self.path.read_bytes())
        self.assertEqual(
            (self.reproduce / "analysis" / "extra.py").read_bytes(),
            (self.root / "analysis" / "extra.py").read_bytes(),
        )

    def test_requirements_pin_the_versions_in_use(self):
        lines = (self.reproduce / "requirements.txt").read_text().splitlines()
        self.assertIn(f"automated-data-analyst=={__version__}", lines)
        self.assertIn(f"# Python {platform.python_version()}", lines)
        for name in ("pandas", "numpy", "scipy", "statsmodels", "matplotlib", "python-docx", "PyYAML"):
            self.assertIn(f"{name}=={metadata.version(name)}", lines)

    def test_the_instructions_and_the_script_give_the_steps(self):
        text = (self.reproduce / "INSTRUCTIONS.md").read_text()
        for step in (
            "## 1. Get the data",
            "## 2. Install",
            "## 3. Build",
            "## 4. Verify",
            "## 5. What to expect",
        ):
            self.assertIn(step, text)
        self.assertIn("pip install -r paper-assets/reproduce/requirements.txt", text)
        self.assertIn(hashlib.sha256((self.root / "assay.csv").read_bytes()).hexdigest(), text)
        self.assertIn("Table 1 (", text)
        script = self.reproduce / "reproduce.sh"
        self.assertTrue(os.access(script, os.X_OK))
        body = script.read_text()
        self.assertTrue(body.startswith("#!/bin/sh\n"))
        self.assertLess(body.index("sha256sum -c"), body.index("ada build"))
        self.assertLess(body.index("ada build"), body.index("ada verify"))
        if shutil.which("sh"):
            self.assertEqual(subprocess.run(["sh", "-n", str(script)]).returncode, 0)

    def test_the_package_files_are_recorded_as_assets(self):
        record = json.loads((self.built.out / "provenance.json").read_text())
        for ident in ("reproduction-package", "repository-metadata"):
            self.assertEqual(record["assets"][ident]["section"], "package")
        self.assertIn("reproduce/checksums.sha256", record["assets"]["reproduction-package"]["files"])
        self.assertIn("repository/.zenodo.json", record["assets"]["repository-metadata"]["files"])
        self.assertIn("Sharing and reproduction", (self.built.out / "README.md").read_text())

    def test_citation_cff_parses_with_the_required_keys(self):
        cff = yaml.safe_load((self.repository / "CITATION.cff").read_text())
        for key in ("cff-version", "message", "title", "authors"):
            self.assertIn(key, cff)
        self.assertEqual(cff["cff-version"], "1.2.0")
        self.assertEqual(cff["license"], "CC-BY-4.0")
        self.assertEqual(cff["keywords"], ["viability", "assay"])
        first, second = cff["authors"]
        self.assertEqual(
            first,
            {
                "given-names": "Ludwig",
                "name-particle": "van",
                "family-names": "Beethoven",
                "orcid": "https://orcid.org/0000-0002-1825-0097",
                "affiliation": "Example Institute",
            },
        )
        self.assertEqual(second, {"given-names": "Ada", "family-names": "Lovelace"})
        self.assertIn("not a valid ORCID", (self.repository / "CITATION.cff").read_text())
        self.assertEqual(cff["references"][0]["title"], "ADA: Automated Data Analyst")
        self.assertEqual(cff["references"][0]["version"], __version__)

    def test_zenodo_metadata_lists_creators_and_related_work(self):
        record = json.loads((self.repository / ".zenodo.json").read_text())
        self.assertEqual(record["upload_type"], "dataset")
        self.assertEqual(record["license"], "cc-by-4.0")
        self.assertEqual(
            record["creators"][0],
            {
                "name": "van Beethoven, Ludwig",
                "affiliation": "Example Institute",
                "orcid": "0000-0002-1825-0097",
            },
        )
        self.assertEqual(
            record["related_identifiers"], [{"identifier": "10.1234/abcd.5678", "relation": "isSupplementTo"}]
        )

    def test_a_rebuild_writes_the_same_package(self):
        before = {
            path: path.read_bytes()
            for path in [*self.reproduce.rglob("*"), *self.repository.rglob("*")]
            if path.is_file()
        }
        build(self.path)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content, path)


class RepositoryTests(unittest.TestCase):
    def test_no_licence_is_invented_when_it_is_blank(self):
        study = assay_study()
        study["study"]["license"] = ""
        with study_folder(assay(), study) as path:
            out = build(path).out / "repository"
            cff = yaml.safe_load((out / "CITATION.cff").read_text())
            zenodo = json.loads((out / ".zenodo.json").read_text())
            readme = (out / "README.md").read_text()
            availability = (out / "data-availability.md").read_text()
        self.assertNotIn("license", cff)
        self.assertNotIn("license", zenodo)
        self.assertIn("TODO: choose a licence for the data", readme)
        self.assertIn("TODO: choose a licence for the code", readme)
        self.assertIn("[licence]", availability)
        for invented in ("CC-BY", "CC0", "MIT licence", "Apache"):
            self.assertNotIn(invented, availability)

    def test_names_are_split_where_they_can_be(self):
        self.assertEqual(
            split_name("A. Researcher"), {"given": "A.", "particle": "", "family": "Researcher", "suffix": ""}
        )
        self.assertEqual(
            split_name("Mary Jane van der Berg"),
            {"given": "Mary Jane", "particle": "van der", "family": "Berg", "suffix": ""},
        )
        self.assertEqual(
            split_name("de la Cruz, María"),
            {"given": "María", "particle": "de la", "family": "Cruz", "suffix": ""},
        )
        self.assertEqual(
            split_name("John Smith Jr."),
            {"given": "John", "particle": "", "family": "Smith", "suffix": "Jr."},
        )
        self.assertEqual(split_name("Consortium"), {"name": "Consortium"})

    def test_orcids_are_checked(self):
        self.assertEqual(orcid_id("https://orcid.org/0000-0002-1825-0097"), "0000-0002-1825-0097")
        self.assertEqual(orcid_id("0000-0001-5109-3700"), "0000-0001-5109-3700")
        self.assertEqual(orcid_id("0000-0002-1694-233x"), "0000-0002-1694-233X")
        self.assertIsNone(orcid_id("0000-0000-0000-0000"))
        self.assertIsNone(orcid_id("not an orcid"))

    def test_the_availability_statements_follow_the_sharing_setting(self):
        for mode, expected in (
            ("processed", "openly available"),
            ("raw", "original data file and the processed data"),
            ("none", "not openly available"),
        ):
            with (
                self.subTest(mode=mode),
                study_folder(with_operator(assay()), operator_study(data=mode)) as path,
            ):
                out = build(path).out / "repository"
                data = (out / "data-availability.md").read_text()
                code = (out / "code-availability.md").read_text()
                self.assertIn(expected, data)
                self.assertIn("[repository DOI]", data)
                self.assertIn("ada build", code)
                if mode != "none":
                    self.assertIn("leave out the column Operator", data)
                instructions = (out.parent / "reproduce" / "INSTRUCTIONS.md").read_text()
                if mode == "none":
                    self.assertIn("[name and email of the corresponding author", instructions)


class SharingTests(unittest.TestCase):
    def test_preview_flags_a_withheld_column_and_share_removes_it(self):
        with study_folder(with_operator(assay()), operator_study()) as path:
            built = build(path)
            self.assertIn("Operator", pd.read_csv(built.out / "processing" / "excluded-rows.csv").columns)
            text = preview(path)
            self.assertIn("! share*", text)
            self.assertIn("processing/excluded-rows.csv: has the withheld column Operator", text)
            shared = share(path)
            self.assertTrue(shared.ok, shared.problems)
            headers = csv_headers(shared.folder)
            self.assertIn("processing/excluded-rows.csv", headers)
            self.assertIn("processing/processed-data.csv", headers)
            for name, columns in headers.items():
                self.assertNotIn("Operator", columns, name)
            self.assertIn("Operator", (shared.folder / "SHARING.md").read_text())
            self.assertIn("Withheld columns removed", shared.text())

    def test_sharing_none_keeps_the_data_out_of_share(self):
        with study_folder(with_operator(assay()), operator_study(data="none")) as path:
            build(path)
            checked = review(path)
            left = {decision.path: decision for decision in checked.decisions if decision.action == "leave"}
            self.assertIn("processing/excluded-rows.csv", left)
            self.assertTrue(left["processing/excluded-rows.csv"].flags)
            shared = share(path)
            files = {item.relative_to(shared.folder).as_posix() for item in shared.folder.rglob("*")}
            self.assertNotIn("processing/processed-data.csv", files)
            self.assertNotIn("processing/excluded-rows.csv", files)
            self.assertFalse(any(name.startswith("data/") for name in files))
            self.assertIn("processing/processing-record.md", files)

    def test_raw_sharing_adds_the_original_file_without_withheld_columns(self):
        with study_folder(with_operator(assay()), operator_study(data="raw")) as path:
            build(path)
            shared = share(path)
            copy = pd.read_csv(shared.folder / "data" / "assay.csv")
            self.assertEqual(list(copy.columns), ["Well", "Treatment", "Passage", "Viability (%)"])
            self.assertEqual(len(copy), len(assay()))

    def test_row_level_false_keeps_the_files_listing_rows_back(self):
        with study_folder(with_operator(assay()), operator_study(row_level=False)) as path:
            build(path)
            shared = share(path)
            files = {item.relative_to(shared.folder).as_posix() for item in shared.folder.rglob("*")}
            for name in (
                "processing/processed-data.csv",
                "processing/excluded-rows.csv",
                "audit/data-audit.csv",
            ):
                self.assertNotIn(name, files)
            self.assertIn("audit/data-audit.md", files)
            self.assertIn("tables/descriptives.csv", files)
            self.assertIn("row_level is false", preview(path))

    def test_files_holding_values_of_a_withheld_column_are_left_out(self):
        study = operator_study()
        study["descriptives"] = {"by": "Operator"}
        with study_folder(with_operator(assay()), study) as path:
            build(path)
            text = preview(path)
            self.assertIn("tables/descriptives.csv: contains a value of the withheld column Operator", text)
            shared = share(path)
            files = {item.relative_to(shared.folder).as_posix() for item in shared.folder.rglob("*")}
            self.assertNotIn("tables/descriptives.csv", files)
            self.assertIn("dictionary/data-dictionary.csv", files)
            dictionary = pd.read_csv(shared.folder / "dictionary" / "data-dictionary.csv")
            self.assertEqual(dictionary.set_index("variable").loc["Operator", "observed"], "withheld")
            self.assertNotIn("Lindqvist", (shared.folder / "SHARING.md").read_text())

    def test_share_refuses_and_deletes_a_copy_that_still_holds_something_withheld(self):
        with study_folder(with_operator(assay()), operator_study()) as path:
            build(path)
            with mock.patch.object(package, "share_record", return_value="Lindqvist read these wells.\n"):
                shared = share(path)
            self.assertFalse(shared.ok)
            self.assertFalse(shared.folder.exists())
            self.assertTrue(any("SHARING.md" in problem for problem in shared.problems))
            self.assertIn("Refused", shared.text())

    def test_share_will_not_replace_a_folder_it_did_not_write(self):
        with study_folder(assay(), assay_study()) as path:
            build(path)
            mine = path.parent / "share"
            mine.mkdir()
            (mine / "notes.txt").write_text("mine")
            with self.assertRaises(StudyError):
                share(path)
            self.assertTrue((mine / "notes.txt").is_file())
            self.assertTrue(share(path, out=path.parent / "deposit").ok)
            self.assertTrue(share(path, out=path.parent / "deposit").ok)

    def test_preview_before_a_build_says_to_build(self):
        with study_folder(assay(), assay_study()) as path:
            self.assertIn("Run `ada build` first", preview(path))


class DemoDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        handler_weights().to_csv(root / "weights.csv", index=False)
        cls.path = root / "study.yml"
        cls.path.write_text(yaml.safe_dump(weights_study(), sort_keys=False), encoding="utf-8")
        cls.built = build(cls.path)
        cls.reproduce = cls.built.out / "reproduce"
        cls.demo = pd.read_csv(cls.reproduce / "demo-data.csv")
        cls.processed = pd.read_csv(cls.built.out / "processing" / "processed-data.csv")

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_demo_data_has_the_columns_and_types_and_no_withheld_column(self):
        self.assertEqual(list(self.demo.columns), [c for c in self.processed.columns if c != "source_row"])
        self.assertNotIn("Handler", self.demo.columns)
        for name in ("Day", "Weight (g)"):
            self.assertTrue(pd.api.types.is_numeric_dtype(self.demo[name]), name)
        self.assertTrue(pd.api.types.is_integer_dtype(self.demo["Day"]))
        self.assertLessEqual(set(self.demo["Diet"]), {"Chow", "High fat"})
        self.assertEqual(self.demo["Diet"].value_counts().sum(), self.processed["Diet"].value_counts().sum())
        self.assertEqual(self.demo["Weight (g)"].isna().sum(), self.processed["Weight (g)"].isna().sum())

    def test_demo_values_follow_the_marginal_distribution_but_not_the_rows(self):
        demo, real = self.demo["Weight (g)"].dropna(), self.processed["Weight (g)"].dropna()
        self.assertGreaterEqual(demo.min(), real.min())
        self.assertLessEqual(demo.max(), real.max())
        self.assertLess(abs(demo.mean() - real.mean()), real.std())
        self.assertFalse(self.demo["Weight (g)"].equals(self.processed["Weight (g)"]))

    def test_the_design_is_kept_under_new_identifiers(self):
        self.assertTrue(self.demo["Mouse"].str.startswith("demo-M").all())
        self.assertFalse(set(self.demo["Mouse"]) & set(self.processed["Mouse"]))
        demo_days = sorted(tuple(group) for _, group in self.demo.groupby("Mouse")["Day"])
        real_days = sorted(tuple(group) for _, group in self.processed.groupby("Mouse")["Day"])
        self.assertEqual(demo_days, real_days)
        self.assertTrue((self.demo.groupby("Mouse")["Diet"].nunique() == 1).all())

    def test_demo_data_is_labelled_synthetic(self):
        self.assertIn("SYNTHETIC", (self.reproduce / "demo-data.README.md").read_text())
        self.assertIn("SYNTHETIC", (self.reproduce / "INSTRUCTIONS.md").read_text())
        self.assertIn("SYNTHETIC", (self.reproduce / "demo-study.yml").read_text())

    def test_the_demo_study_builds_on_the_demo_data(self):
        demo_study = load(self.reproduce / "demo-study.yml")
        self.assertIn("synthetic", demo_study.title)
        self.assertEqual([analysis.id for analysis in demo_study.analyses], ["gain"])
        self.assertNotIn("Handler", demo_study.variables)
        self.assertEqual(demo_study.steps, ())
        built = build(demo_study)
        self.assertEqual(built.out, self.reproduce / "demo-assets")
        self.assertTrue((built.out / "tables" / "descriptives.csv").is_file())
        self.assertEqual(len(built.data), len(self.demo))

    def test_demo_data_is_the_same_at_every_build(self):
        with tempfile.TemporaryDirectory() as folder:
            build(self.path, out=Path(folder))
            self.assertEqual(
                (Path(folder) / "reproduce" / "demo-data.csv").read_bytes(),
                (self.reproduce / "demo-data.csv").read_bytes(),
            )

    def test_a_paired_condition_stays_paired(self):
        frame = pd.DataFrame(
            {
                "Participant": [f"P{number:02d}" for number in range(1, 11) for _ in range(2)],
                "Session": ["Before", "After"] * 10,
                "Score": [50.0 + number for number in range(20)],
            }
        )
        study = {
            "study": {"title": "Training"},
            "data": {"path": "scores.csv", "observation": "one participant in one session"},
            "design": {"type": "paired", "subject": "Participant"},
            "variables": {
                "Participant": {"role": "identifier", "type": "text"},
                "Session": {"role": "group", "type": "categorical", "levels": ["Before", "After"]},
                "Score": {"role": "outcome", "type": "numeric"},
            },
            "sharing": {"data": "demo"},
        }
        with study_folder(frame, study, data_name="scores.csv") as path:
            demo = pd.read_csv(build(path).out / "reproduce" / "demo-data.csv")
        sessions = demo.groupby("Participant")["Session"].apply(lambda values: sorted(values))
        self.assertTrue(all(value == ["After", "Before"] for value in sessions))


class ExampleTests(unittest.TestCase):
    def test_ada_demo_writes_a_study_that_builds(self):
        with tempfile.TemporaryDirectory() as folder:
            files = write_example(Path(folder) / "demo")
            study = load(files[0])
            self.assertEqual(study.design, "repeated")
            self.assertEqual([analysis.planned for analysis in study.analyses], [True, False])
            self.assertTrue(all(step.reason for step in study.steps))
            self.assertTrue(study.authors and study.variable("Body weight (g)").unit == "g")
            self.assertEqual(study.sharing.withhold, ("Handler",))
            built = build(study)
            removed = [record.rows_removed for record in built.processed.records if record.kind == "exclude"]
            self.assertEqual(removed, [9, 1])
            ids = {asset.id for asset in built.assets}
            for ident in (
                "descriptives",
                "weight-gain-by-diet",
                "reproduction-package",
                "repository-metadata",
            ):
                self.assertIn(ident, ids)
            self.assertEqual(study.license, "")

    def test_ada_demo_will_not_write_into_a_folder_in_use(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "notes.txt").write_text("mine")
            with self.assertRaises(FileExistsError):
                write_example(folder)


class CommandLineTests(unittest.TestCase):
    def test_demo_build_preview_and_share_from_the_command_line(self):
        with tempfile.TemporaryDirectory() as folder:
            demo = Path(folder) / "demo"
            code, output = quietly(main, ["demo", str(demo)])
            self.assertEqual(code, 0)
            self.assertIn("ada build", output)
            study = str(demo / "study.yml")
            code, output = quietly(main, ["build", "--study", study])
            self.assertEqual(code, 0, output)
            code, output = quietly(main, ["preview", "--study", study])
            self.assertEqual(code, 0)
            self.assertIn("has the withheld column Handler", output)
            code, output = quietly(main, ["share", "--study", study, "--out", str(Path(folder) / "deposit")])
            self.assertEqual(code, 0, output)
            for name, columns in csv_headers(Path(folder) / "deposit").items():
                self.assertNotIn("Handler", columns, name)
            code, _ = quietly(main, ["research", "share", "--study", study])
            self.assertEqual(code, 0)
            self.assertTrue((demo / "share" / "SHARING.md").is_file())


if __name__ == "__main__":
    unittest.main()
