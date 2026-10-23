import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock
from pathlib import Path

import pandas as pd
import yaml

import ada
import ada.api
import ada.nlq


def rainfall(periods: int = 24) -> pd.DataFrame:
    """Two gauges a month reading 10 and 20 mm, three months of the record missing."""
    months = pd.date_range("2023-01-01", periods=periods, freq="MS").delete([7, 8, 15])
    return pd.DataFrame({
        "Date": months.repeat(2),
        "Gauge": ["North", "South"] * len(months),
        "Rainfall (mm)": [10.0, 20.0] * len(months),
    })


def steady(periods: int = 24) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=periods, freq="MS"),
        "Team": ["A", "B"] * (periods // 2),
        "Revenue": [100.0] * periods,
    })


class AnalyzeTests(unittest.TestCase):

    def test_a_dataframe_in_an_analysis_out(self):
        report = ada.analyze(steady().assign(Revenue=lambda frame: 100.0 + frame.index % 3))
        self.assertIsInstance(report, ada.Analysis)
        self.assertEqual(report.rows_analyzed, 24)
        self.assertEqual(report.roles.measure, "Revenue")

    def test_a_path_works_without_reading_the_file_first(self):
        path = ada.samples()["Ecommerce Orders"]
        report = ada.analyze(path)
        self.assertEqual(report.source_name, "ecommerce-orders.csv")
        self.assertGreater(report.rows_analyzed, 0)

    def test_bytes_work_for_an_upload_handler(self):
        raw = ada.samples()["Ecommerce Orders"].read_bytes()
        self.assertGreater(len(ada.analyze(raw).evidence), 0)

    def test_roles_can_be_overridden(self):
        frame = steady().rename(columns={"Revenue": "Takings"})
        self.assertEqual(ada.analyze(frame, measure="Takings").roles.measure, "Takings")

    def test_printing_the_report_shows_the_calculations(self):
        text = str(ada.analyze(ada.sample()))
        self.assertIn("calc:", text)
        self.assertIn("rows", text)

    def test_an_unusable_input_raises_rather_than_refusing(self):
        with self.assertRaises(ada.NothingToAnalyze):
            ada.analyze(pd.DataFrame())
        with self.assertRaises(ada.UnreadableData):
            ada.analyze("no-such-file-here.csv")
        with self.assertRaises(ada.AdaError):
            ada.analyze(12345)

    def test_errors_are_value_errors(self):
        self.assertTrue(issubclass(ada.AdaError, ValueError))


class RefusalTests(unittest.TestCase):

    def test_a_refusal_is_falsy_and_carries_a_code(self):
        thin = ada.analyze(pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]),
            "Revenue": [1.0, 2.0, 3.0],
        }))
        refusal = thin.forecast()
        self.assertIsInstance(refusal, ada.Refusal)
        self.assertFalse(refusal)
        self.assertEqual(refusal.code, "thin_history")
        self.assertIn(refusal.code, ada.REFUSAL_CODES)
        self.assertEqual(refusal.had, "3 periods")

    def test_every_code_emitted_is_a_documented_code(self):
        no_date = ada.analyze(pd.DataFrame({"Team": list("abcd"), "Revenue": [1.0, 2, 3, 4]}))
        for refusal in (no_date.trend(), no_date.anomalies, no_date.forecast(),
                        no_date.segments(), no_date.focus("x")):
            with self.subTest(code=refusal.code):
                self.assertIn(refusal.code, ada.REFUSAL_CODES)

    def test_rows_per_year_fixed_by_the_design_are_not_forecast(self):
        frame = pd.DataFrame([
            {"Year": f"{year}-01-01", "Plot Code": f"P{plot}", "Crop": "maize" if plot % 2 else "wheat"}
            for year in range(2005, 2016) for plot in range(10)
        ])
        report = ada.analyze(frame.assign(Year=pd.to_datetime(frame["Year"])))

        for refusal in (report.forecast(), report.anomalies):
            with self.subTest(refusal=refusal):
                self.assertIsInstance(refusal, ada.Refusal)
                self.assertEqual(refusal.code, "no_measure_column")
                self.assertIn("at most once a year", refusal.message)

    def test_sampling_trips_are_not_forecast(self):
        starts = pd.to_datetime(["2003-06-16", "2003-09-09", "2003-11-03", "2004-04-26", "2004-08-09",
                                 "2004-10-25", "2005-04-25", "2005-07-25", "2005-10-24"])
        dates = [start + pd.Timedelta(days=day % 4) for start in starts for day in range(20)]
        report = ada.analyze(pd.DataFrame({"date": dates, "secchi": [5.0 + day % 3 for day in range(180)]}))

        refusal = report.forecast()

        self.assertIsInstance(refusal, ada.Refusal)
        self.assertEqual(refusal.code, "not_enough_periods")
        self.assertIn("9 sampling trips", refusal.message)

    def test_a_column_per_month_is_offered_in_long_form(self):
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        values = {name: [float(index) for index in range(10)] for name in months}
        frame = pd.DataFrame({"YEAR": range(1990, 2000), **values})
        report = ada.analyze(frame, date="YEAR")

        long = report.long_form()

        self.assertEqual(long.columns.tolist(), ["Date", "Value"])
        self.assertEqual(len(long), 120)
        self.assertEqual(long["Date"].iloc[13], pd.Timestamp("1991-02-01"))
        self.assertIsInstance(ada.analyze(steady()).long_form(), ada.Refusal)

    def test_an_average_over_sites_that_change_every_year_is_not_forecast(self):
        rows = [
            {"Date": f"{1990 + year}-06-01", "SITE_ID": f"S{year}-{site}" if site < 20 else f"K{site}",
             "COND": 100.0 + 3 * year + site}
            for year in range(25) for site in range(25)
        ]
        frame = pd.DataFrame(rows).assign(Date=lambda table: pd.to_datetime(table["Date"]))

        refusal = ada.analyze(frame, measure="COND", aggregation="mean").forecast()

        self.assertIsInstance(refusal, ada.Refusal)
        self.assertEqual(refusal.code, "changing_units")
        self.assertIn("appear in only one year", refusal.message)

    def test_nothing_found_is_not_the_same_as_could_not_look(self):
        """An empty tuple and a Refusal are both falsy; only the type differs."""
        found = ada.analyze(steady()).anomalies
        self.assertNotIsInstance(found, ada.Refusal)
        self.assertEqual(found, ())

        short = ada.analyze(steady(periods=4)).anomalies
        self.assertIsInstance(short, ada.Refusal)
        self.assertEqual(short.code, "not_enough_periods")

    def test_ask_tells_the_two_refusals_apart(self):
        report = ada.analyze(ada.sample())
        self.assertEqual(report.ask("revenue this year").code, "unsupported_phrasing")
        self.assertEqual(
            report.ask("revenue divided by the moon phase").code, "unrepresentable_question"
        )

    def test_a_refusal_names_the_columns_a_partial_name_may_mean(self):
        report = ada.analyze(pd.DataFrame({
            "island": ["Biscoe", "Dream"] * 10,
            "bill_depth_mm": [18.0 + index / 10 for index in range(20)],
            "body_mass_g": [3_500.0 + index * 40 for index in range(20)],
            "body_length_mm": [400.0 + index for index in range(20)],
        }))

        refusal = report.ask("average body by island")

        self.assertEqual(refusal.code, "unrepresentable_question")
        self.assertIn("Did you mean body_mass_g or body_length_mm?", refusal.message)

    def test_an_answer_says_when_the_row_limit_left_rows_out(self):
        frame = pd.DataFrame({"group": ["a", "b"] * 30, "value": range(60)})

        answer = ada.analyze(frame, row_limit=40).ask("average value by group")
        refusal = ada.analyze(frame, row_limit=40).ask("average value for zz9")

        self.assertIn("Based on the 40 sampled rows (of 60).", answer.answer)
        self.assertIn("the row limit left out 20 others", refusal.message)

    def test_ask_answers_a_supported_question_with_its_calculation(self):
        answer = ada.analyze(ada.sample()).ask("total revenue")
        self.assertNotIsInstance(answer, ada.Refusal)
        self.assertTrue(answer.calculation)


class SurfaceTests(unittest.TestCase):

    def setUp(self):
        self.report = ada.analyze(ada.sample())

    def test_suggestions_are_questions_ask_can_answer(self):
        for question in self.report.suggestions[:4]:
            with self.subTest(question=question):
                self.assertNotIsInstance(self.report.ask(question), ada.Refusal)

    def test_focus_returns_a_whole_analysis(self):
        focused = self.report.focus("Home")
        self.assertIsInstance(focused, ada.Analysis)
        self.assertLess(focused.rows_analyzed, self.report.rows_analyzed)
        self.assertTrue(focused.headline)

    def test_focus_refuses_an_unknown_value(self):
        self.assertEqual(self.report.focus("Nowhere").code, "unknown_segment_value")

    def test_every_evidence_item_carries_its_calculation(self):
        self.assertTrue(self.report.evidence)
        for item in self.report.evidence:
            with self.subTest(title=item.title):
                self.assertTrue(item.calculation.strip())

    def test_to_dict_is_json_safe_and_versioned(self):
        payload = self.report.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["schema_version"], ada.SCHEMA_VERSION)
        self.assertTrue(payload["evidence"][0]["calculation"])

    def test_to_markdown_produces_the_brief(self):
        self.assertIn("#", self.report.to_markdown())

    def test_the_limits_are_readable(self):
        self.assertEqual(ada.DEFAULT_ROW_LIMIT, 1_000_000)
        self.assertEqual(ada.MAX_FILE_BYTES, 25 * 1024 * 1024)

    def test_the_brief_checks_and_answers_share_one_scan_of_the_data(self):
        report = ada.analyze(ada.sample("Lab Measurements"))
        calls = []
        real = ada.api.scan_quality

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        with unittest.mock.patch.object(ada.api, "scan_quality", counted), \
                unittest.mock.patch.object(ada.nlq, "scan_quality", counted):
            self.assertTrue(report.headline)
            report.quality()
            report.ask(report.suggestions[0])
        self.assertEqual(len(calls), 1)

    def test_samples_ship_with_the_package(self):
        self.assertIn("Lab Measurements", ada.samples())
        self.assertGreater(len(ada.sample("Lab Measurements")), 0)
        self.assertGreater(len(ada.sample("lab-measurements")), 0)


class NoStreamlitTests(unittest.TestCase):
    """Runs the public API in a subprocess where the app and AI packages cannot be imported."""

    SCRIPT = textwrap.dedent(
        """
        import sys, warnings
        warnings.filterwarnings("ignore")
        BLOCKED = {"streamlit", "plotly", "openai", "pydantic"}

        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"{name} is not installed (simulated)")

        sys.meta_path.insert(0, Block())

        import ada
        report = ada.analyze(ada.sample())
        assert report.headline
        assert report.evidence and report.evidence[0].calculation
        assert report.kpis and report.recommendations
        assert report.suggestions
        assert report.to_dict()["schema_version"]
        assert report.to_markdown()
        assert str(report)
        assert report.trend().frame is not None
        report.anomalies, report.forecast(), report.segments()
        report.ask("total revenue")
        report.focus("Home")
        leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
        assert not leaked, leaked
        print("CLEAN")
        """
    )

    def test_the_whole_public_surface_runs_without_the_app_stack(self):
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("CLEAN", result.stdout)


class CommandLineTests(unittest.TestCase):
    """Exit codes of `ada check`: 0 nothing flagged, 1 anomaly, 2 unusable input, 3 could not check."""

    def run_ada(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "ada.cli", *args],
            capture_output=True, text=True, timeout=300,
        )

    def write(self, name: str, frame: pd.DataFrame) -> str:
        import tempfile
        path = f"{tempfile.mkdtemp()}/{name}"
        frame.to_csv(path, index=False)
        return path

    def test_sheet_picks_the_worksheet_to_read(self):
        import tempfile
        path = f"{tempfile.mkdtemp()}/book.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            pd.DataFrame({"Note": ["cover sheet"]}).to_excel(writer, index=False, sheet_name="About")
            steady().to_excel(writer, index=False, sheet_name="Data")

        result = self.run_ada("schema", path, "--sheet", "Data")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("24 rows", result.stdout)

    def test_a_quiet_series_exits_zero(self):
        path = self.write("quiet.csv", steady())
        result = self.run_ada("check", path, "--frequency", "M")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing outside", result.stdout)

    def test_a_flagged_period_exits_one(self):
        frame = steady()
        frame.loc[14, "Revenue"] = 480.0
        result = self.run_ada("check", self.write("spike.csv", frame), "--frequency", "M")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("anomaly:", result.stdout)

    def test_too_little_history_exits_three_not_zero(self):
        result = self.run_ada("check", self.write("short.csv", steady(periods=4)), "--frequency", "M")
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertIn("could not look", result.stderr)

    def test_no_timeline_exits_three(self):
        frame = pd.DataFrame({"Team": list("abcd"), "Revenue": [1.0, 2, 3, 4]})
        result = self.run_ada("check", self.write("nodate.csv", frame))
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)

    def test_an_unusable_file_exits_two(self):
        result = self.run_ada("check", "no-such-file-anywhere.csv")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("error:", result.stderr)

    def test_schema_prints_what_ada_decided(self):
        result = self.run_ada("schema", "-")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("measure", result.stdout)

    def test_analyze_json_is_parseable(self):
        result = self.run_ada("analyze", "-", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["schema_version"], ada.SCHEMA_VERSION)

    def test_samples_lists_the_bundled_files(self):
        self.assertIn("Lab Measurements", self.run_ada("samples").stdout)

    def test_the_aggregation_flag_changes_how_the_measure_combines(self):
        path = self.write("rain.csv", rainfall())

        default = self.run_ada("schema", path)
        summed = self.run_ada("schema", path, "--aggregation", "sum")
        analyzed = self.run_ada("analyze", path, "--aggregation", "sum", "--json")
        refused = self.run_ada("analyze", path, "--aggregation", "median")

        self.assertIn("Rainfall (mm) (mean)", default.stdout)
        self.assertIn("Rainfall (mm) (sum)", summed.stdout)
        self.assertEqual(json.loads(analyzed.stdout)["roles"]["aggregation"], "sum")
        self.assertEqual(refused.returncode, 2)


class SkillTests(unittest.TestCase):
    """`ada skill` installs the SKILL.md that tells AI assistants how to use ADA."""

    SKILL = Path(ada.__file__).parent / "skills" / "ada" / "SKILL.md"

    def run_ada(
        self, *args: str, cwd: str | None = None, env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "ada.cli", *args],
            capture_output=True, text=True, timeout=300, cwd=cwd,
            env={**os.environ, "PYTHONPATH": str(Path(ada.__file__).parent.parent), **(env or {})},
        )

    def snapshot(self, root: Path) -> dict:
        """Keep contents and write-related metadata; reading may change access times."""
        result = {}
        for path in [root, *sorted(root.rglob("*"))]:
            info = path.stat()
            result[str(path.relative_to(root))] = (
                info.st_mode, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                path.read_bytes() if path.is_file() else None,
            )
        return result

    def assert_check_status(self, result: subprocess.CompletedProcess, path: Path, status: str):
        self.assertTrue(
            any(str(path) in line and status in line for line in result.stdout.splitlines()),
            result.stdout + result.stderr,
        )

    def test_the_skill_meets_the_agent_skills_format(self):
        text = self.SKILL.read_text(encoding="utf-8")
        front = yaml.safe_load(text.split("---")[1])
        self.assertEqual(front["name"], self.SKILL.parent.name)
        self.assertRegex(front["name"], r"^[a-z0-9]+(-[a-z0-9]+)*$")
        self.assertLessEqual(len(front["description"]), 1024)
        self.assertLessEqual(len(front["compatibility"]), 500)
        self.assertLess(len(text.splitlines()), 500)

    def test_every_command_in_the_skill_exists(self):
        from ada.cli import build_parser

        commands = build_parser()._subparsers._group_actions[0].choices
        named = {
            line.split()[1]
            for line in self.SKILL.read_text(encoding="utf-8").splitlines()
            if line.startswith("ada ") and not line.split()[1].startswith("-")
        }
        self.assertTrue(named)
        self.assertLessEqual(named, set(commands))

    def test_the_skills_study_file_is_a_valid_study(self):
        from ada.research import load

        text = self.SKILL.read_text(encoding="utf-8")
        example = text.split("```yaml\n")[1].split("```")[0]
        folder = Path(tempfile.mkdtemp())
        (folder / "study.yml").write_text("study:\n  title: Example\n" + example, encoding="utf-8")

        study = load(folder / "study.yml")

        self.assertEqual((study.design, study.subject, study.time), ("repeated", "Mouse", "Week"))

    def test_print_writes_the_skill_to_stdout(self):
        for flags in ((), ("--force",)):
            with self.subTest(flags=flags):
                result = self.run_ada("skill", "--print", *flags)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, self.SKILL.read_text(encoding="utf-8"))

    def test_the_skill_goes_where_claude_code_and_codex_look(self):
        folder = tempfile.mkdtemp()
        result = self.run_ada("skill", cwd=folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        for skills in (".claude/skills", ".agents/skills"):
            written = Path(folder, skills, "ada")
            skill = (written / "SKILL.md").read_text(encoding="utf-8")
            self.assertEqual(skill, self.SKILL.read_text(encoding="utf-8"))
            self.assertTrue((written / "agents" / "openai.yaml").is_file())
        self.assertIn("for Claude Code", result.stdout)
        self.assertIn("for Codex", result.stdout)

    def test_for_picks_one_assistant(self):
        folder = tempfile.mkdtemp()
        result = self.run_ada("skill", "--for", "codex", cwd=folder)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(folder, ".agents", "skills", "ada", "SKILL.md").is_file())
        self.assertFalse(Path(folder, ".claude").exists())

    def test_user_writes_to_the_home_folder(self):
        home = tempfile.mkdtemp()
        result = subprocess.run(
            [sys.executable, "-m", "ada.cli", "skill", "--user"], capture_output=True, text=True, timeout=300,
            cwd=tempfile.mkdtemp(),
            env={**os.environ, "HOME": home, "PYTHONPATH": str(Path(ada.__file__).parent.parent)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(Path(home, ".claude", "skills", "ada", "SKILL.md").is_file())
        self.assertTrue(Path(home, ".agents", "skills", "ada", "SKILL.md").is_file())

    def test_codex_gets_a_name_and_a_starting_prompt(self):
        shown = yaml.safe_load((self.SKILL.parent / "agents" / "openai.yaml").read_text(encoding="utf-8"))
        self.assertEqual(shown["interface"]["display_name"], "ADA")
        self.assertTrue(shown["interface"]["short_description"])
        self.assertTrue(shown["interface"]["default_prompt"])

    def test_a_changed_skill_is_kept_unless_forced(self):
        folder = tempfile.mkdtemp()
        target = Path(folder, "ada", "SKILL.md")

        first = self.run_ada("skill", "--dir", folder)
        again = self.run_ada("skill", "--dir", folder)
        target.write_text("my own notes", encoding="utf-8")
        refused = self.run_ada("skill", "--dir", folder)
        kept = target.read_text(encoding="utf-8")
        forced = self.run_ada("skill", "--dir", folder, "--force")

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("already this version", again.stdout)
        self.assertEqual(refused.returncode, 2)
        self.assertEqual(kept, "my own notes")
        self.assertEqual(forced.returncode, 0, forced.stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), self.SKILL.read_text(encoding="utf-8"))

    def test_check_matches_text_without_changing_files_or_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            installed = self.run_ada("skill", cwd=folder)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            skill = Path(folder, ".claude", "skills", "ada", "SKILL.md")
            skill.write_bytes(skill.read_bytes().replace(b"\n", b"\r\n"))
            os.utime(skill, ns=(1_000_000_000, 1_000_000_000))
            before = self.snapshot(Path(folder))

            result = self.run_ada("skill", "--check", cwd=folder)

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertEqual(self.snapshot(Path(folder)), before)

    def test_check_reports_every_missing_file_without_creating_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            before = self.snapshot(Path(folder))

            result = self.run_ada("skill", "--check", "--for", "all", cwd=folder)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual(result.stderr, "")
            for skills in (".claude/skills", ".agents/skills"):
                for name in ("SKILL.md", "agents/openai.yaml"):
                    self.assert_check_status(result, Path(skills, "ada", name), "missing")
            self.assertEqual(self.snapshot(Path(folder)), before)

    def test_check_reports_missing_changed_and_invalid_text_files_together(self):
        with tempfile.TemporaryDirectory() as folder:
            installed = self.run_ada("skill", cwd=folder)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            missing = Path(".claude", "skills", "ada", "agents", "openai.yaml")
            changed = Path(".agents", "skills", "ada", "SKILL.md")
            invalid = Path(".agents", "skills", "ada", "agents", "openai.yaml")
            Path(folder, missing).unlink()
            Path(folder, changed).write_text("my own notes", encoding="utf-8")
            Path(folder, invalid).write_bytes(b"\xff\xfe")
            before = self.snapshot(Path(folder))

            result = self.run_ada("skill", "--check", cwd=folder)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual(result.stderr, "")
            for path, status in ((missing, "missing"), (changed, "differs"), (invalid, "differs")):
                self.assert_check_status(result, path, status)
            self.assertEqual(self.snapshot(Path(folder)), before)

    def test_check_only_reads_the_selected_assistant(self):
        for selected, other, skills in (("claude", "codex", ".agents"), ("codex", "claude", ".claude")):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as folder:
                installed = self.run_ada("skill", cwd=folder)
                self.assertEqual(installed.returncode, 0, installed.stderr)
                changed = Path(skills, "skills", "ada", "SKILL.md")
                Path(folder, changed).write_text("my own notes", encoding="utf-8")
                before = self.snapshot(Path(folder))

                matching = self.run_ada("skill", "--check", "--for", selected, cwd=folder)
                different = self.run_ada("skill", "--check", "--for", other, cwd=folder)

                self.assertEqual(matching.returncode, 0, matching.stdout + matching.stderr)
                self.assertNotIn(str(changed), matching.stdout + matching.stderr)
                self.assertEqual(different.returncode, 1, different.stdout + different.stderr)
                self.assert_check_status(different, changed, "differs")
                self.assertEqual(self.snapshot(Path(folder)), before)

    def test_check_uses_a_custom_directory_even_when_an_assistant_is_selected(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder, "nested", "skills")
            before = self.snapshot(Path(folder))

            missing = self.run_ada(
                "skill", "--check", "--dir", str(destination), "--for", "codex", cwd=folder,
            )

            self.assertEqual(missing.returncode, 1, missing.stdout + missing.stderr)
            for name in ("SKILL.md", "agents/openai.yaml"):
                self.assert_check_status(missing, destination / "ada" / name, "missing")
            self.assertEqual(self.snapshot(Path(folder)), before)
            installed = self.run_ada("skill", "--dir", str(destination), cwd=folder)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            before = self.snapshot(Path(folder))

            matching = self.run_ada(
                "skill", "--check", "--dir", str(destination), "--for", "codex", cwd=folder,
            )

            self.assertEqual(matching.returncode, 0, matching.stdout + matching.stderr)
            self.assertEqual(self.snapshot(Path(folder)), before)

    def test_check_user_reads_the_home_folder_instead_of_the_project(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as home:
            installed = self.run_ada("skill", "--user", cwd=folder, env={"HOME": home})
            self.assertEqual(installed.returncode, 0, installed.stderr)
            before = (self.snapshot(Path(folder)), self.snapshot(Path(home)))

            matching = self.run_ada("skill", "--check", "--user", cwd=folder, env={"HOME": home})
            missing = self.run_ada("skill", "--check", cwd=folder, env={"HOME": home})

            self.assertEqual(matching.returncode, 0, matching.stdout + matching.stderr)
            self.assertEqual(missing.returncode, 1, missing.stdout + missing.stderr)
            for skills in (".claude/skills", ".agents/skills"):
                self.assertIn(str(Path(home, skills, "ada", "SKILL.md")), matching.stdout)
            self.assertEqual((self.snapshot(Path(folder)), self.snapshot(Path(home))), before)

    def test_check_rejects_print_and_force_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            installed = self.run_ada("skill", cwd=folder)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            Path(folder, ".claude", "skills", "ada", "SKILL.md").write_text("my notes", encoding="utf-8")
            before = self.snapshot(Path(folder))

            for conflict in ("--print", "--force"):
                with self.subTest(conflict=conflict):
                    result = self.run_ada("skill", "--check", conflict, cwd=folder)
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertIn("error:", result.stderr)
                    self.assertIn("--check", result.stderr)
                    self.assertIn(conflict, result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(self.snapshot(Path(folder)), before)

    def test_check_reports_path_errors_and_still_lists_other_differences(self):
        cases = (("SKILL.md", "agents/openai.yaml"), ("agents/openai.yaml", "SKILL.md"))
        for error_name, changed_name in cases:
            with self.subTest(error=error_name), tempfile.TemporaryDirectory() as folder:
                destination = Path(folder, "skills")
                installed = self.run_ada("skill", "--dir", str(destination))
                self.assertEqual(installed.returncode, 0, installed.stderr)
                unreadable = destination / "ada" / error_name
                unreadable.unlink()
                unreadable.mkdir()
                changed = destination / "ada" / changed_name
                changed.write_text("my own notes", encoding="utf-8")
                before = self.snapshot(Path(folder))

                result = self.run_ada("skill", "--check", "--dir", str(destination))

                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertIn(str(unreadable), result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assert_check_status(result, changed, "differs")
                self.assertEqual(self.snapshot(Path(folder)), before)

    def test_check_reports_a_file_in_the_destination_path_as_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder, "skills")
            destination.write_text("not a directory", encoding="utf-8")
            before = self.snapshot(Path(folder))

            result = self.run_ada("skill", "--check", "--dir", str(destination))

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("error:", result.stderr)
            self.assertIn(str(destination / "ada" / "SKILL.md"), result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(self.snapshot(Path(folder)), before)

    @unittest.skipUnless(os.name == "posix", "POSIX permissions are needed for an unreadable file")
    def test_check_reports_an_unreadable_file_as_an_error(self):
        with tempfile.TemporaryDirectory() as folder:
            installed = self.run_ada("skill", "--dir", folder)
            self.assertEqual(installed.returncode, 0, installed.stderr)
            target = Path(folder, "ada", "SKILL.md")
            original = target.stat()
            target.chmod(0)
            try:
                try:
                    target.read_bytes()
                except PermissionError:
                    pass
                else:
                    self.skipTest("This user can read files despite their permissions")
                before = target.stat()

                result = self.run_ada("skill", "--check", "--dir", folder)

                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("error:", result.stderr)
                self.assertIn(str(target), result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                after = target.stat()
                for attribute in ("st_mode", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
                    self.assertEqual(getattr(after, attribute), getattr(before, attribute))
            finally:
                target.chmod(original.st_mode)
            self.assertEqual(target.read_bytes(), self.SKILL.read_bytes())


class PluginTests(unittest.TestCase):
    """plugins/automated-data-analyst is the plugin for Claude Code, Codex and the two plugin directories."""

    ROOT = Path(__file__).resolve().parent.parent
    PLUGIN = ROOT / "plugins" / "automated-data-analyst"

    def manifests(self) -> tuple[dict, dict]:
        claude = json.loads((self.PLUGIN / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
        portable = json.loads((self.PLUGIN / "plugin.json").read_text(encoding="utf-8"))
        return claude, portable

    def test_the_plugin_holds_the_skill_ada_skill_installs(self):
        from ada.cli import SKILL_FILES

        for name in SKILL_FILES:
            packaged = (Path(ada.__file__).parent / "skills" / "ada" / name).read_bytes()
            self.assertEqual((self.PLUGIN / "skills" / "ada" / name).read_bytes(), packaged, name)

    def test_both_manifests_name_this_version(self):
        claude, portable = self.manifests()
        for key in ("name", "version", "description", "author", "license"):
            self.assertEqual(claude[key], portable[key], key)
        self.assertEqual(claude["name"], self.PLUGIN.name)
        self.assertEqual(claude["version"], ada.__version__)

    def test_the_marketplace_points_at_the_plugin_and_moves_the_old_name(self):
        catalog = self.ROOT / ".claude-plugin" / "marketplace.json"
        marketplace = json.loads(catalog.read_text(encoding="utf-8"))
        (entry,) = marketplace["plugins"]
        self.assertEqual(entry["name"], self.PLUGIN.name)
        self.assertEqual((self.ROOT / entry["source"]).resolve(), self.PLUGIN)
        self.assertEqual(marketplace["renames"], {"ada": self.PLUGIN.name})

    def test_the_listing_fits_the_directories_limits(self):
        shown = self.manifests()[1]["extensions"]["com.openai"]["interface"]
        self.assertLessEqual(len(shown["displayName"]), 30)
        self.assertLessEqual(len(shown["shortDescription"]), 30)
        self.assertLessEqual(len(shown["longDescription"]), 4000)
        self.assertLessEqual(len(shown["capabilities"]), 20)
        self.assertTrue(all(len(label) <= 120 for label in shown["capabilities"]))
        self.assertLessEqual(len(shown["defaultPrompt"]), 3)
        self.assertTrue(all(len(prompt) <= 128 for prompt in shown["defaultPrompt"]))
        prose = (self.PLUGIN / "README.md").read_text(encoding="utf-8").split("```")[::2]
        self.assertGreaterEqual(sum(len(part.split()) for part in prose), 40)

    def test_the_icon_is_a_square_png_both_directories_accept(self):
        claude, portable = self.manifests()
        shown = portable["extensions"]["com.openai"]["interface"]
        for path in {claude["icon"], shown["logo"], shown["composerIcon"]}:
            icon = (self.PLUGIN / path).read_bytes()
            self.assertEqual(icon[:8], b"\x89PNG\r\n\x1a\n", path)
            width, height = int.from_bytes(icon[16:20], "big"), int.from_bytes(icon[20:24], "big")
            self.assertEqual(width, height, path)
            self.assertTrue(512 <= width <= 2048, path)
            self.assertLess(len(icon), 2 * 1024 * 1024, path)

    def test_the_plugin_holds_only_small_text_files_and_the_icon(self):
        files = [path for path in self.PLUGIN.rglob("*") if path.is_file()]
        self.assertLessEqual(len(files), 512)
        for path in files:
            self.assertIn(path.suffix, {".md", ".json", ".yaml", ".png", ""}, path)
            self.assertLess(path.stat().st_size, 256 * 1024, path)
            if path.suffix != ".png":
                credential = r"[A-Z]+_(API_)?(KEY|TOKEN|SECRET)\b"
                self.assertNotRegex(path.read_text(encoding="utf-8"), credential, path)


class OverrideTests(unittest.TestCase):
    def test_an_unknown_override_column_raises(self):
        with self.assertRaisesRegex(ada.AdaError, "No column named 'Nope'"):
            ada.analyze(steady(), measure="Nope")

    def test_an_override_leaves_other_roles_as_detected(self):
        frame = pd.DataFrame({"Team": ["A", "B"] * 6, "Revenue": [1.0] * 12, "Units": [2.0] * 12})
        report = ada.analyze(frame, measure="Units")
        self.assertEqual(report.roles.measure, "Units")
        self.assertIsNone(report.roles.date)

    def test_profile_has_a_row_per_column(self):
        profile = ada.analyze(steady()).profile()
        self.assertEqual(profile["Column"].tolist(), ["Date", "Team", "Revenue"])

    def test_a_chosen_measure_has_no_runners_up(self):
        frame = pd.DataFrame({
            "HGB": [13.0 + index / 10 for index in range(30)],
            "HCT": [40.0 + index / 7 for index in range(30)],
        })
        self.assertTrue(ada.analyze(frame).roles.measure_alternatives)
        self.assertEqual(ada.analyze(frame, measure="HGB").roles.measure_alternatives, ())


def long_readings() -> pd.DataFrame:
    """Readings of three parameters in three units, one per row."""
    levels = {"Ammonia": ("mg/l", 0.2), "pH": ("phunits", 7.5), "Conductivity": ("us/cm", 450.0)}
    return pd.DataFrame([
        {"Site": f"Site {site % 3}", "Parameter": parameter, "Unit": unit,
         "Result": level * (1 + 0.01 * site + 0.03 * repeat)}
        for site in range(12)
        for parameter, (unit, level) in levels.items()
        for repeat in range(3 if parameter == "Ammonia" else 2)
    ])


class LongTableTests(unittest.TestCase):
    def test_the_brief_covers_the_most_common_parameter_only(self):
        report = ada.analyze(long_readings())

        self.assertTrue((report.frame["Parameter"] == "Ammonia").all())
        self.assertEqual(report.roles.dimension, "Site")
        self.assertIn("Ammonia", report.source_name)
        self.assertTrue(any("never combined" in note for note in report.cleaning.notes))

    def test_naming_the_grouping_keeps_every_row(self):
        report = ada.analyze(long_readings(), dimension="Site")

        self.assertEqual(len(report.frame), len(long_readings()))
        self.assertIsNone(report.roles.focus)

    def test_ask_answers_another_parameter_from_every_row(self):
        report = ada.analyze(long_readings())
        other = next(value for value in long_readings()["Parameter"].unique() if value != "Ammonia")

        answer = report.ask(f"average Result for {other}")
        own = report.ask("average Result by Site")

        self.assertIn(f"Parameter = {other}", answer.answer)
        self.assertIn("This covers the", own.answer)
        self.assertIn("Parameter = Ammonia, which the brief follows", own.answer)


class AggregationTests(unittest.TestCase):
    def test_a_reading_is_averaged_unless_told_otherwise(self):
        averaged = ada.analyze(rainfall())
        summed = ada.analyze(rainfall(), aggregation="sum")

        self.assertEqual(averaged.to_dict()["roles"]["aggregation"], "mean")
        self.assertEqual(summed.to_dict()["roles"]["aggregation"], "sum")
        self.assertEqual(averaged.segments()["Value"].tolist(), [20.0, 10.0])
        self.assertEqual(summed.segments()["Value"].tolist(), [420.0, 210.0])
        self.assertIn("measure=Rainfall (mm) (mean)", str(averaged))

    def test_the_override_survives_a_focus(self):
        focused = ada.analyze(rainfall(), aggregation="sum").focus("North")
        self.assertEqual(focused.roles.aggregation, "sum")

    def test_an_unknown_aggregation_is_refused_plainly(self):
        with self.assertRaisesRegex(ValueError, "'sum' \\(a total\\) or 'mean' \\(an average\\)"):
            ada.analyze(rainfall(), aggregation="median")

    def test_months_without_rows_are_not_anomalies_of_an_averaged_reading(self):
        report = ada.analyze(rainfall(), frequency="M")

        self.assertEqual(report.trend().filled_periods, 3)
        self.assertEqual(report.anomalies, ())

    def test_a_summed_override_counts_months_without_rows_as_zero(self):
        trend = ada.analyze(rainfall(), frequency="M", aggregation="sum").trend()
        self.assertEqual((trend.frame["Value"] == 0).sum(), 3)

    def test_a_falling_temperature_may_be_forecast_below_zero(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "Water Temp (°C)": [16.0 - 1.4 * month + (0.3 if month % 2 else -0.3) for month in range(12)],
        })

        forecast = ada.analyze(frame).forecast()

        self.assertLess(forecast.values[-1], 0.0)
        self.assertLess(min(forecast.lower), forecast.values[-1])


class OverrideTypeTests(unittest.TestCase):
    def test_a_date_override_must_hold_dates(self):
        with self.assertRaisesRegex(ada.AdaError, "could not be read as dates"):
            ada.analyze(steady(), date="Team")

    def test_a_measure_override_must_be_numeric(self):
        with self.assertRaisesRegex(ada.AdaError, "is not numeric"):
            ada.analyze(steady(), measure="Team")

    def test_a_date_override_accepts_a_column_of_years(self):
        frame = pd.DataFrame({"Season": range(1950, 1970), "Catch": [float(n) for n in range(20)]})

        report = ada.analyze(frame, date="Season")

        self.assertEqual(report.roles.date, "Season")
        self.assertEqual(report.period, (pd.Timestamp("1950-01-01"), pd.Timestamp("1969-01-01")))

    def test_a_date_override_still_refuses_numbers_that_are_not_years(self):
        frame = pd.DataFrame({"model_year": [70, 71, 82] * 4, "mpg": [18.0, 15.0, 31.0] * 4})

        with self.assertRaisesRegex(ada.AdaError, "could not be read as dates"):
            ada.analyze(frame, date="model_year")


class FrequencyTests(unittest.TestCase):
    def setUp(self):
        days = pd.date_range("2022-01-01", "2024-12-31", freq="D")
        self.frame = pd.DataFrame({"date": days, "precipitation": [float(day.day % 7) for day in days]})

    def test_the_headline_uses_the_requested_grain(self):
        self.assertIn("Q4 2024", ada.analyze(self.frame).headline)
        monthly = ada.analyze(self.frame, frequency="M")
        self.assertRegex(monthly.headline, r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) 2024")
        trend = next(item for item in monthly.evidence if item.kind == "trend")
        self.assertIn("(Dec 2024)", trend.statement)

    def test_the_headline_and_the_trend_agree_on_the_periods(self):
        monthly = ada.analyze(self.frame, frequency="M")
        self.assertEqual(monthly.trend().frequency, "M")
        self.assertEqual(len(monthly.trend().frame), 36)

    def test_year_month_day_and_hour_columns_make_an_hourly_timeline(self):
        stamps = pd.date_range("2016-12-25", periods=10 * 24, freq="h")
        frame = pd.DataFrame({
            "year": stamps.year, "month": stamps.month, "day": stamps.day, "hour": stamps.hour,
            "PM2.5": [float(10 + stamp.hour % 6) for stamp in stamps], "station": "Aotizhongxin",
        })

        report = ada.analyze(frame)

        self.assertEqual(report.roles.date, "year-month-day-hour")
        self.assertEqual(report.trend().frequency, "h")
        self.assertEqual(len(report.trend().frame), 240)
        self.assertEqual(report.trend().filled_periods, 0)


class YearTimelineTests(unittest.TestCase):
    def setUp(self):
        years = range(1880, 2024)
        self.report = ada.analyze(pd.DataFrame({
            "year": years,
            "temp": [round(0.008 * (year - 1880) + 0.1 * ((year * 7) % 5 - 2), 2) for year in years],
        }))

    def test_a_year_column_is_the_timeline_at_a_yearly_grain(self):
        self.assertEqual(self.report.roles.date, "year")
        trend = self.report.trend()
        self.assertEqual(trend.frequency, "Y")
        self.assertEqual(len(trend.frame), 144)
        self.assertIn("(2023)", self.report.headline)

    def test_anomalies_and_the_forecast_run_on_a_yearly_series(self):
        self.assertNotIsInstance(self.report.anomalies, ada.Refusal)
        self.assertNotIsInstance(self.report.forecast(), ada.Refusal)


class RowsLeftOutTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Material": [name for name in "ABCD" for _ in range(300)],
            "Signal": [float(n % 17) for n in range(1200)],
        })

    def test_every_answer_says_which_rows_it_used(self):
        report = ada.analyze(self.frame, row_limit=400)
        answer = report.ask("average Signal by Material")
        self.assertTrue(answer.answer.endswith("Based on the 400 sampled rows (of 1,200)."))
        self.assertEqual(report.rows_note, "a random sample of 400 of 1,200 rows")
        self.assertEqual(report.to_dict()["rows_note"], report.rows_note)
        self.assertIn("Analyzed a random sample of 400 of 1,200 rows", str(report))
        self.assertNotIn("older", str(report))

    def test_a_table_analyzed_whole_adds_nothing_to_its_answers(self):
        report = ada.analyze(self.frame)
        self.assertNotIn("row limit", report.ask("average Signal by Material").answer)
        self.assertEqual(report.to_dict()["rows_note"], "")


class UndatedTimeTests(unittest.TestCase):
    def setUp(self):
        stamps = pd.date_range("2024-01-01 15:06:28", periods=2974, freq="s")
        self.report = ada.analyze(pd.DataFrame({
            "Time": stamps.strftime("%H:%M:%S"),
            "Signal": [100.0 + (n > 2400) * 200 + n % 5 for n in range(2974)],
        }))

    def test_the_span_is_written_without_a_date(self):
        text = str(self.report)
        self.assertIn("15:06:28 to 15:56:01 (times with no calendar date)", text)
        self.assertNotIn("2026", text)
        period = self.report.to_dict()["period"]
        self.assertEqual((period["from"], period["to"]), ("15:06:28", "15:56:01"))

    def test_segments_give_each_group_its_times_without_the_stand_in_date(self):
        stamps = pd.date_range("2024-01-01 15:06:28", periods=600, freq="s")
        report = ada.analyze(pd.DataFrame({
            "Time": stamps.strftime("%H:%M:%S"),
            "Phase": ["warm-up"] * 200 + ["burn"] * 400,
            "Signal": [100.0 + (n > 200) * 200 + n % 5 for n in range(600)],
        }), dimension="Phase")

        segments = report.segments().set_index("Segment")

        self.assertEqual(segments.loc["warm-up", "From"], "15:06:28")
        self.assertEqual(segments.loc["burn", "To"], "15:16:27")


class WholeNumberDateTests(unittest.TestCase):
    def setUp(self):
        days = pd.date_range("1954-06-01", periods=400)
        self.frame = pd.DataFrame({
            "MESS_DATUM": days.strftime("%Y%m%d").astype(int),
            "TMK": [10.0 + 8 * ((n % 365) / 365) for n in range(400)],
        })

    def test_a_yyyymmdd_column_is_the_timeline(self):
        self.assertEqual(ada.analyze(self.frame).roles.date, "MESS_DATUM")

    def test_a_yyyymmdd_column_can_be_named_as_the_date(self):
        report = ada.analyze(self.frame, date="MESS_DATUM")

        self.assertEqual(report.period[0], pd.Timestamp("1954-06-01"))


def gauges() -> pd.DataFrame:
    """Hourly wind at two stations, four hours logged as the code 999.9, and a humidity over 100%."""
    hours = pd.date_range("2024-01-01", periods=400, freq="h")
    speed = [2.0 + (hour % 7) * 0.4 for hour in range(400)]
    speed[10:14] = [999.9] * 4
    humidity = [60.0 + (hour % 11) for hour in range(400)]
    humidity[50] = 180.0
    return pd.DataFrame({
        "Time": hours,
        "Station": ["North", "South"] * 200,
        "Wind Speed": speed,
        "Humidity": humidity,
    })


class DataCheckTests(unittest.TestCase):
    def test_quality_lists_every_issue_as_a_frame(self):
        report = ada.analyze(gauges(), measure="Humidity")

        table = report.quality()

        self.assertEqual(
            list(table.columns), ["Check", "Columns", "Rows", "Share %", "Detail", "Rule", "Examples"]
        )
        self.assertIn("Impossible values", set(table["Check"]))
        self.assertIn("Sentinel codes", set(table["Check"]))
        self.assertEqual(report.to_dict()["data_checks"], table.to_dict(orient="records"))

    def test_a_clear_code_in_the_measure_is_set_aside_and_noted(self):
        report = ada.analyze(gauges(), measure="Wind Speed")

        self.assertEqual(int(report.frame["Wind Speed"].isna().sum()), 4)
        self.assertTrue(any("999.9" in note for note in report.cleaning.notes), report.cleaning.notes)
        self.assertLess(report.frame["Wind Speed"].max(), 10)

    def test_codes_in_other_columns_are_set_aside_and_still_listed(self):
        report = ada.analyze(gauges(), measure="Humidity")

        self.assertEqual(int(report.frame["Wind Speed"].isna().sum()), 4)
        codes = report.quality().query("Check == 'Sentinel codes'")
        self.assertEqual(codes["Columns"].tolist(), ["Wind Speed"])
        self.assertIn("set those cells to missing", codes["Detail"].iloc[0])
        self.assertEqual(int(codes["Rows"].iloc[0]), 4)

    def test_the_brief_counts_the_codes_set_aside_with_the_other_checks(self):
        report = ada.analyze(gauges(), measure="Humidity")

        card = next(item for item in report.evidence if item.title == "Data checks")

        self.assertEqual(card.value, f"{len(report.quality())} issues")

    def test_missing_counts_include_the_codes_set_aside(self):
        report = ada.analyze(gauges(), measure="Humidity")

        self.assertEqual(int(report.profile().set_index("Column").loc["Wind Speed", "Missing"]), 4)
        self.assertIn("4 of the 400 rows", report.ask("how many rows have missing Wind Speed").answer)

    def test_a_frame_handed_in_is_left_unchanged(self):
        frame = gauges()

        ada.analyze(frame, measure="Humidity")

        self.assertEqual(frame["Wind Speed"].max(), 999.9)

    def test_a_clean_table_has_an_empty_frame(self):
        frame = steady().assign(Revenue=lambda table: 100.0 + table.index % 3)

        self.assertTrue(ada.analyze(frame).quality().empty)


class ChosenMeasureTests(unittest.TestCase):
    def test_a_column_chosen_as_the_measure_is_not_also_a_code(self):
        frame = pd.DataFrame({"Site": ["A", "B"] * 60, "Mark": [n % 7 + 1 for n in range(120)]})

        report = ada.analyze(frame, measure="Mark")

        self.assertNotIn("Mark", report.roles.factors)
        self.assertNotIn("Mark", report.roles.dimensions)


if __name__ == "__main__":
    unittest.main()
