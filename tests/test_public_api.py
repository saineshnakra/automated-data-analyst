import json
import subprocess
import sys
import textwrap
import unittest

import pandas as pd

import ada


def steady(periods: int = 24) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=periods, freq="MS"),
        "Team": ["A", "B"] * (periods // 2),
        "Revenue": [100.0] * periods,
    })


class AnalyzeTests(unittest.TestCase):

    def test_a_dataframe_in_an_analysis_out(self):
        report = ada.analyze(steady())
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
        self.assertEqual(report.ask("revenue greater than 200").code, "unsupported_phrasing")
        self.assertEqual(
            report.ask("revenue divided by the moon phase").code, "unrepresentable_question"
        )

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
        self.assertEqual(ada.DEFAULT_ROW_LIMIT, 250_000)
        self.assertEqual(ada.MAX_FILE_BYTES, 25 * 1024 * 1024)

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
        self.assertIn("(Q4 2024)", ada.analyze(self.frame).headline)
        monthly = ada.analyze(self.frame, frequency="M")
        self.assertIn("(Dec 2024)", monthly.headline)
        trend = next(item for item in monthly.evidence if item.kind == "trend")
        self.assertIn("(Dec 2024)", trend.statement)

    def test_the_headline_and_the_trend_agree_on_the_periods(self):
        monthly = ada.analyze(self.frame, frequency="M")
        self.assertEqual(monthly.trend().frequency, "M")
        self.assertEqual(len(monthly.trend().frame), 36)


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


if __name__ == "__main__":
    unittest.main()
