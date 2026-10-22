import json
import subprocess
import sys
import textwrap
import unittest

import pandas as pd

import ada


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

        self.assertIn("The row limit left out 20 rows", answer.answer)
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


class RowsLeftOutTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Material": [name for name in "ABCD" for _ in range(300)],
            "Signal": [float(n % 17) for n in range(1200)],
        })

    def test_every_answer_says_which_rows_it_used(self):
        report = ada.analyze(self.frame, row_limit=400)
        answer = report.ask("average Signal by Material")
        self.assertTrue(answer.answer.endswith(
            "The row limit left out 800 rows, so this answer covers only a random sample of 400."
        ))
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

    def test_missing_counts_include_the_codes_set_aside(self):
        report = ada.analyze(gauges(), measure="Humidity")

        self.assertEqual(int(report.profile().set_index("Column").loc["Wind Speed", "Missing"]), 4)
        self.assertIn("4 of the 400 rows", report.ask("how many rows have missing Wind Speed").answer)

    def test_a_frame_handed_in_is_left_unchanged(self):
        frame = gauges()

        ada.analyze(frame, measure="Humidity")

        self.assertEqual(frame["Wind Speed"].max(), 999.9)

    def test_a_clean_table_has_an_empty_frame(self):
        self.assertTrue(ada.analyze(steady()).quality().empty)


class ChosenMeasureTests(unittest.TestCase):
    def test_a_column_chosen_as_the_measure_is_not_also_a_code(self):
        frame = pd.DataFrame({"Site": ["A", "B"] * 60, "Mark": [n % 7 + 1 for n in range(120)]})

        report = ada.analyze(frame, measure="Mark")

        self.assertNotIn("Mark", report.roles.factors)
        self.assertNotIn("Mark", report.roles.dimensions)


if __name__ == "__main__":
    unittest.main()
