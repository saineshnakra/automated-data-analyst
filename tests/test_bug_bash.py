"""Bugs found walking every customer journey, each pinned so it cannot return."""

import io
import unittest
import zipfile

import numpy as np
import pandas as pd

import ui
from aggregation import build_trend, measure_aggregation, segment_frame
from analysis import clean_dataframe
from autovis import recommend_chart
from business_insights import analyze_business, increase_is_welcome
from file_io import list_excel_sheets, read_tabular_file
from nlq import answer_question, unsupported_phrasing
from pipeline import prepare_analysis
from schema import ColumnRoles, detect_roles

MONTHS = pd.date_range("2024-01-01", periods=12, freq="MS")


class UserColumnNameCollisionTests(unittest.TestCase):
    """The file may call a column anything, including the names used internally."""

    def test_a_measure_named_period_still_gets_a_brief(self):
        frame = pd.DataFrame(
            {"Date": pd.date_range("2024-01-01", periods=60), "Period": np.linspace(1, 9, 60),
             "Region": ["a", "b"] * 30}
        )

        prepared = prepare_analysis(frame, row_limit=1000)

        self.assertTrue(prepared.analyze().headline)

    def test_a_category_named_records_can_be_explored(self):
        frame = pd.DataFrame(
            {"Date": pd.date_range("2024-01-01", periods=60), "Records": ["x", "y", "z"] * 20,
             "Revenue": np.arange(60.0)}
        )
        prepared = prepare_analysis(frame, row_limit=1000)

        spec = recommend_chart(prepared.dataframe, ["Records"])
        shown = ui._explore_frame(prepared.dataframe, spec)

        self.assertEqual(len(shown), 3)


class RefusedPhrasingTests(unittest.TestCase):
    """A precise answer to a different question is worse than no answer."""

    def setUp(self):
        self.frame = pd.DataFrame(
            {"Month": MONTHS[:4], "Region": ["West", "East", "West", "East"],
             "Revenue": [100.0, 200.0, 300.0, 400.0]}
        )
        self.roles = detect_roles(self.frame)

    def test_unrepresentable_phrasings_are_refused_before_parsing(self):
        for question in (
            "Revenue > 200", "Revenue in January and February", "Revenue this year",
            "Revenue for West or Partner", "total revenue last quarter", "revenue in 2024 and 2025",
            "revenue at least 300", "revenue over 1000", "revenue more than 50",
        ):
            with self.subTest(question=question):
                self.assertTrue(unsupported_phrasing(question))
                self.assertIsNone(answer_question(question, self.frame, self.roles))

    def test_plans_the_executor_would_misanswer_are_refused_by_the_parser(self):
        # These read fine; it is the plan that cannot be honoured.
        for question in ("average Revenue monthly", "top 0 Region by Revenue"):
            with self.subTest(question=question):
                self.assertFalse(unsupported_phrasing(question))
                self.assertIsNone(answer_question(question, self.frame, self.roles))

    def test_supported_phrasings_are_not_caught_by_the_refusal(self):
        for question in ("total revenue", "revenue over time", "top 2 region by revenue",
                         "revenue by region", "revenue in march 2024"):
            with self.subTest(question=question):
                self.assertFalse(unsupported_phrasing(question))
                self.assertIsNotNone(answer_question(question, self.frame, self.roles))

    def test_naming_the_column_does_not_steal_its_value_from_the_filter(self):
        answer = answer_question("total Revenue for Region West", self.frame, self.roles)

        self.assertIsNotNone(answer)
        self.assertIn("$400.00", answer.answer)

    def test_an_explicit_minimum_is_computed_not_summed(self):
        answer = answer_question("minimum Revenue by Region", self.frame, self.roles)

        self.assertIsNotNone(answer)
        self.assertEqual(answer.plan.aggregation, "min")

    def test_top_n_applies_to_a_growth_ranking(self):
        answer = answer_question("top 1 Region growth", self.frame, self.roles)

        self.assertIsNotNone(answer)
        self.assertEqual(len(answer.table), 1)


class RateMeasureTests(unittest.TestCase):
    """A rate is averaged; adding two months of conversion rate means nothing."""

    def setUp(self):
        self.frame = pd.DataFrame(
            {"Date": MONTHS[:4], "Conversion Rate": [0.1, 0.2, 0.3, 0.4], "Channel": ["a", "b"] * 2}
        )
        self.roles = detect_roles(self.frame)

    def test_the_rule_lives_in_one_place(self):
        self.assertEqual(measure_aggregation("Conversion Rate"), "mean")
        self.assertEqual(measure_aggregation("Revenue"), "sum")
        self.assertEqual(measure_aggregation("Margin Amount"), "sum")

    def test_the_headline_averages_a_rate_with_a_matching_number(self):
        brief = analyze_business(pd.DataFrame({"Date": MONTHS[:2], "Conversion Rate": [0.1, 0.2]}))

        self.assertIn("averages", brief.headline)
        self.assertIn("15.0%", brief.headline)

    def test_chat_averages_a_rate_unless_told_to_total_it(self):
        averaged = answer_question("conversion rate", self.frame, self.roles)
        totalled = answer_question("total conversion rate", self.frame, self.roles)

        self.assertEqual(averaged.plan.aggregation, "mean")
        self.assertIn("25.0%", averaged.answer)
        self.assertEqual(totalled.plan.aggregation, "sum")

    def test_a_rate_trend_is_one_of_period_means(self):
        series = build_trend(self.frame, self.roles)

        self.assertAlmostEqual(float(series.frame["Value"].iloc[0]), 0.1)


class DirectionOfGoodTests(unittest.TestCase):
    def test_a_rising_cost_is_bad_news(self):
        self.assertFalse(increase_is_welcome("Cost"))
        self.assertFalse(increase_is_welcome("Refund Amount"))
        self.assertTrue(increase_is_welcome("Revenue"))
        self.assertTrue(increase_is_welcome("Cost Savings"))

        frame = pd.DataFrame({"Month": MONTHS, "Cost": np.linspace(30, 90, 12)})
        brief = analyze_business(frame)
        trend = next(item for item in brief.evidence if item.kind == "trend")

        self.assertEqual(trend.tone, "negative")
        self.assertNotIn("growth", brief.recommendations[0].title.lower())

    def test_a_segment_that_grew_is_not_blamed_for_a_fall(self):
        frame = pd.DataFrame(
            {"Month": list(MONTHS[:2]) * 2, "Segment": ["Alpha", "Alpha", "Beta", "Beta"],
             "Revenue": [100.0, 210.0, 500.0, 370.0]}
        )

        brief = analyze_business(frame)

        for item in brief.recommendations:
            self.assertNotIn("Alpha decline", item.action)


class SegmentDenominatorTests(unittest.TestCase):
    def test_blank_segments_stay_in_the_total(self):
        frame = pd.DataFrame(
            {"Date": MONTHS, "Region": ["West", "East"] + [None] * 10,
             "Revenue": [100.0, 200.0] + [90.0] * 10}
        )
        roles = ColumnRoles(date="Date", measure="Revenue", dimension="Region", identifier=None,
                            numeric=("Revenue",), dimensions=("Region",))

        segments = segment_frame(frame, roles)

        self.assertAlmostEqual(float(segments["Value"].sum()), float(frame["Revenue"].sum()))


class RoleDeterminismTests(unittest.TestCase):
    def test_the_event_date_wins_whatever_order_the_columns_arrive_in(self):
        frame = pd.DataFrame(
            {"Order Date": MONTHS[:10], "Ship Date": MONTHS[:10] + pd.Timedelta(days=3), "Revenue": range(10)}
        )

        self.assertEqual(detect_roles(frame).date, "Order Date")
        self.assertEqual(detect_roles(frame[["Ship Date", "Order Date", "Revenue"]]).date, "Order Date")


class CalendarTests(unittest.TestCase):
    def test_annual_figures_are_years_not_zero_filled_quarters(self):
        frame = pd.DataFrame(
            {"Year": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01"]),
             "Revenue": [100.0, 120.0, 150.0, 170.0]}
        )

        series = build_trend(frame, detect_roles(frame))

        self.assertEqual(series.frequency, "Y")
        self.assertEqual(series.filled_periods, 0)
        self.assertEqual(len(series.frame), 4)


class IngestionTests(unittest.TestCase):
    def test_mixed_utc_offsets_still_become_a_date_column(self):
        frame = pd.DataFrame(
            {"created_at": ["2024-01-15T10:00:00+01:00", "2024-06-15T10:00:00+02:00"] * 10,
             "Revenue": range(20)}
        )

        cleaned, _ = clean_dataframe(frame)

        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned["created_at"]))
        self.assertEqual(detect_roles(cleaned).date, "created_at")

    def test_a_whitespace_only_column_is_empty(self):
        frame = pd.DataFrame({"Date": MONTHS[:5], "Revenue": range(5), "Note": ["  "] * 5})

        self.assertNotIn("Note", clean_dataframe(frame)[0].columns)

    def test_a_zip_that_is_not_a_workbook_is_a_readable_error(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("hello.txt", "hi")

        for reader in (list_excel_sheets, read_tabular_file):
            with self.subTest(reader=reader.__name__), self.assertRaises(ValueError):
                reader(buffer.getvalue(), "x.xlsx")

    def test_na_is_north_america_not_missing(self):
        frame = read_tabular_file(b"Region,Revenue\nNA,10\nEU,20\n", "f.csv")

        self.assertEqual(frame["Region"].tolist(), ["NA", "EU"])

    def test_an_id_column_keeps_its_leading_zeros(self):
        frame = read_tabular_file(b"Customer ID,Revenue\n00042,10\n00007,20\n", "f.csv")

        self.assertEqual(frame["Customer ID"].tolist(), ["00042", "00007"])
        self.assertEqual(frame["Revenue"].tolist(), [10, 20])


class EvidenceBudgetTests(unittest.TestCase):
    def test_a_quality_card_is_never_squeezed_out_by_the_six_card_cap(self):
        from business_insights import Evidence, _shown_evidence

        filler = [
            Evidence(kind=f"k{i}", title="t", value="v", statement="s", calculation="c") for i in range(7)
        ]
        quality = Evidence(kind="quality", title="q", value="v", statement="s", calculation="c")

        shown = _shown_evidence(filler + [quality])

        self.assertEqual(len(shown), 6)
        self.assertIn(quality, shown)
