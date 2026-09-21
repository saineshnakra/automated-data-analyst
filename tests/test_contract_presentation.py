"""What the page shows is a preview of what was computed, and says so.

Direction and desirability are separate things; a cap is a reading budget,
not a result; a sample is disclosed; and a period with no observation is
not a period that measured zero.
"""

from __future__ import annotations

import re
import unittest

import numpy as np
import pandas as pd

from aggregation import build_trend
from autovis import ChartSpec
from business_insights import Evidence, analyze_business, build_business_report, preview_evidence
from metrics import increase_is_welcome
from pipeline import prepare_analysis
from schema import ColumnRoles, detect_roles
from ui import _explore_frame


def cost_fixture() -> pd.DataFrame:
    rows = []
    for month, costs in (
        ("2024-11-15", (100, 100, 100)),
        ("2024-12-15", (100, 100, 100)),
        ("2025-01-15", (100, 100, 100)),
        ("2025-02-15", (20, 130, 130)),
    ):
        rows.extend(
            (month, segment, cost) for segment, cost in zip(("Alpha", "Beta", "Gamma"), costs, strict=True)
        )
    frame = pd.DataFrame(rows, columns=["Date", "Segment", "Cost"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


class DirectionAndDesirabilityTests(unittest.TestCase):
    def test_a_cost_reduction_keeps_its_driver_and_asks_for_more_of_it(self):
        # D-17: total cost 300 -> 280 is good news, driven by Alpha's 80
        # reduction. The driver used to be discarded as "negative" and the
        # recommendation asked which segment created the latest increase.
        frame = cost_fixture()
        roles = detect_roles(frame)
        self.assertEqual(roles.measure, "Cost")
        brief = analyze_business(frame, roles)
        trend = next(item for item in brief.evidence if item.kind == "trend")
        driver = next(item for item in brief.evidence if item.kind == "driver")
        report = build_business_report(frame, brief, source_name="costs.csv")

        self.assertEqual(trend.tone, "positive")
        self.assertIn("decreased", trend.statement)
        self.assertEqual(driver.tone, "positive")
        self.assertEqual(driver.subject, "Alpha")
        self.assertNotIn("latest increase", report)
        self.assertNotIn("growth plan", report.lower())
        self.assertIn("reduction", report.lower())

    def test_direction_of_good_is_read_from_the_whole_name(self):
        self.assertTrue(increase_is_welcome("Revenue After Returns"))
        self.assertTrue(increase_is_welcome("Cost Savings"))
        self.assertFalse(increase_is_welcome("Refund Amount"))
        self.assertFalse(increase_is_welcome("Cost"))
        self.assertTrue(increase_is_welcome("Units"))

    def test_a_flat_metric_is_flat(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]),
                "Revenue": [100.0, 100.0, 100.0, 100.0],
            }
        )
        brief = analyze_business(frame, detect_roles(frame))
        trend = next(item for item in brief.evidence if item.kind == "trend")
        self.assertEqual(trend.tone, "neutral")
        self.assertIn("was unchanged", trend.statement)
        self.assertNotIn("increased 0", trend.statement)
        titles = [item.title for item in brief.recommendations]
        self.assertNotIn("Protect what improved", titles)


class CapTests(unittest.TestCase):
    def test_every_anomaly_is_counted_and_the_preview_says_how_many_it_shows(self):
        # D-18: 60 quarter-ends, every sixth at 1,000. Ten anomalies, of
        # which the card shows five -- and must say so.
        dates = pd.date_range("2010-03-31", periods=60, freq="QE")
        values = np.where(np.arange(60) % 6 == 5, 1000.0, 100.0)
        frame = pd.DataFrame({"Date": dates, "Revenue": values})
        brief = analyze_business(frame, detect_roles(frame))
        anomaly = next(item for item in brief.evidence if item.kind == "anomaly")
        self.assertEqual(anomaly.value, "10")
        self.assertIn("10 periods sit outside", anomaly.statement)
        self.assertIn("showing the sharpest 5", anomaly.statement)

    def test_a_preview_never_drops_the_quality_card(self):
        cards = [
            Evidence(kind=f"k{i}", title=str(i), value="", statement="", calculation="") for i in range(7)
        ]
        quality = Evidence(kind="quality", title="Data quality", value="", statement="", calculation="")
        shown = preview_evidence([*cards, quality], limit=4)
        self.assertEqual(len(shown), 4)
        self.assertIs(shown[-1], quality)
        self.assertEqual([item.title for item in shown[:3]], ["0", "1", "2"])
        untouched = preview_evidence(cards[:3], limit=4)
        self.assertEqual(untouched, cards[:3])

    def test_the_brief_keeps_every_finding(self):
        rng = np.random.default_rng(7)
        months = pd.date_range("2022-01-31", periods=36, freq="ME")
        rows = []
        for index, month in enumerate(months):
            for region, weight in (("North", 3.0), ("South", 1.0), ("East", 1.0)):
                revenue = 1000.0 * weight * (1 + 0.02 * index) + rng.normal(0, 20)
                if index in (10, 20) and region == "North":
                    revenue *= 4
                rows.append(
                    (month, region, revenue, revenue * 0.6 + rng.normal(0, 10), None if index % 5 else 1.0)
                )
        frame = pd.DataFrame(rows, columns=["Date", "Region", "Revenue", "Cost", "Discount"])
        frame.loc[3, "Region"] = None
        brief = analyze_business(frame, detect_roles(frame))
        kinds = [item.kind for item in brief.evidence]
        self.assertGreater(len(kinds), 6, kinds)
        self.assertEqual(len(kinds), len(set(kinds)))
        self.assertLessEqual(len(preview_evidence(brief.evidence, 6)), 6)


class ChartPayloadTests(unittest.TestCase):
    def test_a_distribution_is_binned_over_every_row(self):
        # D-19: 20,000 ones then 1,000 of 10,000; the first-N sample showed
        # no tail at all.
        frame = pd.DataFrame({"Revenue": [1.0] * 20_000 + [10_000.0] * 1_000})
        spec = ChartSpec(form="histogram", rationale="", x="Revenue")
        binned = _explore_frame(frame, spec)
        self.assertEqual(int(binned["Records"].sum()), 21_000)
        self.assertEqual(int(binned["Records"].iloc[-1]), 1_000)
        self.assertIn("All 21,000 values", binned.attrs["note"])

    def test_a_time_series_is_trimmed_in_whole_periods(self):
        # D-19: five series over 81 dates, one missing on the last date, is
        # 404 rows. A 400-row cut split the earliest date across series.
        dates = pd.date_range("2024-01-01", periods=81, freq="D")
        rows = [(date, series, 1.0) for date in dates for series in "ABCDE"]
        rows = [row for row in rows if not (row[0] == dates[-1] and row[1] == "E")]
        frame = pd.DataFrame(rows, columns=["Date", "Series", "Value"])
        spec = ChartSpec(form="line", rationale="", x="Date", y="Value", color="Series")
        plotted = _explore_frame(frame, spec, limit=400)
        per_date = plotted.groupby("Date")["Series"].nunique()
        self.assertTrue((per_date.iloc[:-1] == 5).all(), per_date.value_counts())
        self.assertLessEqual(len(plotted), 400)
        self.assertIn("most recent", plotted.attrs["note"])

    def test_a_scatter_sample_is_spread_and_disclosed(self):
        frame = pd.DataFrame({"X": np.arange(50_000, dtype=float), "Y": np.arange(50_000, dtype=float)})
        spec = ChartSpec(form="scatter", rationale="", x="X", y="Y")
        points = _explore_frame(frame, spec)
        self.assertLessEqual(len(points), 20_000)
        self.assertGreater(points["X"].max(), 49_000)
        self.assertIn("every 3th point", points.attrs["note"].replace("3rd", "3th"))


class CalendarTests(unittest.TestCase):
    def test_a_rate_with_no_observation_in_a_month_has_no_rate_that_month(self):
        # D-13: 20% in January, March and April got a 0% February inserted.
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-31", "2025-03-31", "2025-04-30"]),
                "Conversion Rate": [0.2, 0.2, 0.2],
            }
        )
        series = build_trend(frame, detect_roles(frame))
        february = series.frame.loc[series.frame["Period"] == pd.Timestamp("2025-02-01"), "Value"]
        self.assertEqual(len(february), 1)
        self.assertTrue(np.isnan(float(february.iloc[0])))
        brief = analyze_business(frame, detect_roles(frame))
        report = build_business_report(frame, brief, source_name="rates.csv")
        self.assertIsNone(re.search(r"(?<![\d.])0\.0%", report), report)

    def test_a_row_with_a_missing_value_is_a_missing_period_not_a_zero(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]),
                "Revenue": [100.0, np.nan, 100.0, 100.0],
            }
        )
        series = build_trend(frame, detect_roles(frame))
        february = float(
            series.frame.loc[series.frame["Period"] == pd.Timestamp("2025-02-01"), "Value"].iloc[0]
        )
        self.assertTrue(np.isnan(february))

    def test_two_readings_are_not_spread_into_invented_weeks(self):
        frame = pd.DataFrame(
            {"Month": pd.to_datetime(["2024-01-31", "2024-02-29"]), "Revenue": [100.0, 120.0]}
        )
        series = build_trend(frame, detect_roles(frame))
        self.assertEqual(len(series.frame), 2)
        self.assertEqual(series.filled_periods, 0)

    def test_a_truncated_leading_period_is_dropped_not_reported_as_a_jump(self):
        # D-13: cutting a two-record February to one record produced an
        # artificial February 100 -> March 200 jump.
        rows = [(f"2025-{month:02d}-{day:02d}", 100.0) for month in range(2, 10) for day in (10, 20)]
        frame = pd.DataFrame(rows, columns=["Date", "Revenue"])
        frame["Date"] = pd.to_datetime(frame["Date"])
        # Fifteen of sixteen rows: one February record is cut, so February
        # is a partial period and must not be analyzed as a whole one.
        prepared = prepare_analysis(frame, row_limit=15)
        roles = ColumnRoles(
            date="Date",
            measure="Revenue",
            dimension=None,
            identifier=None,
            numeric=("Revenue",),
            dimensions=(),
        )
        series = build_trend(prepared.dataframe, roles)
        self.assertEqual(series.frame["Period"].min(), pd.Timestamp("2025-03-01"))
        self.assertEqual(prepared.analyzed_from, pd.Timestamp("2025-03-10"))


if __name__ == "__main__":
    unittest.main()
