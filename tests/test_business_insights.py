import unittest

import numpy as np
import pandas as pd

from aggregation import driver_frame, heatmap_frame, segment_frame, trend_frame
from business_insights import analyze_business, build_business_report
from demo_data import make_demo_data
from formatting import format_number
from schema import detect_roles


class RoleDetectionTests(unittest.TestCase):
    def test_detects_demo_business_schema(self):
        dataframe = make_demo_data(rows=400)

        roles = detect_roles(dataframe)

        self.assertEqual(roles.date, "Order Date")
        self.assertEqual(roles.measure, "Revenue")
        self.assertEqual(roles.dimension, "Product")
        self.assertEqual(roles.identifier, "Order ID")

    def test_prefers_business_metric_over_numeric_identifier(self):
        dataframe = pd.DataFrame(
            {
                "Customer ID": range(10_000, 10_020),
                "Revenue": range(100, 120),
                "Region": ["West", "East"] * 10,
            }
        )

        roles = detect_roles(dataframe)

        self.assertEqual(roles.measure, "Revenue")
        self.assertEqual(roles.identifier, "Customer ID")


class BusinessAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.dataframe = make_demo_data(rows=1_200)
        self.roles = detect_roles(self.dataframe)

    def test_builds_decision_ready_brief(self):
        brief = analyze_business(self.dataframe, self.roles)
        evidence_kinds = {item.kind for item in brief.evidence}

        self.assertEqual(len(brief.kpis), 4)
        self.assertIn("trend", evidence_kinds)
        self.assertIn("driver", evidence_kinds)
        self.assertIn("leader", evidence_kinds)
        self.assertIn("concentration", evidence_kinds)
        self.assertTrue(brief.recommendations)
        self.assertTrue(brief.headline)

    def test_trend_and_segment_frames_are_chart_ready(self):
        trend = trend_frame(self.dataframe, self.roles)
        segments = segment_frame(self.dataframe, self.roles)

        self.assertEqual(trend.columns.tolist(), ["Period", "Value"])
        self.assertGreater(len(trend), 2)
        self.assertEqual(segments.columns.tolist(), ["Segment", "Value"])
        self.assertEqual(segments.iloc[0]["Segment"], "Enterprise")

    def test_driver_frame_reconciles_with_the_net_movement(self):
        trend = trend_frame(self.dataframe, self.roles)
        net_movement = float(trend.iloc[-1]["Value"] - trend.iloc[-2]["Value"])

        drivers = driver_frame(self.dataframe, self.roles)

        self.assertEqual(drivers.columns.tolist(), ["Segment", "Change"])
        self.assertTrue(len(drivers) >= 2)
        self.assertAlmostEqual(float(drivers["Change"].sum()), net_movement, places=6)

    def test_driver_frame_needs_full_schema(self):
        roles = detect_roles(self.dataframe.drop(columns=["Order Date"]))
        self.assertTrue(driver_frame(self.dataframe.drop(columns=["Order Date"]), roles).empty)

    def test_heatmap_frame_is_segment_by_period(self):
        heat = heatmap_frame(self.dataframe, self.roles)

        self.assertFalse(heat.empty)
        self.assertLessEqual(len(heat), 8)
        self.assertTrue(all(isinstance(period, pd.Timestamp) for period in heat.columns))
        row_totals = heat.sum(axis=1)
        self.assertTrue(row_totals.is_monotonic_decreasing)
        self.assertEqual(str(row_totals.index[0]), "Enterprise")

    def test_heatmap_frame_without_dimension_is_empty(self):
        frame = self.dataframe[["Order Date", "Revenue"]].copy()
        self.assertTrue(heatmap_frame(frame, detect_roles(frame)).empty)

    def test_report_separates_facts_from_recommendations(self):
        brief = analyze_business(self.dataframe, self.roles)
        report = build_business_report(
            self.dataframe,
            brief,
            source_name="demo.csv",
            context="Operating data",
        )

        self.assertIn("## What the data says", report)
        self.assertIn("## What ADA recommends", report)
        self.assertIn("Calculation:", report)
        self.assertIn("not causal proof", report)

    def test_semantic_percentage_formatting_flows_through_kpi_and_report(self):
        dataframe = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]
                ),
                "Gross Margin": [0.35, 0.42, 0.28, 0.31],
                "Region": ["West", "East", "West", "East"],
            }
        )

        roles = detect_roles(dataframe)
        self.assertEqual(roles.measure, "Gross Margin")

        brief = analyze_business(dataframe, roles)
        report = build_business_report(
            dataframe,
            brief,
            source_name="formatting_test.csv",
        )

        kpi_values = [item.value for item in brief.kpis]
        kpi_labels = [item.label for item in brief.kpis]

        self.assertIn("34.0%", kpi_values)
        # Four monthly margins do not add up to a 136% margin, so no total is
        # offered for a rate -- only the average, which means something.
        self.assertNotIn("Total Gross Margin", kpi_labels)
        self.assertIn("Average Gross Margin", kpi_labels)
        self.assertIn("Gross Margin increased 10.7%", report)
        self.assertIn("from 28.0% to 31.0%", report)

    def test_negative_latest_period_prioritizes_diagnosis(self):
        dataframe = pd.DataFrame(
            {
                "Date": pd.to_datetime(
                    ["2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15", "2025-05-15"]
                ),
                "Revenue": [1000, 1100, 1200, 1300, 600],
                "Region": ["West", "West", "East", "East", "West"],
            }
        )

        brief = analyze_business(dataframe)

        self.assertEqual(brief.recommendations[0].title, "Start with East")
        self.assertIn("moved the most of any region", brief.recommendations[0].rationale)

    def test_offsetting_segments_do_not_produce_an_absurd_share(self):
        """A big rise and a big fall net to almost nothing; a share of that is nonsense."""
        dataframe = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-15"] * 2 + ["2025-02-15"] * 2 + ["2025-03-15"] * 2),
                "Revenue": [1_000, 1_000, 1_000, 1_000, 2_000, 20],
                "Region": ["West", "East"] * 3,
            }
        )

        brief = analyze_business(dataframe)
        driver = next(item for item in brief.evidence if item.kind == "driver")

        self.assertIn("offset each other", driver.statement)
        self.assertIn("of all movement", driver.statement)
        self.assertNotIn("of the net movement", driver.statement)
        self.assertIn("near zero", driver.calculation)

    def test_concentration_reports_an_effective_segment_count(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=60, freq="D"),
                "Revenue": [900.0 if index % 12 == 0 else 10.0 for index in range(60)],
                "Product": [f"SKU {index % 12}" for index in range(60)],
            }
        )

        brief = analyze_business(frame)
        concentration = next(item for item in brief.evidence if item.kind == "concentration")

        self.assertEqual(concentration.tone, "warning")
        self.assertIn("of 12", concentration.value)
        self.assertIn("Herfindahl", concentration.calculation)

    def test_an_evenly_spread_business_is_not_called_concentrated(self):
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=40, freq="D"),
                "Revenue": [100.0] * 40,
                "Product": [f"SKU {index % 10}" for index in range(40)],
            }
        )

        brief = analyze_business(frame)
        concentration = next(item for item in brief.evidence if item.kind == "concentration")

        self.assertEqual(concentration.tone, "neutral")
        self.assertIn("10.0 of 10", concentration.value)

    def test_a_mixed_sign_measure_quotes_no_share_at_all(self):
        """Shares of a total that parts of it subtract from are meaningless.

        Not "degraded to a simpler share" -- absent. A percentage taken against
        a net figure the parts do not sum into is how a segment ends up
        contributing 2,000,000% of profit.
        """
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
                "Profit": [500.0, -300.0, 200.0, 100.0] * 3,
                "Product": ["A", "B", "C", "D"] * 3,
            }
        )

        brief = analyze_business(frame)

        self.assertFalse([item for item in brief.evidence if item.kind == "concentration"])
        leader = next(item for item in brief.evidence if item.kind == "leader")
        self.assertIn("no share of the total can be quoted", leader.statement)
        self.assertNotIn("%", leader.statement)

    def test_an_all_negative_measure_ranks_by_size_of_the_loss(self):
        """A cost column has a usable share: -14.4k of -24k really is 60%."""
        frame = pd.DataFrame(
            {
                "Date": list(pd.date_range("2024-01-01", periods=6, freq="D")) * 3,
                "Profit": [-2400.0] * 6 + [-7200.0] * 6 + [-14400.0] * 6,
                "Region": ["North"] * 6 + ["South"] * 6 + ["East"] * 6,
            }
        )

        brief = analyze_business(frame)
        leader = next(item for item in brief.evidence if item.kind == "leader")

        # East carries the largest loss, so East is the leader -- not North,
        # which is merely the number closest to zero.
        self.assertIn("East", leader.statement)
        self.assertIn("60.0%", leader.statement)

    def test_business_number_formatting(self):
        self.assertEqual(format_number(1_250_000, "Revenue"), "$1.2M")
        self.assertEqual(format_number(12_000, "Units"), "12.0K")
        self.assertEqual(format_number(0.25, "Margin %"), "25.0%")
        self.assertEqual(format_number(25, "Conversion Rate"), "25.0%")
        self.assertEqual(format_number(1250, "Cost (EUR)"), "€1.2K")
        self.assertEqual(format_number(1250, "Price GBP"), "£1.2K")
        self.assertEqual(format_number(1250, "Amount USD"), "$1.2K")

        # Currency semantics must take precedence over rate/margin words.
        self.assertEqual(format_number(350_000, "Corporate Revenue"), "$350.0K")
        self.assertEqual(format_number(500, "Total EUR"), "€500.00")

        # Currency codes must match whole words, not substrings.
        self.assertEqual(format_number(30_000, "Europe Sales"), "$30.0K")

        # Percentage detection/scaling must be consistent for negative values.
        self.assertEqual(format_number(-0.3, "Negative Rate"), "-0.3%")


    def test_percentage_formatting_uses_column_level_range(self):
        negative_values = pd.Series([-0.3, -0.1, 0.1, 0.3])
        mixed_values = pd.Series([0.9, 1.0, 1.1, 25.0])
        fraction_values = pd.Series([0.12, 0.18, 0.25, 0.30])

        self.assertEqual(
            format_number(-0.3, "Negative Rate", column_values=negative_values),
            "-0.3%",
        )
        self.assertEqual(
            format_number(0.9, "Mixed Rate", column_values=mixed_values),
            "0.9%",
        )
        self.assertEqual(
            format_number(25.0, "Mixed Rate", column_values=mixed_values),
            "25.0%",
        )
        self.assertEqual(
            format_number(0.25, "Margin %", column_values=fraction_values),
            "25.0%",
        )


    def test_currency_detection_precedes_percentage_detection(self):
        self.assertEqual(
            format_number(350_000, "Corporate Revenue"),
            "$350.0K",
        )


    def test_integer_currency_is_formatted_as_currency(self):
        self.assertEqual(
            format_number(500, "Total EUR"),
            "€500.00",
        )


    def test_currency_code_requires_a_whole_token(self):
        self.assertEqual(
            format_number(30_000, "Europe Sales"),
            "$30.0K",
        )

    def test_a_ratio_named_column_holding_money_is_not_a_percentage(self):
        """"Gross Margin" is as often an amount as a ratio."""
        self.assertEqual(format_number(1_250_000, "Gross Margin"), "1.2M")
        self.assertEqual(format_number(45.0, "Gross Margin"), "45.0%")

    def test_the_headline_is_not_cut_at_a_decimal_point(self):
        """"Revenue increased 18.5%" used to be truncated to "Revenue increased 18."."""
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=200),
                "Revenue": range(200),
                "Region": [f"r{index}" for index in range(200)],
            }
        )

        headline = analyze_business(frame).headline

        self.assertNotRegex(headline, r"\d+\.$")
        self.assertIn("%", headline)

    def test_non_finite_values_never_print_as_nan(self):
        """A percentage is a division, and division can fail."""
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=20),
                "Revenue": [np.inf] * 10 + [1.0] * 10,
                "Region": ["A", "B"] * 10,
            }
        )

        brief = analyze_business(frame)

        self.assertNotIn("nan", brief.headline.lower())
        for kpi in brief.kpis:
            self.assertNotIn("nan", kpi.value.lower())




class NumberRenderingTests(unittest.TestCase):
    def test_the_minus_sign_belongs_to_the_amount_not_the_currency(self):
        self.assertEqual(format_number(-1_200.0, "Expense Amount"), "-$1.2K")
        self.assertEqual(format_number(-1_500_000.0, "Cost"), "-$1.5M")
        self.assertEqual(format_number(-12.0, "Cost"), "-$12.00")

    def test_negative_zero_is_just_zero(self):
        self.assertEqual(format_number(-0.0, "Cost"), "$0.00")

    def test_totals_larger_than_a_billion_keep_a_readable_scale(self):
        self.assertEqual(format_number(1.5e12, "Revenue"), "$1.5T")
        # Past a quadrillion the exponent says more than the commas do.
        self.assertEqual(format_number(3.6e20, "amount"), "$3.6e+20")


class KpiStripTests(unittest.TestCase):
    def test_a_thin_file_shows_fewer_tiles_rather_than_repeating_one(self):
        frame = pd.DataFrame({"Note": ["a", "b", "c"] * 4})

        brief = analyze_business(frame)
        labels = [item.label for item in brief.kpis]

        self.assertEqual(len(labels), len(set(labels)))


class MovementWordingTests(unittest.TestCase):
    def test_a_flat_period_is_not_called_a_large_swing(self):
        values = [1000.0, 1050.0, 1100.0, 1155.0, 1210.0, 1270.0, 1330.0, 1400.0, 1470.0,
                  1540.0, 1540.0]
        frame = pd.DataFrame(
            {"Month": pd.date_range("2023-01-01", periods=len(values), freq="MS"),
             "Revenue": values}
        )

        brief = analyze_business(frame)
        trend = next(item for item in brief.evidence if item.kind == "trend")

        self.assertIn("0.0%", trend.value)
        self.assertNotIn("large swing", trend.statement)
        self.assertIn("quiet", trend.statement)

    def test_a_quarter_is_named_as_a_quarter(self):
        frame = pd.DataFrame(
            {"Date": pd.date_range("2021-01-01", periods=140, freq="W"),
             "Revenue": [100.0] * 130 + [400.0] * 10}
        )

        brief = analyze_business(frame)
        trend = next(item for item in brief.evidence if item.kind == "trend")

        # Whatever grain was chosen, the label must not name a month for a
        # period that is not one -- the anomaly card and the axis already agree.
        self.assertNotRegex(trend.statement, r"\((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\)")



class PercentageVersusCurrencyTests(unittest.TestCase):
    def test_an_explicit_percent_sign_beats_a_currency_word(self):
        values = pd.Series([0.31, 0.33, 0.35])

        self.assertEqual(
            format_number(0.33, "Profit Margin %", column_values=values), "33.0%"
        )

    def test_a_percentage_word_in_the_head_position_wins(self):
        self.assertEqual(format_number(0.42, "Discount Rate"), "42.0%")

    def test_a_currency_word_in_the_head_position_wins(self):
        self.assertEqual(format_number(1_200.0, "Margin Amount"), "$1.2K")

    def test_a_column_gets_one_unit_for_every_row(self):
        """Judging plausibility per value gave 551.3% and 1.4K in one column."""
        values = pd.Series([1000.0, 551.26, 201.79, 1400.0, 558.08])

        rendered = [format_number(value, "Gross Margin", column_values=values) for value in values]

        self.assertFalse(any(text.endswith("%") for text in rendered))

if __name__ == "__main__":
    unittest.main()