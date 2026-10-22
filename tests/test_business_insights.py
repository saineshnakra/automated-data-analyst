import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ada.aggregation import driver_frame, heatmap_frame, segment_frame, segment_period_change, trend_frame
from ada.business_insights import (
    Evidence,
    _shown_evidence,
    analyze_business,
    build_business_report,
    describe_correlation,
    measure_correlation,
    numeric_readings,
)
from ada.demo_data import make_demo_data
from ada.formatting import format_number
from ada.metrics import increase_is_welcome
from ada.pipeline import prepare_analysis
from ada.schema import ColumnRoles, detect_roles

MONTHS = pd.date_range("2024-01-01", periods=12, freq="MS")


def brief_for(measure: str):
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    frame = pd.DataFrame({
        "Month": list(dates) * 2,
        "Team": ["A"] * 24 + ["B"] * 24,
        measure: [10.0] * 24 + [20.0] * 24,
    })
    prepared = prepare_analysis(frame, row_limit=5000)
    return analyze_business(prepared.dataframe, prepared.detected_roles)


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


class FactorEvidenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        status = np.repeat([0, 1, 2], 40)
        self.frame = pd.DataFrame({
            "Status": status,
            "Signal": status * 10 + rng.normal(0, 1, 120),
            "Noise": rng.normal(0, 1, 120),
        })
        self.roles = ColumnRoles(
            date=None, measure="Signal", dimension="Status", identifier=None,
            numeric=("Signal", "Noise", "Status"), dimensions=("Status",), factors=("Status",),
        )

    def test_a_factor_is_left_out_of_the_strongest_relationship(self):
        evidence = analyze_business(self.frame, self.roles).evidence

        relationships = [item.statement for item in evidence if item.kind == "relationship"]
        self.assertFalse(any("Status" in statement for statement in relationships))

    def test_a_numeric_group_is_named_with_its_column(self):
        evidence = analyze_business(self.frame, self.roles).evidence
        leader = next(item for item in evidence if item.kind == "leader")

        self.assertIn("Status 2 has the highest mean", leader.statement)
        self.assertEqual(leader.subject, "Status 2")


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
        # Summed, the four margins would read 136%, so a rate gets an average and no total.
        self.assertNotIn("Total Gross Margin", kpi_labels)
        self.assertIn("Average Gross Margin", kpi_labels)
        # A rate moves in points: 28% to 31% is 3 points, not +10.7%.
        self.assertIn("Gross Margin increased by 3.0 percentage points", report)
        self.assertIn("from 28.0% to 31.0%", report)

    def test_negative_latest_period_prioritizes_diagnosis(self):
        months = ["2025-01-15", "2025-02-15", "2025-03-15", "2025-04-15", "2025-05-15"]
        dataframe = pd.DataFrame(
            {
                "Date": pd.to_datetime(months * 2),
                "Revenue": [500, 520, 540, 560, 580, 500, 550, 600, 650, 100],
                "Region": ["West"] * 5 + ["East"] * 5,
            }
        )

        brief = analyze_business(dataframe)

        self.assertEqual(brief.recommendations[0].title, "Start with East")
        self.assertIn("moved the most of any region", brief.recommendations[0].rationale)

    def test_a_region_that_stops_reporting_is_not_the_driver(self):
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

        self.assertNotIn("driver", [item.kind for item in brief.evidence])
        trend = next(item for item in brief.evidence if item.kind == "trend")
        self.assertIn("No region value reported in both periods", trend.statement)
        self.assertIn("East reported in Apr 2025 but not in May 2025", trend.statement)

    def test_offsetting_segments_are_measured_against_all_movement(self):
        """West rises by 1,000 and East falls by 980, so the net movement is only 20."""
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
                "Revenue": [90.0 if index // 10 % 2 else 110.0 for index in range(40)],
                "Product": [f"SKU {index % 10}" for index in range(40)],
            }
        )

        brief = analyze_business(frame)
        concentration = next(item for item in brief.evidence if item.kind == "concentration")

        self.assertEqual(concentration.tone, "neutral")
        self.assertIn("10.0 of 10", concentration.value)

    def test_a_mixed_sign_measure_quotes_no_share_at_all(self):
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
        self.assertIn("shares of the total are not shown", leader.statement)
        self.assertNotIn("%", leader.statement)

    def test_an_all_negative_measure_ranks_by_size_of_the_loss(self):
        """Each date has -2,400, -7,200 and -14,400; East's -14,400 of -24,000 is 60%."""
        frame = pd.DataFrame(
            {
                "Date": list(pd.date_range("2024-01-01", periods=6, freq="D")) * 3,
                "Profit": [-2400.0] * 6 + [-7200.0] * 6 + [-14400.0] * 6,
                "Region": ["North"] * 6 + ["South"] * 6 + ["East"] * 6,
            }
        )

        brief = analyze_business(frame)
        leader = next(item for item in brief.evidence if item.kind == "leader")

        # East has the largest loss; North only has the value closest to zero.
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

        # A currency word outranks a rate or margin word.
        self.assertEqual(format_number(350_000, "Corporate Revenue"), "$350.0K")
        self.assertEqual(format_number(500, "Total EUR"), "€500.00")

        # Currency codes match whole words only.
        self.assertEqual(format_number(30_000, "Europe Sales"), "$30.0K")

        # Negative rates are detected and scaled like positive ones.
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
        self.assertEqual(format_number(1_250_000, "Gross Margin"), "1,250,000")
        self.assertEqual(format_number(45.0, "Gross Margin"), "45.0%")

    def test_the_headline_is_not_cut_at_a_decimal_point(self):
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
        # Above a quadrillion the value is shown in scientific notation.
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

        # 140 weeks roll up to quarters, so the label must not name a month.
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
        values = pd.Series([1000.0, 551.26, 201.79, 1400.0, 558.08])

        rendered = [format_number(value, "Gross Margin", column_values=values) for value in values]

        self.assertFalse(any(text.endswith("%") for text in rendered))


class RateMeasureTests(unittest.TestCase):
    def setUp(self):
        # Team A averages 10% and team B 20%; a share would read 20 / (10 + 20) = 66.7%.
        self.brief = brief_for("Churn Rate")

    def leader(self):
        return next(e for e in self.brief.evidence if e.kind == "leader")

    def test_the_leader_card_quotes_no_share(self):
        leader = self.leader()
        self.assertNotIn("66.7%", leader.statement, leader.statement)
        self.assertNotIn("contributing", leader.statement, leader.statement)

    def test_the_leader_calculation_does_not_divide_by_a_total(self):
        self.assertNotIn("÷ total", self.leader().calculation, self.leader().calculation)

    def test_no_concentration_card_for_a_rate(self):
        """The concentration card is a Herfindahl index over segment shares."""
        kinds = [e.kind for e in self.brief.evidence]
        self.assertNotIn("concentration", kinds, kinds)

    def test_the_kpi_is_not_labelled_a_share(self):
        labels = [kpi.label for kpi in self.brief.kpis]
        self.assertIn("Highest average by team", labels)
        self.assertFalse([label for label in labels if "share" in label.lower()], labels)

    def test_the_leader_is_still_named(self):
        self.assertIn("B", self.leader().statement)
        self.assertIn("20", self.leader().statement)

    def test_a_rate_headline_does_not_say_coming_from(self):
        headline = self.brief.headline
        self.assertNotIn("coming from", headline, headline)

    def test_a_rate_headline_still_names_the_leader_level(self):
        headline = self.brief.headline
        expected = "Churn Rate is higher in B than in A: mean 20.0% vs 10.0% (n = 24 and 24)"
        self.assertIn(expected, headline, headline)

    def test_the_headline_averages_a_rate_with_a_matching_number(self):
        # Without a date there is no trend card, so the headline falls back to the rate's average.
        brief = analyze_business(pd.DataFrame({"Conversion Rate": [0.1, 0.2], "Channel": ["a", "b"]}))

        self.assertIn("averages", brief.headline)
        self.assertIn("15.0%", brief.headline)


class AdditiveMeasureTests(unittest.TestCase):
    def setUp(self):
        self.brief = brief_for("Revenue")

    def test_an_additive_leader_still_quotes_a_share(self):
        leader = next(e for e in self.brief.evidence if e.kind == "leader")
        self.assertIn("the largest share of revenue: 66.7%", leader.statement, leader.statement)

    def test_an_additive_measure_keeps_its_concentration_card(self):
        self.assertIn("concentration", [e.kind for e in self.brief.evidence])

    def test_an_additive_headline_keeps_its_share(self):
        headline = self.brief.headline
        self.assertIn("B has the largest share of revenue", headline, headline)
        self.assertIn("66.7%", headline, headline)


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

    def test_a_rise_in_a_reading_is_neither_good_nor_bad_news(self):
        frame = pd.DataFrame({"Month": MONTHS, "CO2": np.linspace(410, 416, 12)})

        brief = analyze_business(frame)

        trend = next(item for item in brief.evidence if item.kind == "trend")
        movement = next(item for item in brief.kpis if item.label == "Latest movement")
        self.assertEqual(trend.tone, "neutral")
        self.assertEqual(movement.tone, "neutral")
        # A steady rise of the usual size is no clear change, so there is nothing to follow up.
        self.assertTrue(trend.quiet)
        self.assertEqual(brief.recommendations[0].title, "Collect more before acting")

    def test_a_reading_named_as_adverse_keeps_its_direction(self):
        frame = pd.DataFrame({"Month": MONTHS, "Resolution Hours": np.linspace(10, 16, 12)})

        trend = next(item for item in analyze_business(frame).evidence if item.kind == "trend")

        self.assertEqual(trend.tone, "negative")

    def test_a_site_moving_against_a_neutral_average_is_not_named(self):
        # North rises 6 degrees, but South's four readings fall 2, so the average falls.
        frame = pd.DataFrame(
            {
                "Month": [MONTHS[0]] * 5 + [MONTHS[1]] * 5,
                "Site": ["North", "South", "South", "South", "South"] * 2,
                "Water Temp (°C)": [10.0, 20.0, 20.0, 20.0, 20.0, 16.0, 18.0, 18.0, 18.0, 18.0],
            }
        )

        brief = analyze_business(frame)

        kinds = {item.kind: item for item in brief.evidence}
        self.assertEqual(kinds["trend"].tone, "neutral")
        self.assertEqual(kinds["driver"].subject, "North")
        self.assertNotIn("North", brief.recommendations[0].title)


class ChangeWordingTests(unittest.TestCase):
    def test_a_small_change_never_reads_from_x_to_x(self):
        frame = pd.DataFrame(
            {"Month": MONTHS, "Units": [1_150.0, 1_160.0, 1_175.0, 1_180.0, 1_190.0, 1_195.0,
                                        1_200.0, 1_204.0, 1_206.0, 1_208.0, 1_211.0, 1_214.0]}
        )

        trend = next(item for item in analyze_business(frame).evidence if item.kind == "trend")

        self.assertIn("from 1,211 to 1,214", trend.statement)


class EvidenceBudgetTests(unittest.TestCase):
    def test_a_quality_card_is_never_squeezed_out_by_the_card_cap(self):
        filler = [
            Evidence(kind=f"k{i}", title="t", value="v", statement="s", calculation="c") for i in range(9)
        ]
        quality = Evidence(kind="quality", title="q", value="v", statement="s", calculation="c")

        shown = _shown_evidence(filler + [quality])

        self.assertEqual(len(shown), 8)
        self.assertIn(quality, shown)


class ColumnNameCollisionTests(unittest.TestCase):
    def test_a_measure_named_period_still_gets_a_brief(self):
        frame = pd.DataFrame(
            {"Date": pd.date_range("2024-01-01", periods=60), "Period": np.linspace(1, 9, 60),
             "Region": ["a", "b"] * 30}
        )

        prepared = prepare_analysis(frame, row_limit=1000)

        self.assertTrue(prepared.analyze().headline)


def site_temperatures() -> pd.DataFrame:
    """Two sites over six months; the last month is 2 °C colder at North."""
    months = pd.date_range("2024-01-01", periods=6, freq="MS")
    rows = []
    for index, month in enumerate(months):
        rows.append((month, "North", 14.0 - (2.0 if index == 5 else 0.0)))
        rows.append((month, "South", 12.0))
    return pd.DataFrame(rows, columns=["Survey Date", "Site", "Water Temp (°C)"])


class LevelMeasureTests(unittest.TestCase):
    def setUp(self):
        self.brief = analyze_business(site_temperatures())
        self.evidence = {item.kind: item for item in self.brief.evidence}

    def test_the_change_is_a_difference_not_a_percentage(self):
        trend = self.evidence["trend"]
        self.assertEqual(trend.value, "-1")
        self.assertIn("decreased by 1 in the latest complete period", trend.statement)
        self.assertIn("from 13 to 12", trend.statement)
        self.assertNotIn("%", trend.value)

    def test_segments_are_compared_by_average_without_shares(self):
        leader = self.evidence["leader"]
        self.assertIn("Water Temp (°C) is higher in North than in South: mean", leader.statement)
        self.assertNotIn("% of", leader.statement)
        self.assertNotIn("concentration", self.evidence)

    def test_the_driver_quotes_no_share_of_the_net_change(self):
        self.assertNotIn("net movement", self.evidence["driver"].statement)

    def test_no_total_kpi_for_a_level(self):
        labels = [kpi.label for kpi in self.brief.kpis]
        self.assertNotIn("Total Water Temp (°C)", labels)
        self.assertIn("Average Water Temp (°C)", labels)

    def test_recommendations_use_no_business_vocabulary(self):
        text = " ".join(f"{item.title} {item.action}" for item in self.brief.recommendations).lower()
        for word in ("pricing", "customer", "channel", "revenue", "playbook"):
            self.assertNotIn(word, text)


class RateDriverTests(unittest.TestCase):
    def test_a_rate_driver_moves_in_points(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        rows = []
        for index, month in enumerate(months):
            rows += [(month, "A", 20.0 + (3.0 if index == 5 else 0.0)), (month, "B", 10.0)]
        brief = analyze_business(pd.DataFrame(rows, columns=["Month", "Team", "Conversion Rate"]))
        driver = next(item for item in brief.evidence if item.kind == "driver")
        self.assertEqual(driver.value, "+3.0 pts")
        self.assertIn("increased by 3.0 percentage points", driver.statement)


class DriverToneTests(unittest.TestCase):
    def test_a_rising_cost_driver_is_named_in_the_recommendation(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        rows = []
        for index, month in enumerate(months):
            rows += [(month, "Ops", 100.0 + (60.0 if index == 5 else 0.0)), (month, "Sales", 50.0)]
        brief = analyze_business(pd.DataFrame(rows, columns=["Month", "Team", "Cost"]))
        self.assertEqual(brief.recommendations[0].title, "Start with Ops")


class ShareDenominatorTests(unittest.TestCase):
    def test_shares_use_every_segment(self):
        frame = pd.DataFrame({"Store": [f"S{i:03d}" for i in range(150)], "Revenue": [1.0] * 150})
        roles = ColumnRoles(date=None, measure="Revenue", dimension="Store", identifier=None,
                            numeric=("Revenue",), dimensions=("Store",))
        leader = next(item for item in analyze_business(frame, roles).evidence if item.kind == "leader")
        self.assertEqual(leader.value, "0.7%")


class NegativeLevelTests(unittest.TestCase):
    def test_the_highest_average_is_the_largest_value_not_the_largest_size(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        rows = [
            (month, site, temperature)
            for month in months
            for site, temperature in (("Arctic", -20.0), ("Coast", 2.0), ("Desert", 12.0))
        ]
        brief = analyze_business(pd.DataFrame(rows, columns=["Date", "Site", "Temperature (°C)"]))
        leader = next(item for item in brief.evidence if item.kind == "leader")
        self.assertIn("Desert has the highest mean, 12, and Arctic the lowest, -20", leader.statement)
        top = next(kpi for kpi in brief.kpis if kpi.label == "Highest average by site")
        self.assertEqual((top.value, top.context), ("12", "Desert"))

    def test_a_mixed_sign_leader_is_described_by_size(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        rows = [(month, region, value) for month in months
                for region, value in (("North", -5000.0), ("South", 3000.0), ("East", 1000.0))]
        brief = analyze_business(pd.DataFrame(rows, columns=["Date", "Region", "Profit"]))
        self.assertIn("North is the largest by size", brief.headline)
        top = next(kpi for kpi in brief.kpis if kpi.label == "Largest region by size")
        self.assertEqual(top.context, "North")


class FixedFrequencyTests(unittest.TestCase):
    def setUp(self):
        days = pd.date_range("2022-01-01", "2024-12-31", freq="D")
        values = 100 + np.random.default_rng(0).normal(0, 5, len(days))
        values[(days >= "2023-06-01") & (days < "2023-07-01")] = 300
        self.frame = pd.DataFrame({
            "Date": days,
            "Site": ["A", "B"] * (len(days) // 2) + ["A"] * (len(days) % 2),
            "Rainfall": values,
        })
        self.roles = detect_roles(self.frame)

    def evidence(self, frequency=None) -> dict[str, Evidence]:
        brief = analyze_business(self.frame, self.roles, frequency=frequency)
        return {item.kind: item for item in brief.evidence}

    def test_three_years_of_days_default_to_quarters(self):
        evidence = self.evidence()
        self.assertIn("(Q4 2024)", evidence["trend"].statement)
        self.assertTrue(evidence["anomaly"].statement.startswith("Q2 2023"))

    def test_a_fixed_frequency_sets_the_grain_of_every_period_finding(self):
        evidence = self.evidence("M")
        self.assertIn("(Dec 2024)", evidence["trend"].statement)
        self.assertTrue(evidence["anomaly"].statement.startswith("Jun 2023"))
        # The driver compares the same two months as the trend, not two quarters.
        grouped, previous, current = segment_period_change(self.frame, self.roles, "M")
        self.assertEqual((previous, current), (pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-01")))
        self.assertEqual(evidence["driver"].subject, str(grouped["Change"].abs().idxmax()))
        self.assertNotEqual(evidence["driver"].subject, self.evidence()["driver"].subject)


def relationship_of(frame: pd.DataFrame) -> Evidence | None:
    brief = analyze_business(frame, detect_roles(frame))
    return next((item for item in brief.evidence if item.kind == "relationship"), None)


class RelationshipCandidateTests(unittest.TestCase):
    def test_a_numbered_participant_is_not_correlated_with_their_score(self):
        rng = np.random.default_rng(7)
        subjects = np.repeat(np.arange(1, 21), 3)
        frame = pd.DataFrame({
            "subject": subjects,
            "attention": ["divided", "focused"] * 30,
            "score": subjects * 0.3 + rng.normal(0, 1, 60),
            "solutions": rng.integers(1, 4, 60),
        })

        relationship = relationship_of(frame)

        self.assertTrue(relationship is None or "subject" not in relationship.statement)

    def test_a_plate_column_is_not_correlated_with_the_dilution_laid_out_across_it(self):
        rng = np.random.default_rng(11)
        columns = np.tile(np.arange(1, 13), 8)
        dose = 100 / 2.0 ** (columns - 1)
        frame = pd.DataFrame({
            "Column": columns,
            "Dose (µM)": dose,
            "Abs 450nm": 0.2 + dose / 100 + rng.normal(0, 0.05, 96),
        })

        relationship = relationship_of(frame)

        assert relationship is not None
        self.assertNotIn("Column", relationship.statement)
        self.assertTrue(relationship.statement.startswith("Abs 450nm and Dose (µM) move together"))

    def test_a_derived_copy_is_named_but_not_reported_as_the_relationship(self):
        rng = np.random.default_rng(5)
        co2 = np.linspace(315, 410, 120) + rng.normal(0, 0.5, 120)
        frame = pd.DataFrame({
            "CO2": co2,
            "adjusted CO2": co2 - 0.3,
            "Temperature anomaly": (co2 - 315) / 100 + rng.normal(0, 0.15, 120),
        })

        relationship = relationship_of(frame)

        assert relationship is not None
        self.assertNotIn("CO2 and adjusted CO2 move together:", relationship.statement)
        self.assertIn(
            "CO2 and adjusted CO2 (r = 1.00) were left out: they move together almost exactly",
            relationship.statement,
        )
        self.assertIn("treat it as a lead to check", relationship.statement)
        self.assertIn("near-duplicates", relationship.calculation)

    def test_no_card_when_the_only_strong_pair_is_a_near_duplicate(self):
        values = np.linspace(1.0, 50.0, 40)
        frame = pd.DataFrame({"x": values, "y": values * 1.001 + 0.01})

        self.assertIsNone(relationship_of(frame))

    def test_an_ordinary_relationship_gets_no_near_duplicate_sentence(self):
        rng = np.random.default_rng(3)
        units = rng.normal(50, 10, 200)
        frame = pd.DataFrame({"Units": units, "Revenue": units * 20 + rng.normal(0, 150, 200)})

        relationship = relationship_of(frame)

        assert relationship is not None
        self.assertNotIn("left out", relationship.statement)


def roles_for(frame: pd.DataFrame, measure: str | None, dimension: str | None, **extra) -> ColumnRoles:
    numeric = tuple(column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column]))
    return ColumnRoles(
        date=extra.pop("date", None), measure=measure, dimension=dimension,
        identifier=extra.pop("identifier", None), numeric=numeric,
        dimensions=(dimension,) if dimension else (), **extra,
    )


@dataclass(frozen=True)
class RolesWithAlternatives(ColumnRoles):
    measure_alternatives: tuple[str, ...] = ()


class GroupWordingTests(unittest.TestCase):
    def leader(self, frame: pd.DataFrame, roles: ColumnRoles) -> Evidence:
        return next(item for item in analyze_business(frame, roles).evidence if item.kind == "leader")

    def test_a_numeric_group_is_named_with_its_column(self):
        frame = pd.DataFrame({"Status": [0, 1, 2] * 20, "MQ139": [104.0, 120.0, 309.0] * 20})
        roles = roles_for(frame, "MQ139", "Status", factors=("Status",))

        brief = analyze_business(frame, roles)
        leader = self.leader(frame, roles)

        self.assertIn("Status 2 has the highest mean", leader.statement)
        self.assertIn("Status 0 the lowest", leader.statement)
        self.assertIn("Status 2 has", brief.headline)
        self.assertEqual(leader.title, "Highest average by status")

    def test_tied_groups_are_all_named(self):
        frame = pd.DataFrame({"Bird": ["47", "68", "46", "70", "12"] * 8,
                              "Approach": [3.0, 3.0, 2.0, 2.0, 2.5] * 8})
        leader = self.leader(frame, roles_for(frame, "Approach", "Bird"))

        self.assertIn("47 and 68 have the highest mean", leader.statement)
        self.assertIn("46 and 70 the lowest", leader.statement)
        self.assertEqual(leader.subject, "47 (tied with 1 other)")

    def test_averages_that_round_alike_print_apart(self):
        frame = pd.DataFrame({"Crop": ["Maize", "Rice"] * 30, "Yield": [3_004.25, 3_004.21] * 30})
        leader = self.leader(frame, roles_for(frame, "Yield", "Crop"))

        self.assertIn("3,004.25 vs 3,004.21", leader.statement)

    def test_groups_with_few_rows_say_so(self):
        frame = pd.DataFrame({"Gp": ["T0", "D6", "D3"] * 4, "Ca": [13.4, 4.8, 8.0] * 4})
        leader = self.leader(frame, roles_for(frame, "Ca", "Gp"))

        self.assertIn("13.40 (T0, n = 4)", leader.statement)
        self.assertIn("too few to call the difference clear", leader.statement)
        self.assertIsNone(leader.subject)

    def test_well_filled_groups_quote_no_row_counts(self):
        frame = pd.DataFrame({"Site": ["A", "B"] * 40, "pH": [7.1, 6.9] * 40})
        leader = self.leader(frame, roles_for(frame, "pH", "Site"))

        self.assertNotIn("rows", leader.statement)

    def test_no_card_title_or_tile_uses_sales_words(self):
        frame = pd.DataFrame({"condition": ["a", "b", "c"] * 30, "realism": [5.6, 5.5, 5.4] * 30})
        brief = analyze_business(frame, roles_for(frame, "realism", "condition"))

        text = " ".join([item.title for item in brief.evidence] + [kpi.label for kpi in brief.kpis]).lower()
        for word in ("leading", "top ", "leader", "segment"):
            self.assertNotIn(word, text)

    def test_counted_rows_are_called_rows(self):
        frame = pd.DataFrame({"condition": ["Employee_Low"] * 69 + ["Contractor_High"] * 50})
        leader = self.leader(frame, roles_for(frame, None, "condition"))

        self.assertEqual(leader.title, "Most rows by condition")
        self.assertIn("Employee_Low has the most rows: 69 of 119", leader.statement)


class RowCountTests(unittest.TestCase):
    def test_a_driver_resting_on_few_rows_says_so(self):
        dates = [pd.Timestamp("2025-01-06")] * 180 + [pd.Timestamp("2025-01-13")] * 12
        frame = pd.DataFrame({
            "Date": dates * 2,
            "Material": ["SEA WATER"] * 192 + ["RIVER WATER"] * 192,
            "result": [15_684.0] * 180 + [70_718.0] * 12 + [5.0] * 192,
        })
        roles = roles_for(frame, "result", "Material", date="Date")

        driver = next(item for item in analyze_business(frame, roles).evidence if item.kind == "driver")

        self.assertTrue(driver.statement.startswith("SEA WATER moved the most"), driver.statement)
        self.assertIn("from 15,684 to 70,718", driver.statement)
        self.assertIn("the average of 180 rows and the latest of 12 rows", driver.statement)

    def test_a_total_quotes_no_row_counts(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        frame = pd.DataFrame({"Month": list(months) * 2, "Region": ["N"] * 6 + ["S"] * 6,
                              "Revenue": [100.0, 110, 120, 130, 140, 200] * 2})
        brief = analyze_business(frame, roles_for(frame, "Revenue", "Region", date="Month"))

        for item in brief.evidence:
            self.assertNotIn("average of", item.statement)


class ConcentrationScopeTests(unittest.TestCase):
    def test_a_summed_reading_gets_no_concentration_card(self):
        frame = pd.DataFrame({"Station": ["63rd", "Oak", "Foster"] * 20,
                              "Total Rain": [184.7, 90.2, 0.4] * 20})
        brief = analyze_business(frame, roles_for(frame, "Total Rain", "Station"))

        self.assertNotIn("concentration", [item.kind for item in brief.evidence])
        self.assertIn("Total Rain", [kpi.label for kpi in brief.kpis])
        self.assertNotIn("Total Total Rain", [kpi.label for kpi in brief.kpis])

    def test_money_and_counts_keep_it_under_a_neutral_title(self):
        for measure, values in (("Revenue", [900.5, 10.25, 10.0]), ("Orders", [90.0, 10.0, 10.0])):
            frame = pd.DataFrame({"Product": ["A", "B", "C"] * 20, measure: values * 20})
            brief = analyze_business(frame, roles_for(frame, measure, "Product"))
            concentration = next(item for item in brief.evidence if item.kind == "concentration")
            self.assertEqual(concentration.title, "Concentration by product")


class SummaryTests(unittest.TestCase):
    def test_a_table_without_a_trend_is_described_not_praised(self):
        frame = pd.DataFrame({"Reading ID": range(1, 41), "Strength (MPa)": np.linspace(10, 60, 40)})
        brief = analyze_business(frame, roles_for(frame, "Strength (MPa)", None, identifier="Reading ID"))

        self.assertNotIn("clearest available signals", brief.summary)
        self.assertEqual(
            brief.summary,
            "40 rows, one per Reading ID. Measure: Strength (MPa), averaged across rows. "
            "No date or grouping column was found.",
        )

    def test_a_row_counter_is_not_called_entities(self):
        frame = pd.DataFrame({"Reading ID": range(1, 41), "MQ139": np.linspace(100, 300, 40)})
        brief = analyze_business(frame, roles_for(frame, "MQ139", None, identifier="Reading ID"))

        rows = next(kpi for kpi in brief.kpis if kpi.label == "Rows analyzed")
        self.assertEqual(rows.context, "One per Reading ID")
        self.assertFalse([kpi for kpi in brief.kpis if kpi.label.startswith("Distinct")])

    def test_a_close_measure_pick_is_named(self):
        frame = pd.DataFrame({"realism": [5.0, 6.0] * 20, "risky_decision": [0.4, 0.6] * 20,
                              "risk_perception": [3.0, 4.0] * 20})
        roles = RolesWithAlternatives(
            date=None, measure="realism", dimension=None, identifier=None,
            numeric=tuple(frame.columns), dimensions=(),
            measure_alternatives=("risky_decision", "risk_perception"),
        )

        summary = analyze_business(frame, roles).summary

        self.assertIn(
            "Measure picked: realism (also plausible: risky_decision and risk_perception). "
            'Change it with measure="risky_decision" or in the app.',
            summary,
        )
        self.assertNotIn("Measure: realism", summary)

    def test_no_close_call_note_without_runners_up(self):
        frame = pd.DataFrame({"realism": [5.0, 6.0] * 20})
        self.assertNotIn("Measure picked", analyze_business(frame, roles_for(frame, "realism", None)).summary)

    def test_a_coded_band_is_not_printed_as_money(self):
        frame = pd.DataFrame({"income": [1.0, 5.0, 12.0, 15.0] * 10})
        brief = analyze_business(frame, roles_for(frame, "income", None))

        for kpi in brief.kpis:
            self.assertNotIn("$", kpi.value)
        self.assertNotIn("$", brief.headline)


def evidence_of(frame, kind, roles=None):
    brief = analyze_business(frame, roles or detect_roles(frame))
    return next((item for item in brief.evidence if item.kind == kind), None)


def stations_by_quarter():
    """Three stations a quarter for six years; the warm South stops after Q3 1945."""
    quarters = pd.date_range("1940-01-01", periods=24, freq="QS")
    season = np.tile([5.0, 15.0, 25.0, 15.0], 6)
    rows = []
    for station, offset in (("North", 0.0), ("Middle", 2.0), ("South", 20.0)):
        for quarter, level in zip(quarters, season, strict=True):
            if station == "South" and quarter > pd.Timestamp("1945-07-01"):
                continue
            rows.append((quarter, station, level + offset))
    return pd.DataFrame(rows, columns=["Date", "Station", "MaxTemp"])


class PeriodComparisonTests(unittest.TestCase):
    def test_a_station_that_stops_reporting_does_not_move_the_change(self):
        trend = evidence_of(stations_by_quarter(), "trend")

        self.assertIn("compares the 2 station values that reported in both periods", trend.statement)
        self.assertIn("South reported in Q3 1945 but not in Q4 1945", trend.statement)
        self.assertIn("With every station included it went from", trend.statement)
        self.assertIn("decreased by 10 ", trend.statement)
        self.assertIn("within the range usual for the time of year", trend.statement)
        self.assertIn("Against Q4 1944 (16), it is level", trend.statement)
        self.assertTrue(trend.calculation.startswith("On the Station values present in both periods"))

    def test_a_winter_to_spring_move_is_set_beside_earlier_years(self):
        frame = stations_by_quarter()
        frame = frame[frame["Station"] == "North"]

        trend = evidence_of(frame, "trend")

        self.assertIn("the move into Q4 ranged from -10 to -10", trend.statement)
        self.assertIn("within the range usual for the time of year", trend.statement)
        self.assertIn("Against Q4 1944 (15), it is level", trend.statement)
        self.assertNotIn("unusually large swing", trend.statement)

    def test_a_short_last_period_is_not_called_complete(self):
        stamps = pd.date_range("2024-09-02", "2025-01-04 23:00", freq="h")
        frame = pd.DataFrame({"Time": stamps, "CO": 400.0 + np.arange(len(stamps)) % 7})

        brief = analyze_business(frame, detect_roles(frame), frequency="W")
        trend = next(item for item in brief.evidence if item.kind == "trend")

        self.assertIn("in the latest period (30 Dec 2024)", trend.statement)
        self.assertIn("only reaches 6 of 7 days", trend.statement)


class GroupCoverageTests(unittest.TestCase):
    def test_a_group_with_no_readings_is_no_data_not_zero(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        frame = pd.DataFrame({
            "Date": list(months) * 3,
            "Station": ["Oak"] * 6 + ["63rd"] * 6 + ["Foster"] * 6,
            "Rainfall Total": [5.0] * 6 + [9.0] * 6 + [np.nan] * 6,
        })

        leader = evidence_of(frame, "leader")
        segments = segment_frame(frame, detect_roles(frame), limit=None).set_index("Segment")

        self.assertIn("Foster has no Rainfall Total values", leader.statement)
        self.assertTrue(np.isnan(segments.loc["Foster", "Value"]))

    def test_a_ranked_group_covering_part_of_the_dates_says_so(self):
        months = pd.date_range("2015-01-01", periods=60, freq="MS")
        frame = pd.DataFrame({
            "Date": list(months) * 2,
            "City": ["Delhi"] * 60 + ["Aizawl"] * 60,
            "PM2.5": [90.0] * 60 + [np.nan] * 54 + [20.0] * 6,
        })

        leader = evidence_of(frame, "leader")

        self.assertIn("Aizawl only has figures from 01 Jul 2019 to 01 Dec 2019", leader.statement)
        self.assertNotIn("Delhi only has", leader.statement)

    def test_the_only_group_in_both_periods_gets_no_share(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        frame = pd.DataFrame({
            "Date": list(months) + list(months[:4]),
            "Station": ["Oak"] * 6 + ["63rd"] * 4,
            "Rain Total": [5.0, 6.0, 7.0, 8.0, 9.0, 3.0, 9.0, 9.0, 9.0, 9.0],
        })

        driver = evidence_of(frame, "driver")

        self.assertIn("It is the only station with figures in both periods.", driver.statement)
        self.assertNotIn("100.0% of the net movement", driver.statement)


def patients(size: int = 300, seed: int = 0) -> tuple[pd.DataFrame, ColumnRoles]:
    """Patients by test result: a marker that separates the results, a sugar level that doesn't."""
    rng = np.random.default_rng(seed)
    result = np.where(np.arange(size) % 5 < 3, "positive", "negative")
    frame = pd.DataFrame({
        "Age": rng.normal(55, 12, size) + np.where(result == "positive", 6, 0),
        "Sugar": rng.normal(145, 70, size),
        "Marker": rng.lognormal(-4, 1.2, size) * np.where(result == "positive", 8, 1),
        "Sex": rng.integers(0, 2, size),
        "Result": result,
    })
    roles = ColumnRoles(
        date=None, measure="Sugar", dimension="Result", identifier=None,
        numeric=("Age", "Sugar", "Marker", "Sex"), dimensions=("Result", "Sex"), factors=("Sex",),
    )
    return frame, roles


class GroupDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.roles = patients()
        self.brief = analyze_business(self.frame, self.roles)
        self.evidence = {item.kind: item for item in self.brief.evidence}

    def test_a_gap_within_the_noise_is_not_presented_as_a_ranking(self):
        leader = self.evidence["leader"]
        self.assertEqual(leader.title, "No clear difference by result")
        self.assertIn("Sugar shows no clear difference between", leader.statement)
        self.assertNotIn("highest", leader.statement)
        self.assertIsNone(leader.subject)
        self.assertFalse(any(kpi.label.startswith("Highest") for kpi in self.brief.kpis))

    def test_the_reading_that_does_differ_is_found_and_headlined(self):
        differences = self.evidence["differences"]
        self.assertEqual(differences.title, "What differs between groups")
        self.assertTrue(differences.statement.startswith("Marker is higher in positive than in negative"))
        self.assertIn("Holm-adjusted", differences.calculation)
        self.assertIn("Sugar by Result is compared on its own card", differences.calculation)
        self.assertTrue(
            self.brief.headline.startswith(
                "Largest difference found: Marker is higher in positive than in negative"
            )
        )

    def test_a_factor_is_a_grouping_not_a_reading(self):
        differences = self.evidence["differences"]
        self.assertIn("across Result and Sex", differences.statement + differences.calculation)
        self.assertNotIn("Sex by", differences.statement)

    def test_a_clear_difference_in_the_measure_leads_the_card(self):
        roles = ColumnRoles(
            date=None, measure="Age", dimension="Result", identifier=None,
            numeric=("Age", "Sugar", "Marker", "Sex"), dimensions=("Result",),
        )
        brief = analyze_business(self.frame, roles)
        leader = next(item for item in brief.evidence if item.kind == "leader")
        self.assertEqual(leader.title, "Highest average by result")
        self.assertEqual(leader.subject, "positive")
        self.assertIn("Age is higher in positive than in negative", brief.headline)
        top = next(kpi for kpi in brief.kpis if kpi.label == "Highest average by result")
        self.assertEqual(top.context, "positive")

    def test_a_nothing_found_scan_is_reported_but_ranks_below_the_findings(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({
            "Crop": rng.choice(["Maize", "Rice", "Wheat"], 900),
            **{name: rng.normal(size=900) for name in ("Yield", "NDVI", "pH", "Rain")},
        })
        roles = ColumnRoles(date=None, measure="Yield", dimension="Crop", identifier=None,
                            numeric=("Yield", "NDVI", "pH", "Rain"), dimensions=("Crop",))
        brief = analyze_business(frame, roles)
        kinds = [item.kind for item in brief.evidence]
        differences = brief.evidence[kinds.index("differences")]
        self.assertEqual(differences.value, "0 of 3")
        self.assertIn("None of the 3 comparisons across Crop shows a clear difference", differences.statement)
        self.assertGreater(kinds.index("differences"), kinds.index("relationship"))

    def test_several_groupings_get_their_share_of_the_variance(self):
        rng = np.random.default_rng(2)
        depth = np.repeat(["0-10", "30-60"], 30)
        frame = pd.DataFrame({
            "Depth": depth,
            "Contour": np.tile(["Top", "Slope", "Foot"], 20),
            "Ca": np.where(depth == "0-10", 12.0, 5.0) + rng.normal(0, 1, 60),
            "N": rng.normal(size=60),
        })
        roles = ColumnRoles(date=None, measure="Ca", dimension="Depth", identifier=None,
                            numeric=("Ca", "N"), dimensions=("Depth", "Contour"))
        differences = next(
            item for item in analyze_business(frame, roles).evidence if item.kind == "differences"
        )
        self.assertIn("Of the variance in Ca, Depth explains", differences.statement)


def card(frame: pd.DataFrame, roles: ColumnRoles, kind: str) -> Evidence:
    return next(item for item in analyze_business(frame, roles).evidence if item.kind == kind)


class RelationshipScanCardTests(unittest.TestCase):
    def test_what_relates_to_the_measure_comes_first(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame({"Cement": rng.normal(300, 50, 200), "Water": rng.normal(180, 20, 200)})
        frame["Plasticizer"] = 30 - frame["Water"] / 10 + rng.normal(0, 1, 200)
        frame["Strength"] = frame["Cement"] / 10 + rng.normal(0, 4, 200)
        roles = ColumnRoles(date=None, measure="Strength", dimension=None, identifier=None,
                            numeric=tuple(frame.columns), dimensions=())

        relationship = card(frame, roles, "relationship")

        self.assertEqual(relationship.title, "What relates to Strength")
        self.assertTrue(relationship.statement.startswith("Strength and Cement move together"))
        self.assertIn(
            "Among the other columns, the strongest pair: Water and Plasticizer", relationship.statement
        )
        self.assertIn("3 correlations with Strength and 6 pairs scanned in all", relationship.calculation)

    def test_a_total_and_its_items_are_left_out_and_listed(self):
        rng = np.random.default_rng(4)
        items = pd.DataFrame(
            rng.uniform(0, 20, (40, 4)).round(2), columns=["Correct1", "Correct2", "Correct3", "Correct4"]
        )
        frame = items.assign(CorrectTotal=items.sum(axis=1), Looking=rng.normal(10, 2, 40))
        roles = ColumnRoles(date=None, measure="CorrectTotal", dimension=None, identifier=None,
                            numeric=tuple(frame.columns), dimensions=())

        relationship = card(frame, roles, "relationship")

        self.assertNotIn("Correct3 relates most", relationship.statement)
        self.assertIn("CorrectTotal = Correct1 + Correct2 + Correct3 + Correct4", relationship.calculation)

    def test_a_relationship_resting_on_one_record_names_it(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"HCT": rng.normal(42, 4, 300), "MCH": rng.normal(29, 2, 300)})
        frame.loc[7, ["HCT", "MCH"]] = [3_715.0, 3_117.0]
        frame["PLT"] = rng.normal(250, 50, 300)
        roles = ColumnRoles(date=None, measure="HCT", dimension=None, identifier=None,
                            numeric=("HCT", "MCH", "PLT"), dimensions=())
        correlation = measure_correlation(frame[["HCT", "MCH"]])

        assert correlation is not None
        self.assertTrue(correlation.rests_on_one_record)
        statement = describe_correlation(correlation)
        # The figure that one record does not decide comes first.
        self.assertTrue(statement.startswith("HCT and MCH move together only weakly once one record is"))
        self.assertIn("It rests on one record: with the one where HCT is 3,715 and MCH is 3,117", statement)
        relationship = card(frame, roles, "relationship")
        self.assertEqual(relationship.value, "|r| < 0.3")

    def test_a_correlation_that_reverses_within_groups_says_so(self):
        rng = np.random.default_rng(6)
        parts = []
        for offset in range(4):
            dose = rng.uniform(0, 10, 50)
            parts.append(pd.DataFrame({
                "Dye": f"dye {offset}",
                "Dose": dose + offset * 10,
                "R": 100 + offset * 40 - 3 * dose + rng.normal(0, 2, 50),
            }))
        frame = pd.concat(parts, ignore_index=True)
        roles = ColumnRoles(date=None, measure="R", dimension="Dye", identifier=None,
                            numeric=("Dose", "R"), dimensions=("Dye",))

        relationship = card(frame, roles, "relationship")

        self.assertIn("Within each dye the relationship runs the other way", relationship.statement)

    def test_a_null_result_is_reported_rather_than_dropped(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(size=(2_000, 6)), columns=list("abcdef"))
        roles = ColumnRoles(date=None, measure="a", dimension=None, identifier=None,
                            numeric=tuple(frame.columns), dimensions=())

        relationship = card(frame, roles, "relationship")

        self.assertEqual(relationship.value, "|r| < 0.3")
        self.assertIn("None of the 5 columns tested against a reaches |r| = 0.3", relationship.statement)
        self.assertIn("No other pair reaches |r| = 0.3", relationship.statement)
        self.assertIn("worth a check of how its values were produced", relationship.statement)

    def test_a_yes_no_flag_is_not_a_reading(self):
        frame, roles = patients()
        roles = ColumnRoles(date=None, measure="Age", dimension=None, identifier=None,
                            numeric=("Age", "Sugar", "Marker", "Sex"), dimensions=())

        self.assertNotIn("Sex", numeric_readings(frame, roles))

    def test_directions_coordinates_and_an_age_in_months(self):
        frame = pd.DataFrame({
            "Age_months": [12 + index % 55 for index in range(110)],
            "Wind Direction": [10, 350, 180, 90, 270] * 22,
            "easting": [300_000 + index * 37 % 500 for index in range(110)],
            "Shannon_Index": np.linspace(2.5, 3.2, 110),
        })

        readings = numeric_readings(frame, detect_roles(frame))

        self.assertEqual(sorted(readings), ["Age_months", "Shannon_Index"])

    def test_the_main_grouping_is_reported_when_the_pair_is_stronger_within_it(self):
        rng = np.random.default_rng(60)
        frame = pd.DataFrame({"Log": np.repeat(["first", "second"], 300), "PM10": rng.normal(size=600)})
        frame["PM25"] = frame["PM10"] + rng.normal(0, 0.4, 600)
        frame.loc[frame["Log"] == "second", "PM10"] += 3.0
        roles = ColumnRoles(date=None, measure="PM10", dimension="Log", identifier=None,
                            numeric=("PM10", "PM25"), dimensions=("Log",))

        relationship = card(frame, roles, "relationship")

        stronger = "Within each Log it is stronger: r = 0.92 in first and 0.92 in second"
        self.assertIn(stronger, relationship.statement)

    def test_a_row_mean_is_not_related_to_the_columns_it_averages(self):
        rng = np.random.default_rng(61)
        counts = rng.poisson(5, (300, 24)).astype(float)
        frame = pd.DataFrame(counts, columns=[f"S{index}" for index in range(24)])
        frame.insert(0, "Taxon", [f"t{index}" for index in range(300)])
        prepared = prepare_analysis(frame, row_limit=5000)

        relationship = card(prepared.dataframe, prepared.detected_roles, "relationship")

        self.assertEqual(relationship.title, "Strongest measurable relationship")
        self.assertIn("is part of it", relationship.statement)

    def test_a_row_mean_of_counts_named_by_sample_is_not_modelled_on_its_own_columns(self):
        samples = ["CL3", "CC1", "SV1", "M31Fcsw", "M11Tong", "AQC1cm", "AQC4cm", "NP2", "TRRsed1", "TS28",
                   "Even1", "Even2"]
        rng = np.random.default_rng(64)
        frame = pd.DataFrame(rng.poisson(30, (300, len(samples))).astype(float), columns=samples)
        frame.insert(0, "OTU", [f"OTU{549000 + index}" for index in range(300)])
        prepared = prepare_analysis(frame, row_limit=5000)

        brief = analyze_business(prepared.dataframe, prepared.detected_roles)

        self.assertEqual(prepared.detected_roles.measure, "Row mean (12 columns)")
        self.assertNotIn("model", [item.kind for item in brief.evidence])
        relationship = next(item for item in brief.evidence if item.kind == "relationship")
        self.assertIn("is part of it", relationship.statement)

    def test_shares_that_break_their_total_are_named(self):
        rng = np.random.default_rng(62)
        shares = (rng.dirichlet([4, 3, 2, 1], 400) * 100).round(1)
        frame = pd.DataFrame(shares, columns=["Bact", "Firm", "Prot", "Other"])
        frame["Other"] = (100 - frame[["Bact", "Firm", "Prot"]].sum(axis=1)).round(1)
        frame.loc[:24, "Other"] = 0.0
        frame.loc[:24, "Bact"] += 50.0
        frame["Weight"] = rng.normal(400, 50, 400)
        roles = ColumnRoles(date=None, measure="Weight", dimension=None, identifier=None,
                            numeric=tuple(frame.columns), dimensions=())

        relationship = card(frame, roles, "relationship")

        self.assertIn("On 25 rows, Bact, Firm, Prot and Other add up to", relationship.statement)
        self.assertNotIn("Bact and Other move", relationship.statement)


class CorrelationWordingTests(unittest.TestCase):
    def test_one_record_hiding_a_relationship_leads_with_the_figure_without_it(self):
        rng = np.random.default_rng(63)
        frame = pd.DataFrame({"RBC": rng.normal(4.5, 0.5, 400)})
        frame["HGB"] = frame["RBC"] * 3 + rng.normal(0, 0.6, 400)
        frame.loc[3, "RBC"] = 90.8

        statement = describe_correlation(measure_correlation(frame))

        self.assertTrue(statement.startswith("RBC and HGB move together once one record is set aside"))
        self.assertIn("One record hides a strong relationship: with the one where RBC is 90.80", statement)
        self.assertNotIn("rests on", statement)

    def test_extreme_records_hiding_a_line_lead_with_r_without_them(self):
        rng = np.random.default_rng(64)
        frame = pd.DataFrame({"MCV": rng.normal(85, 8, 500)})
        frame["MCH"] = frame["MCV"] * 0.33 + rng.normal(0, 1, 500)
        frame.loc[:7, "MCH"] = rng.uniform(2_000, 3_000, 8)

        statement = describe_correlation(measure_correlation(frame))

        self.assertTrue(statement.startswith("MCV and MCH move together once 8 extreme records are set"))
        self.assertIn("with them, Pearson r is only", statement)

    def test_a_long_stretch_and_a_floor_are_named_instead_of_a_few_extremes(self):
        rng = np.random.default_rng(65)
        quiet = rng.uniform(0, 100, 2_000)
        fire = rng.uniform(5_000, 40_000, 400)
        floor = rng.random(2_000) < 0.3
        eco2 = np.r_[np.where(floor, 400.0, 400 + quiet * rng.uniform(0, 3, 2_000)), fire * 1.5]
        frame = pd.DataFrame({"TVOC": np.r_[quiet, fire], "eCO2": eco2})

        statement = describe_correlation(measure_correlation(frame))

        self.assertIn("by rank: the rank correlation is only", statement)
        self.assertIn("one unbroken stretch of", statement)
        self.assertIn("eCO2 sits at its minimum of 400", statement)
        self.assertNotIn("few extreme records", statement)


class ConstructionCardTests(unittest.TestCase):
    def test_a_grouping_set_by_a_reading_is_stated_and_not_compared(self):
        rng = np.random.default_rng(66)
        sex = rng.integers(0, 2, 800)
        hemoglobin = rng.normal(13, 2, 800).round(1)
        frame = pd.DataFrame({
            "Gender": sex,
            "Hemoglobin": hemoglobin,
            "MCV": rng.normal(85, 5, 800),
            "Result": np.where(hemoglobin < np.where(sex == 0, 12.0, 13.5), 1, 0),
        })
        roles = ColumnRoles(date=None, measure="MCV", dimension="Result", identifier=None,
                            numeric=tuple(frame.columns), dimensions=("Gender", "Result"),
                            factors=("Gender", "Result"), flags=("Gender", "Result"))

        brief = analyze_business(frame, roles)
        differences = brief_card(brief, "differences")

        assert differences is not None
        self.assertIn("Result is set by Hemoglobin: Result 1 below 12.0 for Gender 0", differences.statement)
        self.assertIn("left out as set by the reading: Hemoglobin by Result", differences.calculation)
        self.assertNotIn("Hemoglobin is higher", brief.headline)

    def test_no_group_ranking_when_the_grouping_is_cut_from_the_measure(self):
        rng = np.random.default_rng(67)
        acres = rng.integers(1, 101, 500)
        frame = pd.DataFrame({
            "Acres": acres.astype(float),
            "Size": np.select([acres <= 10, acres <= 50], ["Small", "Medium"], "Large"),
            "Weight": rng.normal(400, 50, 500),
        })
        roles = ColumnRoles(date=None, measure="Acres", dimension="Size", identifier=None,
                            numeric=("Acres", "Weight"), dimensions=("Size",), aggregation="mean")

        brief = analyze_business(frame, roles)

        self.assertIsNone(brief_card(brief, "leader"))
        self.assertNotIn("Acres differs across Size", brief.headline)

    def test_a_score_is_not_compared_across_its_own_items(self):
        rng = np.random.default_rng(68)
        items = pd.DataFrame(rng.integers(0, 11, (240, 3)), columns=["RDB1", "RDB2", "RDB3"])
        frame = items.assign(
            arm=np.tile(["A", "B"], 120),
            risky=items.mean(axis=1),
            other=rng.normal(size=240),
        )
        roles = ColumnRoles(date=None, measure="risky", dimension="arm", identifier=None,
                            numeric=("RDB1", "RDB2", "RDB3", "risky", "other"),
                            dimensions=("arm", "RDB1", "RDB2", "RDB3"), factors=("RDB1", "RDB2", "RDB3"))

        brief = analyze_business(frame, roles)
        differences = brief_card(brief, "differences")

        assert differences is not None
        self.assertNotIn("risky differs across RDB", differences.statement)
        self.assertIn("left out as a score compared with its own items: risky by", differences.calculation)
        self.assertNotIn("RDB1 explains", differences.statement)

    def test_date_parts_and_nested_groupings_are_not_compared(self):
        rng = np.random.default_rng(69)
        dates = pd.date_range("2020-01-01", periods=480, freq="D")
        frame = pd.DataFrame({
            "Date": dates,
            "MO": dates.month,
            "Dye": np.repeat(["blue", "green", "red", "yellow"], 120),
            "Concentration": np.r_[np.tile([3, 6, 9], 80), np.tile([15, 25], 120)],
            "B": rng.normal(100, 20, 480),
            "G": rng.normal(100, 20, 480) + np.repeat([0, 40, 80, 120], 120),
        })
        roles = ColumnRoles(date="Date", measure="B", dimension="Dye", identifier=None,
                            numeric=("MO", "Concentration", "B", "G"),
                            dimensions=("Dye", "MO", "Concentration"), factors=("MO", "Concentration"))

        differences = brief_card(analyze_business(frame, roles), "differences")

        assert differences is not None
        self.assertIn("MO is the month of Date, so it is not used as a grouping", differences.calculation)
        self.assertIn("Concentration within Dye", differences.statement)
        self.assertNotIn("by MO", differences.statement)

    def test_the_scan_compares_across_a_handful_of_groupings(self):
        rng = np.random.default_rng(70)
        flags = {f"flag{index}": rng.integers(0, 2, 300) for index in range(9)}
        frame = pd.DataFrame({"Arm": rng.choice(["A", "B"], 300), **flags, "y": rng.normal(size=300)})
        frame["z"] = rng.normal(size=300)
        roles = ColumnRoles(date=None, measure="y", dimension="Arm", identifier=None,
                            numeric=(*flags, "y", "z"), dimensions=("Arm", *flags), factors=tuple(flags))

        differences = brief_card(analyze_business(frame, roles), "differences")

        assert differences is not None
        self.assertIn("first 6 groupings only, so flag5, flag6, flag7 and 1 other", differences.calculation)


def lab_roles(frame: pd.DataFrame, measure: str, dimension: str | None = None, date: str | None = None):
    numeric = tuple(column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column]))
    return ColumnRoles(
        date=date,
        measure=measure,
        dimension=dimension,
        identifier=None,
        numeric=numeric,
        dimensions=(dimension,) if dimension else (),
        aggregation="mean",
    )


def brief_card(brief, kind: str) -> Evidence | None:
    return next((item for item in brief.evidence if item.kind == kind), None)


class OutlierShapeTests(unittest.TestCase):
    def test_a_right_skewed_measure_is_judged_on_a_log_scale(self):
        rng = np.random.default_rng(31)
        frame = pd.DataFrame({"Latency": rng.lognormal(4, 1, 400)})

        outliers = brief_card(analyze_business(frame, lab_roles(frame, "Latency")), "outliers")

        self.assertTrue(outliers is None or "log scale" in outliers.statement)

    def test_each_group_is_judged_against_its_own_quartiles(self):
        rng = np.random.default_rng(32)
        frame = pd.DataFrame({
            "Station": np.repeat(["Tropic", "Arctic"], 300),
            "MaxTemp": np.r_[rng.normal(30, 2, 300), rng.normal(-20, 2, 300)],
        })
        frame.loc[0:11, "MaxTemp"] = 45.0

        outliers = brief_card(analyze_business(frame, lab_roles(frame, "MaxTemp", "Station")), "outliers")

        assert outliers is not None
        self.assertLessEqual(int(outliers.value.replace(",", "")), 20)
        self.assertIn("of their own station", outliers.statement)

    def test_outliers_are_judged_within_the_grouping_that_explains_them(self):
        rng = np.random.default_rng(37)
        frequency = np.tile(np.arange(100.0, 600.0, 10.0), 40)
        resonance = np.where(frequency % 100 == 0, 400.0, 100.0)
        frame = pd.DataFrame({
            "Sample": np.repeat(["A", "B", "C", "D"], 500),
            "Frequency": frequency,
            "Signal": resonance + rng.normal(0, 5, 2000),
        })

        outliers = brief_card(analyze_business(frame, lab_roles(frame, "Signal", "Sample")), "outliers")

        # Pooled, every resonant reading would sit outside 1.5×IQR; within frequency none do.
        self.assertTrue(outliers is None or "of their own frequency" in outliers.statement)

    def test_a_count_within_chance_raises_no_card(self):
        rng = np.random.default_rng(38)
        frame = pd.DataFrame({"Potassium": rng.normal(60, 8, 104), "Nitrogen": rng.normal(22, 2, 104)})
        frame.loc[0, "Potassium"] = 120.0

        outliers = brief_card(analyze_business(frame, lab_roles(frame, "Potassium")), "outliers")

        self.assertIsNone(outliers)

    def test_every_reading_is_counted_and_the_measure_named_when_within_chance(self):
        rng = np.random.default_rng(39)
        nitrogen = rng.normal(22, 2, 400)
        nitrogen[:40] = rng.uniform(40, 60, 40)
        frame = pd.DataFrame({"Potassium": rng.normal(60, 8, 400), "Nitrogen": nitrogen})

        outliers = brief_card(analyze_business(frame, lab_roles(frame, "Potassium")), "outliers")

        assert outliers is not None
        self.assertIn("of Nitrogen", outliers.statement)
        self.assertIn("Potassium, the measure, has", outliers.statement)
        self.assertIn("chance", outliers.statement)

    def test_a_mostly_zero_measure_is_described_as_heavy_tailed(self):
        rng = np.random.default_rng(33)
        counts = np.where(rng.random(500) < 0.6, 0.0, rng.lognormal(2, 1.5, 500).round())
        frame = pd.DataFrame({"Abundance": counts})

        outliers = brief_card(analyze_business(frame, lab_roles(frame, "Abundance")), "outliers")

        assert outliers is not None
        self.assertEqual(outliers.title, "Heavy-tailed measure")
        self.assertIn("none are listed", outliers.statement)

    def test_flagged_values_in_one_run_of_time_are_a_shift(self):
        rng = np.random.default_rng(34)
        times = pd.date_range("2024-01-01 15:00", periods=600, freq="s")
        signal = np.r_[rng.normal(100, 3, 500), rng.normal(300, 10, 100)]
        frame = pd.DataFrame({"Time": times, "MQ139": signal})

        brief = analyze_business(frame, lab_roles(frame, "MQ139", date="Time"))
        outliers = brief_card(brief, "outliers")

        assert outliers is not None
        self.assertEqual(outliers.title, "Level shift")
        self.assertIn("01 Jan 2024 15:08:20", outliers.statement)
        self.assertIn("to the end of the record", outliers.statement)
        self.assertNotIn("Review the extreme records", [item.title for item in brief.recommendations])

    def test_scattered_outliers_keep_the_review_recommendation(self):
        rng = np.random.default_rng(35)
        times = pd.date_range("2024-01-01", periods=300, freq="D")
        values = rng.normal(100, 5, 300)
        far = [160.0, 40.0, 170.0, 150.0, 45.0, 155.0, 165.0, 35.0, 158.0, 42.0]
        values[[20, 50, 80, 110, 140, 170, 200, 230, 260, 290]] = far
        frame = pd.DataFrame({"Day": times, "Reading": values})

        brief = analyze_business(frame, lab_roles(frame, "Reading", date="Day"))

        self.assertEqual(brief_card(brief, "outliers").title, "Exceptional records")


class DataCheckTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(36)
        groups = np.repeat(["Healthy", "Anaemia", "Leukaemia"], 100)
        hct = np.r_[rng.normal(45, 3, 100), rng.normal(35, 3, 100), rng.normal(44, 3, 100)]
        hct[150] = 3715.0  # one impossible record inside the anaemia group
        self.frame = pd.DataFrame({
            "Diagnosis": groups,
            "HCT": hct.round(1),
            "NEUTp": np.r_[rng.uniform(40, 80, 299), [5317.0]],
        })
        self.roles = lab_roles(self.frame, "HCT", "Diagnosis")
        self.brief = analyze_business(self.frame, self.roles)

    def test_the_card_lists_the_leading_issues_with_counts(self):
        checks = brief_card(self.brief, "quality")

        assert checks is not None
        self.assertEqual(checks.title, "Data checks")
        self.assertIn("HCT: 1 extreme value (3,715)", checks.statement)
        self.assertIn("NEUTp: 1 value above 100 (5,317)", checks.statement)
        self.assertEqual(checks.tone, "warning")

    def test_a_flagged_value_cannot_carry_the_leader(self):
        leader = brief_card(self.brief, "leader")

        # One impossible HCT makes the reading skewed, so groups are compared by
        # medians and ranks, which the flagged value can't move.
        self.assertIn("Healthy has the highest median", leader.statement)
        self.assertNotIn("Anaemia has the highest", self.brief.headline)

    def test_the_summary_says_what_to_check_first(self):
        self.assertIn("Check these first: HCT: 1 extreme value (3,715)", self.brief.summary)

    def test_checking_the_flags_comes_first_among_the_next_steps(self):
        self.assertEqual(self.brief.recommendations[0].title, "Check the flagged values first")

    def test_a_clean_table_keeps_the_completeness_card(self):
        frame = pd.DataFrame({"Group": ["a", "b"] * 20, "Value": np.linspace(1.0, 2.0, 40)})
        frame.loc[[3, 7, 11], "Value"] = np.nan

        checks = brief_card(analyze_business(frame, lab_roles(frame, "Value", "Group")), "quality")

        assert checks is not None
        self.assertEqual(checks.title, "Data completeness")

    def test_copied_rows_over_a_tenth_of_the_table_are_checked_first(self):
        rng = np.random.default_rng(63)
        base = pd.DataFrame({
            "Result": rng.choice(["0", "1"], 300),
            "Hemoglobin": rng.normal(13, 2, 300).round(1),
            "MCV": rng.normal(85, 9, 300).round(1),
        })
        frame = pd.concat([base, base.iloc[:150]], ignore_index=True)

        brief = analyze_business(frame, lab_roles(frame, "MCV", "Result"))

        copies = r"Check these first: 15\d rows \(3\d%\) are exact copies of other rows"
        self.assertRegex(brief.summary, copies)

    def test_fill_values_in_most_rows_are_checked_before_a_single_bad_value(self):
        rng = np.random.default_rng(65)
        filled = rng.random(600) < 0.6
        frame = pd.DataFrame({
            "Diagnosis": rng.choice(["Healthy", "Anaemia", "Leukaemia"], 600),
            "MCV": np.r_[rng.normal(86, 6, 599).round(3), [-79.3]],
            "HCT": np.where(filled, 46.1526, rng.normal(40, 4, 600).round(1)),
        })

        brief = analyze_business(frame, lab_roles(frame, "MCV", "Diagnosis"))

        self.assertIn("Check these first: HCT: 46.1526 in", brief.summary)
        self.assertIn("MCV: 1 negative value (-79.3)", brief.summary)

    def test_flagged_codes_are_left_out_of_the_relationship_scan(self):
        rng = np.random.default_rng(64)
        speed = rng.gamma(3, 1.2, 500).round(1)
        frame = pd.DataFrame({
            "Station": rng.choice(["North", "South"], 500),
            "Wind Speed": speed,
            "Maximum Wind Speed": (speed * 1.4 + rng.gamma(2, 2, 500)).round(1),
            "Air Temperature": rng.normal(15, 5, 500).round(1),
        })
        frame.loc[:3, ["Wind Speed", "Maximum Wind Speed"]] = 999.9
        roles = lab_roles(frame, "Air Temperature", "Station")

        relationship = brief_card(analyze_business(frame, roles), "relationship")

        assert relationship is not None
        left_out = "8 cells the data checks flagged (missing-value codes) left out"
        self.assertIn(left_out, relationship.calculation)
        pooled = frame.drop(index=range(4))
        expected = pooled["Wind Speed"].corr(pooled["Maximum Wind Speed"])
        self.assertIn(f"r = {expected:.2f}", relationship.statement)

    def test_the_correlates_listed_leave_out_the_flagged_cells_too(self):
        rng = np.random.default_rng(65)
        load = rng.normal(50, 10, 500).round(1)
        frame = pd.DataFrame({
            "Station": rng.choice(["North", "South"], 500),
            "Load": load,
            "Current": (load * 0.9 + rng.normal(0, 3, 500)).round(1),
            "Heat": (load * 0.5 + rng.normal(0, 8, 500)).round(1),
        })
        frame.loc[:3, "Heat"] = 999.9
        roles = lab_roles(frame, "Load", "Station")

        relationship = brief_card(analyze_business(frame, roles), "relationship")

        assert relationship is not None
        pooled = frame.drop(index=range(4))
        expected = pooled["Load"].corr(pooled["Heat"])
        listed = rf"Next: Heat \(r = {expected:.2f}, Spearman [0-9.]+, n = 496\)"
        self.assertRegex(relationship.statement, listed)


class RateCardTests(unittest.TestCase):
    def test_a_yes_no_outcome_gets_its_rate_in_each_group(self):
        rng = np.random.default_rng(50)
        gender = np.repeat([0, 1], 300)
        frame = pd.DataFrame({
            "Gender": gender,
            "Hemoglobin": rng.normal(13, 1.5, 600),
            "Result": (rng.random(600) < np.where(gender == 1, 0.55, 0.3)).astype(int),
        })
        roles = roles_for(frame, "Hemoglobin", "Result", factors=("Gender",))

        brief = analyze_business(frame, roles)

        rates = brief_card(brief, "rates")
        assert rates is not None
        self.assertEqual(rates.title, "Rate of Result 1 by gender")
        self.assertIn("in Gender 1 (n = 300) against", rates.statement)
        self.assertIn("percentage points (95% CI", rates.statement)
        self.assertIn("chi2_contingency", rates.calculation)

    def test_a_flag_fixed_within_every_group_gets_no_rate_brief_card(self):
        frame = pd.DataFrame({
            "Arm": ["a", "b"] * 30,
            "Score": np.linspace(0, 10, 60),
            "Treated": [1, 0] * 30,
        })
        roles = roles_for(frame, "Score", "Arm", factors=("Treated",))

        self.assertIsNone(brief_card(analyze_business(frame, roles), "rates"))


class JointModelCardTests(unittest.TestCase):
    def test_readings_that_explain_more_together_get_a_model_brief_card(self):
        rng = np.random.default_rng(51)
        frame = pd.DataFrame(rng.uniform(0, 10, (400, 3)), columns=["Temperature", "Humidity", "Moisture"])
        frame["Growth"] = 1.2 * frame["Temperature"] + 0.8 * frame["Humidity"] + 0.5 * frame["Moisture"]
        frame["Growth"] += rng.normal(0, 1, 400)

        brief = analyze_business(frame, roles_for(frame, "Growth", None))

        model = brief_card(brief, "model")
        assert model is not None
        self.assertIn("Together, ", model.statement)
        self.assertIn("Growth = ", model.statement)
        self.assertIn("statsmodels OLS", model.calculation)
        self.assertTrue(brief.headline.startswith("Together, "))

    def test_one_reading_that_says_it_all_gets_no_model_brief_card(self):
        rng = np.random.default_rng(52)
        x = rng.normal(0, 1, 300)
        frame = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.5, 300), "z": rng.normal(0, 1, 300)})
        frame["w"] = frame["x"] * 0.5 + rng.normal(0, 1, 300)

        self.assertIsNone(brief_card(analyze_business(frame, roles_for(frame, "y", None)), "model"))

    def test_an_exact_combination_is_named_instead_of_modelled(self):
        rng = np.random.default_rng(56)
        # More parts than the construction screen traces, so only the model meets the identity.
        items = pd.DataFrame(rng.normal(25, 2, (200, 9)), columns=[f"item{index}" for index in range(9)])
        frame = items.assign(Total=items.sum(axis=1))
        roles = roles_for(frame, "Total", None)

        brief = analyze_business(frame, roles)

        model = brief_card(brief, "model")
        assert model is not None
        self.assertEqual(model.title, "Total is a combination of other columns")
        self.assertIn("Total is almost exactly a fixed combination of", model.statement)
        self.assertEqual(model.lead, "")
        self.assertNotIn("Together", brief.headline)

    def test_only_predictors_that_stay_clear_are_named_in_the_lead(self):
        rng = np.random.default_rng(57)
        frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=["carbon", "nickel", "vanadium"])
        frame["vanadium"] = frame["carbon"] * 0.6 + rng.normal(0, 0.8, 300)
        frame["strength"] = 2 * frame["carbon"] + 1.5 * frame["nickel"] + rng.normal(0, 1, 300)

        model = brief_card(analyze_business(frame, roles_for(frame, "strength", None)), "model")

        assert model is not None
        self.assertIn("vanadium adds little once the others are known", model.statement)
        self.assertIn("carbon and nickel stay clear", model.lead)
        self.assertTrue(model.lead.startswith("Together, 3 readings explain"))

    def test_a_case_number_and_another_candidate_outcome_are_not_predictors(self):
        rng = np.random.default_rng(58)
        case = np.repeat(np.arange(1, 21), 15)
        time = np.tile(np.arange(15.0), 20)
        frame = pd.DataFrame({"case": case, "time": time, "load": rng.normal(5, 1, 300)})
        frame["wear"] = 0.1 * frame["time"] + 0.05 * frame["case"] + 0.3 * frame["load"]
        frame["wear"] += rng.normal(0, 0.3, 300)
        frame["yield"] = frame["wear"] * 2 + rng.normal(0, 0.5, 300)
        roles = RolesWithAlternatives(
            date=None, measure="wear", dimension=None, identifier=None, dimensions=(),
            numeric=("case", "time", "load", "wear", "yield"), measure_alternatives=("yield",),
        )

        model = brief_card(analyze_business(frame, roles), "model")

        assert model is not None
        self.assertNotIn("case", model.statement)
        self.assertIn("case (a case number)", model.calculation)
        self.assertIn("yield (another candidate outcome)", model.calculation)

    def test_a_model_on_sibling_readings_never_leads(self):
        rng = np.random.default_rng(59)
        base = rng.normal(size=(200, 3))
        frame = pd.DataFrame(base, columns=["Propanoic_acid", "Butanoic_acid", "Valeric_acid"])
        frame["Acetic_acid"] = frame.sum(axis=1) + rng.normal(0, 1, 200)

        brief = analyze_business(frame, roles_for(frame, "Acetic_acid", None))

        model = brief_card(brief, "model")
        assert model is not None
        self.assertIn("Every predictor is a reading of the same kind as Acetic_acid", model.statement)
        self.assertEqual(model.lead, "")

    def test_a_reversal_among_overlapping_predictors_is_not_a_finding(self):
        rng = np.random.default_rng(61)
        family = rng.normal(0, 1, 400)
        frame = pd.DataFrame({
            "Site": rng.choice(["a", "b"], 400),
            "turbidity": family + rng.normal(0, 0.2, 400),
            "color": family + rng.normal(0, 0.2, 400),
        })
        frame["secchi"] = -3 * frame["turbidity"] + 0.8 * frame["color"] + rng.normal(0, 0.5, 400)

        self.assertIsNone(brief_card(analyze_business(frame, roles_for(frame, "secchi", "Site")), "model"))

    def test_inputs_that_explain_little_still_get_their_model(self):
        rng = np.random.default_rng(60)
        frame = pd.DataFrame(rng.uniform(100, 500, (400, 4)), columns=["Cement", "Water", "Sand", "Ash"])
        frame["Strength"] = 0.02 * frame["Cement"] - 0.01 * frame["Water"] + rng.normal(0, 4, 400)

        model = brief_card(analyze_business(frame, roles_for(frame, "Strength", None)), "model")

        assert model is not None
        self.assertIn("Together, ", model.statement)
        self.assertEqual(model.lead, "")
        self.assertIn("every reading recorded beside it", model.calculation)


class TwoFactorTests(unittest.TestCase):
    def test_each_factor_of_a_combined_grouping_is_tested_on_its_own(self):
        rng = np.random.default_rng(53)
        contractor = np.tile([0, 1], 60)
        investment = np.repeat([0, 1], 60)
        frame = pd.DataFrame({
            "condition": [f"{c}-{i}" for c, i in zip(contractor, investment, strict=True)],
            "contractor": contractor,
            "investment": investment,
            "decision": 5 - 0.8 * investment + rng.normal(0, 1, 120),
        })
        roles = roles_for(frame, "decision", "condition", factors=("contractor", "investment"))

        leader = brief_card(analyze_business(frame, roles), "leader")

        assert leader is not None
        self.assertIn("condition combines contractor and investment", leader.statement)
        self.assertIn("investment differs", leader.statement)
        self.assertIn("contractor shows no clear difference", leader.statement)


class RepeatedRowTests(unittest.TestCase):
    def test_tests_run_on_distinct_rows_when_many_repeat_exactly(self):
        rng = np.random.default_rng(54)
        distinct = pd.DataFrame({"Group": ["a", "b"] * 50, "Reading": rng.normal(10, 2, 100)})
        frame = pd.concat([distinct] * 3, ignore_index=True)

        brief = analyze_business(frame, roles_for(frame, "Reading", "Group"))

        leader = brief_card(brief, "leader")
        assert leader is not None
        self.assertIn("n = 50 and 50", leader.statement)
        self.assertIn("on the 100 distinct rows, leaving out 200 exact repeats", leader.calculation)
        self.assertIn("200 of the 300 rows", brief.summary)

    def test_repeated_counts_are_kept(self):
        frame = pd.DataFrame({"Site": ["a", "b"] * 40, "Crabs": [0, 0, 1, 0] * 20})

        leader = brief_card(analyze_business(frame, roles_for(frame, "Crabs", "Site")), "leader")

        assert leader is not None
        self.assertNotIn("distinct rows", leader.calculation)


class HeadlineChoiceTests(unittest.TestCase):
    def test_a_quiet_movement_gives_way_to_a_clear_group_difference(self):
        rng = np.random.default_rng(55)
        days = pd.date_range("2024-01-01", periods=360, freq="D")
        frame = pd.DataFrame({
            "Day": days,
            "Treatment": ["Control", "Compound"] * 180,
            "Viability": rng.normal(80, 2, 360) + np.tile([8.0, 0.0], 180),
        })

        brief = analyze_business(frame, roles_for(frame, "Viability", "Treatment", date="Day"))

        trend = brief_card(brief, "trend")
        self.assertTrue(trend.quiet)
        self.assertTrue(brief.headline.startswith("Viability is higher in Control than in Compound"))
        self.assertNotIn("Check whether the change holds", [item.title for item in brief.recommendations])

    def test_a_bare_average_never_leads_when_a_relationship_was_found(self):
        rng = np.random.default_rng(56)
        x = rng.normal(0, 1, 200)
        frame = pd.DataFrame({"Dose": x, "Response": x + rng.normal(0, 0.5, 200)})

        headline = analyze_business(frame, roles_for(frame, "Response", None)).headline

        self.assertTrue(headline.startswith("Response and Dose move together"), headline)

    def test_a_share_of_a_summed_score_does_not_lead_when_the_groups_do_not_differ(self):
        rng = np.random.default_rng(57)
        frame = pd.DataFrame({
            "Condition": [1] * 13 + [2] * 17,
            "CorrectTotal": rng.uniform(40, 80, 30).round(2),
        })
        roles = roles_for(frame, "CorrectTotal", "Condition", aggregation="sum")

        brief = analyze_business(frame, roles)

        self.assertNotIn("largest share", brief.headline)
        leader = brief_card(brief, "leader")
        self.assertTrue(leader.title.startswith("No clear difference"))
        self.assertIn("Summed, Condition", leader.statement)
        self.assertNotIn("Compare Condition", " ".join(item.title for item in brief.recommendations))

    def test_no_relationship_found_leaves_nothing_to_test(self):
        rng = np.random.default_rng(58)
        frame = pd.DataFrame({column: rng.normal(size=200) for column in "abc"})

        brief = analyze_business(frame, roles_for(frame, "a", None))

        self.assertNotIn("Test the relationship", [item.title for item in brief.recommendations])


class LevelShiftCardTests(unittest.TestCase):
    def test_a_lasting_change_in_level_is_reported_with_its_date(self):
        rng = np.random.default_rng(59)
        months = pd.date_range("2015-01-01", periods=60, freq="MS")
        level = np.where(np.arange(60) < 40, 90.0, 35.0)
        frame = pd.DataFrame({"Month": months, "Ozone": level + rng.normal(0, 8, 60)})

        brief = analyze_business(frame, roles_for(frame, "Ozone", None, date="Month"))

        shift = brief_card(brief, "shift")
        assert shift is not None
        self.assertEqual(shift.value, "Q2 2018")
        self.assertIn("Ozone fell to a new level from Q2 2018", shift.statement)
        self.assertIn("ruptures Binseg", shift.calculation)
        self.assertIn("Find what changed in Q2 2018", [item.title for item in brief.recommendations])

    def test_a_steady_series_has_no_shift(self):
        rng = np.random.default_rng(60)
        months = pd.date_range("2015-01-01", periods=60, freq="MS")
        frame = pd.DataFrame({"Month": months, "Ozone": 50 + 0.5 * np.arange(60) + rng.normal(0, 3, 60)})

        brief = analyze_business(frame, roles_for(frame, "Ozone", None, date="Month"))

        self.assertIsNone(brief_card(brief, "shift"))


class MatchedUnitTests(unittest.TestCase):
    def test_a_change_without_a_grouping_is_measured_on_the_units_in_both_periods(self):
        rows = []
        for station in range(10):
            for quarter, start in enumerate(pd.date_range("2020-01-01", periods=8, freq="QS")):
                if station >= 8 and quarter == 7:
                    continue
                hot = 30.0 if station >= 8 else 0.0
                rows.append({"STA": station * 10 + 7, "Date": start, "MAX": 60 + hot + quarter})
        frame = pd.DataFrame(rows)
        roles = roles_for(frame, "MAX", None, date="Date")

        trend = brief_card(analyze_business(frame, roles), "trend")

        self.assertIn("This compares the 8 STA values that reported in both periods", trend.statement)
        self.assertIn("+1", trend.value)


class OutcomeChoiceTests(unittest.TestCase):
    def test_who_was_studied_is_not_an_outcome(self):
        rng = np.random.default_rng(66)
        gender = np.repeat([0, 1], 300)
        frame = pd.DataFrame({
            "gender": gender,
            "activity": rng.normal(10, 3, 600),
            "smoker": (rng.random(600) < np.where(gender == 1, 0.6, 0.3)).astype(int),
        })
        roles = roles_for(frame, "activity", "gender", factors=("smoker",))

        rates = brief_card(analyze_business(frame, roles), "rates")

        assert rates is not None
        self.assertTrue(rates.title.startswith("Rate of smoker 1 by gender"), rates.title)


class QuietTimelineTests(unittest.TestCase):
    def test_a_quiet_movement_outranks_a_correlation_between_two_readings(self):
        rng = np.random.default_rng(69)
        days = pd.date_range("2020-01-01", periods=720, freq="D")
        low = rng.normal(10, 2, 720)
        frame = pd.DataFrame({"Date": days, "MAX": low + 8 + rng.normal(0, 1, 720), "MIN": low})

        brief = analyze_business(frame, roles_for(frame, "MAX", None, date="Date"))

        self.assertTrue(brief.headline.startswith("No clear change in MAX"), brief.headline)
        self.assertIsNotNone(brief_card(brief, "relationship"))


class DatePartTests(unittest.TestCase):
    def test_parts_of_the_date_are_not_compared_as_groupings(self):
        rng = np.random.default_rng(67)
        days = pd.date_range("2020-01-01", periods=730, freq="D")
        frame = pd.DataFrame({
            "Date": days,
            "MO": days.month,
            "Station": ["a", "b"] * 365,
            "Temp": 15 + 10 * np.sin(days.dayofyear / 58) + rng.normal(0, 1, 730),
        })
        roles = roles_for(frame, "Temp", "Station", date="Date", factors=("MO",))

        differences = brief_card(analyze_business(frame, roles), "differences")

        text = "" if differences is None else differences.statement + differences.calculation
        self.assertNotIn("across MO", text)
        self.assertNotIn("by MO", text)


class InputImbalanceTests(unittest.TestCase):
    def test_inputs_that_differ_between_groups_are_named_as_an_imbalance(self):
        rng = np.random.default_rng(68)
        age = np.repeat([3, 7, 28, 91], 40)
        plasticizer = np.where(age == 91, 12.0, 2.0) + rng.normal(0, 1, 160)
        frame = pd.DataFrame({
            "Age": age,
            "Superplasticizer": plasticizer,
            "Water": rng.normal(180, 10, 160),
            "Strength": 20 + 0.3 * age + 2 * plasticizer + rng.normal(0, 3, 160),
        })
        roles = roles_for(frame, "Strength", "Age")

        leader = brief_card(analyze_business(frame, roles), "leader")

        assert leader is not None
        self.assertIn("The inputs differ across Age too: Superplasticizer", leader.statement)
        self.assertNotIn("Water (", leader.statement)


if __name__ == "__main__":
    unittest.main()
