import unittest

import numpy as np
import pandas as pd

from ada.aggregation import driver_frame, heatmap_frame, segment_frame, segment_period_change, trend_frame
from ada.business_insights import (
    Evidence,
    _shown_evidence,
    analyze_business,
    build_business_report,
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
                "Revenue": [100.0] * 40,
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
        tops = [k for k in self.brief.kpis if k.label.startswith("Top ")]
        for kpi in tops:
            self.assertNotIn("Share", kpi.context, f"{kpi.label}: {kpi.context}")

    def test_the_leader_is_still_named(self):
        self.assertIn("B", self.leader().statement)
        self.assertIn("20", self.leader().statement)

    def test_a_rate_headline_does_not_say_coming_from(self):
        headline = self.brief.headline
        self.assertNotIn("coming from", headline, headline)

    def test_a_rate_headline_still_names_the_leader_level(self):
        headline = self.brief.headline
        self.assertIn("B has the highest average, at 20.0%", headline, headline)

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
        self.assertIn("with 66.7% of revenue", leader.statement, leader.statement)

    def test_an_additive_measure_keeps_its_concentration_card(self):
        self.assertIn("concentration", [e.kind for e in self.brief.evidence])

    def test_an_additive_headline_keeps_its_share(self):
        headline = self.brief.headline
        self.assertIn("coming from", headline, headline)
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
        self.assertEqual(brief.recommendations[0].title, "Check whether the change holds")

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
    def test_a_quality_card_is_never_squeezed_out_by_the_six_card_cap(self):
        filler = [
            Evidence(kind=f"k{i}", title="t", value="v", statement="s", calculation="c") for i in range(7)
        ]
        quality = Evidence(kind="quality", title="q", value="v", statement="s", calculation="c")

        shown = _shown_evidence(filler + [quality])

        self.assertEqual(len(shown), 6)
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
        self.assertIn("highest average Water Temp (°C)", leader.statement)
        self.assertIn("South has the lowest", leader.statement)
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
        self.assertTrue(leader.statement.startswith("Desert has the highest average"), leader.statement)
        self.assertIn("Arctic has the lowest", leader.statement)
        top = next(kpi for kpi in brief.kpis if kpi.label == "Top Site")
        self.assertEqual(top.context, "Highest average of any segment")

    def test_a_mixed_sign_leader_is_described_by_size(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        rows = [(month, region, value) for month in months
                for region, value in (("North", -5000.0), ("South", 3000.0), ("East", 1000.0))]
        brief = analyze_business(pd.DataFrame(rows, columns=["Date", "Region", "Profit"]))
        self.assertIn("North is the largest by size", brief.headline)
        top = next(kpi for kpi in brief.kpis if kpi.label == "Top Region")
        self.assertEqual(top.context, "Largest in size of any segment")


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
        self.assertTrue(relationship.statement.startswith("Dose (µM) and Abs 450nm"))

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


if __name__ == "__main__":
    unittest.main()