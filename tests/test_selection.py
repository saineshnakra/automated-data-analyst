import unittest

import numpy as np
import pandas as pd
import pandas.testing as pdt

from ada.aggregation import preferred_frequency
from ada.autovis import ChartSpec
from ada.business_insights import analyze_business
from ada.demo_data import make_demo_data
from ada.nlq import ValueFilter
from ada.pipeline import analyze_selection, prepare_analysis
from ada.schema import detect_roles
from ada.selection import (
    MIN_SLICE_RECORDS,
    RangeFilter,
    RegionFilter,
    Selection,
    apply_selection,
    compare_slice,
    comparison_frame,
    describe_selection,
    selection_from_event,
    selection_modes,
)


def event(points=(), box=(), lasso=()):
    return {
        "points": list(points),
        "point_indices": list(range(len(points))),
        "box": list(box),
        "lasso": list(lasso),
    }


class Prepared:
    """The demo file, prepared the way the app prepares it."""

    def __init__(self):
        prepared = prepare_analysis(make_demo_data(rows=400), row_limit=400)
        self.frame = prepared.dataframe
        self.roles = prepared.detected_roles


class EventTranslationTests(unittest.TestCase):
    def setUp(self):
        self.data = Prepared()

    def test_a_box_on_a_scatter_is_a_range_on_each_axis(self):
        spec = ChartSpec(form="scatter", rationale="", x="Revenue", y="Profit")

        selection = selection_from_event(
            event(box=[{"xref": "x", "yref": "y", "x": [900, 300], "y": [50, 400]}]), spec, self.data.frame
        )

        self.assertEqual(selection.method, "box")
        self.assertEqual(
            selection.ranges,
            (RangeFilter("Revenue", 300.0, 900.0), RangeFilter("Profit", 50.0, 400.0)),
        )
        expected = self.data.frame[
            self.data.frame["Revenue"].between(300, 900) & self.data.frame["Profit"].between(50, 400)
        ]
        self.assertEqual(len(apply_selection(self.data.frame, selection)), len(expected))

    def test_a_lasso_keeps_only_points_inside_the_shape(self):
        frame = pd.DataFrame({"a": [1.0, 2.0, 8.0, 1.0, np.nan], "b": [1.0, 2.0, 8.0, 7.0, 1.0]})
        spec = ChartSpec(form="scatter", rationale="", x="a", y="b")
        triangle = {"x": [0, 6, 0], "y": [0, 0, 6]}

        selection = selection_from_event(event(lasso=[triangle]), spec, frame)

        self.assertEqual(selection.region, RegionFilter("a", "b", ((0.0, 0.0), (6.0, 0.0), (0.0, 6.0))))
        kept = apply_selection(frame, selection)
        # (1,1) and (2,2) are inside; (8,8) and (1,7) are not; a missing value is never inside.
        self.assertEqual(kept[["a", "b"]].values.tolist(), [[1.0, 1.0], [2.0, 2.0]])

    def test_a_box_on_a_trend_selects_the_time_window_and_ignores_the_vertical_extent(self):
        spec = ChartSpec(form="area", rationale="", x="Order Date", y="Revenue")
        box = {"x": ["2025-03-01 00:00:00", "2025-06-30 23:59:59"], "y": [0, 1]}

        selection = selection_from_event(event(box=[box]), spec, self.data.frame)

        self.assertEqual(len(selection.ranges), 1)
        window = selection.ranges[0]
        self.assertEqual(window.column, "Order Date")
        self.assertGreaterEqual(window.low, pd.Timestamp("2025-03-01"))
        self.assertLessEqual(window.high.normalize(), pd.Timestamp("2025-06-30"))
        kept = apply_selection(self.data.frame, selection)
        self.assertTrue(kept["Order Date"].between(window.low, window.high).all())
        self.assertGreater(len(kept), 0)

    def test_a_time_window_is_narrowed_to_whole_periods_and_says_so(self):
        spec = ChartSpec(form="area", rationale="", x="Order Date", y="Revenue")
        box = {"x": ["2025-04-19", "2025-09-08"]}

        selection = selection_from_event(event(box=[box]), spec, self.data.frame)
        window = selection.ranges[0]
        inside = self.data.frame["Order Date"].between("2025-04-19", "2025-09-08")
        grain = preferred_frequency(self.data.frame.loc[inside, "Order Date"])

        # Both ends land on period boundaries, inside what was drawn.
        self.assertEqual(window.low, pd.Period(window.low, freq=grain).start_time)
        self.assertEqual(window.high, pd.Period(window.high, freq=grain).end_time)
        self.assertGreaterEqual(window.low, pd.Timestamp("2025-04-19"))
        self.assertLessEqual(window.high, pd.Timestamp("2025-09-08 23:59:59.999999999"))
        kept = apply_selection(self.data.frame, selection)
        self.assertTrue(kept["Order Date"].between("2025-04-19", "2025-09-08 23:59:59").all())
        self.assertIn("Narrowed to whole", selection.adjustment)
        self.assertIn("19 Apr 2025 to 08 Sep 2025 you selected", selection.adjustment)
        _, comparison = compare_slice(self.data.frame, selection, self.data.roles)
        self.assertIn(selection.adjustment, comparison.notes)

    def test_a_window_already_on_period_boundaries_is_left_as_drawn(self):
        frame = pd.DataFrame(
            {"Month": pd.date_range("2023-01-01", periods=36, freq="MS"), "Revenue": range(36)}
        )
        spec = ChartSpec(form="area", rationale="", x="Month", y="Revenue")
        box = {"x": ["2024-01-01", "2024-12-31 23:59:59.999999999"]}

        selection = selection_from_event(event(box=[box]), spec, frame)

        self.assertEqual(selection.adjustment, "")
        self.assertEqual(len(apply_selection(frame, selection)), 12)

    def test_clicked_points_on_a_trend_span_the_earliest_to_the_latest(self):
        spec = ChartSpec(form="line", rationale="", x="Order Date", y="Revenue", color="Product")
        points = [{"x": "2025-05-04", "y": 10}, {"x": "2025-02-01", "y": 12}]

        selection = selection_from_event(event(points=points), spec, self.data.frame)

        window = selection.ranges[0]
        self.assertLessEqual(window.low, pd.Timestamp("2025-02-01"))
        self.assertGreaterEqual(window.high, pd.Timestamp("2025-05-04"))
        self.assertEqual(selection.method, "click")

    def test_clicked_bars_become_a_value_filter(self):
        spec = ChartSpec(form="column", rationale="", x="Product", y="Revenue")
        points = [{"x": "Enterprise", "y": 1}, {"x": "Growth", "y": 2}, {"x": "Enterprise", "y": 1}]

        selection = selection_from_event(event(points=points), spec, self.data.frame)

        self.assertEqual(selection.values, (ValueFilter("Product", ("Enterprise", "Growth")),))
        kept = apply_selection(self.data.frame, selection)
        self.assertEqual(set(kept["Product"]), {"Enterprise", "Growth"})

    def test_a_horizontal_bar_reads_its_labels_from_the_vertical_axis(self):
        spec = ChartSpec(form="bar", rationale="", x="Region", y="Revenue")

        selection = selection_from_event(event(points=[{"x": 4200.0, "y": "West"}]), spec, self.data.frame)

        self.assertEqual(selection.values, (ValueFilter("Region", ("West",)),))

    def test_a_folded_other_bar_is_not_a_real_value_and_selects_nothing(self):
        spec = ChartSpec(form="column", rationale="", x="Product", y="Revenue")

        selection = selection_from_event(event(points=[{"x": "Other", "y": 1}]), spec, self.data.frame)

        self.assertTrue(selection.is_empty)

    def test_heatmaps_tables_and_stats_accept_no_selection(self):
        for form in ("heatmap", "table", "stat", "none"):
            with self.subTest(form=form):
                spec = ChartSpec(form=form, rationale="", x="Product", y="Region")
                self.assertEqual(selection_modes(spec), ())
                selection = selection_from_event(event(points=[{"x": "Core"}]), spec, self.data.frame)
                self.assertTrue(selection.is_empty)

    def test_a_timezone_aware_date_column_can_be_selected(self):
        frame = pd.DataFrame(
            {
                "When": pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC"),
                "Revenue": range(10),
            }
        )
        spec = ChartSpec(form="area", rationale="", x="When", y="Revenue")

        selection = selection_from_event(event(box=[{"x": ["2025-01-03", "2025-01-05"]}]), spec, frame)

        self.assertEqual(str(selection.ranges[0].low.tz), "UTC")
        kept = apply_selection(frame, selection)
        self.assertTrue(set(pd.date_range("2025-01-03", "2025-01-05", tz="UTC")) <= set(kept["When"]))

    def test_garbage_coordinates_select_nothing_rather_than_raising(self):
        spec = ChartSpec(form="scatter", rationale="", x="Revenue", y="Profit")

        garbage = event(box=[{"x": ["a", "b"], "y": [None, None]}])

        selection = selection_from_event(garbage, spec, self.data.frame)

        self.assertTrue(selection.is_empty)


class ClearingTests(unittest.TestCase):
    def test_clearing_the_selection_restores_the_full_analysis_exactly(self):
        data = Prepared()
        spec = ChartSpec(form="column", rationale="", x="Product", y="Revenue")

        cleared = selection_from_event(event(), spec, data.frame)

        self.assertTrue(cleared.is_empty)
        restored = apply_selection(data.frame, cleared)
        self.assertIs(restored, data.frame)
        self.assertEqual(analyze_business(restored, data.roles), analyze_business(data.frame, data.roles))

    def test_a_missing_event_is_no_selection(self):
        spec = ChartSpec(form="scatter", rationale="", x="Revenue", y="Profit")

        self.assertTrue(selection_from_event(None, spec, Prepared().frame).is_empty)


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.data = Prepared()

    def test_the_description_names_the_count_columns_and_bounds_a_reader_can_check(self):
        selection = Selection(
            values=(ValueFilter("Product", ("Enterprise", "Growth")),),
            ranges=(RangeFilter("Order Date", pd.Timestamp("2025-03-01"), pd.Timestamp("2025-06-30")),),
        )
        part = apply_selection(self.data.frame, selection)

        text = describe_selection(selection, self.data.frame, part)

        self.assertTrue(text.startswith(f"{len(part):,} of {len(self.data.frame):,} records you selected"))
        self.assertIn("Product is Enterprise or Growth", text)
        self.assertIn("Order Date from 01 Mar 2025 to 30 Jun 2025", text)

    def test_many_labels_are_summarized_rather_than_listed_without_end(self):
        selection = Selection(values=(ValueFilter("Order ID", tuple(f"ORD-{n}" for n in range(9))),))

        text = describe_selection(selection, self.data.frame, self.data.frame.head(0))

        self.assertIn("or 5 more", text)

    def test_the_wording_never_claims_ada_found_the_grouping(self):
        selection = Selection(values=(ValueFilter("Region", ("West",)),))
        _, comparison = compare_slice(self.data.frame, selection, self.data.roles)
        wording = " ".join(
            [comparison.description, *comparison.notes, *(stat.calculation for stat in comparison.stats)]
        ).lower()

        for claim in ("discover", "found", "cluster", "detected", "segment emerged"):
            self.assertNotIn(claim, wording)
        self.assertIn("you selected", wording)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = Prepared()

    def test_every_slice_figure_sits_beside_the_whole(self):
        selection = Selection(values=(ValueFilter("Region", ("West",)),))

        _, comparison = compare_slice(self.data.frame, selection, self.data.roles)
        table = comparison_frame(comparison)

        self.assertEqual(
            list(table.columns), ["Measure", "Your selection", "Whole view", "Difference", "Calculation"]
        )
        self.assertGreaterEqual(len(table), 3)
        self.assertTrue((table["Whole view"].str.len() > 0).all())
        self.assertTrue((table["Calculation"].str.len() > 0).all())

    def test_the_slice_figures_are_the_arithmetic_of_the_selected_rows(self):
        selection = Selection(values=(ValueFilter("Region", ("West",)),))
        west = self.data.frame[self.data.frame["Region"] == "West"]

        _, comparison = compare_slice(self.data.frame, selection, self.data.roles)
        stats = {stat.label: stat for stat in comparison.stats}

        self.assertEqual(comparison.records, len(west))
        self.assertEqual(stats["Records"].slice_value, f"{len(west):,}")
        share = west["Revenue"].sum() / self.data.frame["Revenue"].sum() * 100
        self.assertIn(f"{share:.1f}%", stats["Total Revenue"].difference)

    def test_slice_evidence_comes_from_the_same_machinery_and_is_paired_with_the_whole(self):
        selection = Selection(values=(ValueFilter("Product", ("Enterprise", "Growth")),))

        result = analyze_selection(self.data.frame, self.data.roles, selection)

        part = apply_selection(self.data.frame, selection)
        expected = analyze_business(part, self.data.roles).evidence
        self.assertEqual(tuple(item for item, _ in result.evidence), expected)
        whole_kinds = {item.kind for item in analyze_business(self.data.frame, self.data.roles).evidence}
        for item, whole in result.evidence:
            if item.kind in whole_kinds:
                self.assertIsNotNone(whole)
                self.assertEqual(whole.kind, item.kind)

    def test_a_selection_too_small_for_a_statistic_says_so_instead(self):
        rows = self.data.frame.head(MIN_SLICE_RECORDS - 2)
        selection = Selection(values=(ValueFilter("Order ID", tuple(rows["Order ID"])),))

        result = analyze_selection(self.data.frame, self.data.roles, selection)
        labels = [stat.label for stat in result.comparison.stats]

        self.assertEqual(result.comparison.records, MIN_SLICE_RECORDS - 2)
        self.assertNotIn("Average Revenue", labels)
        self.assertFalse(any(label.startswith("Most common") for label in labels))
        self.assertEqual(result.evidence, ())
        notes = " ".join(result.comparison.notes)
        self.assertIn(f"at least {MIN_SLICE_RECORDS}", notes)
        self.assertIn("none is reported", notes)

    def test_an_empty_selection_result_says_there_is_nothing_to_analyze(self):
        selection = Selection(
            values=(ValueFilter("Region", ("West",)),), ranges=(RangeFilter("Units", 99, 100),)
        )

        result = analyze_selection(self.data.frame, self.data.roles, selection)

        self.assertEqual(result.comparison.records, 0)
        self.assertIn("No records fall inside the selection", " ".join(result.comparison.notes))
        self.assertEqual(result.evidence, ())

    def test_selecting_everything_is_not_reported_as_a_difference(self):
        selection = Selection(ranges=(RangeFilter("Units", 0, 1_000),))

        result = analyze_selection(self.data.frame, self.data.roles, selection)

        self.assertEqual(result.comparison.records, len(self.data.frame))
        self.assertEqual([stat.label for stat in result.comparison.stats], ["Records"])
        self.assertIn("slice is the whole", " ".join(result.comparison.notes))

    def test_a_rate_is_averaged_and_never_totalled(self):
        frame = pd.DataFrame(
            {"Team": ["A"] * 6 + ["B"] * 6, "Conversion Rate %": [10, 12, 11, 13, 12, 10, 4, 5, 6, 5, 4, 6]}
        )
        roles = analyze_business(frame).roles
        self.assertEqual(roles.measure, "Conversion Rate %")
        selection = Selection(values=(ValueFilter("Team", ("A",)),))

        _, comparison = compare_slice(frame, selection, roles)
        labels = [stat.label for stat in comparison.stats]

        self.assertNotIn("Total Conversion Rate %", labels)
        self.assertIn("Average Conversion Rate %", labels)

    def test_a_selection_on_a_column_that_is_gone_is_an_error_not_a_silent_whole(self):
        with self.assertRaises(ValueError):
            apply_selection(self.data.frame, Selection(ranges=(RangeFilter("Missing", 0, 1),)))


class SelectionFrameUnchangedTests(unittest.TestCase):
    def test_selecting_never_modifies_the_whole_view(self):
        data = Prepared()
        before = data.frame.copy()

        analyze_selection(data.frame, data.roles, Selection(values=(ValueFilter("Region", ("West",)),)))

        pdt.assert_frame_equal(data.frame, before)


class AveragedMeasureComparisonTests(unittest.TestCase):
    def comparison(self, measure):
        frame = pd.DataFrame({"Region": list("ABCD") * 10, measure: [10.0, 20.0, 30.0, 40.0] * 10})
        spec = ChartSpec(form="column", rationale="", x="Region", y=measure)
        selection = selection_from_event(event(points=[{"x": "D"}]), spec, frame)
        _, comparison = compare_slice(frame, selection, detect_roles(frame))
        return {stat.label: stat for stat in comparison.stats}

    def test_a_rate_differs_from_the_whole_in_points(self):
        stats = self.comparison("Conversion Rate")
        self.assertEqual(stats["Average Conversion Rate"].difference, "+15.0 pts vs the whole")

    def test_a_level_differs_from_the_whole_by_a_plain_difference(self):
        stats = self.comparison("pH")
        self.assertEqual(stats["Average pH"].difference, "+15 vs the whole")

    def test_no_segment_mix_when_the_selection_filters_that_column(self):
        stats = self.comparison("Conversion Rate")
        self.assertFalse(any(label.startswith("Most common") for label in stats))


if __name__ == "__main__":
    unittest.main()
