import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from ada.demo_data import make_demo_data
from ada.pipeline import (
    apply_focus,
    apply_role_selection,
    cleaning_audit_frame,
    focus_options,
    prepare_analysis,
    schema_frame,
)


def dated(pairs):
    frame = pd.DataFrame(pairs, columns=["Date", "Revenue"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


class PipelineTests(unittest.TestCase):
    def test_prepare_analysis_bounds_work_and_detects_schema(self):
        raw = make_demo_data(rows=80)

        prepared = prepare_analysis(raw, row_limit=50)

        self.assertEqual(len(prepared.dataframe), 50)
        self.assertEqual(prepared.truncated_rows, 30)
        self.assertEqual(prepared.detected_roles.measure, "Revenue")
        self.assertTrue(prepared.analyze().headline)

    def test_role_overrides_preserve_detected_candidates(self):
        prepared = prepare_analysis(make_demo_data(rows=100), row_limit=100)

        selected = apply_role_selection(
            prepared.detected_roles,
            date="None",
            measure="Profit",
            dimension="Region",
        )

        self.assertIsNone(selected.date)
        self.assertEqual(selected.measure, "Profit")
        self.assertEqual(selected.dimension, "Region")
        self.assertEqual(selected.numeric, prepared.detected_roles.numeric)
        self.assertIsNone(selected.aggregation)

    def test_the_chosen_way_to_combine_rows_is_kept_through_a_focus(self):
        prepared = prepare_analysis(make_demo_data(rows=600), row_limit=600)
        selected = apply_role_selection(
            prepared.detected_roles, date="Order Date", measure="Revenue", dimension="Product",
            aggregation="mean",
        )

        _, focused_roles = apply_focus(prepared.dataframe, selected, "Enterprise")

        self.assertEqual(selected.aggregation, "mean")
        self.assertEqual(focused_roles.aggregation, "mean")
        schema = schema_frame(focused_roles).set_index("Role")
        self.assertEqual(schema.loc["Combine rows by", "Column"], "Average")

    def test_focus_options_rank_by_frequency(self):
        prepared = prepare_analysis(make_demo_data(rows=600), row_limit=600)

        options = focus_options(prepared.dataframe, prepared.detected_roles)

        self.assertEqual(len(options), 4)
        self.assertEqual(options[0], "Core")

    def test_focus_filters_and_regroups_by_the_next_dimension(self):
        prepared = prepare_analysis(make_demo_data(rows=600), row_limit=600)

        focused, focused_roles = apply_focus(
            prepared.dataframe, prepared.detected_roles, "Enterprise"
        )

        self.assertTrue((focused["Product"] == "Enterprise").all())
        self.assertLess(len(focused), len(prepared.dataframe))
        self.assertEqual(focused_roles.dimension, "Region")
        self.assertEqual(focused_roles.measure, prepared.detected_roles.measure)

    def test_focus_is_a_no_op_for_missing_or_unknown_values(self):
        prepared = prepare_analysis(make_demo_data(rows=200), row_limit=200)

        for focus in (None, "Atlantis"):
            dataframe, roles = apply_focus(prepared.dataframe, prepared.detected_roles, focus)
            self.assertEqual(len(dataframe), len(prepared.dataframe))
            self.assertEqual(roles, prepared.detected_roles)

    def test_changing_a_role_drops_what_was_detected_for_it(self):
        detected = replace(
            prepare_analysis(make_demo_data(rows=100), row_limit=100).detected_roles,
            measure_alternatives=("Profit",), focus="Core",
        )

        kept = apply_role_selection(
            detected, date="None", measure=detected.measure, dimension=detected.dimension
        )
        changed = apply_role_selection(detected, date="None", measure="Profit", dimension="Region")

        self.assertEqual((kept.measure_alternatives, kept.focus), (("Profit",), "Core"))
        self.assertEqual((changed.measure_alternatives, changed.focus), ((), None))

    def test_a_focus_is_spent_once_applied(self):
        prepared = prepare_analysis(make_demo_data(rows=200), row_limit=200)
        roles = replace(prepared.detected_roles, focus="Core")

        _, focused_roles = apply_focus(prepared.dataframe, roles, "Core")

        self.assertIsNone(focused_roles.focus)

    def test_what_detection_concluded_is_listed_with_the_cleaning_notes(self):
        raw = pd.DataFrame({"Treatment": ["A", "B"] * 20, "Reopened": [0, 1, 0, 0] * 10,
                            "Hours": [float(index % 7) + 0.5 for index in range(40)]})

        notes = prepare_analysis(raw, row_limit=100).cleaning_report.notes

        self.assertTrue(any("categories, not quantities: Reopened." in note for note in notes))

    def test_a_matrix_is_measured_by_its_row_mean(self):
        raw = pd.DataFrame({f"S{index}": [index + row for row in range(10)] for index in range(25)})
        raw.insert(0, "Gene", [f"gene {row}" for row in range(10)])

        prepared = prepare_analysis(raw, row_limit=100)

        self.assertEqual(prepared.detected_roles.measure, "Row mean (25 columns)")
        self.assertEqual(prepared.dataframe["Row mean (25 columns)"].iloc[0], 12.0)

    def test_audit_and_schema_frames_are_export_ready(self):
        prepared = prepare_analysis(make_demo_data(rows=80), row_limit=80)

        audit = cleaning_audit_frame(prepared.cleaning_report)
        schema = schema_frame(prepared.detected_roles)

        self.assertEqual(audit.columns.tolist(), ["Operation", "Count"])
        self.assertEqual(schema.columns.tolist(), ["Role", "Column"])
        self.assertIn("Primary metric", schema["Role"].tolist())
        self.assertEqual(schema.set_index("Role").loc["Combine rows by", "Column"], "Total")


class RowLimitTests(unittest.TestCase):
    def test_a_half_present_oldest_period_is_dropped(self):
        # Two records a month for eight months. Keeping fifteen of sixteen
        # cuts one February record, so February is partial.
        rows = [(f"2025-{month:02d}-{day:02d}", 100.0) for month in range(2, 10) for day in (10, 20)]
        prepared = prepare_analysis(dated(rows), row_limit=15)
        self.assertEqual(prepared.analyzed_from, pd.Timestamp("2025-03-10"))
        self.assertNotIn(2, prepared.dataframe["Date"].dt.month.unique())

    def test_a_period_that_survives_the_cut_whole_is_kept(self):
        # All sixteen rows fit, so nothing is cut.
        rows = [(f"2025-{month:02d}-{day:02d}", 100.0) for month in range(2, 10) for day in (10, 20)]
        prepared = prepare_analysis(dated(rows), row_limit=16)
        self.assertEqual(prepared.analyzed_from, pd.Timestamp("2025-02-10"))

    def test_a_file_that_is_all_one_period_is_not_emptied(self):
        # Every kept row falls in the truncated period, so dropping it would leave no rows.
        rows = [("2025-03-03", float(n)) for n in range(400)]
        prepared = prepare_analysis(dated(rows), row_limit=100)
        self.assertEqual(len(prepared.dataframe), 100)
        self.assertGreater(prepared.dataframe["Revenue"].sum(), 0)

    def test_a_file_under_the_cap_is_untouched(self):
        rows = [(f"2025-0{month}-10", 100.0) for month in range(1, 6)]
        prepared = prepare_analysis(dated(rows), row_limit=1000)
        self.assertEqual(len(prepared.dataframe), 5)
        self.assertEqual(prepared.analyzed_from, pd.Timestamp("2025-01-10"))

    def test_a_file_with_no_dates_is_sampled_in_file_order(self):
        # Sorted by material: the tail would hold one material only.
        frame = pd.DataFrame({
            "Material": [name for name in "ABCD" for _ in range(500)],
            "Signal": [float(n) for n in range(2000)],
        })
        prepared = prepare_analysis(frame, row_limit=400)
        signal = prepared.dataframe["Signal"]
        self.assertEqual(len(prepared.dataframe), 400)
        self.assertTrue(signal.is_monotonic_increasing)
        self.assertEqual(set(prepared.dataframe["Material"]), set("ABCD"))
        self.assertEqual(prepared.row_selection, "sample")
        self.assertEqual(prepared.rows_note, "a random sample of 400 of 2,000 rows")
        self.assertNotIn("newest", prepared.rows_note)

    def test_the_sample_is_the_same_on_every_run(self):
        frame = pd.DataFrame({"Segment": list("abcdefghij") * 30, "Revenue": [float(n) for n in range(300)]})
        first = prepare_analysis(frame, row_limit=40).dataframe["Revenue"].tolist()
        second = prepare_analysis(frame, row_limit=40).dataframe["Revenue"].tolist()
        self.assertEqual(first, second)

    def test_a_sample_missing_a_group_says_so(self):
        frame = pd.DataFrame({
            "Site": ["Rare"] + ["Common"] * 999 + ["Other"] * 1000,
            "Reading": [float(n % 7) for n in range(2000)],
        })
        prepared = prepare_analysis(frame, row_limit=50)
        self.assertEqual(prepared.dataframe["Site"].nunique(), 2)
        self.assertTrue(
            any("Site has 3 values in the file but 2 in the sample" in note
                for note in prepared.cleaning_report.notes)
        )

    def test_a_dated_file_keeps_its_newest_rows_and_says_so(self):
        rows = [(f"2025-{month:02d}-{day:02d}", 100.0) for month in range(2, 10) for day in (10, 20)]
        prepared = prepare_analysis(dated(rows), row_limit=15)
        self.assertEqual(prepared.row_selection, "newest")
        self.assertEqual(prepared.rows_note, "the newest 14 of 16 rows, from 10 Mar 2025")

    def test_a_file_that_fits_has_no_rows_note(self):
        prepared = prepare_analysis(make_demo_data(rows=80), row_limit=80)
        self.assertEqual(prepared.rows_note, "")


def hourly(start, hours, level):
    rng = np.random.default_rng(len(start) + hours)
    return pd.DataFrame({
        "Datetime": pd.date_range(start, periods=hours, freq="h"),
        "CO": level + rng.random(hours),
    })


class LogStructureTests(unittest.TestCase):
    def test_daily_summaries_below_hourly_rows_are_left_out(self):
        daily = pd.DataFrame({"Datetime": pd.date_range("2024-01-01", periods=30, freq="D"), "CO": 5.0})
        frame = pd.concat([hourly("2024-01-01 01:00", 24 * 30, 1.0), daily], ignore_index=True)

        prepared = prepare_analysis(frame, row_limit=10_000)

        self.assertEqual(len(prepared.dataframe), 24 * 30)
        self.assertLess(prepared.dataframe["CO"].max(), 5.0)
        self.assertTrue(any("set aside" in note for note in prepared.cleaning_report.notes))

    def test_two_logs_in_one_file_become_two_groups(self):
        frame = pd.concat(
            [hourly("2024-01-01", 600, 1.0), hourly("2024-01-01", 600, 3.0)], ignore_index=True
        )

        prepared = prepare_analysis(frame, row_limit=10_000)

        self.assertEqual(prepared.detected_roles.dimension, "Block")
        means = prepared.dataframe.groupby("Block")["CO"].mean()
        self.assertLess(means["Block 1"], 2.0)
        self.assertGreater(means["Block 2"], 3.0)
        self.assertTrue(any("holds 2 logs" in note for note in prepared.cleaning_report.notes))


if __name__ == "__main__":
    unittest.main()
