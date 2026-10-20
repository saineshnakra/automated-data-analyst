import unittest

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

    def test_audit_and_schema_frames_are_export_ready(self):
        prepared = prepare_analysis(make_demo_data(rows=80), row_limit=80)

        audit = cleaning_audit_frame(prepared.cleaning_report)
        schema = schema_frame(prepared.detected_roles)

        self.assertEqual(audit.columns.tolist(), ["Operation", "Count"])
        self.assertEqual(schema.columns.tolist(), ["Role", "Column"])
        self.assertIn("Primary metric", schema["Role"].tolist())


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

    def test_a_file_with_no_dates_still_takes_its_tail(self):
        frame = pd.DataFrame({"Segment": list("abcdefghij"), "Revenue": [float(n) for n in range(10)]})
        prepared = prepare_analysis(frame, row_limit=4)
        self.assertEqual(len(prepared.dataframe), 4)
        self.assertEqual(prepared.dataframe["Revenue"].tolist(), [6.0, 7.0, 8.0, 9.0])


if __name__ == "__main__":
    unittest.main()
