import unittest

from demo_data import make_demo_data
from pipeline import apply_role_selection, cleaning_audit_frame, prepare_analysis, schema_frame


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

    def test_audit_and_schema_frames_are_export_ready(self):
        prepared = prepare_analysis(make_demo_data(rows=80), row_limit=80)

        audit = cleaning_audit_frame(prepared.cleaning_report)
        schema = schema_frame(prepared.detected_roles)

        self.assertEqual(audit.columns.tolist(), ["Operation", "Count"])
        self.assertEqual(schema.columns.tolist(), ["Role", "Column"])
        self.assertIn("Primary metric", schema["Role"].tolist())


if __name__ == "__main__":
    unittest.main()
