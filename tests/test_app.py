import unittest

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_demo_renders_complete_product(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()

        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Executive brief", "Ask ADA", "Live dashboard", "Explore", "Evidence ledger", "Data room"],
        )
        # Six dashboard charts, plus the one Explore draws for its default columns.
        self.assertEqual(len(app.get("plotly_chart")), 7)
        self.assertEqual(len(app.dataframe), 4)

    def test_drill_down_focuses_the_whole_analysis(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        focus_box = next(box for box in app.selectbox if box.label.startswith("Drill into"))
        focus_box.set_value("Enterprise").run()

        self.assertFalse(app.exception)
        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("Focus · Enterprise", rendered)
        self.assertIn("Segment · Region", rendered)

    def test_ask_ada_answers_a_question(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        app.chat_input[0].set_value("top 3 products by revenue").run()

        self.assertFalse(app.exception)
        history = app.session_state["chat_history"]
        self.assertEqual(len(history), 1)
        self.assertIsNotNone(history[0]["result"])
        self.assertIn("Product", history[0]["result"].answer)

    def test_ask_ada_explains_unreadable_questions(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        app.chat_input[0].set_value("tell me a joke").run()

        self.assertFalse(app.exception)
        history = app.session_state["chat_history"]
        self.assertEqual(len(history), 1)
        self.assertIsNone(history[0]["result"])

    def test_upload_mode_waits_for_a_file(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        app.segmented_control[0].set_value("Upload your file").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(len(app.tabs), 0)

    def test_repeating_a_question_does_not_break_the_transcript(self):
        """Two answers that render the same chart need distinct element ids."""
        app = AppTest.from_file("app.py", default_timeout=45).run()
        app.chat_input[0].set_value("top 3 products by revenue").run()
        app.chat_input[0].set_value("top 3 products by revenue").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["chat_history"]), 2)

    def test_a_sample_dataset_can_be_analyzed_without_uploading(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        app.segmented_control[0].set_value("Try a sample dataset").run()

        picker = next(box for box in app.selectbox if box.label == "Sample dataset")
        self.assertIn("SaaS Subscriptions", picker.options)

        picker.set_value("SaaS Subscriptions").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)
        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("SaaS Subscriptions · sample", rendered)

    def test_explore_charts_any_columns_and_says_why(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        picker = next(box for box in app.multiselect if box.label == "Columns to chart")

        picker.set_value(["Product", "Revenue"]).run()

        self.assertFalse(app.exception)
        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("WHY THIS CHART", rendered)

    def test_explore_handles_every_recommended_form(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        for columns in (
            ["Order Date", "Revenue"],
            ["Order Date", "Revenue", "Product"],
            ["Product", "Region", "Revenue"],
            ["Revenue", "Units"],
            ["Product"],
        ):
            with self.subTest(columns=columns):
                picker = next(box for box in app.multiselect if box.label == "Columns to chart")
                picker.set_value(columns).run()
                self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
