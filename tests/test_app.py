import unittest

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_demo_renders_complete_product(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()

        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Executive brief", "Ask ADA", "Live dashboard", "Evidence ledger", "Data room"],
        )
        self.assertEqual(len(app.get("plotly_chart")), 4)
        self.assertEqual(len(app.dataframe), 4)

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


if __name__ == "__main__":
    unittest.main()
