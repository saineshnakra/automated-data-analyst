import unittest

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_demo_renders_complete_product(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()

        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Executive view", "Dashboard", "Evidence", "Data room"],
        )
        self.assertEqual(len(app.get("plotly_chart")), 4)
        self.assertEqual(len(app.dataframe), 4)

    def test_upload_mode_waits_for_a_file(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()
        app.segmented_control[0].set_value("Upload your file").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(len(app.tabs), 0)


if __name__ == "__main__":
    unittest.main()
