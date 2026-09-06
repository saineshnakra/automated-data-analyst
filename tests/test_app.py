import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from schema import ColumnRoles


class _BlockImport:
    """Make a package unimportable, the way a broken deployment does."""

    def __init__(self, *names: str, evict: tuple[str, ...] = ()) -> None:
        self.names = set(names)
        # A module that already imported the blocked package is cached and
        # would import again quite happily, so it has to be evicted too.
        self.evict = set(evict)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ImportError(f"No module named {name!r} (simulated)")
        return None

    def __enter__(self):
        sys.meta_path.insert(0, self)
        self.dropped = {
            key: sys.modules.pop(key)
            for key in list(sys.modules)
            if key.split(".")[0] in self.names | self.evict
        }
        return self

    def __exit__(self, *_):
        sys.meta_path.remove(self)
        sys.modules.update(self.dropped)


class DatasetIdentityTests(unittest.TestCase):
    """An answer must not outlive the table it was computed from."""

    def setUp(self):
        import app  # noqa: PLC0415 - importing runs the page once, in bare mode

        self.fingerprint = app.dataset_fingerprint
        self.roles = ColumnRoles(
            date=None, measure="revenue", dimension="region",
            identifier=None, numeric=("revenue",), dimensions=("region",),
        )
        self.frame = pd.DataFrame({"region": ["N", "S"], "revenue": [1.0, 2.0]})

    def test_same_shape_and_name_but_different_numbers_is_a_different_dataset(self):
        other = self.frame.assign(revenue=[10.0, 20.0])

        self.assertNotEqual(
            self.fingerprint(self.frame, self.roles, "q.csv"),
            self.fingerprint(other, self.roles, "q.csv"),
        )

    def test_drilling_into_a_segment_is_a_different_dataset(self):
        slice_ = self.frame[self.frame["region"] == "S"]

        self.assertNotEqual(
            self.fingerprint(self.frame, self.roles, "q.csv"),
            self.fingerprint(slice_, self.roles, "q.csv"),
        )

    def test_changing_which_column_is_the_measure_is_a_different_dataset(self):
        rerolled = ColumnRoles(
            date=None, measure=None, dimension="region",
            identifier=None, numeric=("revenue",), dimensions=("region",),
        )

        self.assertNotEqual(
            self.fingerprint(self.frame, self.roles, "q.csv"),
            self.fingerprint(self.frame, rerolled, "q.csv"),
        )

    def test_the_same_table_is_the_same_dataset(self):
        self.assertEqual(
            self.fingerprint(self.frame, self.roles, "q.csv"),
            self.fingerprint(self.frame.copy(), self.roles, "q.csv"),
        )


class SourceSelectionTests(unittest.TestCase):
    def test_clearing_the_source_control_falls_back_instead_of_crashing(self):
        app = AppTest.from_file("app.py", default_timeout=90).run()

        app.segmented_control[0].set_value(None).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)


class StaleModuleTests(unittest.TestCase):
    """A deploy that adds a symbol must not strand a long-lived server process.

    Streamlit re-executes app.py on every rerun but keeps already-imported
    modules cached. The public app was down for a day because file_io.py
    gained a function the cached copy did not have. This reproduces that in a
    subprocess -- one process, two runs, source changed between them.
    """

    SIM = textwrap.dedent(
        r"""
        import sys, runpy, warnings, logging, io, contextlib
        warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
        def run():
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    runpy.run_path("app.py", run_name="__main__")
            except SystemExit:
                pass
        run()
        with open("file_io.py", "a") as handle:
            handle.write("\n\ndef brand_new_export(frame):\n    return 'fresh'\n")
        src = open("app.py").read().replace(
            "from file_io import list_excel_sheets,",
            "from file_io import brand_new_export, list_excel_sheets,",
        )
        open("app.py", "w").write(src)
        run()
        import file_io
        print("FRESH" if hasattr(file_io, "brand_new_export") else "STALE")
        """
    )

    def test_a_symbol_added_by_a_deploy_is_importable_without_a_restart(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as scratch:
            for path in root.glob("*.py"):
                shutil.copy(path, scratch)
            for folder in ("samples", ".streamlit", "assets"):
                if (root / folder).exists():
                    shutil.copytree(root / folder, Path(scratch) / folder)
            result = subprocess.run(
                [sys.executable, "-c", self.SIM],
                cwd=scratch,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "PYTHONPATH": scratch},
            )

        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertIn("FRESH", result.stdout, result.stdout[-2000:])


class OptionalAiLayerTests(unittest.TestCase):
    """Without a key ADA is a complete product, so it must load without the AI stack."""

    def test_the_app_is_whole_when_the_ai_layer_cannot_be_imported(self):
        with _BlockImport("pydantic", "openai", evict=("ai_insights",)):
            app = AppTest.from_file("app.py", default_timeout=90).run()

        self.assertFalse(app.exception)
        # The deterministic product is untouched: every tab, every KPI, every chart.
        self.assertEqual(len(app.tabs), 6)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(len(app.get("plotly_chart")), 7)
        self.assertTrue(
            any("optional AI layer could not be loaded" in str(w.value) for w in app.warning)
        )


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
