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
from streamlit.util import calc_hash

from ada.schema import ColumnRoles


def first_look(timeout: float = 45) -> AppTest:
    """The app on its first-look page. st.navigation keys each page by the hash of its URL path."""
    app = AppTest.from_file("app.py", default_timeout=timeout)
    app._page_hash = calc_hash("first-look")
    return app


class _BlockImport:
    """Make the named packages unimportable inside a with block."""

    def __init__(self, *names: str, evict: tuple[str, ...] = ()) -> None:
        self.names = set(names)
        # Modules that already imported a blocked package stay cached, so they are evicted too.
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
            if key.split(".")[0] in self.names
            or key in self.evict
            or any(key.startswith(f"{name}.") for name in self.evict)
        }
        return self

    def __exit__(self, *_):
        sys.meta_path.remove(self)
        sys.modules.update(self.dropped)


class DatasetIdentityTests(unittest.TestCase):
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
        app = first_look(90).run()

        app.segmented_control[0].set_value(None).run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)


class StaleModuleTests(unittest.TestCase):
    """Streamlit reruns app.py but keeps imported modules cached. SIM runs the app, adds a
    function to ada/file_io.py and an import of it to app.py, then runs the app again."""

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
        with open("ada/file_io.py", "a") as handle:
            handle.write("\n\ndef brand_new_export(frame):\n    return 'fresh'\n")
        src = open("app.py").read().replace(
            "from ada.file_io import list_excel_sheets,",
            "from ada.file_io import brand_new_export, list_excel_sheets,",
        )
        open("app.py", "w").write(src)
        run()
        import ada.file_io as file_io
        print("FRESH" if hasattr(file_io, "brand_new_export") else "STALE")
        """
    )

    def test_a_symbol_added_by_a_deploy_is_importable_without_a_restart(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as scratch:
            for path in root.glob("*.py"):
                shutil.copy(path, scratch)
            for folder in ("ada", ".streamlit", "assets"):
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


class StaleModuleCacheTests(unittest.TestCase):
    """An edited engine module is reloaded even when Streamlit's caches were cleared,
    which happens when a deploy changes a cached function's source."""

    SIM = textwrap.dedent(
        r"""
        import sys, runpy, warnings, logging, io, contextlib
        warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
        import streamlit as st
        def run():
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    runpy.run_path("app.py", run_name="__main__")
            except SystemExit:
                pass
        run()
        with open("ada/business_insights.py", "a") as handle:
            handle.write("\n\nDEPLOY_MARKER = 'fresh'\n")
        st.cache_resource.clear()
        st.cache_data.clear()
        run()
        import ada.business_insights as business_insights
        print("FRESH" if hasattr(business_insights, "DEPLOY_MARKER") else "STALE")
        """
    )

    def test_an_edited_module_is_reloaded_after_the_caches_are_cleared(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as scratch:
            shutil.copy(root / "app.py", scratch)
            for folder in ("ada", ".streamlit", "assets"):
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
    def test_the_app_is_whole_when_the_ai_layer_cannot_be_imported(self):
        with _BlockImport("pydantic", "openai", evict=("ada.ai_insights",)):
            app = first_look(90).run()

        self.assertFalse(app.exception)
        # All six tabs, four KPIs and seven charts still render.
        self.assertEqual(len(app.tabs), 6)
        self.assertEqual(len(app.metric), 4)
        self.assertEqual(len(app.get("plotly_chart")), 7)
        self.assertTrue(
            any("optional AI layer could not be loaded" in str(w.value) for w in app.warning)
        )


def _clock_log_dashboard():
    """A fifty-minute sensor log with clock times only, drawn on the dashboard."""
    import numpy as np
    import pandas as pd

    from ada.pipeline import prepare_analysis
    from ada.ui import render_dashboard

    stamps = pd.date_range("2024-01-01 15:06:28", periods=2974, freq="s").strftime("%H:%M:%S")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "Time": stamps,
        "MQ139": 104 + (np.arange(2974) > 2500) * 200 + rng.normal(0, 2, 2974),
        "Detector": ["OFF"] * 2900 + ["ON"] * 74,
    })
    prepared = prepare_analysis(frame, row_limit=1_000_000)
    render_dashboard(prepared.dataframe, prepared.detected_roles)


class DashboardForecastTests(unittest.TestCase):
    def test_a_forecast_worse_than_no_change_is_drawn_only_when_asked(self):
        app = AppTest.from_function(_clock_log_dashboard, default_timeout=60).run()

        self.assertFalse(app.exception)
        self.assertTrue(any("The forecast is not drawn" in caption.value for caption in app.caption))
        self.assertTrue(any("15:56 is still in progress" in caption.value for caption in app.caption))
        self.assertEqual([toggle.label for toggle in app.toggle], ["Draw it anyway"])

        app.toggle[0].set_value(True).run()

        self.assertFalse(app.exception)
        self.assertTrue(any(caption.value.startswith("Baseline forecast") for caption in app.caption))


class AppSmokeTests(unittest.TestCase):
    def test_demo_renders_complete_product(self):
        app = first_look().run()

        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Summary", "Ask ADA", "Charts", "Explore", "Findings", "Data"],
        )
        # Six dashboard charts, plus the one Explore draws for its default columns.
        self.assertEqual(len(app.get("plotly_chart")), 7)
        # Detected columns, cleaning audit, data checks, cleaned data and data dictionary.
        self.assertEqual(len(app.dataframe), 5)

    def test_the_demo_is_the_lab_sample(self):
        app = first_look().run()

        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("Lab Measurements · demo", rendered)
        self.assertIn("Metric · Viability %", rendered)

    def test_drill_down_focuses_the_whole_analysis(self):
        app = first_look().run()
        focus_box = next(box for box in app.selectbox if box.label.startswith("Drill into"))
        focus_box.set_value("Compound B").run()

        self.assertFalse(app.exception)
        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("Focus · Compound B", rendered)
        self.assertIn("Segment · Replicate", rendered)

    def test_combining_rows_follows_the_metric_and_can_be_switched(self):
        app = first_look().run()
        combine = next(box for box in app.selectbox if box.label == "Combine rows by")
        self.assertEqual(combine.value, "Average")
        self.assertNotIn("Total Viability %", " ".join(str(block.value) for block in app.markdown))

        combine.set_value("Total").run()

        self.assertFalse(app.exception)
        self.assertIn("Total Viability %", " ".join(str(block.value) for block in app.markdown))

    def test_ask_ada_answers_a_question(self):
        app = first_look().run()
        app.chat_input[0].set_value("average viability by treatment").run()

        self.assertFalse(app.exception)
        history = app.session_state["chat_history"]
        self.assertEqual(len(history), 1)
        self.assertIsNotNone(history[0]["result"])
        self.assertIn("Treatment", history[0]["result"].answer)

    def test_ask_ada_explains_unreadable_questions(self):
        app = first_look().run()
        app.chat_input[0].set_value("tell me a joke").run()

        self.assertFalse(app.exception)
        history = app.session_state["chat_history"]
        self.assertEqual(len(history), 1)
        self.assertIsNone(history[0]["result"])

    def test_upload_mode_waits_for_a_file(self):
        app = first_look().run()
        app.segmented_control[0].set_value("Upload your file").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.file_uploader), 1)
        self.assertEqual(len(app.tabs), 0)

    def test_repeating_a_question_does_not_break_the_transcript(self):
        """Two answers that render the same chart need distinct element ids."""
        app = first_look().run()
        app.chat_input[0].set_value("average viability by treatment").run()
        app.chat_input[0].set_value("average viability by treatment").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["chat_history"]), 2)

    def test_a_sample_dataset_can_be_analyzed_without_uploading(self):
        app = first_look().run()
        app.segmented_control[0].set_value("Try a sample dataset").run()

        picker = next(box for box in app.selectbox if box.label == "Sample dataset")
        self.assertIn("SaaS Subscriptions", picker.options)

        picker.set_value("SaaS Subscriptions").run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)
        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("SaaS Subscriptions · sample", rendered)

    def test_explore_charts_any_columns_and_says_why(self):
        app = first_look().run()
        picker = next(box for box in app.multiselect if box.label == "Columns to chart")

        picker.set_value(["Treatment", "Viability %"]).run()

        self.assertFalse(app.exception)
        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("WHY THIS CHART", rendered)

    def test_explore_handles_every_recommended_form(self):
        app = first_look().run()
        for columns in (
            ["Run Date", "Viability %"],
            ["Run Date", "Viability %", "Treatment"],
            ["Treatment", "Replicate", "Viability %"],
            ["Viability %", "Live Cells"],
            ["Treatment"],
        ):
            with self.subTest(columns=columns):
                picker = next(box for box in app.multiselect if box.label == "Columns to chart")
                picker.set_value(columns).run()
                self.assertFalse(app.exception)

    def test_explore_says_how_it_fitted_a_long_date_range(self):
        app = first_look().run()
        app.segmented_control[0].set_value("Try a sample dataset").run()
        samples = next(box for box in app.selectbox if box.label == "Sample dataset")
        samples.set_value("Ecommerce Orders").run()
        picker = next(box for box in app.multiselect if box.label == "Columns to chart")

        picker.set_value(["Order Date", "Revenue"]).run()

        self.assertFalse(app.exception)
        captions = " ".join(str(block.value) for block in app.caption)
        self.assertIn("so each point is one week and the whole range fits", captions)


if __name__ == "__main__":
    unittest.main()


class HomePageTests(unittest.TestCase):
    """The app opens on the page for researchers, which leads to the example study and the first look."""

    def test_the_app_opens_on_the_page_for_researchers(self):
        app = AppTest.from_file("app.py", default_timeout=45).run()

        self.assertFalse(app.exception)
        rendered = " ".join(str(block.value) for block in app.markdown)
        self.assertIn("For researchers", rendered)
        self.assertIn("traceable to it", rendered)
        self.assertEqual(
            [button.label for button in app.button],
            [
                "See an example study built here",
                "Look at a table now",
                "Open the example study",
                "Look at a table",
            ],
        )

    def test_the_example_study_is_built_and_shown(self):
        app = AppTest.from_file("app.py", default_timeout=180).run()
        app.button(key="home_example").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Figures", "Tables", "Methods draft", "Data audit", "Processing record", "study.yml", "Verify"],
        )
        self.assertEqual(app.metric[0].value, "17")
        self.assertEqual(app.metric[2].value, "2 of 2")
        self.assertTrue(any("Every file matches a rebuild" in str(box.value) for box in app.success))

    def test_the_first_look_is_one_click_away(self):
        app = AppTest.from_file("app.py", default_timeout=90).run()
        app.button(key="home_first_look").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)
