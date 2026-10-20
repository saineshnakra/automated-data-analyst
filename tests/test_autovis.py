import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from ada import ui
from ada.autovis import MAX_SERIES, ChartSpec, fold_small_series, recommend_chart
from ada.pipeline import prepare_analysis
from ada.schema import detect_roles


def frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "Order Date": pd.date_range("2024-01-01", periods=rows, freq="D"),
            "Revenue": rng.gamma(3, 400, rows).round(2),
            "Units": rng.integers(1, 9, rows),
            "Product": rng.choice(["Alpha", "Beta", "Gamma"], rows),
            "Region": rng.choice(["West", "East"], rows),
        }
    )


class FormChoiceTests(unittest.TestCase):
    def setUp(self):
        self.frame = frame()

    def spec(self, columns: list[str]) -> ChartSpec:
        return recommend_chart(self.frame, columns)

    def test_time_and_a_measure_is_a_trend(self):
        spec = self.spec(["Order Date", "Revenue"])
        self.assertEqual(spec.form, "area")
        self.assertEqual((spec.x, spec.y), ("Order Date", "Revenue"))

    def test_a_segment_over_time_becomes_one_line_each(self):
        spec = self.spec(["Order Date", "Revenue", "Product"])
        self.assertEqual(spec.form, "line")
        self.assertEqual(spec.color, "Product")

    def test_a_category_and_a_measure_compare_magnitude(self):
        spec = self.spec(["Product", "Revenue"])
        self.assertIn(spec.form, ("bar", "column"))
        self.assertEqual(spec.y, "Revenue")

    def test_two_categories_and_a_measure_make_a_grid(self):
        spec = self.spec(["Product", "Region", "Revenue"])
        self.assertEqual(spec.form, "heatmap")
        self.assertEqual(spec.color, "Revenue")

    def test_two_measures_ask_whether_they_move_together(self):
        spec = self.spec(["Revenue", "Units"])
        self.assertEqual(spec.form, "scatter")

    def test_a_single_value_is_not_a_chart(self):
        single = pd.DataFrame({"Revenue": [42.0]})
        self.assertEqual(recommend_chart(single, ["Revenue"]).form, "stat")

    def test_nothing_selected_asks_for_a_column(self):
        self.assertEqual(self.spec([]).form, "none")

    def test_too_many_categories_become_a_table(self):
        wide = pd.DataFrame(
            {"Account": [f"a{n}" for n in range(80)], "Revenue": range(80)}
        )
        self.assertEqual(recommend_chart(wide, ["Account", "Revenue"]).form, "table")

    def test_long_category_names_lay_the_bars_down(self):
        wordy = pd.DataFrame(
            {
                "Account Name": ["A very long account name indeed", "Another lengthy one"] * 10,
                "Revenue": range(20),
            }
        )
        self.assertEqual(recommend_chart(wordy, ["Account Name", "Revenue"]).form, "bar")

    def test_a_second_measure_is_declined_rather_than_given_its_own_axis(self):
        spec = self.spec(["Order Date", "Revenue", "Units"])
        self.assertEqual(spec.y, "Revenue")
        self.assertTrue(any("two scales" in note for note in spec.notes))

    def test_every_recommendation_explains_itself(self):
        for columns in (["Order Date", "Revenue"], ["Product", "Revenue"], ["Revenue", "Units"]):
            with self.subTest(columns=columns):
                self.assertTrue(self.spec(columns).rationale.strip())


class SeriesFoldingTests(unittest.TestCase):
    def test_the_tail_folds_into_other_rather_than_growing_hues(self):
        rng = np.random.default_rng(9)
        many = pd.DataFrame(
            {"Product": [f"p{n}" for n in range(12)] * 5, "Revenue": rng.integers(1, 99, 60)}
        )

        folded = fold_small_series(many, "Product", "Revenue")

        self.assertLessEqual(folded["Product"].nunique(), MAX_SERIES + 1)
        self.assertIn("Other", set(folded["Product"]))
        self.assertEqual(folded["Revenue"].sum(), many["Revenue"].sum())

    def test_a_small_number_of_series_is_left_alone(self):
        small = pd.DataFrame({"Product": ["A", "B"] * 5, "Revenue": range(10)})

        self.assertEqual(set(fold_small_series(small, "Product", "Revenue")["Product"]), {"A", "B"})


class ChartSizeTests(unittest.TestCase):
    def test_two_high_cardinality_categories_do_not_become_a_heatmap(self):
        rows = 4_000
        frame = pd.DataFrame(
            {
                "Customer ID": [f"c{index % 900}" for index in range(rows)],
                "SKU": [f"s{index % 400}" for index in range(rows)],
                "Revenue": [1.0] * rows,
            }
        )

        spec = recommend_chart(frame, ["Customer ID", "SKU", "Revenue"])

        # 900 x 400 = 360,000 cells, too many to draw or send to the browser.
        self.assertEqual(spec.form, "table")
        self.assertIn("cell grid", spec.rationale)

    def test_a_small_grid_is_still_a_heatmap(self):
        frame = pd.DataFrame(
            {
                "Region": ["N", "S"] * 20,
                "Product": ["a", "b", "c", "d"] * 10,
                "Revenue": [1.0] * 40,
            }
        )

        spec = recommend_chart(frame, ["Region", "Product", "Revenue"])

        self.assertEqual(spec.form, "heatmap")

    def test_a_lone_continuous_measure_is_binned_not_counted_per_value(self):
        frame = pd.DataFrame({"Revenue": [float(index) / 7 for index in range(500)]})

        spec = recommend_chart(frame, ["Revenue"])

        self.assertEqual(spec.form, "histogram")


class FoldSmallSeriesTests(unittest.TestCase):
    def test_a_real_other_category_is_not_absorbed_by_the_folded_tail(self):
        frame = pd.DataFrame(
            {
                "Region": ["Other", "N", "S", "E", "W", "X", "Y"],
                "Revenue": [20_000.0, 1_200.0, 1_000.0, 800.0, 400.0, 300.0, 100.0],
            }
        )

        folded = fold_small_series(frame, "Region", "Revenue")
        totals = folded.groupby("Region", observed=True)["Revenue"].sum()

        # The existing "Other" region keeps its own total; the folded tail is "Other (2)".
        self.assertEqual(totals["Other"], 20_000.0)
        self.assertIn("Other (2)", totals.index)

    def test_a_categorical_column_can_be_folded(self):
        frame = pd.DataFrame(
            {
                "Product": pd.Categorical([f"p{index}" for index in range(8)]),
                "Revenue": [float(8 - index) for index in range(8)],
            }
        )

        folded = fold_small_series(frame, "Product", "Revenue")

        self.assertIn("Other", set(folded["Product"]))


class ExploreFrameTests(unittest.TestCase):
    def test_the_other_series_of_an_average_is_weighted_by_rows(self):
        day = pd.Timestamp("2024-01-01")
        rows = [(day, f"Big{i}", 10.0) for i in range(MAX_SERIES) for _ in range(99)]
        rows += [(day, "Small", 47.0)] * 5 + [(day, "Rest", 10.0)] * 99
        frame = pd.DataFrame(rows, columns=["Date", "Site", "pH"])
        spec = ChartSpec(form="line", x="Date", y="pH", color="Site", aggregation="mean", rationale="")
        plotted = ui._explore_frame(frame, spec)
        other = plotted.loc[plotted["Site"].astype(str).str.startswith("Other"), "pH"]
        # 5 rows at 47 and 99 at 10, not the mean of the two series means (28.5).
        self.assertAlmostEqual(float(other.iloc[0]), (5 * 47 + 99 * 10) / 104)

    def test_a_category_named_records_can_be_explored(self):
        frame = pd.DataFrame(
            {"Date": pd.date_range("2024-01-01", periods=60), "Records": ["x", "y", "z"] * 20,
             "Revenue": np.arange(60.0)}
        )
        prepared = prepare_analysis(frame, row_limit=1000)

        spec = recommend_chart(prepared.dataframe, ["Records"])
        shown = ui._explore_frame(prepared.dataframe, spec)

        self.assertEqual(len(shown), 3)


class ExploreDateAxisTests(unittest.TestCase):
    def monthly(self, periods: int, **columns) -> pd.DataFrame:
        frame = pd.DataFrame({
            "Date": pd.date_range("1958-03-01", periods=periods, freq="MS"),
            "pH": np.linspace(7.0, 8.0, periods),
        })
        return frame.assign(**columns)

    def test_a_long_monthly_series_is_drawn_by_quarter_from_its_first_reading(self):
        frame = self.monthly(750)
        spec = recommend_chart(frame, ["Date", "pH"])

        rows, caption = ui._fit_date_axis(frame, spec)
        shown = ui._explore_frame(rows, spec)

        self.assertLessEqual(len(shown), ui.EXPLORE_POINTS)
        self.assertEqual(shown["Date"].min(), pd.Timestamp("1958-01-01"))
        self.assertEqual(shown["Date"].max(), pd.Timestamp("2020-07-01"))
        self.assertIn("each point is one quarter", caption)

    def test_a_short_series_is_drawn_as_it_is(self):
        frame = self.monthly(120)
        spec = recommend_chart(frame, ["Date", "pH"])

        rows, caption = ui._fit_date_axis(frame, spec)

        self.assertIs(rows, frame)
        self.assertIsNone(caption)

    def test_a_range_too_long_even_for_years_says_how_much_is_shown(self):
        frame = self.monthly(360)
        spec = recommend_chart(frame, ["Date", "pH"])

        rows, caption = ui._fit_date_axis(frame, spec, limit=10)
        shown = ui._explore_frame(rows, spec, limit=10)

        # March 1958 to February 1988 touches 31 calendar years.
        self.assertEqual(caption, "Showing the latest 10 of 31 years; earlier ones are not drawn.")
        self.assertEqual(shown["Date"].dt.year.tolist(), list(range(1979, 1989)))

    def test_every_series_keeps_its_older_dates(self):
        frame = pd.concat(
            [self.monthly(300, Site=site) for site in ("North", "South", "East")], ignore_index=True
        )
        spec = recommend_chart(frame, ["Date", "pH", "Site"])

        rows, caption = ui._fit_date_axis(frame, spec)
        shown = ui._explore_frame(rows, spec)

        self.assertIsNone(caption)
        self.assertEqual(shown["Date"].nunique(), 300)
        self.assertEqual(len(shown), 900)


class AggregationDefaultTests(unittest.TestCase):
    def test_an_averaged_measure_over_time_is_a_line(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "pH": np.linspace(6.8, 7.4, 12),
        })
        spec = recommend_chart(frame, ["Date", "pH"])
        self.assertEqual(spec.form, "line")
        self.assertEqual(spec.aggregation, "mean")

    def test_an_amount_keeps_the_filled_area(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "Revenue": np.linspace(100, 200, 12),
        })
        spec = recommend_chart(frame, ["Date", "Revenue"])
        self.assertEqual(spec.form, "area")
        self.assertEqual(spec.aggregation, "sum")

    def test_the_roles_override_reaches_the_chart_of_their_measure_only(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=12, freq="MS"),
            "Rainfall (mm)": np.linspace(20, 80, 12),
            "pH": np.linspace(6.8, 7.4, 12),
        })
        roles = replace(detect_roles(frame), measure="Rainfall (mm)", aggregation="sum")

        self.assertEqual(recommend_chart(frame, ["Date", "Rainfall (mm)"]).form, "line")
        self.assertEqual(recommend_chart(frame, ["Date", "Rainfall (mm)"], roles=roles).form, "area")
        self.assertEqual(recommend_chart(frame, ["Date", "pH"], roles=roles).aggregation, "mean")


if __name__ == "__main__":
    unittest.main()