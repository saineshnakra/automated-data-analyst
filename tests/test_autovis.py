import unittest

import numpy as np
import pandas as pd

from autovis import MAX_SERIES, ChartSpec, fold_small_series, recommend_chart


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

        # 360,000 cells is not a picture, and it exceeds the websocket limit
        # before it reaches a screen.
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

        # The real "Other" segment keeps its own money.
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

if __name__ == "__main__":
    unittest.main()