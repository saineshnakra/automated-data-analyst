import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ada
from ada import explain
from ada.cli import main

try:
    import lightgbm  # noqa: F401
    import sklearn  # noqa: F401
    HAS_EXTRA = True
except ImportError:
    HAS_EXTRA = False


class CompareTests(unittest.TestCase):
    """Two models are compared fold by fold, with the mean difference and its standard error."""

    def test_a_clear_win_names_what_it_means(self):
        found = explain.compare("effects bend", "additive", "linear", [0.90, 0.91, 0.92, 0.90, 0.91],
                                [0.80, 0.80, 0.81, 0.80, 0.80], True, "the effects are not straight lines")
        self.assertAlmostEqual(found.difference, 0.106)
        self.assertIn("additive is better beyond twice its standard error: the effects are not straight",
                      found.detail)

    def test_lower_is_better_turns_the_difference_round(self):
        found = explain.compare("cost", "full", "readable", [1.0, 1.1, 1.0], [2.0, 2.1, 2.0], False, "x")
        self.assertGreater(found.difference, 0)
        self.assertIn("full is better", found.verdict)

    def test_noise_is_no_clear_difference(self):
        found = explain.compare("cost", "a", "b", [0.5, 0.7, 0.4], [0.6, 0.5, 0.6], True, "x")
        self.assertIn("no clear difference", found.verdict)

    def test_one_fold_has_no_standard_error(self):
        found = explain.compare("cost", "a", "b", [0.9], [0.8], True, "x")
        self.assertIn("no clear difference", found.verdict)


class EffectTests(unittest.TestCase):
    """Effect curves are read from the additive model's contributions."""

    def test_a_straight_effect_and_a_bent_one(self):
        x = np.linspace(-3, 3, 600)
        frame = pd.DataFrame({"straight": x, "bent": x})
        contributions = np.column_stack([2 * x, np.sin(2 * x)])
        found = {effect.column: effect
                 for effect in explain.effects(contributions, frame, ["straight", "bent"])}
        self.assertIn("close to a straight line", found["straight"].shape)
        self.assertIn("bends", found["bent"].shape)
        self.assertEqual(len(found["straight"].points), explain.BANDS)

    def test_few_values_get_a_point_each_and_categories_a_level_each(self):
        frame = pd.DataFrame({"class": [1, 2, 3] * 100, "sex": pd.Categorical(["f", "m"] * 150)})
        contributions = np.column_stack([frame["class"].to_numpy() * 0.5,
                                         np.where(frame["sex"] == "f", 1.0, -1.0)])
        found = {effect.column: effect for effect in explain.effects(contributions, frame, ["class", "sex"])}
        self.assertEqual([at for at, _ in found["class"].points], ["1", "2", "3"])
        self.assertEqual(dict(found["sex"].points), {"f": 1.0, "m": -1.0})

    def test_a_column_with_no_effect_is_left_out(self):
        frame = pd.DataFrame({"used": np.arange(100.0), "unused": np.ones(100)})
        contributions = np.column_stack([np.arange(100.0), np.zeros(100)])
        found = explain.effects(contributions, frame, ["used", "unused"])
        self.assertEqual([effect.column for effect in found], ["used"])


class CoefficientTests(unittest.TestCase):
    """The linear model's coefficients and 95% intervals come from statsmodels."""

    def test_a_slope_per_standard_deviation_with_its_interval(self):
        rng = np.random.default_rng(0)
        x = rng.normal(10, 4, 2000)
        city = rng.choice(["a", "b", "c"], 2000, p=[0.5, 0.3, 0.2])
        y = 3 * x + np.where(city == "b", 5, 0) + rng.normal(0, 1, 2000)
        frame = pd.DataFrame({"x": x, "city": pd.Categorical(city), "copy": 2 * x})
        terms, note = explain.coefficients(frame, y, "regression", 1)
        found = {term.name.split(" (")[0]: term for term in terms}
        # x and its copy are the same column twice; one is kept, and per standard deviation they agree.
        slope = found["x"] if "x" in found else found["copy"]
        self.assertAlmostEqual(slope.estimate, 3 * x.std(), delta=0.1)
        self.assertLess(slope.high - slope.low, 0.2)
        self.assertIn("city = b", [term.name.split(" (")[0] for term in terms])
        self.assertIn("1 columns repeating others left out", note)

    def test_several_classes_are_explained_by_curves(self):
        terms, note = explain.coefficients(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), np.array([0, 1, 2]),
                                           "classification", 3)
        self.assertEqual(terms, ())
        self.assertIn("3 classes", note)

    def test_nothing_varies(self):
        terms, note = explain.coefficients(pd.DataFrame({"x": [1.0] * 5}), np.arange(5.0), "regression", 1)
        self.assertEqual((terms, note), ((), "no column varies"))


@unittest.skipUnless(HAS_EXTRA, "needs the model extra")
class ExplainedModelTests(unittest.TestCase):
    """`explain` predicts with a readable model and says what it costs."""

    def test_a_curved_effect_picks_the_additive_model(self):
        rng = np.random.default_rng(1)
        rows = 1500
        x, z = rng.uniform(-3, 3, rows), rng.normal(0, 1, rows)
        frame = pd.DataFrame({"x": x.round(3), "z": z.round(3),
                              "y": 4 * np.sin(2 * x) + z + rng.normal(0, 0.3, rows)})
        report = ada.model(frame[:1200], frame[1200:].drop(columns="y"), target="y", metric="rmse",
                           explain=True)
        explanation = report.explanation
        self.assertEqual(explanation.chosen, "the additive model")
        bend = next(found for found in explanation.comparisons if found.name == "effects bend")
        self.assertIn("the additive model is better", bend.verdict)
        effect = next(found for found in explanation.effects if found.column == "x")
        self.assertIn("bends", effect.shape)
        self.assertIn("readable model with the better out-of-fold score", report.calculation)
        self.assertEqual(len(report.submission), 300)
        self.assertIn("explanation", report.to_dict())
        self.assertIn("explained by: the additive model", str(report))

    def test_columns_acting_together_show_in_the_comparison(self):
        rng = np.random.default_rng(2)
        rows = 1500
        a, b = rng.choice([-1.0, 1.0], rows), rng.choice([-1.0, 1.0], rows)
        frame = pd.DataFrame({"a": a, "b": b, "noise": rng.normal(0, 1, rows).round(3),
                              "y": 3 * a * b + rng.normal(0, 0.5, rows)})
        report = ada.model(frame[:1200], frame[1200:].drop(columns="y"), target="y", metric="rmse",
                           explain=True)
        compared = {found.name: found for found in report.explanation.comparisons}
        self.assertIn("LightGBM is better", compared["columns act together"].verdict)
        cost = compared["cost of explaining"]
        self.assertGreater(cost.difference, 0)

    def test_the_command_line_flag(self):
        rng = np.random.default_rng(3)
        rows = 400
        x = rng.normal(0, 1, rows)
        frame = pd.DataFrame({"id": np.arange(rows), "x": x.round(3), "y": (x + rng.normal(0, 1, rows) > 0)})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame[:300].to_csv(root / "train.csv", index=False)
            frame[300:].drop(columns="y").to_csv(root / "test.csv", index=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(["model", str(root / "train.csv"), str(root / "test.csv"), "--target", "y",
                             "--metric", "auc", "--explain", "--json"])
        self.assertEqual(code, 0)
        found = json.loads(out.getvalue())["explanation"]
        self.assertIn(found["chosen"], ("the linear model", "the additive model"))
        self.assertEqual([item["name"] for item in found["comparisons"]],
                         ["effects bend", "columns act together", "cost of explaining"])
        self.assertTrue(found["coefficients"][0]["term"].startswith("x ("))


if __name__ == "__main__":
    unittest.main()
