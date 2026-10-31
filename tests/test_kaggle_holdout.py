import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ada.modelling import METRICS

try:
    import lightgbm  # noqa: F401
    import sklearn  # noqa: F401
    HAS_EXTRA = True
except ImportError:
    HAS_EXTRA = False

SPEC = importlib.util.spec_from_file_location(
    "holdout", Path(__file__).resolve().parents[1] / "results" / "kaggle" / "holdout.py")
holdout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(holdout)


def readings(rows: int, seed: int = 0) -> pd.DataFrame:
    """Daily readings of five stations: a level that rises with the temperature."""
    rng = np.random.default_rng(seed)
    temperature = rng.normal(15, 5, rows)
    return pd.DataFrame({
        "id": np.arange(rows),
        "station": [f"S{index % 5}" for index in range(rows)],
        "temperature": temperature.round(2),
        "level": (2 * temperature + rng.normal(0, 2, rows)).round(2),
    })


class SplitTests(unittest.TestCase):
    """The holdout is cut from the training rows the way the test rows differ from them."""

    def test_rows_by_hash_of_id(self):
        train, test = readings(1000), readings(200, seed=1).drop(columns="level")
        test["id"] += 1000
        held, how = holdout.split(train, test, None)
        self.assertEqual(int(held.sum()), 200)
        self.assertIn("by a hash of id", how)
        again, _ = holdout.split(train.sample(frac=1, random_state=3).sort_index(), test, None)
        self.assertTrue((held == again).all())

    def test_last_stretch_when_test_dates_come_later(self):
        train = readings(100).assign(date=pd.date_range("2024-01-01", periods=100).strftime("%Y-%m-%d"))
        test = readings(10, seed=1).drop(columns="level").assign(
            date=pd.date_range("2024-04-10", periods=10).strftime("%Y-%m-%d"))
        test["id"] += 100
        held, how = holdout.split(train, test, None)
        self.assertEqual(held.tolist(), [False] * 90 + [True] * 10)
        self.assertIn("the last 10 days of date", how)

    def test_a_fifth_of_groups_the_test_never_shares(self):
        train = readings(1000).assign(station=[f"S{index % 50}" for index in range(1000)])
        test = readings(100, seed=1).drop(columns="level")
        test["station"] = [f"T{index % 5}" for index in range(100)]
        test["id"] += 1000
        held, how = holdout.split(train, test, None)
        groups = set(train.loc[held, "station"])
        self.assertEqual(len(groups), 10)
        self.assertFalse(groups & set(train.loc[~held, "station"]))
        self.assertIn("a fifth of the station values (10 of 50)", how)

    def test_a_fifth_of_id_groups_the_test_never_shares(self):
        train = readings(1000).assign(id=[f"{index // 2:04d}_{index % 2 + 1:02d}" for index in range(1000)])
        test = readings(100, seed=1).drop(columns="level")
        test["id"] = [f"9{index:03d}_01" for index in range(100)]
        held, how = holdout.split(train, test, None)
        groups = train["id"].str[:4]
        self.assertEqual(groups[held].nunique(), 100)
        self.assertFalse(set(groups[held]) & set(groups[~held]))
        self.assertIn("a fifth of the groups in the first part of id (100 of 500)", how)

    def test_thin_table_still_holds_out_a_row(self):
        train, test = readings(3), readings(2, seed=1).drop(columns="level")
        test["id"] += 3
        held, _ = holdout.split(train, test, None)
        self.assertEqual(int(held.sum()), 1)


class ReadingTests(unittest.TestCase):
    """Submissions are read back as the metric scores them."""

    def test_true_false_labels_are_compared_as_labels(self):
        truth = pd.Series([True, False, True, False])
        submission = pd.DataFrame({"id": range(4), "Transported": [True, True, True, False]})
        observed, predicted, _ = holdout.readings(METRICS["accuracy"], submission, truth, [False, True],
                                                  "Transported")
        self.assertEqual(METRICS["accuracy"].score(observed, predicted), 0.75)

    def test_whole_numbers_match_however_they_were_read(self):
        truth = pd.Series([1, 0, 2])
        submission = pd.DataFrame({"id": range(3), "y": [1.0, 0.0, 1.0]})
        observed, predicted, _ = holdout.readings(METRICS["accuracy"], submission, truth, [0, 1, 2], "y")
        self.assertAlmostEqual(METRICS["accuracy"].score(observed, predicted), 2 / 3)

    def test_one_probability_column_is_the_second_class(self):
        truth = pd.Series(["no", "yes", "yes", "no"])
        submission = pd.DataFrame({"id": range(4), "y": [0.1, 0.9, 0.8, 0.3]})
        observed, predicted, _ = holdout.readings(METRICS["auc"], submission, truth, ["no", "yes"], "y")
        self.assertEqual(observed.tolist(), [0, 1, 1, 0])
        self.assertEqual(METRICS["auc"].score(observed, predicted), 1.0)

    def test_three_labels_a_row_score_by_their_order(self):
        truth = pd.Series(["a", "b", "c"])
        submission = pd.DataFrame({"id": range(3), "y": ["a b c", "a b c", "a b d"]})
        classes = ["a", "b", "c", "d"]
        observed, predicted, labels = holdout.readings(METRICS["map@3"], submission, truth, classes, "y")
        self.assertAlmostEqual(METRICS["map@3"].score(observed, predicted, labels), (1 + 1 / 2 + 0) / 3)


@unittest.skipUnless(HAS_EXTRA, "needs the model extra")
class EvaluateTests(unittest.TestCase):
    """A competition folder is split, modelled on its development part and scored on its holdout."""

    def test_holdout_is_scored_with_the_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            folder, work = Path(directory) / "data", Path(directory) / "work"
            folder.mkdir()
            train, test = readings(500), readings(100, seed=1).drop(columns="level")
            test["id"] += 500
            train.to_csv(folder / "train.csv", index=False)
            test.to_csv(folder / "test.csv", index=False)
            test.assign(level=0.0)[["id", "level"]].to_csv(folder / "sample_submission.csv", index=False)
            entry = holdout.evaluate(folder, work, "rmse")
            written = pd.read_csv(work / "test.csv")
        self.assertEqual((entry["rows_development"], entry["rows_holdout"]), (400, 100))
        self.assertLess(entry["holdout"], 3.0)
        self.assertGreater(entry["holdout_se"], 0)
        self.assertEqual(list(written.columns), ["id", "station", "temperature"])

    def test_thin_data_is_refused_with_the_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "data"
            folder.mkdir()
            train, test = readings(20), readings(5, seed=1).drop(columns="level")
            test["id"] += 20
            train.to_csv(folder / "train.csv", index=False)
            test.to_csv(folder / "test.csv", index=False)
            entry = holdout.evaluate(folder, Path(directory) / "work", "rmse")
        self.assertIn("training rows", entry["refused"])


if __name__ == "__main__":
    unittest.main()
