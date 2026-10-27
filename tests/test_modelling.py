import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import ada
from ada.api import Refusal
from ada.cli import main

try:
    import lightgbm  # noqa: F401
    import sklearn  # noqa: F401
    HAS_EXTRA = True
except ImportError:
    HAS_EXTRA = False


def passengers(rows: int, seed: int = 0) -> pd.DataFrame:
    """People with a cabin code, a titled name and a fare; women and first class mostly survive."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], rows)
    klass = rng.choice([1, 2, 3], rows, p=[0.25, 0.25, 0.5])
    title = np.where(sex == "female", rng.choice(["Mrs", "Miss"], rows), rng.choice(["Mr", "Master"], rows))
    chance = 0.15 + 0.6 * (sex == "female") + 0.15 * (klass == 1)
    return pd.DataFrame({
        "PassengerId": np.arange(1, rows + 1),
        "Name": [f"Surname{index % 97}, {title[index]}. Given{index}" for index in range(rows)],
        "Sex": sex,
        "Pclass": klass,
        "Cabin": [f"{rng.choice(list('ABCDEFG'))}/{rng.integers(1, 300)}/{rng.choice(['P', 'S'])}"
                  for _ in range(rows)],
        "Fare": np.round(rng.gamma(2, 15, rows) * (4 - klass), 2),
        "Survived": (rng.random(rows) < chance).astype(int),
    })


@unittest.skipUnless(HAS_EXTRA, "needs the model extra")
class ModelTests(unittest.TestCase):
    def setUp(self):
        frame = passengers(600)
        self.train = frame.iloc[:450].reset_index(drop=True)
        self.test = frame.iloc[450:].drop(columns="Survived").reset_index(drop=True)

    def test_a_classifier_is_cross_validated_and_predicts_every_test_row(self):
        report = ada.model(self.train, self.test, metric="accuracy")
        self.assertEqual((report.target, report.task, report.metric.name),
                         ("Survived", "classification", "accuracy"))
        self.assertGreater(report.score, 0.7)
        self.assertEqual(len(report.folds), 15)
        self.assertEqual(list(report.submission.columns), ["PassengerId", "Survived"])
        self.assertEqual(report.submission["PassengerId"].tolist(), self.test["PassengerId"].tolist())
        self.assertTrue(all(check.passed for check in report.checks), report.checks)
        self.assertIn(("PassengerId", "the row's id"), report.dropped)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_the_same_data_give_the_same_model(self):
        first = ada.model(self.train, self.test, metric="auc")
        again = ada.model(self.train, self.test, metric="auc")
        self.assertEqual(first.folds, again.folds)
        pd.testing.assert_frame_equal(first.submission, again.submission)

    def test_text_columns_give_parts_words_and_counts(self):
        report = ada.model(self.train, self.test, metric="accuracy")
        names = dict(report.features)
        self.assertIn("Cabin_part1", names)
        self.assertIn("split at '/'", names["Cabin_part1"])
        self.assertIn("Name_word", names)
        self.assertIn("Name_count", names)

    def test_a_probability_metric_writes_probabilities(self):
        report = ada.model(self.train, self.test, metric="Roc Auc Score")
        values = report.submission["Survived"]
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        self.assertGreater(values.nunique(), 10)
        self.assertLess(abs(report.test_wobble(100) - report.test_wobble(100)), 1e-12)

    def test_a_leaked_target_is_warned_about(self):
        leaky = self.train.assign(Outcome_copy=self.train["Survived"] * 2 + 1)
        test = self.test.assign(Outcome_copy=1)
        report = ada.model(leaky, test, target="Survived", metric="accuracy")
        warned = [text for text in report.warnings if "nearly perfect" in text or "carries" in text]
        self.assertTrue(warned)

    def test_test_rows_unlike_the_training_rows_fail_the_shift_check(self):
        shifted = self.test.assign(Fare=self.test["Fare"] + 1000)
        report = ada.model(self.train, shifted, metric="accuracy")
        shift = next(check for check in report.checks if check.name == "test like training")
        self.assertFalse(shift.passed)

    def test_a_regression_on_a_logged_scale(self):
        rng = np.random.default_rng(3)
        size = rng.integers(40, 300, 400)
        frame = pd.DataFrame({"id": np.arange(400), "size": size,
                              "rooms": size // 30 + rng.integers(0, 3, 400),
                              "price": np.round(size * 900 * rng.lognormal(0, 0.2, 400))})
        report = ada.model(frame.iloc[:300], frame.iloc[300:].drop(columns="price"), metric="rmsle")
        self.assertEqual(report.task, "regression")
        self.assertIn("log1p(price)", report.calculation)
        self.assertLess(report.score, 0.4)
        self.assertTrue((report.submission["price"] > 0).all())

    def test_balanced_accuracy_gives_rare_classes_their_share(self):
        rng = np.random.default_rng(4)
        signal = rng.normal(0, 1, 900)
        label = np.where(signal > 1.6, "rare", np.where(signal > 0, "middle", "common"))
        frame = pd.DataFrame({"row": np.arange(900), "signal": signal + rng.normal(0, 0.3, 900),
                              "label": label})
        report = ada.model(frame.iloc[:700], frame.iloc[700:].drop(columns="label"), target="label",
                           metric="Balanced Accuracy Score", id_column="row")
        self.assertIn("rare", set(report.submission["label"]))
        self.assertGreater(report.score, 0.6)


@unittest.skipUnless(HAS_EXTRA, "needs the model extra")
class ModelFolderTests(unittest.TestCase):
    def folder(self, train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        train.to_csv(root / "train.csv", index=False)
        test.to_csv(root / "test.csv", index=False)
        if sample is not None:
            sample.to_csv(root / "sample_submission.csv", index=False)
        return root

    def test_a_kaggle_folder_names_the_target_and_the_layout(self):
        frame = passengers(500).rename(columns={"Survived": "Transported"})
        frame["Transported"] = frame["Transported"].astype(bool)
        test = frame.iloc[400:].drop(columns="Transported")
        sample = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": False})
        root = self.folder(frame.iloc[:400], test, sample)
        report = ada.model(root, metric="accuracy")
        self.assertEqual(report.target, "Transported")
        self.assertEqual(report.submission["Transported"].dtype, bool)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["model", str(root), "--metric", "accuracy", "--out", str(root / "submission.csv")])
        self.assertEqual(code, 0)
        written = pd.read_csv(root / "submission.csv")
        self.assertEqual(list(written.columns), ["PassengerId", "Transported"])
        self.assertEqual(len(written), 100)


@unittest.skipUnless(HAS_EXTRA, "needs the model extra")
class ModelRefusalTests(unittest.TestCase):
    def test_too_few_rows(self):
        frame = passengers(140)
        report = ada.model(frame.iloc[:99], frame.iloc[100:].drop(columns="Survived"))
        self.assertIsInstance(report, Refusal)
        self.assertEqual(report.code, "too_few_rows")
        self.assertIn("at least 100", report.message)

    def test_exactly_enough_rows(self):
        frame = passengers(140)
        report = ada.model(frame.iloc[:100], frame.iloc[100:].drop(columns="Survived"))
        self.assertNotIsInstance(report, Refusal)

    def test_a_class_too_rare_to_cross_validate(self):
        frame = passengers(300)
        frame.loc[:2, "Survived"] = 2
        report = ada.model(frame.iloc[:250], frame.iloc[250:].drop(columns="Survived"))
        self.assertEqual(report.code, "too_few_rows")
        self.assertIn("Survived = 2 has 3 rows", report.message)

    def test_no_test_table_an_unknown_metric_and_no_single_target(self):
        frame = passengers(300)
        self.assertEqual(ada.model(frame).code, "no_target")
        test = frame.iloc[250:].drop(columns=["Survived", "Fare"])
        self.assertEqual(ada.model(frame.iloc[:250], test).code, "no_target")
        self.assertEqual(ada.model(frame.iloc[:250], test, target="Survived", metric="kappa").code,
                         "unknown_metric")

    def test_the_command_line_reports_a_refusal(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        passengers(50).to_csv(root / "train.csv", index=False)
        passengers(20).drop(columns="Survived").to_csv(root / "test.csv", index=False)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["model", str(root)]), 3)
        self.assertIn("too_few_rows", err.getvalue())


if __name__ == "__main__":
    unittest.main()
