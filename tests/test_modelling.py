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

    def test_a_structured_id_gives_the_size_of_its_group(self):
        # "0013_02": the id names one row, and its first part a group of rows travelling together.
        sizes = np.random.default_rng(1).choice([1, 2, 3, 4], 300)
        groups = np.repeat(np.arange(len(sizes)), sizes)[:600]
        frame = self.train.copy()
        ids = [f"{group:04d}_{index:02d}" for index, group in enumerate(groups)]
        train = frame.assign(PassengerId=ids[:450])
        test = self.test.assign(PassengerId=ids[450:600])
        report = ada.model(train, test, metric="accuracy")
        self.assertIn("PassengerId_group_size", dict(report.features))
        self.assertIn(("PassengerId", "the row's id"), report.dropped)
        # A plain running number gives no group.
        self.assertNotIn("PassengerId_group_size", dict(ada.model(self.train, self.test).features))

    def test_a_wide_table_encodes_numbers_by_value(self):
        # A playground table is drawn from a smaller original: a number whose values repeat is also
        # encoded by each value's mean target and by how many rows share it. Fare is nearly unique.
        from ada import modelling
        train = self.train.assign(Age=np.random.default_rng(2).integers(1, 40, len(self.train)))
        test = self.test.assign(Age=np.random.default_rng(3).integers(1, 40, len(self.test)))
        original = modelling.WIDE_TABLE
        modelling.WIDE_TABLE = 400
        try:
            features = dict(ada.model(train, test, metric="auc").features)
        finally:
            modelling.WIDE_TABLE = original
        self.assertIn("Age_value", features)
        self.assertIn("Age_value_count", features)
        self.assertIn("Pclass_value", features)
        self.assertNotIn("Fare_value", features)
        # Below the size, numbers stay as they are.
        self.assertNotIn("Age_value", dict(ada.model(train, test, metric="auc").features))

    def test_xgboost_joins_the_blend_when_installed(self):
        from ada import modelling
        report = ada.model(self.train, self.test, metric="auc")
        if modelling._xgboost_installed():
            self.assertIn("XGBoost", report.calculation)
        original = modelling._xgboost_installed
        modelling._xgboost_installed = lambda: False
        try:
            without = ada.model(self.train, self.test, metric="auc")
        finally:
            modelling._xgboost_installed = original
        self.assertNotIn("XGBoost", without.calculation)
        self.assertGreater(without.score, 0.7)

    def test_three_learners_are_weighed_together(self):
        from ada import modelling
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        good = (y + rng.normal(0, 0.3, 400))[:, None]
        noise = rng.random((400, 1))
        weights = modelling._blend_weights({"lightgbm": noise, "linear": noise, "xgboost": good},
                                           modelling.METRICS["auc"], "classification", np.array([0, 1]), y,
                                           None)
        self.assertEqual(weights["xgboost"], 1.0)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

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

    def test_a_skewed_target_is_fitted_on_logs_for_absolute_error(self):
        rng = np.random.default_rng(5)
        size = rng.integers(40, 300, 400)
        frame = pd.DataFrame({"id": np.arange(400), "size": size,
                              "price": np.round(size * 900 * rng.lognormal(0, 0.6, 400))})
        report = ada.model(frame.iloc[:300], frame.iloc[300:].drop(columns="price"), metric="mae")
        self.assertIn("log1p(price)", report.calculation)
        self.assertGreater(report.submission["price"].min(), 0)
        self.assertGreater(report.test_wobble(50), 0)
        even = frame.assign(price=rng.normal(100, 10, 400))
        report = ada.model(even.iloc[:300], even.iloc[300:].drop(columns="price"), metric="mae")
        self.assertNotIn("log1p", report.calculation)

    def test_numbers_written_as_text_in_one_table_are_read_as_numbers_in_both(self):
        from ada.modelling import _align_types

        train = pd.DataFrame({"cars": [0, 1, 2, 2], "name": ["a", "b", "c", "d"]})
        test = pd.DataFrame({"cars": ["1", "NA", "3", "2"], "name": ["e", "f", "g", "h"]})
        self.assertEqual(_align_types(train, test, ["cars", "name"]), ["cars"])
        self.assertTrue(pd.api.types.is_numeric_dtype(test["cars"]))
        self.assertTrue(test["cars"].isna().iloc[1])
        words = pd.DataFrame({"cars": ["one", "two", "NA", "3"]})
        self.assertEqual(_align_types(train, words, ["cars"]), [])

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


def stocks(codes: list[str], rows_each: int, seed: int) -> pd.DataFrame:
    """Rows of several stocks; y follows f_0 the same way in every stock, around a level of its own."""
    rng = np.random.default_rng(seed)
    frames = []
    for code in codes:
        signal = rng.normal(0, 1, rows_each)
        frames.append(pd.DataFrame({
            "code": code, "f_0": signal, "f_1": rng.normal(0, 1, rows_each),
            "y": 0.5 * signal + rng.normal(0, 1) + rng.normal(0, 1, rows_each),
        }))
    frame = pd.concat(frames, ignore_index=True)
    return frame.assign(id=np.arange(len(frame)) + 1000 * seed)


@unittest.skipUnless(HAS_EXTRA, "needs the model extra")
class ModelGroupTests(unittest.TestCase):
    def test_test_rows_of_new_groups_group_the_folds(self):
        train = stocks([f"code_{index}" for index in range(8)], 60, 0)
        test = stocks([f"code_{index}" for index in range(8, 12)], 60, 1).drop(columns="y")
        report = ada.model(train, test, target="y", metric="RankIC", id_column="id")
        self.assertEqual(report.metric.name, "spearman")
        self.assertIn("code-grouped", report.calculation)
        self.assertIn("folds are grouped by it", dict(report.dropped)["code"])
        self.assertGreater(report.score, 0.2)
        self.assertLessEqual(report.score, 1)

    def test_groups_the_test_rows_share_with_training_do_not_group_the_folds(self):
        train = stocks([f"code_{index}" for index in range(8)], 60, 0)
        test = stocks([f"code_{index}" for index in range(6, 10)], 60, 1).drop(columns="y")
        report = ada.model(train, test, target="y", metric="rmse", id_column="id")
        self.assertNotIn("grouped", report.calculation)

    def test_fewer_groups_than_folds_do_not_group_the_folds(self):
        train = stocks(["a", "b", "c"], 100, 0)
        test = stocks(["d", "e"], 50, 1).drop(columns="y")
        report = ada.model(train, test, target="y", metric="rmse", id_column="id")
        self.assertNotIn("grouped", report.calculation)

    def test_each_fold_holds_whole_groups(self):
        from ada.modelling import _group_folds

        groups = pd.Series(["a"] * 5 + ["b"] * 3 + ["c"] * 3 + ["d"] * 2 + ["e"] * 1)
        folds = _group_folds(groups, 3, seed=0)
        held = [set(groups.iloc[rows]) for _, rows in folds]
        self.assertEqual(sorted(name for names in held for name in names), ["a", "b", "c", "d", "e"])
        self.assertEqual(sorted(len(rows) for _, rows in folds), [4, 5, 5])


def daily_sales(days: int, stores: int = 3, seed: int = 0) -> pd.DataFrame:
    """Daily sales of two families in each store, with a weekly cycle and a level per store."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for store in range(1, stores + 1):
        for family in ("BREAD", "MILK"):
            level = 20 * store + (5 if family == "MILK" else 0)
            weekly = 1 + 0.4 * (dates.dayofweek >= 5)
            sales = np.round(level * weekly * rng.lognormal(0, 0.1, days))
            rows.append(pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "store_nbr": store,
                                      "family": family, "onpromotion": rng.integers(0, 3, days),
                                      "sales": sales}))
    frame = pd.concat(rows, ignore_index=True)
    return frame.assign(id=np.arange(len(frame)))


@unittest.skipUnless(HAS_EXTRA, "needs the model extra")
class ForecastTests(unittest.TestCase):
    def test_test_dates_after_training_give_time_folds_and_lags(self):
        frame = daily_sales(300)
        cut = pd.to_datetime(frame["date"]) <= pd.Timestamp("2020-01-01") + pd.Timedelta(days=285)
        train, test = frame[cut], frame[~cut].drop(columns="sales")
        report = ada.model(train, test, target="sales", metric="rmsle", id_column="id")
        self.assertIn("time-ordered folds", report.calculation)
        self.assertIn("store_nbr, family", report.calculation)
        self.assertIn("holding out 14 days of date", report.calculation)
        names = dict(report.features)
        self.assertIn("sales_lag14", names)
        self.assertIn("date_dayofweek", names)
        self.assertEqual(len(report.folds), 5)
        self.assertLess(len(report.out_of_fold_predictions), len(train))
        self.assertEqual(len(report.submission), len(test))
        self.assertLess(report.score, 0.2)
        self.assertTrue(all(check.passed for check in report.checks), report.checks)

    def test_a_series_that_sold_nothing_lately_predicts_zero(self):
        frame = daily_sales(300)
        # Store 3's bread stops selling after day 200.
        stopped = (frame["store_nbr"] == 3) & (frame["family"] == "BREAD") & (
            pd.to_datetime(frame["date"]) > pd.Timestamp("2020-01-01") + pd.Timedelta(days=200))
        frame.loc[stopped, "sales"] = 0.0
        cut = pd.to_datetime(frame["date"]) <= pd.Timestamp("2020-01-01") + pd.Timedelta(days=285)
        train, test = frame[cut], frame[~cut].drop(columns="sales")
        report = ada.model(train, test, target="sales", metric="rmsle", id_column="id")
        self.assertIn("predict 0", report.calculation)
        quiet = test.loc[(test["store_nbr"] == 3) & (test["family"] == "BREAD"), "id"]
        predicted = report.submission.set_index("id")["sales"]
        self.assertTrue((predicted.loc[quiet] == 0).all())
        self.assertTrue((predicted.drop(quiet) > 0).all())

    def test_a_lag_is_never_nearer_than_the_horizon(self):
        from ada.modelling import _lag_features

        dates = pd.Series(pd.date_range("2021-01-01", periods=60, freq="D"))
        train = pd.DataFrame({"date": dates[:50].to_numpy(), "shop": "a"})
        test = pd.DataFrame({"date": dates[50:].to_numpy(), "shop": "a"})
        values = np.arange(50, dtype=float)
        past, ahead, _ = _lag_features(train, test, "date", ["shop"], values, 10, "y")
        self.assertEqual(past["y_lag10"].iloc[30], 20)
        self.assertEqual(past["y_mean7"].iloc[30], np.mean(np.arange(14, 21)))
        self.assertTrue(np.isnan(past["y_lag10"].iloc[5]))
        self.assertEqual(ahead["y_lag10"].iloc[-1], 49)
        self.assertFalse(ahead.isna().any().any())

    def test_too_little_history_keeps_shuffled_folds_and_says_so(self):
        frame = daily_sales(40)
        cut = pd.to_datetime(frame["date"]) <= pd.Timestamp("2020-01-01") + pd.Timedelta(days=19)
        train, test = frame[cut], frame[~cut].drop(columns="sales")
        report = ada.model(train, test, target="sales", metric="rmse", id_column="id")
        self.assertNotIn("time-ordered", report.calculation)
        self.assertTrue(any("flatters" in warning for warning in report.warnings))


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
