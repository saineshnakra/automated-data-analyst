"""Predict a column of a table from its other columns, as a tabular Kaggle competition asks.

    import ada
    report = ada.model("titanic/", metric="accuracy")    # a folder with train.csv and test.csv
    report.score          # the mean cross-validated score; each fold's is in report.folds
    report.submission     # one row per test row, in the sample submission's layout
    report.write("submission.csv")

Every score is out of fold: each training row is predicted by models that
never saw it, and the test rows by the average of the fold models. The
features come from rules on the shape of each column (see ``build_features``),
never from knowledge of one competition. The model is LightGBM gradient
boosting, averaged over folds and, on small tables, over several seeds.

Checks run around the fit: a score so close to perfect that a column may leak
the target, a test set unlike the training rows, and the submission's rows,
ids and values. A model that can't be fitted honestly (no target, too few rows,
a class too rare to cross-validate) gives a ``Refusal`` with the reason.

Needs the optional extra: ``pip install "automated-data-analyst[model]"``
(scikit-learn and LightGBM).
"""

from __future__ import annotations

import inspect
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .formatting import normalized_name
from .numeric import to_numbers

#: Fewest training rows a model is cross-validated on.
MIN_MODEL_ROWS = 100
#: A text column with at most this many distinct values is a category.
MAX_CATEGORIES = 64
#: Most common words of a text column offered as a feature, and the share of rows a word needs.
FREQUENT_WORDS = 40
MIN_WORD_SHARE = 0.01
#: Below this many rows, cross-validation is repeated with more seeds, as small folds are noisy.
SMALL_TABLE = 20_000
#: From this many rows, boosting takes larger steps and stops sooner, so a fit takes minutes, not hours.
LARGE_TABLE = 1_000_000
#: From this many rows, trees grow larger, and numbers whose values repeat are also encoded by value.
WIDE_TABLE = 100_000
#: Most distinct values, as a share of the rows, a number may have to be encoded by value.
VALUE_SHARE = 0.1
#: A cross-validated score this close to perfect suggests a column that leaks the target.
NEAR_PERFECT = 0.995
#: Train-against-test AUC above which the test rows count as unlike the training rows.
SHIFT_AUC = 0.8
#: Rows of each side the shift check samples.
SHIFT_SAMPLE = 20_000
#: Skewness of a target at 0 or above beyond which an absolute error is fitted on log(1 + target).
LOG_SKEW = 1.0
#: Fewest rows per value, on average, before a column whose test values are all new groups the folds.
MIN_GROUP_ROWS = 2
#: A text column is read as dates when this share of a sample of its values is written as ISO dates.
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?")
#: Most columns tried, and most at once, for the columns that name each series of a forecast.
SERIES_CANDIDATES, SERIES_SIZE = 6, 3
#: Share of the series-by-date grid the training rows must fill to count as series over time.
SERIES_COVER = 0.5
#: Widths in days of the target's past means offered to a daily forecast.
LAG_WINDOWS = (7, 14, 28, 56, 112)
#: Days back to the same weekday a year earlier, for series with two years of history.
YEAR_DAYS = 364
# A series whose target was 0 on every day of this many before a forecast predicts 0, when that scores better.
QUIET_DAYS = 28


@dataclass(frozen=True)
class Metric:
    """How a competition scores predictions: the name, whether higher is better and what it scores."""

    name: str
    higher: bool
    #: "label" (classes), "probability" (of the positive class or each class) or "value" (a number).
    output: str
    task: str

    def score(self, truth: np.ndarray, predicted: np.ndarray, classes: np.ndarray | None = None) -> float:
        from sklearn import metrics  # noqa: PLC0415

        if self.name == "accuracy":
            return float(metrics.accuracy_score(truth, predicted))
        if self.name == "balanced_accuracy":
            return float(metrics.balanced_accuracy_score(truth, predicted))
        if self.name == "f1":
            average = "binary" if len(set(truth)) <= 2 else "macro"
            return float(metrics.f1_score(truth, predicted, average=average))
        if self.name == "auc":
            if predicted.ndim == 2:
                return float(metrics.roc_auc_score(truth, predicted, multi_class="ovr", labels=classes))
            return float(metrics.roc_auc_score(truth, predicted))
        if self.name == "logloss":
            return float(metrics.log_loss(truth, predicted, labels=classes))
        if self.name == "map@3":
            top = np.argsort(-predicted, axis=1)[:, :3]
            index = {label: position for position, label in enumerate(classes)}
            wanted = np.array([index[label] for label in truth])
            hits = top == wanted[:, None]
            return float((hits / np.arange(1, 4)).sum(axis=1).mean())
        if self.name == "rmse":
            return float(np.sqrt(metrics.mean_squared_error(truth, predicted)))
        if self.name == "rmsle":
            return float(np.sqrt(metrics.mean_squared_log_error(truth, np.clip(predicted, 0, None))))
        if self.name == "mae":
            return float(metrics.mean_absolute_error(truth, predicted))
        if self.name == "spearman":
            from scipy.stats import spearmanr  # noqa: PLC0415

            return float(spearmanr(truth, predicted).statistic)
        raise ValueError(f"Unknown metric {self.name}")


METRICS = {
    "accuracy": Metric("accuracy", True, "label", "classification"),
    "balanced_accuracy": Metric("balanced_accuracy", True, "label", "classification"),
    "f1": Metric("f1", True, "label", "classification"),
    "auc": Metric("auc", True, "probability", "classification"),
    "logloss": Metric("logloss", False, "probability", "classification"),
    "map@3": Metric("map@3", True, "top3", "classification"),
    "rmse": Metric("rmse", False, "value", "regression"),
    "rmsle": Metric("rmsle", False, "value", "regression"),
    "mae": Metric("mae", False, "value", "regression"),
    "spearman": Metric("spearman", True, "value", "regression"),
}
#: Other names for the metrics, as Kaggle and scikit-learn write them.
METRIC_NAMES = {
    "categorization accuracy": "accuracy", "accuracy score": "accuracy", "roc auc score": "auc",
    "roc_auc": "auc",
    "balanced accuracy score": "balanced_accuracy", "balanced accuracy": "balanced_accuracy",
    "area under roc curve": "auc", "log loss": "logloss", "multiclass loss": "logloss", "map@3": "map@3",
    "mean average precision @ 3": "map@3",
    "root mean squared error": "rmse", "root mean squared logarithmic error": "rmsle",
    "mean absolute error": "mae", "f-score": "f1", "f-score (micro)": "f1", "f1 score": "f1",
    "rankic": "spearman", "rank ic": "spearman", "spearman correlation": "spearman",
    "spearman rank correlation": "spearman", "spearman's rank correlation": "spearman",
}


def metric_named(name: str) -> Metric | None:
    key = name.strip().lower()
    return METRICS.get(METRIC_NAMES.get(key, key))


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ModelReport:
    """A cross-validated model and its predictions for the test rows."""

    target: str
    task: str
    metric: Metric
    folds: tuple[float, ...]
    score: float
    spread: float
    out_of_fold: float
    rows_train: int
    rows_test: int
    calculation: str
    features: tuple[tuple[str, str], ...]
    dropped: tuple[tuple[str, str], ...]
    checks: tuple[Check, ...]
    importances: tuple[tuple[str, float], ...]
    submission: pd.DataFrame = field(repr=False)
    warnings: tuple[str, ...] = ()
    #: Each training row's out-of-fold prediction beside its target, in the training table's order.
    out_of_fold_predictions: pd.DataFrame | None = field(default=None, repr=False)
    #: Whether the model was fitted on log(1 + target), so its predictions are turned back with expm1.
    logged: bool = False

    def test_wobble(self, rows: int | None = None, draws: int = 200, seed: int = 0) -> float:
        """How much the score moves between test sets of ``rows`` rows (default: the test table's).

        The standard deviation of the score over ``draws`` bootstrap samples of
        that many out-of-fold predictions: on a small test set, chance alone
        moves a leaderboard score this much.
        """
        frame = self.out_of_fold_predictions
        size = rows or self.rows_test
        rng = np.random.default_rng(seed)
        truth = frame["target"].to_numpy()
        predicted = frame.drop(columns=["target"]).to_numpy()
        priors = np.bincount(truth.astype(int)) / len(truth) if self.task == "classification" else None
        classes = np.arange(predicted.shape[1] if predicted.shape[1] > 1 else 2)
        scores = []
        for _ in range(draws):
            rows_drawn = rng.integers(0, len(frame), size)
            try:
                drawn = predicted[rows_drawn]
                scores.append(_score(self.metric, self.task, classes, truth, rows_drawn, drawn, priors,
                                     self.logged))
            except ValueError:
                continue
        return float(np.std(scores)) if scores else float("nan")

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        self.submission.to_csv(out, index=False)
        return out

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "target": self.target,
            "task": self.task,
            "metric": {"name": self.metric.name, "higher_is_better": self.metric.higher},
            "score": self.score,
            "spread": self.spread,
            "folds": list(self.folds),
            "out_of_fold": self.out_of_fold,
            "rows": {"train": self.rows_train, "test": self.rows_test},
            "calculation": self.calculation,
            "features": [{"name": name, "from": how} for name, how in self.features],
            "dropped": [{"column": column, "reason": reason} for column, reason in self.dropped],
            "checks": [check.to_dict() for check in self.checks],
            "importances": [{"feature": name, "share": share} for name, share in self.importances],
            "warnings": list(self.warnings),
        }

    def __str__(self) -> str:
        better = "higher" if self.metric.higher else "lower"
        lines = [
            f"{self.metric.name} {self.score:.5f} (sd {self.spread:.5f} over {len(self.folds)} folds; "
            f"{better} is better), predicting {self.target} ({self.task}) for {self.rows_test:,} test rows",
            f"  calc: {self.calculation}",
            f"  features: {len(self.features)} from {self.rows_train:,} training rows; "
            + ", ".join(f"{name} {share:.0%}" for name, share in self.importances[:5]),
        ]
        lines.extend(f"  dropped: {column} ({reason})" for column, reason in self.dropped)
        lines.extend(f"  check: {'passed' if check.passed else 'FAILED'} {check.name}: {check.detail}"
                     for check in self.checks)
        lines.extend(f"  warning: {warning}" for warning in self.warnings)
        return "\n".join(lines)


def _refusal(code: str, message: str):
    from .api import Refusal  # noqa: PLC0415

    return Refusal(code, message)


def _read(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy()
    from .file_io import read_tabular_file  # noqa: PLC0415

    path = Path(source)
    return read_tabular_file(path.read_bytes(), path.name)


def _find(folder: Path, stem: str) -> Path | None:
    return next((path for path in sorted(folder.iterdir()) if path.stem.lower() == stem and path.is_file()),
                None)


# ---------------------------------------------------------------- features

#: Separators a structured code may use: "B/45/P", "0013_02", "Smith, Mr. John".
SEPARATORS = ("/", "_", ",", "-", " ")


def _identifier(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None) -> str | None:
    """The column naming each row: the sample submission's first column, or a unique id-like column."""
    if sample is not None and sample.columns[0] in test.columns:
        return str(sample.columns[0])
    for column in test.columns:
        words = normalized_name(str(column)).split()
        if column in train.columns and words and words[-1] in ("id", "key"):
            if test[column].is_unique and train[column].is_unique:
                return str(column)
    return None


# Text that marks a missing value in a column of numbers.
MISSING_TEXT = frozenset({"", "na", "n/a", "nan", "null", "none", "-", "?"})


def _align_types(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> list[str]:
    """Read a column as numbers in both tables when one table has numbers and the other number-like text.

    House Prices' test.csv writes "NA" among GarageCars' counts, so it reads as
    text there and as numbers in train.csv; text that is not a number becomes
    missing. Returns the columns changed.
    """
    changed = []
    for column in columns:
        sides = (train[column], test[column])
        numeric = [is_numeric_dtype(side) and not is_bool_dtype(side) for side in sides]
        if numeric[0] == numeric[1]:
            continue
        text = sides[numeric.index(False)]
        text = text.where(~text.astype("string").str.strip().str.lower().isin(MISSING_TEXT))
        numbers = to_numbers(text)
        if numbers.notna().sum() < 0.95 * text.notna().sum():
            continue
        frame = train if not numeric[0] else test
        frame[column] = numbers
        changed.append(str(column))
    return changed


def _parse_dates(frames: list[pd.DataFrame], columns: list[str]) -> list[str]:
    """Read text columns written as ISO dates ("2017-08-16") as dates in every frame; returns them."""
    parsed = []
    for column in columns:
        if not all(frame[column].dtype == object for frame in frames):
            continue
        sample = pd.concat([frame[column].dropna().astype(str).head(500) for frame in frames])
        if sample.empty or not sample.str.fullmatch(ISO_DATE).all():
            continue
        for frame in frames:
            frame[column] = pd.to_datetime(frame[column], format="ISO8601", errors="coerce")
        parsed.append(column)
    return parsed


def _forecast(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[str, pd.Timedelta] | None:
    """The date column every test row comes after, and how far the test rows reach past training.

    When every test date is later than every training date the task is a
    forecast: the folds then hold out the latest stretches of time, each as
    long as the test's, and fit on the time before them.
    """
    later = [
        column for column in columns
        if is_datetime64_any_dtype(train[column]) and train[column].notna().all()
        and test[column].notna().all() and test[column].min() > train[column].max()
    ]
    if len(later) != 1:
        return None
    column = later[0]
    return column, test[column].max() - train[column].max()


def _time_folds(dates: pd.Series, folds: int, horizon: pd.Timedelta) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds that each hold out a stretch of ``horizon`` before the last date and fit on the time before it.

    The first fold holds out the last stretch, the next the one before, and so
    on; a stretch with no rows, or with fewer than MIN_MODEL_ROWS rows before
    it, is skipped.
    """
    rows = np.arange(len(dates))
    values = dates.to_numpy()
    last = dates.max()
    found = []
    for index in range(folds):
        end = last - index * horizon
        start = end - horizon
        held = rows[(values > start.to_datetime64()) & (values <= end.to_datetime64())]
        fit = rows[values <= start.to_datetime64()]
        if len(held) and len(fit) >= MIN_MODEL_ROWS:
            found.append((fit, held))
    return found


def _series_keys(train: pd.DataFrame, date: str, columns: list[str]) -> list[str] | None:
    """The fewest columns that, with the date, name each training row once: a store and a product family.

    The columns tried are text, or whole numbers with at most 1,000 values,
    fewest values first; together they must fill at least SERIES_COVER of
    the grid of their series by the training dates, so the rows are series
    over time. An empty list means the table is one series.
    """
    dates = train[date].nunique()
    if not train[date].duplicated().any():
        return []
    candidates = [
        column for column in columns
        if column != date and train[column].notna().all() and (
            not is_numeric_dtype(train[column])
            or (pd.api.types.is_integer_dtype(train[column]) and train[column].nunique() <= 1000)
        )
    ]
    candidates = sorted(candidates, key=lambda column: train[column].nunique())[:SERIES_CANDIDATES]
    for size in range(1, SERIES_SIZE + 1):
        for keys in combinations(candidates, size):
            series = len(train.drop_duplicates(list(keys)))
            if len(train) < SERIES_COVER * series * dates:
                continue
            if not train.duplicated([*keys, date]).any():
                return [column for column in columns if column in keys]
    return None


def _lag_features(train: pd.DataFrame, test: pd.DataFrame, date: str, keys: list[str], values: np.ndarray,
                  horizon: int, target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[str, str]]]:
    """The target's past in each series, ``horizon`` days back or more, for the training and test rows.

    Every value a row is given is at least ``horizon`` days older than the
    row, so the latest test row has it, and a fold that holds out a stretch of
    ``horizon`` days uses only target values from before the stretch.
    """
    both = pd.concat([train[[date, *keys]], test[[date, *keys]]], ignore_index=True)
    series = both.groupby(keys, sort=False).ngroup().to_numpy() if keys else np.zeros(len(both), dtype=int)
    first = both[date].min()
    days = ((both[date] - first).dt.days).to_numpy()
    grid = np.full((series.max() + 1, days.max() + 1), np.nan)
    grid[series[:len(train)], days[:len(train)]] = values
    present = ~np.isnan(grid)
    zeros = np.zeros((grid.shape[0], 1))
    sums = np.hstack([zeros, np.cumsum(np.where(present, grid, 0.0), axis=1)])
    counts = np.hstack([zeros, np.cumsum(present, axis=1)])

    def mean_before(back: int, width: int) -> np.ndarray:
        high = np.clip(days - back + 1, 0, grid.shape[1])
        low = np.clip(high - width, 0, grid.shape[1])
        total = sums[series, high] - sums[series, low]
        seen = counts[series, high] - counts[series, low]
        return np.where(seen > 0, total / np.maximum(seen, 1), np.nan)

    def value_at(back: int) -> np.ndarray:
        position = days - back
        found = np.full(len(days), np.nan)
        inside = position >= 0
        found[inside] = grid[series[inside], position[inside]]
        return found

    week = -(-horizon // 7) * 7
    columns: dict[str, np.ndarray] = {}
    made = []
    where = f"in the same {', '.join(keys)}" if keys else "in the table"
    for back in dict.fromkeys((horizon, week, week + 7, week + 14, week + 21)):
        columns[f"{target}_lag{back}"] = value_at(back)
        made.append((f"{target}_lag{back}", f"{target} {back} days before, {where}"))
    for width in LAG_WINDOWS:
        columns[f"{target}_mean{width}"] = mean_before(horizon, width)
        made.append((f"{target}_mean{width}", f"mean {target} over the {width} days ending {horizon} days "
                                              f"before, {where}"))
    weeks = np.column_stack([value_at(week + 7 * step) for step in range(4)])
    seen = (~np.isnan(weeks)).sum(axis=1)
    weekday = np.where(seen > 0, np.nansum(weeks, axis=1) / np.maximum(seen, 1), np.nan)
    columns[f"{target}_weekday4"] = weekday
    made.append((f"{target}_weekday4", f"mean {target} on the same weekday in the 4 weeks from {week} days "
                                       f"before, {where}"))
    if grid.shape[1] > 2 * YEAR_DAYS and horizon < YEAR_DAYS - 7:
        # A season that comes once a year (school supplies before September) shows in the same weeks a year
        # before, which no lag of a few weeks reaches.
        columns[f"{target}_lag{YEAR_DAYS}"] = value_at(YEAR_DAYS)
        made.append((f"{target}_lag{YEAR_DAYS}", f"{target} {YEAR_DAYS} days before, the same weekday a year "
                                                 f"earlier, {where}"))
        columns[f"{target}_year15"] = mean_before(YEAR_DAYS - 7, 15)
        made.append((f"{target}_year15", f"mean {target} over the 15 days around the same date a year "
                                         f"earlier, {where}"))
    frame = pd.DataFrame(columns)
    fit, ahead = frame.iloc[:len(train)], frame.iloc[len(train):]
    return fit.reset_index(drop=True), ahead.reset_index(drop=True), made


def _series_ids(train: pd.DataFrame, test: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """A number for each series, the same in the training and the test rows."""
    both = pd.concat([train[keys], test[keys]], ignore_index=True).astype(str).agg("\x1f".join, axis=1)
    codes = pd.factorize(both)[0]
    return codes[:len(train)], codes[len(train):]


def _quiet(ids: np.ndarray, dates: np.ndarray, y: np.ndarray, fit_rows: np.ndarray, size: int) -> np.ndarray:
    """Which series had a target of 0 on every row of the QUIET_DAYS before the fit rows end."""
    fit_dates = dates[fit_rows]
    recent = fit_rows[fit_dates > fit_dates.max() - np.timedelta64(QUIET_DAYS, "D")]
    totals = np.bincount(ids[recent], weights=np.abs(y[recent].astype(float)), minlength=size)
    return (np.bincount(ids[recent], minlength=size) > 0) & (totals == 0)


def _new_groups(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], folds: int) -> str | None:
    """A column whose test values never appear in training: the test rows are new stocks, stores or people.

    Folds that mix such a group's rows score the model on groups it has seen,
    which the test set never does, so the folds are grouped by it instead. The
    column repeats (two rows a value at least, on average), has a value for
    each fold, and is not a date. With several such columns, the one with the
    fewest values is used.
    """
    found = []
    for column in columns:
        values = train[column]
        if is_datetime64_any_dtype(values) or values.isna().any():
            continue
        count = values.nunique()
        if count < folds or len(values) < MIN_GROUP_ROWS * count:
            continue
        if not set(test[column].dropna().unique()) & set(values.unique()):
            found.append((count, str(column)))
    return min(found)[1] if found else None


def _group_folds(groups: pd.Series, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds that keep each group's rows together, the largest groups first into the lightest fold.

    The order of groups of the same size is shuffled with ``seed``, so each
    repeat splits differently.
    """
    sizes = groups.value_counts(sort=False)
    names = sizes.index.to_numpy(dtype=object)
    names = names[np.random.default_rng(seed).permutation(len(names))]
    names = sorted(names, key=lambda name: -sizes[name])
    load, fold_of = np.zeros(folds), {}
    for name in names:
        lightest = int(load.argmin())
        fold_of[name] = lightest
        load[lightest] += sizes[name]
    assigned = groups.map(fold_of).to_numpy()
    rows = np.arange(len(groups))
    return [(rows[assigned != index], rows[assigned == index]) for index in range(folds)]


def _id_group_sizes(ids: pd.Series) -> pd.Series | None:
    """How many rows share the first part of a structured id ("0013" of "0013_02"), when some do.

    Spaceship Titanic's PassengerId is a travel group and a place in it: the
    id itself names one row, but its group is shared by the people who
    travelled together.
    """
    text = ids.astype("string").str.strip()
    structured = _structured_parts(text)
    if structured is None:
        return None
    first = text.str.split(structured[0], n=1, regex=False).str[0]
    sizes = first.map(first.value_counts()).astype(float)
    return sizes if sizes.nunique() > 1 else None


def _structured_parts(text: pd.Series) -> tuple[str, int] | None:
    """The separator and part count when nearly every value splits into the same 2 to 4 parts."""
    present = text.dropna()
    if present.empty:
        return None
    for separator in SEPARATORS:
        sizes = present.str.count(re.escape(separator)) + 1
        common = sizes.mode().iloc[0]
        if 2 <= common <= 4 and (sizes == common).mean() >= 0.95:
            return separator, int(common)
    return None


def _as_number_or_category(values: pd.Series) -> pd.Series:
    numbers = to_numbers(values)
    if numbers.notna().sum() >= 0.95 * values.notna().sum():
        return numbers
    return values.astype("category")


def build_features(both: pd.DataFrame, skip: set[str]) -> tuple[pd.DataFrame, list[tuple[str, str]],
                                                                 list[tuple[str, str]], list[str]]:
    """Model features from every column but ``skip``, by rules on each column's shape.

    - A number stays a number; a yes/no column becomes 0 and 1.
    - A date becomes its year, month, day of the month, day of the week and day of the year.
    - Text with at most ``MAX_CATEGORIES`` values is a category.
    - Other text gives, when they apply: the parts of a structured code ("B/45/P"
      gives three), the most common word it holds among the column's most common
      words ("Mr" in a name), its first character, and how many rows share it.
      Its raw values are encoded in each fold from the training rows' targets.

    Returns the features, how each was made, the columns left out with the
    reason, and the high-cardinality text columns to target-encode.
    """
    features: dict[str, pd.Series] = {}
    made: list[tuple[str, str]] = []
    dropped: list[tuple[str, str]] = []
    encode: list[str] = []
    for column in both.columns:
        if column in skip:
            continue
        values = both[column]
        name = str(column)
        if values.notna().sum() == 0:
            dropped.append((name, "every value missing"))
            continue
        if values.nunique(dropna=False) <= 1:
            dropped.append((name, "one value only"))
            continue
        if is_bool_dtype(values):
            features[name] = values.astype(float)
            made.append((name, "yes/no as 1 and 0"))
            continue
        if is_numeric_dtype(values):
            features[name] = values.astype(float)
            made.append((name, "number"))
            continue
        if is_datetime64_any_dtype(values):
            for part in ("year", "month", "day", "dayofweek", "dayofyear"):
                features[f"{name}_{part}"] = getattr(values.dt, part).astype(float)
                made.append((f"{name}_{part}", f"the {part} of {name}"))
            continue
        text = values.astype("string").str.strip()
        distinct = text.nunique()
        if distinct <= MAX_CATEGORIES:
            features[name] = text.astype("category")
            made.append((name, "category"))
            continue
        structured = _structured_parts(text)
        if structured:
            separator, count = structured
            parts = text.str.split(separator, n=count - 1, regex=False)
            for index in range(count):
                part = _as_number_or_category(parts.str[index].str.strip())
                part_name = f"{name}_part{index + 1}"
                if isinstance(part.dtype, pd.CategoricalDtype) and part.nunique() > MAX_CATEGORIES:
                    shared = part.astype("string").map(part.value_counts())
                    features[f"{part_name}_count"] = shared.astype(float)
                    made.append((f"{part_name}_count", f"how many rows share part {index + 1} of {name}"))
                    features[part_name] = part.astype("string")
                    encode.append(part_name)
                else:
                    features[part_name] = part
                made.append((part_name, f"part {index + 1} of {name}, split at {separator!r}"))
        words = text.fillna("").str.findall(r"[A-Za-z]+")
        counts = Counter(word for row in words for word in set(row))
        least = MIN_WORD_SHARE * len(text)
        common = [word for word, seen in counts.most_common(FREQUENT_WORDS) if seen >= least]
        if common:
            features[f"{name}_word"] = words.map(_commonest(common)).astype("category")
            made.append((f"{name}_word", f"the most common of {name}'s frequent words it holds"))
        first = text.str[0]
        if first.nunique() <= MAX_CATEGORIES:
            features[f"{name}_first"] = first.astype("category")
            made.append((f"{name}_first", f"the first character of {name}"))
        features[f"{name}_count"] = text.map(text.value_counts()).astype(float)
        made.append((f"{name}_count", f"how many rows share {name}'s value"))
        features[name] = text
        encode.append(name)
        made.append((name, "its training rows' mean target, from the other folds"))
    return pd.DataFrame(features, index=both.index), made, dropped, encode


def _commonest(common: list[str]):
    """A function giving the first of ``common`` a row's words hold, or "none"."""
    rank = {word: position for position, word in enumerate(common)}
    return lambda row: min((word for word in row if word in rank), key=rank.get, default="none")


def _target_encode(fit: pd.Series, y_fit: np.ndarray, apply: pd.Series, smoothing: float = 10.0
                   ) -> np.ndarray:
    """Each value's mean target in ``fit``, pulled toward the overall mean when it has few rows."""
    frame = pd.DataFrame({"value": fit.fillna("<missing>").to_numpy(), "y": y_fit})
    overall = float(np.mean(y_fit))
    stats = frame.groupby("value")["y"].agg(["sum", "count"])
    smoothed = (stats["sum"] + smoothing * overall) / (stats["count"] + smoothing)
    return apply.fillna("<missing>").map(smoothed).fillna(overall).to_numpy(dtype=float)


# ---------------------------------------------------------------- fitting

def _held_out(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> dict:
    """The held-out rows for early stopping, in the form this LightGBM takes: eval_X and eval_y from 4.6."""
    if "eval_X" in inspect.signature(estimator.fit).parameters:
        return {"eval_X": X, "eval_y": y}
    return {"eval_set": [(X, y)]}


def _xgboost(task: str, classes: int, metric: Metric, rows: int, seed: int, patience: int | None):
    """XGBoost with depth-wise trees: a second view of the same features for the blend, when installed."""
    import xgboost as xgb  # noqa: PLC0415

    small = rows < SMALL_TABLE
    wide = WIDE_TABLE <= rows < LARGE_TABLE
    rate = 0.02 if small else 0.1 if rows >= LARGE_TABLE else 0.03 if wide else 0.05
    common = dict(
        n_estimators=5000, learning_rate=rate, max_depth=4 if small else 8 if wide else 6,
        min_child_weight=1 if small else 10, subsample=0.8, colsample_bytree=0.5 if wide else 0.7,
        reg_lambda=1.0, tree_method="hist", max_bin=1023 if wide else 256, enable_categorical=True,
        random_state=seed, n_jobs=4, early_stopping_rounds=patience, verbosity=0,
    )
    if task == "regression":
        objective = "reg:absoluteerror" if metric.name == "mae" else "reg:squarederror"
        return xgb.XGBRegressor(objective=objective, **common)
    return xgb.XGBClassifier(**common)


def _xgboost_installed() -> bool:
    try:
        import xgboost  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True


def _lightgbm(task: str, classes: int, metric: Metric, rows: int, seed: int):
    import lightgbm as lgb  # noqa: PLC0415

    small = rows < SMALL_TABLE
    wide = WIDE_TABLE <= rows < LARGE_TABLE
    rate = 0.02 if small else 0.1 if rows >= LARGE_TABLE else 0.03 if wide else 0.05
    common = dict(
        n_estimators=5000, learning_rate=rate, num_leaves=15 if small else 255 if wide else 63,
        min_child_samples=10 if small else 100 if wide else 40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.5 if wide else 0.7, reg_lambda=1.0, random_state=seed, verbose=-1,
        deterministic=True, force_row_wise=True, n_jobs=4, **({"max_bin": 1023} if wide else {}),
    )
    if task == "regression":
        objective = "regression_l1" if metric.name == "mae" else "regression"
        return lgb.LGBMRegressor(objective=objective, **common)
    return lgb.LGBMClassifier(**common)


def _fold_matrix(X: pd.DataFrame, encode: list[str], fit_index: np.ndarray, y_fit: np.ndarray,
                 others: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """The features for one fold: high-cardinality text replaced by its target encoding from the fit rows.

    ``y_fit`` holds one column per class when there are more than two; each
    encoded column then gives one share per class.
    """
    if not encode:
        return X.iloc[fit_index], [frame for frame in others]
    fit = X.iloc[fit_index].copy()
    done = [frame.copy() for frame in others]
    targets = y_fit if y_fit.ndim == 2 else y_fit[:, None]
    order = np.arange(len(fit)) % 5
    for column in encode:
        values = fit[column]
        for index in range(targets.shape[1]):
            name = column if targets.shape[1] == 1 else f"{column}_{index}"
            # Inside the fit rows, each row's encoding comes from the other fit rows (5 inner folds).
            inner = np.zeros(len(fit))
            for part in range(5):
                mask = order == part
                inner[mask] = _target_encode(values[~mask], targets[~mask, index], values[mask])
            for frame in done:
                frame[name] = _target_encode(values, targets[:, index], frame[column])
            fit[name] = inner
        if targets.shape[1] > 1:
            fit = fit.drop(columns=column)
            done = [frame.drop(columns=column) for frame in done]
    return fit, done


def model(
    train,
    test=None,
    *,
    target: str | None = None,
    metric: str | None = None,
    id_column: str | None = None,
    folds: int = 5,
    seed: int = 0,
):
    """Cross-validate a model of ``target`` on ``train`` and predict it for ``test``.

    ``train`` may be a folder holding train.csv, test.csv and a sample
    submission, as Kaggle gives them; the target is then the column the sample
    submission names. ``metric`` is one of ``METRICS`` or a Kaggle name for it
    ("Categorization Accuracy"); by default accuracy for classes and RMSE for
    numbers. Returns a ``ModelReport``, or a ``Refusal`` with the reason.
    """
    try:
        import lightgbm  # noqa: F401, PLC0415
        import sklearn  # noqa: F401, PLC0415
    except ImportError:
        return _refusal("missing_extra", 'Modelling needs scikit-learn and LightGBM: '
                                         'pip install "automated-data-analyst[model]".')
    sample = None
    if not isinstance(train, pd.DataFrame) and Path(train).is_dir():
        folder = Path(train)
        found = _find(folder, "train")
        if found is None:
            return _refusal("no_target", f"{folder} holds no train file.")
        train, test = found, test if test is not None else _find(folder, "test")
        sample_path = next((path for path in sorted(folder.iterdir()) if "sample" in path.stem.lower()
                            or path.stem.lower().endswith("submission")), None)
        sample = _read(sample_path) if sample_path else None
    train_frame = _read(train)
    test_frame = _read(test) if test is not None else None
    if test_frame is None:
        return _refusal("no_target", "A test table is needed to predict for: pass test.csv or a folder "
                                     "holding one.")
    if target is None:
        candidates = [str(column) for column in train_frame.columns if column not in test_frame.columns]
        if sample is not None:
            named = [column for column in sample.columns[1:] if column in train_frame.columns]
            candidates = named or candidates
        if len(candidates) != 1:
            return _refusal("no_target", "Name the column to predict: "
                            + (f"{', '.join(candidates)} are only in the training table." if candidates
                               else "every training column is also in the test table."))
        target = candidates[0]
    if target not in train_frame.columns:
        return _refusal("no_target", f"{target} is not a column of the training table.")
    rows_missing = int(train_frame[target].isna().sum())
    train_frame = train_frame[train_frame[target].notna()].reset_index(drop=True)
    if len(train_frame) < MIN_MODEL_ROWS:
        return _refusal("too_few_rows", f"{len(train_frame)} training rows have {target}; a cross-validated "
                                        f"model needs at least {MIN_MODEL_ROWS}.")
    y_raw = train_frame[target]
    chosen = metric_named(metric) if metric else None
    if metric and chosen is None:
        return _refusal("unknown_metric",
                        f"ADA has no metric called {metric!r}; it has {', '.join(METRICS)}.")
    numeric_target = is_numeric_dtype(y_raw) and not is_bool_dtype(y_raw)
    whole = numeric_target and bool((y_raw % 1 == 0).all())
    task = chosen.task if chosen else (
        "regression" if numeric_target and not (whole and y_raw.nunique() <= 20) else "classification"
    )
    chosen = chosen or (METRICS["accuracy"] if task == "classification" else METRICS["rmse"])
    identifier = id_column or _identifier(train_frame, test_frame, sample)
    skip = {target, *([identifier] if identifier else [])}
    shared = [column for column in train_frame.columns if column in test_frame.columns and column not in skip]
    _align_types(train_frame, test_frame, shared)
    _parse_dates([train_frame, test_frame], shared)
    forecast = _forecast(train_frame, test_frame, shared)
    group = None if forecast else _new_groups(train_frame, test_frame, shared, folds)
    if group is not None:
        skip.add(group)
        shared.remove(group)
    only_train = [str(column) for column in train_frame.columns
                  if column not in test_frame.columns and column != target]
    both = pd.concat([train_frame[shared], test_frame[shared]], keys=["train", "test"])
    features, made, dropped, encode = build_features(both, skip)
    dropped = [(column, "not in the test table") for column in only_train] + dropped
    if group is not None:
        dropped.insert(0, (group, "its test values never appear in training; the folds are grouped by it"))
    if identifier:
        dropped.insert(0, (identifier, "the row's id"))
        ids = pd.concat([train_frame[identifier], test_frame[identifier]], keys=["train", "test"])
        if (sizes := _id_group_sizes(ids)) is not None:
            features[f"{identifier}_group_size"] = sizes
            made.append((f"{identifier}_group_size", f"how many rows share the first part of {identifier}"))
    X, X_test = features.loc["train"].reset_index(drop=True), features.loc["test"].reset_index(drop=True)

    classes = None
    if task == "classification":
        counts = y_raw.value_counts()
        if counts.min() < folds:
            return _refusal("too_few_rows",
                            f"{target} = {counts.idxmin()} has {counts.min()} rows; each class needs at "
                            f"least {folds} to be cross-validated in {folds} folds.")
        classes = np.array(sorted(counts.index, key=str), dtype=object)
        y = pd.Categorical(y_raw, categories=classes).codes.astype(int)
    else:
        y = y_raw.to_numpy(dtype=float)
        if chosen.name == "rmsle" and (y < 0).any():
            return _refusal("unknown_metric", f"RMSLE needs {target} at 0 or above.")
    # A median is kept by any rising transform, so for absolute error a skewed target at 0 or above
    # is fitted on log(1 + y), where its spread is even, and the predictions are still medians.
    logged = chosen.name == "rmsle" or (
        chosen.name == "mae" and bool((y >= 0).all()) and float(pd.Series(y).skew()) > LOG_SKEW
    )
    fitted_y = np.log1p(y) if logged else y
    if len(X) >= WIDE_TABLE and not forecast:
        # Playground tables are drawn from a smaller original, and its exact values carry signal of their own:
        # a number whose values repeat is also encoded by each value's mean target, from the other folds.
        for name in [str(column) for column in X.columns if is_numeric_dtype(X[column])
                     and 3 <= X[column].nunique() <= VALUE_SHARE * len(X)]:
            X[f"{name}_value"] = X[name].astype("string")
            X_test[f"{name}_value"] = X_test[name].astype("string")
            encode.append(f"{name}_value")
            shared = pd.concat([X[f"{name}_value"], X_test[f"{name}_value"]]).value_counts()
            X[f"{name}_value_count"] = X[f"{name}_value"].map(shared).astype(float)
            X_test[f"{name}_value_count"] = X_test[f"{name}_value"].map(shared).astype(float)
            made.append((f"{name}_value_count", f"how many training and test rows share {name}'s value"))
            made.append((f"{name}_value", f"the training rows' mean target for each value of {name}, "
                                          "from the other folds"))
    if task == "regression":
        encoded_y = fitted_y
    elif len(classes) == 2:
        encoded_y = (y == 1).astype(float)
    else:
        # With several classes a value is encoded by its share of each class.
        encoded_y = (y[:, None] == np.arange(len(classes))[None, :]).astype(float)
    priors = np.bincount(y, minlength=len(classes)) / len(y) if task == "classification" else None
    time_pairs, lagged, keys = [], "", None
    if forecast:
        date, horizon = forecast
        time_pairs = _time_folds(train_frame[date], folds, horizon)
        whole_days = bool((train_frame[date].dt.normalize() == train_frame[date]).all()) and horizon.days >= 1
        keys = _series_keys(train_frame, date, shared) if whole_days and task == "regression" else None
        if keys is not None and len(time_pairs) >= 2:
            past, ahead, lag_made = _lag_features(train_frame, test_frame, date, keys, fitted_y, horizon.days,
                                                  str(target))
            X, X_test = pd.concat([X, past], axis=1), pd.concat([X_test, ahead], axis=1)
            made += lag_made
            lagged = (f"{len(lag_made)} features of {target} {horizon.days} days or more before each row"
                      + (f", in each series of {', '.join(keys)}" if keys else ""))
    if forecast and len(time_pairs) < 2:
        warnings_early = [f"Every test date is after the training dates, but the training rows hold under "
                          f"two stretches of {forecast[1]} to hold out; the folds are shuffled, which "
                          "flatters the score of a forecast."]
        forecast = None
    else:
        warnings_early = []

    from sklearn.model_selection import KFold, StratifiedKFold  # noqa: PLC0415

    seeds = [seed, seed + 1, seed + 2] if len(X) < SMALL_TABLE and not forecast else [seed]
    width = len(classes) if classes is not None and len(classes) > 2 else 1
    splits = []
    for repeat in seeds:
        if forecast:
            pairs = time_pairs
        elif group is not None:
            pairs = _group_folds(train_frame[group].reset_index(drop=True), folds, repeat)
        else:
            splitter = (StratifiedKFold(folds, shuffle=True, random_state=repeat) if task == "classification"
                        else KFold(folds, shuffle=True, random_state=repeat))
            pairs = list(splitter.split(X, y))
        splits += [(repeat, index, fit_rows, held_rows) for index, (fit_rows, held_rows) in enumerate(pairs)]
    # Time folds never hold out the earliest rows; scores use the rows some fold held out.
    scored = np.unique(np.concatenate([held_rows for _, _, _, held_rows in splits]))
    gains = Counter()
    learners = {}
    for learner in ("lightgbm", "linear", *(["xgboost"] if _xgboost_installed() else [])):
        learners[learner] = _cross_validate(learner, X, X_test, y, fitted_y, encoded_y, encode, splits, task,
                                            width, chosen, gains if learner == "lightgbm" else None,
                                            refit=forecast is not None)
    weights = _blend_weights({name: found[0][scored] for name, found in learners.items()}, chosen, task,
                             classes, y[scored], priors, logged)
    oof = sum(weights[name] * learners[name][0] for name in learners)
    predicted = sum(weights[name] * learners[name][1] for name in learners)
    quiet_held, quiet_note = [np.zeros(0, dtype=int) for _ in splits], ""
    if forecast and keys and task == "regression" and (y >= 0).all():
        # A series that sold nothing for QUIET_DAYS predicts 0, judged in each fold on the time before it.
        train_ids, test_ids = _series_ids(train_frame, test_frame, keys)
        dates, size = train_frame[forecast[0]].to_numpy(), int(max(train_ids.max(), test_ids.max())) + 1
        quiet_held = [held_rows[_quiet(train_ids, dates, y, fit_rows, size)[train_ids[held_rows]]]
                      for _, _, fit_rows, held_rows in splits]
        trial = oof.copy()
        trial[np.concatenate(quiet_held)] = 0.0
        # Judged fold by fold: a few series that sell again in one stretch of time should not outweigh the
        # many that stay at 0 in the others. The rule holds when it scores better in most folds.
        better = 0
        for (_, _, _, held_rows), quiet_rows in zip(splits, quiet_held, strict=True):
            if not len(quiet_rows):
                continue
            before = _score(chosen, task, classes, y, held_rows, oof[held_rows], priors, logged)
            after = _score(chosen, task, classes, y, held_rows, trial[held_rows], priors, logged)
            better += 1 if (after > before if chosen.higher else after < before) else -1
        if better > 0:
            oof = trial
            quiet_test = _quiet(train_ids, dates, y, np.arange(len(y)), size)[test_ids]
            predicted = predicted.copy()
            predicted[quiet_test] = 0.0
            quiet_note = (f"; series whose {target} was 0 on every day of the {QUIET_DAYS} before predict 0 "
                          f"({int(quiet_test.sum()):,} test rows), as it scored better in most folds")
        else:
            quiet_held = [np.zeros(0, dtype=int) for _ in splits]
    fold_scores = []
    for (repeat, _, _, held_rows), quiet_rows in zip(splits, quiet_held, strict=True):
        # Each repeat predicts every row once; a fold's score uses that repeat's predictions.
        held = sum(weights[name] * learners[name][2][repeat][held_rows] for name in learners)
        held[np.isin(held_rows, quiet_rows)] = 0.0
        fold_scores.append(_score(chosen, task, classes, y, held_rows, held, priors, logged))
    out_of_fold = _score(chosen, task, classes, y, scored, oof[scored], priors, logged)
    oof_frame = pd.DataFrame(oof[scored], columns=[f"p{index}" for index in range(oof.shape[1])])
    oof_frame["target"] = y[scored]
    total = sum(gains.values()) or 1.0
    importances = tuple((name, gain / total) for name, gain in gains.most_common(10))

    submission = _submission(test_frame, identifier, target, sample, chosen, task, classes, predicted, priors,
                             logged)
    # A forecast's test rows are later by design, so the shift check leaves out the date's own features.
    timed = [name for name, _ in made if forecast and name.startswith(f"{forecast[0]}_")]
    checks = _checks(submission, test_frame, identifier, task, classes, chosen, X.drop(columns=timed),
                     X_test.drop(columns=timed), seed)
    warnings = list(warnings_early)
    perfect = out_of_fold >= NEAR_PERFECT if chosen.higher else out_of_fold <= 1e-6
    if perfect:
        warnings.append(f"The out-of-fold {chosen.name} is {out_of_fold:.4f}, nearly perfect: check that no "
                        f"column gives {target} away.")
    if importances and importances[0][1] > 0.8:
        warnings.append(f"{importances[0][0]} carries {importances[0][1]:.0%} of the model's gain; check "
                        f"it is known before {target} is.")
    if rows_missing:
        warnings.append(f"Left out {rows_missing:,} training rows with {target} missing.")
    kind = f"{group}-grouped " if group is not None else "stratified " if task == "classification" else ""
    target_note = ""
    if logged:
        target_note = f"log1p({target}) fitted, predictions turned back with expm1; "
    linear = "ridge regression" if task == "regression" else "logistic regression"
    parts = [f"{weights['lightgbm']:.2f} LightGBM gradient boosting", f"{weights['linear']:.2f} {linear}"]
    if "xgboost" in weights:
        parts.append(f"{weights['xgboost']:.2f} XGBoost gradient boosting")
    blend = (f"{', '.join(parts[:-1])} and {parts[-1]} (weights chosen on the out-of-fold predictions)")
    if forecast:
        date, horizon = forecast
        reach = f"{horizon.days} days" if horizon == pd.Timedelta(days=horizon.days) else str(horizon)
        calculation = (
            f"{blend}, forecast: {len(splits)} time-ordered folds, each holding out {reach} of {date} "
            "before the last training date (the test's reach) and fitted on the time before it"
            + (f"; {lagged}" if lagged else "")
            + f"; early stopping on each held-out stretch; {target_note}the test predictions refitted on "
            f"every training row{quiet_note}; scored with {chosen.name} on the {len(scored):,} rows held out"
        )
    else:
        calculation = (
            f"{blend}, {folds}-fold {kind}cross-validation"
            + (f" repeated with {len(seeds)} seeds" if len(seeds) > 1 else "")
            + f", early stopping on each held-out fold; {target_note}test predictions averaged over the "
            f"{folds * len(seeds)} fold models; scored with {chosen.name}"
        )
    return ModelReport(
        target=target, task=task, metric=chosen, folds=tuple(fold_scores), score=float(np.mean(fold_scores)),
        spread=float(np.std(fold_scores)), out_of_fold=out_of_fold, rows_train=len(X), rows_test=len(X_test),
        calculation=calculation, features=tuple(made), dropped=tuple(dropped), checks=tuple(checks),
        importances=importances, submission=submission, warnings=tuple(warnings),
        out_of_fold_predictions=oof_frame, logged=logged,
    )


def _labels(predicted: np.ndarray, metric: Metric, priors: np.ndarray) -> np.ndarray:
    """The class each row is given: the likeliest, or for balanced accuracy the likeliest against its share.

    Balanced accuracy weighs each class equally, so a probability is divided
    by the class's share of the training rows before the largest is taken.
    """
    table = np.column_stack([1 - predicted[:, 0], predicted[:, 0]]) if predicted.shape[1] == 1 else predicted
    if metric.name == "balanced_accuracy":
        table = table / priors
    return table.argmax(axis=1)


def _linear_design(fit: pd.DataFrame, others: list[pd.DataFrame]) -> tuple[np.ndarray, list[np.ndarray]]:
    """The fold's features for a linear model: numbers filled, logged when skewed, scaled; categories one-hot.

    Every step is learned from the fit rows only: the medians, the skew, the
    scale and the categories (those with at least 5 fit rows).
    """
    blocks_fit, blocks_other = [], [[] for _ in others]
    for column in fit.columns:
        values = fit[column]
        if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
            counts = values.astype("string").value_counts()
            kept = list(counts[counts >= 5].index)
            if not kept:
                continue
            text = values.astype("string")
            blocks_fit.append(np.column_stack([(text == level).fillna(False).to_numpy(dtype=float)
                                               for level in kept]))
            for position, frame in enumerate(others):
                other = frame[column].astype("string")
                blocks_other[position].append(np.column_stack(
                    [(other == level).fillna(False).to_numpy(dtype=float) for level in kept]))
            continue
        numbers = values.to_numpy(dtype=float)
        median = float(np.nanmedian(numbers)) if np.isfinite(numbers).any() else 0.0
        missing = ~np.isfinite(numbers)
        filled = np.where(missing, median, numbers)
        skewed = bool(filled.min() >= 0 and pd.Series(filled).skew() > 1)
        transform = np.log1p if skewed else (lambda array: array)
        shaped = transform(filled)
        centre, scale = float(shaped.mean()), float(shaped.std()) or 1.0
        blocks_fit.append(((shaped - centre) / scale)[:, None])
        flag = missing.any()
        if flag:
            blocks_fit.append(missing.astype(float)[:, None])
        for position, frame in enumerate(others):
            other = frame[column].to_numpy(dtype=float)
            other_missing = ~np.isfinite(other)
            other_filled = np.where(other_missing, median, other)
            if skewed:
                other_filled = np.clip(other_filled, 0, None)
            blocks_other[position].append(((transform(other_filled) - centre) / scale)[:, None])
            if flag:
                blocks_other[position].append(other_missing.astype(float)[:, None])
    matrix = np.hstack(blocks_fit) if blocks_fit else np.zeros((len(fit), 0))
    rest = [np.hstack(blocks) if blocks else np.zeros((len(frame), 0))
            for blocks, frame in zip(blocks_other, others, strict=True)]
    return matrix, rest


def _cross_validate(learner: str, X: pd.DataFrame, X_test: pd.DataFrame, y: np.ndarray, fitted_y: np.ndarray,
                    encoded_y: np.ndarray, encode: list[str], splits: list, task: str, width: int,
                    metric: Metric, gains: Counter | None, refit: bool = False):
    """One learner's out-of-fold predictions (averaged over repeats), test predictions and each repeat's.

    With ``refit`` the test predictions come from one more model fitted on
    every training row (LightGBM with the folds' mean best number of trees),
    as a forecast's latest rows matter most; otherwise they average the folds'.
    """
    import lightgbm as lgb  # noqa: PLC0415
    from sklearn.linear_model import LogisticRegression, RidgeCV  # noqa: PLC0415

    by_repeat: dict[int, np.ndarray] = {}
    predicted = np.zeros((len(X_test), width))
    trees = []
    for repeat, index, fit_rows, held_rows in splits:
        fit_X, (held_X, test_X) = _fold_matrix(X, encode, fit_rows, encoded_y[fit_rows],
                                               [X.iloc[held_rows], X_test])
        if learner == "lightgbm":
            estimator = _lightgbm(task, width, metric, len(X), repeat * 100 + index)
            patience = 100 if len(X) >= LARGE_TABLE else 200
            estimator.fit(fit_X, fitted_y[fit_rows], **_held_out(estimator, held_X, fitted_y[held_rows]),
                          callbacks=[lgb.early_stopping(patience, verbose=False)])
            trees.append(estimator.best_iteration_ or estimator.n_estimators)
            if gains is not None:
                shares = estimator.booster_.feature_importance("gain")
                for name, gain in zip(fit_X.columns, shares, strict=True):
                    gains[str(name)] += float(gain)
            held_input, test_input = held_X, test_X
        elif learner == "xgboost":
            patience = 100 if len(X) >= LARGE_TABLE else 200
            estimator = _xgboost(task, width, metric, len(X), repeat * 100 + index, patience)
            estimator.fit(fit_X, fitted_y[fit_rows], eval_set=[(held_X, fitted_y[held_rows])], verbose=False)
            trees.append(int(estimator.best_iteration) + 1)
            held_input, test_input = held_X, test_X
        else:
            fit_matrix, (held_input, test_input) = _linear_design(fit_X, [held_X, test_X])
            if task == "regression":
                estimator = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(fit_matrix, fitted_y[fit_rows])
            else:
                estimator = LogisticRegression(C=0.5, max_iter=2000).fit(fit_matrix, y[fit_rows])
        if task == "regression":
            # A linear model can run far past the targets it saw; trees never do. Keep it within them.
            low, high = float(fitted_y[fit_rows].min()), float(fitted_y[fit_rows].max())
            held_pred = np.clip(estimator.predict(held_input), low, high)[:, None]
            test_pred = np.clip(estimator.predict(test_input), low, high)[:, None]
        else:
            held_pred = estimator.predict_proba(held_input)
            test_pred = estimator.predict_proba(test_input)
            if width == 1:
                held_pred, test_pred = held_pred[:, 1:], test_pred[:, 1:]
        by_repeat.setdefault(repeat, np.zeros((len(X), width)))[held_rows] = held_pred
        predicted += test_pred / len(splits)
    oof = sum(by_repeat.values()) / len(by_repeat)
    if refit:
        every = np.arange(len(X))
        fit_X, (test_X,) = _fold_matrix(X, encode, every, encoded_y, [X_test])
        if learner == "lightgbm":
            estimator = _lightgbm(task, width, metric, len(X), 0).set_params(
                n_estimators=max(1, round(float(np.mean(trees)))))
            estimator.fit(fit_X, fitted_y)
        elif learner == "xgboost":
            estimator = _xgboost(task, width, metric, len(X), 0, None).set_params(
                n_estimators=max(1, round(float(np.mean(trees)))))
            estimator.fit(fit_X, fitted_y, verbose=False)
        else:
            fit_matrix, (test_input,) = _linear_design(fit_X, [test_X])
            if task == "regression":
                estimator = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(fit_matrix, fitted_y)
            else:
                estimator = LogisticRegression(C=0.5, max_iter=2000).fit(fit_matrix, y)
            test_X = test_input
        if task == "regression":
            predicted = np.clip(estimator.predict(test_X), fitted_y.min(), fitted_y.max())[:, None]
        else:
            predicted = estimator.predict_proba(test_X)
            predicted = predicted[:, 1:] if width == 1 else predicted
    return oof, predicted, by_repeat


def _blend_weights(oofs: dict[str, np.ndarray], metric: Metric, task: str, classes, y: np.ndarray,
                   priors, logged: bool = False) -> dict[str, float]:
    """The weights, 0 to 1 in steps of 0.05 and adding to 1, that give the best out-of-fold score.

    For a metric of labels (accuracy, balanced accuracy, F1) the weights are
    chosen by log loss instead: a count of right labels jumps between weights,
    so it would pick a weight by chance.
    """
    names = list(oofs)
    if metric.output == "label":
        metric = METRICS["logloss"]
    steps = 20
    best, best_score = {names[0]: 1.0, **{name: 0.0 for name in names[1:]}}, None
    for parts in product(range(steps + 1), repeat=len(names) - 1):
        if sum(parts) > steps:
            continue
        shares = [part / steps for part in parts]
        weights = dict(zip(names, [1 - sum(shares), *shares], strict=True))
        mixed = sum(weight * oofs[name] for name, weight in weights.items())
        try:
            score = _score(metric, task, classes, y, np.arange(len(y)), mixed, priors, logged)
        except ValueError:
            continue
        better = best_score is None or (score > best_score if metric.higher else score < best_score)
        if better:
            best, best_score = weights, score
    return best


def _score(metric: Metric, task: str, classes, y: np.ndarray, rows: np.ndarray, predicted: np.ndarray,
           priors: np.ndarray | None = None, logged: bool = False) -> float:
    truth = y[rows]
    if task == "regression":
        values = predicted[:, 0]
        return metric.score(truth, np.expm1(values) if logged else values)
    if metric.output == "probability" and predicted.shape[1] == 1:
        return metric.score(truth, predicted[:, 0])
    if metric.output in ("probability", "top3"):
        return metric.score(truth, predicted, np.arange(len(classes)))
    return metric.score(truth, _labels(predicted, metric, priors))


def _submission(test: pd.DataFrame, identifier: str | None, target: str, sample: pd.DataFrame | None,
                metric: Metric, task: str, classes, predicted: np.ndarray, priors: np.ndarray | None = None,
                logged: bool = False) -> pd.DataFrame:
    """The predictions in the sample submission's layout, or as id and target."""
    out = pd.DataFrame({identifier: test[identifier].to_numpy()}) if identifier else pd.DataFrame(
        index=test.index)
    if task == "regression":
        values = predicted[:, 0]
        out[target] = np.expm1(values) if logged else values
        return out
    if predicted.shape[1] == 1:
        positive = predicted[:, 0]
        if metric.output == "probability":
            out[target] = positive
        else:
            out[target] = classes[_labels(predicted, metric, priors)]
        return _like_sample(out, sample, target)
    if metric.output == "top3":
        top = np.argsort(-predicted, axis=1)[:, :3]
        out[target] = [" ".join(str(classes[index]) for index in row) for row in top]
    elif metric.output == "probability":
        columns = [f"{target}_{label}" for label in classes]
        if sample is not None and len(sample.columns) == len(classes) + 1:
            columns = list(sample.columns[1:])
        for position, column in enumerate(columns):
            out[column] = predicted[:, position]
    else:
        out[target] = classes[_labels(predicted, metric, priors)]
    return _like_sample(out, sample, target)


def _like_sample(out: pd.DataFrame, sample: pd.DataFrame | None, target: str) -> pd.DataFrame:
    """Labels written as the sample submission writes them: True/False, 0/1 or text."""
    if sample is None or target not in sample.columns or target not in out.columns:
        return out
    example = sample[target]
    if is_bool_dtype(example) and not is_bool_dtype(out[target]):
        out[target] = out[target].map(lambda value: str(value).lower() in ("true", "1", "1.0"))
    elif is_numeric_dtype(example) and not is_bool_dtype(example):
        converted = to_numbers(out[target])
        if converted.notna().all():
            out[target] = converted.astype(example.dtype) if (converted % 1 == 0).all() else converted
    return out


def _checks(submission: pd.DataFrame, test: pd.DataFrame, identifier: str | None, task: str, classes,
            metric: Metric, X: pd.DataFrame, X_test: pd.DataFrame, seed: int) -> list[Check]:
    checks = [Check("rows", len(submission) == len(test),
                    f"{len(submission):,} predictions for {len(test):,} test rows")]
    if identifier:
        same = submission[identifier].tolist() == test[identifier].tolist()
        said = "every test id once, in the test table's order" if same else "ids differ from the test table"
        checks.append(Check("ids", same and submission[identifier].is_unique, said))
    values = submission.drop(columns=[identifier] if identifier else [])
    missing = int(values.isna().sum().sum())
    checks.append(Check("complete", not missing, f"{missing} missing predictions"))
    numeric = values.select_dtypes("number")
    if metric.output == "probability" and not numeric.empty:
        inside = bool(((numeric >= 0) & (numeric <= 1)).all().all())
        checks.append(Check("probabilities", inside, "every probability between 0 and 1" if inside
                            else "probabilities outside 0 to 1"))
    if not numeric.empty:
        varied = bool((numeric.nunique() > 1).any())
        checks.append(Check("varied", varied, "the predictions differ between rows" if varied
                            else "every row gets the same prediction"))
    checks.append(_shift(X, X_test, seed))
    return checks


def _shift(X: pd.DataFrame, X_test: pd.DataFrame, seed: int) -> Check:
    """Whether a model can tell training rows from test rows: AUC near 0.5 means they look alike."""
    import lightgbm as lgb  # noqa: PLC0415
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    columns = [column for column in X.columns if not pd.api.types.is_string_dtype(X[column])]
    left = X[columns].sample(min(len(X), SHIFT_SAMPLE), random_state=seed)
    right = X_test[columns].sample(min(len(X_test), SHIFT_SAMPLE), random_state=seed)
    joined = pd.concat([left, right], ignore_index=True)
    side = np.r_[np.zeros(len(left)), np.ones(len(right))]
    scores = np.zeros(len(joined))
    for fit_rows, held_rows in StratifiedKFold(3, shuffle=True, random_state=seed).split(joined, side):
        estimator = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, verbose=-1,
                                       random_state=seed, deterministic=True, force_row_wise=True, n_jobs=4)
        estimator.fit(joined.iloc[fit_rows], side[fit_rows])
        scores[held_rows] = estimator.predict_proba(joined.iloc[held_rows])[:, 1]
    auc = float(roc_auc_score(side, scores))
    return Check("test like training", auc <= SHIFT_AUC,
                 f"a model tells training rows from test rows with AUC {auc:.3f} (0.5: alike)")


__all__ = ["METRICS", "Metric", "ModelReport", "build_features", "metric_named", "model"]
