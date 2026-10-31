"""Score ADA's modelling mode on a holdout cut from each competition's own training rows.

    python -I holdout.py ADA_CHECKOUT CACHE out.json [--only=a,b] [--auto] [--jobs=N]

Each competition in competitions.yml is split inside itself, the way its test
rows differ from its training rows:

- when every test date comes after the training dates, the holdout is the
  last stretch of training dates as long as the test's (Store Sales: 16 days);
- when a column's test values never appear in training (new stores, stocks or
  people), the holdout is a fifth of that column's values;
- otherwise it is a fifth of the rows.

Values and rows are chosen by a hash of their text (a row by its id), so the
same data always gives the same split. ADA models the rest, the development
part, cross-validated as usual, and predicts the holdout, which is scored with
the competition's metric. The holdout score's standard error is from 200
bootstrap samples of its rows. Its leaderboard percentile is where the score
would place among the teams' test scores; those models were fitted on all the
training rows, a fifth more than ADA's here. The data must be in CACHE, as
run.py downloads it.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandas.api.types import is_bool_dtype

HERE = Path(__file__).parent
SHARE = 0.2
DRAWS = 200


def chosen(values: pd.Series) -> np.ndarray:
    """Whether each value is in the fifth of distinct values with the lowest hash of their text."""
    distinct = sorted({str(value) for value in values},
                      key=lambda text: hashlib.sha256(text.encode()).digest())
    picked = set(distinct[:max(1, round(SHARE * len(distinct)))])
    return values.astype(str).isin(picked).to_numpy()


def files(folder: Path) -> tuple[Path | None, Path | None, Path | None]:
    """A competition folder's train, test and sample submission files, found as ``ada.model`` finds them."""
    import ada.modelling as modelling  # noqa: PLC0415

    sample = next((path for path in sorted(folder.iterdir()) if "sample" in path.stem.lower()
                   or path.stem.lower().endswith("submission")), None)
    return modelling._find(folder, "train"), modelling._find(folder, "test"), sample


def target_of(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None) -> str | None:
    """The column to predict: the one the sample submission names, or the one only training has."""
    named = [column for column in sample.columns[1:] if column in train.columns] if sample is not None else []
    candidates = named or [column for column in train.columns if column not in test.columns]
    return str(candidates[0]) if len(candidates) == 1 else None


def split(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame | None,
          folds: int = 5) -> tuple[np.ndarray, str]:
    """Which training rows to hold out, and how they were chosen, made like the test rows."""
    import ada.modelling as modelling  # noqa: PLC0415

    identifier = modelling._identifier(train, test, sample)
    shared = [column for column in train.columns if column in test.columns and column != identifier]
    train, test = train.copy(), test.copy()
    modelling._align_types(train, test, shared)
    modelling._parse_dates([train, test], shared)
    forecast = modelling._forecast(train, test, shared)
    if forecast:
        date, reach = forecast
        held = (train[date] > train[date].max() - reach).to_numpy()
        span = f"{reach.days} days" if reach == pd.Timedelta(days=reach.days) else str(reach)
        return held, f"the last {span} of {date}, as every test date is after the training dates"
    group = modelling._new_groups(train, test, shared, folds)
    if group is not None:
        held = chosen(train[group])
        return held, (f"a fifth of the {group} values ({train.loc[held, group].nunique()} of "
                      f"{train[group].nunique()}), as the test's {group} values never appear in training")
    first = modelling._new_id_groups(train[identifier], test[identifier], folds) if identifier else None
    if first is not None:
        held = chosen(first)
        return held, (f"a fifth of the groups in the first part of {identifier} ({first[held].nunique()} of "
                      f"{first.nunique()}), as no test row's group is in training")
    if identifier:
        return chosen(train[identifier]), f"a fifth of the rows, by a hash of {identifier}"
    return chosen(pd.Series(np.arange(len(train)))), "a fifth of the rows, by a hash of each row's position"


def as_text(values: pd.Series) -> pd.Series:
    """Labels as text that compares equal whether they were read as 1, 1.0 or "1"; True stays "True"."""
    if not is_bool_dtype(values) and not values.map(lambda value: isinstance(value, bool | np.bool_)).all():
        numbers = pd.to_numeric(values, errors="coerce")
        if numbers.notna().all() and (numbers % 1 == 0).all():
            return numbers.astype("int64").astype(str)
    return values.astype(str)


def readings(metric, submission: pd.DataFrame, truth: pd.Series, classes: list,
             target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """The holdout's truth and predictions as ``metric.score`` reads them, from the submission's columns.

    ``classes`` are the development rows' classes sorted by their text, as
    ``ada.model`` sorts them: a single probability column is the chance of the
    second, and probability columns follow their order.
    """
    if metric.task == "regression":
        return truth.to_numpy(dtype=float), submission[target].to_numpy(dtype=float), None
    names = as_text(pd.Series(classes, dtype=object)).tolist()
    truth = as_text(truth)
    if metric.output == "probability" and target in submission.columns:
        return (truth == names[1]).to_numpy(dtype=int), submission[target].to_numpy(dtype=float), None
    if metric.output == "probability":
        columns = [column for column in submission.columns if column != submission.columns[0]]
        return truth.to_numpy(dtype=object), submission[columns].to_numpy(dtype=float), np.array(names,
                                                                                                 dtype=object)
    if metric.output == "top3":
        # Each row's three labels get 3, 2 and 1, so the metric's own ranking reads them back in order.
        place = {name: position for position, name in enumerate(names)}
        ranks = np.zeros((len(submission), len(names)))
        for row, labels in enumerate(submission[target].astype(str).str.split()):
            for position, label in enumerate(labels[:3]):
                if label in place:
                    ranks[row, place[label]] = 3 - position
        return truth.to_numpy(dtype=object), ranks, np.array(names, dtype=object)
    # Labels are compared as codes in the classes' order, so F1's positive class is the second, as in the
    # model. A label that is not one of the classes gets -1.
    code = {name: position for position, name in enumerate(names)}
    predicted = as_text(submission[target]).map(code).fillna(-1).to_numpy(dtype=int)
    return truth.map(code).fillna(-1).to_numpy(dtype=int), predicted, None


def evaluate(folder: Path, work: Path, metric_name: str, auto: bool = False, jobs: int = 1) -> dict:
    """Split a competition folder inside itself, model the development part and score the holdout.

    The two parts are written to ``work`` as train.csv and test.csv beside the
    sample submission, so ``ada.model`` reads them as it reads the competition.
    """
    import ada.modelling as modelling  # noqa: PLC0415

    train_path, test_path, sample_path = files(folder)
    if train_path is None or test_path is None:
        return {"refused": f"{folder} holds no train or test file"}
    train, test = modelling._read(train_path), modelling._read(test_path)
    sample = modelling._read(sample_path) if sample_path else None
    target = target_of(train, test, sample)
    if target is None:
        return {"refused": "the column to predict is not clear from the files"}
    train = train[train[target].notna()].reset_index(drop=True)
    held, how = split(train, test, sample)
    if held.all() or not held.any():
        return {"refused": f"the holdout ({how}) holds {int(held.sum())} of {len(train)} rows"}
    development, holdout = train[~held], train[held]
    work.mkdir(parents=True, exist_ok=True)
    development.to_csv(work / "train.csv", index=False)
    columns = [column for column in test.columns if column in train.columns]
    holdout[columns].to_csv(work / "test.csv", index=False)
    if sample_path is not None:
        (work / sample_path.name).write_bytes(sample_path.read_bytes())
    started = time.time()
    report = modelling.model(work, metric=metric_name, auto=auto, jobs=jobs)
    seconds = round(time.time() - started)
    if not isinstance(report, modelling.ModelReport):
        return {"refused": report.message, "holdout": how}
    metric = report.metric
    classes = sorted(pd.unique(development[target]), key=str) if metric.task == "classification" else []
    truth, predicted, labels = readings(metric, report.submission, holdout[target].reset_index(drop=True),
                                        classes, target)
    if metric.task == "classification" and labels is None and (truth < 0).all():
        return {"refused": "no holdout label matches a class of the development rows", "holdout": how}
    score = metric.score(truth, predicted, labels)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(DRAWS):
        rows = rng.integers(0, len(truth), len(truth))
        try:
            draws.append(metric.score(truth[rows], predicted[rows], labels))
        except ValueError:
            continue
    return {
        "metric": metric.name, "holdout_rows": how, "rows_development": len(development),
        "rows_holdout": len(holdout), "cv": report.score, "cv_sd": report.spread, "holdout": score,
        "holdout_se": float(np.std(draws)) if draws else None, "gap": score - report.score,
        "warnings": list(report.warnings), "auto": auto, "seconds": seconds,
    }


def main() -> None:
    checkout, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    only = next((arg.split("=", 1)[1].split(",") for arg in sys.argv if arg.startswith("--only=")), None)
    jobs = next((int(arg.split("=", 1)[1]) for arg in sys.argv if arg.startswith("--jobs=")), 1)
    auto = "--auto" in sys.argv
    sys.path.insert(0, str(checkout))
    sys.path.insert(0, str(HERE))
    from run import bar, download, percentile  # noqa: PLC0415

    import ada.modelling as modelling  # noqa: PLC0415

    spec = yaml.safe_load((HERE / "competitions.yml").read_text())["competitions"]
    results = json.loads(out.read_text()) if out.exists() else {}
    for item in spec:
        slug = item["slug"]
        if only and slug not in only:
            continue
        entry = evaluate(download(slug, cache), cache / "holdout" / slug, item["metric"], auto, jobs)
        if "refused" in entry:
            print(f"{slug}: refused: {entry['refused']}")
        else:
            higher = modelling.METRICS[entry["metric"]].higher
            top, teams = bar(slug, higher, cache)
            entry.update(bar_top10=top, teams=teams,
                         holdout_percentile=percentile(entry["holdout"], higher, slug, cache))
            print(f"{slug} {entry['metric']} cv {entry['cv']:.5f} holdout {entry['holdout']:.5f} "
                  f"(se {entry['holdout_se']:.5f}) bar {top:.5f} -> about top "
                  f"{entry['holdout_percentile']:.1f}%; holdout: {entry['holdout_rows']}")
        results[slug] = entry
        out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
