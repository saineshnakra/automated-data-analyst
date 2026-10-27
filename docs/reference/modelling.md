# Predicting a column

`ada.model(train, test)` and `ada model` predict one column of a table from its
other columns, as a tabular Kaggle competition asks: a training table with the
answers, a test table without them, and a score to beat. Every score is
cross-validated, the features come from rules on the columns' shapes, and the
submission is checked before it is written. Code: `ada/modelling.py`.

It needs the optional extra, which adds scikit-learn and LightGBM:

```bash
pip install "automated-data-analyst[model]"
```

```bash
ada model titanic/ --metric accuracy --out submission.csv     # train.csv, test.csv, a sample submission
ada model train.csv test.csv --target price --metric rmsle --json
```

```python
report = ada.model("titanic/", metric="accuracy")
report.score             # mean cross-validated accuracy; report.folds holds each fold's
report.submission        # one row per test row, in the sample submission's layout
report.write("submission.csv")
```

## What it reads

A folder holding `train.csv`, `test.csv` and a sample submission (any file
whose name holds "sample" or ends in "submission"), as Kaggle gives them, or
two tables. The target is the column the sample submission names, or the one
column the training table has and the test table lacks; `--target` names it
otherwise. The id column is the sample submission's first column, or a column
in both tables whose last name word is "id" and whose values are unique.
Training rows with no target are left out, and a warning counts them.

## Metrics

| `--metric` | Kaggle names it | Predicts |
|---|---|---|
| `accuracy` | Categorization Accuracy, Accuracy Score | Labels |
| `balanced_accuracy` | Balanced Accuracy Score | Labels, each class weighed equally |
| `f1` | F-Score | Labels |
| `auc` | Roc Auc Score | Probabilities |
| `logloss` | Log Loss | Probabilities |
| `map@3` | MAP@3 | The three likeliest labels |
| `rmse` | Root Mean Squared Error | Numbers |
| `rmsle` | Root Mean Squared Logarithmic Error | Numbers, fitted on log(1 + y) |
| `mae` | Mean Absolute Error | Numbers, fitted on absolute error |

With no metric, a class target gets accuracy and a number gets RMSE. A
number target with at most 20 whole values is read as classes. For balanced
accuracy, each class's probability is divided by its share of the training
rows before the largest is taken, so rare classes are not drowned out.

## Features

`build_features` makes them from the training and test rows together, by the
shape of each column, never from knowledge of one competition:

| Column | Becomes |
|---|---|
| A number | The number |
| Yes/no | 1 and 0 |
| A date | Its year, month, day, day of the week and day of the year |
| Text with at most 64 values | A category |
| A structured code ("B/45/P", "0013_02", "Smith, Mr. John") | Its 2 to 4 parts, as numbers or categories, when 95% of values split the same way |
| Other text | The commonest of the column's frequent words it holds ("Mr"), its first character, how many rows share it, and its mean target in the training rows (see below) |
| The id, a column with one value, a column with no values, a column the test table lacks | Left out, with the reason in `dropped` |

The mean target of a many-valued text value is computed inside each fold from
that fold's training rows only, and within them from the other four fifths, so
no row sees its own target. It is pulled toward the overall mean when the value
has few rows. With more than two classes such text keeps only its other features.

## The model

Two learners are cross-validated on the same folds:

- LightGBM gradient boosting, stopped early on each held-out fold
  (learning rate 0.02 and 15 leaves under 20,000 rows; 0.05 and 63 leaves
  above)
- a regularized linear model: ridge regression for numbers, logistic
  regression for classes, on the numbers filled with their median (with a
  flag for each missing value), logged when skewed, and scaled, and the
  categories with at least 5 rows one-hot

Their out-of-fold predictions are blended with the weight, in steps of 0.05,
that gives the best out-of-fold score (by log loss for a metric of labels, as a
count of right labels jumps from one weight to the next). Folds are stratified
for classes. Under 20,000 rows, cross-validation is repeated with 3 seeds. The
test predictions average every fold's model. A fixed seed gives the same
result on every run.

## Checks and warnings

| Check | Passes when |
|---|---|
| `rows` | One prediction for each test row |
| `ids` | Every test id once, in the test table's order |
| `complete` | No prediction is missing |
| `probabilities` | Every probability is between 0 and 1 |
| `varied` | Not every row gets the same prediction |
| `test like training` | A model tells training rows from test rows with AUC at most 0.8 |

A warning says when the out-of-fold score is nearly perfect, or one feature
carries more than 80% of the model's gain: either may be a column that gives
the target away.

`report.test_wobble(rows)` is how much the score moves between test sets of
that many rows: the standard deviation of the score over 200 bootstrap samples
of the out-of-fold predictions. On a small test set, chance alone moves a
leaderboard score this much.

## Refusals

| Code | When |
|---|---|
| `missing_extra` | scikit-learn or LightGBM is not installed |
| `no_target` | No test table, no train file in the folder, or no single column to predict |
| `too_few_rows` | Fewer than 100 training rows with the target, or a class with fewer rows than folds |
| `unknown_metric` | A metric ADA does not have, or RMSLE on a target below 0 |

## The JSON record

`ModelReport.to_dict()` and `ada model --json` give the target, the task, the
metric and whether higher is better, the score, its spread and each fold's,
the out-of-fold score, the rows, the calculation, every feature and how it was
made, the columns left out and why, the checks, the ten features with the
largest share of the model's gain, and the warnings.

## Kaggle

[results/kaggle](../../results/kaggle/README.md) scores the modelling mode on
tabular Kaggle competitions, against each leaderboard's top 10%.
