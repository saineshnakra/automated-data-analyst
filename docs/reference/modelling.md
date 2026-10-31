# Predicting a column

`ada.model(train, test)` and `ada model` predict one column of a table from its
other columns, as a tabular Kaggle competition asks: a training table with the
answers, a test table without them, and a score to beat. Every score is
cross-validated, the features come from rules on the columns' shapes, and the
submission is checked before it is written. Code: `ada/modelling.py`.

It needs the optional extra, which adds scikit-learn, LightGBM, XGBoost and CatBoost:

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

A column that is numbers in one table and number-like text in the other (House
Prices' test.csv writes "NA" among `GarageCars`' counts) is read as numbers in
both, the text that is no number as missing. Text written as ISO dates
("2017-08-16") is read as dates.

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
| `mae` | Mean Absolute Error | Numbers, fitted on absolute error; on log(1 + y) when y is 0 or above and skewed |
| `spearman` | RankIC, Spearman correlation | Numbers, ranked |

For MAE, a target at 0 or above with a skewness above 1 (prices, counts) is
fitted on log(1 + y) and the predictions turned back with expm1: a rising
transform keeps the median, which absolute error asks for, and on the log
scale the spread is even.

With no metric, a class target gets accuracy and a number gets RMSE. A number
target with at most 20 whole values is read as classes. For balanced
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
| A structured id whose first part repeats ("0013" of "0013_02") | How many rows share that first part, the size of the group |
| In a table of 100,000 rows or more, a number with at most a tenth as many values as rows | Also its mean target for each value (see below) and how many training and test rows share its value |

A playground table is drawn from a smaller original, and the exact values it
copied carry signal of their own; a forecast's numbers are not encoded this
way.

The mean target of a many-valued text value is computed inside each fold from
that fold's training rows only, and within them from the other four fifths, so
no row sees its own target. It is pulled toward the overall mean when the value
has few rows. With more than two classes a value gives its share of each class
instead, one feature per class.

## The model

Three learners are cross-validated on the same folds:

- LightGBM gradient boosting, stopped early on each held-out fold
  (learning rate 0.02 and 15 leaves under 20,000 rows; 0.05 and 63 leaves
  above; 0.03 and 255 leaves, at least 100 rows a leaf, half the features for
  each tree and values in up to 1,023 bins, from 100,000 rows; 0.1 and 63
  leaves, stopping after 100 rounds without gain, from a million rows)
- a regularized linear model: ridge regression for numbers, logistic
  regression for classes, on the numbers filled with their median (with a
  flag for each missing value), logged when skewed, and scaled, and the
  categories with at least 5 rows one-hot; its predictions of a number are
  kept within the targets it was fitted on, as a linear model can run far
  past them where trees never do
- XGBoost gradient boosting with depth-wise trees on the same features
  (depth 4 under 20,000 rows, 6 above and 8 from 100,000 rows, with the
  learning rates LightGBM takes), stopped early the same way; it is left out
  when XGBoost is not installed

When the test rows belong to groups the training rows never hold (new
stocks, new stores, new patients), folds that mix a group's rows would score
the model on groups it has seen. A column whose test values never appear in
training, with at least one value per fold and two rows per value on average,
groups the folds instead: each group's rows stay in one fold, the largest
groups go first into the lightest fold, and the column is left out of the
features. With several such columns the one with the fewest values is used.
When no column does but the first part of a structured id never appears in
training ("0013" of "0013_02", and some first parts repeat), the folds keep
the rows of each first part together the same way: Spaceship Titanic's test
passengers travel in groups no training passenger is in.

Importing ADA sets `OMP_WAIT_POLICY` to `PASSIVE` unless it is set already,
so LightGBM's and XGBoost's threads sleep while they wait instead of spinning.
On a machine whose cores are busy with other work, spinning threads make a fit
about a hundred times slower: 300 XGBoost trees on House Prices take 101
seconds instead of 0.65. The libraries read the setting when they load, so it
has no effect if they were imported before ADA.

## Forecasts

When every test date is later than every training date, the task is a
forecast, and shuffled folds would score the model on days in between days it
was fitted on. The folds then hold out the latest stretches of time instead:
the first the stretch before the last training date as long as the test's
reach (the last test date less the last training date), the next the stretch
before that, each fitted on the time before it. A stretch with fewer than 100
rows before it is skipped; with fewer than two stretches the folds stay
shuffled and a warning says the score flatters the forecast. The score uses the
rows some fold held out. The test predictions come from one more model fitted
on every training row, LightGBM with the folds' mean best number of trees.

For a daily forecast of a number, the rows are read as series: the fewest
columns that with the date name each training row once (a store and a product
family), tried among text columns and whole-number columns with at most 1,000
values, which must fill at least half the grid of series by dates. Each row
then gets its series' target from the test's reach or further back: the value
that many days before, the value on the same weekday 1 to 4 weeks further
back, the mean over the 7, 14, 28, 56 and 112 days ending that many days
before, and the mean of the four same weekdays. With two years of dates, it
also gets the value 364 days before (the same weekday a year earlier) and the
mean over the 15 days around that day, so a season that comes once a year
(school supplies before September) shows. No value is nearer than the
reach, so a held-out stretch uses only targets from before it, and the last
test day has them all. The date's own parts are left out of the shift check,
as a forecast's test rows are later by design.

A series whose target was 0 on every day of the 28 before a fold's held-out
stretch predicts 0 for that stretch, and one whose target was 0 on every day of
the last 28 training days predicts 0 for the test, when that scores better
than the model's own predictions in most of the folds that hold such series. A
few series that sell again in one stretch of time do not outweigh the many that
stay at 0 in the others. Each fold decides from the time before it only, and
the calculation says how many test rows were set so.

## Blending

Their out-of-fold predictions are blended with the weights, in steps of 0.05
and adding to 1, that give the best out-of-fold score (by log loss for a metric of labels, as a
count of right labels jumps from one weight to the next). Folds are stratified
for classes. Under 20,000 rows, cross-validation is repeated with 3 seeds. The
test predictions average every fold's model. A fixed seed gives the same
result on every run.

## The full search: `ada auto`

`ada auto` and `ada.model(..., auto=True)` read the same data and build the
same features, then search wider. It takes longer: on a table of 600,000 rows,
hours instead of minutes.

```bash
ada auto playground/ --metric auc --out submission.csv --jobs 2
```

- More learners on the same folds: CatBoost (balanced trees with its own
  encoding of category columns), LightGBM with random split points, a neural
  network with hidden layers of 256 and 128 units on the linear model's
  numbers, and under 100,000 rows extremely randomized trees. CatBoost is left
  out when it is not installed. A random forest and the 20 nearest training
  rows were tried as well; across seven competitions neither earned more than
  0.04 of a blend, and the forest took longest to fit, so they are not used.
- A second layer: LightGBM fitted on the features and every first-layer
  learner's out-of-fold predictions, on the same folds. A row's predictions
  come from models that never saw it. It joins the blend only when, on its
  own, it beats every first-layer learner on the metric; otherwise the
  calculation gives both scores. The blend's weights for a metric of labels
  come from log loss, and on Spaceship Titanic the second layer had the
  better log loss and the worse accuracy.
- The blend's weights come from 25 rounds of greedy selection: each round adds
  the learner, the second layer among them, that most improves the
  out-of-fold score of the mean so far, and a learner's weight is its share of
  the rounds. A grid of weights grows as a power of the number of learners;
  greedy selection grows with it.
- A regression under 100,000 rows also tries fitting without the rows the
  first LightGBM fit misses by more than 4 robust standard deviations (1.4826
  times the median absolute deviation of its out-of-fold errors), keeping them
  in scoring. The rule holds when it scores better in most folds; the
  calculation says how many rows were left out. Under RMSE, where the left-out
  rows' own errors count most, it rarely holds.
- `--jobs N` fits N learners at once, in threads that share the cores.
  LightGBM, XGBoost and CatBoost give the same predictions with any number of
  threads, so the model is the same.

## Readable models: `--explain`

`ada model --explain` and `ada auto --explain` (`ada.model(..., explain=True)`)
predict with a model a person can read, and say what that costs against the
full search on the same folds.

```bash
ada auto data/ --metric auc --explain --out submission.csv
```

- Two readable models are fitted: the linear model (ridge regression, or
  logistic regression for classes) and an additive model, LightGBM with one
  split a tree, so each tree reads one column and each column's effect adds
  up on its own. The one with the better out-of-fold score makes the
  predictions.
- The linear model's coefficients come from statsmodels on every training
  row: ordinary least squares with heteroscedasticity-robust (HC3) errors for
  numbers, logistic regression for two classes. Numbers are scaled to a
  standard deviation of 1, so a coefficient is the change for one standard
  deviation; each category's most common level is the reference. The 12 terms
  with the largest z are shown with their 95% intervals and p-values. Columns
  that repeat others exactly are left out, and the report says how many. For
  more than two classes there is a table for each class, so none is shown.
- The additive model's effect curves come from one more fit on every row,
  stopped early on every tenth row: each column's contribution to each
  prediction, read by LightGBM, averaged in ten bands of a number's values (or
  at each value when it has ten or fewer) and at a category's most common
  levels. A column's share is the spread of its contributions over all the
  columns'. A number's curve "bends" when a straight line explains under 90%
  of it. Effects are in the units the model fits: the target (log1p of it
  when the target is logged) or log-odds.
- Three comparisons, each the mean difference of two models' fold scores and
  its standard error over the folds, called clear only beyond twice the
  error:
  - effects bend: the additive model against the linear one;
  - columns act together: the full LightGBM against the additive model;
  - cost of explaining: the full blend against the readable model.

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
