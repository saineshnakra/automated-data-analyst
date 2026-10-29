# Kaggle competitions

`ada model` is scored on 22 tabular Kaggle competitions, against the score
at the top 10% of each leaderboard. `competitions.yml` lists them and their
metrics; `run.py` downloads the data, runs `ada.model`, reads the leaderboard
and, only when asked and only when every gate passes, submits.

```bash
KAGGLE_API_TOKEN=... python -I results/kaggle/run.py . CACHE out.json --only=titanic
```

The account must have accepted each competition's rules on kaggle.com before
its data can be downloaded. The token is read from the environment and never
written anywhere.

## Splits

The competitions are split as the claim cases are: train competitions may be
used to find and fix problems; validation ones are scored each round, and one
whose failure leads to a fix moves to train; test ones are locked and scored
once, at the end.

| Split | Competitions |
|---|---|
| train | Titanic, House Prices, Spaceship Titanic, Playground S6E7 to S6E9 and S5E12, Housing Prices (Kaggle Learn), Store Sales, Stocks Predict challenge |
| validation | Playground S5E10, S5E11 and S6E1 to S6E3 |
| test | Playground S5E7 to S5E9, S6E4 to S6E6 and S6E10 |

Housing Prices (Kaggle Learn), Store Sales, the Stocks Predict challenge and
S5E12 moved to train when their runs led to fixes: numbers read as text in one
table, forecasts, test rows from new groups, and the gap between
cross-validated and leaderboard scores.

Three other competitions the account has entered are not used: House Price
Prediction Project (two teams and no test table), Home Credit Credit Risk
Model Stability (submissions only through notebooks) and Stocks Return
Prediction V3 (data only as pickle files, which are not loaded from an
untrusted source).

## The bar and the gate

The bar is the leaderboard score at the top 10%: the 90th percentile when a
higher score is better, the 10th when lower is. A submission is sent only
when:

- every check on the submission passes and the model gives no warning (see
  [Predicting a column](../../docs/reference/modelling.md#checks-and-warnings));
- the competition has a scored submission of ADA's, and the cross-validated
  score, moved by that submission's gap between its leaderboard and
  cross-validated scores, beats the bar by at least twice the larger of the
  folds' standard error and the test wobble, how much the score moves between
  test sets the size of the public leaderboard's share of the test rows. The
  gap is read from the submission's description ("ada model, auc cv 0.95549").
  Test rows can be harder than the training rows: on S5E12 a cross-validated
  AUC of 0.731 scored 0.699 and on S6E2 0.9555 scored 0.9536, each above the
  bar by cross-validation and below it on the leaderboard;
- the competition got no submission in the last 24 hours, and no competition
  got one in the last 2 hours.

`--first` sends one submission to each competition that has none yet, when
the checks pass and the model gives no warning, whatever the margin, so each
competition has a leaderboard score beside its cross-validated one.
Submissions go through the `kaggle` package.

## Where it stands

Public leaderboard scores of the submissions sent, with the model of this
version (value encoding and larger trees from 100,000 rows) where the table
is that large:

| Competition | Split | Metric | Cross-validated | Public score | Top-10% bar | Place |
|---|---|---|---|---|---|---|
| Titanic | train | accuracy | 0.8358 | 0.79665 | 0.79425 | top 9.6% of 11,692 |
| Housing Prices (Kaggle Learn) | train | MAE | 14,012 | 13,331.6 | 15,009.7 | top 4.4% of 3,855 |
| House Prices | train | RMSLE | 0.1195 | 0.12119 | 0.12006 | top 14.4% of 4,613 |
| Spaceship Titanic | train | accuracy | 0.8074 | 0.80804 | 0.80897 | top 15.6% of 1,791 |
| S6E7 | train | balanced accuracy | 0.9500 | 0.94972 | 0.95112 | top 42% of 3,356 |
| S6E8 | train | AUC | 0.9671 | 0.96847 | 0.97117 | top 31% of 3,532 |
| S6E9 | train | AUC | 0.9454 | 0.94568 | 0.94656 | top 39% of 3,576 |
| Store Sales | train | RMSLE | 0.3914 | 0.47814 | 0.38724 | top 58% of 565 |
| Stocks Predict challenge (late) | train | Spearman | 0.0506 | 0.04490 | 0.05333 | top 46% of 26 |
| S5E10 (the earlier model) | validation | RMSE | 0.0561 | 0.05563 | 0.05546 | top 46% of 4,083 |
| S5E11 | validation | AUC | 0.9254 | 0.92542 | 0.92731 | top 19% of 3,726 |
| S5E12 | train, after this round | AUC | 0.7308 | 0.69903 | 0.70378 | top 26% of 4,208 |
| S6E2 | validation | AUC | 0.9555 | 0.95358 | 0.95390 | top 28% of 4,371 |

Value encoding and larger trees moved S6E8 from 0.9656 (top 51%) to 0.96847
and S6E9 from 0.94164 (top 65%) to 0.94568 on the leaderboard; S6E7, scored
on labels, stayed within its wobble. S5E12 moves to train: its leaderboard
score is 0.032 below the cross-validated one, which led to the gap in the
gate. Weighting the training rows by how much they look like the test rows
did not close it (cross-validated 0.7271 against 0.7280 unweighted), so the
test rows differ in how the target follows the columns, not only in where
they lie.

Store Sales' public score is 0.087 worse than its cross-validated 0.391,
twenty times its wobble: the time-ordered folds do not score the forecast as
the test is made, and that is a fix still to make.

The locked test competitions (S5E7 to S5E9, S6E4 to S6E6, S6E10) are scored
once, at the end. S6E1, S6E3, S6E4 and S6E5 wait on their rules being
accepted for the account.
