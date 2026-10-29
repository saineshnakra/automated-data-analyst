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
| train | Titanic, House Prices, Spaceship Titanic, Playground S6E7 to S6E9, Housing Prices (Kaggle Learn), Store Sales, Stocks Predict challenge |
| validation | Playground S5E10 to S5E12 and S6E1 to S6E3 |
| test | Playground S5E7 to S5E9, S6E4 to S6E6 and S6E10 |

Housing Prices (Kaggle Learn), Store Sales and the Stocks Predict challenge
moved to train when their runs led to fixes: numbers read as text in one
table, forecasts, and test rows from new groups.

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
- the cross-validated score beats the bar by at least twice the larger of the
  folds' standard error and the test wobble, how much the score moves between
  test sets the size of the public leaderboard's share of the test rows;
- the competition got no submission in the last 24 hours, and no competition
  got one in the last 2 hours.

`--first` sends one submission to each competition that has none yet, when
the checks pass and the model gives no warning, whatever the margin, so each
competition has a leaderboard score beside its cross-validated one.
Submissions go through the `kaggle` package.

## Where it stands

| Competition | Split | Metric | Cross-validated | Top-10% bar | About | Gate |
|---|---|---|---|---|---|---|
| Titanic | train | accuracy | 0.8358 (sd 0.022) | 0.7943 | top 3.4% | held: the public leaderboard's 209 rows leave a wobble of about 0.025, and the margin is under twice it; sent with `--first` |
| Housing Prices (Kaggle Learn) | train | MAE | 14,012 (sd 1,325) | 15,010 | top 7.1% | held: the margin of 998 is under twice the test wobble of 848 on half the 1,459 test rows (1,696); sent with `--first` |
| Store Sales | train | RMSLE | 0.3914 (sd 0.008, 5 time-ordered folds) | 0.3872 | top 13.6% | held: not past the bar; sent by hand |
| Stocks Predict challenge | train | Spearman | 0.0506 (sd 0.015) | 0.0533 | top 27% | held: not past the bar; sent by hand after it closed |

Public leaderboard scores of the submissions sent:

| Competition | Sent | Public score | Top-10% bar | Place |
|---|---|---|---|---|
| Titanic | 28 Oct 2026 | 0.79665 | 0.79425 | top 9.6% of 11,692 |
| Housing Prices (Kaggle Learn) | 28 Oct 2026 | 13,331.6 | 15,009.7 | top 4.4% of 3,855 |
| Store Sales | 29 Oct 2026 | 0.47814 | 0.38724 | top 58% of 565 |
| Stocks Predict challenge | 29 Oct 2026 (late) | 0.04490 (private 0.03545) | 0.05333 | top 46% of 26 |

Titanic's public score is 0.039 below its cross-validated one, within twice
the wobble of 209 rows; Housing's is 681 better, within its wobble of 848;
Stocks' is 0.006 below, within its wobble of 0.013. Store Sales' is 0.087
worse than its cross-validated 0.391, twenty times its wobble: the
time-ordered folds do not score the forecast as the test is made, and that is
the next fix.

The other competitions are not scored yet: their rules have not been accepted
for the account, so their data cannot be downloaded.
