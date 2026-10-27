# Kaggle competitions

`ada model` is scored on 21 tabular Kaggle competitions, against the score
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
| train | Titanic, House Prices, Spaceship Titanic, Playground S6E7 to S6E9 |
| validation | Housing Prices (Kaggle Learn), Store Sales, Playground S5E10 to S5E12 and S6E1 to S6E3 |
| test | Playground S5E7 to S5E9, S6E4 to S6E6 and S6E10 |

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

## Where it stands

| Competition | Split | Metric | Cross-validated | Top-10% bar | About | Gate |
|---|---|---|---|---|---|---|
| Titanic | train | accuracy | 0.8260 (sd 0.022) | 0.7943 | top 3.5% | held: the public leaderboard's 209 rows leave a wobble of 0.025, and the margin of 0.032 is under twice it (0.049) |
| Housing Prices (Kaggle Learn) | validation | MAE | 15,459 (sd 1,423) | 15,010 | top 11.6% | held: not past the bar |

No submission has been sent. The other competitions are not scored yet: their
rules have not been accepted for the account, so their data cannot be
downloaded.
