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

Each competition is split inside itself. `holdout.py` holds out training
rows the way the competition's test rows differ from them, models the rest
with `ada.model`, cross-validated as usual, and scores its predictions for
the held-out rows with the competition's metric:

- when every test date comes after the training dates, the holdout is the
  last stretch of training dates as long as the test's (Store Sales: 16
  days);
- when a column's test values never appear in training, a fifth of that
  column's values (the Stocks Predict challenge: 3 of its 14 stocks);
- when the first part of a structured id never appears in training, a
  fifth of those groups (Spaceship Titanic: 1,243 of 6,217 travel groups);
- otherwise a fifth of the rows.

Values and rows are chosen by a hash of their text, so the same data always
gives the same split. Every competition may be used to find and fix
problems: a change is kept when it moves the holdout scores, and a
submission is the last check.

```bash
python -I results/kaggle/holdout.py . CACHE holdout.json --only=titanic [--auto]
```

Until this version whole competitions were set aside: train ones were used
to find problems, validation ones were scored each round, and test ones
were locked and scored once. The tables below come from those rounds. It
left seven competitions out of every change, and a change was judged on a
leaderboard score whose public share can be a few hundred rows.

Splitting each competition the way its test is made also found a leak:
Spaceship Titanic's test passengers travel in groups no training passenger
is in, so a holdout of random rows scored 0.8166 accuracy, where the
leaderboard gave 0.8080. Holding out whole groups gives 0.8087. The folds
of `ada.model` now keep such groups together too.

## Holdout scores

`ada model` of this version on each competition's holdout. The cross-validated
score is on the other rows; the standard error is from 200 bootstrap samples
of the held-out rows.

| Competition | Metric | Held out | Cross-validated | Holdout | Standard error |
|---|---|---|---|---|---|
| Titanic | accuracy | 178 rows | 0.83123 | 0.83146 | 0.02790 |
| House Prices | RMSLE | 292 rows | 0.11858 | 0.11957 | 0.00998 |
| Spaceship Titanic | accuracy | 1,725 rows | 0.80626 | 0.80870 | 0.00892 |
| S6E7 | balanced accuracy | 138,018 rows | 0.95002 | 0.95155 | 0.00095 |
| S6E8 | AUC | 138,274 rows | 0.96712 | 0.96731 | 0.00039 |
| S6E9 | AUC | 133,733 rows | 0.94493 | 0.94674 | 0.00066 |
| Housing Prices (Kaggle Learn) | MAE | 292 rows | 14,067 | 15,406 | 1,102 |
| Store Sales | RMSLE | 28,512 rows | 0.39397 | 0.39944 | 0.00287 |
| Stocks Predict challenge | Spearman | 1,935 rows | 0.04716 | 0.00626 | 0.02451 |
| S5E10 | RMSE | 103,551 rows | 0.05609 | 0.05599 | 0.00017 |
| S5E11 | AUC | 118,799 rows | 0.92561 | 0.92484 | 0.00106 |
| S5E12 | AUC | 140,000 rows | 0.73006 | 0.73346 | 0.00133 |
| S6E1 | RMSE | 126,000 rows | 8.68793 | 8.64713 | 0.01747 |
| S6E2 | AUC | 126,000 rows | 0.95560 | 0.95513 | 0.00051 |
| S6E3 | AUC | 118,839 rows | 0.91796 | 0.91788 | 0.00088 |
| S6E10 | AUC | 139,927 rows | 0.96015 | 0.96073 | 0.00054 |
| S6E4 | balanced accuracy | 126,000 rows | 0.97840 | 0.97987 | 0.00085 |
| S6E5 | AUC | 87,828 rows | 0.94995 | 0.95018 | 0.00072 |
| S6E6 | balanced accuracy | 115,469 rows | 0.96619 | 0.96667 | 0.00058 |
| S5E7 | accuracy | 3,705 rows | 0.96986 | 0.96707 | 0.00280 |
| S5E8 | AUC | 150,000 rows | 0.97466 | 0.97483 | 0.00039 |
| S5E9 | RMSE | 104,833 rows | 26.46421 | 26.45598 | 0.05137 |

On 19 of the 22 the holdout score is within twice its standard error of the
cross-validated one. S6E1, S6E9 and S5E12 are 2.3 to 2.7 standard errors
apart, and on each the holdout is the better. The held-out rows come from
the training table, so they can be easier than the test rows: S5E12's
holdout AUC is 0.733 and its leaderboard score 0.699. A holdout score is for
comparing two versions of ADA on the same rows, not for guessing a
leaderboard place; a submission stays the last check.

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
| S6E1 | validation | RMSE | 8.6703 | 8.62682 | 8.54913 | top 17% of 4,319 |
| S6E3 | validation | AUC | 0.9182 | 0.91607 | 0.91687 | top 16% of 4,143 |

Value encoding and larger trees moved S6E8 from 0.9656 (top 51%) to 0.96847
and S6E9 from 0.94164 (top 65%) to 0.94568 on the leaderboard; S6E7, scored
on labels, stayed within its wobble. S5E12 moves to train: its leaderboard
score is 0.032 below the cross-validated one, which led to the gap in the
gate. Weighting the training rows by how much they look like the test rows
did not close it (cross-validated 0.7271 against 0.7280 unweighted), so the
test rows differ in how the target follows the columns, not only in where
they lie.

Store Sales' public score is 0.087 worse than its cross-validated 0.391,
twenty times its wobble. That submission came before the yearly lags and the
rule for series that sell nothing (#135). The holdout of the last 16 training
days, cut as the test is, scores the current model 0.399 against its
cross-validated 0.394; a new submission is what will tell.

## `ada auto`

Each train and validation competition got one more submission from `ada auto`
(`run.py --auto`), the wider search: CatBoost, LightGBM with random split
points, a neural network, a second layer when it beats every first-layer
learner, greedy blend weights and, for a regression under 100,000 rows, the
outlier rule. Kaggle ranks a team by its best public score, so the place is
the better of the two.

| Competition | Test rows | Metric | `ada model` | `ada auto` | Place |
|---|---|---|---|---|---|
| Housing Prices (Kaggle Learn) | 1,459 | MAE | 13,331.6 | 12,602.8 | top 2.0% of 3,855 |
| S6E2 | 270,000 | AUC | 0.95358 | 0.95384 | top 13.5% of 4,371 |
| S5E10 | 172,585 | RMSE | 0.05563 | 0.05550 | top 14.3% of 4,083 |
| S6E3 | 254,655 | AUC | 0.91607 | 0.91612 | top 15.4% of 4,143 |
| S6E1 | 270,000 | RMSE | 8.62682 | 8.61700 | top 16.6% of 4,319 |
| S5E11 | 254,569 | AUC | 0.92542 | 0.92572 | top 18.1% of 3,726 |
| S6E8 | 296,302 | AUC | 0.96847 | 0.96882 | top 30.1% of 3,532 |
| S6E7 | 295,753 | balanced accuracy | 0.94972 | 0.94990 | top 35% of 3,356 |
| S6E9 | 286,571 | AUC | 0.94568 | 0.94581 | top 37.8% of 3,576 |
| S5E12 | 300,000 | AUC | 0.69903 | 0.69883 | top 26% of 4,208 |
| Titanic | 418 | accuracy | 0.79665 | 0.78947 | top 9.6% of 11,692 |
| Spaceship Titanic | 4,277 | accuracy | 0.80804 | 0.80570 | top 15.6% of 1,791 |
| House Prices | 1,459 | RMSLE | 0.12119 | 0.12247 | top 14.4% of 4,613 |

On the Playground tables, with 170,000 to 300,000 test rows, eight of nine
scored better and S5E12, whose test rows follow the columns differently from
its training rows, held. Titanic, Spaceship Titanic and House Prices scored
worse by less than their test wobble: on a few hundred to a few thousand
public rows, chance moves the score more than the gain. The Stocks Predict
challenge was not sent, as its test rows are told apart from the training
rows. Store Sales, a forecast, was not run: its folds, not its learners, are
what is still to fix.

Two ideas from the top Playground solutions were tried and not kept. A
linear second layer (logistic regression on the learners' log-odds, or ridge
regression, with negative weights allowed) scored within 0.00001 of the
greedy weights on the saved out-of-fold predictions of seven competitions:
with learners this alike, how they are weighed matters little. CatBoost given
the repeated-value columns as categories of its own, instead of their target
encoding, gained 0.00024 holdout AUC on S6E8 and nothing on S6E2, and took
three to nine times as long; without its combinations of categories it lost
on both.

The locked test competitions, scored once with the model of XGBoost, value
encoding and larger trees, and not used to choose any fix:

| Competition | Metric | Cross-validated | Public score | Top-10% bar | Place |
|---|---|---|---|---|---|
| S5E7 | accuracy | 0.9692 | 0.97571 | 0.97652 | top 31% of 4,331 |
| S5E8 | AUC | 0.9751 | 0.97607 | 0.97710 | top 13% of 3,367 |
| S5E9 | RMSE | 26.462 | 26.3886 | 26.3836 | top 48% of 2,583 |
| S6E6 | balanced accuracy | 0.9668 | 0.96749 | 0.97227 | top 26% of 2,817 |
| S6E10 | AUC | 0.9605 | 0.96007 | 0.96207 | top 48% of 3,429 |
| S6E4 | balanced accuracy | 0.9791 | 0.97791 | 0.98110 | top 17% of 4,316 |
| S6E5 | AUC | 0.9502 | 0.94983 | 0.95453 | top 34% of 3,023 |

S5E9's public score is 0.005 from the bar and half the leaderboard away from
it: on these leaderboards the middle and the top 10% are close together.
S6E1 and S6E3 to S6E5 were scored with the same model once their rules were
accepted. S6E3's cross-validated AUC of 0.9182 was above the bar and its
public score 0.0008 below it, another leaderboard harder than its folds.
