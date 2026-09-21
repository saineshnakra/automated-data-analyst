# Handoff — where this work stands

Working notes so a fresh session can pick this up cold. Last updated 2026-09-21.

## 1. Standing rules (these override anything else)

- **Every commit and PR goes out under my name.** Commit with
  `git -c user.name="saineshnakra" -c user.email="saineshnakra@gmail.com" commit`.
  No `Co-Authored-By` trailer. No session-URL footer — the server appends one to PR
  bodies, strip it afterwards with `update_pull_request`.
- **Do not post comments on GitHub.** I reply to reviewers myself. Bring findings to me
  in chat instead.
- **Branch names say what the branch does.** No tool names, no assistant names, no
  competitor names — in branches, files, or commit messages.
- **Never put a model identifier in anything that lands in a repo.**
- **Do not execute any SQL.** Not the wellFinity setup file, not the verify script, not
  the query helper. Running anything against the database is mine to do.
- **Plain language.** Write the way I write. No marketing voice, no filler.

## 2. What ADA is

Streamlit + pandas data analyst. `app.py`/`ui.py` → `pipeline.py` → the engine
(`analysis`, `schema`, `aggregation`, `timeseries`, `anomalies`, `forecasting`,
`business_insights`, `nlq`, `autovis`, `formatting`, `metrics`) → `file_io`, with
`ai_insights` optional. Python 3.11/3.12/3.13, ruff, unittest, `AppTest` for the UI.

Two ideas hold the whole thing together:

- **The metric contract** (`metrics.py`). `resolve_metric(name)` returns a `MetricSpec`:
  `aggregation` (sum/mean/count), `unit` (amount/rate/count), `additive`,
  `increase_is_welcome`, `combines_as`. Rates get averaged, never summed; a rate moves in
  percentage points; there are no shares, no HHI, no waterfall, no net decomposition.
- **Span accounting in `nlq.py`.** A `_Spans` ledger records which characters of the
  question a rule has claimed. The grammar reads from `blanked()` — text with the claimed
  spans removed. If a meaningful span is still unclaimed at the end, the question is
  refused rather than half-answered.

Private intermediate columns are `__date`, `__measure`, `__segment`, `__period`,
`__value`; collisions are avoided with `distinct_label()`. `NO_SELECTION = "(none)"` is
the empty-selection sentinel.

## 3. Repo state as of 2026-09-21 (later the same day)

`main` is now `6a8b890`. Merged: #48 (star ask), #45, #46, #42, #47. ruff clean,
309 tests green.

**The trap that caught us:** #43 and #44 were merged into `metric-contract-module`,
their base branch, *after* #42 had already gone to main from `96ca93c`. The branch moved
on, main did not follow, and both fixes silently missed main. **PR #49** recovers them by
merging `metric-contract-module` into main, and also carries a parser fix (below).
Lesson: when a stacked PR merges into its base rather than main, main does not get it.

### PR #49 also fixes the number parser

`_read_formatted_number` on main was refusing `$(100)` and `$-100` (cell becomes NaN,
money vanishes from every total) and *accepting* `1,2,3` as 123, `12 34` as 1234, and
unclosed `(100` as positive 100. Replaced with the marker-at-a-time version from
`consistency-release` plus `_digit_groups`/`_integer_and_fraction`. Tests in
`tests/test_accounting_negatives.py` — 7 tests, 8 failures on unfixed main.
330 tests green with it.

### Review findings that did NOT need porting to main

- partial-period trim guard: already on main via #47
- ai_insights uniqueness rule: main's logic was already correct; release differs in comments only
- ui.py histogram collision: main uses `px.histogram`, has no manual binning, bug absent
- nlq.py duplicate spellings: main collects every match per column already, bug absent

### Earlier state (superseded, kept for context)

- `main` is `66c494c` and has not moved since Sep 9. **Nothing of mine has been merged.**
- 28 stars, 10 forks, 34 open issues.
- `consistency-release` is `fae5680` — all 26 audit items (D-01…D-26) plus the eight
  review defects below. CI green on 3.11/3.12/3.13.

### Open PRs

| PR | Author | What it is | Verdict |
|---|---|---|---|
| #47 | me | 6/N partial-period trim | mine, ready |
| #46 | me | 5/N rate change in points | mine, ready |
| #45 | me | 4/N identifier-name rule | mine, ready |
| #44 | me | 3/N rate trends average (base #42) | mine, ready |
| #43 | me | 2/N trend gaps are not numbers (base #42) | mine, ready |
| #42 | me | 1/N metric contract module | mine, ready |
| #48 | me | README star ask, 14 lines, text only | mine, ready |
| #41 | Uday029 | accessible text summaries for charts | reviewed — changes requested, see section 9 |
| #40 | me | the whole consistency release, ~4k lines | **closed unmerged 2026-09-21** — see below |
| #36 | Saket7002 | interactive cleaning suggestions, 946 lines | too big — author needs to split it |
| #35 | Som0111 | QUERY_STOPWORDS additions | **take it** — not superseded, the release keeps the same short list |
| #31 | HNK69 | pass column values to format_number | real bug on main; the release's `_shown()` helper covers more call sites — leave open until that lands |

#33 and #34 were already closed (they duplicated each other).

### #40 is closed

Closed without merging on 2026-09-21, before the split finished. **`origin/consistency-release`
is still at `fae5680` and holds everything** — all 26 audit items and the eight review
fixes. Only six slices (#42–#47) were cut before it closed, so the remaining ~22 slices
exist only on that branch. Cut them from `consistency-release`, not from a reopened #40.
Do not reopen #40 or open a replacement for the whole thing.

## 4. The split of #40

#40 is ~4,000 lines. I asked for no more than ~100 lines of *executable* code per PR
(prose and tests can be longer) so they are actually reviewable. `CONTRIBUTING.md` on the
stack carries the rule under "Size: about 100 lines of code".

Six are pushed:

| PR | Branch | Code lines | Base | Files |
|---|---|---|---|---|
| #42 | `metric-contract-module` | 88 | main | `metrics.py`, `tests/test_metrics.py` |
| #43 | `trend-gaps-are-not-numbers` | 50 | #42 | `timeseries.py`, `anomalies.py`, `forecasting.py`, `tests/test_observed_periods.py` |
| #44 | `rate-trends-average` | 90 | #42 | `aggregation.py`, `tests/test_rate_trends.py`, `tests/test_bug_bash.py` |
| #45 | `identifier-name-rule` | 32 | main | `schema.py`, `tests/test_identifier_names.py` |
| #46 | `rate-change-in-points` | 18 | main | `formatting.py`, `tests/test_rate_change_units.py` |
| #47 | `partial-period-and-selection` | 23 | main | `pipeline.py` (trim only), `tests/test_partial_period_trim.py` |

**Merge order:** #45, #46, #42 first (no behaviour change at all), then #47, then #43 and
#44 on top of #42. Then #41. Then close #31, #35, #40; #36 goes back to its author.

**Still to slice, roughly 22 PRs:** `nlq.py` is 884 lines and wants about nine (ledger →
time scope → mentions → filters → execution → refusal gates); `analysis.py` 342 lines,
about four; `business_insights.py` 234, about three; `file_io.py` 166, about two;
`app.py` 149, about two; then `ui.py`, `ai_insights.py`, `forecasting.py`, and one
text-only docs/CI PR.

**Open question I have not answered yet:** 28 PRs at ≤100 lines, or about 12 at 150–200
lines each. Nothing past #47 should be cut until that is settled.

## 5. The eight defects a review of #40 found

A high-effort review of my own #40 turned up eight real bugs. Each is fixed on
`consistency-release` with a test that fails without the fix. They are the reason the
100-line rule is worth keeping.

1. **`timeseries.py`** — new `observed_periods(trend)` drops rows whose `Value` is NaN.
   NaN must not reach the arithmetic.
2. **`forecasting.py`** — call `observed_periods` *before* the `min_periods` length check.
   Note: the import line on `main` is
   `from timeseries import fit_trendline, period_grain, robust_scale` and differs from the
   release branch. A blind sed against the wrong line silently no-ops and leaves `F821`.
3. **`anomalies.py`** — same fix, returns `()` instead of `None`.
4. **`pipeline.py`** — the partial-period trim could empty the frame; guard so it only
   drops the oldest bucket when whole periods remain.
5. **`nlq.py`** — two different spellings of a value in the same column must both become
   filters; a second spelling naming a *different* column refuses.
6. **`analysis.py`** — a parenthesised negative must survive a sign prefix
   (`negative = negative or raw[0] == "-"`).
7. **`ai_insights.py`** — near-uniqueness alone disqualifies a column as a dimension.
8. **`ui.py`** — the histogram count column must go through `distinct_label` so it cannot
   collide with the binned column's own name.

Tests live in `tests/test_contract_review_fixes.py` (9 tests; verified 8 failures and 1
error against the unfixed tree).

## 6. Traps already hit — do not repeat

- `_usable_dimension()` takes `(dataframe, column)`, not a Series. `TrendSeries.notes` is
  a property, not a method.
- A daily fixture can resolve to a non-daily grain, so the gap row never exists. Use a
  monthly fixture for gap tests.
- `pipeline.py` held three unrelated changes; two depend on code that has not landed
  (`CleaningReport.numeric_cells_unreadable`). #47 takes the trim only.
- Jest hoisting: a mock factory cannot close over an outer `filters`; name it `mockFilters`.
- Literal dates in fixtures are caught by the no-expiring-fixtures guard. Use
  `localDateString()`.

## 7. The other repo

Three branches are pushed on the mobile app and **they are stacked, not independent**:

`home-streak-counts-goal-days` ← `name-your-own-activity` ← `search-finds-your-own-food`

The last one contains all three. No PRs opened for them yet. A setup SQL file on that
branch adds a column and **has not been run** — running it is mine to do, and nothing
should run it automatically.

## 9. Review of #41 — changes requested

Reviewed 2026-09-21. Good idea, four real defects, CI red. All four reproduced against
the branch; 293 tests pass but `ruff check` fails on import order in `ui.py`.

1. `heatmap_frame` fills empty cells with `0` (`unstack(fill_value=0)`, `aggregation.py:400`),
   so `summarize_heatmap`'s `idxmin` reports a segment that did not exist in a period as a
   real low of 0.00. Same defect as review finding 1 — a gap is not a number.
2. `segment_frame` cuts to the top 12 with no "Other" row (`aggregation.py:288`), so
   "lowest" names the twelfth-ranked segment, not the lowest. The sentence is false.
3. `summarize_trend` reports a rate as a percentage of itself: churn 5% → 7% reads
   "+40.0%". A rate moves in percentage points.
4. `abs(first_value)` makes a sign flip read "−100 → +50 (+150.0%)".

Also: numbers bypass `format_number` so currency and `%` are lost and the heatmap summary
prints a raw timestamp where the axis prints `Jan 2025`; `summarize_movement` can name
"Other segments" as the largest increase.

The blocker to state first is the diff itself — the feature is ~40 lines but `ui.py` was
reflowed for ~400, which buries the change and will conflict with the stack.

## 8. What is left

1. Review the open PRs one at a time, starting with the no-behaviour-change ones.
2. Answer the 28-vs-12 question, then cut the rest of the #40 slices.
3. Merge the stack in the order in section 4, then close #31, #35, #40.
4. #36 goes back to its author to split.
5. Decide whether to unstack and open PRs for the three mobile branches; run the SQL.
