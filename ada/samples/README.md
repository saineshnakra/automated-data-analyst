# Sample datasets

Four synthetic files for trying ADA without your own data. They are generated,
not real business or lab data, so they can be used freely in demos, issues and
tests.

Each one exercises a different part of the analysis.

| File | Rows | Contents | What it shows |
|---|---|---|---|
| `lab-measurements.csv` | 936 | Weekly cell-culture assay by treatment arm | A rate averaged while the count beside it is summed, one arm declining, one bad plate |
| `saas-subscriptions.csv` | 216 | Monthly MRR by plan and region | A revenue drop in April 2025 that anomaly detection finds, and a forecast that beats no change |
| `support-tickets.csv` | 2,400 | Support tickets by team and priority | No money column, so resolution hours are averaged rather than summed, and a forecast that only just beats no change |
| `ecommerce-orders.csv` | 4,000 | Orders by category and channel | Returns as negative rows, so totals and shares have to handle mixed signs |

## Using them

- In the app, choose **Try a sample dataset**.
- In Python, `ada.sample("Lab Measurements")` returns a DataFrame and
  `ada.samples()` lists the files. With no name, `ada.sample()` loads Ecommerce
  Orders.
- On the command line, `ada samples` lists them, and `-` in place of a file
  name (`ada analyze -`) uses Ecommerce Orders.

The files are installed with the package.

## Things to try

**`lab-measurements.csv`**
- ADA picks `Viability %` as the measure and averages it. Asked about `Live
  Cells`, it sums them.
- Anomaly detection flags July 2025, where a planted bad plate lowers viability
  across every arm at once.
- Ask *"lowest treatment by viability %"*: the answer is Compound B, which
  declines across the run. Then ask *"live cells by treatment"*.
- Nine `Viability %` cells are blank, as in an instrument export. The data
  checks card reports them; ADA does not fill them in. It also flags the bad
  plate's 12 readings as extreme values, cut off from the rest, and the summary
  says to check them first.

**`saas-subscriptions.csv`**
- Anomaly detection flags April 2025, where a drop is planted.
- Ask *"which plan grew fastest?"* and *"MRR by region"*.
- The forecast backtest reports MASE 0.40, better than assuming no change.

**`support-tickets.csv`**
- ADA picks `Resolution Hours` as the measure, since there is no revenue
  column, and averages it: hours are neither money nor a count, and a sum of
  resolution times means little. A rise is still flagged as bad news, because
  the name ends in `hours`.
- The forecast backtest reports MASE 0.96, only just better than assuming no
  change.
- Ask *"average resolution hours by team"*, *"how many tickets?"* and
  *"reopened by team"*, which averages the 0/1 `Reopened` flag into a reopen
  rate.

**`ecommerce-orders.csv`**
- About 7% of rows are returns, with negative units and revenue.
- Ask *"revenue by category"* and *"which channel declined most?"*.

## Regenerating

The committed files are the version of record; the repository has no script
that regenerates them. `ada/demo_data.py` shows the fixed-seed approach the
built-in demo uses, if you want to build a sample of your own the same way.
