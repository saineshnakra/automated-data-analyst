# Sample datasets

Four synthetic files for trying ADA without uploading anything of your own.
They are generated, not real business data, so they carry no privacy or
licensing baggage — use them freely in demos, issues and tests.

Each one is shaped to exercise a different part of the analysis.

| File | Rows | Shape | What it shows |
|---|---|---|---|
| `lab-measurements.csv` | 936 | Weekly cell-culture assay by treatment arm | A rate column averaged while the count beside it is summed, one arm declining, and one bad plate |
| `saas-subscriptions.csv` | 216 | Monthly MRR by plan and region | A deliberate revenue drop in April 2025 that anomaly detection finds, and a forecast that beats no-change |
| `support-tickets.csv` | 2,400 | Operational tickets by team and priority | A dataset with no money column, and a forecast honest enough to say it is no better than assuming no change |
| `ecommerce-orders.csv` | 4,000 | Orders by category and channel | Returns as negative rows, so totals and shares have to cope with mixed signs |

## Things worth trying

**`lab-measurements.csv`**
- ADA picks `Viability %` as the measure and **averages** it, while `Live Cells`
  beside it is **summed**. A rate and a count in one file, handled differently.
- The anomaly radar should flag **July 2025**. That week is a planted bad plate:
  viability collapses across every arm at once.
- Ask: *"lowest treatment by viability %"* — it should name **Compound B**,
  which declines steadily across the run. And *"live cells by treatment"*.
- Nine `Viability %` cells are blank, the way an instrument export is. The
  cleaning report says so rather than filling them in.

**`saas-subscriptions.csv`**
- The anomaly radar should flag **April 2025**. The drop is real and planted.
- Ask: *"which plan grew fastest?"* and *"MRR by region"*
- The forecast reports MASE 0.40 — comfortably better than assuming no change.

**`support-tickets.csv`**
- ADA picks `Resolution Hours` as the metric because there is no revenue column.
- The forecast reports MASE above 1 and says so. That refusal is the point.
- Ask: *"average resolution hours by team"* and *"how many tickets?"*

**`ecommerce-orders.csv`**
- About 7% of rows are returns, carrying negative units and revenue.
- Ask: *"top 5 categories by revenue"* and *"which channel declined most?"*

## Regenerating

These files are committed so they can be downloaded straight from GitHub, and
they are the version of record — there is no generator script in the repository
to reproduce them from. `demo_data.py` shows the fixed-seed approach the
built-in demo uses, if you want to build a sample of your own the same way.
