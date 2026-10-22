# What ADA does, in detail

The [README](../README.md) gives the short version. This page lists what each
step does, in the order ADA runs them, with links to the reference pages that
hold the exact rules and thresholds.

- **Cleans conservatively and says what it changed**: encodings (UTF-8,
  UTF-16, Latin-1), `;` and tab delimiters, a title line above the header, a
  second header row under group labels, `$1,203.55` and `(48.10)` style
  numbers, day/month order in dates, dates written as `19540601`, missing-value
  markers (including a lone `T` for a trace, `<3,87` or `ND` below a detection
  limit, "not stated" phrases such as `no indica` or `tidak ada data`, and
  placeholders such as `Unknown`), ranges a spreadsheet turned into dates
  (`Oct-30` for `10-30`), labels spelled differently only in case or accents
  (`Con riego`, `CON RIEGO`). Identical rows are counted, not dropped;
  repeated headers and dropped empty columns are named.
- **Checks the data before trusting it**: every numeric column is scanned for
  values its name or a physical limit rules out (a percentage over 100, a
  negative yield, pH past 14, potassium of 19 mmol/L, a wind speed of 280 m/s),
  codes such as `-999` or `999.9`, values piled up at a limit, filled-in gaps,
  values far from the rest, second clusters, swapped and copied columns, max
  below min, stuck sensors, and gaps that follow a group or another column;
  the table as a whole for copied rows and for flat, unrelated columns that
  suggest simulated data. The brief lists the issues covering the most rows
  and says which to check first; `report.quality()` returns them all. Only a
  clear code in the measure is set aside, with a note, but relationships and
  group comparisons are computed without the cells the checks flagged.
- **Detects column roles**: the date, the measure, the grouping column and an
  ID column, from names, values and cardinality. Numeric category codes (a 0/1
  flag, a status, a set dose, a swept frequency) group rows instead of being
  added up, while rating items (`RDB1` to `RDB3` on 0 to 10) stay quantities; a
  column of unique row numbers is an ID; a running total such as a rain gauge's,
  a constant, a p-value or a date stored as a number is never the measure.
  Coordinates, compass directions and quality flags are never compared as
  readings. A long table (one column naming the parameter, one holding the
  result) is analyzed one parameter at a time, a file stacking several tables
  is read as its largest one, and a matrix of sample or channel columns is
  measured by its row mean. Override any role with
  `ada.analyze(df, measure="OD600")` or in the app.
- **Knows what can be added up**: money and counts are summed; rates
  (`Viability %`, `Churn Rate`), measured levels (`Water Temp (°C)`, `pH`,
  `Concentration (mg/L)`), per-unit prices and anything it doesn't recognise as
  money or a count (`CO2`, `mpg`) are averaged, and so is a score or count per
  subject when each row is one participant or sample. Averages get no totals
  and no "share of total", and a change in a rate is reported in percentage
  points.
  When the name misleads, `aggregation="sum"` or `"mean"` (or **Combine rows
  by** in the app) says how the measure combines.
- **Prints readings in full**: `K` and `M` are kept for money and large
  counts, so a heart rate of 1,111 is not shown as `1.1K`; very small or very
  large readings use scientific notation (`3.624e+11`). A `$` needs both a
  money name and values that look like money, so an income band coded 1 to 15
  is not printed as dollars. Figures compared in one sentence never print
  alike, numeric groups are named with their column ("Status 2"), and an
  average resting on fewer than 30 rows says so.
- **Builds a timeline**: picks a period grain the data can fill, down to the
  hour, minute or second for a sensor log, drops a final period the data only
  partly covers, and notes both under the chart. A `Year` column (with a
  `Month` column, if there is one) counts as a date. Clock times with no date
  are read as times of day, never given today's date, and a column of elapsed
  seconds or minutes can be the time axis.
- **Compares like with like**: the latest change is measured on the groups
  that reported in both periods, a series with a yearly cycle is set against
  the same time of year, and a group's figure says when it covers only part
  of the dates. Logs stacked one after another in a file, and daily summaries
  appended below hourly readings, are found and kept apart.
- **Flags anomalies and changes in level**: periods outside a band around a
  Theil-Sen trendline, or around the usual level for the time of year when the
  series has a yearly cycle, checked at a finer grain when the chosen one has
  too few periods. A lasting change in level (an instrument swap, a method
  change, a fire) is found by binary segmentation and reported with its start
  and the levels before and after. Both are calibrated by simulation so a
  stable series raises a false alarm in about one analysis in twenty.
- **Forecasts with a backtest**: a trendline plus month-of-year seasonality,
  scored on held-out periods against a no-change forecast. It refuses to run on
  too little history, and says plainly when the forecast did worse than
  assuming no change.
- **Answers plain-English questions** ("average pH by site", "mean signal by
  timepoint", "which month of the year has the highest wind", "is pH
  correlated with temperature", "does pH differ by site", "what predicts
  strength", "does antibiotic use affect risk", "distribution of pH",
  "outliers in HGB", "90th percentile of lead", "how many distinct species
  per reef", "count by crop and year", "share of Result 1 by Gender", "which
  column has the highest average", "average growth by temperature") by
  turning them into a fixed query plan that pandas runs. Every column, value,
  grouping and condition the question names must reach the plan; otherwise
  the question is refused with the part ADA couldn't use, rather than
  answered approximately. A short name that fits one column ("strength" for
  `Concrete compressive strength (MPa)`) is used and the answer says so. Each
  answer states its filters, how many rows it used and whether the row limit
  left rows out; a ranking gives each group's n and says whether the groups
  really differ, a trend gives its Theil-Sen slope and peak, and an answer
  that uses values the data checks flag says so.
- **Tests group differences**: a measure compared across groups gets the n per
  group, Welch's test with a Mann-Whitney check (or Welch's ANOVA and
  Kruskal-Wallis for more groups) and an effect size, by medians and ranks when
  the data are skewed or capped, and Tukey's HSD names the pairs that differ.
  A gap within the noise, one whose two tests disagree, or one resting on a
  group of fewer than five reads as "no clear difference", never as a ranking.
  Ordered numeric levels (a dose, a depth) also get a trend test, a mostly-zero
  reading is compared on its means and the share above zero, and a grouping
  that combines two factors gets each factor's own test. A scan of every
  reading across every grouping names the largest differences, with p-values
  adjusted for the number of tests (a false-discovery rate for a screen of
  hundreds of readings), repeated rows of one subject count once, and exact
  copies of rows are set aside. A grouping cut from a reading (a size band
  from acres, an anaemia flag from haemoglobin) is stated as a rule rather
  than reported as a difference.
- **Compares rates and fits joint models**: a yes/no outcome gets its rate in
  each group with a chi-squared or Fisher test and the risk difference with
  its interval, and the readings that relate to the measure are fitted
  together by least squares, with coefficients, intervals, R² and a
  collinearity warning, when that says more than any one of them alone.
- **Reports what relates to the measure**: every numeric column's Pearson and
  Spearman r with the measure, with a 95% interval and the sample size, then
  the strongest other pair, or a plain "none of N pairs reaches |r| = 0.3".
  A total and its items, a mean or row maximum of other columns, a score
  standardised within groups, shares that add up to 100, a censoring flag and
  its capped reading, near-duplicates, participant numbers and plate positions
  are left out and listed. It leads with the figure one record doesn't
  decide, says when one record makes or hides a correlation, and when it is
  stronger, weaker or reversed within groups.
  Ask about a pair and it reports the same figures for that pair.
- **Analyzes a slice you select**: draw a box or lasso on the Explore chart, or
  click some bars, and get the same analysis for those rows beside the whole.

The web app accepts files up to 25 MB. By default up to 1,000,000 rows are
analyzed (`row_limit=` changes that): the newest rows of a table with a date,
or a random sample of one without, and every answer says which.

See [How it works](how-it-works.md) for the steps from upload to report, and
the [Reference](README.md#reference) for one page per step.
