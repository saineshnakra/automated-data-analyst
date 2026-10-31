# Changelog

## Unreleased

### Predicting a column

- The folds keep together the rows that share the first part of a structured
  id ("0013" of "0013_02") when no test row's first part is in training, as
  for Spaceship Titanic's travel groups.
- `results/kaggle/holdout.py` splits each Kaggle competition inside itself,
  the way its test rows differ from its training rows (the last stretch of
  time, a fifth of new groups, or a fifth of the rows), models the rest and
  scores the held-out rows with the competition's metric. Whole competitions
  are no longer set aside as validation and test sets.
- `ada auto` (and `ada.model(..., auto=True)`) searches wider than `ada model`
  on the same features and folds: CatBoost, LightGBM with random split points,
  a neural network and, under 100,000 rows, extremely randomized trees join
  LightGBM, XGBoost and the linear model;
  a second layer, LightGBM on the features and the first layer's out-of-fold
  predictions, joins the blend when it beats every first-layer learner on its
  own; the weights come from 25 rounds of greedy selection; and a regression
  under 100,000 rows also tries fitting without the rows the first fit misses
  by far, kept when that scores better in most folds. `--jobs N` fits N
  learners at once with the same result. Out-of-fold scores, `ada model`
  against `ada auto` on the same folds: House Prices RMSLE 0.1183 to 0.1174,
  Housing Prices (Kaggle Learn) MAE 13,624 to 13,131, Spaceship Titanic
  accuracy 0.8077 to 0.8119, Titanic 0.8451 to 0.8462, S5E10 RMSE 0.05604 to
  0.05599, S6E2 AUC 0.95555 to 0.95570, S5E12 AUC 0.73108 to 0.73122. It
  takes longer: an hour or two for a Playground table of 500,000 to 700,000
  rows, two learners at once. A random forest and nearest neighbours were
  tried and left out; neither earned more than 0.04 of a blend.
- LightGBM's and XGBoost's threads sleep while they wait (`OMP_WAIT_POLICY`
  is set to `PASSIVE` unless already set). With other work on the cores,
  spinning threads made a fit about a hundred times slower: 300 XGBoost trees
  on House Prices took 101 seconds instead of 0.65.
- A daily forecast with two years of dates also gets each series' value 364
  days before and the mean over the 15 days around it. On Store Sales, 400
  series held out over 16 to 31 August 2016 score an RMSLE of 0.3790 against
  0.3823 (stores closed in that stretch aside), and school supplies, which
  sell before September, 0.551 against 0.597.
- A series that sold nothing for 28 days predicts 0 when that scores better in
  most folds; before, the folds' total decided. On Store Sales the rule did not
  hold for the test, so 114 series with no sales in the last 28 days got
  forecasts of up to 80; in the last stretch the folds hold out, 8 of 108 such
  series sold again, and the other folds had 2 to 13.
- XGBoost joins LightGBM and the linear model, on the same features and
  folds, and the blend weighs all three on the out-of-fold predictions. The
  `model` extra installs it (`xgboost-cpu` on Linux). Cross-validated: House
  Prices RMSLE 0.1195 to 0.1185, S6E8 AUC 0.9671 to 0.9674, S6E9 0.9454 to
  0.9456, S6E2 0.9555 to 0.9556; Spaceship Titanic keeps LightGBM alone.
- In a table of 100,000 rows or more, a number whose values repeat is also
  encoded by each value's mean target, from the other folds, and by how many
  rows share its value, and the trees grow larger (255 leaves, learning rate
  0.03, values in up to 1,023 bins). On Playground S6E8 the cross-validated
  AUC rises from 0.9639 to 0.9671, and on S6E9 from 0.9418 to 0.9454.
- With more than two classes, a many-valued text value and an encoded number
  give their share of each class; before, such text was left out.
- A structured id whose first part repeats ("0013" of Spaceship Titanic's
  "0013_02") gives the size of its group.
- `ada.model(train, test)` and `ada model` predict one column of a table from
  the others, as a tabular Kaggle competition asks: features from rules on
  the columns' shapes, LightGBM and a regularized linear model blended on
  out-of-fold predictions, every score cross-validated, and the submission
  checked (one row per test id, no gaps, probabilities in range, test rows
  like the training rows) before it is written. Metrics: accuracy, balanced
  accuracy, F1, AUC, log loss, MAP@3, RMSE, RMSLE and MAE. It needs the new
  `model` extra (scikit-learn and LightGBM). See
  [Predicting a column](docs/reference/modelling.md).
- `results/kaggle` scores it on 22 tabular competitions split into train,
  validation and locked test, against each leaderboard's top 10%, and
  submits only when the score clears the bar by twice its uncertainty.
- When the test rows are groups the training rows never hold (stocks, stores
  or people that are all new), the folds are grouped by the column that names
  them, so the score is for unseen groups as the test set's is; the column is
  left out of the features.
- `spearman` (Kaggle's RankIC) scores ranked predictions, and MAE on a skewed
  target at 0 or above is fitted on log(1 + y), which keeps the median.
- Forecasts: when every test date is after the training dates, the folds hold
  out the latest stretches of time as long as the test's reach, each fitted on
  the time before it, and a daily target gets its own past per series (lags
  and means from the reach back) as features. The test predictions are
  refitted on every training row. A series whose target was 0 for the 28 days
  before predicts 0 when that scores better out of fold.
- A column read as numbers in one table and as text in the other ("NA" among
  House Prices' test counts) is read as numbers in both, and a linear model's
  predictions stay within the targets it saw.

### Reading claims and questions

- "More than 300,000 migrants arrived in New Zealand in 2023" in Stats NZ's
  migration estimates adds up estimate for direction Arrivals at the TOTAL
  sex and age: 228,001. "arrived" names Arrivals and "departed" Departures;
  a plural that names nothing else ("migrants") in a table of counts with
  total levels adds up its count; and standard_error, beside estimate, is
  the uncertainty of the count, not a second count.
- A level called a record in a year ("Net migration hit a record of about
  126,000 in 2023") must be the highest of the years up to it, or the lowest
  for "a record low" or a record it "fell to"; when an earlier year holds
  it, the claim gets `failed_check` naming that year. A record in the
  table's first year has nothing before it, and gets `missing_context`.
- A date the figures were published on (month_of_release) is no longer
  tried as another reading of the claim's year.
- "March was the busiest month in 2015, with 2,259 tickets" counts March's
  rows and checks that no month of 2015 has more. "The busiest day of the
  year was Thursday 12 March 2015, when 159 tickets were written" is read
  like "12 March 2015 was the busiest day".
- A weekday that is not the date's own ("Friday 12 March 2015", a Thursday)
  fails the claim.
- "which month had the most tickets" counts the tickets, as "how many
  tickets" does, and a question that only names a time ("rows in March
  2015") counts the rows in it.
- Year and Month make the dates when the table also holds a time of day with
  no date (a crash's Time), so "in 2022" picks 2022's rows; a time of day is
  never tried as another reading of a claim's year.
- Times of day: "before 10 am", "after 6:30 pm" and "between 22:00 and
  23:00" keep rows by the time on the date column, or on the one column of
  times with no date. A time is never the number a claim states.
- "Most tickets are written before 10 am" is more than half of them. With a
  number of its own ("Most of the 20,071 tickets cost $55") "most" is left
  as it is.
- Australian states: "Queensland" keeps State = Qld in a column of the short
  names (NSW, Vic, Qld and the rest). An adjective of nationality
  ("Australian roads") is not a territory named with it.
- "Queensland had the most deaths in 2022, with 297, more than New South
  Wales" and "Queensland's roads were the deadliest in 2022, with 297
  deaths" rank the states that year, not Queensland's years.
- Money: "cents" and "dollars" are units, a unit column such as UNITS gives a
  column's unit for the rows a claim picks, and an amount of money is never
  a count of rows (see Units and scale in the claims reference). "price"
  names a value in money.
- "eggs in New Zealand" keeps the series for New Zealand, not "Overseas
  accommodation prepaid in New Zealand", a thing there; "a dozen eggs" names
  "Eggs, dozen".
- A change "between September 2016 and September 2026" compares the two
  months, and "over the past ten years" is the ten years to the table's
  latest date. When the rows do not reach both months, no change is given.
  "since January 2022" keeps the dates from 1 January 2022 on, and a change
  since it compares January 2022 with the latest month.
- "written" and "issued" after what is counted say how the rows came about.
- "Ohio's Sherrod Brown raised about $96.5 million, the most of any Senate
  candidate" makes the same total for every Senate candidate and holds only
  if Brown's is the highest; "Hakeem Jeffries raised more than any other
  House candidate, almost $23 million" is read the same way. Before, a
  closing rank left the claim at `missing_context`.
- "Treasury bonds still carried an average interest rate above 6%" and
  "TIPS paid an average interest rate of 1.1%" had that rate: before a rate,
  "still carried" and "paid" read as "had", as a rate is no payment; and
  "interest rate" names
  avg_interest_rate_amt, whose "amt" only says it is an amount, so a wrong
  rate fails rather than getting `missing_context`.
- "raised" names receipts, and a word that names a total and the rest of it
  (Total_Receipt and Other_Receipts) names the total. "House", "Senate" and
  "President" keep the letter a column of office codes writes (H, S, P).
- "About 2,664 agencies began reporting to NIBRS in 2021": a verb of
  beginning before a word of a start date's name ("nibrs" of
  nibrs_start_date) names that date. "More agencies began reporting in 2021
  than in any other year: 2,664" is a peak over the years.
- Fixed a wrong verdict: a peak over the years counted the rows with no date
  as a year of their own, so 4,061 agencies yet to start NIBRS outnumbered
  2021's 2,664. Rows with no date are in no year.
- "There were 5,034 film permits in 2025" in Film_Permits.csv: the table's
  own name before its rows' noun names every row, so "film" picks no rows
  though Category holds Film.
- A word that fits part of a column's name ("roads" in Road User) no longer
  turns "2,000 people died" into a count of Road User values.
- "Eviction notices in San Francisco fell to 778 in 2020, the fewest of any
  year since records began in 1997": the file's noun for its rows at the
  head of a claim counts them for a level as for a change, also before
  "in", a participle or a relative word ("Terry stops in Seattle fell",
  "Notices citing non-payment rose"); "the fewest of any year" is a record
  low set against every year; "citing" before a yes/no column keeps its
  yes rows. Before, the claims guessed a column.
- "Women make up about a third of the artists" in MoMA's Artists.csv:
  "women" keeps Gender = female, though the column also holds labels such
  as "transgender woman" once; other labels on under 1% of the answered rows
  leave it a column of sexes.
- A dash, "?" or "." among Y and N (Seattle's Frisk Flag: Y, N or "-") is a
  missing answer, so the column is a yes/no column and "share of stops
  frisked" is 17,280 of the 69,121 stops with an answer.
- Fixed a crash: a share said as "two men in five" raised TypeError, because
  a list of plurals and a map of singulars to plurals shared one name.
- Fixed a wrong table shape: a dozen columns numbering districts and
  neighbourhoods from 1 (Supervisor District 1 to 11) made San Francisco's
  eviction notices a matrix of counts, measured by their row mean. A
  column numbering a few categories from 1 is no count.
- A change that agrees with ends that do not is made again by each other
  date column and, when rows are counted and the ID repeats, with each ID
  once; when one gives the change and both ends, the claim gets
  `ambiguous_intent` ("Terry stops ... from 8,798 to 3,651": 3,670 rows by
  Occurred Date, 3,651 stops by Reported Date).
- A date only some rows have (Constraints Date, on 9% of the notices) or
  the date the data are as of (data_as_of, data_loaded_at) is no longer
  tried as another reading of a claim's year.
- "The number of girls named Sophia fell by about a third between 2015 and
  2021, from 525 to 342" in NYC's baby names: a change between two years
  of a column of whole years (Year of Birth) compares those years when the
  table has no date, and "the number of girls" or "babies" adds up the one
  column named only as a count (Count, Freq or N) when no ID names each
  row once. Before, it was the share of rows and failed.
- In a table whose rows are deaths (ardd_fatalities.csv), "died" and
  "killed" name every row, and "deaths" names the rows as "fatalities"
  does: "Deaths rose by about 8% between 2020 and 2022" counts them.
- "in the year to September 2026" with a word of change, on monthly rows,
  is September 2025 to September 2026.
- "cyclists" names "Pedal cyclist": the last word of a value is matched with
  its plural too. "motorcyclists" may mean Motorcycle rider or Motorcycle
  pillion passenger, and the question asks which.
- "Pennsylvania and Texas each have 1,477 agencies" is checked for each name
  on its own. A closing rank whose words only name the groups ("the most of
  any state", "more than any other state") is left out of the reading and
  checked after, over every state.
- A table with a column for each year ("1969 Deaths" to "2024 Deaths", as
  Statistics Finland gives deaths by cause) is read with a row for each
  year, so a year is a date like any other. Its areas and causes nest under
  subtotals; "WHOLE COUNTRY" and "00-54 Total" are read as their columns'
  totals, a column of up to 150 labels qualifies when one reads as a total,
  and "in Finland" is the whole country rather than every area named for it.
  "Deaths in Finland hit a record of more than 63,000 in 2022" is that
  level, set against every year; "how many people died" names Deaths.
- A workbook that opens on a sheet of notes (the Maddison Project's "Notes")
  is read from the sheet with the most numbers ("Full data"), and a note says
  so. Its short names are read in full: `pop` is population, `gdppc` GDP per
  capita, and "GDP per head" or "per person" means per capita.
- In a table of crime counts, "Motorcycle thefts" names
  theft_vehicle_motorcycle, a code of words in another order, and the one
  column of numbers, `crimes`, is what the claim counts rather than a guess.
  "how many rapes in Malaysia in 2023" gives the count when its share of all
  crimes cannot be read. "the most of any year since 2016" is a record set
  against the years from 2016. "collapsed", "plunged", "soared" and other
  strong verbs before "by", "to" or a percent read as "fell" or "rose".
- Fixed a wrong count: "how many thefts" in a table whose types are
  theft_other and theft_vehicle_motorcycle counted every crime; the question
  is refused and names the kinds of theft.
- "The SPD won Germany's 2021 election with 25.7% of the vote" is the share
  won in that election, as "won 25.7% of the vote in Germany's 2021 election"
  is.
- Fixed a wrong count: "how many episodes are in the data for each year" in
  `daily_show_guests.csv` counted the guests, two or three on some show dates.
  When a file names its rows as people and the question counts something
  else, the question is refused with what to ask instead.
- Statistics Denmark's files read: a semicolon-separated file whose values
  hold commas ("Tickets sold (1,000)") no longer fails as a CSV, a column of
  years named `TID` (or `år`, `Vuosi`) is the date, and the
  subtotals after a comma ("Feature films, all audiences") and a label marked
  "(all)" are total levels. Fixed a wrong total: "how many tickets sold in
  2025" added up every type and premiere row and gave 56,874 thousand, where
  the table gives 9,854.
- Year columns may give the figure's name before the year ("Number 2000" to
  "Number 2024", Statistics Sweden).
- "American films sold more than half of all tickets in Denmark" reads "in
  Denmark" as where, not as a second nationality; "a third of all tickets" is
  no rank.
- "imported rice" names "Rice (imported)", "their own water supply" names
  "self-supply", "permanently" names "permanent", and "Gotland" names "0980
  Gotland" in a column of coded regions.
- A question whose values are recorded only in another unit than the one the
  brief follows is answered in that unit ("average price of Maize", priced by
  the KG), or refused with the units when there are several.
- `S$`, `NZ$`, `HK$`, `₦`, `₩` and `₱` are currencies, "cost more than twice
  as much in 2021 as in 2019" is the change between the years, and "rather
  than ..." is set aside. A year's prices are never added up as another
  reading.
- A title line with a semicolon above comma-separated rows is skipped, rather
  than read as a header of two columns.
- Fixed wrong verdicts: in "About a quarter of samples at Bondi Beach
  exceeded 200 enterococci", the samples at Bondi Beach were counted out of
  every sample in the table, so a claim of a quarter verified at 26.5% where
  53% of Bondi's samples exceed 200, and a claim of a half failed. A verb in
  the past tense ("exceeded", "ended") now ends what a share is of, as "had"
  and "that" do.
- "100 COVID-19 deaths" no longer reads "COVID" as the unit of 100, which left
  "-19" in the question.
- "on Saturdays", "at the weekend", "on a Saturday or Sunday" and "on
  weekdays" pick the days of the week of the date column.
- "aged 40 and over" reads ages in years when an age class column ("40-44")
  sits beside them, where it read age = 40.
- "the men's 100 metres" is the race whose Length is 100, not a number
  claimed; "has come down by 0.013 seconds a year" is a yearly fall.
- "per 100 mL" after a column whose name gives its values per 100 mL
  (enterococci_cfu_100ml) says what it measures, "the median enterococci
  count" is the median of that column, and units at the end of a column's
  name (cfu, ml) are not the kind of quantity it holds.
- `UID`, `FIPS`, `GEOID` and `OBJECTID` columns are codes, not measures.
- A column that is a log or a root of another is looked for in at most 2,000
  pairs of columns, the closest first. A table with a cumulative column for
  each day (JHU's COVID-19 deaths, 1,155 columns) ranked almost every pair
  alike, and its brief took over half an hour; it takes about a minute.
- "Australia's wheat crop grew by about a quarter in 2024-25, to 34.8 million
  tonnes" checks the change and the level it reached as two claims, and holds
  when both do. Before, two numbers in one claim were refused.
- Months written "1993M01", as Statistics Norway gives them, are read as the
  first day of each month. `CODIGO DANE` and other columns named `codigo`,
  `codice` or `cod` are codes, not measures. In Colombia's road deaths, where
  no column could hold a count of deaths, "more than 1,500 road deaths in
  Antioquia" counts the rows, and "Colombia", also a town among the
  municipalities, is read as the whole table too: when the town and the whole
  table disagree, the claim gets `ambiguous_intent`. A column of 30 labels or
  fewer, a column of countries and a US state are left out of this.
- In BEA's personal income, whose Unit differs from line to line, an amount
  of money is read from Value in the unit of the rows used: "Texans' personal
  income topped $3 trillion in 2025" compares 2,319,750.7 million dollars,
  and a Description with one value says what Value holds. Before, "topped $2
  trillion" failed against 2,319,750.7.
- The last word of a value is matched without its plural ("cyclists" names
  Pedal cyclist), but not with one added: "the lowest of any state" no longer
  keeps GeoName = United States beside the state named.
- "unemployed people" names Spain's "Unemployed persons", and "Both genders"
  is a total like "Both sexes".
- "China" in a column of countries is not also "China, Hong Kong SAR", a
  place after a comma, so "China's GDP per head rose roughly tenfold" is
  checked.
- A failed check is no verdict when the reading is likely of something else:
  when each end the claim gives is ten times or more off the reading's ("from
  280 to 123" read as deaths from Duty related illness, 2 and 4), and when a
  Value column is averaged over rows that differ in a breakdown (Sex, Age,
  Category), or a share is of the rows of a table whose rows each hold a
  Value. These get `missing_context`.
- Fixed a wrong count: "how many deaths in each region", where each row holds
  a number of deaths, counted the distinct values of Deaths (6 in each
  region). It adds them up now, as "how many deaths in 2020" did, and so does
  "how many deaths each year", which was refused.
- Fixed a wrong count: "how many cyclists in 2020" in a table of road deaths
  counted 998 distinct Crash ID values. An ID that numbers events (a crash,
  an order, a visit) counts those events only; the people in them are rows.
- Fixed lost quarters: "2026Q3", "2026QIII" (INE), "2026 T3" and "Q3 2026"
  were read as the number 2026, so the four quarters of a year became one
  year four times over. They are the first day of each quarter now.
- The words after a scaled number ("2.6 million unemployed people") are no
  longer dropped when they name values the reading leaves out: the claim
  gets `missing_context` rather than a total over every value.
- Fixed a meaningless average: Stats NZ gives "Food" as an index and as two
  percent changes, and a mean over the three rows mixed them (452.57). A
  measure is no longer combined over rows whose unit column holds more than
  one unit; the answer names the units.

### Wrong verdicts removed

- A claim false under ADA's reading but true under another the words allow
  is no longer failed. A name several places share (Lincoln County in four
  states, Washington state and the District of Columbia), each occasion of a
  repeated design (ages 8 to 14 of each child), and another column named for
  the same thing (Deaths and Age-adjusted Death Rate) are tried first; when
  one agrees the claim gets `ambiguous_intent` or `missing_context`. A
  failure through a column ADA picked because nothing else fit gets
  `missing_context`, unless that column is the table's date and only places
  the year ("Sweden's 2022 election"), and the things counted after a number
  are never averaged from a column named for them: "Sudan had about 1.1
  million refugees" adds up refugees in each reading of Sudan (hosted 1,097,128,
  from Sudan 836,756). "casualties" is also tried as the dead alone, and a
  sum over a column holding Net beside Arrivals and Departures, with none
  named, is made for each of them.
  On sets 6 to 34 this removes 16 of the 17 failed checks of claims that
  hold under some reading.
- A word said of two values ("neutered" of Neutered and Spayed, "no hair"
  of No Hair and Bald) is checked with both: when the verdict rests on which
  is meant, the claim gets `ambiguous_intent`. "About 2,000 Marvel characters
  have no hair" failed on No Hair's 1,176; with Bald it is 2,014.
- "The average time in space was about 1,050 hours" in the astronauts table
  read total_eva_hrs as the only column in hours and failed. hours_mission
  and total_hrs_sum hold hours too, so no one column is guessed and the
  claim gets `missing_context`.
- Rows repeated exactly: NYC's baby names give every 2014 row twice, so a
  sum over 2014 counts each baby twice (1,156 girls named Sophia, not 578).
  A claim on a table whose rows each hold a count, with such rows, is
  checked with each counted once too; when the two give different verdicts
  it gets `ambiguous_intent` with both numbers, and when only the number
  moves the check says so. Rows that are records may repeat as separate
  records, so a table of them is checked as it is.

- Temperatures: "°C" and "°F" are units, and "Maximum Temperature (C)" is
  in degrees Celsius. "Edmonton International fell to -46.6 °C" is the
  lowest Minimum Temperature (C) at that station, "Blatchford hit 37 °C"
  the highest Maximum Temperature (C), and "Blatchford averages a high of
  about 24 °C in July" the mean of the maximum in July. "the average high
  temperature" and "the lowest temperature" pick the maximum and minimum
  columns. A claim in °F about a column in °C gets `conflicting_evidence`.
  Before, these claims guessed Station Elevation (m), and "fell to" was read
  as a change between periods.
- "fell below -40 °C on more than 20 days" counts the days whose minimum is
  below -40, rather than reading "fell" as a fall from one year to the next;
  "rose above 30 °C" uses the maximum.
- A day's reading called a record ("The coldest reading in the records came
  on 12 January 2024, when Edmonton International fell to -46.6 °C") is
  checked against every day too: when another day is colder, the claim gets
  `failed_check` naming the record.
- "on 5 days" counts days; it no longer reads as Day = 5 in a table with a
  day-of-the-month column. "Day 5" still does.
- Times in UTC: a date column whose name says UTC (START_DT_UTC) or whose
  values carry the UTC zone gives answers by hour, weekday or date with a
  caution that these are UTC's. A claim checked on such dates is checked
  again in every time zone from UTC-12 to UTC+14; when some zone turns the
  verdict ("110 incidents on 25 March 2026": 101 in UTC, 110 in Calgary's
  time), it gets `missing_context` with the range the zones give.
- Fixed lost dates: Stats NZ writes months as numbers ("2006.10" for October
  2006), which read as 2006.1, so its price indexes had no date column. A
  column named for a period, month or date whose values are a year and a
  month after the point is read as monthly dates.
- Fixed wrong totals: in Stats NZ's migration table the date was
  month_of_release, when each estimate was published, rather than
  year_month, the month it describes, so "net migration in 2023" gave
  -10,805 instead of 126,196. A date named for its release, publication,
  report or revision gives way to another date.
- Fixed a wrong answer: "Hakeem Jeffries" in a column of names written
  surname first picked both "JEFFRIES, HAKEEM" and "HAKEEM, JAWAD HASHEM",
  another candidate, and added up their receipts. A full name now names the
  value written surname first.
- "NIBRS" names is_nibrs, a flag of whether each agency reports through
  NIBRS, rather than nibrs_start_date, when one name fits a flag and a date.
- "California has more than 1,500 law enforcement agencies on the FBI's
  list" in fbi-agencies/agencies.csv counts the rows: the rows' own noun
  right after the number is what it counts, and "list", "register" or
  "database" names the table as a whole.
- Fixed lost rows: a date column holding "NA" for missing dates
  (nibrs_start_date, "NA" on 4,061 of 19,166 rows) stayed text, because the
  "NA" cells counted against the share of cells that read as dates. They are
  missing values now, and the column is read as dates.
- Fixed a dropped row: "IDA" among US county names (Ida County, Iowa) was
  read as the World Bank's IDA group and left out of every answer. A group
  known only by its initials (IDA, IBRD, HIPC, OECD) is an aggregate only
  beside other aggregate rows such as World or an income group.
- Fixed a wrong answer: "how many incidents in 2024" in Traffic_Incidents.csv
  counted the one row whose description says "multiple incidents in the
  area", because "incidents in" was read as part of that value. The file's
  own noun for its rows, with query words around it, names no value.
- "on more than 20 days" counts the rows of one station when each of its
  rows is one day, in a table of several stations.

- "Half of all rail disruptions were over within 51 minutes" is checked as
  the median; "the first of July 2025" is a date; "D was the worst day of the
  year, with fifty-one disruptions" is checked against every day of the
  year, and fails when another day has more.
- Fixed wrong verdicts on a table of deaths by single year of age (NISRA):
  "ALL" and "All ages" among the ages were read as blanks, so a share of
  people who died aged 80 or over was taken of rows. A total label in a
  column of numbers now keeps the column as labels, ages 0 to 115 count as
  levels beside it, a code beside its label (SEX, Sex Label) is read with
  it, and "centenarians" are every age from 100.
- Cancer registry tables (PHS): "All cancer types" is the total of sites
  that overlap (Colon within Colorectal cancer), found as the level at least
  as large as every other in the largest column of counts; "diagnoses" are
  the incidences for all ages; "prostate cancer" is CancerSite = Prostate,
  read from the label over its code; "melanoma" is not "Non-melanoma skin
  cancer"; "women" is Sex = Females beside "All".
- Counts by age band kept in columns (IncidencesAge75To79 to
  IncidencesAllAges) are read as rows of an Age column when a question names
  ages, so "aged 75 or over" adds up those bands.
- "sea level" names MSL, mean sea level; "since records began in 1856" is
  from 1856; a closing comparison ("three times the long-run pace") fails a
  wrong number but never verifies a right one; words of a table's name that
  name a column stay in the claim.
- Notes printed below a table ("Last updated: ...") are left out, with a
  note; "personal income" names "Personal income (millions of dollars)", a
  value in sentence case, not "Per capita personal income"; "deaths
  registered" leaves no word out; "per year" never divides by a column of
  years.
- "Number Of Trees" counts trees, and trees_in_camden.csv's rows are trees.
  Such a count is 1 on most rows; n_cats, above 1 on most rows of
  cats_uk_reference.csv, counts each cat's household, so "how many cats"
  still counts the rows there.
- "how many drivers" in a table whose one column of counts is Drivers adds
  them up (19,543,658 in Texas in 2024), with no rows of totals needed.
  Before, it was refused as a question about a measurement.
- A claimed total that names no year is not failed by the total over every
  year when one year's total fits it: "California recorded about 5,000
  traffic deaths" over 1982 to 1988 gives 35,315 in all and 5,020 in 1984,
  so ADA asks which year is meant. "In all", "altogether" or a named year
  is checked as said.
- Fixed a wrong verdict: "rose by about 1 a year" over a single year was
  checked against a plain average. A change each year needs a slope over
  time, or the claim is refused; "per year" never divides by a column of
  years.
- Age bands written "[70-80)" (70 to 79) or "06-11 Months" (age 0) are read
  as ages, and "Americans under 18" or "women over 65" bound the age as
  "people under 18" does.
- An amount of money ("valued at almost £1 billion") is never checked
  against a count of things.
- "per 1,000 inhabitants" in a claim is read as the base the table already
  gives its figures in ("Passenger cars - per thousand inhabitants"), and
  "Germany had about 590 cars" is checked against that figure, never against
  a count of rows; a rate per thousand is never added up over rows.
- A highest, lowest or total over every row the claim keeps that already
  falls short of the claim fails it, whatever words were left out, since
  they could only keep fewer rows ("topped 8,000 vehicles an hour" against a
  busiest hour of 7,280); a left-out span of time longer than a row keeps
  the claim open.
- Digits after a letter and a hyphen ("I-94") are a name, not a number.
- Parts of a whole kept in columns that add up to it (Good Days to
  Hazardous Days beside Days with AQI; the methods of travel beside the
  census total) are read as rows of a Kind column when a question asks for
  a share of one, so "good on 75% of the days" is the good days over all
  days, added up over the rows; "across US counties" is every row, not each
  county.
- "Air quality was good on 75% of the days", "Air quality in Maricopa County
  was good on 8.8% of days" and "Maricopa County had good air on fewer than
  one day in ten" are shares of the days; "one day in ten" is 10% of days;
  ", home to Phoenix," says what a place is.
- Fixed a wrong verdict: on a small table, "Number of residents" with a
  different count on every row was read as an id and counted as distinct
  values. A column named "Number of ..." holds counts.
- Fixed a wrong answer on a table of deaths by age (CDC): "total COVID-19
  Deaths under 18" was read as the rows with fewer than 18 deaths. Deaths,
  cases and admissions under an age are of people that age, and "0-05
  Months" beside "06-11 Months" are both age 0, so "under 18" keeps every
  age below 18.
- "died of COVID-19" names the column "COVID-19 Deaths"; "what share of
  COVID-19 Deaths were 65 or older" adds up the deaths of those ages over
  all of them, for a column whose name says it counts (a plural or "count";
  Day, a day of the month, is never added up); "between January 2020 and late June 2023" is no filter when
  every row covers that period (Start Date and End Date the same on every
  row), and any other span is still refused.
- A claim whose word names two columns of numbers alike ("the deaths" for
  Total deaths and COVID-19 Deaths) gets `ambiguous_intent` when the two
  give different verdicts, with both readings.
- "COVID-19 was involved in nearly one in ten deaths" is the total of
  COVID-19 Deaths over the total of Total deaths, a column of counts that
  never exceeds the total of its kind on a row being a part of it; "nearly
  one in ten" stands for 9% to 10%, not 0% to 10%.
- "the EU" in capitals names "European Union - 27 countries (from 2020)",
  the one value whose words before a dash spell it, as "the UK" names
  "United Kingdom"; in lower case, or with two such values, it names none.
- A word inside a web address ("map" in ".../Domus-augustana-map.png") names
  no value, and "on the map" says where the rows are, as "in the dataset"
  does.
- In a table of ONS geography codes (E06000001, W06000015), "Welsh workers"
  are the rows whose code starts with W and "England and Wales" is every row
  when each code starts with E or W; "drove to work" names "Driving a car or
  van", the one part whose first word it is.
- Fixed a wrong verdict: "Texas had more than 25 million drivers" against
  17.8 million was verified, reading "million drivers" as the unit Drivers
  is recorded in, since 17.8 million millions is more than 25. Only a claim
  bounded at both ends ("about 37,528 million tonnes") is read that way.
- "Texans", "Ohioan" and the other names for the people of a US state pick
  that state's rows, also after a number ("More than 20 million Texans").
- Fixed lost rows: a column of classes such as "3 ROOM", "4 ROOM" and
  "EXECUTIVE" (Singapore's flat types) was read as numbers, leaving the
  4,194 Executive flats blank. A word with no number on 1% of a column's
  cells or more makes it a column of classes.
- "four-room flats" names flat_type = 4 ROOM, a number word written in
  digits, and "resales" counts the resales rather than naming resale_price:
  the plural of a name's first word counts the things the rest of the name
  measures, as "orders" does beside order_value. The first word of several
  names ("candidates" for Candidate family name and Candidate given name)
  still asks which column is meant.
- "founded before the common era" is year below 0 in a table that writes
  those years as negative numbers, and a 0/1 flag such as year_approx is no
  column of years.

- Fixed wrong answers on published tables of counts (CSO, StatCan): each
  row is a cell of counts, so a count adds up the column of counts (`VALUE`)
  instead of counting rows, a change or total adds it up unless the question
  says average, and a share of a level is of its total level ("25 - 44
  years" over "All ages"). It is refused, with the reason, when another
  column would add up different counts (immigrants with emigrants), a count
  is missing or below zero, or the unit column gives a rate or two units.
  Counts beside percentages of them ("Number of live births", "Percentage
  of live births") are read at the counts.
- Total levels are found behind words every label shares ("Age of mother,
  all ages"), cell by cell when some counts are blank or a level starts
  partway through the years, and with no total word when a label's counts
  are the others' added up ("Canada" over the provinces).
- A unit column that gives one scale for every row ("Thousand") scales the
  value checked, the ends of a change and a base; "from about 74,000 to
  149,000" says "about" of both ends.
- Ages: "aged 18 and 19", "a child under 15", "teenage mothers" (under 20),
  "in their early forties" (40 to 44), and bands written "25 - 44 years",
  "65 years and over" or "Under 15 years" beside "All ages" and "not stated".
- "immigrants" names "Immigrants: All origins", the words before a value's
  colon; "immigration" and "emigrated" name immigrants and emigrants; "broken
  down trains" names "broken down train"; "caused" beside `cause_en` and
  `cause_group` names neither.
- "hit points" names `hp_number` by its initials, the number `hp` writes
  first, and "challenge rating" names `cr`; `hp_number` beside `hp` is read
  as numbers, not as a key.
- "the year to April 2024" on a table of one figure a year is 2024; dotted
  codes ("1.1.1") are not dates; a date-named column of whole years
  (`REF_DATE`) is the date; a start date is the date over an end date.
- Words that only describe a base the data confirm ("the 149,200 immigrants
  who arrived in Ireland") no longer keep a share from being checked; a base
  in round thousands is the denominator rounded; "the 15 parks", every value
  of a column, spans the table.
- When every column a claim's words fit gives the same verdict, the claim
  stands or falls on it; "nearly half" reaches down to 4/9, the next simple
  fraction below; words for births, babies and mothers name a table of
  births, and unknown words are left out up to three times over.
- A folder answer grouped by codes names them from the file they link to:
  airports by `iso_country` say "US is United States" from `countries.csv`.
- Fixed a wrong reading: "made up half of" lost the space before "50%" when
  "half" was written as a number, and an age range after it was misread.
- Fixed a wrong verdict: "rose by about 40 centimetres" failed against a
  column in metres whose name gives no unit. A claim that agrees once the
  column is read in another unit of its kind (metres for centimetres), or a
  bound that would, gets `missing_context`.
- "the investment" in a share weighs by `invest`, the column it names, before
  any other column of money (`value`).
- Fixed a wrong answer: "which national parks had the most visitors in 2016"
  gave the largest single number. Which rows of a kind are now ranked by the
  column that names each row (`unit_name`), or refused when none does.

- Fixed a wrong reading: "fell by more than three quarters" was read as
  "more than 3" and "fell by two thirds" as 2, so a fall of 10% agreed with
  the first. A change by several parts of a whole is that fraction.
- "The euro gained about 10 yen a year" and "averaged about $1.18" read the
  rate column the currency names (JPY, USD) in a table of exchange rates; a
  name two columns share ("krone" for DKK and NOK) names neither.
- "hit its all-time high on 15 July 2008, at $1.599" is checked against the
  highest value of the column too, and fails when another row is higher.
- "Prices in 2023 averaged five times their 2019 level" is the change from
  2019 to 2023; "a thousand" is 1,000; "below zero" bounds a column at 0,
  and "prices dropped below zero" links the column to its bound.
- "sales of light trucks" names `Light truck sales`, and "urban rail
  ridership" names `Transit Ridership - Urban Rail - Adjusted` rather than
  asking which ridership column is meant.
- A claim on a table of one row per date where a date repeats (BTS gives
  April 2023 twice) is checked with the repeat counted once and twice; when
  the verdicts differ it is ambiguous, and when they agree the check says so.
- ", the category that takes in SUVs and pickups," is a description and is
  left out; "Price (EUR/MWhe)" is named by "prices", the word outside its
  brackets.
- Fixed a cleaning error: "country" was merged into "county" as a typo, one
  letter apart, so a salmon table's national rows were counted as a county
  and "total losses of salmon in 2023" gave 214 million where there were
  71.4 million. A spelling on ten rows or more that never sits beside the
  same value of another column as the commoner one ("country" beside
  "Norge", "county" beside the counties) is a label of its own.
- A column of geographic levels (area, county and country, each row at one)
  is read at the whole unless the question names a part (decision
  `total_levels`).
- Fixed a wrong verdict: "Fentanyl was found in about 68% of the deaths"
  read "found in" as the value "Found in car". A value only partly written
  right after "was" or "were" is a verb, not that value.
- A column that holds only a mark or a blank ("Y" where xylazine was found)
  reads a blank as a mark not made, so a share is of every row: "what share
  of deaths had xylazine in 2024" is 342 of 982, not 342 of the 342 rows
  marked. "with", "involving", "was found in", "turned up in" and
  "without" pick its rows.
- "about 12 of every 100 cases" and "8 in every 100" are shares, as "one in
  five" is; a rate per 100,000 is not.
- "By 2024, xylazine turned up in 35 of every 100 deaths" is the share in
  2024 or over every year up to it; when the two give different verdicts,
  the claim is ambiguous.
- A description between commas (", a veterinary sedative,") is left out of
  a claim.
- "Fish farms lost about 71.4 million salmon in 2023" adds up `losses`: a
  verb of amount ("lost", "spent", "sold", "paid") names the column and
  adds it up unless the claim says how rows combine ("on average").
- "Illinois" picks State = IL in a column of postal codes that also holds
  PR and GU.
- "bike thefts" no longer keeps the two rows where BIKE_MAKE is "BIKE": a
  value that only repeats its column's name picks rows only when typed as
  the data spell it. A word that starts the names of three or more columns
  (BIKE_MAKE, BIKE_TYPE, BIKE_COST) names what each row is about, so "how
  many bikes" counts rows.
- "July" in a table with OCC_MONTH and REPORT_MONTH is the month a theft
  happened, unless the question says reported; "Created Date" wins over
  "Closed Date" the same way.
- "Births in Ireland rose by 20%" checks a table of births with no column
  of places: a country or US state that no column could hold is read as the
  place the whole table covers, and the check says so.
- "Prices were negative for 511 hours in 2025" counts rows in a table of
  one reading an hour; "for 511 hours" is no longer read as a base like
  "across the 48 states".
- A year-named text column of single years and spans ("2019", "2023 - 25")
  gets a column of its single years, so OHID's Fingertips tables have dates;
  the note says how many rows hold a span.
- "England" in Area Name, Parent Name and Area Type is the area of that name.
- "three quarters of a percentage point a year" is 0.75 points, not 75%, and
  a slope written in words is compared as one.
- In a long table, a filter on its indicator column (Indicator Name =
  Smoking prevalence) says what `Value` holds, so the check does not call
  `Value` a guess. "Persons" is a total level, as "Both sexes" is.
- "women" picks a gender column that also holds "transman".
- "Half of the people who died were 44 or younger" reads the age bound ("44
  or younger", "18 and under") and the share past it; "within 19 days" and
  "mothers under 20" bound the rows, and the share of a group ("more than
  85% of the babies") is the number claimed. "died" names a table of deaths.
- "Fewer than one in five of the 85 candies" and "Of the District's 187
  homicides" name the base the share is of, and the base is checked. "is a
  hard candy" picks the 0/1 column `hard`.
- "shortlisted writers" is no part of "The Crime Writers' Association": one
  lower-case word picks a capitalised name of three or more words only when
  another of its words follows.
- Fixed wrong answers on cross-tabulated tables: CDC's injury deaths give
  every count under each sex, age band, race and intent and again under
  "Both sexes", "All Ages", "All races" and "All Intentions", so "total
  Deaths for Poisoning in 2016" added up 1.1 million deaths where there were
  68,995. A column the question does not name is now read at its total level,
  and a grouping leaves its total level out (decision `total_levels`).
- Fixed a wrong answer: "how many ships were wrecked in Irish waters in each
  year of the First World War" kept only the two wrecks whose descriptions
  hold "ships were wrecked" and "First World War". A column of long text is
  free text even when many of its values repeat, and words inside it pick no
  rows.
- Fixed a wrong verdict: "The number of votes fell by more than a quarter
  between 2020 and 2022" failed against the 8% fall in the number of rows
  (candidates). A plural that columns of counts end in (`candidatevotes`,
  `totalvotes`) is no longer counted as rows: one such column is added up,
  and several are named in a refusal.
- "median salary each year" is a trend of `Annual Salary Median`, not a
  median of it: a summary word that names a column is part of its name.
- "Starr County, Texas" picks the state: a US state's name that a column of
  states holds is the state, even when a column of county names holds Texas
  County, Missouri, unless "County" follows it.
- Fixed wrong answers on NOAA's storm events: the unit column ADA adds for
  `DAMAGE_PROPERTY` ("0.00K", "1.5M") was taken as the unit of the whole
  table, so every answer covered only the 54,231 rows in K of 69,801. A unit
  column read off another column no longer splits the table, and the
  fatalities file now joins the details file on `EVENT_ID` (every key found).
- "the tracked frogs" and "the located earthquakes" pick no rows: words for
  how the rows were gathered are read as grammar before a noun, as
  "recorded" is; "how many cats were tracked" is still refused.
- "the median salary" names `Annual Salary Median` rather than offering it
  beside `Annual Salary Avg`, and "earned" in "nurses earned a median salary
  of about $33,000" says what a column of money holds.

- Fixed a wrong verdict: "US milk production rose 46%, from 147.7 billion
  pounds in 1990 to 215.5 billion in 2017" failed because the change was read
  as one in the states' average, whose ends are 2.95 and 4.31 billion. Ends
  that count something are now totals, as a level in pounds is, and "total"
  in a change ("total milk rose") compares the totals.
- Fixed a wrong answer: "which pies do people serve at Thanksgiving" found no
  rows, reading "pies" as a value of a free-text column. A select-all question
  that a survey export splits into a column per option ("... - Pumpkin",
  "... - Apple") is now read as one question, and the answer counts the rows
  that tick each option. A share of one option is of the rows that ticked any.
- Fixed a wrong verdict: "More than half of the outbreaks since 2009 were
  norovirus" failed at 23.9%, counting only the label `Norovirus`; NORS also
  writes it under 251 other labels ("Norovirus Genogroup II", mixtures joined
  by ";"), which make 57%. A one-word value that starts other values of a long
  list is now another spelling, for a share as for a filter, and the claim
  asks which is meant. "Animal Type" no longer counts the outbreaks as kinds
  of animal: an entity column's name must end in the entity word or an ID.
- A change each year is read as the slope of the trend: "Idaho's milk
  production has grown by about 500 million pounds a year since 1990" is
  checked against the least-squares slope of the yearly values (500,640,120)
  and the Theil-Sen slope (506,284,524), and is ambiguous when only one agrees.
  A trend's answer now gives both slopes per year, and a trend takes "total" or
  "average" ("total milk per year") instead of refusing a word that differs
  from the default.
- Fixed a wrong verdict: "Visits to Great Smoky Mountains National Park rose
  to a record 11.3 million in 2016" failed against a count of 1 row, reading
  "record" as `number_of_records` and "Visits" as the folder's name. "rose to
  a record N" is now a record level, "visits" names `visitors`, and a count of
  rows read through the table's name gives `missing_context` when it
  disagrees.
- "has added about 80,000 visitors a year" and "has lost ..." are slopes, with
  the words after the number naming the column that changes.
- "Milk production" names `milk_produced`: the noun of a verb in a column's
  name says what it measures, so a wrong number there fails instead of asking
  which column is meant.
- Fixed a crash: a column of a few large whole numbers (900,000,000 pounds)
  made the survey-code check list every number between them and run out of
  memory. A missing date beside timed rows no longer overflows the check for
  daily rows appended to hourly ones.

- Fixed a wrong verdict: "South Korea's fertility rate fell to a record low of
  0.72 in 2023" failed because "low" picked the value Low-income countries; a
  word of degree now picks only a value spelled so. The record low is checked
  as the level, the fall and the lowest of the years up to it.
- A change "since 1950" compares 1950 with the latest year, not the last two
  years; a multiple or fraction of a rate ("halved") is in percent of the first
  value, and any other percent of a rate is ambiguous.
- Fixed wrong answers on DC's crime incidents: the `ANC` codes ("7D", "6C")
  were read as readings in units, and every answer then covered only the 6,693
  rows "in unit C" of 29,304. Whole numbers glued to one of several letters now
  stay codes.
- A verdict that rests on a year is made again with the table's other date
  columns; when one turns it, the claim is ambiguous. A column of codes such as
  `WARD` is never summed.
- Fixed a wrong verdict: "Day 7 was the slowest for births" verified as
  `day_of_week` = 7 in a table whose `date_of_month` also holds 7; a bare "day
  N" that both day columns hold is now refused, naming both.
- "which countries have the most large airports" lists the first five
  countries; "which country has the most" still gives one.
- "magnitude" names `mag` rather than every column starting with mag, and
  "capacity" names `capacity_mw`; "magnitude 7 or more" bounds it; a column
  named for a time or period that holds only years is the year; "more than
  1,000 earthquakes of magnitude 4.5 or above" claims 1,000.
- Fixed a wrong answer: a cross-table of a table of counts counted its rows,
  not the counts they hold, so "how does breathlessness change with age" in
  CoalMiners found no association; it now adds up `Freq` and gives the share
  with the yes value in each age band (0.8% at 20-24, 42.1% at 60-64).
- Fixed a wrong reading: "Half of the miners aged 50 and over had a wheeze"
  was checked as the share of distinct Wheeze values among the 50-54 band. A
  column holding only a yes and a no (W and NoW) now keeps its yes after "had a",
  "50 and over" reaches the oldest band, "in their fifties" is 50 to 59, and a
  share of a two-valued column's distinct values is no longer given.
- "the Hot 100", "No. 1", "September 11" and "10 p.m." are not the number
  claimed.
- Fixed a wrong answer: "how many senators were over 65" counted every
  member of Congress; a plural formed from a value ("senators" for senate) is
  now refused and names the value.
- "papers with COVID in the title", "the first half of 2021", "the last
  quarter of 2019", "was under 40" and "65 or older" are read; "went from
  28,000 to 5.7 million, more than two hundred times as many" checks the
  change and both ends.
- Fixed wrong verdicts: "The number of fox rescues rose in 2020" took 2020 as
  the number claimed; "rose to 61 in 2020" failed and "fell below 40 in 2020"
  passed because the level was compared with the size of the change. A level
  after a change verb is now the value that year, checked with the change
  against the year before. A peak year in a table of counts compared the rows
  of each year rather than the counts they hold; it now makes the answer's own
  count for every year.
- Fixed a wrong answer: a change "in 2020" compared the last two months of
  2020, and groups ranked by their change "from 2019 to 2020" compared months
  too. Both now compare whole years.
- "Fox rescues nearly doubled in 2020, from 32 the year before to 61" counts
  the rescues and checks both ends; "2021-22" is one period; "polio cases in
  the United States" adds up the `count` column over all the states.
- Fixed a wrong answer: "which London boroughs have the most animal rescues"
  filtered to City of London and ranked a cost column; "London boroughs" now
  picks no rows, and "most" with no measure counts each group's rows.
- "In 2016 the Democrats topped 60% in nine states" counts the states above
  60%; "topped", "exceeded" and "surpassed" bound a value; "Just 48 women"
  is 48; "by 2016" is up to 2016, not in 2016.
- "rose by more than 60% between 1999 and 2017, to 47,173" and "halved in
  twenty years, from 51.5% in 1996 to 26.2% in 2016" check the change, then
  the values it went from and to.
- Fixed a wrong verdict: "the vote halved between 1996 and 2016" failed
  against a fall of 49%, as "halved" was not read as a fall when the sizes
  were compared. "Democratic vote" names `demVote` in a claim, as in a
  question.
- "In 2015, 565 boys were named Jack" in a table of names with a `Number` of
  babies each is checked against the rows and against the total of `Number`,
  with Sex = Boy and Name = Jack: it gets `ambiguous_intent` when only the
  total agrees, and fails when neither does.
- "how many homes were sold in Austin in 2014" adds up `sales` instead of
  counting the city's monthly rows; "a lake" names `LAKE` before `LAKES`; "new
  cases" no longer picks New Zealand.
- "The typical strike cost about $27,000" is checked against the median as
  well as the mean, and gets `ambiguous_intent` when only one agrees.
- Fixed a wrong verdict: "Of the strikes where the time of day was recorded,
  almost a third happened at night" was compared with the sum of the `time`
  column, as "recorded" read as a total; a percentage is never a total now.
- Fixed a wrong verdict: "Total rain at Oxford peaked at 984 mm in 2012"
  failed because the peak was checked across stations in 2012; a peak said in
  a year is now checked across the years.
- "Chorleywood had the best education score of any town, at 9.4" is checked
  as a peak, and any other claim that ranks a group against all the others is
  not verified on its value alone.
- A bound beside the claimed number picks rows: "Drivers under 25 accounted
  for 12% of all claims", "less than 1 km deep", "deeper than 50 cm" (with
  its unit converted to the column's), "on 25 or more days". "Of every 100
  birds, about 25 were put down" is a share.
- "dogs" in lower case is a kind (`animal_type`), not a pet's name; "birds"
  picks `animal_type` = bird; "were adopted" names "adoption"; "the", "other"
  and "any" are no values unless typed as the data spell them; words of the
  table's name found inside several columns' values name the table.
- Fixed wrong verdicts: "Oxford got 984 mm of rain in 2012" failed against
  the monthly average (82 mm) though the year's total agrees; a claim that
  names no way to combine rows is now checked both ways, and gets
  `ambiguous_intent` when only the other way agrees. "About 25% of the birds had outcome_type euthanasia"
  failed against the share of all animals. A count or share of 0, where no row fits the
  words as read, no longer fails a claim.
- "TCD won about 23% of all the money" is the share of the money column's
  total; "Men made up 64% of the 53,182 suicides" checks its base; verbs of
  showing ("featured", "showed", "included", "appeared in") keep a yes/no
  column's yes rows.
- Fixed silent drops: a word found only inside a column of titles filtered
  on one title or was dropped without a note; it now picks no rows unless the
  whole title is typed, and a category that holds the word too is meant.
- "The average price of a gallon fell by about 11%" is checked: a unit left
  out of a change in percent does not change it.
- "doubled", "more than sixfold", "tripled" and "halved" are changes in
  percent; "nearly half went on to college" is a fraction without "of"; and
  in a column of state names "NM" and "Calif." pick their states.
- Fixed wrong answers: "how many encounters in 2023" counted rows when each
  row holds a count (`encounter_count`), and "Mexican nationals accounted for
  29% of encounters" was the share of rows. A plural that names a column of
  whole counts now adds it up in counts, shares, groupings and trends.
  "Venezuelans" and "Mexican nationals" read as their countries.
- Rates: "Claims per 100 Holders" and "about 20 claims for every 100
  holders" are the total of one column over the total of the other, times
  the number they are per (`sum(Claims) ÷ sum(Holders) × 100`), with both
  totals in the answer. A code after "per" ("per District") and an average
  asked for keep their own readings.

- "6,657 metres high" reads a height, and "Customers paid R$16,008,872" a
  payment: an adjective after the unit and a verb before the number name the
  quantity (high, tall, long, wide, deep, heavy, old; paid, spent, earned,
  sold, charged, cost).
- A people noun as a claim's subject that names nothing in the table, and a
  capitalised word before the rows' own noun ("the average Lego set"), stand
  for every row, said in the assumptions; a word that is a value still
  filters.
- A number with a currency symbol picks the one money column among the
  columns the words fit.
- "Sets from 2019" means 2019's sets, while "from 2019 onwards" stays a
  range; "released in 2020" and "on board" no longer name nothing.
- Words a claim uses that name nothing in the table are left out, and the
  rest is checked: "Brazil lost 2,564,000 hectares of forest to pasture in
  2004". Such a check verifies only a lookup of one row, a total, a count or
  a slope; a mismatch, an average and a bound give `missing_context`, so a
  left-out word that picks rows can never give a wrong verdict.
  `words_left_out` in the JSON lists them.
- Slopes: "Reaction time rises by about 10.5 ms per day" is compared with
  the slope (`scipy.stats.linregress`, or the mean of each subject's own
  slope), and a fall with the slope reversed. Correlation answers carry
  `slope of Y on X` figures.
- Fractions and numbers in words ("about a quarter of", "more than half
  of", "Seventeen cases"), "with 4 cylinders" as a filter, `--value "about
  25%"` keeping its qualifier, feet, inches, miles, pounds and tonnes as
  units, and columns named for a length or a mass (`height`, `wgt`) for a
  number in that unit.
- A number that counts things ("1,524 Kenyans") is checked as a total, not an
  average. "73 HR" names the `HR` column, not hours, and digits inside a code
  ("bondsba01") are not a number.
- Fixed wrong verdicts: a count of rows pinned to one row by a key ("Nairobi
  had 5.2 million residents" on a table of counties) and a column guessed
  from part of a word ("seedlings" for `Seed`) are no longer compared with
  the claim; a bound is never read in thousands or millions.
- More fixed wrong verdicts, found on data ADA had not seen: "55 of the cats
  were neutered" counts the cats (the rows), not the mean of `n_cats`; "the
  fourth quarter of 2020" is a period, not 25%; squirrel IDs such as
  `37F-PM-1014-03` are no longer read as numbers in 42 units; and a claim
  about a column of medians (`median_weekly_earn`) that matches several rows
  gets `missing_context` instead of an average of medians.
- "At 14, girls averaged 24.1 mm" reads "At 14" as `age` = 14, the one column
  of few values that holds 14, and "the fourth quarter" as `quarter` = 4.
- "high satisfaction", "a high level of contact" and "highly satisfied" pick
  the one of several columns holding High that is named beside it; "Belgian"
  and "French" pick Belgium and France in a nationality column; "the 2010s"
  is 2010 to 2019; "won 20 times" counts rows; "their" and "its" are grammar.
- "Almost half of the squirrels were seen foraging" is the share of distinct
  squirrel IDs (a count of distinct values under a filter now gives its
  share), "seen" before a yes/no column keeps its yes rows, and "Owners
  estimated that" is set aside. Fixed wrong verdict: a percent claim read as
  a count with no share was compared with the count and failed; it now gets
  `missing_context`. Fixed wrong verdict: "Cars from USA averaged 18.6
  dollars" was read as a count of rows, because a plural after the number
  that names no column counts rows; an average or an amount of money no
  longer does.
- Claims with a second number that is part of the question are read: a day
  written out ("207 tornadoes on April 27, 2011", which also filters
  questions to that day), a rating scale ("6.5 out of 10") and a base,
  which is checked too ("34 of the 93 cars", "The 93 models came from 32
  manufacturers", "Of the 5,144 Danes in the study,"). "Seven in ten" and
  "1 in 20" are shares; "came from 32 manufacturers" counts distinct
  manufacturers; "reached", "surveyed" and "came from" are grammar.
- Fixed wrong verdict: "John McCain was a guest on the show 11 times" failed
  because the filter kept "John McCain" (5 rows) and not "Sen. John McCain"
  (6 more). A claim whose filter value, a name in a long list, the column
  also holds inside a longer value now gets `missing_context` naming the
  other spellings.
- Ages: "30- to 34-year-olds", "aged 50 and over", "people 65 and older" and
  "20-year-olds" keep those ages, on a column of ages or of age bands ("50-64"
  and "65+" make up "50 and over"); ages that cut a band in two are refused.
  "Among victims aged 50 and over, 9.5% died" is the share of those victims.
- Fixed wrong verdicts on shares of a group: "79% of the victims who died
  were men" and "4.95% of male victims died" were divided by every victim,
  and "Men were 79% of the victims who died" by the men. The base now ends
  at the sentence's own verb, a past tense with no verb before it is what is
  counted, and a share stated after what it counts is read the other way
  round.
- Words for dying and surviving ("deaths", "killed", "survivors") name the
  value or yes/no column that records the outcome, or the one column of
  numbers named for it ("deaths" is `fatal`).
- Fixed wrong verdict: a change "between 1982 and 1988" compared the last
  two years of the range (1987 and 1988); it now compares the two years
  named. A level's change is given as a percent too, "fell by about 19%" is
  compared with the size of the fall, and "rose", "risen" and "climbed" ask
  for a change.
- Peaks: "Cocaine use peaked at 4.9% among 20-year-olds", "California had
  more traffic deaths than any other state in 1988, with 5,390" and "had its
  busiest year on record in 2024, with 1,315 earthquakes" are verified only
  when the group named gives the highest value; a peak elsewhere fails and
  says where it is.
- Fixed wrong answer: "which day had the most earthquakes" ranked days by the
  average of a column the question never named; with no measure named,
  "most", "fewest", "busiest" and "quietest" count rows. "strongest",
  "deepest", "weakest" and "shallowest" ask for the highest or lowest value.
- "rose by about half" and "fell by a third" are changes of 50% and 33%, a
  change in "the number of" rows counts rows, two years named are compared as
  whole years, and the years of a range ("from 2023 to 2024") are never the
  number claimed.
- Words of the table's file or folder name ("Marvel's" in
  `marvel-characters/`) name the whole table and pick no rows; a base the
  claim spans ("across the 48 states") is checked against the distinct
  values of the column it names; US state names pick postal codes ("Texas"
  is `tx`); a plural every value shares ("Male Characters", "Good
  Characters") names the rows; and a column of genders that also holds
  values such as Agender or Non-binary reads "men" and "women".
- Fixed silent drop: in a table of counts every plural that named nothing
  was taken for the rows, so "Cyclists made up 15% of the people killed"
  ignored "Cyclists". Only the first one is now.
- Fixed wrong answers: a table of counts with one category and an outcome
  (age, result, Freq) was read as a long table and kept only its commonest
  outcome; and the word after a percent was taken as its unit, so "9.5%
  died" lost what it counts.
- The column that numbers the units (a `Rat` column of 16 rats) is never
  averaged or ranked, and the word for the units names the rows when the
  question names a measure too: "The heaviest rat reached 628 grams" is the
  largest weight, where before it was the largest rat number.
- "The distance increased by about 0.66 mm per year of age" is the slope of
  distance on age: a column named for a quantity (age, dose, time) at a few
  levels still correlates, and with several rows per subject both slopes of
  the subjects' own lines are given.

### Answers

- A grouping split at its quartiles with one row in scope ("average infant
  mortality by life expectancy for Japan") says the rows cannot be split
  into groups. Before, the empty grouping raised an IndexError.
- A share of a group is of that group: "what percentage of Adelie are
  female" gives 50% of the Adelie, not 21.9% of all penguins, and "what share
  of women survived" 73.2% of the women. Before, the base was every row,
  which made a claim such as "50% of Adelie penguins are female" fail.
- "Men", "women", "boys" and "girls" name the values of a column of sexes,
  "department" names `Dept` and "weight" `wgt`, and in a table of counts a
  plural that names nothing else ("applicants") names the cases.
- A claim's lead that names its base ("Of the 4,526 applicants,") is checked
  against the share's base, "at a rate of" is grammar, "1,835 women" counts
  women, a plural after the number that names no column counts rows ("1,950
  captures in 2007"), and "rose to 20.9 tonnes per hectare" is a level, not a
  slope.
- Tables of counts weight shares and counts by group too: Titanic's "share of
  Survived = Yes by Class" gives 62.5% for 1st class (203 of 325), not 50% of
  rows, and its chi-squared test runs on the weighted table.
- "how many trees" is refused, not answered with the row count, when each
  tree has several rows; "how many reviews in 2019" counts rows, not
  distinct years; "how many Gold medals were won" reads "were won" as
  grammar.
- A column named as a date that holds only years (`review_date`) scopes "in
  2019", a code in a column too large to index ("bondsba01") filters, and "on
  day 21" picks the time-point column's value.
- A clock time read onto one day ("1:27.452") is no longer the table's date,
  and joins leave out birth and death dates, so "in 2019" in a join of F1
  results, drivers and races is the race year: "Lewis Hamilton scored 413
  points in 2019" checks across the three files.

### Decisions

- Answers and claim checks list the choices they rest on: `decisions` in the
  JSON and a `decides:` line in the text, each with the question, the
  options, ADA's default, the option used and why the default is the
  default. `Analysis.ask(question, decisions=...)`, `ada.check_claim(...,
  decisions=...)` and `--decide KEY=OPTION` on `ada ask` and `ada claim` pick
  another option; an unknown key or option raises `UnknownDecision` (exit
  code 2). A person, or an agent that knows the person's goal, decides; with
  no one to ask, the defaults give the same answer every time. See
  [Decisions](docs/reference/decisions.md).

### Questions first, and answers across files

- `ada scan` asks what the analysis should find out first, and asks about the
  decisions too: for each file with World or region rows, and for joining
  files. It also scans a single file (`ada scan data.csv`). The answers file
  `--template` writes is read by `ada ask` and `ada claim` with `--answers`
  (`answers=` in Python): `decide:KEY` for every file, `decide:FILE:KEY` for
  one, and `--decide` over both.
- `ada ask` takes a folder (`ada.ask_folder` in Python), and a question or
  claim about a folder that no single file answers is tried on two files
  joined along a link: "Barry Bonds hit 73 home runs in 2001." is verified on
  `Batting.csv` joined to `People.csv`. Only many-to-one and one-to-one links
  are joined, by default only those that need no confirmation (`join_links`:
  `sure`, `all`, `none`), and the answer says which join it used.
- A person named by first and last name, in a column each, is found even
  when there are more names than the value index keeps.
- After a verb that adds things up (hit, scored, earned, collected,
  recorded), the words after the number name the column that is totalled:
  "scored 3,999.5 points", "hit 73 home runs". "home runs" reads `HR` by its
  initials wherever a claim names a column.

### Rows that add up others

- World, region and Total rows listed beside the rows they add up (OWID's
  World and Africa among countries, a spreadsheet's Total) are found by their
  labels and left out of averages, rankings, counts and comparisons, and the
  answer says which (`aggregate_rows`, default `leave_out`). A total over the
  scope is read from the World or Total row, with what the other rows add up
  to beside it (`total_from`, default `total_row`). "total co2 in 2022" on
  the OWID file gives 37,527.773, the World row, instead of 240,370.227.
- A claim that agrees only when one of these decisions goes the other way is
  `ambiguous_intent`, with both readings and the question to settle it.

### Fixes

- A year in a question scopes a numeric column named for years when the table
  has no date column (OWID's `year`, from 1750), and a year before 1900 is
  read when that column holds it.
- An aggregate over one row says "calculated from 1 row".

## 0.6.0 (2026-10-27)

ADA now checks numbers as well as computing them. `ada claim` says whether a
number stated about a table is verified or unverified, and why; `ada ask
--json` gives an answer's numbers and rows as data; `ada scan` maps a folder
of related files and the keys that join them. Answers count each subject once
when several rows come from one, and say when groups reverse a pooled
correlation. Evaluated on 73 cases from 20 public datasets
([results/verification](results/verification/README.md)).

```bash
pip install --upgrade automated-data-analyst
ada skill      # the updated skill: check every number with ada claim
```

### The skill

- The skill tells an assistant to compute with `ada ask`, check every number it
  states with `ada claim`, say which numbers are verified (with their
  calculation) and which are not and why, ask ADA's clarifying question
  instead of picking a reading, and run `ada scan` before combining files.
  Run `ada skill` again, or `ada skill --check`, after upgrading.

### Reading a folder of tables

- `ada.scan(folder)` and `ada scan FOLDER` profile every table in a folder,
  propose the keys that join them from their values (with the coverage both
  ways, the cardinality, the rows a join leaves unmatched and how many rows
  it repeats), find composite keys such as `(yearID, teamID)`, read a
  codebook's units, and list what a person should confirm, grouped by topic:
  what a row stands for, units, repeated measurements, links and the
  question to answer. `--template FILE` writes the questions with ADA's
  proposed answers; `--answers FILE` reads the confirmed answers back.
  `--json` prints all of it. See [Reading a folder of tables](docs/reference/folder.md).
  Whole numbers link only under matching names, since ids, positions and
  counters overlap by chance, and a link where both sides repeat needs a key
  word (`id`, `code`, `zip`, `number`) in its name. On the Formula 1 tables
  this leaves the 7 links the schema documents and one to confirm. Two bare
  `id` columns, each its own file's row id, link only when each holds 90% of
  the other's values.
- `ada.check_claim` and `ada claim` take a folder and check the claim against
  the file it fits; when no single file does, the answer names the keys that
  would join two of them.

### Checking numbers against the data

- `ada.check_claim(data, claim)` and `ada claim FILE "claim"` check a number
  someone states about a table, such as "The average departure delay was
  12.64 minutes". The result is `verified`, or `unverified` with one of six
  reasons: `missing_context`, `ambiguous_intent`, `inaccessible_data`,
  `unsupported_analysis`, `conflicting_evidence` or `failed_check`. Each check
  gives the interval the claim's number stands for ("12.64" is 12.635 to
  12.645, "about 13" is 12 to 14), the calculation, the rows used and left
  out, the file's sha256, its assumptions and warnings, and, when the claim
  can't be checked as stated, the one question that would let it go ahead.
  `--json` prints all of it; the exit code is 0 verified, 1 failed check, 2
  unreadable file, 3 unverified for another reason.
  See [Checking claims](docs/reference/claims.md).
- `ada ask FILE "question" --json` and `QueryAnswer.to_dict()` give an
  answer as data: the full-precision `value`, the `values` per group, other
  `figures` (counts, shares and their denominators, p-values, intervals), how
  the question was read, and the rows read, used and left out with a reason
  for each.
- An answer names its source: `Analysis.source` and `QueryAnswer.source`
  hold the file's name, path, sha256, rows and columns.
- A question can name the rows by the file's name: "how many flights" in
  flights.csv counts rows instead of distinct values of a `flight` column.

### Fixes

- A table of counts is counted by its counts. When each row is one
  combination of categories with a column such as `Freq` holding how many
  cases it stands for, a count or share adds up that column, and the answer
  says so: Titanic's "what share survived" is 711 of 2,201 people (32.3%),
  not half of its 32 rows. A share of a Yes/No column counts its Yes rows
  ("what share survived" was answered with the number of distinct values),
  and a percent claim that names no column ("32.3% survived") is checked as
  a share.
- Ordinals are labels. Titanic's `Class` ("1st", "2nd", "3rd", "Crew") was
  read as numbers in units called "st", "nd" and "rd", with "Crew" missing; it
  stays a column of classes.
- A mismatch is a failed check only when the claim names what the column
  measures. "Expeditions recorded 788 member deaths" read "deaths" as
  `death_height_metres`, summed the heights at which climbers died (11,050)
  and called the claim false; such a mismatch is now `missing_context`, with
  the question "Is the number about death_height_metres, or about another
  column?".
- The file's own noun for its rows names the rows, not a value: "Expeditions"
  in expeditions.csv no longer matches trekking agencies named "...
  Expeditions", and "what share of members had success" in members.csv counts
  all 76,519 members (38.2%), not the 9 whose role is "Member", which a
  question still reaches by typing "Member". In a folder, the file whose column the claim names by
  what it measures is chosen over one read from a lesser word, so the claim
  above is verified against `expeditions.csv`'s `member_deaths` (788).
- One repeated id is not repeated measurement: the warning that an average
  weights each unit by its rows needs 1.5 rows per unit on average.
- Words that a column's name shortens word by word name it: "life
  expectancy" names Gapminder's `lifeExp`, and the answer says so ('Read
  "life expectancy" as lifeExp'). The question was refused.
- A camelCase column name reads as its words, and a claim's unit can hold a
  slash. "The average fastest lap speed is 202.6 km/h" about Formula 1's
  results was refused ("lap" named `laps`, and "/h" named nothing); it is now
  verified against `fastestLapSpeed` (202.59 over 6,777 rows), with the note
  that only the numbers are compared.
- A table out of date order is no longer split into logs. Formula 1's races,
  listed season by season from 2009 back to 1950, were read as 8 logs one
  after another and given a Block column; a file with more steps back than 24
  and 1% of its rows is now left as it is.
- A year in a question is the year even when it also appears inside a longer
  text value. "how many races in 2020" on Formula 1's races table was refused
  because 2020 is part of 17 race web addresses; it now counts the 17 races
  dated 2020.
- A correlation over rows that come several to a subject counts each subject
  once. "does Reaction increase with Days" on the sleepstudy table (18
  people, 10 days each) used to be refused; it now gives the average of the
  18 people's own slopes, 10.47 per day (95% CI 7.21 to 13.73, the same
  standard error as a random-slope mixed model), with the repeated-measures
  r of 0.72. A column fixed within each subject is correlated over the
  subjects' means.
- A correlation answer says when the relation inside the groups of a
  grouping differs from the pooled one, as the brief does: over all penguins
  bill length and depth fall together (r = -0.24), within each species they
  rise together (0.39 to 0.65). The sentence also reaches a claim check's
  warnings.
- "average order value" and "average payment value per order" total each
  order first and average over orders (160.99 on the Olist payments table),
  and give the row-by-row average beside it (154.10). A claim that matches
  the row average is told so.
- Elapsed time counted within each subject (`Days`) is a time axis, not a
  running total to set aside.

- A comparison counts each subject once when the subject column is numbered
  with a sign (`subject#`, `Patient #`): `#` is now a word of its own, so such
  a column is read as the unit. On the UCI Parkinson's telemonitoring table,
  "does Jitter(Abs) differ by sex" compared 5,875 recordings as if each were a
  separate person (p = 2e-151); it now compares the 42 people (Welch's p =
  0.14 on their medians) and says the two tests disagree.
- `.data` and `.dat` files are read, as tab-separated text when the header
  splits on tabs and as comma-separated text otherwise (the UCI repository's
  `.data` files).
- Lines above the header are skipped: a data portal's "Data Source" and "Last
  Updated Date" lines as well as a report's title. A World Bank indicator file
  failed to read.
- `\N`, the null MySQL and PostgreSQL write when they export a table, is read
  as missing in every column. Formula 1's `fastestLap` and `fastestLapSpeed`,
  three quarters `\N`, stayed text.
- `NA` in a text column of words (`male`, `female`, `NA`) is read as missing,
  with a note; among codes (`EU`, `APAC`, `NA`) it stays a code. Penguins'
  `sex` had an "NA" group of 11.
- A year column shared by many rows (three years across a table sorted by
  species) no longer splits the file into "logs" with a made-up `Block`
  column.
- A comparison's rows used leave out rows with no group: "compare body mass
  between sex" rests on 333 rows, not 342.
- The data checks no longer call a share's 100 (the World row's share of world
  emissions) a cap or a missing-value code, and no longer pair a share, a rate
  or a cumulative total with the total it shares a word with as part and
  whole (`share_global_co2` beside `co2`).
- Two to four words whose initials spell a column in capitals name it, and the
  answer says so: "home runs" reads as `HR`.
- Reading a table with hashed identifiers no longer crashes Python. pandas
  2.3 crashes on text such as `689e8820798347950a2649f71c734bd0`, which it
  reads as a number with an exponent past its limit; ADA 0.5.0 crashed on the
  public Olist payments table this way. Such text is now set to missing
  before pandas reads it as a number, which gives the same result without the
  crash.

- The plugin is now its own folder, `plugins/automated-data-analyst/`, named
  `automated-data-analyst`, with manifests for the Claude and ChatGPT plugin
  directories, an icon, a README and the license. It holds the same skill as
  `ada skill` and no code. Install it with
  `/plugin marketplace add saineshnakra/automated-data-analyst` then
  `/plugin install automated-data-analyst@automated-data-analyst` in Claude
  Code, or `codex plugin add automated-data-analyst@automated-data-analyst`
  in Codex. An install of `ada@automated-data-analyst` moves to the new
  name; run the install command once to fetch it.

## 0.5.0 (2026-10-23)

ADA now works as a skill in Claude Code and Codex, and the study workflow is
safer on real research data: an outside evaluation of 0.4.0 found seven
defects on public datasets, and every one is fixed.

```bash
pip install --upgrade automated-data-analyst
ada skill      # the skill, for Claude Code and Codex in this project
```

### A data-analysis skill for Claude Code and Codex

- `ada skill` installs ADA's skill where Claude Code (`.claude/skills/ada`) and
  Codex (`.agents/skills/ada`) look for skills. `--for claude` or
  `--for codex` picks one, `--user` writes to your home folder for every
  project, and `--dir` to any other tool's skills folder.
- It also installs as a plugin. In Claude Code:
  `/plugin install ada --marketplace saineshnakra/automated-data-analyst`.
  In Codex: `codex plugin marketplace add saineshnakra/automated-data-analyst`,
  then `codex plugin add ada@automated-data-analyst`.
- The skill tells the assistant to ask you for what only you know (what a row
  is, the subject column, the design, the planned analyses), to let ADA do
  the arithmetic, to report ADA's numbers with the calculation behind each,
  and to pass a refusal on instead of working around it.
- `llms.txt` at the root of the repository gives a language model a short
  map of the docs.

### Study workflow: `ada init`, `ada build`, `ada verify`

Found by an outside evaluation on UCI Wine, Wine Quality, Parkinsons
Telemonitoring and Beijing air-quality data, and fixed:

- **Repeated subjects.** `ada init` proposes a column of repeated numbered
  codes (`subject#`, `Patient ID`) as the subject of a repeated design and
  asks you to confirm it. A guessed design is written as
  `design.confirmed: false`, and the methods draft says it was not confirmed.
- **Mixed models that can't be trusted are refused**, with the reason: fewer
  than 2 subjects or clusters, a subject alone in its group, a fit that did
  not converge, or a singular fit.
- **Rows that form a series.** An independent design with a time column gets
  a Durbin–Watson check in time order, and a warning when successive rows
  stay alike, since its p-values are then too small.
- **Small numbers never print as zero** or lose their sign: `−1.20 × 10⁻⁵`,
  not `0.0000`.
- **Proposed analyses are checked** with the build's own rules before
  `study.yml` is written; groups coded as numbers are written as
  categorical.
- **`ada build` counts** the analyses requested, fitted and refused, with the
  reason for each refusal. `ada build --strict` exits with 1 when any was
  refused.
- **Failed builds keep the last good assets.** A build works in a scratch
  folder and replaces the old files only once every step has succeeded.
- **Scatter figures** draw the linked analysis's own fitted line and band,
  with covariates held at stated values, and their source data carry the
  table's slope and correctly paired confidence limits.
- **Hourly data** keeps its timestamps: year, month, day, hour, minute and
  second columns combine to the finest grain given.
- Smaller fixes: the "on the boundary" caveat is judged against the residual
  variance, captions say "1 subject", and a bracketed word becomes a unit
  only when it is a known unit.

### Web app

- The app opens on a page for researchers, with an example study that ADA
  builds on the server and you can download, and the first look on its own
  page (`/first-look`).
- Starlette is kept below 1.0, since Streamlit 1.59 cannot serve images or
  downloads on Starlette 1.x.

### Upgrading from 0.4.0

- A `study.yml` that names `year-month` as a variable stops with "no such
  column" when the file also has day or hour columns, because the time is
  now built at the finer grain. Run `ada init` again, or rename the variable.
  Your previous assets are kept.
- A mixed model that 0.4.0 reported although its fit did not converge or was
  singular is now refused. Rebuild, and `CHANGES.md` lists what changed.
- Tables print very small values in scientific notation, so rebuilding a
  study with such values changes those files.
- A `study.yml` without `design.confirmed` counts as confirmed, as before.

## 0.4.0 (2026-10-22)

- **Paper assets for a study.** `ada init` writes a `study.yml` from the data
  and your answers; `ada build` makes a data audit, a data dictionary, a
  processing record, tables in Word, LaTeX and CSV, figures sized for
  journals with their source data and captions, a methods draft, a
  supplement, a reproduction package and repository metadata, each traced in
  `provenance.json`; `ada verify` rebuilds in a scratch folder and checks
  every number. `ada preview`, `ada share` and `ada demo` handle sharing and
  an example study.
- **The first look**, tested on 60 research datasets in three rounds, with
  the fixes from each round and the results in `results/`.

## Earlier versions

See the [releases on GitHub](https://github.com/saineshnakra/automated-data-analyst/releases).
