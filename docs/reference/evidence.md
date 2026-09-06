# Evidence and recommendations

**Code:** `business_insights.py` → `analyze_business`
**Tests:** `tests/test_business_insights.py`, `tests/test_significance.py`

Turns the calculated frames into what the user actually reads: a headline, four
KPIs, evidence cards, and prioritized next steps.

## The split that matters

| | Evidence | Recommendation |
|---|---|---|
| What it is | Something ADA calculated | Something ADA suggests |
| Claim strength | Fact about your data | Interpretation, never proof of cause |
| Always carries | The exact calculation | The evidence it rests on |
| Type | `Evidence` | `Recommendation` |

Nothing in ADA says "X caused Y". Evidence says what moved; recommendations say
where to look.

## The output: `BusinessBrief`

| Field | Contents |
|---|---|
| `headline` | One sentence, built from the strongest available signals |
| `summary` | A short plain-English paragraph |
| `roles` | The `ColumnRoles` used |
| `kpis` | Exactly 4 `KPI`s |
| `evidence` | Up to 6 `Evidence` cards |
| `recommendations` | Prioritized `Recommendation`s |

## The evidence kinds

Each is produced by its own function and skipped when the data does not support
it. Order below is the order they are assembled.

| Kind | What it reports | Skipped when |
|---|---|---|
| `trend` | Latest period-over-period movement, in context of the series' own volatility | No date or measure |
| `driver` | Which segment moved the measure most, and whether gross movement offsets | No segment |
| `leader` | Largest segment and its share | No segment |
| `concentration` | How concentrated the measure is across segments | Total is zero, or the segment values are mixed-sign (a share of a signed total is not a share) |
| `leader` | *(mixed signs)* Largest segment by size, with no share quoted | — |
| `anomaly` | The most severe flagged period | Under 8 periods, or nothing flagged |
| `relationship` | Strongest numeric correlation, with a confidence interval | See thresholds below |
| `outliers` | Share of extreme measure values | No measure |
| `quality` | Data completeness | Nothing missing worth reporting |

### Movement in context

A metric that routinely swings 30% has not told you anything by swinging 30%
again. So the `trend` card compares the latest change against the series' own
typical movement, and says which it is. Needs at least **5 periods**
(`MIN_PERIODS_FOR_VOLATILITY`) to have a sense of typical.

### Offsetting movement

When gross movement across segments is much larger than the net change — the
threshold is a net that is under **25%** of gross (`OFFSETTING_NET_RATIO`) — the
`driver` card says so. "Up 2% overall" hides "one region up 40%, another down
38%", and those are very different situations.

### Concentration

Measured as **effective segments** rather than a raw top-N share, so it does not
depend on how many groups happen to exist. Below **3.0** effective segments the
measure is called concentrated. A top-three share of **70%** or more is toned as
a warning.

### Correlation, reported honestly

The strongest numeric pair is reported only when all of these hold:

| Requirement | Value |
|---|---|
| Absolute correlation | ≥ **0.45** (`MATERIAL_CORRELATION`) |
| Complete pairs | ≥ **12** (`MIN_CORRELATION_PAIRS`) |
| Columns | Not identifiers, not time parts |

It ships with a 95% confidence interval via the Fisher transform and the sample
size. If the interval spans zero, the card says the sample cannot rule out no
relationship at all.

There is also a rank check: when Pearson and rank correlation diverge by more
than **0.25** (`RANK_DIVERGENCE`), the relationship is likely driven by a few
outliers, and the card says so.

This is the difference between "correlation 0.62" and something a reader can
weigh.

## The four KPIs

Filled in this order until there are four:

1. Total measure
2. Average measure
3. Latest movement (if a trend card exists)
4. Top segment share (if a leader card exists)

If fewer than four are available, ADA falls back to distinct identifiers, then
records analyzed, then data completeness. There are always exactly four, so the
layout never breaks.

## Recommendations

`_recommendations` maps evidence to actions with a priority. Each one names the
calculation it came from. A recommendation with no supporting evidence is not
produced — that is the whole design.

## The Markdown report

`build_business_report` renders the brief as a downloadable Markdown executive
brief, including the calculation behind every card.

## Changing this

A new evidence card needs:

1. A function returning `Evidence | None` — `None` when the data cannot support it
2. A `calculation` string a reader could reproduce by hand
3. A test for the happy path **and** the skip path
4. Honest degradation: no history, one segment, all zeros, negative values

If a heuristic can produce a misleading result, add the edge case to the tests
and state the limitation in the card's own text.
