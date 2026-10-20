"""Compare a reading across groups, and find what relates to it.

A gap between two group averages says little without the spread and the sample
behind it, so every comparison here carries the group sizes, a test and an
effect size, and a gap within the noise reads as no clear difference rather
than as a ranking. The relationship helpers screen every pair of numeric
columns at once and set aside pairs that are related by construction: a total
and its parts, a flag marking where a reading hit its cap, one quantity
recorded twice.
"""

from __future__ import annotations

import itertools
import re
import warnings
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from .formatting import format_number, format_values_apart, group_label, list_names, normalized_name
from .metrics import metric_for
from .schema import ColumnRoles, is_entity_name

ALPHA = 0.05
# A group needs two values to have a spread to test against.
MIN_GROUP_SIZE = 2
# Within-group skewness from which medians and ranks describe a reading better than means.
SKEWED = 2.0
# Skewness from fewer values than this is too unstable to act on.
SKEW_MIN_VALUES = 20
# A reading is capped when this share of its values (and at least CAP_MIN_ROWS)
# sit exactly at its maximum or minimum: a timer stopped at 3600 s, an assay
# ceiling. Scales with few distinct values (a 1-7 rating) are not called capped.
CAP_SHARE = 0.02
CAP_MIN_ROWS = 3
CAP_MIN_DISTINCT = 10
# Below this share of variance explained a difference is too small to matter,
# however small its p-value (Cohen's "small" is 0.01).
NEGLIGIBLE_EXPLAINED = 0.01
HEDGES_SIZES = ((0.8, "large"), (0.5, "medium"), (0.2, "small"))
EXPLAINED_SIZES = ((0.14, "large"), (0.06, "medium"), (0.01, "small"))
# The scan of every reading compares groupings with up to this many levels.
MAX_SCAN_LEVELS = 12
# Wider tables are scanned on their most variable columns, and longer ones on a sample of rows.
MAX_SCAN_COLUMNS = 300
SCAN_ROWS = 50_000

MIN_CORRELATION_PAIRS = 12
NEAR_DUPLICATE_CORRELATION = 0.99
RANK_DIVERGENCE = 0.25
# Columns sharing a name stem ("MaxTemp" and "MeanTemp") and correlating above
# this are one measurement in two forms.
SHARED_STEM_CORRELATION = 0.95
# Below this neither Pearson nor Spearman r is reported as a relationship.
NOTABLE_CORRELATION = 0.3
# Dropping one record moving r by more than this means the relationship rests on it.
INFLUENCE_CHANGE = 0.2
# Robust z-score beyond which a value counts as extreme when checking a correlation.
EXTREME_Z = 5.0
# A column is derived when a fixed combination of other columns reproduces it to
# within 1% of its spread on 90% of rows; the 10% allows for rows entered by hand.
DERIVED_TOLERANCE = 0.01
DERIVED_SHARE = 0.9
# Rounding allowance: half the column's own step, but never more than this share of its spread.
DERIVED_STEP_LIMIT = 0.1
DERIVED_EXPLAINED = 0.99
MAX_PARTS = 8
PART_CANDIDATES = 15
DERIVATION_ROWS = 2_000
# Rows that must hold a column and all its candidate parts: this many, and this share of the column.
DERIVATION_MIN_ROWS = 12
DERIVATION_MIN_SHARE = 0.1
# Words that qualify a name rather than say what was measured, so sharing one
# ("CorrectTotal", "IncorrectTotal") does not make two columns one quantity.
GENERIC_NAME_WORDS = frozenset({
    "total", "sum", "mean", "average", "avg", "max", "maximum", "min", "minimum", "median",
    "std", "sd", "count", "number", "rate", "ratio", "percent", "pct", "value", "score", "index",
    "level", "change", "diff", "difference", "delta", "new", "old", "raw", "adjusted", "adj",
    "extended", "net", "gross", "first", "second", "third", "final", "initial", "pre", "post",
    "time", "day", "date", "the", "and", "per", "of",
})
# Words naming a column as built from others, which may then come before its parts.
AGGREGATE_NAME_WORDS = frozenset({
    "total", "sum", "mean", "average", "avg", "overall", "combined", "composite", "aggregate",
})


# --- Group comparisons --------------------------------------------------------

# A pooled standard deviation below this share of the means counts as no spread at all.
NO_SPREAD = 1e-9


@dataclass(frozen=True)
class GroupStats:
    """One group's values of the reading."""

    value: object
    label: str
    n: int
    mean: float
    median: float
    sd: float


@dataclass(frozen=True)
class GroupComparison:
    """A reading compared across the groups of one column.

    ``groups`` runs from the highest centre to the lowest: the mean, or the
    median when ``uses_ranks`` is set because the reading is skewed or capped.
    The first and last groups are the two compared for two groups, and the pair
    that differs most otherwise; ``difference``, ``hedges_g`` and
    ``superiority`` describe that pair.
    """

    measure: str
    grouping: str
    groups: tuple[GroupStats, ...]
    uses_ranks: bool
    #: Why ranks are used: "it is strongly skewed", "13 values sit at its maximum of 3,600".
    shape: str
    p_value: float
    check_p_value: float
    #: eta squared from the means, or epsilon squared from the ranks.
    explained: float
    difference: float
    #: Welch 95% interval for the difference in means of the pair.
    low: float
    high: float
    hedges_g: float
    #: Share of cross-pairs in which the higher group's value is the larger; ties count half.
    superiority: float
    entity: str | None = None
    #: How rows were reduced to one per entity: "mean" or "median".
    entity_summary: str = "mean"
    #: An entity appears in more than one group, so the groups are not independent.
    shared_entities: bool = False
    dropped: tuple[str, ...] = ()
    adjusted_p: float | None = None
    tests: int = 1

    @property
    def higher(self) -> GroupStats:
        return self.groups[0]

    @property
    def lower(self) -> GroupStats:
        return self.groups[-1]

    @property
    def two_groups(self) -> bool:
        return len(self.groups) == 2

    @property
    def test(self) -> str:
        if self.two_groups:
            return "Mann-Whitney" if self.uses_ranks else "Welch's t-test"
        return "Kruskal-Wallis" if self.uses_ranks else "Welch ANOVA"

    @property
    def check(self) -> str:
        if self.two_groups:
            return "Welch's t-test" if self.uses_ranks else "Mann-Whitney"
        return "Welch ANOVA" if self.uses_ranks else "Kruskal-Wallis"

    @property
    def explained_name(self) -> str:
        return "ε²" if self.uses_ranks else "η²"

    @property
    def decisive_p(self) -> float:
        return self.p_value if self.adjusted_p is None else self.adjusted_p

    @property
    def verdict(self) -> str:
        """"clear", "negligible" (a passing test on a trivial gap) or "none"."""
        if not self.decisive_p < ALPHA:
            return "none"
        if self.explained < NEGLIGIBLE_EXPLAINED:
            return "negligible"
        return "clear"

    def centre(self, group: GroupStats) -> float:
        return group.median if self.uses_ranks else group.mean


@dataclass(frozen=True)
class _Omnibus:
    """Group statistics and both omnibus tests for every column of a block."""

    levels: list
    n: np.ndarray
    mean: np.ndarray
    median: np.ndarray
    sd: np.ndarray
    mean_rank: np.ndarray
    welch_p: np.ndarray
    rank_p: np.ndarray
    eta2: np.ndarray
    epsilon2: np.ndarray
    skew: np.ndarray
    at_max: np.ndarray
    at_min: np.ndarray
    maximum: np.ndarray
    minimum: np.ndarray
    distinct: np.ndarray


@dataclass(frozen=True)
class _Ranked:
    """Average ranks of each column (missing values stay missing), from one sort per column."""

    ranks: np.ndarray
    #: Sum of t³ − t over each column's runs of tied values, for the tie correction.
    ties: np.ndarray
    distinct: np.ndarray
    at_max: np.ndarray
    at_min: np.ndarray


def _rank_columns(values: np.ndarray) -> _Ranked:
    ranks = np.full(values.shape, np.nan)
    width = values.shape[1]
    ties, distinct = np.zeros(width), np.zeros(width, dtype=int)
    at_max, at_min = np.zeros(width, dtype=int), np.zeros(width, dtype=int)
    for column in range(width):
        present = np.flatnonzero(~np.isnan(values[:, column]))
        if not len(present):
            continue
        order = present[np.argsort(values[present, column])]
        ordered = values[order, column]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        runs = np.diff(np.r_[starts, len(ordered)])
        ranks[order, column] = np.repeat(starts + (runs + 1) / 2, runs)
        ties[column] = float((runs.astype(float) ** 3 - runs).sum())
        distinct[column], at_min[column], at_max[column] = len(runs), runs[0], runs[-1]
    return _Ranked(ranks, ties, distinct, at_max, at_min)


def _omnibus(block: pd.DataFrame, groups: pd.Series) -> _Omnibus:
    """Compare every column of ``block`` across ``groups`` in one pass.

    Every group must hold at least two values of every column. Welch's ANOVA is
    the test on means (with two groups it equals Welch's t-test), Kruskal-Wallis
    the test on ranks (with two groups the Mann-Whitney test, by the normal
    approximation with ties corrected). Group sums come from products with a
    one-hot group matrix, so the cost grows with rows times columns.
    """
    codes, levels = pd.factorize(groups, sort=True)
    k = len(levels)
    values = block.to_numpy(dtype=float)
    present = ~np.isnan(values)
    filled = np.where(present, values, 0.0)
    member = np.zeros((len(codes), k))
    member[np.arange(len(codes)), codes] = 1.0
    n = member.T @ present
    total = n.sum(axis=0)
    ranked = _rank_columns(values)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = member.T @ filled / n
        deviation = np.where(present, values - mean[codes], 0.0)
        variance = member.T @ deviation**2 / (n - 1)
        grand = filled.sum(axis=0) / total
        ss_total = (np.where(present, values - grand, 0.0) ** 2).sum(axis=0)
        eta2 = (n * (mean - grand) ** 2).sum(axis=0) / ss_total
        # A group with no spread would get infinite weight, so variances are
        # floored far below anything measurable; equal constant groups stay equal.
        floor = np.maximum(ss_total / np.maximum(total - 1, 1) * 1e-12, np.finfo(float).tiny)
        weights = n / np.maximum(variance, floor)
        weight_sum = weights.sum(axis=0)
        weighted_mean = (weights * mean).sum(axis=0) / weight_sum
        between = (weights * (mean - weighted_mean) ** 2).sum(axis=0) / (k - 1)
        spread = (((1 - weights / weight_sum) ** 2) / (n - 1)).sum(axis=0)
        welch_f = between / (1 + 2 * (k - 2) / (k**2 - 1) * spread)
        welch_p = stats.f.sf(welch_f, k - 1, (k**2 - 1) / (3 * spread))

        rank_sum = member.T @ np.where(present, ranked.ranks, 0.0)
        h = (12 / (total * (total + 1)) * (rank_sum**2 / n).sum(axis=0) - 3 * (total + 1)) / (
            1 - ranked.ties / (total**3 - total)
        )
        rank_p = stats.chi2.sf(h, k - 1)
        epsilon2 = h / (total - 1)
        # Adjusted Fisher-Pearson skewness of the values around their own group's mean.
        second = (deviation**2).sum(axis=0) / total
        third = (deviation**3).sum(axis=0) / total
        skew = third / second**1.5 * np.sqrt(total * (total - 1)) / (total - 2)

    median = block.groupby(codes, sort=True).median().to_numpy(dtype=float)
    return _Omnibus(
        levels=list(levels),
        n=n,
        mean=mean,
        median=median,
        sd=np.sqrt(variance),
        mean_rank=rank_sum / n,
        welch_p=np.nan_to_num(welch_p, nan=1.0),
        rank_p=np.nan_to_num(rank_p, nan=1.0),
        eta2=np.nan_to_num(eta2, nan=0.0),
        epsilon2=np.clip(np.nan_to_num(epsilon2, nan=0.0), 0.0, 1.0),
        skew=np.nan_to_num(skew, nan=0.0),
        at_max=ranked.at_max,
        at_min=ranked.at_min,
        maximum=np.nanmax(values, axis=0),
        minimum=np.nanmin(values, axis=0),
        distinct=ranked.distinct,
    )


def _shape(omnibus: _Omnibus, column: int, measure: str) -> str:
    """Why medians and ranks suit this reading better than means, or "" when they don't."""
    present = int(omnibus.n[:, column].sum())
    piled = max(int(omnibus.at_max[column]), int(omnibus.at_min[column]))
    if omnibus.distinct[column] >= CAP_MIN_DISTINCT and piled >= max(CAP_MIN_ROWS, CAP_SHARE * present):
        top = omnibus.at_max[column] >= omnibus.at_min[column]
        edge = omnibus.maximum[column] if top else omnibus.minimum[column]
        return (
            f"{piled:,} values sit at its {'maximum' if top else 'minimum'} of "
            f"{format_number(float(edge), measure, compact=False)}"
        )
    if present >= SKEW_MIN_VALUES and abs(omnibus.skew[column]) >= SKEWED:
        return "it is strongly skewed"
    return ""


def _robust_on_rows(values: pd.Series) -> bool:
    """Whether a reading is skewed or capped across its rows, so a median summarises it best."""
    present = values.dropna()
    if len(present) < 3:
        return False
    piled = max(int((present == present.max()).sum()), int((present == present.min()).sum()))
    if present.nunique() >= CAP_MIN_DISTINCT and piled >= max(CAP_MIN_ROWS, CAP_SHARE * len(present)):
        return True
    skew = present.skew()
    return bool(len(present) >= SKEW_MIN_VALUES and np.isfinite(skew) and abs(skew) >= SKEWED)


def _pair(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float, float, float]:
    """Difference in means with its Welch 95% interval, Hedges' g, and the share of pairs ``first`` wins."""
    n1, n2 = len(first), len(second)
    difference = float(first.mean() - second.mean())
    var1, var2 = float(first.var(ddof=1)), float(second.var(ddof=1))
    error = np.sqrt(var1 / n1 + var2 / n2)
    if error > 0:
        df = error**4 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1))
        margin = float(stats.t.ppf(0.975, df) * error)
    else:
        margin = 0.0
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    # Groups that barely vary inside themselves leave g undefined, not astronomically large.
    scale = max(abs(float(first.mean())), abs(float(second.mean())), 1.0)
    spread = pooled > NO_SPREAD * scale
    hedges = difference / pooled * (1 - 3 / (4 * (n1 + n2) - 9)) if spread else float("nan")
    ranks = stats.rankdata(np.concatenate([first, second]))
    superiority = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return difference, difference - margin, difference + margin, float(hedges), float(superiority)


def _comparison(
    omnibus: _Omnibus,
    column: int,
    values: pd.Series,
    groups: pd.Series,
    measure: str,
    grouping: str,
    **extra,
) -> GroupComparison:
    shape = _shape(omnibus, column, measure)
    uses_ranks = bool(shape)
    centre = omnibus.median[:, column] if uses_ranks else omnibus.mean[:, column]
    order = sorted(
        range(len(omnibus.levels)),
        key=lambda index: (centre[index], omnibus.mean_rank[index, column]),
        reverse=True,
    )
    summaries = tuple(
        GroupStats(
            value=omnibus.levels[index],
            label=group_label(grouping, omnibus.levels[index]),
            n=int(omnibus.n[index, column]),
            mean=float(omnibus.mean[index, column]),
            median=float(omnibus.median[index, column]),
            sd=float(omnibus.sd[index, column]),
        )
        for index in order
    )
    top, bottom = omnibus.levels[order[0]], omnibus.levels[order[-1]]
    first = values[groups == top].dropna().to_numpy(dtype=float)
    second = values[groups == bottom].dropna().to_numpy(dtype=float)
    mean_difference, low, high, hedges, superiority = _pair(first, second)
    welch, ranked = float(omnibus.welch_p[column]), float(omnibus.rank_p[column])
    return GroupComparison(
        measure=measure,
        grouping=grouping,
        groups=summaries,
        uses_ranks=uses_ranks,
        shape=shape,
        p_value=ranked if uses_ranks else welch,
        check_p_value=welch if uses_ranks else ranked,
        explained=float(omnibus.epsilon2[column] if uses_ranks else omnibus.eta2[column]),
        difference=float(centre[order[0]] - centre[order[-1]]) if uses_ranks else mean_difference,
        low=low,
        high=high,
        hedges_g=hedges,
        superiority=superiority,
        **extra,
    )


def _testing_sample(
    dataframe: pd.DataFrame, readings: list[str], grouping: str, entity: str | None
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """The rows to test, one per entity and group when ``entity`` repeats.

    Repeated readings of one subject are not independent evidence, so each
    entity contributes its mean in each group, or its median for a reading that
    is skewed or capped. Returns the readings, the groups, and the fields that
    record the reduction on a comparison.
    """
    columns = list(dict.fromkeys([grouping, *readings, *([entity] if entity else [])]))
    frame = dataframe[columns].dropna(subset=[grouping])
    collapse = (
        entity is not None
        and entity not in (grouping, *readings)
        and bool(frame.duplicated(subset=[entity, grouping]).any())
    )
    if not collapse:
        return frame[readings], frame[grouping], {}
    robust = {reading: _robust_on_rows(frame[reading]) for reading in readings}
    keyed = frame.dropna(subset=[entity]).groupby([entity, grouping], observed=True, sort=False)[readings]
    reduced = keyed.mean()
    medians = keyed.median()
    for reading in readings:
        if robust[reading]:
            reduced[reading] = medians[reading]
    reduced = reduced.reset_index()
    summary = "median" if all(robust.values()) else "mean"
    if len(readings) == 1:
        summary = "median" if robust[readings[0]] else "mean"
    fields = {
        "entity": entity,
        "entity_summary": summary,
        "shared_entities": bool(reduced[entity].duplicated().any()),
    }
    return reduced[readings], reduced[grouping], fields


def _single(
    values: pd.Series, groups: pd.Series, measure: str, grouping: str, fields: dict
) -> GroupComparison | None:
    data = pd.DataFrame({"value": values.to_numpy(), "group": groups.to_numpy()}).dropna()
    if data.empty:
        return None
    sizes = data.groupby("group", observed=True).size()
    small = sizes[sizes < MIN_GROUP_SIZE].index
    data = data[~data["group"].isin(small)]
    if data["group"].nunique() < 2 or data["value"].nunique() < 2:
        return None
    dropped = tuple(group_label(grouping, value) for value in small)
    omnibus = _omnibus(data[["value"]], data["group"])
    return _comparison(
        omnibus, 0, data["value"], data["group"], measure, grouping, dropped=dropped, **fields
    )


def compare_groups(
    dataframe: pd.DataFrame, measure: str, grouping: str, *, entity: str | None = None
) -> GroupComparison | None:
    """Compare ``measure`` across the groups of ``grouping``, or None when there is nothing to compare.

    Two groups get the difference in means with a Welch 95% interval, Hedges'
    g and a Mann-Whitney check; more groups get Welch's ANOVA, eta squared, a
    Kruskal-Wallis check and the pair that differs most. A skewed or capped
    reading is compared by medians and ranks instead. Groups with fewer than
    two values are left out and named in ``dropped``. With ``entity`` (a
    subject or sample column) each entity counts once per group.
    """
    if measure == grouping or measure not in dataframe.columns or grouping not in dataframe.columns:
        return None
    readings, groups, fields = _testing_sample(dataframe, [measure], grouping, entity)
    return _single(readings[measure], groups, measure, grouping, fields)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm's step-down adjustment, which keeps the chance of any false finding at the stated level."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    scaled = np.minimum(1.0, p[order] * (len(p) - np.arange(len(p))))
    adjusted = np.empty_like(p)
    adjusted[order] = np.maximum.accumulate(scaled)
    return adjusted


@dataclass(frozen=True)
class DifferenceScan:
    """Every reading compared across every grouping, with p-values adjusted for the number of tests."""

    #: The largest clear differences, by share of variance explained.
    named: tuple[GroupComparison, ...]
    #: The largest difference found, clear or not.
    largest: GroupComparison | None
    tested: int
    clear: int
    readings: tuple[str, ...]
    groupings: tuple[str, ...]
    entity: str | None
    #: Pairs left out because the grouping only marks where the reading hit its cap.
    flagged: tuple[tuple[str, str], ...] = ()
    #: Pairs left out because the reading has one value per group: another code for the grouping.
    recoded: tuple[tuple[str, str], ...] = ()
    #: Readings before the column cap.
    available: int = 0
    #: Rows in the table; above SCAN_ROWS the scan ran on a sample of SCAN_ROWS.
    rows: int = 0

    @property
    def sampled(self) -> bool:
        return self.rows > SCAN_ROWS


def scan_differences(
    dataframe: pd.DataFrame,
    readings: list[str],
    groupings: list[str],
    *,
    entity: str | None = None,
    skip: tuple[tuple[str, str], ...] = (),
    limit: int = 3,
) -> DifferenceScan | None:
    """Compare each reading across each grouping and rank the differences by size.

    ``skip`` holds (reading, grouping) pairs reported elsewhere. P-values are
    Holm-adjusted over every test run, so the scan's chance of naming any
    difference that isn't there stays at 5%. Very wide tables are scanned on
    their MAX_SCAN_COLUMNS most variable readings, and very long ones on a
    fixed random sample of SCAN_ROWS rows, which still detects any difference
    large enough to report.
    """
    available = len(readings)
    if available > MAX_SCAN_COLUMNS:
        variance = dataframe[readings].var()
        kept = set(variance.nlargest(MAX_SCAN_COLUMNS).index)
        readings = [reading for reading in readings if reading in kept]
    rows = len(dataframe)
    if rows > SCAN_ROWS:
        dataframe = dataframe.sample(SCAN_ROWS, random_state=0)
    entries: list[tuple[float, float, object]] = []
    flagged: list[tuple[str, str]] = []
    recoded: list[tuple[str, str]] = []
    used_entity = None
    for grouping in groupings:
        columns = [
            reading for reading in readings if reading != grouping and (reading, grouping) not in skip
        ]
        two_levels = dataframe[grouping].nunique(dropna=True) == 2
        marks = [
            reading for reading in columns
            if two_levels and cap_flag(dataframe[grouping], dataframe[reading])
        ]
        flagged.extend((reading, grouping) for reading in marks)
        columns = [reading for reading in columns if reading not in marks]
        if not columns:
            continue
        block, groups, fields = _testing_sample(dataframe, columns, grouping, entity)
        used_entity = fields.get("entity", used_entity)
        sizes = groups.value_counts()
        kept = groups.isin(sizes[sizes >= MIN_GROUP_SIZE].index)
        block, groups = block[kept], groups[kept]
        if groups.nunique() < 2:
            continue
        present = block.notna().groupby(groups, observed=True).sum()
        regular = [
            column
            for column in columns
            if int(present[column].min()) >= MIN_GROUP_SIZE and block[column].nunique() >= 2
        ]
        if regular:
            omnibus = _omnibus(block[regular], groups)
            for index, column in enumerate(regular):
                # A reading with one value in each group is another code for the
                # grouping (a 0/1 flag behind a condition label), not a finding.
                if not np.nansum(omnibus.sd[:, index]) > 0:
                    recoded.append((column, grouping))
                    continue
                pending = (omnibus, index, block[column], groups, column, grouping, fields)
                uses_ranks = bool(_shape(omnibus, index, column))
                explained = omnibus.epsilon2[index] if uses_ranks else omnibus.eta2[index]
                p = omnibus.rank_p[index] if uses_ranks else omnibus.welch_p[index]
                entries.append((float(explained), float(p), pending))
        for column in columns:
            if column in regular:
                continue
            single = _single(block[column], groups, column, grouping, fields)
            if single is not None:
                entries.append((single.explained, single.p_value, single))
    if not entries:
        return None

    adjusted = holm_adjust(np.array([p for _, p, _ in entries]))
    tests = len(entries)
    ranked = sorted(range(tests), key=lambda index: entries[index][0], reverse=True)

    def build(index: int) -> GroupComparison:
        item = entries[index][2]
        if not isinstance(item, GroupComparison):
            omnibus, column_index, values, groups, column, grouping, fields = item
            item = _comparison(omnibus, column_index, values, groups, column, grouping, **fields)
        return replace(item, adjusted_p=float(adjusted[index]), tests=tests)

    clear = [
        index
        for index in ranked
        if adjusted[index] < ALPHA and entries[index][0] >= NEGLIGIBLE_EXPLAINED
    ]
    return DifferenceScan(
        named=tuple(build(index) for index in clear[:limit]),
        largest=build(ranked[0]),
        tested=tests,
        clear=len(clear),
        readings=tuple(readings),
        groupings=tuple(groupings),
        entity=used_entity,
        flagged=tuple(flagged),
        recoded=tuple(recoded),
        available=available,
        rows=rows,
    )


def variance_explained(
    dataframe: pd.DataFrame, measure: str, groupings: list[str]
) -> tuple[tuple[str, float], ...]:
    """Share of the measure's variance each grouping explains on its own (eta squared), largest first."""
    shares = []
    for grouping in groupings:
        data = dataframe[[grouping, measure]].dropna()
        if data[grouping].nunique() < 2:
            continue
        values = data[measure]
        total = float(((values - values.mean()) ** 2).sum())
        if total == 0:
            continue
        means = values.groupby(data[grouping], observed=True).transform("mean")
        shares.append((grouping, float(((means - values.mean()) ** 2).sum()) / total))
    return tuple(sorted(shares, key=lambda item: item[1], reverse=True))


def combined_groupings(dataframe: pd.DataFrame, groupings: list[str]) -> dict[str, tuple[str, str]]:
    """Groupings that are the combination of two others: each level is exactly one pair of their levels.

    A cell code such as "T0" for contour Top at depth 0-10 explains at least as
    much as either factor, which says nothing about which factor matters.
    """
    # Each column as integer codes, -1 for missing, so a candidate is checked
    # with array operations rather than a groupby per pair of columns.
    codes = {grouping: pd.factorize(dataframe[grouping])[0].astype(np.int64) for grouping in groupings}
    levels = {grouping: int(codes[grouping].max()) + 1 for grouping in groupings}
    found: dict[str, tuple[str, str]] = {}
    for grouping in groupings:
        for first, second in itertools.combinations([g for g in groupings if g != grouping], 2):
            # A combination has more levels than either part and at most their product.
            if not max(levels[first], levels[second]) < levels[grouping] <= levels[first] * levels[second]:
                continue
            kept = (codes[grouping] >= 0) & (codes[first] >= 0) & (codes[second] >= 0)
            cells = codes[grouping][kept]
            if cells.size == 0 or np.unique(cells).size < 3:
                continue
            pairs = codes[first][kept] * levels[second] + codes[second][kept]
            if np.unique(codes[first][kept]).size < 2 or np.unique(codes[second][kept]).size < 2:
                continue
            # One pair per level and one level per pair: as many distinct (level,
            # pair) couples as distinct levels and as distinct pairs.
            couples = np.unique(cells * (levels[first] * levels[second]) + pairs).size
            if couples == np.unique(cells).size == np.unique(pairs).size:
                found[grouping] = (first, second)
                break
    return found


def entity_column(dataframe: pd.DataFrame, roles: ColumnRoles) -> str | None:
    """The column naming the unit each row belongs to, when some unit has several rows.

    The identifier, or a column named like one ("Subject", "Patient ID"); a
    column of unique values describes one row each and needs no reduction.
    """
    candidates = [roles.identifier, *(column for column in dataframe.columns if is_entity_name(column))]
    for column in dict.fromkeys(column for column in candidates if column):
        if column not in dataframe.columns or column in (roles.measure, roles.date):
            continue
        values = dataframe[column].dropna()
        if values.nunique() >= 2 and bool(values.duplicated().any()):
            return column
    return None


# --- Describing a comparison --------------------------------------------------


def format_p(p: float) -> str:
    """A p-value for a sentence: "p < 0.001", "p = 0.032", "p > 0.99"."""
    if not np.isfinite(p):
        return "p undefined"
    if p < 0.001:
        return "p < 0.001"
    if p > 0.99:
        return "p > 0.99"
    return f"p = {p:.2g}"


def size_word(comparison: GroupComparison) -> str:
    """Conventional size of the difference: negligible, small, medium or large."""
    if comparison.two_groups and not comparison.uses_ranks and np.isfinite(comparison.hedges_g):
        value, sizes = abs(comparison.hedges_g), HEDGES_SIZES
    else:
        value, sizes = comparison.explained, EXPLAINED_SIZES
    return next((word for threshold, word in sizes if value >= threshold), "negligible")


def _amount(value: float, comparison: GroupComparison, values: pd.Series | None) -> str:
    """A signed difference in the measure's own terms: percentage points for a rate."""
    if metric_for(comparison.measure).unit == "rate":
        points = value * 100 if values is not None and values.dropna().between(0, 1).all() else value
        return f"{points:.1f} percentage points"
    return format_number(value, comparison.measure, column_values=values)


def effect_text(comparison: GroupComparison) -> str:
    """The effect size the comparison is judged on, as "Hedges' g = 0.07"."""
    if comparison.two_groups and not comparison.uses_ranks and np.isfinite(comparison.hedges_g):
        return f"Hedges' g = {comparison.hedges_g:.2f}"
    return f"{comparison.explained_name} = {comparison.explained:.2f}"


def _test_text(comparison: GroupComparison) -> str:
    adjusted = "Holm-adjusted " if comparison.adjusted_p is not None else ""
    return f"{comparison.test}, {adjusted}{format_p(comparison.decisive_p)}"


def _tied(comparison: GroupComparison, centre: float) -> list[str]:
    """The groups whose centre equals ``centre``, so a tie is not reported as one group."""
    return [group.label for group in comparison.groups if np.isclose(comparison.centre(group), centre)]


def describe_comparison(comparison: GroupComparison, column_values: pd.Series | None = None) -> str:
    """State whether the reading differs across the groups, with the sizes, test and effect behind it.

    A difference the test can't tell from noise is written as "no clear
    difference", never as a ranking, and one that passes the test but explains
    under 1% of the variance is called negligible.
    """
    c = comparison
    word = "median" if c.uses_ranks else "mean"
    high, low = c.higher, c.lower
    first, second = format_values_apart(
        c.centre(high), c.centre(low), column=c.measure, column_values=column_values
    )
    sizes = f"n = {high.n:,} and {low.n:,}"
    verdict = c.verdict
    if c.two_groups:
        quoted = f"{word} {first} vs {second} ({sizes})"
        interval = (
            f"{_amount(c.low, c, column_values)} to {_amount(c.high, c, column_values)}"
        )
        if verdict == "clear":
            text = f"{c.measure} is higher in {high.label} than in {low.label}: {quoted}"
            if c.uses_ranks:
                text += (
                    f". A value from {high.label} is the larger in {c.superiority:.0%} of pairs "
                    f"({_test_text(c)})."
                )
            else:
                text += (
                    f", a difference of {_amount(c.difference, c, column_values)} (95% CI {interval}); "
                    f"{effect_text(c)}, a {size_word(c)} difference ({_test_text(c)})."
                )
            if c.check_p_value < ALPHA:
                text += f" {c.check} agrees ({format_p(c.check_p_value)})."
            else:
                text += (
                    f" {c.check} does not confirm it ({format_p(c.check_p_value)}), so treat it with care."
                )
        elif verdict == "negligible":
            text = (
                f"{c.measure} differs between {high.label} and {low.label} only by a negligible "
                f"amount: {quoted}, {effect_text(c)} ({_test_text(c)}). With this many records "
                "even a trivial gap passes the test."
            )
        else:
            text = (
                f"{c.measure} shows no clear difference between {high.label} and {low.label}: "
                f"{quoted}."
            )
            if c.uses_ranks:
                text += f" The rank test can't tell them apart ({_test_text(c)}; {effect_text(c)})."
            elif c.low <= 0 <= c.high:
                text += (
                    f" The 95% CI for the difference, {interval}, includes zero "
                    f"({_test_text(c)}; {effect_text(c)})."
                )
            else:
                text += (
                    f" The gap does not hold up once all {c.tests} tests run are allowed for "
                    f"({_test_text(c)}; {effect_text(c)})."
                )
    else:
        span = (
            f"{word}s run from {second} ({low.label}, n = {low.n:,}) to {first} "
            f"({high.label}, n = {high.n:,})"
        )
        variation = "of its variance" if not c.uses_ranks else "of the variation in its ranks"
        if verdict == "clear":
            top, bottom = _tied(c, c.centre(high)), _tied(c, c.centre(low))
            verb = "have" if len(top) > 1 else "has"
            text = (
                f"{c.measure} differs across {c.grouping}: {list_names(top)} {verb} the highest {word}, "
                f"{first}, and {list_names(bottom)} the lowest, {second} ({sizes}). {c.grouping} explains "
                f"{c.explained:.0%} {variation} ({c.explained_name} = {c.explained:.2f}, {_test_text(c)})"
            )
            if c.uses_ranks:
                text += (
                    f"; a value from {high.label} is the larger in {c.superiority:.0%} of pairs "
                    f"against {low.label}."
                )
            elif np.isfinite(c.hedges_g):
                text += f"; Hedges' g between the two is {c.hedges_g:.2f}."
            else:
                text += "."
        elif verdict == "negligible":
            text = (
                f"{c.measure} differs across {c.grouping}, but {c.grouping} explains only "
                f"{c.explained:.1%} {variation} ({c.explained_name} = {c.explained:.3f}, "
                f"{_test_text(c)}): {span}."
            )
        else:
            text = (
                f"{c.measure} shows no clear difference across the {len(c.groups)} {c.grouping} groups "
                f"({_test_text(c)}; {effect_text(c)}): {span}."
            )
    if c.uses_ranks:
        text += f" Medians and ranks are used because {c.shape}."
    if c.entity:
        text += f" Each {c.entity} counts once per group."
        if c.shared_entities:
            text += (
                f" The same {c.entity} appears in more than one group, so the groups are not "
                "independent and the p-value is only a guide."
            )
    if c.dropped:
        text += f" Left out with fewer than {MIN_GROUP_SIZE} values: {', '.join(c.dropped)}."
    return text


def lead_sentence(comparison: GroupComparison, column_values: pd.Series | None = None) -> str:
    """One sentence naming a clear difference and the groups' centres, for a headline."""
    c = comparison
    word = "median" if c.uses_ranks else "mean"
    first, second = format_values_apart(
        c.centre(c.higher), c.centre(c.lower), column=c.measure, column_values=column_values
    )
    if c.two_groups:
        return (
            f"{c.measure} is higher in {c.higher.label} than in {c.lower.label}: {word} {first} vs "
            f"{second} (n = {c.higher.n:,} and {c.lower.n:,})."
        )
    return (
        f"{c.measure} differs across {c.grouping}: {word}s from {second} ({c.lower.label}) to "
        f"{first} ({c.higher.label}), {c.explained_name} = {c.explained:.2f}."
    )


def effect_clause(comparison: GroupComparison) -> str:
    """The effect and p-value of a comparison in a few words: "Hedges' g = 0.50, p < 0.001"."""
    c = comparison
    if c.two_groups and c.uses_ranks:
        effect = f"{c.higher.label} larger in {c.superiority:.0%} of pairs"
    else:
        effect = effect_text(c)
    raw, adjusted = format_p(c.p_value), format_p(c.decisive_p)
    if adjusted == raw:
        return f"{effect}, {adjusted}"
    # A sizeable effect next to "p > 0.99" reads as a contradiction unless the adjustment shows.
    return f"{effect}; {raw}, Holm-adjusted {adjusted.removeprefix('p ')}"


def comparison_calculation(comparison: GroupComparison) -> str:
    """The calculation behind ``describe_comparison``, step by step."""
    c = comparison
    word = "median" if c.uses_ranks else "mean"
    steps = [f"{word} {c.measure} per {c.grouping}"]
    if c.two_groups and c.uses_ranks:
        steps.append(
            "Mann-Whitney test on ranks, with the share of cross-group pairs in which the higher "
            "group's value is the larger; checked with Welch's t-test"
        )
    elif c.two_groups:
        steps.append(
            "difference in means with a Welch 95% CI and Welch's t-test, Hedges' g; checked with "
            "the Mann-Whitney test"
        )
    elif c.uses_ranks:
        steps.append(
            "Kruskal-Wallis with ε² = H ÷ (n − 1); checked with Welch's ANOVA; highest vs lowest median"
        )
    else:
        steps.append(
            "Welch's ANOVA with η² as the between-group share of the total sum of squares; checked with "
            "Kruskal-Wallis; Hedges' g for the highest vs lowest mean"
        )
    if c.uses_ranks:
        steps.append(f"ranks because {c.shape}")
    if c.entity:
        steps.append(f"one value per {c.entity} in each group (its {c.entity_summary}) before testing")
    if c.adjusted_p is not None:
        steps.append(f"p Holm-adjusted for {c.tests} test{'' if c.tests == 1 else 's'}")
    return "; ".join(steps)


def comparison_table(comparison: GroupComparison) -> pd.DataFrame:
    """One row per group, highest centre first: the centre, n, the other centre and the SD."""
    word, other = ("Median", "Mean") if comparison.uses_ranks else ("Mean", "Median")
    return pd.DataFrame(
        {
            comparison.grouping: [group.label for group in comparison.groups],
            f"{word} {comparison.measure}": [comparison.centre(group) for group in comparison.groups],
            "n": [group.n for group in comparison.groups],
            f"{other} {comparison.measure}": [
                group.mean if comparison.uses_ranks else group.median for group in comparison.groups
            ],
            "SD": [group.sd for group in comparison.groups],
        }
    )


# --- Relationships ------------------------------------------------------------


def correlation_interval(value: float, pairs: int) -> tuple[float, float]:
    """95% confidence interval for a correlation, via the Fisher transform."""
    if pairs <= 3 or abs(value) >= 1.0:
        return value, value
    centre = np.arctanh(value)
    margin = stats.norm.ppf(0.975) / np.sqrt(pairs - 3)
    return float(np.tanh(centre - margin)), float(np.tanh(centre + margin))


def correlation_p(value: float, pairs: int) -> float:
    """Two-sided p-value for a Pearson r of zero."""
    if pairs <= 2 or not np.isfinite(value):
        return 1.0
    if abs(value) >= 1.0:
        return 0.0
    t = value * np.sqrt((pairs - 2) / (1 - value**2))
    return float(2 * stats.t.sf(abs(t), pairs - 2))


def pairwise_correlations(frame: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r and the number of complete pairs for every pair of columns, as matrices.

    Each pair uses the rows where both columns are recorded, computed with a
    handful of matrix products rather than one pair at a time. Columns are
    standardised first so very large readings keep their precision.
    """
    values = np.asarray(frame, dtype=float)
    present = ~np.isnan(values)
    if present.all():
        rows = len(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = (values - values.mean(axis=0)) / values.std(axis=0)
            correlation = scaled.T @ scaled / rows
        return np.clip(correlation, -1.0, 1.0), np.full(correlation.shape, float(rows))
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scaled = np.where(present, (values - centre) / np.where(scale > 0, scale, 1.0), 0.0)
        mask = present.astype(float)
        counts = mask.T @ mask
        sums = scaled.T @ mask
        squares = (scaled**2).T @ mask
        products = scaled.T @ scaled
        covariance = counts * products - sums * sums.T
        spread = (counts * squares - sums**2) * (counts * squares - sums**2).T
        correlation = covariance / np.sqrt(spread)
    return np.clip(correlation, -1.0, 1.0), counts


def most_influential(first: np.ndarray, second: np.ndarray) -> tuple[int, float] | None:
    """The position of the record whose removal changes r most, and r without it.

    Every leave-one-out r comes from the full sums at once, so the check costs
    one pass over the data.
    """
    n = len(first)
    if n < 4:
        return None
    x, y = first - first.mean(), second - second.mean()
    shrink = n / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        without = ((x * y).sum() - shrink * x * y) / np.sqrt(
            ((x**2).sum() - shrink * x**2) * ((y**2).sum() - shrink * y**2)
        )
        full = (x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum())
    change = np.abs(np.nan_to_num(without, nan=full) - full)
    position = int(np.argmax(change))
    return position, float(np.nan_to_num(without[position], nan=full))


def _robust_z(values: np.ndarray) -> np.ndarray | None:
    centre = np.median(values)
    spread = 1.4826 * np.median(np.abs(values - centre))
    if spread == 0:
        quartiles = np.percentile(values, [25, 75])
        spread = (quartiles[1] - quartiles[0]) / 1.349
    if spread == 0:
        return None
    return np.abs(values - centre) / spread


def without_extremes(first: np.ndarray, second: np.ndarray) -> tuple[float, int] | None:
    """Pearson r after dropping records more than EXTREME_Z robust SDs out in either column.

    Returns None when nothing is that extreme, or when so much is (over 5% of
    records) that the extremes are part of the distribution rather than errors.
    """
    z_first, z_second = _robust_z(first), _robust_z(second)
    if z_first is None and z_second is None:
        return None
    keep = np.ones(len(first), dtype=bool)
    for z in (z_first, z_second):
        if z is not None:
            keep &= z <= EXTREME_Z
    dropped = int((~keep).sum())
    if dropped == 0 or dropped > max(3, 0.05 * len(first)) or keep.sum() < MIN_CORRELATION_PAIRS:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.corrcoef(first[keep], second[keep])[0, 1]
    return (float(value), dropped) if np.isfinite(value) else None


@dataclass(frozen=True)
class WithinGroups:
    """A correlation recomputed inside each group of a grouping column."""

    grouping: str
    #: r between the two columns after removing each group's own means.
    within: float
    groups: int
    median: float
    low: float
    high: float


def within_group_correlation(
    dataframe: pd.DataFrame, first: str, second: str, grouping: str, *, min_rows: int = 5
) -> WithinGroups | None:
    """The correlation within groups, which can differ in size and even sign from the pooled one.

    Groups that differ in both columns produce a pooled correlation that
    describes the groups rather than how the columns move together inside any
    of them.
    """
    if grouping in (first, second):
        return None
    data = dataframe[[first, second, grouping]].dropna()
    sizes = data.groupby(grouping, observed=True)[first].transform("size")
    data = data[sizes >= min_rows]
    per_group = []
    for _, part in data.groupby(grouping, observed=True):
        if part[first].nunique() > 1 and part[second].nunique() > 1:
            per_group.append(float(np.corrcoef(part[first], part[second])[0, 1]))
    if len(per_group) < 2:
        return None
    centred = data[[first, second]] - data.groupby(grouping, observed=True)[[first, second]].transform("mean")
    within = float(centred[first].corr(centred[second]))
    if not np.isfinite(within):
        return None
    return WithinGroups(
        grouping=grouping,
        within=within,
        groups=len(per_group),
        median=float(np.median(per_group)),
        low=float(min(per_group)),
        high=float(max(per_group)),
    )


@dataclass(frozen=True)
class Derivation:
    """A column that a fixed combination of other columns reproduces."""

    column: str
    parts: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    #: "sum", "mean", "difference" or "combination".
    kind: str

    def describe(self) -> str:
        if self.kind == "sum":
            return f"{self.column} = {' + '.join(self.parts)}"
        if self.kind == "mean":
            return f"{self.column} = mean of {_joined(self.parts)}"
        if self.kind == "difference":
            ordered = sorted(zip(self.coefficients, self.parts, strict=True), key=lambda item: -item[0])
            formula = ordered[0][1] + "".join(
                f" {'+' if weight > 0 else '−'} {part}" for weight, part in ordered[1:]
            )
            return f"{self.column} = {formula}"
        return f"{self.column} is a fixed combination of {_joined(self.parts)}"


def _joined(names: tuple[str, ...] | list[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _simple_fraction(weight: float) -> float:
    """The nearest fraction with a denominator up to 12, when it is within 2% of ``weight``."""
    for denominator in range(1, 13):
        numerator = round(weight * denominator)
        if abs(weight - numerator / denominator) <= 0.02 * max(1.0, abs(weight)):
            return numerator / denominator
    return weight


def _classify(coefficients: np.ndarray, intercept: float, tolerance: float) -> str:
    near = np.isclose
    if abs(intercept) <= tolerance:
        if near(coefficients, 1.0, rtol=0, atol=0.02).all():
            return "sum"
        if near(coefficients * len(coefficients), 1.0, rtol=0, atol=0.02).all():
            return "mean"
        if near(np.abs(coefficients), 1.0, rtol=0, atol=0.02).all():
            return "difference"
    return "combination"


def find_derivation(frame: pd.DataFrame, column: str, candidates: list[str]) -> Derivation | None:
    """Whether ``column`` is a fixed combination of a few of ``candidates``: a sum, a mean, a difference.

    Candidates should come strongest-correlated first. They are added one at a
    time, each the one closest to what the others leave unexplained, until the
    combination reproduces the column on DERIVED_SHARE of its rows. A fit
    needs three rows per term, so a few rows can't be matched by chance.
    """
    # A part recorded on only some rows (acidity measured only on acid soils)
    # still shows the identity on the rows that have it, so candidates are
    # kept while enough rows have all of them.
    complete = frame[column].notna()
    needed = max(DERIVATION_MIN_ROWS, DERIVATION_MIN_SHARE * int(complete.sum()))
    admitted = []
    for candidate in candidates:
        joint = complete & frame[candidate].notna()
        if joint.sum() >= needed:
            admitted.append(candidate)
            complete = joint
    candidates = admitted
    data = frame.loc[complete, [column, *candidates]]
    if len(data) > DERIVATION_ROWS:
        data = data.iloc[np.linspace(0, len(data) - 1, DERIVATION_ROWS).astype(int)]
    rows = len(data)
    usable = min(len(candidates), rows // 3 - 1)
    if usable < 2:
        return None
    candidates = candidates[:usable]
    target = data[column].to_numpy(dtype=float)
    inputs = data[candidates].to_numpy(dtype=float)
    centred = target - target.mean()
    total = float((centred**2).sum())
    if total == 0:
        return None
    # Measured on the middle half of the values, so a few huge readings can't
    # widen the tolerance until anything matches.
    quartiles = np.percentile(target, [25, 75])
    spread = float(quartiles[1] - quartiles[0]) / 1.349 or float(target.std())
    # A column stored at a coarser step (whole degrees, then converted) can only
    # match its formula to within half that step.
    grain = spread * 1e-6
    steps = np.diff(np.unique(np.round(target / grain))) * grain
    half_step = float(steps[steps > 0].min()) / 2 * (1 + 1e-3) if (steps > 0).any() else 0.0
    tolerance = max(DERIVED_TOLERANCE * spread, min(half_step, DERIVED_STEP_LIMIT * spread))
    ones = np.ones(rows)

    def matched(residual: np.ndarray) -> float:
        return float(np.mean(np.abs(residual) <= tolerance))

    def fit(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Least squares, refitted on the rows it matches best, or its weights as simple fractions.

        A few rows entered by hand, or a column rounded after it was computed,
        pull least squares slightly off the formula (0.49 and 0.52 instead of
        a half each), so the fit is also tried with each weight snapped to
        the nearest simple fraction.
        """
        design = np.column_stack([ones, inputs[:, indices]])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        residual = np.abs(target - design @ coefficients)
        close = residual <= np.quantile(residual, DERIVED_SHARE)
        coefficients = np.linalg.lstsq(design[close], target[close], rcond=None)[0]
        residual = target - design @ coefficients
        snapped = np.array([_simple_fraction(weight) for weight in coefficients[1:]])
        offset = float(np.median(target - design[:, 1:] @ snapped))
        offset = 0.0 if abs(offset) <= tolerance else offset
        simple = np.r_[offset, snapped]
        simple_residual = target - design @ simple
        if matched(simple_residual) >= matched(residual):
            return simple, simple_residual
        return coefficients, residual

    def reproduces(residual: np.ndarray) -> bool:
        # Matching most rows is not enough when they are mostly one value (a
        # column of zeros with a few readings): the fit must also explain the spread.
        explained = 1 - float((residual**2).sum()) / total
        return matched(residual) >= DERIVED_SHARE and explained >= DERIVED_EXPLAINED

    if not reproduces(fit(list(range(usable)))[1]):
        return None
    scale = inputs.std(axis=0)
    standardised = (inputs - inputs.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    chosen: list[int] = []
    residual = target - target.mean()
    for _ in range(min(MAX_PARTS, usable)):
        scores = np.abs(standardised.T @ residual)
        scores[chosen] = -np.inf
        chosen.append(int(np.argmax(scores)))
        coefficients, residual = fit(chosen)
        if reproduces(residual):
            weights = coefficients[1:]
            used = np.abs(weights) * scale[chosen] > 1e-6 * spread
            if used.sum() < 2:
                return None
            parts = tuple(candidates[index] for index, keep in zip(chosen, used, strict=True) if keep)
            weights = weights[used]
            return Derivation(
                column=column,
                parts=parts,
                coefficients=tuple(float(weight) for weight in weights),
                intercept=float(coefficients[0]),
                kind=_classify(weights, float(coefficients[0]), tolerance),
            )
    return None


@dataclass(frozen=True)
class CapFlag:
    """A two-valued column that only marks where another column sits at its maximum or minimum."""

    flag: str
    column: str
    value: float
    edge: str
    rows: int

    def describe(self) -> str:
        return (
            f"{self.flag} marks the {self.rows:,} records where {self.column} sits at its "
            f"{self.edge} ({format_number(self.value, self.column, compact=False)})"
        )


def cap_flag(flag: pd.Series, values: pd.Series) -> CapFlag | None:
    """Whether a two-valued ``flag`` is set exactly on the rows where ``values`` hits its cap.

    A censoring flag next to a timer stopped at 3600 s separates the groups by
    definition, so comparing or correlating the two says nothing new.
    """
    if flag.name == values.name or not pd.api.types.is_numeric_dtype(values):
        return None
    # Codes and floats rather than a frame per pair: the scan calls this for
    # every reading against every two-valued grouping.
    codes = pd.factorize(flag)[0]
    numbers = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    kept = (codes >= 0) & ~np.isnan(numbers)
    codes, numbers = codes[kept], numbers[kept]
    if numbers.size == 0 or np.unique(codes).size != 2 or np.unique(numbers).size < 3:
        return None
    for edge, value in (("maximum", numbers.max()), ("minimum", numbers.min())):
        at_edge = numbers == value
        if at_edge.sum() < 2:
            continue
        marked = np.unique(codes[at_edge])
        if marked.size == 1 and not np.isin(codes[~at_edge], marked).any():
            return CapFlag(str(flag.name), str(values.name), float(value), edge, int(at_edge.sum()))
    return None


def _stems(name: str) -> set[str]:
    """The words of a name that say what was measured, without its units or bracketed notes.

    Two readings in the same unit ("cmol(+)/kg") share nothing but the unit.
    """
    bare = re.sub(r"\([^)]*\)|\[[^\]]*\]", "", str(name))
    bare = " ".join(chunk for chunk in bare.split() if "/" not in chunk)
    words = re.findall(r"[a-z]+", normalized_name(bare))
    return {word for word in words if len(word) >= 3 and word not in GENERIC_NAME_WORDS}


@dataclass(frozen=True)
class Pair:
    """Two columns' correlation over their complete rows."""

    first: str
    second: str
    pearson: float
    spearman: float
    pairs: int
    #: The larger of |Spearman r| and |Pearson r without its most influential record|,
    #: so one stray record can neither make nor hide a relationship in the ranking.
    strength: float
    #: Two-sided p-value of ``strength`` against no relationship.
    p_value: float


@dataclass(frozen=True)
class RelationshipScan:
    """Every pair of numeric columns screened, with pairs related by construction set aside."""

    measure: str | None
    columns: tuple[str, ...]
    #: Numeric columns before the cap on very wide tables.
    available: int
    #: Columns correlated with the measure, strongest first.
    correlates: tuple[Pair, ...]
    #: Pairs without the measure, strongest first.
    others: tuple[Pair, ...]
    #: Correlations with the measure that were tested, and pairs scanned in all.
    measure_tests: int
    pair_tests: int
    derived: tuple[Derivation, ...]
    flags: tuple[CapFlag, ...]
    near_duplicates: tuple[tuple[str, str, float], ...]
    shared_stems: tuple[tuple[str, str, float], ...]


def _families(
    columns: list[str], pearson: np.ndarray, spearman: np.ndarray
) -> tuple[list[int], list[tuple[str, str, float]]]:
    """Group columns that are one quantity recorded twice: |r| ≥ 0.99 with ranks that agree."""
    parent = list(range(len(columns)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    duplicates = []
    first, second = np.where(np.triu(np.abs(pearson) >= NEAR_DUPLICATE_CORRELATION, k=1))
    for i, j in zip(first, second, strict=True):
        if abs(pearson[i, j]) - abs(spearman[i, j]) <= RANK_DIVERGENCE:
            parent[root(i)] = root(j)
            duplicates.append((columns[i], columns[j], float(pearson[i, j])))
    return [root(index) for index in range(len(columns))], duplicates


def _names_an_aggregate(column: str) -> bool:
    return bool(set(normalized_name(column).split()) & AGGREGATE_NAME_WORDS)


def _derivations(
    frame: pd.DataFrame, columns: list[str], pearson: np.ndarray, counts: np.ndarray, family: list[int]
) -> dict[str, Derivation]:
    """Columns built from two or more others, with their parts traced to columns not built themselves.

    An exact identity can be read either way round (a total is the sum of its
    items, and an item is the total minus the others), so a column is matched
    against the columns before it, where computed columns are usually
    appended. A column named as a total or mean may also be the sum or mean of
    columns after it. A total can be rebuilt from other totals ("CorIncor =
    CorrectTotal − IncorrectTotal"), so each derivation is then retraced to the
    columns that are not derived themselves, which names the items it was made
    from.
    """
    present = counts.diagonal()

    def candidates(index: int, allowed: set[int]) -> list[str]:
        strength = np.nan_to_num(np.abs(pearson[index]), nan=-1.0)
        order = np.argsort(-strength)
        return [
            columns[other]
            for other in order
            if other in allowed
            and family[other] != family[index]
            and strength[other] > 0
            and counts[index, other] >= DERIVATION_MIN_SHARE * present[index]
        ][:PART_CANDIDATES]

    def search(index: int, allowed: set[int]) -> Derivation | None:
        column = columns[index]
        found = find_derivation(frame, column, candidates(index, allowed & set(range(index))))
        if found is None and _names_an_aggregate(column):
            found = find_derivation(frame, column, candidates(index, allowed))
            if found is not None and found.kind not in ("sum", "mean"):
                found = None
        return found

    everything = set(range(len(columns)))
    derived = {}
    for index, column in enumerate(columns):
        found = search(index, everything)
        if found is not None:
            derived[column] = found
    leaves = {index for index, column in enumerate(columns) if column not in derived}
    for column in list(derived):
        refined = search(columns.index(column), leaves)
        if refined is not None:
            derived[column] = refined
    # Name the parts in table order, as a reader would write the formula.
    for column, found in derived.items():
        ordered = sorted(
            zip(found.parts, found.coefficients, strict=True), key=lambda item: columns.index(item[0])
        )
        derived[column] = replace(
            found,
            parts=tuple(part for part, _ in ordered),
            coefficients=tuple(weight for _, weight in ordered),
        )
    return derived


def _cap_flags(frame: pd.DataFrame, ranked: _Ranked, counts: np.ndarray) -> list[CapFlag]:
    """Every two-valued column that only marks another column's cap.

    A flag set on m rows can only mark a cap reached on m rows (or on all the
    others), so most pairs are ruled out by counts alone.
    """
    columns = list(frame.columns)
    present = counts.diagonal()
    flags = []
    for index, flag in enumerate(columns):
        if ranked.distinct[index] != 2:
            continue
        minority = int(min(ranked.at_max[index], ranked.at_min[index]))
        for other, column in enumerate(columns):
            if other == index:
                continue
            sizes = {minority, int(present[other]) - minority}
            if int(ranked.at_max[other]) not in sizes and int(ranked.at_min[other]) not in sizes:
                continue
            found = cap_flag(frame[flag], frame[column])
            if found is not None:
                flags.append(found)
    return flags


def scan_relationships(
    dataframe: pd.DataFrame,
    columns: list[str],
    measure: str | None = None,
    *,
    limit: int = 12,
    others: bool = True,
) -> RelationshipScan | None:
    """Rank what relates to ``measure``, then the strongest pairs among the other columns.

    Pearson and Spearman r for every pair come from matrix products over the
    complete rows of each pair; Spearman r ranks each column once, so with
    missing values it is a close screen, and the pairs returned are recomputed
    exactly. Pairs related by construction are left out and returned for the
    calculation: near-duplicates, a column built from the other (or both built
    from the same items), a flag marking the other's cap, and same-stem names
    above SHARED_STEM_CORRELATION. Returns None when no pair has
    MIN_CORRELATION_PAIRS complete rows.
    """
    columns = list(dict.fromkeys(columns))
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        variance = np.nanvar(dataframe[columns].to_numpy(dtype=float), axis=0)
    # A column that never varies relates to nothing.
    columns = [column for column, spread in zip(columns, variance, strict=True) if spread > 0]
    available = len(columns)
    if available > MAX_SCAN_COLUMNS:
        spread = pd.Series(variance[variance > 0], index=columns)
        rest = spread.drop(measure) if measure in spread.index else spread
        kept = set(rest.nlargest(MAX_SCAN_COLUMNS - (measure in spread.index)).index) | {measure}
        columns = [column for column in columns if column in kept]
    if len(columns) < 2:
        return None
    frame = dataframe[columns].astype(float)
    values = frame.to_numpy()
    ranked = _rank_columns(values)
    pearson, counts = pairwise_correlations(values)
    spearman, _ = pairwise_correlations(ranked.ranks)
    usable = counts >= MIN_CORRELATION_PAIRS
    np.fill_diagonal(usable, False)
    if not usable.any():
        return None
    pearson = np.where(usable, pearson, np.nan)
    spearman = np.where(usable, spearman, np.nan)
    screen = np.fmax(np.abs(pearson), np.abs(spearman))
    family, duplicates = _families(columns, pearson, spearman)
    # Exact identities hold on every row, so a spread of rows is enough to find them.
    thinned = frame.iloc[np.linspace(0, len(frame) - 1, min(len(frame), 5 * DERIVATION_ROWS)).astype(int)]
    derived = _derivations(thinned, columns, pearson, counts, family)
    flags = _cap_flags(frame, ranked, counts)
    flagged = {frozenset((flag.flag, flag.column)) for flag in flags}
    position = {column: index for index, column in enumerate(columns)}

    def parts_of(column: str) -> set[int] | None:
        found = derived.get(column)
        return {family[position[part]] for part in found.parts} if found else None

    stems: list[tuple[str, str, float]] = []

    def related(first: str, second: str) -> bool:
        i, j = position[first], position[second]
        if family[i] == family[j] or frozenset((first, second)) in flagged:
            return True
        first_parts, second_parts = parts_of(first), parts_of(second)
        if first_parts and family[j] in first_parts or second_parts and family[i] in second_parts:
            return True
        if first_parts and second_parts and (first_parts <= second_parts or second_parts <= first_parts):
            return True
        if abs(pearson[i, j]) > SHARED_STEM_CORRELATION and _stems(first) & _stems(second):
            stems.append((first, second, float(pearson[i, j])))
            return True
        return False

    def detail(first: str, second: str) -> Pair:
        i, j = position[first], position[second]
        both = ~np.isnan(values[:, i]) & ~np.isnan(values[:, j])
        x, y = values[both, i], values[both, j]
        value = float(pearson[i, j])
        rank_r = float(spearman[i, j])
        if counts[i, j] < min(counts[i, i], counts[j, j]):
            # Ranks taken over each whole column differ from the pair's own ranks.
            rank_r = float(np.corrcoef(_rank_columns(np.column_stack([x, y])).ranks, rowvar=False)[0, 1])
        influence = most_influential(x, y)
        steady = abs(influence[1]) if influence else abs(value)
        strength = float(np.nanmax([abs(rank_r), steady]))
        # Tested on the figure it is ranked by, so a rank correlation hidden by
        # a few extreme values is not dismissed as chance.
        return Pair(first, second, value, rank_r, len(x), strength, correlation_p(strength, len(x)))

    def best(candidates: list[tuple[str, str]], keep: int) -> list[Pair]:
        found: list[Pair] = []
        # MaxTemp ~ MinTemp and its copy in other units, MAX ~ MIN, are one pair.
        seen: set[frozenset[int]] = set()
        for first, second in candidates:
            key = frozenset((family[position[first]], family[position[second]]))
            if key in seen or related(first, second):
                continue
            seen.add(key)
            found.append(detail(first, second))
            if len(found) >= keep:
                break
        return sorted(found, key=lambda pair: pair.strength, reverse=True)

    correlates: list[Pair] = []
    measure_tests = 0
    if measure in position:
        i = position[measure]
        order = [j for j in np.argsort(-np.nan_to_num(screen[i], nan=-1.0)) if usable[i, j]]
        measure_tests = len(order)
        correlates = best([(measure, columns[j]) for j in order], max(limit, 15))
    pair_list: list[Pair] = []
    if others:
        upper = np.triu(usable, k=1)
        if measure in position:
            # A pair with the measure's own duplicate is a pair with the measure.
            same = [index for index in range(len(columns)) if family[index] == family[position[measure]]]
            upper[same, :] = False
            upper[:, same] = False
        first, second = np.where(upper)
        order = np.argsort(-screen[first, second], kind="stable")[:200]
        pair_list = best([(columns[first[k]], columns[second[k]]) for k in order], 30)
    pair_tests = int(np.triu(usable, k=1).sum())
    return RelationshipScan(
        measure=measure if measure in position else None,
        columns=tuple(columns),
        available=available,
        correlates=tuple(correlates[:limit]),
        others=tuple(pair_list[:limit]),
        measure_tests=measure_tests,
        pair_tests=pair_tests,
        derived=tuple(derived.values()),
        flags=tuple(flags),
        near_duplicates=tuple(sorted(duplicates, key=lambda item: -abs(item[2]))),
        shared_stems=tuple(dict.fromkeys(stems)),
    )
