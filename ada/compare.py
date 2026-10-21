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
from functools import cached_property

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.proportion import confint_proportions_2indep

from .formatting import format_number, format_values_apart, group_label, list_names, normalized_name
from .metrics import metric_for
from .schema import COORDINATE_TOKENS, LABEL_NAME_WORDS, ColumnRoles, is_design_name, is_entity_name

ALPHA = 0.05
# A group needs two values to have a spread to test against.
MIN_GROUP_SIZE = 2
# Groups smaller than this are compared but never named as the highest or the
# lowest, and a difference resting on one is not called clear: a few values can
# sit far from the rest by chance.
MIN_NAMED_GROUP = 5
# Welch's test weights each group by its own variance, which fewer values than
# this estimate too poorly, so such comparisons pool the variance instead
# (Student's t-test, one-way ANOVA).
MIN_WELCH_GROUP = 3
# Beyond this many records even a trivial gap passes a test.
LARGE_SAMPLE = 1_000
# A reading with at least this share of zeros is compared on its means and on
# the share of values that are not zero, since every group's median is 0.
ZERO_HEAVY = 0.5
# Below this rank correlation a trend across ordered levels is too small to matter.
NEGLIGIBLE_TREND = 0.1
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
# A difference scan over more readings than MAX_SCAN_COLUMNS is a screen: every
# reading is tested, up to this many cells, and p-values control the false
# discovery rate rather than the chance of any false finding.
SCREEN_CELLS = 20_000_000
HOLM = "Holm"
FALSE_DISCOVERY = "Benjamini-Hochberg"
ADJUSTED = {HOLM: "Holm-adjusted", FALSE_DISCOVERY: "false-discovery-rate adjusted"}

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
# A free combination of columns (weights other than ±1 or 1/n) must hold on this share of rows.
COMBINATION_SHARE = 0.99
# Weights snapped to simple fractions are kept when they match at most this share of rows fewer.
SIMPLE_MARGIN = 0.01
# Times a fit is redone on the rows it matches best.
REFITS = 3
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
# A reading whose groups explain this share of its variance barely varies within a
# group: it describes the group (a site's coordinates, a station's battery), not its members.
ATTRIBUTE_EXPLAINED = 0.99
# A numeric column repeats a part of the date when it matches on this share of rows.
DATE_PART_SHARE = 0.99
# A grouping is nested in another when this share of the cells expected to hold
# NESTING_EXPECTED or more rows are empty.
NESTED_EMPTY_SHARE = 0.5
NESTING_EXPECTED = 5


@dataclass(frozen=True)
class GroupStats:
    """One group's values of the reading."""

    value: object
    label: str
    n: int
    mean: float
    median: float
    sd: float
    #: Share of the group's values that are not zero.
    nonzero: float = float("nan")


@dataclass(frozen=True)
class OrderedTrend:
    """A reading's rank correlation with the numeric levels of a grouping: a dose, a depth, a speed.

    An unordered test asks whether any level differs; this asks whether the
    reading rises or falls as the level does, which is the question for a dose
    or a distance and has more power when the answer is yes.
    """

    measure: str
    grouping: str
    #: Spearman's rank correlation between the level and the reading.
    rho: float
    p_value: float
    n: int
    levels: int
    adjusted_p: float | None = None
    tests: int = 1
    correction: str = HOLM

    @property
    def decisive_p(self) -> float:
        return self.p_value if self.adjusted_p is None else self.adjusted_p

    @property
    def explained(self) -> float:
        return self.rho**2

    @property
    def verdict(self) -> str:
        """"clear", "negligible" (a passing test on a trivial trend) or "none"."""
        if not self.decisive_p < ALPHA:
            return "none"
        return "negligible" if abs(self.rho) < NEGLIGIBLE_TREND else "clear"

    @property
    def direction(self) -> str:
        return "rises" if self.rho > 0 else "falls"


@dataclass(frozen=True)
class GroupComparison:
    """A reading compared across the groups of one column.

    ``groups`` runs from the highest centre to the lowest: the mean, or the
    median when ``uses_ranks`` is set because the reading is skewed or capped
    (the mean again when most values are zero). ``named`` holds the positions
    of the two groups compared: the two there are, or the highest and lowest
    among groups of at least MIN_NAMED_GROUP values; ``difference``,
    ``hedges_g`` and ``superiority`` describe that pair.
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
    correction: str = HOLM
    #: A group has fewer than MIN_WELCH_GROUP values, so the test on means pools the variance.
    pooled: bool = False
    named: tuple[int, int] = (0, -1)
    #: Share of the reading's values that are zero.
    zero_share: float = 0.0
    #: Chi-squared p-value for the share of non-zero values across the groups, for a zero-heavy reading.
    presence_p: float | None = None
    trend: OrderedTrend | None = None
    #: Share of the table's rows with both the reading and the grouping recorded.
    coverage: float = 1.0

    @property
    def higher(self) -> GroupStats:
        return self.groups[self.named[0]]

    @property
    def lower(self) -> GroupStats:
        return self.groups[self.named[1]]

    @property
    def two_groups(self) -> bool:
        return len(self.groups) == 2

    @property
    def zero_heavy(self) -> bool:
        return self.zero_share >= ZERO_HEAVY

    @property
    def test(self) -> str:
        if self.uses_ranks:
            return "Mann-Whitney" if self.two_groups else "Kruskal-Wallis"
        return self._means_test

    @property
    def check(self) -> str:
        if self.uses_ranks:
            return self._means_test
        return "Mann-Whitney" if self.two_groups else "Kruskal-Wallis"

    @property
    def _means_test(self) -> str:
        if self.pooled:
            return "Student's t-test" if self.two_groups else "one-way ANOVA"
        return "Welch's t-test" if self.two_groups else "Welch ANOVA"

    @property
    def explained_name(self) -> str:
        return "ε²" if self.uses_ranks else "η²"

    @property
    def decisive_p(self) -> float:
        return self.p_value if self.adjusted_p is None else self.adjusted_p

    @property
    def check_decisive_p(self) -> float:
        """The check's p-value, adjusted by the same factor as the main test's."""
        return _adjusted_like(self.check_p_value, self.p_value, self.adjusted_p, self.tests)

    @property
    def trivial(self) -> bool:
        """Whether the gap is below the conventional "small" effect, however small its p-value."""
        if self.two_groups and not self.uses_ranks and np.isfinite(self.hedges_g):
            return abs(self.hedges_g) < SMALL_HEDGES
        return self.explained < NEGLIGIBLE_EXPLAINED

    @property
    def thin(self) -> bool:
        """Whether one of the two groups compared has fewer than MIN_NAMED_GROUP values."""
        return min(self.higher.n, self.lower.n) < MIN_NAMED_GROUP

    @property
    def verdict(self) -> str:
        """"clear"; "negligible" (a passing test on a trivial gap); "thin" (a passing test
        that rests on a group of fewer than MIN_NAMED_GROUP values); "disputed" (the main
        test passes and its check does not); or "none"."""
        return _verdict(self.decisive_p, self.check_decisive_p, self.trivial, self.thin)

    @property
    def total(self) -> int:
        return sum(group.n for group in self.groups)

    def centre(self, group: GroupStats) -> float:
        return group.median if self.uses_ranks and not self.zero_heavy else group.mean


SMALL_HEDGES = HEDGES_SIZES[-1][0]


def _adjusted_like(p: float, raw: float, adjusted: float | None, tests: int) -> float:
    """``p`` scaled by the factor that turned ``raw`` into ``adjusted``, so a check is judged alike."""
    if adjusted is None:
        return p
    factor = adjusted / raw if raw > 0 else float(tests)
    return float(min(1.0, p * max(factor, 1.0)))


def _verdict(p: float, check_p: float, trivial: bool, thin: bool) -> str:
    if not p < ALPHA:
        return "none"
    if trivial:
        return "negligible"
    if thin:
        return "thin"
    if not check_p < ALPHA:
        return "disputed"
    return "clear"


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
    #: The classical one-way ANOVA (Student's t-test for two groups), which pools the variance.
    anova_p: np.ndarray
    rank_p: np.ndarray
    eta2: np.ndarray
    epsilon2: np.ndarray
    skew: np.ndarray
    at_max: np.ndarray
    at_min: np.ndarray
    maximum: np.ndarray
    minimum: np.ndarray
    distinct: np.ndarray
    #: Values that are not zero, per group and column.
    nonzero: np.ndarray


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
    the test on means (with two groups it equals Welch's t-test), with the
    classical ANOVA beside it for groups too small for Welch's weights;
    Kruskal-Wallis is the test on ranks (with two groups the Mann-Whitney test,
    by the normal approximation with ties corrected). Group sums come from
    products with a one-hot group matrix, so the cost grows with rows times columns.
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
        ss_within = (deviation**2).sum(axis=0)
        anova_f = (ss_total - ss_within) / (k - 1) / (ss_within / (total - k))
        anova_p = stats.f.sf(anova_f, k - 1, total - k)

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
        anova_p=np.nan_to_num(anova_p, nan=1.0),
        rank_p=np.nan_to_num(rank_p, nan=1.0),
        eta2=np.nan_to_num(eta2, nan=0.0),
        epsilon2=np.clip(np.nan_to_num(epsilon2, nan=0.0), 0.0, 1.0),
        skew=np.nan_to_num(skew, nan=0.0),
        at_max=ranked.at_max,
        at_min=ranked.at_min,
        maximum=np.nanmax(values, axis=0),
        minimum=np.nanmin(values, axis=0),
        distinct=ranked.distinct,
        nonzero=member.T @ (present & (filled != 0)),
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


def _hedges(n1: float, n2: float, mean1: float, mean2: float, sd1: float, sd2: float) -> float:
    """Hedges' g between two groups from their summaries, or NaN when neither varies."""
    pooled = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    # Groups that barely vary inside themselves leave g undefined, not astronomically large.
    scale = max(abs(mean1), abs(mean2), 1.0)
    if not pooled > NO_SPREAD * scale:
        return float("nan")
    return float((mean1 - mean2) / pooled * (1 - 3 / (4 * (n1 + n2) - 9)))


def _pair(
    first: np.ndarray, second: np.ndarray, *, pooled: bool = False
) -> tuple[float, float, float, float, float]:
    """Difference in means with its 95% interval, Hedges' g, and the share of pairs ``first`` wins.

    The interval is Welch's, or uses the pooled variance when ``pooled`` is set
    because a group is too small to estimate its own.
    """
    n1, n2 = len(first), len(second)
    difference = float(first.mean() - second.mean())
    var1, var2 = float(first.var(ddof=1)), float(second.var(ddof=1))
    if pooled:
        shared = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        error, df = np.sqrt(shared * (1 / n1 + 1 / n2)), n1 + n2 - 2
    else:
        error = np.sqrt(var1 / n1 + var2 / n2)
        df = error**4 / ((var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)) if error > 0 else 1.0
    margin = float(stats.t.ppf(0.975, df) * error) if error > 0 else 0.0
    hedges = _hedges(n1, n2, float(first.mean()), float(second.mean()), np.sqrt(var1), np.sqrt(var2))
    ranks = stats.rankdata(np.concatenate([first, second]))
    superiority = (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)
    return difference, difference - margin, difference + margin, hedges, float(superiority)


@dataclass(frozen=True)
class _Judged:
    """What decides one column's comparison, read off the omnibus arrays without the raw values."""

    shape: str
    uses_ranks: bool
    pooled: bool
    zero_share: float
    #: Group indices from the highest centre to the lowest.
    order: list
    #: Positions in ``order`` of the two groups compared.
    named: tuple[int, int]
    p: float
    check_p: float
    explained: float
    trivial: bool
    thin: bool


def _judge(omnibus: _Omnibus, column: int, measure: str) -> _Judged:
    """Pick the test, the groups to name and the effect for one column of ``omnibus``.

    Ranks replace means for a skewed or capped reading, but a reading that is
    mostly zeros is still ranked and named by its means, since every median is
    0. The highest and lowest named are the extremes among groups with
    MIN_NAMED_GROUP values or more, when at least two have that many.
    """
    shape = _shape(omnibus, column, measure)
    uses_ranks = bool(shape)
    sizes = omnibus.n[:, column]
    present = float(sizes.sum())
    zero_share = 1 - float(omnibus.nonzero[:, column].sum()) / present if present else 0.0
    if omnibus.distinct[column] < 3:
        zero_share = 0.0
    by_mean = not uses_ranks or zero_share >= ZERO_HEAVY
    centre = omnibus.mean[:, column] if by_mean else omnibus.median[:, column]
    order = sorted(
        range(len(omnibus.levels)),
        key=lambda index: (centre[index], omnibus.mean_rank[index, column]),
        reverse=True,
    )
    large = [position for position, index in enumerate(order) if sizes[index] >= MIN_NAMED_GROUP]
    named = (large[0], large[-1]) if len(large) >= 2 else (0, len(order) - 1)
    pooled = bool(sizes.min() < MIN_WELCH_GROUP)
    means_p = float(omnibus.anova_p[column] if pooled else omnibus.welch_p[column])
    ranked = float(omnibus.rank_p[column])
    explained = float(omnibus.epsilon2[column] if uses_ranks else omnibus.eta2[column])
    top, bottom = order[named[0]], order[named[1]]
    trivial = explained < NEGLIGIBLE_EXPLAINED
    if len(order) == 2 and not uses_ranks:
        g = _hedges(
            sizes[top], sizes[bottom], omnibus.mean[top, column], omnibus.mean[bottom, column],
            omnibus.sd[top, column], omnibus.sd[bottom, column],
        )
        trivial = abs(g) < SMALL_HEDGES if np.isfinite(g) else trivial
    return _Judged(
        shape=shape,
        uses_ranks=uses_ranks,
        pooled=pooled,
        zero_share=zero_share,
        order=order,
        named=named,
        p=ranked if uses_ranks else means_p,
        check_p=means_p if uses_ranks else ranked,
        explained=explained,
        trivial=trivial,
        thin=bool(min(sizes[top], sizes[bottom]) < MIN_NAMED_GROUP),
    )


def _comparison(
    omnibus: _Omnibus,
    column: int,
    values: pd.Series,
    groups: pd.Series,
    measure: str,
    grouping: str,
    **extra,
) -> GroupComparison:
    judged = _judge(omnibus, column, measure)
    order = judged.order
    sizes = omnibus.n[:, column]
    summaries = tuple(
        GroupStats(
            value=omnibus.levels[index],
            label=group_label(grouping, omnibus.levels[index]),
            n=int(sizes[index]),
            mean=float(omnibus.mean[index, column]),
            median=float(omnibus.median[index, column]),
            sd=float(omnibus.sd[index, column]),
            nonzero=float(omnibus.nonzero[index, column] / sizes[index]),
        )
        for index in order
    )
    top, bottom = order[judged.named[0]], order[judged.named[1]]
    first = values[groups == omnibus.levels[top]].dropna().to_numpy(dtype=float)
    second = values[groups == omnibus.levels[bottom]].dropna().to_numpy(dtype=float)
    mean_difference, low, high, hedges, superiority = _pair(first, second, pooled=judged.pooled)
    zero_heavy = judged.zero_share >= ZERO_HEAVY
    medians = omnibus.median[:, column]
    presence = None
    if zero_heavy:
        presence = _rate_test(omnibus.nonzero[:, column], sizes)[1]
    return GroupComparison(
        measure=measure,
        grouping=grouping,
        groups=summaries,
        uses_ranks=judged.uses_ranks,
        shape=judged.shape,
        p_value=judged.p,
        check_p_value=judged.check_p,
        explained=judged.explained,
        difference=(
            float(medians[top] - medians[bottom])
            if judged.uses_ranks and not zero_heavy
            else mean_difference
        ),
        low=low,
        high=high,
        hedges_g=hedges,
        superiority=superiority,
        pooled=judged.pooled,
        named=(judged.named[0], judged.named[1] % len(order)),
        zero_share=judged.zero_share,
        presence_p=presence,
        **extra,
    )


CHI_SQUARED = "chi-squared test"
FISHER = "Fisher's exact test"


def _rate_test(events: np.ndarray, sizes: np.ndarray) -> tuple[str, float, float]:
    """Test whether a share differs across groups: the test's name, its p-value and Cramér's V.

    Pearson's chi-squared test on the groups-by-outcome table
    (``scipy.stats.chi2_contingency``), or, for two groups with an expected
    count below 5, where chi-squared is unreliable, Fisher's exact test
    (``scipy.stats.fisher_exact``).
    """
    events = np.asarray(events, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    table = np.column_stack([events, sizes - events])[sizes > 0].round().astype(int)
    if table.shape[0] < 2 or (table.sum(axis=0) == 0).any():
        return CHI_SQUARED, 1.0, 0.0
    statistic, p, _, expected = stats.chi2_contingency(table, correction=False)
    cramers_v = float(np.sqrt(statistic / table.sum()))
    if table.shape[0] == 2 and expected.min() < 5:
        return FISHER, float(stats.fisher_exact(table)[1]), cramers_v
    return CHI_SQUARED, float(p), cramers_v


# Values that mark the "yes" and "no" of a two-valued text outcome.
YES_WORDS = frozenset({"yes", "y", "true", "positive", "present", "pass"})
NO_WORDS = frozenset({"no", "n", "false", "negative", "absent", "fail"})
# Below this Cramér's V an association is too small to matter (Cohen's "small" is 0.1).
NEGLIGIBLE_ASSOCIATION = 0.1


def outcome_event(values: pd.Series) -> object | None:
    """The value a yes/no column counts (1, True or "yes"), or None when the column is not one."""
    present = values.dropna()
    levels = list(present.unique())
    if len(levels) != 2:
        return None
    if is_bool_dtype(present):
        return True
    if is_numeric_dtype(present):
        return 1 if set(levels) == {0, 1} else None
    words = {str(level).strip().casefold(): level for level in levels}
    yes = [level for word, level in words.items() if word in YES_WORDS]
    no = [level for word, level in words.items() if word in NO_WORDS]
    return yes[0] if len(yes) == 1 and len(no) == 1 else None


@dataclass(frozen=True)
class RateGroup:
    value: object
    label: str
    n: int
    events: int

    @property
    def rate(self) -> float:
        return self.events / self.n if self.n else float("nan")


@dataclass(frozen=True)
class RateComparison:
    """How often a yes/no outcome is "yes" in each group of another column.

    ``groups`` runs from the highest rate to the lowest. ``named`` holds the
    positions of the two groups compared: the two there are, or the highest and
    lowest among groups of at least MIN_NAMED_GROUP rows. ``difference`` is
    their risk difference, with Newcombe's 95% interval in ``low`` and ``high``.
    """

    outcome: str
    event: object
    grouping: str
    groups: tuple[RateGroup, ...]
    test: str
    p_value: float
    cramers_v: float
    difference: float
    low: float
    high: float
    named: tuple[int, int] = (0, -1)
    adjusted_p: float | None = None
    tests: int = 1
    correction: str = HOLM

    @property
    def higher(self) -> RateGroup:
        return self.groups[self.named[0]]

    @property
    def lower(self) -> RateGroup:
        return self.groups[self.named[1]]

    @property
    def event_label(self) -> str:
        return group_label(self.outcome, self.event)

    @property
    def decisive_p(self) -> float:
        return self.p_value if self.adjusted_p is None else self.adjusted_p

    @property
    def verdict(self) -> str:
        """"clear", "negligible", "thin" or "none", as for ``GroupComparison``."""
        thin = min(self.higher.n, self.lower.n) < MIN_NAMED_GROUP
        trivial = self.cramers_v < NEGLIGIBLE_ASSOCIATION
        return _verdict(self.decisive_p, self.decisive_p, trivial, thin)


def compare_rates(dataframe: pd.DataFrame, outcome: str, grouping: str) -> RateComparison | None:
    """Compare how often the yes/no ``outcome`` is "yes" across the groups of ``grouping``.

    Each group gets its rate and n; the test is chi-squared, or Fisher's exact
    test for two small groups; the highest and lowest rates get their risk
    difference with Newcombe's 95% interval
    (``statsmodels.stats.proportion.confint_proportions_2indep``). Returns None
    when ``outcome`` is not a yes/no column or fewer than two groups have two rows.
    """
    if outcome == grouping or outcome not in dataframe.columns or grouping not in dataframe.columns:
        return None
    event = outcome_event(dataframe[outcome])
    if event is None:
        return None
    data = dataframe[[outcome, grouping]].dropna()
    hits = (data[outcome] == event).groupby(data[grouping], observed=True).agg(["size", "sum"])
    hits = hits[hits["size"] >= MIN_GROUP_SIZE]
    if len(hits) < 2:
        return None
    rates = hits["sum"] / hits["size"]
    hits = hits.loc[rates.sort_values(ascending=False, kind="stable").index]
    groups = tuple(
        RateGroup(value=value, label=group_label(grouping, value), n=int(row["size"]), events=int(row["sum"]))
        for value, row in hits.iterrows()
    )
    large = [position for position, group in enumerate(groups) if group.n >= MIN_NAMED_GROUP]
    named = (large[0], large[-1]) if len(large) >= 2 else (0, len(groups) - 1)
    test, p, cramers_v = _rate_test(hits["sum"].to_numpy(), hits["size"].to_numpy())
    first, second = groups[named[0]], groups[named[1]]
    low, high = confint_proportions_2indep(
        first.events, first.n, second.events, second.n, method="newcomb", compare="diff"
    )
    return RateComparison(
        outcome=outcome,
        event=event,
        grouping=grouping,
        groups=groups,
        test=test,
        p_value=p,
        cramers_v=cramers_v,
        difference=first.rate - second.rate,
        low=float(low),
        high=float(high),
        named=named,
    )


def _points(share: float) -> str:
    return f"{share * 100:.1f}"


def describe_rates(comparison: RateComparison) -> str:
    """State whether the share of "yes" differs across the groups, with each group's rate and n."""
    c = comparison
    high, low = c.higher, c.lower
    test = f"{c.test}, {_adjusted_p_text(c.p_value, c.adjusted_p, c.correction)}"
    effect = f"Cramér's V = {c.cramers_v:.2f}"
    gap = (
        f"a difference of {_points(c.difference)} percentage points (95% CI {_points(c.low)} to "
        f"{_points(c.high)})"
    )
    share = f"The share of rows with {c.event_label}"
    rates = (
        f"{high.rate:.1%} in {high.label} (n = {high.n:,}) against {low.rate:.1%} in {low.label} "
        f"(n = {low.n:,})"
    )
    verdict = c.verdict
    if verdict == "clear":
        if len(c.groups) == 2:
            text = f"{share} is {rates}: {gap}; {test}; {effect}."
        else:
            text = (
                f"{share} differs across {c.grouping}: highest {rates}, {gap}; {test} across all "
                f"{len(c.groups)} groups; {effect}."
            )
    elif verdict == "negligible":
        text = f"{share} differs across {c.grouping} only by a negligible amount: {rates} ({test}; {effect})."
    elif verdict == "thin":
        small = high if high.n <= low.n else low
        text = (
            f"{share} is {rates} ({test}), but {small.n:,} rows in {small.label} are too few to call "
            "the difference clear."
        )
    else:
        text = f"{share} shows no clear difference across {c.grouping}: {rates} ({test}; {effect})."
    small = [group.label for group in c.groups if group.n < MIN_NAMED_GROUP]
    if small and verdict != "thin":
        text += f" Not ranked, with fewer than {MIN_NAMED_GROUP} rows: {list_names(small)}."
    return text


def rate_calculation(comparison: RateComparison) -> str:
    """The calculation behind ``describe_rates``."""
    c = comparison
    function = "scipy.stats.fisher_exact" if c.test == FISHER else "scipy.stats.chi2_contingency"
    steps = [
        f"rows with {c.event_label} ÷ rows per {c.grouping}",
        f"{c.test} on the {len(c.groups)}×2 table ({function}), Cramér's V",
        "risk difference of the highest and lowest rate with Newcombe's 95% interval "
        "(statsmodels confint_proportions_2indep)",
    ]
    if c.adjusted_p is not None:
        method = "fdr_bh" if c.correction == FALSE_DISCOVERY else "holm"
        steps.append(f"p {ADJUSTED[c.correction]} for {c.tests} tests (statsmodels multipletests, {method})")
    return "; ".join(steps)


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
    two values are left out and named in ``dropped``; a group with fewer than
    MIN_WELCH_GROUP switches the test on means to the pooled-variance one, and
    groups with fewer than MIN_NAMED_GROUP are never named as the highest or
    lowest. A grouping of three or more numeric levels also gets a rank trend
    test in ``trend``. With ``entity`` (a subject or sample column) each entity
    counts once per group.
    """
    if measure == grouping or measure not in dataframe.columns or grouping not in dataframe.columns:
        return None
    readings, groups, fields = _testing_sample(dataframe, [measure], grouping, entity)
    comparison = _single(readings[measure], groups, measure, grouping, fields)
    if comparison is not None and ordered_levels(dataframe[grouping], grouping):
        comparison = replace(comparison, trend=_ordered_trend(readings[measure], groups, measure, grouping))
    return comparison


# A trend test needs this many values before its p-value means much.
MIN_TREND_VALUES = 10


def ordered_levels(values: pd.Series, name: str) -> bool:
    """Whether a grouping's levels are numbers on a scale, like a dose, a depth or a speed.

    Three or more numeric levels, under a name that does not mark labels:
    status 2 is not more of anything than status 1, nor plot 7 than plot 3.
    """
    if is_bool_dtype(values) or not is_numeric_dtype(values) or values.nunique(dropna=True) < 3:
        return False
    words = re.findall(r"[a-z]+", normalized_name(re.sub(r"\([^)]*\)|\[[^\]]*\]", "", str(name))))
    if words and words[-1] in LABEL_NAME_WORDS:
        return False
    return not (is_entity_name(name) or is_design_name(name))


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.corrcoef(stats.rankdata(first), stats.rankdata(second))[0, 1]
    return float(value) if np.isfinite(value) else 0.0


def _ordered_trend(values: pd.Series, groups: pd.Series, measure: str, grouping: str) -> OrderedTrend | None:
    """Spearman's rank correlation between a reading and the numeric levels it was grouped by."""
    data = pd.DataFrame({
        "value": pd.to_numeric(pd.Series(values.to_numpy()), errors="coerce"),
        "level": pd.to_numeric(pd.Series(groups.to_numpy()), errors="coerce"),
    }).dropna()
    if len(data) < MIN_TREND_VALUES or data["level"].nunique() < 3 or data["value"].nunique() < 2:
        return None
    rho = _spearman(data["level"].to_numpy(), data["value"].to_numpy())
    return OrderedTrend(
        measure=measure,
        grouping=grouping,
        rho=rho,
        p_value=correlation_p(rho, len(data)),
        n=len(data),
        levels=int(data["level"].nunique()),
    )


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm's step-down adjustment, which keeps the chance of any false finding at the stated level.

    Computed by ``statsmodels.stats.multitest.multipletests(method="holm")``.
    """
    p = np.asarray(p_values, dtype=float)
    return multipletests(p, method="holm")[1] if p.size else p


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, which keep the expected share of false findings among those named.

    A screen of thousands of readings expects some true differences, and Holm's
    guard against any false one would hide nearly all of them. Computed by
    ``statsmodels.stats.multitest.multipletests(method="fdr_bh")``.
    """
    p = np.asarray(p_values, dtype=float)
    return multipletests(p, method="fdr_bh")[1] if p.size else p


@dataclass(frozen=True)
class DifferenceScan:
    """Every reading compared across every grouping, with p-values adjusted for the number of tests."""

    #: The largest clear findings: well-covered ones first, then by share of variance explained.
    #: A finding is a clear difference, or a clear trend across ordered levels (``trend``).
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
    #: Pairs left out because the reading barely varies within each group: it
    #: describes the group (a reef's coordinates, a station's battery), not its members.
    attributes: tuple[tuple[str, str], ...] = ()
    #: Readings never compared: coordinates, which place a row, and compass directions,
    #: whose mean means nothing.
    placed: tuple[str, ...] = ()
    #: HOLM, or FALSE_DISCOVERY for a screen of more than MAX_SCAN_COLUMNS readings.
    correction: str = HOLM
    #: Tests with an unadjusted p-value below ALPHA, against ALPHA × tested expected by chance.
    passing: int = 0
    #: Trend tests among ``tested``, one for each reading against each ordered numeric grouping.
    trends: int = 0
    #: Reading and grouping pairs compared; ``tested`` adds the trend tests.
    compared: int = 0

    @property
    def sampled(self) -> bool:
        return self.rows > SCAN_ROWS


def _explained_share(values: pd.Series, groups: pd.Series) -> float:
    """Eta squared: the share of the values' variance between the groups' means."""
    data = pd.DataFrame({"value": values.to_numpy(dtype=float), "group": groups.to_numpy()}).dropna()
    total = float(((data["value"] - data["value"].mean()) ** 2).sum())
    if total == 0:
        return 0.0
    means = data.groupby("group", observed=True)["value"].transform("mean")
    return float(((means - data["value"].mean()) ** 2).sum()) / total
# Below this share of the table's rows a scan finding is listed after the better-covered ones.
WELL_COVERED = 0.2


@dataclass
class _Entry:
    """One reading against one grouping in a scan, built into a comparison only when reported."""

    explained: float
    p: float
    check_p: float
    trivial: bool
    thin: bool
    coverage: float
    #: A built comparison, or the omnibus, column position and fields that build one.
    item: object
    values: pd.Series
    groups: pd.Series
    #: (rho, p-value) of the trend test, for an ordered numeric grouping.
    trend: tuple[float, float] | None = None


def _trend_screen(block: pd.DataFrame, groups: pd.Series) -> dict[str, tuple[float, float]]:
    """Spearman's rho and its p-value for every column of ``block`` against the numeric levels in ``groups``.

    Ranks come from one sort per column over all its values, so with missing
    values rho is a close screen; a reported trend is recomputed exactly.
    """
    levels = pd.to_numeric(groups, errors="coerce").to_numpy(dtype=float)
    values = np.column_stack([levels, block.to_numpy(dtype=float)])
    rho, counts = pairwise_correlations(_rank_columns(values).ranks)
    found = {}
    for index, column in enumerate(block.columns, start=1):
        pairs, value = int(counts[0, index]), float(rho[0, index])
        present = ~np.isnan(values[:, index]) & ~np.isnan(levels)
        if pairs >= MIN_TREND_VALUES and np.isfinite(value) and np.unique(levels[present]).size >= 3:
            found[column] = (value, correlation_p(value, pairs))
    return found


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

    ``skip`` holds (reading, grouping) pairs reported elsewhere or related by
    construction. A grouping of three or more numeric levels also gets a trend
    test for each reading. A finding is judged as ``GroupComparison.verdict``
    judges one comparison, after the p-values are adjusted over every test run:
    by Holm, so the scan's chance of naming any difference that isn't there
    stays at 5%, or, for a screen of more than MAX_SCAN_COLUMNS readings, by
    Benjamini-Hochberg, so the expected share of false findings among those
    named stays at 5%. A reading that barely varies within the groups of a
    grouping is an attribute of the group and is not compared across it;
    coordinates and compass directions are never compared. A screen tests every
    reading up to SCREEN_CELLS cells, then its most variable ones. Very long
    tables are scanned on a fixed random sample of SCAN_ROWS rows, which still
    detects any difference large enough to report. Findings resting on under
    WELL_COVERED of the rows are listed after the others.
    """
    placed = tuple(
        reading for reading in readings if is_coordinate(reading) or is_angle(reading, dataframe[reading])
    )
    readings = [reading for reading in readings if reading not in placed]
    available = len(readings)
    rows = len(dataframe)
    if rows > SCAN_ROWS:
        dataframe = dataframe.sample(SCAN_ROWS, random_state=0)
    correction = FALSE_DISCOVERY if available > MAX_SCAN_COLUMNS else HOLM
    budget = max(MAX_SCAN_COLUMNS, SCREEN_CELLS // max(len(dataframe), 1))
    if available > budget:
        variance = dataframe[readings].var()
        kept = set(variance.nlargest(budget).index)
        readings = [reading for reading in readings if reading in kept]
    entries: list[_Entry] = []
    flagged: list[tuple[str, str]] = []
    recoded: list[tuple[str, str]] = []
    attributes: list[tuple[str, str]] = []
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
        recorded = dataframe[columns].notna().to_numpy() & dataframe[grouping].notna().to_numpy()[:, None]
        coverage = dict(zip(columns, recorded.mean(axis=0).tolist(), strict=True))
        block, groups, fields = _testing_sample(dataframe, columns, grouping, entity)
        used_entity = fields.get("entity", used_entity)
        sizes = groups.value_counts()
        kept = groups.isin(sizes[sizes >= MIN_GROUP_SIZE].index)
        block, groups = block[kept], groups[kept]
        if groups.nunique() < 2:
            continue
        trends = _trend_screen(block, groups) if ordered_levels(dataframe[grouping], grouping) else {}
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
                if omnibus.eta2[index] >= ATTRIBUTE_EXPLAINED:
                    attributes.append((column, grouping))
                    continue
                judged = _judge(omnibus, index, column)
                entries.append(_Entry(
                    explained=judged.explained,
                    p=judged.p,
                    check_p=judged.check_p,
                    trivial=judged.trivial,
                    thin=judged.thin,
                    coverage=coverage[column],
                    item=(omnibus, index, column, grouping, fields),
                    values=block[column],
                    groups=groups,
                    trend=trends.get(column),
                ))
        for column in columns:
            if column in regular:
                continue
            if _explained_share(block[column], groups) >= ATTRIBUTE_EXPLAINED:
                attributes.append((column, grouping))
                continue
            single = _single(block[column], groups, column, grouping, fields)
            if single is not None:
                entries.append(_Entry(
                    explained=single.explained,
                    p=single.p_value,
                    check_p=single.check_p_value,
                    trivial=single.trivial,
                    thin=single.thin,
                    coverage=coverage[column],
                    item=single,
                    values=block[column],
                    groups=groups,
                    trend=trends.get(column),
                ))
    if not entries:
        return None

    p_values = [entry.p for entry in entries]
    trend_at: dict[int, int] = {}
    for position, entry in enumerate(entries):
        if entry.trend is not None:
            trend_at[position] = len(p_values)
            p_values.append(entry.trend[1])
    tests = len(p_values)
    raw = np.array(p_values)
    adjusted = (benjamini_hochberg if correction == FALSE_DISCOVERY else holm_adjust)(raw)

    def strength(position: int) -> float:
        """The share of variance behind the entry's clear findings, or -1 when it has none."""
        entry = entries[position]
        found = []
        check = _adjusted_like(entry.check_p, entry.p, adjusted[position], tests)
        if _verdict(adjusted[position], check, entry.trivial, entry.thin) == "clear":
            found.append(entry.explained)
        if position in trend_at and adjusted[trend_at[position]] < ALPHA:
            rho = entry.trend[0]
            if abs(rho) >= NEGLIGIBLE_TREND:
                found.append(rho**2)
        return max(found, default=-1.0)

    strengths = [strength(position) for position in range(len(entries))]
    clear = [position for position in range(len(entries)) if strengths[position] >= 0]
    clear.sort(
        key=lambda position: (entries[position].coverage >= WELL_COVERED, strengths[position]), reverse=True
    )
    largest = max(range(len(entries)), key=lambda position: entries[position].explained)

    def build(position: int) -> GroupComparison:
        entry = entries[position]
        item = entry.item
        if not isinstance(item, GroupComparison):
            omnibus, column_index, column, grouping, fields = item
            item = _comparison(omnibus, column_index, entry.values, entry.groups, column, grouping, **fields)
        trend = None
        if position in trend_at:
            trend = _ordered_trend(entry.values, entry.groups, item.measure, item.grouping)
            if trend is not None:
                screened = entry.trend[1]
                trend = replace(
                    trend,
                    adjusted_p=_adjusted_like(trend.p_value, screened, adjusted[trend_at[position]], tests),
                    tests=tests,
                    correction=correction,
                )
        return replace(
            item,
            adjusted_p=float(adjusted[position]),
            tests=tests,
            correction=correction,
            trend=trend,
            coverage=entry.coverage,
        )

    return DifferenceScan(
        named=tuple(build(position) for position in clear[:limit]),
        largest=build(largest),
        tested=tests,
        clear=len(clear),
        readings=tuple(readings),
        groupings=tuple(groupings),
        entity=used_entity,
        flagged=tuple(flagged),
        recoded=tuple(recoded),
        available=available,
        rows=rows,
        attributes=tuple(attributes),
        placed=placed,
        correction=correction,
        passing=int((raw < ALPHA).sum()),
        trends=len(trend_at),
        compared=len(entries),
    )


def variance_explained(
    dataframe: pd.DataFrame, measure: str, groupings: list[str], *, entity: str | None = None
) -> tuple[tuple[str, float], ...]:
    """Share of the measure's variance each grouping explains on its own (eta squared), largest first.

    With ``entity``, each entity counts once per group, as in ``compare_groups``,
    so the shares match the comparisons beside them.
    """
    shares = []
    for grouping in groupings:
        readings, groups, _ = _testing_sample(dataframe, [measure], grouping, entity)
        data = pd.DataFrame({"value": readings[measure].to_numpy(), "group": groups.to_numpy()}).dropna()
        if data["group"].nunique() < 2:
            continue
        values = data["value"].astype(float)
        total = float(((values - values.mean()) ** 2).sum())
        if total == 0:
            continue
        means = values.groupby(data["group"], observed=True).transform("mean")
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
            couples = _distinct(cells * (levels[first] * levels[second]) + pairs)
            if couples == _distinct(cells) == _distinct(pairs):
                found[grouping] = (first, second)
                break
    return found


def _distinct(codes: np.ndarray) -> int:
    """How many different non-negative integer codes an array holds, by counting rather than sorting."""
    return int(np.count_nonzero(np.bincount(codes))) if codes.size else 0


def date_part(values: pd.Series, dates: pd.Series) -> str | None:
    """Which part of the date a numeric column repeats ("month" for 1-12 matching each row's date), or None.

    A month or year column beside the date it came from is the date again:
    grouping by it shows the seasonal cycle or the trend, not a factor.
    """
    numbers = pd.to_numeric(values, errors="coerce")
    stamps = pd.to_datetime(dates, errors="coerce")
    both = numbers.notna() & stamps.notna()
    if int(both.sum()) < MIN_CORRELATION_PAIRS:
        return None
    numbers, stamps = numbers[both], stamps[both].dt
    parts = (
        ("year", stamps.year),
        ("year", stamps.year % 100),
        ("quarter", stamps.quarter),
        ("month", stamps.month),
        ("week", stamps.isocalendar().week.astype(int)),
        ("day", stamps.day),
        ("weekday", stamps.dayofweek),
        ("weekday", stamps.dayofweek + 1),
        ("hour", stamps.hour),
    )
    for name, part in parts:
        if float((numbers == part).mean()) >= DATE_PART_SHARE and part.nunique() > 1:
            return name
    return None


def nested_within(dataframe: pd.DataFrame, finer: str, coarser: str) -> bool:
    """Whether the levels of ``finer`` each occur with only some levels of ``coarser``.

    Concentrations 3 to 12 used only with the blue dyes and 15 to 25 only with
    the yellow ones repeat the dye difference: across such a grouping the two
    can't be told apart. A cell counts as missing only where enough rows were
    expected in it, so thin data is not mistaken for nesting.
    """
    data = dataframe[[finer, coarser]].dropna()
    if data.empty:
        return False
    rows, columns = pd.factorize(data[finer])[0], pd.factorize(data[coarser])[0]
    height, width = int(rows.max()) + 1, int(columns.max()) + 1
    if width < 2 or height <= width:
        return False
    table = np.bincount(rows * width + columns, minlength=height * width).reshape(height, width).astype(float)
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    testable = expected >= NESTING_EXPECTED
    if testable.sum() < table.shape[0]:
        return False
    return bool(((table == 0) & testable).sum() >= NESTED_EMPTY_SHARE * testable.sum())


# --- Groupings set by the readings --------------------------------------------

# A grouping is cut from a reading when the best set of cut points puts this
# share of rows in their own group's band (a value rounded onto a cut may sit
# on the wrong side) ...
RULE_SHARE = 0.995
# ... and neighbouring bands meet: no more than one step of the reading, or
# this share of its spread, lies between them. Groups that merely differ a
# lot leave a wider gap.
RULE_GAP = 0.02
RULE_MIN_ROWS = 20
RULE_MIN_VALUES = 10
RULE_MIN_BAND = 5
# Pooled bands that place less than this share of the rows outside the largest
# group are not worth trying within each level of another grouping.
RULE_TRY_WITHIN = 0.5
# Rows a likely rule is screened on before it is checked on every row.
RULE_SCREEN_ROWS = 1_000
# A grouping that names the largest of several columns does so on this share of rows.
LARGEST_SHARE = 0.9


@dataclass(frozen=True)
class GroupingRule:
    """A grouping that the readings set, row by row, rather than one measured beside them.

    "bands": each level is a range of one reading (sizes of 1-10, 11-50 and
    51-100 acres). "cut-offs": the bands move with the level of another
    grouping (anaemic below 12 g/dL for women and below 13.5 for men).
    "largest": each level names whichever of several columns is largest on
    that row. Comparing those readings across the grouping only restates how
    it was made.
    """

    grouping: str
    kind: str
    readings: tuple[str, ...]
    share: float
    rows: int
    by: str | None = None
    #: For "bands": (level, lowest, highest) from the lowest band up.
    bands: tuple[tuple[object, float, float], ...] = ()
    #: For "cut-offs" between two levels: (level of ``by``, lower level, the cut).
    cuts: tuple[tuple[object, object, float], ...] = ()
    #: The step the reading was recorded at, so cuts print as it does (12.0, not 12).
    step: float = 0.0

    def describe(self) -> str:
        share = _share_text(self.share)
        if self.kind == "largest":
            return f"{self.grouping} names the largest of {_joined(self.readings)} on {share} of rows"
        reading = self.readings[0]

        def number(value: float) -> str:
            decimals = _decimals(self.step)
            if decimals is None:
                return format_number(value, reading, compact=False)
            return f"{value:,.{decimals}f}"

        if self.kind == "bands":
            spans = []
            for index, (level, low, high) in enumerate(self.bands):
                label = group_label(self.grouping, level)
                if index == 0:
                    spans.append(f"{label} up to {number(high)}")
                elif index == len(self.bands) - 1:
                    spans.append(f"{label} from {number(low)}")
                else:
                    spans.append(f"{label} {number(low)} to {number(high)}")
            return f"{self.grouping} is set by {reading}: {_joined(spans)}, on {share} of rows"
        lower = {lower for _, lower, _ in self.cuts}
        if self.cuts and len(lower) == 1:
            cuts = _joined(
                [f"below {number(cut)} for {group_label(self.by, level)}" for level, _, cut in self.cuts]
            )
            label = group_label(self.grouping, next(iter(lower)))
            return f"{self.grouping} is set by {reading}: {label} {cuts}, on {share} of rows"
        return f"{self.grouping} is set by {reading}, in bands that differ by {self.by}, on {share} of rows"


def _decimals(step: float) -> int | None:
    """The decimals a value recorded at this step is written with: 1 for 0.1, 0 for whole numbers."""
    for places in range(7):
        scaled = step * 10**places
        if scaled > 0 and abs(scaled - round(scaled)) <= 1e-3 * scaled:
            return places
    return None


@dataclass(frozen=True)
class _BandFit:
    share: float
    #: (level, lowest, highest) of the rows each band places correctly, lowest band first.
    bands: tuple[tuple[object, float, float], ...]
    #: Whether every pair of neighbouring bands meets.
    meets: bool
    step: float = 0.0


def _band_fit(x: np.ndarray, codes: np.ndarray, labels: list) -> _BandFit | None:
    """The cut points that put the most rows in their own group's band, with groups ordered by median.

    ``x`` and ``codes`` hold complete rows only; ``codes`` index ``labels``.
    A dynamic programme over the sorted values finds the cuts in one pass
    per group; cuts fall only between distinct values, so tied values stay
    together.
    """
    present = np.flatnonzero(np.bincount(codes, minlength=len(labels)))
    k = present.size
    if k < 2:
        return None
    medians = np.array([np.median(x[codes == code]) for code in present])
    order = present[np.argsort(medians, kind="stable")]
    rank = np.zeros(len(labels), dtype=int)
    rank[order] = np.arange(k)
    sorting = np.argsort(x, kind="stable")
    x = x[sorting]
    ranked = rank[codes[sorting]]
    n = len(x)
    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, n]
    cumulative = np.zeros((k, n + 1))
    cumulative[:, 1:] = np.cumsum(ranked[None, :] == np.arange(k)[:, None], axis=1)
    counts = cumulative[:, starts]
    best = counts[0]
    choices = []
    positions = np.arange(len(starts))
    for level in range(1, k):
        offset = best - counts[level]
        running = np.maximum.accumulate(offset)
        choices.append(np.maximum.accumulate(np.where(offset == running, positions, 0)))
        best = running + counts[level]
    cuts = [len(starts) - 1]
    for choice in reversed(choices):
        cuts.append(int(choice[cuts[-1]]))
    cuts.append(0)
    cuts.reverse()
    share = float(best[-1]) / n
    bands, sizes = [], []
    for level in range(k):
        rows = slice(starts[cuts[level]], starts[cuts[level + 1]])
        placed = x[rows][ranked[rows] == level]
        if not placed.size:
            return _BandFit(share, (), False)
        bands.append((labels[order[level]], float(placed.min()), float(placed.max())))
        sizes.append(placed.size)
    spread = _spread(x)
    step = _step(x, spread)
    allowed = max(step * (1 + 1e-3), RULE_GAP * spread)
    # A band of a handful of rows can't show where its edge lies, so only
    # neighbours that both have RULE_MIN_BAND rows are held to meeting.
    checked = [index for index in range(k - 1) if min(sizes[index], sizes[index + 1]) >= RULE_MIN_BAND]
    meets = bool(checked) and all(bands[index + 1][1] - bands[index][2] <= allowed for index in checked)
    return _BandFit(share, tuple(bands), meets, step)


def _screened(rows: np.ndarray) -> np.ndarray:
    """At most RULE_SCREEN_ROWS of the row positions, spread evenly."""
    if len(rows) <= RULE_SCREEN_ROWS:
        return rows
    return rows[np.linspace(0, len(rows) - 1, RULE_SCREEN_ROWS).astype(int)]


@dataclass(frozen=True)
class _Codes:
    """A grouping column as integer codes (-1 where missing) and the labels they index."""

    codes: np.ndarray
    labels: list


def _as_codes(values: pd.Series) -> _Codes:
    codes, labels = pd.factorize(values, sort=True)
    return _Codes(codes.astype(np.int64), list(labels))


def _reading_rule(
    x: np.ndarray,
    reading: str,
    grouping: str,
    groups: _Codes,
    others: dict[str, _Codes],
    needed: float = RULE_SHARE,
) -> GroupingRule | None:
    """Whether ``grouping`` is cut from ``reading`` on ``needed`` of rows, overall or at moving cut-offs."""
    rows = np.flatnonzero(~np.isnan(x) & (groups.codes >= 0))
    if len(rows) < RULE_MIN_ROWS:
        return None
    sample = _screened(rows)
    screen = _band_fit(x[sample], groups.codes[sample], groups.labels)
    # Putting every row in the largest group already places that group's
    # share, so the screen asks how much of the rest the bands place.
    base = float(np.bincount(groups.codes[sample]).max()) / len(sample)
    if screen is None or base >= 1 or (screen.share - base) / (1 - base) < RULE_TRY_WITHIN:
        return None
    if screen.share >= needed - 0.01:
        fit = _band_fit(x[rows], groups.codes[rows], groups.labels) if len(rows) > len(sample) else screen
        if fit is not None and fit.share >= needed and fit.meets:
            return GroupingRule(
                grouping, "bands", (reading,), fit.share, len(rows), bands=fit.bands, step=screen.step
            )
    for by, split in others.items():
        if by in (reading, grouping):
            continue
        both = rows[split.codes[rows] >= 0]
        if not 2 <= np.unique(split.codes[both]).size <= MAX_SCAN_LEVELS:
            continue
        screened = _cut_offs(x, groups, split, _screened(both))
        if screened is None or screened[0] < needed - 0.01:
            continue
        found = _cut_offs(x, groups, split, both) if len(both) > RULE_SCREEN_ROWS else screened
        if found is not None and found[0] >= needed:
            return GroupingRule(
                grouping, "cut-offs", (reading,), found[0], len(both), by=by, cuts=found[1], step=screen.step
            )
    return None


def _cut_offs(
    x: np.ndarray, groups: _Codes, split: _Codes, rows: np.ndarray
) -> tuple[float, tuple[tuple[object, object, float], ...]] | None:
    """Bands of ``x`` fitted within each level of ``split``: the share of rows placed, and two-level cuts.

    None unless the bands meet in every level and at least two levels hold
    two groups to separate.
    """
    placed, compared, cuts = 0.0, 0, []
    for level in np.unique(split.codes[rows]):
        part = rows[split.codes[rows] == level]
        sizes = np.bincount(groups.codes[part])
        if (sizes >= 3).sum() < 2:
            placed += float(sizes.max())
            continue
        fit = _band_fit(x[part], groups.codes[part], groups.labels)
        if fit is None or not fit.meets:
            return None
        placed += fit.share * len(part)
        compared += 1
        if len(fit.bands) == 2:
            cuts.append((split.labels[level], fit.bands[0][0], fit.bands[1][1]))
    if compared < 2:
        return None
    return placed / len(rows), tuple(cuts) if len(cuts) == compared else ()


def _match_key(name: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _largest_rule(frame: pd.DataFrame, grouping: str, columns: list[str]) -> GroupingRule | None:
    """Whether each level of ``grouping`` names the column that is largest on its row."""
    keys = {_match_key(column): column for column in columns if column != grouping}
    levels = frame[grouping].dropna().unique()
    named = {level: keys[_match_key(level)] for level in levels if _match_key(level) in keys}
    if len(set(named.values())) < 2:
        return None
    rows = frame[frame[grouping].isin(list(named))]
    block = rows[list(dict.fromkeys(named.values()))].apply(pd.to_numeric, errors="coerce")
    recorded = block.notna().any(axis=1)
    rows, block = rows[recorded], block[recorded]
    if len(rows) < RULE_MIN_ROWS:
        return None
    top = block.max(axis=1).to_numpy()
    chosen = np.full(len(rows), np.nan)
    labels = rows[grouping].to_numpy()
    for level, column in named.items():
        mask = labels == level
        chosen[mask] = block[column].to_numpy(dtype=float)[mask]
    share = float(np.mean(chosen == top))
    if share < LARGEST_SHARE:
        return None
    used = tuple(column for column in columns if column in set(named.values()))
    return GroupingRule(grouping, "largest", used, share, len(rows))


def _in_stretches(dataframe: pd.DataFrame, grouping: str, date: str | None) -> bool:
    """Whether each level of ``grouping`` covers one unbroken stretch of rows or of time.

    Phases of a run and logs stacked one after another are such stretches,
    and a reading that drifts through them, or that another instrument
    records on another scale, can band by stretch without being what they
    were cut from.
    """
    orders = [dataframe[grouping]]
    if date is not None and date in dataframe.columns:
        orders.append(dataframe[[date, grouping]].dropna().sort_values(date, kind="stable")[grouping])
    for values in orders:
        codes = pd.factorize(values.dropna())[0]
        if codes.size >= 2 and 1 + int(np.count_nonzero(codes[1:] != codes[:-1])) <= 2 * int(codes.max() + 1):
            return True
    return False


def grouping_rules(
    dataframe: pd.DataFrame,
    readings: list[str],
    groupings: list[str],
    *,
    date: str | None = None,
) -> tuple[GroupingRule, ...]:
    """The groupings that the readings set row by row: bands of a reading, or the largest of several.

    A grouping cut from a reading separates it perfectly by definition, so
    the brief leaves such pairs out of every comparison and states the rule
    instead. A grouping whose levels are unbroken stretches of rows or of
    time (phases of a run, stacked logs) can separate a reading almost
    perfectly for other reasons, so it must separate it on every row. Long
    tables are checked on the same fixed sample of SCAN_ROWS rows the
    comparisons use.
    """
    groupings = [grouping for grouping in dict.fromkeys(groupings) if grouping in dataframe.columns]
    if not groupings:
        return ()
    frame = dataframe.sample(SCAN_ROWS, random_state=0) if len(dataframe) > SCAN_ROWS else dataframe
    readings = [reading for reading in readings if reading in frame.columns]
    if len(readings) > MAX_SCAN_COLUMNS:
        # As in the comparisons themselves, a very wide table is checked on its most variable readings.
        kept = set(frame[readings].var().nlargest(MAX_SCAN_COLUMNS).index)
        readings = [reading for reading in readings if reading in kept]
    codes = {grouping: _as_codes(frame[grouping]) for grouping in groupings}
    matrix = frame[readings].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # A reading of a few values is a code itself, not something to cut into bands.
    values = {
        reading: matrix[:, index]
        for index, reading in enumerate(readings)
        if np.unique(matrix[~np.isnan(matrix[:, index]), index]).size >= RULE_MIN_VALUES
    }
    rules: list[GroupingRule] = []
    for grouping in groupings:
        largest = _largest_rule(frame, grouping, readings)
        if largest is not None:
            rules.append(largest)
        if not 2 <= len(codes[grouping].labels) <= MAX_SCAN_LEVELS:
            continue
        needed = 1.0 if _in_stretches(dataframe, grouping, date) else RULE_SHARE
        others = {
            other: codes[other]
            for other in groupings
            if other != grouping and 2 <= len(codes[other].labels) <= MAX_SCAN_LEVELS
        }
        for reading in values:
            if reading == grouping or (largest is not None and reading in largest.readings):
                continue
            rule = _reading_rule(values[reading], reading, grouping, codes[grouping], others, needed)
            if rule is not None:
                rules.append(rule)
    return tuple(rules)


def entity_column(dataframe: pd.DataFrame, roles: ColumnRoles) -> str | None:
    """The column naming the unit each row belongs to, when some unit has several rows.

    The identifier, or a column named like one ("Subject", "Patient ID"), whose
    values can name units: text, or whole-number codes that repeat. A column
    of unique values describes one row each and needs no reduction, and a
    decimal reading named like a unit ("NO", nitric oxide) or a column that
    could be the measure is a measurement, not a unit.
    """
    measures = {roles.measure, roles.date, *(getattr(roles, "measure_alternatives", ()) or ())}
    candidates = [roles.identifier, *(column for column in dataframe.columns if is_entity_name(column))]
    for column in dict.fromkeys(column for column in candidates if column):
        if column not in dataframe.columns or column in measures:
            continue
        values = dataframe[column].dropna()
        if values.nunique() >= 2 and bool(values.duplicated().any()) and _names_units(values):
            return column
    return None


def _names_units(values: pd.Series) -> bool:
    """Whether a column's values can label units: text, or whole numbers far fewer than its rows."""
    if is_bool_dtype(values) or not is_numeric_dtype(values):
        return True
    numbers = values.to_numpy(dtype=float)
    return bool(np.all(numbers % 1 == 0)) and values.nunique() <= 0.5 * len(values)


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


def _adjusted_p_text(p: float, adjusted: float | None, correction: str) -> str:
    if adjusted is None:
        return format_p(p)
    return f"{ADJUSTED[correction]} {format_p(adjusted)}"


def _test_text(comparison: GroupComparison) -> str:
    adjusted = _adjusted_p_text(comparison.p_value, comparison.adjusted_p, comparison.correction)
    return f"{comparison.test}, {adjusted}"


def _tied(comparison: GroupComparison, centre: float) -> list[str]:
    """The groups whose centre equals ``centre``, so a tie is not reported as one group."""
    return [group.label for group in comparison.groups if np.isclose(comparison.centre(group), centre)]


def centre_word(comparison: GroupComparison, *, plural: bool = False) -> str:
    """What each group's figure is: "mean", "median", or "mean of per-Bird medians" after a reduction.

    With an entity, each group's figure summarises one value per entity, which
    is not the plain mean of its rows that a table or a KPI shows.
    """
    word = "median" if comparison.uses_ranks and not comparison.zero_heavy else "mean"
    word += "s" if plural else ""
    if comparison.entity:
        word += f" of per-{comparison.entity} {comparison.entity_summary}s"
    return word


def _unranked(comparison: GroupComparison, column_values: pd.Series | None) -> str:
    """Name the groups too small to rank whose figures sit beyond the named highest and lowest."""
    c = comparison
    top, bottom = c.centre(c.higher), c.centre(c.lower)
    beyond = [
        group for group in c.groups
        if group.n < MIN_NAMED_GROUP and (c.centre(group) > top or c.centre(group) < bottom)
    ]
    if not beyond or c.thin:
        return ""
    word = centre_word(c)
    named = [
        f"{group.label} ({word} "
        f"{format_number(c.centre(group), c.measure, column_values=column_values)}, n = {group.n:,})"
        for group in beyond[:3]
    ]
    if len(beyond) > 3:
        named.append(f"{len(beyond) - 3} more")
    return f" Not ranked, with fewer than {MIN_NAMED_GROUP} values: {list_names(named, limit=len(named))}."


def _zero_text(comparison: GroupComparison) -> str:
    """For a reading that is mostly zeros, the share of non-zero values in each group and its test."""
    c = comparison
    if not c.zero_heavy or c.presence_p is None:
        return ""
    candidates = [group for group in c.groups if group.n >= MIN_NAMED_GROUP] or list(c.groups)
    ranked = sorted(candidates, key=lambda group: group.nonzero)
    low, high = ranked[0], ranked[-1]
    text = (
        f" {c.zero_share:.0%} of {c.measure} values are 0, so every median is 0 and the groups are "
        f"compared by their means and by the share of values that are not 0: from {low.nonzero:.1%} "
        f"({low.label}) to {high.nonzero:.1%} ({high.label}) (chi-squared test, {format_p(c.presence_p)})."
    )
    if c.uses_ranks:
        text += f" The test on all values uses ranks because {c.shape}."
    return text


def trend_sentence(trend: OrderedTrend) -> str:
    """Whether the reading rises or falls across the ordered levels of its grouping."""
    figures = (
        f"Spearman ρ = {trend.rho:.2f}, "
        f"{_adjusted_p_text(trend.p_value, trend.adjusted_p, trend.correction)}, n = {trend.n:,}"
    )
    if trend.verdict == "clear":
        return (
            f"{trend.measure} {trend.direction} as {trend.grouping} increases across its {trend.levels} "
            f"levels ({figures})."
        )
    if trend.verdict == "negligible":
        return (
            f"Across the {trend.levels} levels of {trend.grouping}, {trend.measure} {trend.direction} "
            f"only by a negligible amount ({figures})."
        )
    return (
        f"Across the {trend.levels} ordered levels of {trend.grouping}, {trend.measure} shows no clear "
        f"trend ({figures})."
    )


def _disagreement(comparison: GroupComparison) -> str:
    c = comparison
    return (
        f"{c.test} gives {format_p(c.decisive_p)} but {c.check} gives {format_p(c.check_decisive_p)}, "
        f"and when the two tests disagree the gap is not called clear ({effect_text(c)})"
    )


def describe_comparison(comparison: GroupComparison, column_values: pd.Series | None = None) -> str:
    """State whether the reading differs across the groups, with the sizes, test and effect behind it.

    A difference the test can't tell from noise is written as "no clear
    difference", never as a ranking. So is one whose check disagrees, or one
    resting on a group of fewer than MIN_NAMED_GROUP values. One that passes
    but falls short of a small effect is called negligible.
    """
    c = comparison
    word = centre_word(c)
    high, low = c.higher, c.lower
    first, second = format_values_apart(
        c.centre(high), c.centre(low), column=c.measure, column_values=column_values
    )
    sizes = f"n = {high.n:,} and {low.n:,}"
    verdict = c.verdict
    small = high if high.n <= low.n else low
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
            text += f" {c.check} agrees ({format_p(c.check_p_value)})."
        elif verdict == "negligible":
            text = (
                f"{c.measure} differs between {high.label} and {low.label} only by a negligible "
                f"amount: {quoted}, {effect_text(c)} ({_test_text(c)})."
            )
            if c.total > LARGE_SAMPLE:
                text += " With this many records even a trivial gap passes the test."
        elif verdict == "thin":
            text = (
                f"{c.measure} looks higher in {high.label} than in {low.label} ({quoted}; {_test_text(c)}), "
                f"but {small.n:,} values in {small.label} are too few to call the difference clear."
            )
        elif verdict == "disputed":
            text = (
                f"{c.measure} shows no clear difference between {high.label} and {low.label}: {quoted}. "
                f"{_disagreement(c)}."
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
        words = centre_word(c, plural=True)
        span = (
            f"{words} run from {second} ({low.label}, n = {low.n:,}) to {first} "
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
            if c.total > LARGE_SAMPLE:
                text += " With this many records even a trivial gap passes the test."
        elif verdict == "thin":
            text = (
                f"{c.measure} differs across {c.grouping} by the test ({_test_text(c)}; {effect_text(c)}), "
                f"but fewer than two groups have {MIN_NAMED_GROUP} or more values, too few to call the "
                f"difference clear: {span}."
            )
        elif verdict == "disputed":
            text = (
                f"{c.measure} shows no clear difference across the {len(c.groups)} {c.grouping} groups: "
                f"{_disagreement(c)}. The {span}."
            )
        else:
            text = (
                f"{c.measure} shows no clear difference across the {len(c.groups)} {c.grouping} groups "
                f"({_test_text(c)}; {effect_text(c)}): {span}."
            )
    text += _unranked(c, column_values)
    if c.trend is not None:
        text += " " + trend_sentence(c.trend)
    if c.zero_heavy:
        text += _zero_text(c)
    elif c.uses_ranks:
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


def leads_with_trend(comparison: GroupComparison) -> bool:
    """Whether a comparison's finding is its trend across ordered levels rather than a difference."""
    trend = comparison.trend
    return comparison.verdict != "clear" and trend is not None and trend.verdict == "clear"


def lead_sentence(comparison: GroupComparison, column_values: pd.Series | None = None) -> str:
    """One sentence naming a clear finding, the groups' centres and their sizes, for a headline."""
    c = comparison
    coverage = ""
    if c.coverage < 0.5:
        coverage = f" Only {c.coverage:.0%} of rows record both {c.measure} and {c.grouping}."
    if leads_with_trend(c):
        trend = c.trend
        return (
            f"{c.measure} {trend.direction} as {c.grouping} increases: Spearman ρ = {trend.rho:.2f} across "
            f"{trend.levels} levels (n = {trend.n:,}).{coverage}"
        )
    word = centre_word(c)
    first, second = format_values_apart(
        c.centre(c.higher), c.centre(c.lower), column=c.measure, column_values=column_values
    )
    if c.two_groups:
        return (
            f"{c.measure} is higher in {c.higher.label} than in {c.lower.label}: {word} {first} vs "
            f"{second} (n = {c.higher.n:,} and {c.lower.n:,}).{coverage}"
        )
    return (
        f"{c.measure} differs across {c.grouping}: {centre_word(c, plural=True)} from {second} "
        f"({c.lower.label}, n = {c.lower.n:,}) to {first} ({c.higher.label}, n = {c.higher.n:,}), "
        f"{c.explained_name} = {c.explained:.2f}.{coverage}"
    )


def effect_clause(comparison: GroupComparison) -> str:
    """The effect and p-value of a comparison in a few words: "Hedges' g = 0.50, p < 0.001"."""
    c = comparison
    if leads_with_trend(c):
        trend = c.trend
        effect, raw, adjusted = f"Spearman ρ = {trend.rho:.2f}", trend.p_value, trend.adjusted_p
    else:
        if c.two_groups and c.uses_ranks:
            effect = f"{c.higher.label} larger in {c.superiority:.0%} of pairs"
        else:
            effect = effect_text(c)
        raw, adjusted = c.p_value, c.adjusted_p
    shown, decisive = format_p(raw), format_p(raw if adjusted is None else adjusted)
    if decisive == shown:
        return f"{effect}, {decisive}"
    # A sizeable effect next to "p > 0.99" reads as a contradiction unless the adjustment shows.
    return f"{effect}; {shown}, {ADJUSTED[c.correction]} {decisive.removeprefix('p ')}"


def comparison_calculation(comparison: GroupComparison) -> str:
    """The calculation behind ``describe_comparison``, step by step."""
    c = comparison
    word = "median" if c.uses_ranks and not c.zero_heavy else "mean"
    steps = [f"{word} {c.measure} per {c.grouping}"]
    means = c._means_test
    if c.two_groups and c.uses_ranks:
        steps.append(
            "Mann-Whitney test on ranks, with the share of cross-group pairs in which the higher "
            f"group's value is the larger; checked with {means}"
        )
    elif c.two_groups:
        steps.append(
            f"difference in means with a 95% CI and {means}, Hedges' g; checked with the Mann-Whitney test"
        )
    elif c.uses_ranks:
        steps.append(
            f"Kruskal-Wallis with ε² = H ÷ (n − 1); checked with {means}; highest vs lowest {word}"
        )
    else:
        steps.append(
            f"{means} with η² as the between-group share of the total sum of squares; checked with "
            f"Kruskal-Wallis; Hedges' g for the highest vs lowest mean"
        )
    if c.pooled:
        steps.append(f"variances pooled because a group has fewer than {MIN_WELCH_GROUP} values")
    steps.append("p-values from scipy.stats distributions")
    if any(group.n < MIN_NAMED_GROUP for group in c.groups):
        steps.append(f"groups with fewer than {MIN_NAMED_GROUP} values are not named as highest or lowest")
    if c.uses_ranks:
        steps.append(f"ranks because {c.shape}")
    if c.zero_heavy:
        steps.append("share of values that are not 0 per group, chi-squared test")
    if c.trend is not None:
        steps.append(f"trend: Spearman's ρ between the {c.grouping} level and {c.measure}")
    if c.entity:
        steps.append(f"one value per {c.entity} in each group (its {c.entity_summary}) before testing")
    if c.adjusted_p is not None:
        method = "fdr_bh" if c.correction == FALSE_DISCOVERY else "holm"
        steps.append(
            f"p {ADJUSTED[c.correction]} for {c.tests} test{'' if c.tests == 1 else 's'} "
            f"(statsmodels multipletests, {method})"
        )
    return "; ".join(steps)


def comparison_table(comparison: GroupComparison) -> pd.DataFrame:
    """One row per group, highest centre first: the centre, n, the other centre and the SD."""
    by_median = comparison.uses_ranks and not comparison.zero_heavy
    word, other = ("Median", "Mean") if by_median else ("Mean", "Median")
    table = pd.DataFrame(
        {
            comparison.grouping: [group.label for group in comparison.groups],
            f"{word} {comparison.measure}": [comparison.centre(group) for group in comparison.groups],
            "n": [group.n for group in comparison.groups],
            f"{other} {comparison.measure}": [
                group.mean if by_median else group.median for group in comparison.groups
            ],
            "SD": [group.sd for group in comparison.groups],
        }
    )
    if comparison.zero_heavy:
        table["Share not 0"] = [group.nonzero for group in comparison.groups]
    return table


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


def far_out(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """Which records lie over EXTREME_Z robust SDs out in either column, or None when neither has a spread."""
    z_first, z_second = _robust_z(first), _robust_z(second)
    if z_first is None and z_second is None:
        return None
    outside = np.zeros(len(first), dtype=bool)
    for z in (z_first, z_second):
        if z is not None:
            outside |= z > EXTREME_Z
    return outside


def longest_stretch(marked: np.ndarray) -> int:
    """The longest run of consecutive marked records."""
    if not marked.any():
        return 0
    edges = np.diff(np.r_[0, marked.astype(int), 0])
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def without_extremes(first: np.ndarray, second: np.ndarray) -> tuple[float, int] | None:
    """Pearson r after dropping records more than EXTREME_Z robust SDs out in either column.

    Returns None when nothing is that extreme, or when so much is (over 5% of
    records) that the extremes are part of the distribution rather than errors.
    """
    outside = far_out(first, second)
    if outside is None:
        return None
    keep = ~outside
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
    #: (group, r, rows) for each group with enough rows, in the grouping's order.
    per_group: tuple[tuple[object, float, int], ...] = ()


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
    per_group, details = [], []
    for level, part in data.groupby(grouping, observed=True):
        if part[first].nunique() > 1 and part[second].nunique() > 1:
            per_group.append(float(np.corrcoef(part[first], part[second])[0, 1]))
            details.append((level, per_group[-1], len(part)))
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
        per_group=tuple(details),
    )


@dataclass(frozen=True)
class Derivation:
    """A column that other columns reproduce row by row."""

    column: str
    parts: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    #: "sum", "mean", "difference", "combination", "complement" (the column and its
    #: parts add up to a fixed total), "maximum" or "minimum" (the largest or
    #: smallest of the parts), or "standardised" (a z-score of its part, within
    #: each group of ``within`` when that is set).
    kind: str
    #: Share of the rows holding the column and its parts on which the formula holds.
    share: float = 1.0
    #: Rows that break the formula, and the range of what the formula gives there
    #: (for a complement, what the column and its parts add up to).
    breaks: int = 0
    broken: tuple[float, float] | None = None
    within: str | None = None
    #: Rows where the column reads 0 although every part is blank: a placeholder, not a value.
    blank_rows: int = 0

    def describe(self) -> str:
        if self.kind == "sum":
            return f"{self.column} = {' + '.join(self.parts)}"
        if self.kind == "mean":
            return f"{self.column} = mean of {_listed_parts(self.parts)}"
        if self.kind in ("maximum", "minimum"):
            word = "largest" if self.kind == "maximum" else "smallest"
            return f"{self.column} = the {word} of {_listed_parts(self.parts)}"
        if self.kind == "complement":
            total = format_number(self.intercept, self.column, compact=False)
            return f"{_joined([*self.parts, self.column])} add up to {total}"
        if self.kind == "standardised":
            scope = f" within each {self.within}" if self.within else ""
            source = f"{self.parts[0]} standardised" if self.parts else "standardised"
            return f"{self.column} is {source}{scope} (mean 0, SD 1)"
        if self.kind == "difference":
            ordered = sorted(zip(self.coefficients, self.parts, strict=True), key=lambda item: -item[0])
            formula = ordered[0][1] + "".join(
                f" {'+' if weight > 0 else '−'} {part}" for weight, part in ordered[1:]
            )
            return f"{self.column} = {formula}"
        return f"{self.column} is a fixed combination of {_joined(self.parts)}"

    @property
    def block(self) -> tuple[str, ...]:
        """The column and its parts."""
        return (*self.parts, self.column)

    def exceptions(self) -> list[str]:
        """Sentences on the rows that break a block of shares or hold a placeholder 0."""
        notes = []
        if self.kind == "complement" and self.breaks and self.broken:
            low, high = (format_number(value, self.column, compact=False) for value in self.broken)
            spread = low if low == high else f"{low} to {high}"
            total = format_number(self.intercept, self.column, compact=False)
            notes.append(
                f"On {_count_rows(self.breaks)}, {_joined([*self.parts, self.column])} add up to {spread} "
                f"rather than {total}, as they do on {_share_text(self.share)} of rows."
            )
        if self.blank_rows:
            notes.append(
                f"On {_count_rows(self.blank_rows)}, {self.column} reads 0 although every column it "
                "summarises is blank, so those zeros are treated as missing."
            )
        return notes


def _joined(names: tuple[str, ...] | list[str]) -> str:
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _listed_parts(names: tuple[str, ...]) -> str:
    """The parts of a formula, or the first and last of a long run of them."""
    if len(names) <= 6:
        return _joined(names)
    return f"the {len(names):,} columns {names[0]} … {names[-1]}"


def _share_text(share: float) -> str:
    """A share of rows as "100%" or "99.6%", never rounding a near-miss up to 100%."""
    return "100%" if share >= 1 else f"{min(share, 0.999):.1%}"


def _count_rows(count: int) -> str:
    return f"{count:,} row{'' if count == 1 else 's'}"


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
    elif near(coefficients, -1.0, rtol=0, atol=0.02).all():
        # The column is what the others leave of a fixed total: shares of 100.
        return "complement"
    return "combination"


def _spread(values: np.ndarray) -> float:
    """A spread measured on the middle half of the values, so a few huge readings can't widen it."""
    quartiles = np.percentile(values, [25, 75])
    return float(quartiles[1] - quartiles[0]) / 1.349 or float(values.std())


def _step(values: np.ndarray, spread: float) -> float:
    """The smallest gap between distinct values: the step a column was recorded at."""
    if not spread > 0:
        return 0.0
    grain = spread * 1e-6
    steps = np.diff(np.unique(np.round(values / grain))) * grain
    return float(steps[steps > 0].min()) if (steps > 0).any() else 0.0


def _tolerance(values: np.ndarray) -> tuple[float, float]:
    """How closely a formula must reproduce a column, and the column's spread.

    1% of the spread, or half the column's own step when it was rounded after
    it was computed (whole degrees, then converted), but never more than
    DERIVED_STEP_LIMIT of the spread.
    """
    spread = _spread(values)
    half_step = _step(values, spread) / 2 * (1 + 1e-3)
    return max(DERIVED_TOLERANCE * spread, min(half_step, DERIVED_STEP_LIMIT * spread)), spread


def find_derivation(
    frame: pd.DataFrame, column: str, candidates: list[str], steps: dict[str, float] | None = None
) -> Derivation | None:
    """Whether ``column`` is a fixed combination of a few of ``candidates``: a sum, a mean, a difference.

    Candidates should come strongest-correlated first. They are added one at a
    time, each the one closest to what the others leave unexplained, until the
    combination reproduces the column on DERIVED_SHARE of its rows. A fit
    needs three rows per term, so a few rows can't be matched by chance.
    ``steps`` caches the step each column was recorded at, when many columns
    are searched over the same rows.
    """
    # A part recorded on only some rows (acidity measured only on acid soils)
    # still shows the identity on the rows that have it, so candidates are
    # kept while enough rows have all of them.
    recorded = frame[[column, *candidates]].notna().to_numpy()
    complete = recorded[:, 0]
    needed = max(DERIVATION_MIN_ROWS, DERIVATION_MIN_SHARE * int(complete.sum()))
    admitted = []
    for index, candidate in enumerate(candidates, start=1):
        joint = complete & recorded[:, index]
        if joint.sum() >= needed:
            admitted.append(candidate)
            complete = joint
    candidates = admitted
    data = frame[[column, *candidates]].iloc[np.flatnonzero(complete)]
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
    tolerance, spread = _tolerance(target)
    # Parts recorded at a coarser step than the column (whole-number sensor
    # readings averaged, then truncated) move the formula by up to that step
    # times their weight.
    steps = {} if steps is None else steps
    for index, candidate in enumerate(candidates):
        if candidate not in steps:
            steps[candidate] = _step(inputs[:, index], _spread(inputs[:, index]))
    part_steps = np.array([steps[candidate] for candidate in candidates])
    ones = np.ones(rows)

    def allowance(indices: list[int], weights: np.ndarray) -> float:
        rounding = float((np.abs(weights) * part_steps[indices]).sum())
        return tolerance + min(rounding, max(DERIVED_STEP_LIMIT * spread - tolerance, 0.0))

    def matched(residual: np.ndarray, limit: float) -> float:
        return float(np.mean(np.abs(residual) <= limit))

    def fit(indices: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
        """Least squares, refitted on the rows it matches best, or its weights as simple fractions.

        A few rows entered by hand, or a column rounded after it was computed,
        pull least squares slightly off the formula (0.49 and 0.52 instead of
        a half each), so the fit is also tried with each weight snapped to
        the nearest simple fraction.
        """
        design = np.column_stack([ones, inputs[:, indices]])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        # A few far-off rows can pull the first fit towards them, so the rows
        # it matches best are refitted until they settle.
        for _ in range(REFITS):
            residual = np.abs(target - design @ coefficients)
            close = residual <= np.quantile(residual, DERIVED_SHARE)
            coefficients = np.linalg.lstsq(design[close], target[close], rcond=None)[0]
        residual = target - design @ coefficients
        limit = allowance(indices, coefficients[1:])
        snapped = np.array([_simple_fraction(weight) for weight in coefficients[1:]])
        simple_limit = allowance(indices, snapped)
        offset = float(np.median(target - design[:, 1:] @ snapped))
        offset = 0.0 if abs(offset) <= simple_limit else offset
        simple = np.r_[offset, snapped]
        simple_residual = target - design @ simple
        # A simple formula that matches about as many rows is the likelier one:
        # least squares bends towards the few rows that break it.
        if matched(simple_residual, simple_limit) >= matched(residual, limit) - SIMPLE_MARGIN:
            return simple, simple_residual, simple_limit
        return coefficients, residual, limit

    def reproduces(residual: np.ndarray, limit: float, simple: bool = True) -> bool:
        # A free combination has more weights to fit than a sum or a mean, so
        # it must hold on nearly every row.
        needed = DERIVED_SHARE if simple else COMBINATION_SHARE
        return _reproduces(target, residual, limit, spread, needed)

    if not reproduces(*fit(list(range(usable)))[1:]):
        return None
    scale = inputs.std(axis=0)
    standardised = (inputs - inputs.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    chosen: list[int] = []
    residual = target - target.mean()
    for _ in range(min(MAX_PARTS, usable)):
        scores = np.abs(standardised.T @ residual)
        scores[chosen] = -np.inf
        chosen.append(int(np.argmax(scores)))
        coefficients, residual, limit = fit(chosen)
        if reproduces(residual, limit):
            weights = coefficients[1:]
            used = np.abs(weights) * scale[chosen] > 1e-6 * spread
            if used.sum() < 2:
                return None
            parts = tuple(candidates[index] for index, keep in zip(chosen, used, strict=True) if keep)
            weights = weights[used]
            kind = _classify(weights, float(coefficients[0]), limit)
            if kind == "combination" and not reproduces(residual, limit, simple=False):
                return None
            return Derivation(
                column=column,
                parts=parts,
                coefficients=tuple(float(weight) for weight in weights),
                intercept=float(coefficients[0]),
                kind=kind,
                share=matched(residual, limit),
            )
    return None


def _reproduces(target: np.ndarray, residual: np.ndarray, limit: float, spread: float, needed: float) -> bool:
    """Whether a formula with these residuals reproduces ``target`` on ``needed`` of its rows."""
    close = np.abs(residual) <= limit
    if close.mean() < needed:
        return False
    # Matching most rows is not enough when they are mostly one value (a
    # column of zeros with a few readings): the formula must also explain the spread.
    total = float(((target - target.mean()) ** 2).sum())
    if total > 0 and 1 - float((residual**2).sum()) / total >= DERIVED_EXPLAINED:
        return True
    # A few rows that break the identity (shares that no longer add up once
    # one was clipped at zero) must not hide it, provided the rows that keep
    # it span the column's range.
    kept = target[close]
    held = float(((kept - kept.mean()) ** 2).sum())
    return (
        held > 0
        and 1 - float((residual[close] ** 2).sum()) / held >= DERIVED_EXPLAINED
        and _spread(kept) >= 0.5 * spread
    )


def find_row_extreme(frame: pd.DataFrame, column: str, candidates: list[str]) -> Derivation | None:
    """Whether ``column`` is the largest (or smallest) of some of ``candidates`` on each row.

    A daily index reported as the worst of several sub-indices is a row
    maximum: never below any of them, and equal to one of them on almost
    every row. The parts are the candidates that never exceed the column and
    supply its value on some rows; each row uses the parts it has.
    """
    if len(candidates) < 2:
        return None
    matrix = frame[[column, *candidates]].to_numpy(dtype=float)
    matrix = matrix[~np.isnan(matrix[:, 0])]
    if len(matrix) < DERIVATION_MIN_ROWS:
        return None
    if len(matrix) > 5 * DERIVATION_ROWS:
        matrix = matrix[np.linspace(0, len(matrix) - 1, 5 * DERIVATION_ROWS).astype(int)]
    target = matrix[:, 0]
    # A first look at a few rows rules out most columns: a row maximum is at
    # least as large as two or more of the others on nearly every row.
    head = matrix[np.linspace(0, len(matrix) - 1, min(len(matrix), 200)).astype(int)]
    margin = 0.01 * (np.nanmax(head[:, 0]) - np.nanmin(head[:, 0]))
    directions = [
        (kind, sign)
        for kind, sign in (("maximum", 1.0), ("minimum", -1.0))
        if (np.mean(~(sign * head[:, 1:] > sign * head[:, :1] + margin), axis=0) >= 0.97).sum() >= 2
    ]
    if not directions or np.unique(target).size < 3:
        return None
    tolerance, spread = _tolerance(target)
    for kind, sign in directions:
        values = sign * matrix[:, 1:]
        reference = sign * target
        recorded = ~np.isnan(values)
        within = np.where(recorded, values <= reference[:, None] + tolerance, True)
        attains = recorded & (np.abs(values - reference[:, None]) <= tolerance)
        # A part never exceeds the column (bar a stray row) and supplies its value on some rows.
        keep = (within.mean(axis=0) >= 0.99) & (attains.sum(axis=0) >= 2)
        if keep.sum() < 2:
            continue
        rows = recorded[:, keep].any(axis=1)
        if rows.sum() < DERIVATION_MIN_ROWS:
            continue
        extreme = np.max(np.where(recorded[rows][:, keep], values[rows][:, keep], -np.inf), axis=1)
        residual = reference[rows] - extreme
        if _reproduces(reference[rows], residual, tolerance, spread, DERIVED_SHARE):
            parts = tuple(candidate for candidate, kept in zip(candidates, keep, strict=True) if kept)
            share = float(np.mean(np.abs(residual) <= tolerance))
            return Derivation(column, parts, (1.0,) * len(parts), 0.0, kind, share=share)
    return None


# A score is standardised when its mean is within this of 0 and its SD within this of 1:
# loosely overall, and tightly within each group, where chance can't produce it.
STANDARDISED_CLOSE = 0.05
STANDARDISED_EXACT = 0.005
STANDARDISED_MIN_GROUPS = 3


def find_standardised(
    frame: pd.DataFrame, column: str, candidates: list[str], groupings: list[str]
) -> Derivation | None:
    """Whether ``column`` is a z-score: mean 0 and SD 1 overall, or within each group of a grouping.

    A score standardised within each drug of a screen has mean 0 and SD 1 for
    every drug and is its source column rescaled drug by drug, so its mean says
    nothing and its correlation with the source is built in. ``frame`` must
    hold every row, since a sample of a group is not standardised. Any
    grouping made of whole groups passes too, so the finest one that passes
    is named. The source is named when one of ``candidates`` rescales to the
    column on DERIVED_SHARE of rows.
    """
    values = frame[column]
    present = values.dropna()
    if len(present) < DERIVATION_MIN_ROWS:
        return None
    # Mean 0 and SD 1 within every group leaves mean 0 and SD 1 overall.
    if abs(present.mean()) > STANDARDISED_CLOSE or abs(present.std() - 1) > STANDARDISED_CLOSE:
        return None
    share, within, levels = 0.0, None, 0
    for grouping in groupings:
        if grouping == column or grouping not in frame.columns:
            continue
        summary = values.groupby(frame[grouping], observed=True).agg(["mean", "std", "size"])
        summary = summary[summary["size"] >= 3]
        if len(summary) < STANDARDISED_MIN_GROUPS:
            continue
        fits = (summary["mean"].abs() <= STANDARDISED_EXACT) & (
            (summary["std"] - 1).abs() <= STANDARDISED_EXACT
        )
        covered = float(summary.loc[fits, "size"].sum() / summary["size"].sum())
        if covered >= DERIVED_SHARE and len(summary) > levels:
            share, within, levels = covered, grouping, len(summary)
    for candidate in candidates:
        source = frame[candidate]
        if within is None:
            rescaled = (source - source.mean()) / source.std()
        else:
            grouped = source.groupby(frame[within], observed=True)
            rescaled = (source - grouped.transform("mean")) / grouped.transform("std")
        both = values.notna() & rescaled.notna()
        if both.sum() < DERIVATION_MIN_ROWS:
            continue
        close = float(((values[both] - rescaled[both]).abs() <= 2 * DERIVED_TOLERANCE).mean())
        if close >= DERIVED_SHARE:
            return Derivation(column, (candidate,), (1.0,), 0.0, "standardised", share=close, within=within)
    if within is None:
        return None
    return Derivation(column, (), (), 0.0, "standardised", share=share, within=within)


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


# Words naming a compass direction. Its 0 and 360 are the same place, so a
# linear mean or correlation of it means nothing.
DIRECTION_NAME_WORDS = frozenset({
    "direction", "dir", "wdir", "winddir", "wd", "dr", "bearing", "heading", "azimuth",
})
MIN_ANGLE_VALUES = 8


def is_angle(name: str, values: pd.Series) -> bool:
    """Whether a column holds compass directions: a direction word in its name and values from 0 to 360."""
    if not set(re.findall(r"[a-z]+", normalized_name(name))) & DIRECTION_NAME_WORDS:
        return False
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    return bool(numbers.nunique() >= MIN_ANGLE_VALUES and numbers.min() >= 0 and numbers.max() <= 360)


def is_coordinate(name: str) -> bool:
    """Whether the name marks a map coordinate, which places a row rather than measures it."""
    return bool(set(re.findall(r"[a-z]+", normalized_name(name))) & COORDINATE_TOKENS)


def _families(
    columns: list[str], pearson: np.ndarray, spearman: np.ndarray, values: np.ndarray
) -> tuple[list[int], list[tuple[str, str, float]]]:
    """Group columns that are one quantity recorded twice: |r| ≥ 0.99 with ranks that agree.

    One huge record (a taxon holding half of all reads) can make two columns
    look identical, so the pair must also stay at 0.99 without its most
    influential record.
    """
    parent = list(range(len(columns)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    duplicates = []
    first, second = np.where(np.triu(np.abs(pearson) >= NEAR_DUPLICATE_CORRELATION, k=1))
    for i, j in zip(first, second, strict=True):
        if abs(pearson[i, j]) - abs(spearman[i, j]) > RANK_DIVERGENCE:
            continue
        both = ~np.isnan(values[:, i]) & ~np.isnan(values[:, j])
        influence = most_influential(values[both, i], values[both, j])
        if influence is not None and abs(influence[1]) < NEAR_DUPLICATE_CORRELATION:
            continue
        parent[root(i)] = root(j)
        duplicates.append((columns[i], columns[j], float(pearson[i, j])))
    return [root(index) for index in range(len(columns))], duplicates


def _names_an_aggregate(column: str) -> bool:
    return bool(set(normalized_name(column).split()) & AGGREGATE_NAME_WORDS)


def _derivations(
    frame: pd.DataFrame,
    columns: list[str],
    targets: set[int],
    pearson: np.ndarray,
    counts: np.ndarray,
    family: list[int],
) -> dict[str, Derivation]:
    """Columns built from two or more others, with their parts traced to columns not built themselves.

    An exact identity can be read either way round (a total is the sum of its
    items, and an item is the total minus the others), so a column is matched
    against the columns before it, where computed columns are usually
    appended. A column named as a total or mean may also be the sum or mean of
    columns after it. A total can be rebuilt from other totals ("CorIncor =
    CorrectTotal − IncorrectTotal"), so each derivation is then retraced to the
    columns that are not derived themselves, which names the items it was made
    from. Only the columns in ``targets`` are tried as derived; the others can
    only be parts.
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

    steps: dict[str, float] = {}

    def search(index: int, allowed: set[int]) -> Derivation | None:
        column = columns[index]
        found = find_derivation(frame, column, candidates(index, allowed & set(range(index))), steps)
        if found is None and _names_an_aggregate(column):
            found = find_derivation(frame, column, candidates(index, allowed), steps)
            if found is not None and found.kind not in ("sum", "mean"):
                found = None
        return found

    everything = set(range(len(columns)))
    derived = {}
    for index in sorted(targets):
        found = search(index, everything)
        if found is not None:
            derived[columns[index]] = found
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


def _summaries(
    dataframe: pd.DataFrame,
    frame: pd.DataFrame,
    columns: list[str],
    targets: set[int],
    pearson: np.ndarray,
    family: list[int],
) -> dict[str, Derivation]:
    """Columns that summarise others row by row without a fixed formula: a row maximum, a z-score.

    ``frame`` holds a spread of rows, enough for a row maximum; a z-score is
    checked on every row of ``dataframe``.
    """
    found: dict[str, Derivation] = {}
    groupings: list[str] | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        highs, lows = np.nanmax(frame.to_numpy(), axis=0), np.nanmin(frame.to_numpy(), axis=0)
    for index in sorted(targets):
        column = columns[index]
        # A part that rarely sets the maximum need not correlate with it.
        strength = np.nan_to_num(np.abs(pearson[index]), nan=-1.0)
        nearest = [
            other
            for other in np.argsort(-strength)
            if other != index and family[other] != family[index] and strength[other] >= 0
        ][:PART_CANDIDATES]
        # A row maximum reaches the largest value of its parts, and a row minimum the smallest.
        close = 0.01 * (highs[index] - lows[index])
        reaches = nearest and (
            abs(highs[index] - np.nanmax(highs[nearest])) <= close
            or abs(lows[index] - np.nanmin(lows[nearest])) <= close
        )
        extreme = find_row_extreme(frame, column, [columns[other] for other in nearest]) if reaches else None
        if extreme is not None:
            found[column] = extreme
            continue
        values = frame[column].dropna()
        if len(values) < DERIVATION_MIN_ROWS or abs(values.mean()) > STANDARDISED_CLOSE:
            continue
        if groupings is None:
            # Columns of repeated labels, such as the drug each row was screened with.
            labels = dataframe.select_dtypes(include=["object", "category", "string"])
            groupings = [
                name for name in labels.columns
                if STANDARDISED_MIN_GROUPS <= labels[name].nunique() <= len(labels) / 3
            ]
        closest = np.argsort(-np.abs(np.nan_to_num(pearson[index])))
        sources = [columns[other] for other in closest if other != index][:5]
        standardised = find_standardised(dataframe, column, sources, groupings)
        if standardised is not None:
            found[column] = standardised
    return found


def _checked(dataframe: pd.DataFrame, derivation: Derivation) -> tuple[Derivation, pd.Index]:
    """The derivation with its exceptions counted on every row, and the rows holding a placeholder 0.

    Shares that should add up to a fixed total are checked row by row, so the
    rows that break the total can be named. A sum, mean or row maximum that
    reads 0 where every part is blank was filled in, not measured.
    """
    parts = [part for part in derivation.parts if part in dataframe.columns]
    blank = dataframe.index[:0]
    if not parts or derivation.column not in dataframe.columns:
        return derivation, blank
    target = pd.to_numeric(dataframe[derivation.column], errors="coerce")
    if derivation.kind in ("sum", "mean", "maximum", "minimum") and len(parts) == len(derivation.parts):
        blank = dataframe.index[(target == 0) & dataframe[parts].isna().all(axis=1)]
    if derivation.kind == "complement" and len(parts) == len(derivation.parts):
        block = dataframe[[*parts, derivation.column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(block):
            totals = block.sum(axis=1).to_numpy(dtype=float)
            limit = _tolerance(block[derivation.column].to_numpy(dtype=float))[0] + sum(
                _step(block[part].to_numpy(dtype=float), _spread(block[part].to_numpy(dtype=float)))
                for part in parts
            )
            off = np.abs(totals - derivation.intercept) > limit
            derivation = replace(
                derivation,
                share=float(1 - off.mean()),
                breaks=int(off.sum()),
                broken=(float(totals[off].min()), float(totals[off].max())) if off.any() else None,
            )
    return replace(derivation, blank_rows=len(blank)), blank


@dataclass(frozen=True, eq=False)
class Construction:
    """How a table's numeric columns are built from one another.

    One quantity recorded twice (a temperature in °C and in °F), a column
    computed from others (a total, a mean, the largest of several readings, a
    score standardised within each group, shares that add up to 100), and a
    flag marking where another column hits its cap. A pair related this way
    says nothing new, so the comparison and relationship scans both set it
    aside. The correlations it is found from serve the relationship scan.
    """

    #: Every column screened, in table order: the readings and any items.
    columns: tuple[str, ...]
    readings: tuple[str, ...]
    #: Readings that vary, before the cap on very wide tables.
    available: int
    #: Rows in the table; above SCAN_ROWS the screen ran on a fixed random sample of SCAN_ROWS.
    rows: int
    frame: pd.DataFrame
    ranked: _Ranked
    pearson: np.ndarray
    spearman: np.ndarray
    counts: np.ndarray
    family: tuple[int, ...]
    near_duplicates: tuple[tuple[str, str, float], ...]
    derived: tuple[Derivation, ...]
    flags: tuple[CapFlag, ...]
    #: Compass directions, left out of every linear calculation.
    angles: tuple[str, ...] = ()
    #: For each summary that reads 0 where its parts are blank, the rows holding that 0.
    blanks: tuple[tuple[str, pd.Index], ...] = ()

    @property
    def sampled(self) -> bool:
        return self.rows > SCAN_ROWS

    @cached_property
    def position(self) -> dict[str, int]:
        return {column: index for index, column in enumerate(self.columns)}

    @cached_property
    def _built(self) -> dict[str, Derivation]:
        return {item.column: item for item in self.derived}

    def derivation(self, column: str) -> Derivation | None:
        return self._built.get(column)

    def same_quantity(self, first: str, second: str) -> bool:
        i, j = self.position.get(first), self.position.get(second)
        return i is not None and j is not None and self.family[i] == self.family[j]

    def representative(self, column: str, prefer: tuple[str, ...] = ()) -> str:
        """The column that stands for ``column``'s family: a preferred one, or the first in the table."""
        if column not in self.position:
            return column
        members = [other for other in self.columns if self.same_quantity(column, other)]
        return next((other for other in prefer if other in members), members[0])

    @cached_property
    def _parts(self) -> dict[str, frozenset]:
        """Each derived column's parts, as the families they belong to (or their names, when not screened)."""
        return {
            item.column: frozenset(
                self.family[self.position[part]] if part in self.position else part for part in item.parts
            )
            for item in self.derived
        }

    def _part_families(self, column: str) -> frozenset:
        return self._parts.get(column, frozenset())

    def built_from(self, column: str, other: str) -> bool:
        """Whether ``column`` is computed from ``other``, or from a copy of it."""
        families = self._part_families(column)
        key = self.family[self.position[other]] if other in self.position else other
        return key in families

    def relation(self, first: str, second: str) -> str | None:
        """Why two columns are related by construction: "copy", "flag", "built", "total", or None."""
        if self.same_quantity(first, second):
            return "copy"
        if any({flag.flag, flag.column} == {first, second} for flag in self.flags):
            return "flag"
        if self.built_from(first, second) or self.built_from(second, first):
            return "built"
        one, other = self._part_families(first), self._part_families(second)
        if one and other and (one <= other or other <= one):
            return "built"
        for item in self.derived:
            if item.kind == "complement" and first in item.block and second in item.block:
                return "total"
        return None

    def masked(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """The table with each summary's placeholder zeros set to missing."""
        if not self.blanks:
            return dataframe
        dataframe = dataframe.copy()
        for column, rows in self.blanks:
            dataframe.loc[rows, column] = np.nan
        return dataframe


def construction(
    dataframe: pd.DataFrame,
    columns: list[str],
    measure: str | None = None,
    *,
    items: tuple[str, ...] | list[str] = (),
    known: tuple[Derivation, ...] = (),
) -> Construction | None:
    """Screen numeric ``columns`` for pairs related by construction, on at most SCAN_ROWS rows.

    ``items`` are further numeric columns, such as rating items read as
    categories, that a score in ``columns`` may be built from; they are tried
    as parts but never as derived columns. ``known`` holds derivations the
    caller already knows, such as a row mean it added. Returns None when fewer
    than two columns vary.
    """
    columns = list(dict.fromkeys(column for column in columns if column in dataframe.columns))
    angles = tuple(column for column in columns if column != measure and is_angle(column, dataframe[column]))
    columns = [column for column in columns if column not in angles]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        variance = np.nanvar(dataframe[columns].to_numpy(dtype=float), axis=0)
    # A column that never varies relates to nothing.
    varying = pd.Series(variance, index=columns)
    varying = varying[varying > 0]
    columns = [column for column in columns if column in varying.index]
    available = len(columns)
    if available > MAX_SCAN_COLUMNS:
        rest = varying.drop(measure) if measure in varying.index else varying
        kept = set(rest.nlargest(MAX_SCAN_COLUMNS - (measure in varying.index)).index) | {measure}
        columns = [column for column in columns if column in kept]
    if len(columns) < 2:
        return None
    extra = [
        column
        for column in dict.fromkeys(items)
        if column in dataframe.columns
        and column not in columns
        and column not in angles
        and pd.api.types.is_numeric_dtype(dataframe[column])
        and dataframe[column].nunique() > 1
    ]
    order = {column: index for index, column in enumerate(dataframe.columns)}
    screened = sorted([*columns, *extra], key=order.__getitem__)
    rows = len(dataframe)
    sample = dataframe.sample(SCAN_ROWS, random_state=0) if rows > SCAN_ROWS else dataframe
    frame = sample[screened].astype(float)

    def correlations(values: np.ndarray) -> tuple[_Ranked, np.ndarray, np.ndarray, np.ndarray]:
        ranked = _rank_columns(values)
        pearson, counts = pairwise_correlations(values)
        spearman, _ = pairwise_correlations(ranked.ranks)
        usable = counts >= MIN_CORRELATION_PAIRS
        np.fill_diagonal(usable, False)
        return ranked, np.where(usable, pearson, np.nan), np.where(usable, spearman, np.nan), counts

    values = frame.to_numpy()
    ranked, pearson, spearman, counts = correlations(values)
    family, duplicates = _families(screened, pearson, spearman, values)
    # Exact identities hold on every row, so a spread of rows is enough to find them.
    thinned = frame.iloc[np.linspace(0, len(frame) - 1, min(len(frame), 5 * DERIVATION_ROWS)).astype(int)]
    targets = {screened.index(column) for column in columns}
    derived = _derivations(thinned, screened, targets, pearson, counts, family)
    remaining = {index for index in targets if screened[index] not in derived}
    for column, found in _summaries(dataframe, thinned, screened, remaining, pearson, family).items():
        derived[column] = replace(found, parts=tuple(sorted(found.parts, key=order.__getitem__)))
    for item in known:
        derived[item.column] = item
    blanks = []
    checked = []
    for item in derived.values():
        item, blank = _checked(dataframe, item)
        checked.append(item)
        if len(blank):
            blanks.append((item.column, blank))
    if blanks:
        # A summary's placeholder zeros are missing values, not readings of 0.
        for column, blank in blanks:
            frame.loc[frame.index.intersection(blank), column] = np.nan
        ranked, pearson, spearman, counts = correlations(frame.to_numpy())
    return Construction(
        columns=tuple(screened),
        readings=tuple(columns),
        available=available,
        rows=rows,
        frame=frame,
        ranked=ranked,
        pearson=pearson,
        spearman=spearman,
        counts=counts,
        family=tuple(family),
        near_duplicates=tuple(sorted(duplicates, key=lambda item: -abs(item[2]))),
        derived=tuple(checked),
        flags=tuple(_cap_flags(frame, ranked, counts)),
        angles=angles,
        blanks=tuple(blanks),
    )


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
    #: Pairs of map coordinates, which describe where the rows were taken rather than how they relate.
    coordinates: tuple[tuple[str, str], ...] = ()
    #: Compass directions left out, since a linear correlation of an angle means nothing.
    angles: tuple[str, ...] = ()
    #: Rows in the table; above SCAN_ROWS the pairs were screened on a sample of SCAN_ROWS.
    rows: int = 0
    #: The measure's columns set aside as related to it by construction, with the reason:
    #: "copy" (the same quantity, or the same name above SHARED_STEM_CORRELATION),
    #: "built", "flag", "total" or "place" (two map coordinates).
    set_aside: tuple[tuple[str, str], ...] = ()

    @property
    def sampled(self) -> bool:
        return self.rows > SCAN_ROWS


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
    built: Construction | None = None,
) -> RelationshipScan | None:
    """Rank what relates to ``measure``, then the strongest pairs among the other columns.

    Pearson and Spearman r for every pair come from matrix products over the
    complete rows of each pair; Spearman r ranks each column once, so with
    missing values it is a close screen, and the pairs returned are recomputed
    exactly. Tables longer than SCAN_ROWS are screened on a fixed random
    sample of that many rows. Pairs related by construction are left out and
    returned for the calculation: near-duplicates, a column built from the
    other (or both built from the same items), shares that add up to a fixed
    total, a flag marking the other's cap, and same-stem names above
    SHARED_STEM_CORRELATION. ``built`` is a screen already made of the same
    columns. Returns None when no pair has MIN_CORRELATION_PAIRS complete rows.
    """
    built = built if built is not None else construction(dataframe, columns, measure)
    if built is None:
        return None
    columns = [column for column in built.readings if column in set(columns) | {measure}]
    if len(columns) < 2:
        return None
    position = {column: built.position[column] for column in columns}
    indices = np.array([position[column] for column in columns])
    values = built.frame.to_numpy()
    pearson = built.pearson[np.ix_(indices, indices)]
    spearman = built.spearman[np.ix_(indices, indices)]
    counts = built.counts[np.ix_(indices, indices)]
    family = [built.family[index] for index in indices]
    usable = counts >= MIN_CORRELATION_PAIRS
    np.fill_diagonal(usable, False)
    if not usable.any():
        return None
    screen = np.fmax(np.abs(pearson), np.abs(spearman))
    local = {column: index for index, column in enumerate(columns)}
    stems: list[tuple[str, str, float]] = []
    coordinates: list[tuple[str, str]] = []

    def related(first: str, second: str) -> str | None:
        reason = built.relation(first, second)
        if reason:
            return reason
        i, j = local[first], local[second]
        if is_coordinate(first) and is_coordinate(second):
            coordinates.append((first, second))
            return "place"
        if abs(pearson[i, j]) > SHARED_STEM_CORRELATION and _stems(first) & _stems(second):
            # One huge record can make two columns look like one, so the ranks,
            # and r without that record, must agree as closely.
            x, y = paired(first, second)
            influence = most_influential(x, y)
            steady = abs(influence[1]) if influence is not None else abs(pearson[i, j])
            if min(steady, abs(spearman[i, j])) > SHARED_STEM_CORRELATION:
                stems.append((first, second, float(pearson[i, j])))
                return "copy"
        return None

    def paired(first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
        both = ~np.isnan(values[:, position[first]]) & ~np.isnan(values[:, position[second]])
        return values[both, position[first]], values[both, position[second]]

    def detail(first: str, second: str) -> Pair:
        i, j = local[first], local[second]
        x, y = paired(first, second)
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
            key = frozenset((family[local[first]], family[local[second]]))
            if key in seen or related(first, second):
                continue
            seen.add(key)
            found.append(detail(first, second))
            if len(found) >= keep:
                break
        return sorted(found, key=lambda pair: pair.strength, reverse=True)

    correlates: list[Pair] = []
    measure_tests = 0
    set_aside: list[tuple[str, str]] = []
    if measure in local:
        i = local[measure]
        order = [j for j in np.argsort(-np.nan_to_num(screen[i], nan=-1.0)) if usable[i, j]]
        measure_tests = len(order)
        correlates = best([(measure, columns[j]) for j in order], max(limit, 15))
        for j in order:
            reason = related(measure, columns[j])
            if reason:
                set_aside.append((columns[j], reason))
    pair_list: list[Pair] = []
    if others:
        upper = np.triu(usable, k=1)
        if measure in local:
            # A pair with the measure's own duplicate is a pair with the measure.
            same = [index for index in range(len(columns)) if family[index] == family[local[measure]]]
            upper[same, :] = False
            upper[:, same] = False
        first, second = np.where(upper)
        order = np.argsort(-screen[first, second], kind="stable")[:200]
        pair_list = best([(columns[first[k]], columns[second[k]]) for k in order], 30)
    pair_tests = int(np.triu(usable, k=1).sum())
    scanned = set(columns)
    return RelationshipScan(
        measure=measure if measure in local else None,
        columns=tuple(columns),
        available=built.available,
        correlates=tuple(correlates[:limit]),
        others=tuple(pair_list[:limit]),
        measure_tests=measure_tests,
        pair_tests=pair_tests,
        derived=tuple(
            item for item in built.derived if item.column in scanned and (item.parts or item.within)
        ),
        flags=tuple(flag for flag in built.flags if {flag.flag, flag.column} <= scanned),
        near_duplicates=tuple(
            item for item in built.near_duplicates if item[0] in scanned and item[1] in scanned
        ),
        shared_stems=tuple(dict.fromkeys(stems)),
        coordinates=tuple(dict.fromkeys(coordinates)),
        angles=built.angles,
        rows=built.rows,
        set_aside=tuple(set_aside),
    )


# --- Joint model --------------------------------------------------------------

# Predictors in one model, and rows needed per predictor for its coefficients to settle.
MAX_MODEL_TERMS = 6
ROWS_PER_TERM = 10
# A variance inflation factor above this means a predictor is largely a mix of the others.
MAX_VIF = 5.0
# Rows a model is fitted on; longer tables use a fixed random sample.
MODEL_ROWS = 50_000


@dataclass(frozen=True)
class ModelTerm:
    """One predictor's coefficient in a joint model, beside its correlation on its own."""

    name: str
    coefficient: float
    low: float
    high: float
    p_value: float
    #: The coefficient in standard deviations of the measure per standard deviation of the predictor.
    standardized: float
    #: Pearson r of the predictor with the measure on the model's rows.
    pairwise: float
    vif: float

    @property
    def reversed(self) -> bool:
        """Whether a clear coefficient points the other way from the predictor's own clear correlation."""
        return (
            self.p_value < ALPHA
            and abs(self.pairwise) >= NOTABLE_CORRELATION
            and np.sign(self.coefficient) != np.sign(self.pairwise)
        )


@dataclass(frozen=True)
class JointModel:
    """An ordinary least-squares fit of a measure on several readings at once.

    Each coefficient is the change in the measure per unit of its reading with
    the other readings held fixed. When readings move together, that can differ
    in size and even in sign from the reading's correlation on its own.
    """

    measure: str
    terms: tuple[ModelTerm, ...]
    intercept: float
    n: int
    r2: float
    adjusted_r2: float
    residual_sd: float
    #: Rows with the measure and every predictor recorded; above MODEL_ROWS the fit used a sample.
    rows: int = 0

    @property
    def best_single(self) -> ModelTerm:
        """The predictor that relates most to the measure on its own."""
        return max(self.terms, key=lambda term: abs(term.pairwise))

    @property
    def collinear(self) -> tuple[str, ...]:
        """Predictors whose variance inflation factor exceeds MAX_VIF."""
        return tuple(term.name for term in self.terms if term.vif > MAX_VIF)

    @property
    def reversed(self) -> tuple[ModelTerm, ...]:
        return tuple(term for term in self.terms if term.reversed)

    @property
    def predictors(self) -> tuple[str, ...]:
        return tuple(term.name for term in self.terms)


def fit_joint_model(frame: pd.DataFrame, measure: str, predictors: list[str]) -> JointModel | None:
    """Fit ``measure`` on up to MAX_MODEL_TERMS of ``predictors`` together, by ordinary least squares.

    The fit uses the rows where the measure and every predictor are recorded
    and needs ROWS_PER_TERM of them per predictor. Predictors are taken in the
    order given; one that does not vary, or that the earlier ones reproduce
    exactly, is dropped. Coefficients with 95% intervals, R² and adjusted R²
    come from ``statsmodels`` OLS, and each predictor's variance inflation
    factor from ``statsmodels.stats.outliers_influence.variance_inflation_factor``.
    Returns None with fewer than two usable predictors or too few rows.
    """
    chosen = [
        column for column in dict.fromkeys(predictors) if column != measure and column in frame.columns
    ][:MAX_MODEL_TERMS]
    if measure not in frame.columns or len(chosen) < 2:
        return None
    data = frame[[measure, *chosen]].apply(pd.to_numeric, errors="coerce").dropna()
    rows = len(data)
    if rows > MODEL_ROWS:
        data = data.sample(MODEL_ROWS, random_state=0)
    kept: list[str] = []
    for column in chosen:
        if data[column].nunique() < 2:
            continue
        design = np.column_stack([np.ones(len(data)), data[[*kept, column]].to_numpy(dtype=float)])
        if np.linalg.matrix_rank(design) == design.shape[1]:
            kept.append(column)
    if len(kept) < 2 or len(data) < ROWS_PER_TERM * len(kept) or data[measure].nunique() < 2:
        return None
    target = data[measure].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(data)), data[kept].to_numpy(dtype=float)])
    fit = OLS(target, design).fit()
    intervals = fit.conf_int(ALPHA)
    spread = data[kept].std().to_numpy(dtype=float)
    terms = tuple(
        ModelTerm(
            name=column,
            coefficient=float(fit.params[index]),
            low=float(intervals[index][0]),
            high=float(intervals[index][1]),
            p_value=float(fit.pvalues[index]),
            standardized=float(fit.params[index] * spread[index - 1] / target.std(ddof=1)),
            pairwise=float(np.corrcoef(data[column].to_numpy(dtype=float), target)[0, 1]),
            vif=float(variance_inflation_factor(design, index)),
        )
        for index, column in enumerate(kept, start=1)
    )
    return JointModel(
        measure=measure,
        terms=terms,
        intercept=float(fit.params[0]),
        n=len(data),
        r2=float(fit.rsquared),
        adjusted_r2=float(fit.rsquared_adj),
        residual_sd=float(np.sqrt(fit.mse_resid)),
        rows=rows,
    )
