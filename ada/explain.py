"""Readable models for ``ada model --explain`` and ``ada auto --explain``, and what they cost.

Two models a person can read are fitted on the same folds as the full search:

- a linear model (ridge regression, or logistic regression for classes), whose
  coefficients and their 95% intervals come from statsmodels on every training
  row;
- an additive model, LightGBM with one split a tree, so each column's effect
  adds up on its own and can be drawn as a curve.

The better of the two by out-of-fold score makes the predictions. Three checks
compare models fold by fold, each with the mean difference and its standard
error over the folds: whether the effects bend (additive against linear),
whether columns act together (the full LightGBM against additive), and what
explaining costs (the full blend against the readable model).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .numeric import to_numbers

#: Rows the effect curves are read from at most, taken evenly through the training rows.
EFFECT_ROWS = 50_000
#: Bands each number's effect curve is read in, and the most levels shown for a category.
BANDS, LEVELS = 10, 8
#: Coefficients and effect curves shown, those with the largest z or share first.
TERMS = EFFECTS = 12
#: A number's effect bends when a straight line explains less than this share of its curve's variance.
STRAIGHT = 0.9


@dataclass(frozen=True)
class Effect:
    """One column's effect in the additive model: its share, its shape and its curve."""

    column: str
    share: float
    shape: str
    points: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict:
        return {"column": self.column, "share": self.share, "shape": self.shape,
                "points": [{"at": at, "effect": effect} for at, effect in self.points]}


@dataclass(frozen=True)
class Term:
    """One coefficient of the linear model, with its 95% interval and p-value."""

    name: str
    estimate: float
    low: float
    high: float
    p_value: float

    def to_dict(self) -> dict:
        return {"term": self.name, "estimate": self.estimate, "ci95": [self.low, self.high],
                "p_value": self.p_value}


@dataclass(frozen=True)
class Comparison:
    """Two models compared fold by fold: the mean difference of their scores and its standard error."""

    name: str
    better: str
    worse: str
    difference: float
    error: float
    verdict: str

    @property
    def detail(self) -> str:
        return (f"{self.better} against {self.worse}: {self.difference:+.5f} ± {self.error:.5f} "
                f"(mean fold difference ± its standard error); {self.verdict}")

    def to_dict(self) -> dict:
        return {"name": self.name, "compared": [self.better, self.worse], "difference": self.difference,
                "standard_error": self.error, "verdict": self.verdict}


@dataclass(frozen=True)
class Explanation:
    """The readable model that makes the predictions, its coefficients and curves, and the comparisons."""

    chosen: str
    unit: str
    terms: tuple[Term, ...]
    terms_note: str
    effects: tuple[Effect, ...]
    effects_note: str
    comparisons: tuple[Comparison, ...]

    def to_dict(self) -> dict:
        return {"chosen": self.chosen, "unit": self.unit,
                "coefficients": [term.to_dict() for term in self.terms], "coefficients_note": self.terms_note,
                "effects": [effect.to_dict() for effect in self.effects], "effects_note": self.effects_note,
                "comparisons": [comparison.to_dict() for comparison in self.comparisons]}

    def lines(self) -> list[str]:
        out = [f"  explained by: {self.chosen}; effects and coefficients in {self.unit}"]
        out += [f"  compared: {comparison.name}: {comparison.detail}" for comparison in self.comparisons]
        if self.terms:
            out += [f"  coefficient: {term.name} {term.estimate:+.4g} (95% CI {term.low:+.4g} to "
                    f"{term.high:+.4g}, p {term.p_value:.2g})" for term in self.terms]
        elif self.terms_note:
            out.append(f"  coefficients: {self.terms_note}")
        for effect in self.effects:
            curve = ", ".join(f"{at} {value:+.3g}" for at, value in effect.points)
            out.append(f"  effect: {effect.column} {effect.share:.0%}, {effect.shape}: {curve}")
        if not self.effects and self.effects_note:
            out.append(f"  effects: {self.effects_note}")
        return out


def compare(name: str, first: str, second: str, first_scores: list[float], second_scores: list[float],
            higher: bool, what: str) -> Comparison:
    """``first`` against ``second`` fold by fold; ``what`` says what a clear win for ``first`` means."""
    gaps = np.array(first_scores) - np.array(second_scores)
    gaps = gaps if higher else -gaps
    difference = float(gaps.mean())
    error = float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")
    if not np.isfinite(error) or abs(difference) <= 2 * error:
        verdict = "within twice its standard error, so no clear difference"
    elif difference > 0:
        verdict = f"{first} is better beyond twice its standard error: {what}"
    else:
        verdict = f"{second} is better beyond twice its standard error"
    return Comparison(name, first, second, difference, error, verdict)


def effects(contributions: np.ndarray, frame: pd.DataFrame, columns: list[str]) -> tuple[Effect, ...]:
    """Each column's curve from the additive model's contributions on the training rows.

    A number's curve is its mean contribution in each of ``BANDS`` bands of
    its values; a category's, its mean contribution at each of its most common
    levels. A column's share is the spread of its contributions over the sum
    of every column's.
    """
    spreads = contributions.std(axis=0)
    total = float(spreads.sum()) or 1.0
    found = []
    for position, column in enumerate(columns):
        share = float(spreads[position]) / total
        if share < 0.005:
            continue
        values, effect = frame[column], contributions[:, position]
        if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object \
                or pd.api.types.is_string_dtype(values):
            text = values.astype("string").fillna("(missing)")
            common = text.value_counts().index[:LEVELS]
            means = pd.Series(effect).groupby(text.to_numpy()).mean()
            points = tuple((str(level), float(means[level])) for level in common)
            found.append(Effect(str(column), share, "a level each", points))
            continue
        numbers = to_numbers(values).to_numpy(dtype=float)
        present = np.isfinite(numbers)
        if present.sum() < 2:
            continue
        distinct = np.unique(numbers[present])
        if len(distinct) <= BANDS:
            # Few values: a point at each.
            means = pd.Series(effect[present]).groupby(numbers[present]).mean()
            points = tuple((f"{value:.4g}", float(means[value])) for value in means.index)
        else:
            # A point at the start of each band of a tenth of the rows.
            edges = np.unique(np.quantile(numbers[present], np.linspace(0, 1, BANDS + 1)))
            bands = np.clip(np.searchsorted(edges, numbers[present], side="right") - 1, 0, len(edges) - 2)
            means = pd.Series(effect[present]).groupby(bands).mean()
            points = tuple((f"{edges[band]:.4g}", float(means[band])) for band in means.index)
        if (~present).any():
            points += (("(missing)", float(effect[~present].mean())),)
        straight = np.corrcoef(numbers[present], effect[present])[0, 1] ** 2 if np.std(effect[present]) > 0 \
            and np.std(numbers[present]) > 0 else 1.0
        shape = (f"bends (a straight line explains {straight:.0%} of it)" if straight < STRAIGHT
                 else f"close to a straight line ({straight:.0%})")
        found.append(Effect(str(column), share, shape, points))
    return tuple(sorted(found, key=lambda effect: -effect.share)[:EFFECTS])


def coefficients(frame: pd.DataFrame, y: np.ndarray, task: str, classes: int,
                 encoded: list[str] = ()) -> tuple[tuple[Term, ...], str]:
    """The linear model's coefficients with 95% intervals from statsmodels, or why there are none.

    Numbers are filled with their median and scaled to a standard deviation of
    1, so a coefficient is the change for one standard deviation. Each
    category's most common level is the reference; levels with under 5 rows
    join it. An ``encoded`` column holds each value's mean target from the
    other rows. Regressions use heteroscedasticity-robust (HC3) errors.
    """
    import statsmodels.api as sm  # noqa: PLC0415

    if task == "classification" and classes > 1:
        return (), (f"a model of {classes + 1 if classes == 2 else classes} classes has a table of "
                    "coefficients for each class against the first; the effect curves cover it instead")
    blocks, names = [], []
    for column in frame.columns:
        values = frame[column]
        if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object \
                or pd.api.types.is_string_dtype(values):
            text = values.astype("string").fillna("(missing)")
            counts = text.value_counts()
            kept = [level for level in counts.index[1:] if counts[level] >= 5][:30]
            for level in kept:
                blocks.append((text == level).to_numpy(dtype=float))
                names.append(f"{column} = {level} (against {counts.index[0]})")
            continue
        numbers = to_numbers(values).to_numpy(dtype=float)
        if not np.isfinite(numbers).any():
            continue
        filled = np.where(np.isfinite(numbers), numbers, np.nanmedian(numbers))
        if filled.std() == 0:
            continue
        blocks.append((filled - filled.mean()) / filled.std())
        what = "its value's mean target, " if column in encoded else ""
        names.append(f"{column} ({what}per standard deviation, {filled.std():.4g})")
    if not blocks:
        return (), "no column varies"
    design = np.column_stack(blocks)
    # Columns that repeat others exactly make the fit singular; QR with pivoting keeps an independent set.
    from scipy.linalg import qr  # noqa: PLC0415

    _, upper, order = qr(design - design.mean(axis=0), mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(upper))
    rank = int((diagonal > diagonal.max() * 1e-8).sum()) if diagonal.size else 0
    keep = np.sort(order[:rank])
    design, names = design[:, keep], [names[index] for index in keep]
    design = sm.add_constant(design, has_constant="add")
    try:
        if task == "regression":
            fit = sm.OLS(y, design).fit(cov_type="HC3")
        else:
            fit = sm.Logit(y, design).fit(disp=0, maxiter=200)
    except (np.linalg.LinAlgError, ValueError) as error:
        return (), f"statsmodels could not fit the linear model ({error})"
    if not np.all(np.isfinite(fit.bse)):
        return (), "statsmodels gave no finite standard errors: some columns separate the classes completely"
    interval = fit.conf_int(0.05)
    terms = [Term(name, float(fit.params[index + 1]), float(interval[index + 1][0]),
                  float(interval[index + 1][1]), float(fit.pvalues[index + 1]))
             for index, name in enumerate(names)]
    z = np.abs(fit.params[1:] / fit.bse[1:])
    ranked = np.argsort(-z, kind="stable")[:TERMS]
    dropped = len(blocks) - len(names)
    note = f"; {dropped} columns repeating others left out" if dropped else ""
    return tuple(terms[index] for index in ranked), (
        f"{'ordinary least squares, HC3 errors' if task == 'regression' else 'logistic regression'} on "
        f"{len(y):,} rows and {len(names)} terms, the {min(TERMS, len(names))} with the largest z "
        f"shown{note}")
