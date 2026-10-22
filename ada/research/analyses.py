"""The study's analyses, each fitted with the test its design and variables call for.

Each analysis in ``study.yml`` gets one test, chosen from the design (independent,
paired, repeated or clustered), the outcome's type and the predictor's type;
``test:`` may name another test the same design allows. Every analysis uses its
complete cases and reports what it left out. Designs ADA does not fit are refused
with the reason and what to do instead, rather than fitted with the wrong model.
The key p-value of each analysis is adjusted within its family. Diagnostics
describe the model's assumptions in plain sentences and never change the test.
"""

from __future__ import annotations

import keyword
import math
import re
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .build import DATA_SECTIONS
from .processing import levels_of
from .provenance import Asset
from .results import ESTIMATE_COLUMNS, MEAN_COLUMNS, AnalysisResult
from .study import AnalysisSpec, Study
from .tables import Table, decimals_for, interval, number, p_value

#: Columns of ``AnalysisResult.estimates`` after ESTIMATE_COLUMNS. ``key`` marks the one row whose
#: p-value is adjusted across the family; ``measure`` says what the estimate is ("mean difference",
#: "slope", "odds ratio", "overall" for a test without an estimate, ...); ``statistic_name`` and
#: ``df2`` complete the statistic ("F" with ``df`` and ``df2``); ``test`` names the test behind the
#: row; ``adjustment`` says how ``p_adjusted`` was adjusted.
EXTRA_COLUMNS = ["key", "measure", "statistic_name", "df2", "test", "adjustment"]
COLUMNS = [*ESTIMATE_COLUMNS, *EXTRA_COLUMNS]

#: Time with at most this many distinct points is modelled as categorical in a change analysis.
CATEGORICAL_TIME_POINTS = 6
#: Fewer subjects or clusters than this make a variance component and Wald z tests unreliable.
FEW_UNITS = 10
#: Shapiro–Wilk is computed up to this many residuals (SciPy's p-value is not accurate beyond it).
SHAPIRO_LIMIT = 5000
#: An expected count below this switches a 2 × 2 comparison of proportions to Fisher's exact test.
SMALL_EXPECTED = 5
#: Tables drawn for the Monte Carlo version of Fisher's exact test on larger tables; seeded.
MONTE_CARLO_TABLES = 20_000
MONTE_CARLO_SEED = 20240501
#: Events per model parameter below this make a logistic regression's estimates unstable.
EVENTS_PER_PARAMETER = 10

ADJUSTMENT_NAMES = {
    "holm": "Holm's method",
    "bonferroni": "the Bonferroni method",
    "fdr_bh": "the Benjamini–Hochberg false discovery rate",
    "none": "no adjustment",
}
SHORT_ADJUSTMENT = {
    "holm": "Holm",
    "bonferroni": "Bonferroni",
    "fdr_bh": "Benjamini–Hochberg",
    "none": "none",
}

#: Tests ``test:`` may name, in words.
TEST_NAMES = {
    "welch-t": "Welch's t-test",
    "student-t": "Student's t-test",
    "mann-whitney": "Mann–Whitney U test",
    "welch-anova": "Welch's ANOVA",
    "anova": "One-way ANOVA",
    "kruskal": "Kruskal–Wallis test",
    "ols": "Linear regression (OLS)",
    "paired-t": "Paired t-test",
    "wilcoxon": "Wilcoxon signed-rank test",
    "mixed-model": "Linear mixed model",
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "rates": "Comparison of proportions",
    "logistic": "Logistic regression",
}

REFERENCES = {
    "welch": "Welch, B. L. (1947). The generalization of 'Student's' problem when several different "
    "population variances are involved. Biometrika, 34(1–2), 28–35. https://doi.org/10.1093/biomet/34.1-2.28",
    "welch_anova": "Welch, B. L. (1951). On the comparison of several mean values: an alternative approach. "
    "Biometrika, 38(3–4), 330–336. https://doi.org/10.1093/biomet/38.3-4.330",
    "student": "Student (1908). The probable error of a mean. Biometrika, 6(1), 1–25. "
    "https://doi.org/10.1093/biomet/6.1.1",
    "hedges": "Hedges, L. V. and Olkin, I. (1985). Statistical Methods for Meta-Analysis. Academic Press.",
    "hedges_1981": "Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size and "
    "related estimators. Journal of Educational Statistics, 6(2), 107–128. "
    "https://doi.org/10.3102/10769986006002107",
    "noncentral": "Steiger, J. H. and Fouladi, R. T. (1997). Noncentrality interval estimation and the "
    "evaluation of statistical models. In L. L. Harlow, S. A. Mulaik and J. H. Steiger (Eds.), What If "
    "There Were No Significance Tests? (pp. 221–257). Lawrence Erlbaum.",
    "cohen": "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). Lawrence "
    "Erlbaum.",
    "games_howell": "Games, P. A. and Howell, J. F. (1976). Pairwise multiple comparison procedures with "
    "unequal n's and/or variances: a Monte Carlo study. Journal of Educational Statistics, 1(2), 113–125. "
    "https://doi.org/10.3102/10769986001002113",
    "tukey": "Tukey, J. W. (1949). Comparing individual means in the analysis of variance. Biometrics, "
    "5(2), 99–114. https://doi.org/10.2307/3001913",
    "kramer": "Kramer, C. Y. (1956). Extension of multiple range tests to group means with unequal numbers "
    "of replications. Biometrics, 12(3), 307–310. https://doi.org/10.2307/3001469",
    "omega": "Olejnik, S. and Algina, J. (2003). Generalized eta and omega squared statistics: measures of "
    "effect size for some common research designs. Psychological Methods, 8(4), 434–447. "
    "https://doi.org/10.1037/1082-989X.8.4.434",
    "steiger": "Steiger, J. H. (2004). Beyond the F test: effect size confidence intervals and tests of "
    "close fit in the analysis of variance and contrast analysis. Psychological Methods, 9(2), 164–182. "
    "https://doi.org/10.1037/1082-989X.9.2.164",
    "mann_whitney": "Mann, H. B. and Whitney, D. R. (1947). On a test of whether one of two random "
    "variables is stochastically larger than the other. Annals of Mathematical Statistics, 18(1), 50–60. "
    "https://doi.org/10.1214/aoms/1177730491",
    "kruskal": "Kruskal, W. H. and Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. "
    "Journal of the American Statistical Association, 47(260), 583–621. "
    "https://doi.org/10.1080/01621459.1952.10483441",
    "wilcoxon": "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), "
    "80–83. https://doi.org/10.2307/3001968",
    "hodges_lehmann": "Hodges, J. L. and Lehmann, E. L. (1963). Estimates of location based on rank tests. "
    "Annals of Mathematical Statistics, 34(2), 598–611. https://doi.org/10.1214/aoms/1177704172",
    "bauer": "Bauer, D. F. (1972). Constructing confidence sets using rank statistics. Journal of the "
    "American Statistical Association, 67(339), 687–690. https://doi.org/10.1080/01621459.1972.10481279",
    "laird_ware": "Laird, N. M. and Ware, J. H. (1982). Random-effects models for longitudinal data. "
    "Biometrics, 38(4), 963–974. https://doi.org/10.2307/2529876",
    "reml": "Patterson, H. D. and Thompson, R. (1971). Recovery of inter-block information when block sizes "
    "are unequal. Biometrika, 58(3), 545–554. https://doi.org/10.1093/biomet/58.3.545",
    "icc": "Snijders, T. A. B. and Bosker, R. J. (2012). Multilevel Analysis: An Introduction to Basic and "
    "Advanced Multilevel Modeling (2nd ed.). Sage.",
    "pearson": "Pearson, K. (1896). Mathematical contributions to the theory of evolution. III. Regression, "
    "heredity, and panmixia. Philosophical Transactions of the Royal Society A, 187, 253–318. "
    "https://doi.org/10.1098/rsta.1896.0007",
    "fisher_z": "Fisher, R. A. (1915). Frequency distribution of the values of the correlation coefficient "
    "in samples from an indefinitely large population. Biometrika, 10(4), 507–521. "
    "https://doi.org/10.2307/2331838",
    "spearman": "Spearman, C. (1904). The proof and measurement of association between two things. American "
    "Journal of Psychology, 15(1), 72–101. https://doi.org/10.2307/1412159",
    "newcombe": "Newcombe, R. G. (1998). Interval estimation for the difference between independent "
    "proportions: comparison of eleven methods. Statistics in Medicine, 17(8), 873–890. "
    "https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I",
    "chi2": "Pearson, K. (1900). On the criterion that a given system of deviations from the probable in "
    "the case of a correlated system of variables is such that it can be reasonably supposed to have arisen "
    "from random sampling. Philosophical Magazine, 50(302), 157–175. "
    "https://doi.org/10.1080/14786440009463897",
    "fisher_exact": "Fisher, R. A. (1922). On the interpretation of χ² from contingency tables, and the "
    "calculation of P. Journal of the Royal Statistical Society, 85(1), 87–94. "
    "https://doi.org/10.2307/2340521",
    "freeman_halton": "Freeman, G. H. and Halton, J. H. (1951). Note on an exact treatment of contingency, "
    "goodness of fit and other problems of significance. Biometrika, 38(1–2), 141–149. "
    "https://doi.org/10.1093/biomet/38.1-2.141",
    "woolf": "Woolf, B. (1955). On estimating the relation between blood group and disease. Annals of Human "
    "Genetics, 19(4), 251–253. https://doi.org/10.1111/j.1469-1809.1955.tb01348.x",
    "wilson": "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. "
    "Journal of the American Statistical Association, 22(158), 209–212. "
    "https://doi.org/10.1080/01621459.1927.10502953",
    "cramer": "Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.",
    "logistic": "Cox, D. R. (1958). The regression analysis of binary sequences. Journal of the Royal "
    "Statistical Society B, 20(2), 215–242. https://doi.org/10.1111/j.2517-6161.1958.tb00292.x",
    "holm": "Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal "
    "of Statistics, 6(2), 65–70.",
    "bonferroni": "Dunn, O. J. (1961). Multiple comparisons among means. Journal of the American "
    "Statistical Association, 56(293), 52–64. https://doi.org/10.1080/01621459.1961.10482090",
    "fdr_bh": "Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and "
    "powerful approach to multiple testing. Journal of the Royal Statistical Society B, 57(1), 289–300. "
    "https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
    "shapiro": "Shapiro, S. S. and Wilk, M. B. (1965). An analysis of variance test for normality (complete "
    "samples). Biometrika, 52(3–4), 591–611. https://doi.org/10.1093/biomet/52.3-4.591",
    "levene": "Brown, M. B. and Forsythe, A. B. (1974). Robust tests for the equality of variances. Journal "
    "of the American Statistical Association, 69(346), 364–367. "
    "https://doi.org/10.1080/01621459.1974.10482955",
    "breusch_pagan": "Breusch, T. S. and Pagan, A. R. (1979). A simple test for heteroscedasticity and "
    "random coefficient variation. Econometrica, 47(5), 1287–1294. https://doi.org/10.2307/1911963",
    "vif": "Marquardt, D. W. (1970). Generalized inverses, ridge regression, biased linear estimation, and "
    "nonlinear estimation. Technometrics, 12(3), 591–612. https://doi.org/10.2307/1267205",
}

#: Units read in the singular after "per": a slope of 0.2 kg per year.
SINGULAR_UNITS = {
    "years": "year",
    "months": "month",
    "weeks": "week",
    "days": "day",
    "hours": "hour",
    "minutes": "minute",
    "seconds": "second",
}

#: Short symbols for effect sizes in the tables.
EFFECT_SYMBOLS = {
    "Hedges' g": "g",
    "Cohen's dz": "dz",
    "omega squared": "ω²",
    "partial omega squared": "partial ω²",
    "Pearson's r": "r",
    "odds ratio": "OR",
    "Cramér's V": "V",
}


class Unsupported(Exception):
    """ADA does not fit this analysis; the message says why and what to do instead."""


def run(result) -> None:
    """Run each analysis in ``study.yml``, adjust within families, and write the results assets."""
    study = result.study
    fitted = {spec.id: analyse(study, result.data, spec) for spec in study.analyses}
    adjust_families(study, fitted)
    result.results.update(fitted)
    for ident, found in fitted.items():
        if not found.supported:
            result.warnings.append(f"Analysis {ident} was not fitted: {found.reason}")
        for warning in found.warnings:
            result.warnings.append(f"Analysis {ident}: {warning}")
    for planned in (True, False):
        _add_results_table(result, [found for found in fitted.values() if found.planned == planned], planned)
    for found in fitted.values():
        _add_model_record(result, found)


def analyse(study: Study, data: pd.DataFrame, spec: AnalysisSpec) -> AnalysisResult:
    """Fit one analysis on its complete cases; one ADA does not fit comes back with the reason."""
    found = AnalysisResult(spec=spec, design=study.design)
    try:
        _Analysis(study, data, spec, found).fit()
    except (Unsupported, ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError) as reason:
        found.supported = False
        found.reason = (
            str(reason)
            if isinstance(reason, Unsupported)
            else f"The model could not be fitted ({reason}). Write it under `custom:`."
        )
        found.estimates = _estimates([])
        found.effect = {}
        found.group_means = found.fitted = found.residuals = None
        found.summary = ""
    return found


def key_row(found: AnalysisResult) -> pd.Series | None:
    """The headline row of an analysis: the one whose p-value is adjusted across its family."""
    estimates = found.estimates
    if not found.supported or estimates is None or estimates.empty or "key" not in estimates:
        return None
    chosen = estimates[estimates["key"].astype(bool)]
    return chosen.iloc[0] if len(chosen) else None


def adjust_families(study: Study, results: dict[str, AnalysisResult]) -> None:
    """Adjust the key p-value of each analysis with the others in its family.

    Families are what the researcher declared under ``family``; an analysis without one is its
    own family. ``multiple_comparisons: none`` leaves ``p_adjusted`` empty, so nothing reads as
    adjusted that was not.
    """
    from statsmodels.stats.multitest import multipletests  # noqa: PLC0415

    method = study.decisions.multiple_comparisons
    families: dict[str, list[str]] = {}
    for ident, found in results.items():
        row = key_row(found)
        if row is not None and pd.notna(row["p"]):
            families.setdefault(found.family, []).append(ident)
    for family, members in families.items():
        p_values = [float(key_row(results[ident])["p"]) for ident in members]
        if method == "none":
            adjusted = [math.nan] * len(p_values)
            text = "not adjusted (multiple_comparisons: none)"
        else:
            adjusted = multipletests(p_values, alpha=study.decisions.alpha, method=method)[1]
            size = len(members)
            noun = "analysis" if size == 1 else "analyses"
            text = f"{SHORT_ADJUSTMENT[method]} over family {family} ({size} {noun})"
        for ident, value in zip(members, adjusted, strict=True):
            estimates = results[ident].estimates
            position = estimates.index[estimates["key"].astype(bool)][0]
            estimates.loc[position, "p_adjusted"] = float(value)
            estimates.loc[position, "adjustment"] = text
            reference = REFERENCES.get(method)
            if reference and reference not in results[ident].references:
                results[ident].references = (*results[ident].references, reference)


def _estimates(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    for column in ("estimate", "ci_low", "ci_high", "statistic", "df", "p", "p_adjusted", "df2"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    frame["key"] = frame["key"].fillna(False).astype(bool)
    for column in ("term", "label", "measure", "statistic_name", "test", "adjustment"):
        frame[column] = frame[column].fillna("").astype(str)
    return frame


def _row(
    term: str,
    label: str,
    estimate=math.nan,
    low=math.nan,
    high=math.nan,
    statistic=math.nan,
    df=math.nan,
    p=math.nan,
    *,
    measure: str,
    statistic_name: str = "",
    df2=math.nan,
    test: str = "",
    key: bool = False,
    p_adjusted=math.nan,
    adjustment: str = "",
) -> dict:
    return {
        "term": term,
        "label": label,
        "estimate": _float(estimate),
        "ci_low": _float(low),
        "ci_high": _float(high),
        "statistic": _float(statistic),
        "df": _float(df),
        "p": _float(p),
        "p_adjusted": _float(p_adjusted),
        "key": key,
        "measure": measure,
        "statistic_name": statistic_name,
        "df2": _float(df2),
        "test": test,
        "adjustment": adjustment,
    }


def _float(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def _plain(value: float, digits: int = 3) -> str:
    """A number for a diagnostic sentence, to a few significant figures."""
    if value is None or not math.isfinite(value):
        return "–"
    return f"{value:.{digits}g}".replace("-", "−")


def _p_text(value: float) -> str:
    text = p_value(value)
    return f"p {text.replace('<', '< ')}" if text.startswith("<") else f"p = {text}"


def _safe_names(names: list[str]) -> dict[str, str]:
    """Formula-safe names, readable in a model summary: "Body weight (g)" → Body_weight_g."""
    taken: set[str] = set()
    safe = {}
    for name in names:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or "v"
        if base[0].isdigit():
            base = f"v_{base}"
        if keyword.iskeyword(base) or base in {"C", "I", "Q"}:
            base = f"{base}_"
        candidate, count = base, 2
        while candidate in taken:
            candidate, count = f"{base}_{count}", count + 1
        taken.add(candidate)
        safe[name] = candidate
    return safe


def _time_label(value) -> str:
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _clean_summary(text: str) -> str:
    """The summary without the clock time and date statsmodels prints, so a rebuild gives the same file."""
    text = re.sub(r"(Date:\s+)(\w{3}, \d{2} \w{3} \d{4})", lambda m: m.group(1) + " " * len(m.group(2)), text)
    return re.sub(r"(Time:\s+)(\d{2}:\d{2}:\d{2})", lambda m: m.group(1) + " " * len(m.group(2)), text)


def _record_warnings(caught) -> list[str]:
    messages = []
    for item in caught:
        if issubclass(item.category, FutureWarning | DeprecationWarning):
            continue
        message = " ".join(str(item.message).split())
        if message not in messages:
            messages.append(message)
    return messages


class _Analysis:
    """One analysis being fitted: its complete cases, the chosen test, and the result being filled."""

    def __init__(self, study: Study, data: pd.DataFrame, spec: AnalysisSpec, found: AnalysisResult):
        self.study = study
        self.data = data
        self.spec = spec
        self.found = found
        self.confidence = study.decisions.confidence
        self.alpha = 1 - study.decisions.confidence
        self.adjust = study.decisions.multiple_comparisons
        self.references: list[str] = []
        self.unit: str | None = None
        self.unit_word = ""
        self.fitted_formula = ""

    # The steps every analysis goes through.

    def fit(self) -> None:
        spec, study, found = self.spec, self.study, self.found
        self._check_columns()
        self.unit, self.unit_word = self._unit()
        needed = [spec.outcome, spec.predictor, spec.by, *spec.covariates, self.unit]
        self.needed = list(dict.fromkeys(name for name in needed if name))
        complete = self.data[self.needed].notna().all(axis=1)
        found.n_missing = int((~complete).sum())
        self.rows = self.data.loc[complete]
        found.n = len(self.rows)
        if found.n_missing:
            counts = [
                f"{name} {int(self.data[name].isna().sum()):,}"
                for name in self.needed
                if self.data[name].isna().any()
            ]
            found.diagnostics.append(
                f"{found.n_missing:,} of {len(self.data):,} rows were left out because a value this analysis "
                f"needs was missing (missing values: {', '.join(counts)})."
            )
        if found.n < 3:
            raise Unsupported(
                f"Only {found.n} rows have every value this analysis needs; it needs at least 3."
            )
        self.names = _safe_names(self.needed)
        self.outcome_kind = self._outcome_kind()
        if spec.by and spec.kind != "change":
            raise Unsupported(
                f"`by` applies to change analyses only. To compare {spec.outcome} across two variables at "
                "once, add the second as a covariate, or write the model under `custom:`."
            )
        test = self._choose_test()
        self.test = test
        if self.outcome_kind == "binary":
            self._fit_binary(test)
        elif spec.kind == "compare":
            self._fit_compare(test)
        elif spec.kind == "relate":
            self._fit_relate(test)
        else:
            self._fit_change()
        found.references = tuple(dict.fromkeys(self.references))
        if found.subjects is None and self.unit and study.design != "clustered":
            found.subjects = int(self.rows[self.unit].nunique())
        if self.fitted_formula:
            renamed = [f"{safe} = {name}" for name, safe in self.names.items() if safe != name]
            header = [f"Formula as fitted: {self.fitted_formula}"]
            if renamed:
                header.append(f"Names in the fitted formula: {'; '.join(renamed)}")
            found.summary = "\n".join(header) + "\n\n" + found.summary

    def _check_columns(self) -> None:
        spec, study = self.spec, self.study
        for name in [spec.outcome, spec.predictor, spec.by, *spec.covariates]:
            if name and name not in self.data:
                raise Unsupported(f"{name} is not a column of the processed data.")
        predictor = study.variable(spec.predictor)
        if spec.kind == "compare" and predictor.type not in ("categorical", "binary"):
            hint = " Use `relate:` for a numeric predictor." if predictor.type == "numeric" else ""
            raise Unsupported(
                f"`compare:` needs a categorical or binary group, but {spec.predictor} is {predictor.type}."
                + hint
            )
        if spec.kind == "relate" and predictor.type != "numeric":
            hint = (
                " Use `compare:` for a categorical group."
                if predictor.type in ("categorical", "binary")
                else ""
            )
            raise Unsupported(
                f"`relate:` needs a numeric predictor, but {spec.predictor} is {predictor.type}." + hint
            )
        if spec.kind == "change" and predictor.type not in ("numeric", "categorical", "binary"):
            raise Unsupported(
                f"The time variable {spec.predictor} is {predictor.type}. Add a numeric column of elapsed "
                "time (such as days since the first measurement) or a categorical time point, and use that."
            )
        if spec.by and study.variable(spec.by).type not in ("categorical", "binary"):
            raise Unsupported(
                f"`by` needs a categorical group, but {spec.by} is {study.variable(spec.by).type}."
            )
        for name in spec.covariates:
            kind = study.variable(name).type
            if kind not in ("numeric", "categorical", "binary"):
                raise Unsupported(
                    f"The covariate {name} is {kind}; covariates must be numeric, categorical or binary."
                )

    def _unit(self) -> tuple[str | None, str]:
        """The column whose rows are not independent, and what to call it: subject or cluster."""
        study, spec = self.study, self.spec
        if spec.kind == "change":
            if study.design == "independent":
                raise Unsupported(
                    "A change analysis follows the same subjects over time, but the design is independent "
                    "(each row a different subject). Set `design.type: repeated` and `design.subject`."
                )
            if not study.subject:
                raise Unsupported(
                    "A change analysis needs `design.subject`: the column naming who or what was measured "
                    "at each time."
                )
            if study.design == "clustered" and study.cluster and study.cluster != study.subject:
                raise Unsupported(
                    f"Subjects measured over time within clusters ({study.cluster}) need random effects at "
                    "two levels, which ADA does not fit. Write the model under `custom:`."
                )
            return study.subject, "subject"
        if study.design in ("paired", "repeated"):
            return study.subject, "subject"
        if study.design == "clustered":
            return study.cluster, "cluster"
        return None, ""

    def _outcome_kind(self) -> str:
        variable = self.study.variable(self.spec.outcome)
        if variable.type == "numeric":
            values = self.rows[self.spec.outcome].astype(float)
            if values.nunique() < 2:
                raise Unsupported(f"{self.spec.outcome} has the same value in every row this analysis uses.")
            return "continuous"
        if variable.type in ("categorical", "binary"):
            levels = levels_of(self.rows, self.spec.outcome)
            if len(levels) > 2:
                raise Unsupported(
                    f"{self.spec.outcome} has {len(levels)} categories ({', '.join(levels)}). Ordinal and "
                    "multinomial outcomes are not fitted; write the model under `custom:`, or recode the "
                    "outcome into two categories with a `recode` step and a reason."
                )
            if len(levels) < 2:
                raise Unsupported(
                    f"{self.spec.outcome} has only one value ({', '.join(levels)}) in the rows used."
                )
            self.event = _event_level(variable, levels)
            self.non_event = next(level for level in levels if level != self.event)
            return "binary"
        raise Unsupported(
            f"The outcome {self.spec.outcome} is {variable.type}; outcomes must be numeric or binary."
        )

    def _choose_test(self) -> str:
        allowed, auto, where = self._allowed_tests()
        chosen = self.spec.test
        if chosen == "auto":
            return auto
        if chosen not in allowed:
            raise Unsupported(
                f"`test: {chosen}` does not fit {where}. Allowed here: auto, {', '.join(allowed)}."
            )
        return chosen

    def _allowed_tests(self) -> tuple[list[str], str, str]:
        spec, design = self.spec, self.study.design
        covariates = bool(spec.covariates)
        if self.outcome_kind == "binary":
            if design != "independent" or spec.kind == "change":
                raise Unsupported(
                    f"{spec.outcome} is binary, and binary outcomes in {design} designs (rows that are not "
                    "independent) are not fitted. Write the model, such as a generalised linear mixed model "
                    "or GEE, under `custom:`."
                )
            if spec.kind == "compare":
                if covariates:
                    return ["logistic"], "logistic", "a binary outcome compared with covariates"
                return ["rates", "logistic"], "rates", "a binary outcome compared between groups"
            return ["logistic"], "logistic", "a binary outcome related to a numeric predictor"
        if spec.kind == "change":
            return ["mixed-model"], "mixed-model", "a change over time"
        if spec.kind == "relate":
            if design == "independent":
                if covariates:
                    return ["ols"], "ols", "a relationship adjusted for covariates"
                return ["ols", "pearson", "spearman"], "ols", "a relationship in an independent design"
            return ["mixed-model"], "mixed-model", f"a relationship in a {design} design"
        levels = levels_of(self.rows, spec.compare)
        if design == "independent":
            if covariates:
                return ["ols"], "ols", "a comparison adjusted for covariates"
            if len(levels) == 2:
                return (
                    ["welch-t", "student-t", "mann-whitney", "ols"],
                    "welch-t",
                    "a comparison of two independent groups",
                )
            return (
                ["welch-anova", "anova", "kruskal", "ols"],
                "welch-anova",
                "a comparison of three or more independent groups",
            )
        if design == "paired":
            if covariates:
                return ["mixed-model"], "mixed-model", "a paired comparison adjusted for covariates"
            return ["paired-t", "wilcoxon", "mixed-model"], "paired-t", "a paired comparison"
        return ["mixed-model"], "mixed-model", f"a comparison in a {design} design"

    # Shared pieces.

    def _levels(self, column: str) -> list[str]:
        """The levels of a group in the rows used, refusing a group that cannot be compared."""
        present = levels_of(self.rows, column)
        counts = self.rows[column].astype(str).value_counts()
        if len(present) < 2:
            shown = ", ".join(present) or "none"
            raise Unsupported(
                f"{column} has fewer than 2 levels in the rows this analysis uses ({shown}), so there is "
                "nothing to compare."
            )
        small = [f"{level} ({int(counts[level])})" for level in present if counts[level] < 2]
        if small:
            raise Unsupported(
                f"Each level of {column} needs at least 2 rows; {', '.join(small)} "
                f"{'has' if len(small) == 1 else 'have'} fewer after leaving out missing values."
            )
        declared = [str(level) for level in self.study.variable(column).levels]
        absent = [level for level in declared if level not in present]
        if absent:
            self.found.diagnostics.append(
                f"{', '.join(absent)} of {column} had no rows with every value this analysis needs and "
                f"{'is' if len(absent) == 1 else 'are'} left out."
            )
        if declared and declared[0] not in present:
            self.found.diagnostics.append(
                f"The declared reference level {declared[0]} has no rows, so {present[0]} is the reference "
                "instead."
            )
        return present

    def _work(self, categorical: tuple[str, ...] = ()) -> pd.DataFrame:
        """The rows used, with formula-safe names, numbers as floats and groups in study order."""
        work = pd.DataFrame(index=self.rows.index)
        for name in self.needed:
            safe = self.names[name]
            column = self.rows[name]
            variable = self.study.variable(name)
            if name == self.spec.outcome and self.outcome_kind == "binary":
                work[safe] = (column.astype(str) == self.event).astype(int)
            elif name == self.unit:
                work[safe] = column.astype(str)
            elif name in categorical or variable.type in ("categorical", "binary"):
                labels = column.astype(str) if variable.type != "numeric" else column.map(_time_label)
                order = (
                    levels_of(self.rows, name)
                    if variable.type != "numeric"
                    else [_time_label(value) for value in sorted(column.unique())]
                )
                work[safe] = pd.Categorical(labels, categories=order)
            else:
                work[safe] = column.astype(float)
        return work

    def _covariate_terms(self) -> list[str]:
        terms = []
        for name in self.spec.covariates:
            safe = self.names[name]
            if self.study.variable(name).type in ("categorical", "binary"):
                if self.rows[name].nunique() < 2:
                    raise Unsupported(f"The covariate {name} has a single value in the rows used.")
                terms.append(f"C({safe})")
            else:
                if self.rows[name].astype(float).nunique() < 2:
                    raise Unsupported(f"The covariate {name} has a single value in the rows used.")
                terms.append(safe)
        return terms

    def _readable_formula(self, right: str, random: bool = False) -> str:
        covariates = " + ".join(self.spec.covariates)
        text = f"{self.spec.outcome} ~ {right}" + (f" + {covariates}" if covariates else "")
        return text + (f" + (1 | {self.unit})" if random else "")

    def _means(
        self,
        values: pd.Series,
        groups: pd.Series | None,
        group_order: list[str],
        times: pd.Series | None = None,
        time_order: list | None = None,
        units: pd.Series | None = None,
        binary: bool = False,
    ) -> pd.DataFrame:
        """Observed means with t-based intervals (Wilson for proportions), averaging each unit's rows first.

        Averaging within each subject or cluster first keeps the intervals honest in repeated and
        clustered designs: they reflect the number of subjects, not the number of rows.
        """
        frame = pd.DataFrame({"value": values.astype(float)})
        frame["group"] = groups.astype(str) if groups is not None else "All"
        frame["time"] = times.to_numpy() if times is not None else 0
        if units is not None:
            frame["unit"] = units.astype(str)
            frame = frame.groupby(["unit", "group", "time"], sort=False)["value"].mean().reset_index()
        order = group_order if groups is not None else ["All"]
        rows = []
        for group in order:
            for moment in time_order if times is not None else [None]:
                mask = frame["group"] == group
                if moment is not None:
                    mask &= frame["time"] == moment
                selected = frame.loc[mask, "value"]
                n = int(selected.size)
                mean = float(selected.mean()) if n else math.nan
                sd = float(selected.std(ddof=1)) if n > 1 else math.nan
                se = sd / math.sqrt(n) if n > 1 else math.nan
                if binary and n:
                    from statsmodels.stats.proportion import proportion_confint  # noqa: PLC0415

                    low, high = proportion_confint(
                        int(round(selected.sum())), n, alpha=self.alpha, method="wilson"
                    )
                elif n > 1:
                    margin = stats.t.ppf(1 - self.alpha / 2, n - 1) * se
                    low, high = mean - margin, mean + margin
                else:
                    low = high = math.nan
                rows.append(
                    {
                        "group": group,
                        "time": moment,
                        "n": n,
                        "mean": mean,
                        "sd": sd,
                        "se": se,
                        "ci_low": float(low),
                        "ci_high": float(high),
                    }
                )
        return pd.DataFrame(rows, columns=MEAN_COLUMNS)

    def _shapiro(self, residuals, what: str = "residuals") -> None:
        values = np.asarray(residuals, dtype=float)
        values = values[np.isfinite(values)]
        if values.size < 3:
            return
        if values.size > SHAPIRO_LIMIT:
            self.found.diagnostics.append(
                f"Normality of the {what} was not tested: Shapiro–Wilk is limited to {SHAPIRO_LIMIT:,} "
                f"values and there are {values.size:,}; see the Q–Q plot."
            )
            return
        if np.ptp(values) == 0:
            return
        test = stats.shapiro(values)
        verdict = (
            "consistent with a normal distribution"
            if test.pvalue >= self.study.decisions.alpha
            else "departing from a normal distribution"
        )
        self.found.diagnostics.append(
            f"Shapiro–Wilk test of the {what}: W = {test.statistic:.3f}, {_p_text(test.pvalue)} (n = "
            f"{values.size:,}), {verdict}."
        )
        self.references.append(REFERENCES["shapiro"])

    def _levene(self, samples: list[np.ndarray], group: str, note: str = "") -> None:
        if len(samples) < 2 or any(len(sample) < 2 for sample in samples):
            return
        if all(np.ptp(sample) == 0 for sample in samples):
            return
        test = stats.levene(*samples, center="median")
        self.found.diagnostics.append(
            f"Levene's test (median-centred) for equal variances across {group}: "
            f"W = {test.statistic:.2f}, {_p_text(test.pvalue)}." + (f" {note}" if note else "")
        )
        self.references.append(REFERENCES["levene"])

    def _sizes(self, counts: dict[str, int], what: str = "Rows per group") -> None:
        self.found.groups = {str(level): int(count) for level, count in counts.items()}
        listed = ", ".join(f"{level} (n = {count:,})" for level, count in counts.items())
        self.found.diagnostics.append(f"{what}: {listed}.")

    def _adjust_within(self, rows: list[dict], label: str) -> None:
        """Adjust the contrasts inside one analysis with the study's method: each set controls its error."""
        from statsmodels.stats.multitest import multipletests  # noqa: PLC0415

        chosen = [row for row in rows if math.isfinite(row["p"])]
        if not chosen:
            return
        if self.adjust == "none":
            for row in chosen:
                row["adjustment"] = "not adjusted (multiple_comparisons: none)"
            return
        adjusted = multipletests([row["p"] for row in chosen], method=self.adjust)[1]
        for row, value in zip(chosen, adjusted, strict=True):
            row["p_adjusted"] = float(value)
            row["adjustment"] = f"{SHORT_ADJUSTMENT[self.adjust]} over the {len(chosen)} {label}"
        self.references.append(REFERENCES[self.adjust])

    # Continuous outcomes compared between groups.

    def _fit_compare(self, test: str) -> None:
        design = self.study.design
        if design == "independent":
            if test == "ols":
                self._ols_compare()
            elif test in ("welch-t", "student-t"):
                self._two_sample(equal_var=test == "student-t")
            elif test == "mann-whitney":
                self._mann_whitney()
            elif test in ("welch-anova", "anova"):
                self._one_way(welch=test == "welch-anova")
            else:
                self._kruskal()
        elif design == "paired" and test in ("paired-t", "wilcoxon"):
            self._paired(test)
        else:
            self._mixed_compare()

    def _samples(self) -> tuple[list[str], list[np.ndarray], pd.DataFrame]:
        levels = self._levels(self.spec.compare)
        work = self._work()
        group, outcome = self.names[self.spec.compare], self.names[self.spec.outcome]
        samples = [work.loc[work[group] == level, outcome].to_numpy(dtype=float) for level in levels]
        self._sizes({level: len(sample) for level, sample in zip(levels, samples, strict=True)})
        self.found.group_means = self._means(work[outcome], work[group], levels)
        return levels, samples, work

    def _group_fit(self, work: pd.DataFrame) -> None:
        group, outcome = self.names[self.spec.compare], self.names[self.spec.outcome]
        fitted = work.groupby(group, observed=True)[outcome].transform("mean")
        self.found.fitted = fitted.rename("fitted")
        self.found.residuals = (work[outcome] - fitted).rename("residual")

    def _two_sample(self, equal_var: bool) -> None:
        found = self.found
        levels, (reference, other), work = self._samples()
        result = stats.ttest_ind(other, reference, equal_var=equal_var)
        ci = result.confidence_interval(confidence_level=self.confidence)
        name = "Welch's t-test" if not equal_var else "Student's t-test"
        found.test = name
        found.method = f"scipy.stats.ttest_ind(equal_var={equal_var})"
        found.statistic_name = "t"
        difference = float(other.mean() - reference.mean())
        found.estimates = _estimates(
            [
                _row(
                    f"{self.spec.compare}[{levels[1]} - {levels[0]}]",
                    f"{levels[1]} − {levels[0]}",
                    difference,
                    ci.low,
                    ci.high,
                    result.statistic,
                    result.df,
                    result.pvalue,
                    measure="mean difference",
                    statistic_name="t",
                    test=name,
                    key=True,
                )
            ]
        )
        found.effect = hedges_g(reference, other, self.confidence)
        self._group_fit(work)
        self._shapiro(found.residuals)
        self._levene(
            [reference, other],
            self.spec.compare,
            "Welch's test does not assume equal variances." if not equal_var else "",
        )
        self.references += [REFERENCES["welch" if not equal_var else "student"]]
        self.references += [REFERENCES["hedges"], REFERENCES["hedges_1981"], REFERENCES["noncentral"]]
        found.summary = self._test_summary(levels, [reference, other])

    def _one_way(self, welch: bool) -> None:
        from statsmodels.stats.oneway import anova_oneway  # noqa: PLC0415

        found = self.found
        levels, samples, work = self._samples()
        result = anova_oneway(samples, use_var="unequal" if welch else "equal")
        df1, df2 = float(result.df_num), float(result.df_denom)
        name = "Welch's ANOVA" if welch else "One-way ANOVA"
        found.test = name + (" with Games–Howell contrasts" if welch else " with Tukey–Kramer contrasts")
        found.method = (
            f"statsmodels.stats.oneway.anova_oneway(use_var='{'unequal' if welch else 'equal'}'); "
            + (
                "Games–Howell contrasts from scipy.stats.studentized_range"
                if welch
                else "Tukey–Kramer contrasts from scipy.stats.tukey_hsd"
            )
        )
        found.statistic_name = "F"
        rows = [
            _row(
                self.spec.compare,
                f"{self.spec.compare} (overall)",
                statistic=result.statistic,
                df=df1,
                p=result.pvalue,
                measure="overall",
                statistic_name="F",
                df2=df2,
                test=name,
                key=True,
            )
        ]
        contrasts = (
            games_howell(levels, samples, self.confidence)
            if welch
            else tukey_kramer(levels, samples, self.confidence)
        )
        for contrast in contrasts:
            rows.append(
                _row(
                    f"{self.spec.compare}[{contrast['level']} - {contrast['reference']}]",
                    f"{contrast['level']} − {contrast['reference']}",
                    contrast["difference"],
                    contrast["ci_low"],
                    contrast["ci_high"],
                    contrast["t"],
                    contrast["df"],
                    contrast["p"],
                    measure="mean difference",
                    statistic_name="t",
                    test="Games–Howell" if welch else "Tukey–Kramer",
                    p_adjusted=contrast["p_adjusted"],
                    adjustment=(
                        f"{'Games–Howell' if welch else 'Tukey–Kramer'} over the {len(contrasts)} pairs"
                    ),
                )
            )
        found.estimates = _estimates(rows)
        found.effect = omega_squared(float(result.statistic), df1, df2, self.confidence)
        self._group_fit(work)
        self._shapiro(found.residuals)
        self._levene(
            samples,
            self.spec.compare,
            "Welch's ANOVA does not assume equal variances." if welch else "",
        )
        self.references += [
            REFERENCES["welch_anova"] if welch else REFERENCES["student"],
            REFERENCES["games_howell"] if welch else REFERENCES["tukey"],
        ]
        if not welch:
            self.references.append(REFERENCES["kramer"])
        self.references += [REFERENCES["omega"], REFERENCES["steiger"]]
        found.summary = self._test_summary(levels, samples)

    def _mann_whitney(self) -> None:
        found = self.found
        levels, (reference, other), work = self._samples()
        result = stats.mannwhitneyu(other, reference, alternative="two-sided")
        estimate, low, high = hodges_lehmann_shift(reference, other, self.confidence)
        found.test = "Mann–Whitney U test"
        found.method = "scipy.stats.mannwhitneyu; Hodges–Lehmann shift with a test-inversion interval"
        found.statistic_name = "U"
        found.estimates = _estimates(
            [
                _row(
                    f"{self.spec.compare}[{levels[1]} - {levels[0]}]",
                    f"{levels[1]} − {levels[0]}",
                    estimate,
                    low,
                    high,
                    result.statistic,
                    p=result.pvalue,
                    measure="median difference",
                    statistic_name="U",
                    test="Mann–Whitney U test",
                    key=True,
                )
            ]
        )
        self._interval_note(low, high)
        self.references += [REFERENCES["mann_whitney"], REFERENCES["hodges_lehmann"], REFERENCES["bauer"]]
        found.summary = self._test_summary(levels, [reference, other])

    def _kruskal(self) -> None:
        found = self.found
        levels, samples, work = self._samples()
        result = stats.kruskal(*samples)
        found.test = "Kruskal–Wallis test with pairwise Mann–Whitney tests"
        found.method = "scipy.stats.kruskal; scipy.stats.mannwhitneyu for each pair"
        found.statistic_name = "H"
        rows = [
            _row(
                self.spec.compare,
                f"{self.spec.compare} (overall)",
                statistic=result.statistic,
                df=len(levels) - 1,
                p=result.pvalue,
                measure="overall",
                statistic_name="H",
                test="Kruskal–Wallis test",
                key=True,
            )
        ]
        pairs = []
        for first in range(len(levels)):
            for second in range(first + 1, len(levels)):
                test = stats.mannwhitneyu(samples[second], samples[first], alternative="two-sided")
                estimate, low, high = hodges_lehmann_shift(samples[first], samples[second], self.confidence)
                pairs.append(
                    _row(
                        f"{self.spec.compare}[{levels[second]} - {levels[first]}]",
                        f"{levels[second]} − {levels[first]}",
                        estimate,
                        low,
                        high,
                        test.statistic,
                        p=test.pvalue,
                        measure="median difference",
                        statistic_name="U",
                        test="Mann–Whitney U test",
                    )
                )
        self._adjust_within(pairs, "pairs")
        found.estimates = _estimates(rows + pairs)
        found.diagnostics.append(
            f"Intervals for the pairwise median differences are unadjusted {self._percent()} intervals."
        )
        self.references += [
            REFERENCES["kruskal"],
            REFERENCES["mann_whitney"],
            REFERENCES["hodges_lehmann"],
            REFERENCES["bauer"],
        ]
        found.summary = self._test_summary(levels, samples)

    def _ols_compare(self) -> None:
        import statsmodels.formula.api as smf  # noqa: PLC0415

        found = self.found
        levels = self._levels(self.spec.compare)
        work = self._work()
        group, outcome = self.names[self.spec.compare], self.names[self.spec.outcome]
        right = " + ".join([f"C({group})", *self._covariate_terms()])
        formula = f"{outcome} ~ {right}"
        fit = smf.ols(formula, data=work).fit()
        self.fitted_formula = formula
        terms = [f"C({group})[T.{level}]" for level in levels[1:]]
        adjusted_for = ", ".join(self.spec.covariates)
        found.test = "Linear regression (OLS)" + (f" adjusted for {adjusted_for}" if adjusted_for else "")
        found.method = "statsmodels.formula.api.ols"
        found.formula = self._readable_formula(self.spec.compare)
        rows = self._coefficient_rows(
            fit,
            terms,
            [f"{level} − {levels[0]}" for level in levels[1:]],
            measure="mean difference",
            statistic_name="t",
            df=fit.df_resid,
            test="Linear regression",
        )
        statistic, p, df1, df2 = self._f_test(fit, terms)
        if len(terms) == 1:
            rows[0]["key"] = True
            found.statistic_name = "t"
        else:
            self._adjust_within(rows, "contrasts")
            rows.insert(
                0,
                _row(
                    self.spec.compare,
                    f"{self.spec.compare} (overall)",
                    statistic=statistic,
                    df=df1,
                    p=p,
                    measure="overall",
                    statistic_name="F",
                    df2=df2,
                    test="Linear regression",
                    key=True,
                ),
            )
            found.statistic_name = "F"
        found.estimates = _estimates(rows)
        effect = omega_squared(statistic, df1, df2, self.confidence)
        found.effect = {**effect, "name": "partial omega squared"} if self.spec.covariates else effect
        found.group_means = self._means(work[outcome], work[group], levels)
        self._sizes({level: int((work[group] == level).sum()) for level in levels})
        self._ols_diagnostics(fit, work, group)
        self.references += [REFERENCES["omega"], REFERENCES["steiger"]]
        found.summary = _clean_summary(str(fit.summary()))

    def _f_test(self, fit, terms: list[str]) -> tuple[float, float, float, float]:
        names = list(fit.params.index)
        matrix = np.zeros((len(terms), len(names)))
        for row, term in enumerate(terms):
            matrix[row, names.index(term)] = 1
        test = fit.f_test(matrix)
        return float(np.squeeze(test.fvalue)), float(test.pvalue), float(test.df_num), float(test.df_denom)

    def _coefficient_rows(
        self, fit, terms, labels, *, measure, statistic_name, df=math.nan, test="", exponentiate=False
    ) -> list[dict]:
        interval_frame = fit.conf_int(alpha=self.alpha)
        rows = []
        for term, label in zip(terms, labels, strict=True):
            estimate = float(fit.params[term])
            low, high = (float(value) for value in interval_frame.loc[term])
            if exponentiate:
                with np.errstate(over="ignore"):
                    estimate, low, high = (float(np.exp(value)) for value in (estimate, low, high))
            rows.append(
                _row(
                    term,
                    label,
                    estimate,
                    low,
                    high,
                    fit.tvalues[term],
                    df,
                    fit.pvalues[term],
                    measure=measure,
                    statistic_name=statistic_name,
                    test=test,
                )
            )
        return rows

    def _slope_rows(self, fit, statistic_name: str, df=math.nan, test: str = "") -> list[dict]:
        """The slope on the predictor, named by its column, and the intercept that the figures draw it from.

        The intercept row (measure "intercept") stays out of the results tables.
        """
        predictor = self.names[self.spec.relate]
        slope, intercept = self._coefficient_rows(
            fit,
            [predictor, "Intercept"],
            [f"Slope on {self.spec.relate}", "Intercept"],
            measure="slope",
            statistic_name=statistic_name,
            df=df,
            test=test,
        )
        slope["term"] = self.spec.relate
        intercept["measure"] = "intercept"
        return [slope, intercept]

    def _ols_diagnostics(self, fit, work: pd.DataFrame, group: str | None = None) -> None:
        from statsmodels.stats.diagnostic import het_breuschpagan  # noqa: PLC0415
        from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: PLC0415

        found = self.found
        found.fitted = pd.Series(np.asarray(fit.fittedvalues), index=work.index, name="fitted")
        found.residuals = pd.Series(np.asarray(fit.resid), index=work.index, name="residual")
        self._shapiro(found.residuals)
        if group is not None:
            samples = [
                found.residuals[work[group] == level].to_numpy() for level in work[group].cat.categories
            ]
            self._levene(samples, self.spec.compare, "Computed on the residuals.")
        exog = np.asarray(fit.model.exog, dtype=float)
        if exog.shape[1] > 1 and np.linalg.matrix_rank(exog) == exog.shape[1]:
            statistic, p_lm, _, _ = het_breuschpagan(fit.resid, exog)
            found.diagnostics.append(
                f"Breusch–Pagan test for constant residual variance: LM = {statistic:.2f}, {_p_text(p_lm)}."
            )
            self.references.append(REFERENCES["breusch_pagan"])
        if self.spec.covariates and exog.shape[1] > 2:
            names = list(fit.model.exog_names)
            values = []
            for position, name in enumerate(names):
                if name == "Intercept":
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    values.append((name, float(variance_inflation_factor(exog, position))))
            listed = ", ".join(f"{name} {_plain(value, 3)}" for name, value in values[:8])
            worst = max(value for _, value in values)
            more = f" (and {len(values) - 8} more)" if len(values) > 8 else ""
            caution = " Values above 5 suggest collinear predictors." if worst > 5 else ""
            found.diagnostics.append(f"Variance inflation factors: {listed}{more}.{caution}")
            self.references.append(REFERENCES["vif"])

    # Continuous outcomes in paired designs.

    def _paired(self, test: str) -> None:
        found, spec = self.found, self.spec
        levels = self._levels(spec.compare)
        if len(levels) != 2:
            raise Unsupported(
                f"A paired comparison needs exactly two conditions, but {spec.compare} has {len(levels)} "
                f"({', '.join(levels)}). For three or more, set `design.type: repeated` (a mixed model)."
            )
        work = self._work()
        group, outcome, unit = self.names[spec.compare], self.names[spec.outcome], self.names[self.unit]
        cells = work.groupby([unit, group], observed=True)[outcome].agg(["mean", "size"])
        repeated = cells[cells["size"] > 1]
        if len(repeated):
            subjects = repeated.index.get_level_values(0).nunique()
            found.diagnostics.append(
                f"Rows were averaged within each {self.unit} and condition before pairing ({subjects:,} "
                f"{self.unit} value{'s' if subjects != 1 else ''} had more than one row under a condition)."
            )
        wide = cells["mean"].unstack(group)
        complete = wide.dropna()
        unpaired = wide.index.difference(complete.index)
        if len(unpaired):
            dropped_rows = int(work[unit].isin(unpaired).sum())
            shown = ", ".join(map(str, sorted(unpaired)[:10])) + (" …" if len(unpaired) > 10 else "")
            where = "under only one of the two conditions"
            if len(unpaired) == 1:
                sentence = (
                    f"{self.unit} {shown} had {spec.outcome} {where} and was left out ({dropped_rows:,} "
                    f"row{'s' if dropped_rows != 1 else ''})."
                )
            else:
                sentence = (
                    f"{len(unpaired):,} {self.unit} values had {spec.outcome} {where} and were left out "
                    f"({dropped_rows:,} rows): {shown}."
                )
            found.diagnostics.append(sentence)
        if len(complete) < 2:
            raise Unsupported(
                f"Fewer than 2 {self.unit} values have {spec.outcome} under both {levels[0]} and "
                f"{levels[1]}. If {spec.compare} differs between subjects rather than within them, set "
                "`design.type: repeated` (a mixed model)."
            )
        used = work[work[unit].isin(complete.index)]
        found.n = len(used)
        found.subjects = len(complete)
        reference, other = complete[levels[0]].to_numpy(), complete[levels[1]].to_numpy()
        differences = other - reference
        self._sizes({level: int(len(complete)) for level in levels}, "Subjects per condition")
        found.groups = {level: int((used[group] == level).sum()) for level in levels}
        found.group_means = self._means(used[outcome], used[group], levels, units=used[unit])
        subject_means = complete.mean(axis=1)
        condition_effect = complete.mean(axis=0) - complete.to_numpy().mean()
        fitted = used[unit].map(subject_means).astype(float) + used[group].map(condition_effect).astype(float)
        found.fitted = fitted.rename("fitted")
        found.residuals = (used[outcome] - fitted).rename("residual")
        label = f"{levels[1]} − {levels[0]}"
        term = f"{spec.compare}[{levels[1]} - {levels[0]}]"
        if test == "paired-t":
            result = stats.ttest_rel(other, reference)
            ci = result.confidence_interval(confidence_level=self.confidence)
            found.test = "Paired t-test"
            found.method = "scipy.stats.ttest_rel"
            found.statistic_name = "t"
            row = _row(
                term,
                label,
                differences.mean(),
                ci.low,
                ci.high,
                result.statistic,
                result.df,
                result.pvalue,
                measure="mean difference",
                statistic_name="t",
                test="Paired t-test",
                key=True,
            )
            found.effect = cohens_dz(differences, self.confidence)
            self.references += [REFERENCES["student"], REFERENCES["cohen"], REFERENCES["noncentral"]]
        else:
            result = stats.wilcoxon(differences)
            estimate, low, high = hodges_lehmann_location(differences, self.confidence)
            found.test = "Wilcoxon signed-rank test"
            found.method = "scipy.stats.wilcoxon; Hodges–Lehmann estimate with a test-inversion interval"
            found.statistic_name = "W"
            row = _row(
                term,
                label,
                estimate,
                low,
                high,
                result.statistic,
                p=result.pvalue,
                measure="median difference",
                statistic_name="W",
                test="Wilcoxon signed-rank test",
                key=True,
            )
            self._interval_note(low, high)
            self.references += [REFERENCES["wilcoxon"], REFERENCES["hodges_lehmann"], REFERENCES["bauer"]]
            found.fitted = found.residuals = None
        found.estimates = _estimates([row])
        self._shapiro(differences, f"paired differences ({label})")
        found.summary = self._test_summary(levels, [reference, other], note=f"{len(complete):,} pairs.")

    # Linear mixed models.

    def _fit_mixed(self, formula: str, work: pd.DataFrame):
        import statsmodels.formula.api as smf  # noqa: PLC0415

        found = self.found
        unit = self.names[self.unit]
        self._nested_check()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = smf.mixedlm(formula, data=work, groups=unit).fit(reml=True)
            except (np.linalg.LinAlgError, ValueError) as error:
                raise Unsupported(
                    f"The linear mixed model could not be fitted ({error}). Check that each {self.unit_word} "
                    "has enough rows, or write the model under `custom:`."
                ) from error
        self.fitted_formula = f"{formula}, groups = {unit}"
        for message in _record_warnings(caught):
            found.warnings.append(f"statsmodels: {message}")
        if not fit.converged:
            found.warnings.append("The mixed model did not converge; treat its estimates with caution.")
        units = work[unit].nunique()
        per_unit = work.groupby(unit).size()
        if self.unit_word == "cluster":
            found.clusters = int(units)
        else:
            found.subjects = int(units)
        found.diagnostics.append(
            f"{units:,} {self.unit_word}s ({self.unit}), {int(per_unit.min()):,} to {int(per_unit.max()):,} "
            f"rows each; the model {'converged' if fit.converged else 'did not converge'} (REML)."
        )
        variance = float(np.asarray(fit.cov_re)[0, 0])
        residual = float(fit.scale)
        icc = variance / (variance + residual) if variance + residual > 0 else math.nan
        found.diagnostics.append(
            f"Variance between {self.unit_word}s {_plain(variance)}, residual variance {_plain(residual)}; "
            f"intraclass correlation (ICC) {_plain(icc, 2)}."
        )
        self.icc = icc
        if variance <= 1e-8 * max(residual, 1e-300):
            found.warnings.append(
                f"The variance between {self.unit_word}s was estimated as zero, at the edge of what the "
                f"model allows; the {self.unit_word}s look no more alike than unrelated rows."
            )
        if units < FEW_UNITS:
            found.warnings.append(
                f"Only {units} {self.unit_word}s: with fewer than {FEW_UNITS}, the variance between them is "
                "poorly estimated and Wald z tests can give p-values that are too small."
            )
        found.fitted = pd.Series(np.asarray(fit.fittedvalues), index=work.index, name="fitted")
        found.residuals = pd.Series(np.asarray(fit.resid), index=work.index, name="residual")
        self._shapiro(found.residuals, "conditional residuals")
        found.test = f"Linear mixed model, random intercept per {self.unit_word} ({self.unit})"
        found.method = "statsmodels.formula.api.mixedlm(...).fit(reml=True)"
        found.statistic_name = "z"
        found.summary = _clean_summary(str(fit.summary()))
        self.references += [REFERENCES["laird_ware"], REFERENCES["reml"]]
        if self.unit_word == "cluster" or self.study.design == "repeated":
            self.references.append(REFERENCES["icc"])
        return fit

    def _nested_check(self) -> None:
        study = self.study
        if study.design != "clustered" or not study.subject or study.subject == study.cluster:
            return
        if study.subject in self.rows and self.rows.groupby(study.subject).size().max() > 1:
            raise Unsupported(
                f"Rows are grouped by {study.subject} within {study.cluster}; random effects at two levels "
                "are not fitted. Write the model under `custom:`."
            )

    def _wald(self, fit, terms: list[str]) -> tuple[float, float, int]:
        names = list(fit.params.index)
        matrix = np.zeros((len(terms), len(names)))
        for row, term in enumerate(terms):
            matrix[row, names.index(term)] = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test = fit.wald_test(matrix, use_f=False, scalar=True)
        return float(np.squeeze(test.statistic)), float(test.pvalue), len(terms)

    def _key_terms(self, fit, rows: list[dict], overall_label: str, overall_term: str, what: str):
        """Mark the single term as key, or add an overall Wald test over several and adjust them within."""
        if len(rows) == 1:
            rows[0]["key"] = True
            return rows
        statistic, p, df = self._wald(fit, [row["term"] for row in rows])
        self._adjust_within(rows, what)
        overall = _row(
            overall_term,
            overall_label,
            statistic=statistic,
            df=df,
            p=p,
            measure="overall",
            statistic_name="χ²",
            test="Linear mixed model",
            key=True,
        )
        self.found.statistic_name = "χ²"
        return [overall, *rows]

    def _mixed_compare(self) -> None:
        found, spec = self.found, self.spec
        levels = self._levels(spec.compare)
        work = self._work()
        group, outcome = self.names[spec.compare], self.names[spec.outcome]
        right = " + ".join([f"C({group})", *self._covariate_terms()])
        fit = self._fit_mixed(f"{outcome} ~ {right}", work)
        found.formula = self._readable_formula(spec.compare, random=True)
        terms = [f"C({group})[T.{level}]" for level in levels[1:]]
        rows = self._coefficient_rows(
            fit,
            terms,
            [f"{level} − {levels[0]}" for level in levels[1:]],
            measure="mean difference",
            statistic_name="z",
            test="Linear mixed model",
        )
        found.estimates = _estimates(
            self._key_terms(fit, rows, f"{spec.compare} (overall)", spec.compare, "contrasts")
        )
        self._sizes({level: int((work[group] == level).sum()) for level in levels})
        found.group_means = self._means(work[outcome], work[group], levels, units=work[self.names[self.unit]])
        found.diagnostics.append(
            f"Group means average each {self.unit_word}'s rows first, so their intervals reflect the number "
            f"of {self.unit_word}s."
        )
        if self.study.time and self.study.time not in (spec.compare, *spec.covariates):
            found.diagnostics.append(
                f"The design's time variable ({self.study.time}) is not in this model, so the difference is "
                f"averaged over time points; a `change: {self.study.time}` analysis models time."
            )
        if self.study.design == "paired":
            found.diagnostics.append(
                "In this paired design the comparison is a mixed model with a random intercept per "
                f"{self.unit}, which allows covariates and subjects measured under only one condition."
            )

    # Continuous outcomes related to a numeric predictor.

    def _fit_relate(self, test: str) -> None:
        if self.study.design == "independent":
            self._relate_independent(test)
        else:
            self._relate_mixed()

    def _relate_independent(self, test: str) -> None:
        import statsmodels.formula.api as smf  # noqa: PLC0415

        found, spec = self.found, self.spec
        work = self._work()
        outcome, predictor = self.names[spec.outcome], self.names[spec.relate]
        if work[predictor].nunique() < 2:
            raise Unsupported(f"{spec.relate} has the same value in every row this analysis uses.")
        right = " + ".join([predictor, *self._covariate_terms()])
        formula = f"{outcome} ~ {right}"
        fit = smf.ols(formula, data=work).fit()
        self.fitted_formula = formula
        found.formula = self._readable_formula(spec.relate)
        slope, intercept = self._slope_rows(fit, "t", fit.df_resid, "Linear regression")
        rows = [slope]
        if not spec.covariates:
            x, y = work[predictor].to_numpy(), work[outcome].to_numpy()
            pearson = stats.pearsonr(x, y)
            ci = pearson.confidence_interval(confidence_level=self.confidence)
            n = len(x)
            r_row = _row(
                "pearson",
                "Pearson's r",
                pearson.statistic,
                ci.low,
                ci.high,
                pearson.statistic * math.sqrt((n - 2) / max(1 - pearson.statistic**2, 1e-300)),
                n - 2,
                pearson.pvalue,
                measure="correlation",
                statistic_name="t",
                test="Pearson correlation",
            )
            spearman = stats.spearmanr(x, y)
            rho_row = _row(
                "spearman",
                "Spearman's ρ (sensitivity check)",
                spearman.statistic,
                p=spearman.pvalue,
                measure="correlation",
                statistic_name="",
                test="Spearman correlation",
            )
            rows = [slope, r_row, rho_row]
            found.effect = {
                "name": "Pearson's r",
                "value": float(pearson.statistic),
                "ci_low": float(ci.low),
                "ci_high": float(ci.high),
            }
            self.references += [REFERENCES["pearson"], REFERENCES["fisher_z"], REFERENCES["spearman"]]
        key = {"pearson": 1, "spearman": 2}.get(test, 0)
        rows[key]["key"] = True
        found.statistic_name = rows[key]["statistic_name"]
        found.estimates = _estimates([*rows, intercept])
        adjusted = ", ".join(spec.covariates)
        found.test = {
            "pearson": "Pearson correlation",
            "spearman": "Spearman correlation",
        }.get(test, "Linear regression (OLS)" + (f" adjusted for {adjusted}" if adjusted else ""))
        found.method = "statsmodels.formula.api.ols" + (
            "; scipy.stats.pearsonr (Fisher z interval); scipy.stats.spearmanr" if not spec.covariates else ""
        )
        found.diagnostics.append(f"{len(work):,} rows with both {spec.outcome} and {spec.relate}.")
        self._ols_diagnostics(fit, work)
        found.summary = _clean_summary(str(fit.summary()))

    def _relate_mixed(self) -> None:
        found, spec = self.found, self.spec
        work = self._work()
        outcome, predictor = self.names[spec.outcome], self.names[spec.relate]
        if work[predictor].nunique() < 2:
            raise Unsupported(f"{spec.relate} has the same value in every row this analysis uses.")
        right = " + ".join([predictor, *self._covariate_terms()])
        fit = self._fit_mixed(f"{outcome} ~ {right}", work)
        found.formula = self._readable_formula(spec.relate, random=True)
        rows = self._slope_rows(fit, "z", test="Linear mixed model")
        rows[0]["key"] = True
        found.estimates = _estimates(rows)
        if self.study.design == "paired":
            found.diagnostics.append(
                f"Rows come in pairs from the same {self.unit}, so the slope comes from a mixed model with a "
                f"random intercept per {self.unit}."
            )

    # Change over time.

    def _fit_change(self) -> None:
        found, spec, study = self.found, self.spec, self.study
        time = spec.change
        variable = study.variable(time)
        points = (
            sorted(self.rows[time].unique()) if variable.type == "numeric" else levels_of(self.rows, time)
        )
        if len(points) < 2:
            raise Unsupported(
                f"{time} has fewer than 2 distinct values in the rows used; there is no change to model."
            )
        if spec.time_as == "linear" and variable.type != "numeric":
            raise Unsupported(
                f"`time_as: linear` needs a numeric time variable, but {time} is {variable.type}."
            )
        linear = spec.time_as == "linear" or (
            spec.time_as is None and variable.type == "numeric" and len(points) > CATEGORICAL_TIME_POINTS
        )
        if spec.time_as is None:
            found.diagnostics.append(
                f"{time} has {len(points)} distinct values, so it is modelled as "
                + (
                    f"a straight line (more than {CATEGORICAL_TIME_POINTS} points)."
                    if linear
                    else f"categorical, one level per time point ({CATEGORICAL_TIME_POINTS} or fewer)."
                )
            )
        levels = self._levels(spec.by) if spec.by else []
        work = self._work(categorical=() if linear else (time,))
        outcome, tname = self.names[spec.outcome], self.names[time]
        by = self.names[spec.by] if spec.by else None
        time_term = tname if linear else f"C({tname})"
        right = (f"C({by}) * {time_term}" if by else time_term) + "".join(
            f" + {term}" for term in self._covariate_terms()
        )
        fit = self._fit_mixed(f"{outcome} ~ {right}", work)
        found.formula = self._readable_formula(f"{spec.by} * {time}" if by else time, random=True)
        found.test = (
            f"Linear mixed model over {time} ({'linear' if linear else 'categorical'}), random intercept per "
            f"subject ({self.unit})"
        )
        labels = [_time_label(point) for point in points] if variable.type == "numeric" else list(points)
        if linear:
            if by:
                terms = [f"C({by})[T.{level}]:{tname}" for level in levels[1:]]
                rows = self._coefficient_rows(
                    fit,
                    terms,
                    [f"Slope difference, {level} − {levels[0]}" for level in levels[1:]],
                    measure="slope difference",
                    statistic_name="z",
                    test="Linear mixed model",
                )
                rows = self._key_terms(
                    fit,
                    rows,
                    f"{spec.by} × {time} (overall)",
                    f"{spec.by}:{time}",
                    "slope differences",
                )
                rows += self._group_slopes(fit, by, tname, levels)
            else:
                rows = self._coefficient_rows(
                    fit,
                    [tname],
                    [f"Slope over {time}"],
                    measure="slope",
                    statistic_name="z",
                    test="Linear mixed model",
                )
                rows[0]["key"] = True
            found.diagnostics.append(
                f"The model has a random intercept per {self.unit} but no random slope: it lets subjects "
                "differ in level, not in rate of change. If rates differ between subjects, the slope's "
                "interval is too narrow; write a random-slope model under `custom:`."
            )
        else:
            first = labels[0]
            if by:
                terms = [
                    f"C({by})[T.{level}]:C({tname})[T.{point}]"
                    for level in levels[1:]
                    for point in labels[1:]
                ]
                rows = self._coefficient_rows(
                    fit,
                    terms,
                    [
                        f"{level} − {levels[0]}, change {time} {first} to {point}"
                        for level in levels[1:]
                        for point in labels[1:]
                    ],
                    measure="difference in change",
                    statistic_name="z",
                    test="Linear mixed model",
                )
                rows = self._key_terms(
                    fit,
                    rows,
                    f"{spec.by} × {time} (overall)",
                    f"{spec.by}:{time}",
                    "interaction terms",
                )
            else:
                terms = [f"C({tname})[T.{point}]" for point in labels[1:]]
                rows = self._coefficient_rows(
                    fit,
                    terms,
                    [f"{time} {point} − {first}" for point in labels[1:]],
                    measure="change",
                    statistic_name="z",
                    test="Linear mixed model",
                )
                rows = self._key_terms(fit, rows, f"{time} (overall)", time, "time points")
        found.estimates = _estimates(rows)
        groups = work[by] if by else None
        times = self.rows[time]
        found.group_means = self._means(
            work[outcome],
            groups,
            levels,
            times=times,
            time_order=list(points),
            units=work[self.names[self.unit]],
        )
        if by:
            self._sizes({level: int((work[by] == level).sum()) for level in levels})
        counts = times.value_counts()
        found.diagnostics.append(
            f"Rows per {time}: "
            + ", ".join(f"{_time_label(point)} (n = {int(counts[point]):,})" for point in points)
            + "."
        )

    def _group_slopes(self, fit, by: str, tname: str, levels: list[str]) -> list[dict]:
        """Each group's own slope over time, as linear combinations of the fixed effects."""
        names = list(fit.fe_params.index)
        rows = []
        for position, level in enumerate(levels):
            vector = np.zeros(len(names))
            vector[names.index(tname)] = 1
            if position:
                vector[names.index(f"C({by})[T.{level}]:{tname}")] = 1
            test = fit.t_test(vector[None, :])
            low, high = np.asarray(test.conf_int(alpha=self.alpha))[0]
            rows.append(
                _row(
                    f"{tname}|{level}",
                    f"Slope over {self.spec.change} in {level}",
                    float(np.squeeze(test.effect)),
                    low,
                    high,
                    float(np.squeeze(test.tvalue)),
                    p=float(np.squeeze(test.pvalue)),
                    measure="slope",
                    statistic_name="z",
                    test="Linear mixed model",
                )
            )
        return rows

    # Binary outcomes.

    def _fit_binary(self, test: str) -> None:
        self.found.diagnostics.append(
            f"{self.spec.outcome}: {self.event} is counted as the event and {self.non_event} as its absence."
        )
        if self.spec.kind == "compare" and test == "rates":
            self._rates()
        else:
            self._logistic()

    def _rates(self) -> None:
        from statsmodels.stats.contingency_tables import Table2x2  # noqa: PLC0415
        from statsmodels.stats.proportion import confint_proportions_2indep  # noqa: PLC0415

        found, spec = self.found, self.spec
        levels = self._levels(spec.compare)
        work = self._work()
        group, outcome = self.names[spec.compare], self.names[spec.outcome]
        events = [int(work.loc[work[group] == level, outcome].sum()) for level in levels]
        totals = [int((work[group] == level).sum()) for level in levels]
        self._sizes(dict(zip(levels, totals, strict=True)))
        found.group_means = self._means(work[outcome], work[group], levels, binary=True)
        table = np.array([[event, total - event] for event, total in zip(events, totals, strict=True)])
        statistic, df, p, used = self._table_test(table)
        name = f"Comparison of proportions ({used})"
        found.test = name
        found.method = (
            "statsmodels.stats.proportion.confint_proportions_2indep(method='newcomb'); statsmodels "
            "Table2x2.oddsratio_confint; "
            + (
                "scipy.stats.chi2_contingency(correction=False)"
                if used.startswith("χ²")
                else "scipy.stats.fisher_exact"
            )
        )
        found.statistic_name = "χ²" if used.startswith("χ²") else ""
        rows = []
        odds_rows = []
        for position in range(1, len(levels)):
            e1, n1, e0, n0 = events[position], totals[position], events[0], totals[0]
            low, high = confint_proportions_2indep(
                e1, n1, e0, n0, method="newcomb", compare="diff", alpha=self.alpha
            )
            pair = table[[0, position]]
            pair_statistic, pair_df, pair_p, pair_test = self._table_test(pair)
            odds = Table2x2(np.array([[e1, n1 - e1], [e0, n0 - e0]]), shift_zeros=True)
            odds_low, odds_high = odds.oddsratio_confint(alpha=self.alpha)
            rows.append(
                _row(
                    f"{spec.compare}[{levels[position]} - {levels[0]}]",
                    f"Risk difference, {levels[position]} − {levels[0]}",
                    e1 / n1 - e0 / n0,
                    low,
                    high,
                    pair_statistic,
                    pair_df,
                    pair_p,
                    measure="risk difference",
                    statistic_name="χ²" if pair_test.startswith("χ²") else "",
                    test=pair_test,
                )
            )
            odds_rows.append(
                {
                    "name": "odds ratio",
                    "value": float(odds.oddsratio),
                    "ci_low": float(odds_low),
                    "ci_high": float(odds_high),
                    "level": levels[position],
                }
            )
            if 0 in (e1, n1 - e1, e0, n0 - e0):
                found.diagnostics.append(
                    f"A cell of the {levels[position]} vs {levels[0]} table is zero, so 0.5 was added to "
                    "every cell for its odds ratio."
                )
        if len(levels) == 2:
            rows[0]["key"] = True
            found.effect = {key: value for key, value in odds_rows[0].items() if key != "level"}
        else:
            self._adjust_within(rows, "contrasts with the reference")
            rows.insert(
                0,
                _row(
                    spec.compare,
                    f"{spec.compare} (overall)",
                    statistic=statistic,
                    df=df,
                    p=p,
                    measure="overall",
                    statistic_name="χ²" if used.startswith("χ²") else "",
                    test=used,
                    key=True,
                ),
            )
            for odds in odds_rows:
                rows.append(
                    _row(
                        f"{spec.compare}[{odds['level']} / {levels[0]}]",
                        f"Odds ratio, {odds['level']} vs {levels[0]}",
                        odds["value"],
                        odds["ci_low"],
                        odds["ci_high"],
                        measure="odds ratio",
                        test="Woolf interval",
                    )
                )
            from scipy.stats.contingency import association  # noqa: PLC0415

            found.effect = {
                "name": "Cramér's V",
                "value": float(association(table, method="cramer")),
                "ci_low": math.nan,
                "ci_high": math.nan,
            }
            self.references.append(REFERENCES["cramer"])
        found.estimates = _estimates(rows)
        found.diagnostics.append(
            "Events per group: "
            + ", ".join(
                f"{level} {event:,} of {total:,}"
                for level, event, total in zip(levels, events, totals, strict=True)
            )
            + "."
        )
        self.references += [REFERENCES["newcombe"], REFERENCES["woolf"], REFERENCES["wilson"]]
        found.summary = self._rates_summary(levels, events, totals)

    def _table_test(self, table: np.ndarray) -> tuple[float, float, float, str]:
        """χ² test, or Fisher's exact test when an expected count is below 5."""
        expected = stats.contingency.expected_freq(table)
        if (expected < SMALL_EXPECTED).any():
            if table.shape == (2, 2):
                result = stats.fisher_exact(table)
                self.references.append(REFERENCES["fisher_exact"])
                self._small_expected_note(expected, "Fisher's exact test")
                return math.nan, math.nan, float(result.pvalue), "Fisher's exact test"
            method = stats.MonteCarloMethod(
                n_resamples=MONTE_CARLO_TABLES, rng=np.random.default_rng(MONTE_CARLO_SEED)
            )
            result = stats.fisher_exact(table, method=method)
            self.references += [REFERENCES["fisher_exact"], REFERENCES["freeman_halton"]]
            self._small_expected_note(
                expected,
                f"Fisher's exact test (Freeman–Halton, Monte Carlo with {MONTE_CARLO_TABLES:,} tables, seed "
                f"{MONTE_CARLO_SEED})",
            )
            return math.nan, math.nan, float(result.pvalue), "Fisher's exact test (Monte Carlo)"
        statistic, p, df, _ = stats.chi2_contingency(table, correction=False)
        self.references.append(REFERENCES["chi2"])
        return float(statistic), float(df), float(p), "χ² test"

    def _small_expected_note(self, expected: np.ndarray, test: str) -> None:
        sentence = (
            f"The smallest expected count was {_plain(float(expected.min()), 3)} (below {SMALL_EXPECTED}), "
            f"so the p-value is from {test}."
        )
        if sentence not in self.found.diagnostics:
            self.found.diagnostics.append(sentence)

    def _logistic(self) -> None:
        import statsmodels.formula.api as smf  # noqa: PLC0415
        from statsmodels.tools.sm_exceptions import PerfectSeparationError  # noqa: PLC0415

        found, spec = self.found, self.spec
        work = self._work()
        outcome, predictor = self.names[spec.outcome], self.names[spec.predictor]
        if spec.kind == "compare":
            levels = self._levels(spec.compare)
            term_names = [f"C({predictor})[T.{level}]" for level in levels[1:]]
            labels = [f"Odds ratio, {level} vs {levels[0]}" for level in levels[1:]]
            first = f"C({predictor})"
            self._sizes({level: int((work[predictor] == level).sum()) for level in levels})
            found.group_means = self._means(work[outcome], work[predictor], levels, binary=True)
        else:
            if work[predictor].nunique() < 2:
                raise Unsupported(f"{spec.relate} has the same value in every row this analysis uses.")
            term_names = [predictor]
            unit = self.study.variable(spec.relate).unit
            labels = [f"Odds ratio per {unit or 'unit'} of {spec.relate}"]
            first = predictor
        formula = f"{outcome} ~ " + " + ".join([first, *self._covariate_terms()])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = smf.logit(formula, data=work).fit(disp=0)
            except (PerfectSeparationError, np.linalg.LinAlgError) as error:
                raise Unsupported(
                    f"The logistic regression could not be fitted ({error}). Write a penalised (Firth) model "
                    "under `custom:`."
                ) from error
        messages = _record_warnings(caught)
        separated = any("separation" in message.lower() for message in messages)
        if separated or not np.isfinite(fit.bse).all() or not fit.mle_retvals.get("converged", True):
            raise Unsupported(
                f"The logistic regression did not converge to finite estimates: {spec.predictor} or a "
                "covariate separates the outcome perfectly or nearly, so the odds ratio is not estimable by "
                "maximum likelihood. Write a penalised (Firth) model under `custom:`."
            )
        for message in messages:
            found.warnings.append(f"statsmodels: {message}")
        self.fitted_formula = formula
        found.formula = self._readable_formula(spec.predictor)
        found.test = "Logistic regression" + (
            f" adjusted for {', '.join(spec.covariates)}" if spec.covariates else ""
        )
        found.method = "statsmodels.formula.api.logit"
        found.statistic_name = "z"
        rows = self._coefficient_rows(
            fit,
            term_names,
            labels,
            measure="odds ratio",
            statistic_name="z",
            test="Logistic regression",
            exponentiate=True,
        )
        if len(rows) == 1:
            rows[0]["key"] = True
        else:
            statistic, p, df = self._wald(fit, term_names)
            self._adjust_within(rows, "contrasts")
            rows.insert(
                0,
                _row(
                    spec.predictor,
                    f"{spec.predictor} (overall)",
                    statistic=statistic,
                    df=df,
                    p=p,
                    measure="overall",
                    statistic_name="χ²",
                    test="Logistic regression",
                    key=True,
                ),
            )
            found.statistic_name = "χ²"
        found.estimates = _estimates(rows)
        events = int(work[outcome].sum())
        fewest = min(events, len(work) - events)
        parameters = len(fit.params) - 1
        found.diagnostics.append(
            f"{events:,} events in {len(work):,} rows; {parameters} "
            f"parameter{'s' if parameters != 1 else ''} besides the intercept."
        )
        if parameters and fewest / parameters < EVENTS_PER_PARAMETER:
            found.warnings.append(
                f"Fewer than {EVENTS_PER_PARAMETER} of the rarer outcome per parameter ({fewest} for "
                f"{parameters}); the odds ratios may be biased away from 1 and their intervals unreliable."
            )
        self.references.append(REFERENCES["logistic"])
        found.summary = _clean_summary(str(fit.summary()))

    # Text for the supplement.

    def _percent(self) -> str:
        return f"{self.confidence * 100:g}%"

    def _interval_note(self, low: float, high: float) -> None:
        if not (math.isfinite(low) and math.isfinite(high)):
            self.found.diagnostics.append(
                f"There are too few rows for a finite {self._percent()} interval from the rank test."
            )

    def _test_summary(self, levels: list[str], samples: list[np.ndarray], note: str = "") -> str:
        lines = [self.found.test, ""]
        for level, sample in zip(levels, samples, strict=True):
            sd = float(np.std(sample, ddof=1)) if len(sample) > 1 else math.nan
            lines.append(
                f"  {level}: n = {len(sample)}, mean = {float(np.mean(sample)):.6g}, SD = {sd:.6g}, median = "
                f"{float(np.median(sample)):.6g}"
            )
        if note:
            lines += ["", note]
        return "\n".join(lines)

    def _rates_summary(self, levels, events, totals) -> str:
        lines = [self.found.test, ""]
        for level, event, total in zip(levels, events, totals, strict=True):
            lines.append(f"  {level}: {event} of {total} ({event / total:.4f})")
        return "\n".join(lines)


def _event_level(variable, levels: list[str]) -> str:
    """The level counted as the event: the second declared level, or the one whose name says so."""
    declared = [str(level) for level in variable.levels]
    if declared:
        ordered = [level for level in declared if level in levels] + [
            level for level in levels if level not in declared
        ]
        return ordered[1]
    words = ("1", "true", "yes", "y", "present", "positive", "event", "case", "died", "dead")
    marked = [level for level in levels if level.strip().lower() in words]
    if len(marked) == 1:
        return marked[0]
    return levels[1]


def _estimates_text(estimates: pd.DataFrame, percent: str) -> str:
    frame = estimates[
        ["label", "estimate", "ci_low", "ci_high", "statistic_name", "statistic", "df", "p", "p_adjusted"]
    ]
    frame = frame.rename(
        columns={"ci_low": f"{percent} CI low", "ci_high": f"{percent} CI high", "statistic_name": "stat"}
    )
    return frame.to_string(index=False, float_format=lambda value: f"{value:.6g}", na_rep="")


# The statistics behind the tests, each checked against an independent computation in the tests.


def hedges_g(reference: np.ndarray, other: np.ndarray, confidence: float) -> dict:
    """Hedges' g for other − reference, with a noncentral-t interval (exact under normality).

    g is Cohen's d on the pooled SD, times the exact small-sample correction
    J = Γ(m/2) / (√(m/2) Γ((m − 1)/2)), m = n₁ + n₂ − 2 (Hedges 1981). The interval inverts the
    noncentral t distribution of the pooled t statistic (Steiger and Fouladi 1997) and scales it the
    same way.
    """
    n1, n2 = len(reference), len(other)
    freedom = n1 + n2 - 2
    pooled = math.sqrt(((n1 - 1) * np.var(reference, ddof=1) + (n2 - 1) * np.var(other, ddof=1)) / freedom)
    if pooled == 0:
        return {}
    d = (float(np.mean(other)) - float(np.mean(reference))) / pooled
    correction = hedges_correction(freedom)
    scale = math.sqrt(1 / n1 + 1 / n2)
    low, high = noncentral_t_interval(d / scale, freedom, confidence)
    return {
        "name": "Hedges' g",
        "value": correction * d,
        "ci_low": correction * low * scale,
        "ci_high": correction * high * scale,
    }


def hedges_correction(freedom: float) -> float:
    return math.exp(
        special.gammaln(freedom / 2) - math.log(math.sqrt(freedom / 2)) - special.gammaln((freedom - 1) / 2)
    )


def cohens_dz(differences: np.ndarray, confidence: float) -> dict:
    """Cohen's dz, the mean paired difference over its SD, with a noncentral-t interval."""
    n = len(differences)
    sd = float(np.std(differences, ddof=1))
    if sd == 0 or n < 2:
        return {}
    dz = float(np.mean(differences)) / sd
    low, high = noncentral_t_interval(dz * math.sqrt(n), n - 1, confidence)
    return {"name": "Cohen's dz", "value": dz, "ci_low": low / math.sqrt(n), "ci_high": high / math.sqrt(n)}


def noncentral_t_interval(t: float, freedom: float, confidence: float) -> tuple[float, float]:
    """The noncentrality parameters whose t distributions put ``t`` at the interval's two tail points.

    SciPy's noncentral t returns NaN far in its tails; the cdf falls from 1 to 0 as the noncentrality
    grows, so a NaN is read as the limit on its side. An end that cannot be found is NaN, never an error.
    """
    alpha = 1 - confidence

    def cdf(ncp: float) -> float:
        value = float(stats.nct.cdf(t, freedom, ncp))
        return (0.0 if ncp > t else 1.0) if math.isnan(value) else value

    def solve(target: float) -> float:
        def gap(ncp: float) -> float:
            return cdf(ncp) - target

        width = 10.0 + abs(t)
        low, high = t - width, t + width
        for _ in range(60):
            if gap(low) > 0 > gap(high):
                try:
                    return float(optimize.brentq(gap, low, high, xtol=1e-12))
                except (ValueError, RuntimeError):
                    return math.nan
            low, high = low - width, high + width
            width *= 2
        return math.nan

    return solve(1 - alpha / 2), solve(alpha / 2)


def omega_squared(f_statistic: float, df1: float, df2: float, confidence: float) -> dict:
    """ω² from an F statistic and its degrees of freedom, with a noncentral-F interval.

    ω² = df₁(F − 1) / (df₁F + df₂ + 1), which for a one-way ANOVA equals Hays' estimate from the sums of
    squares and, for a term in a larger model, gives partial ω² (Olejnik and Algina 2003). The interval
    inverts the noncentral F distribution for the noncentrality λ (statsmodels
    ``confint_noncentrality``; Steiger 2004) and maps each end through the same formula with F = λ/df₁,
    so the point estimate and its interval use one definition. Negative values are reported as 0.
    """
    from statsmodels.stats.oneway import confint_noncentrality  # noqa: PLC0415

    def omega(f: float) -> float:
        return max(0.0, df1 * (f - 1) / (df1 * f + df2 + 1))

    if not math.isfinite(f_statistic):
        return {}
    noncentrality = np.asarray(
        confint_noncentrality(f_statistic, (df1, df2), alpha=1 - confidence), dtype=float
    )
    low, high = (float(omega(max(float(value), 0.0) / df1)) for value in noncentrality)
    return {"name": "omega squared", "value": omega(f_statistic), "ci_low": low, "ci_high": high}


def games_howell(levels: list[str], samples: list[np.ndarray], confidence: float) -> list[dict]:
    """Every pair of groups compared without assuming equal variances (Games and Howell 1976).

    For each pair, the difference in means over √(s²ᵢ/nᵢ + s²ⱼ/nⱼ) is a Welch t with
    Welch–Satterthwaite degrees of freedom; √2·|t| is referred to the studentized range for k groups
    (``scipy.stats.studentized_range``), which gives p-values and intervals simultaneous over the pairs.
    """
    k = len(samples)
    critical = {}
    rows = []
    for first in range(k):
        for second in range(first + 1, k):
            a, b = samples[first], samples[second]
            va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
            se = math.sqrt(va + vb)
            freedom = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
            difference = float(np.mean(b) - np.mean(a))
            t = difference / se
            if freedom not in critical:
                critical[freedom] = float(stats.studentized_range.ppf(confidence, k, freedom))
            margin = critical[freedom] / math.sqrt(2) * se
            rows.append(
                {
                    "level": levels[second],
                    "reference": levels[first],
                    "difference": difference,
                    "ci_low": difference - margin,
                    "ci_high": difference + margin,
                    "t": t,
                    "df": freedom,
                    "p": float(2 * stats.t.sf(abs(t), freedom)),
                    "p_adjusted": float(
                        min(1.0, stats.studentized_range.sf(abs(t) * math.sqrt(2), k, freedom))
                    ),
                }
            )
    return rows


def tukey_kramer(levels: list[str], samples: list[np.ndarray], confidence: float) -> list[dict]:
    """Every pair of groups compared on the pooled variance: ``scipy.stats.tukey_hsd`` (Tukey–Kramer)."""
    result = stats.tukey_hsd(*samples)
    bounds = result.confidence_interval(confidence_level=confidence)
    total = sum(len(sample) for sample in samples)
    freedom = total - len(samples)
    pooled = sum((len(sample) - 1) * np.var(sample, ddof=1) for sample in samples) / freedom
    rows = []
    for first in range(len(samples)):
        for second in range(first + 1, len(samples)):
            difference = float(result.statistic[second, first])
            t = difference / math.sqrt(pooled * (1 / len(samples[first]) + 1 / len(samples[second])))
            rows.append(
                {
                    "level": levels[second],
                    "reference": levels[first],
                    "difference": difference,
                    "ci_low": float(bounds.low[second, first]),
                    "ci_high": float(bounds.high[second, first]),
                    "t": t,
                    "df": freedom,
                    "p": float(2 * stats.t.sf(abs(t), freedom)),
                    "p_adjusted": float(result.pvalue[second, first]),
                }
            )
    return rows


def hodges_lehmann_shift(
    reference: np.ndarray, other: np.ndarray, confidence: float
) -> tuple[float, float, float]:
    """The Hodges–Lehmann shift (median of other − reference over all pairs) with a test-inversion interval.

    The interval holds every shift δ that the Mann–Whitney test of other − δ against reference does
    not reject (Bauer 1972), using SciPy's own p-values, so it agrees with the reported test.
    """
    differences = np.subtract.outer(np.asarray(other, float), np.asarray(reference, float)).ravel()
    estimate = float(np.median(differences))

    def p_at(delta: float) -> float:
        return float(
            stats.mannwhitneyu(np.asarray(other, float) - delta, reference, alternative="two-sided").pvalue
        )

    low, high = _inverted_interval(np.unique(differences), estimate, p_at, 1 - confidence)
    return estimate, low, high


def hodges_lehmann_location(differences: np.ndarray, confidence: float) -> tuple[float, float, float]:
    """The Hodges–Lehmann estimate of the paired difference (median of the Walsh averages), with an interval
    from inverting the Wilcoxon signed-rank test."""
    values = np.asarray(differences, float)
    first, second = np.triu_indices(len(values))
    walsh = (values[first] + values[second]) / 2
    estimate = float(np.median(walsh))

    def p_at(delta: float) -> float:
        return float(stats.wilcoxon(values - delta).pvalue)

    low, high = _inverted_interval(np.unique(walsh), estimate, p_at, 1 - confidence)
    return estimate, low, high


def _inverted_interval(candidates: np.ndarray, estimate: float, p_at, alpha: float) -> tuple[float, float]:
    """The ends of {δ : p(δ) ≥ α} for a rank test whose p-value is unimodal in δ and steps at ``candidates``.

    p is constant between consecutive candidates, so it is evaluated at the midpoints and the interval's
    ends are candidates. Bisection keeps the number of tests small.
    """
    if len(candidates) < 2:
        return math.nan, math.nan
    gaps = (candidates[:-1] + candidates[1:]) / 2
    last = len(gaps) - 1
    peak = int(np.clip(np.searchsorted(candidates, estimate) - 1, 0, last))
    best = max(range(max(0, peak - 1), min(last, peak + 1) + 1), key=lambda index: p_at(gaps[index]))
    if p_at(gaps[best]) < alpha:
        return math.nan, math.nan
    lo, hi = 0, best
    while lo < hi:
        middle = (lo + hi) // 2
        if p_at(gaps[middle]) >= alpha:
            hi = middle
        else:
            lo = middle + 1
    lower_index = lo
    lo, hi = best, last
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if p_at(gaps[middle]) >= alpha:
            lo = middle
        else:
            hi = middle - 1
    upper_index = lo
    spacing = float(np.ptp(candidates)) + 1.0
    low = candidates[lower_index]
    if lower_index == 0 and p_at(candidates[0] - spacing) >= alpha:
        low = math.nan
    high = candidates[upper_index + 1]
    if upper_index == last and p_at(candidates[-1] + spacing) >= alpha:
        high = math.nan
    return float(low), float(high)


# The results tables and the model records.


def _decimals(study: Study, found: AnalysisResult, data: pd.DataFrame, row: pd.Series) -> int:
    measure = row["measure"]
    if measure in ("odds ratio", "correlation"):
        return 2
    if measure == "risk difference":
        return 3
    if measure in ("slope", "slope difference"):
        return (
            decimals_for(pd.Series([row["ci_low"], row["ci_high"]]).dropna())
            if pd.notna(row["ci_low"])
            else 3
        )
    outcome = found.spec.outcome
    fixed = study.variable(outcome).decimals
    return decimals_for(data[outcome], fixed) if outcome in data else 2 if fixed is None else fixed


def _unit_text(study: Study, found: AnalysisResult, measure: str) -> str:
    outcome_unit = study.variable(found.spec.outcome).unit
    if measure in ("mean difference", "median difference", "change", "difference in change"):
        return outcome_unit
    if measure in ("slope", "slope difference"):
        predictor_unit = study.variable(found.spec.predictor).unit
        per = f"per {SINGULAR_UNITS.get(predictor_unit, predictor_unit)}" if predictor_unit else "per unit"
        return f"{outcome_unit} {per}".strip()
    return ""


def _statistic_text(row: pd.Series) -> str:
    name, value = row["statistic_name"], row["statistic"]
    if not name or pd.isna(value):
        return "–"

    def freedom(df) -> str:
        return f"{df:.0f}" if float(df).is_integer() else f"{df:.1f}"

    if pd.notna(row["df"]) and pd.notna(row["df2"]):
        return f"{name}({freedom(row['df'])}, {freedom(row['df2'])}) = {number(value, 2)}"
    if pd.notna(row["df"]) and name != "z":
        return f"{name}({freedom(row['df'])}) = {number(value, 2)}"
    return f"{name} = {number(value, 2)}"


def _effect_text(effect: dict) -> str:
    if not effect or pd.isna(effect.get("value")):
        return "–"
    symbol = EFFECT_SYMBOLS.get(effect["name"], effect["name"])
    text = f"{symbol} = {number(effect['value'], 2)}"
    if pd.notna(effect.get("ci_low")) and pd.notna(effect.get("ci_high")):
        text += f" ({interval(effect['ci_low'], effect['ci_high'], 2)})"
    return text


def _n_text(found: AnalysisResult) -> str:
    if found.design == "paired" and found.test in ("Paired t-test", "Wilcoxon signed-rank test"):
        return f"{found.n:,} ({found.subjects:,} pairs)"
    if found.clusters:
        return f"{found.n:,} ({found.clusters:,} clusters)"
    if found.subjects and found.design != "independent":
        return f"{found.n:,} ({found.subjects:,} subjects)"
    return f"{found.n:,}"


#: Columns of the results tables' CSV: one row per estimate, at full precision.
RESULT_COLUMNS = [
    "analysis",
    "question",
    "planned",
    "family",
    "outcome",
    "predictor",
    "design",
    "test",
    "method",
    "n",
    "n_missing",
    "subjects",
    "clusters",
    "term",
    "label",
    "row_test",
    "measure",
    "unit",
    "estimate",
    "ci_low",
    "ci_high",
    "confidence",
    "statistic_name",
    "statistic",
    "df",
    "df2",
    "p",
    "p_adjusted",
    "adjustment",
    "key",
    "effect_name",
    "effect",
    "effect_ci_low",
    "effect_ci_high",
]


def results_table(study: Study, data: pd.DataFrame, chosen: list[AnalysisResult], planned: bool) -> Table:
    """One table of the planned analyses, or of the exploratory ones.

    An analysis that could not be fitted is named in the notes with the reason, so a planned
    analysis never drops out of the paper unseen, even when it leaves the table without rows.
    """
    fitted = [found for found in chosen if found.supported]
    percent = f"{study.decisions.confidence * 100:g}%"
    display_rows, data_rows = [], []
    symbols, statistics, contrast_methods, measures = set(), set(), set(), set()
    for found in fitted:
        first = True
        for _, row in found.estimates[found.estimates["measure"] != "intercept"].iterrows():
            decimals = _decimals(study, found, data, row)
            unit = _unit_text(study, found, row["measure"])
            estimate = number(row["estimate"], decimals) if pd.notna(row["estimate"]) else "–"
            if estimate != "–" and unit:
                estimate = f"{estimate} {unit}"
            effect = found.effect if row["key"] else {}
            if effect:
                symbols.add(effect["name"])
            if row["statistic_name"]:
                statistics.add(row["statistic_name"])
            measures.add(row["measure"])
            if not row["key"] and row["adjustment"]:
                contrast_methods.add(
                    row["test"] if row["test"] in ("Games–Howell", "Tukey–Kramer") else "within"
                )
            display_rows.append(
                [
                    (found.spec.question or found.spec.id) if first else "",
                    study.label(found.spec.outcome) if first else "",
                    row["label"],
                    row["test"] or found.test,
                    _n_text(found) if first else "",
                    estimate,
                    interval(row["ci_low"], row["ci_high"], decimals) if pd.notna(row["ci_low"]) else "–",
                    _statistic_text(row),
                    p_value(row["p"]),
                    p_value(row["p_adjusted"]),
                    _effect_text(effect),
                    ("Planned" if found.planned else "Exploratory") if first else "",
                ]
            )
            data_rows.append(
                {
                    "analysis": found.spec.id,
                    "question": found.spec.question,
                    "planned": found.planned,
                    "family": found.family,
                    "outcome": found.spec.outcome,
                    "predictor": found.spec.predictor,
                    "design": found.design,
                    "test": found.test,
                    "method": found.method,
                    "n": found.n,
                    "n_missing": found.n_missing,
                    "subjects": found.subjects,
                    "clusters": found.clusters,
                    "term": row["term"],
                    "label": row["label"],
                    "row_test": row["test"],
                    "measure": row["measure"],
                    "unit": unit,
                    "estimate": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "confidence": study.decisions.confidence,
                    "statistic_name": row["statistic_name"],
                    "statistic": row["statistic"],
                    "df": row["df"],
                    "df2": row["df2"],
                    "p": row["p"],
                    "p_adjusted": row["p_adjusted"],
                    "adjustment": row["adjustment"],
                    "key": bool(row["key"]),
                    "effect_name": effect.get("name", ""),
                    "effect": effect.get("value"),
                    "effect_ci_low": effect.get("ci_low"),
                    "effect_ci_high": effect.get("ci_high"),
                }
            )
            first = False
    columns = [
        "Analysis",
        "Outcome",
        "Comparison or term",
        "Test",
        "n",
        "Estimate",
        f"{percent} CI",
        "Statistic",
        "p",
        "Adjusted p",
        f"Effect size ({percent} CI)",
        "Plan",
    ]
    notes = _table_notes(
        study, chosen, fitted, percent, symbols, statistics, contrast_methods, measures, planned
    )
    title = "Results of the planned analyses" if planned else "Exploratory analyses"
    return Table(
        id="results" if planned else "exploratory-analyses",
        title=title,
        display=pd.DataFrame(display_rows, columns=columns),
        data=pd.DataFrame(data_rows, columns=RESULT_COLUMNS),
        notes=notes,
    )


def _table_notes(
    study, chosen, fitted, percent, symbols, statistics, contrast_methods, measures, planned
) -> list[str]:
    notes = (
        _fitted_notes(study, fitted, percent, symbols, statistics, contrast_methods, measures)
        if fitted
        else []
    )
    notes.append(
        "Only the analyses listed in study.yml were run."
        + (
            " These were planned before the data were seen."
            if planned
            else " They were not planned before the data were seen; read them as exploratory."
        )
    )
    for found in chosen:
        if not found.supported:
            notes.append(f"Not fitted: {found.spec.id}. {found.reason}")
    return notes


def _fitted_notes(study, fitted, percent, symbols, statistics, contrast_methods, measures) -> list[str]:
    method = study.decisions.multiple_comparisons
    abbreviations = ["CI, confidence interval"]
    words = {
        "Hedges' g": "g, Hedges' g",
        "Cohen's dz": "dz, Cohen's dz",
        "omega squared": "ω², omega squared",
        "partial omega squared": "partial ω², partial omega squared",
        "Pearson's r": "r, Pearson's correlation",
        "odds ratio": "OR, odds ratio",
        "Cramér's V": "V, Cramér's V",
    }
    abbreviations += [words[name] for name in sorted(symbols) if name in words]
    if "odds ratio" in measures and "odds ratio" not in symbols:
        abbreviations.append("OR, odds ratio")
    notes = [
        "; ".join(abbreviations)
        + "."
        + (
            f" Test statistics ({', '.join(sorted(statistics))}) show their degrees of freedom in brackets "
            "where they have them."
            if statistics
            else ""
        ),
        f"Intervals are {percent} confidence intervals; p-values are two-sided.",
        "Each analysis uses the rows with every value it needs (complete cases); n counts the rows used, "
        "with the subjects, clusters or pairs in brackets where rows are not independent.",
    ]
    if "median difference" in measures:
        notes.append(
            "Rank-based rows estimate the Hodges–Lehmann median difference, with the interval found by "
            "inverting the rank test."
        )
    if "risk difference" in measures:
        notes.append(
            "Risk differences are differences in proportions with Newcombe's hybrid score interval; odds "
            "ratios beside them have Woolf's logit interval."
        )
    families: dict[str, list[str]] = {}
    for found in fitted:
        families.setdefault(found.family, []).append(found.spec.id)
    listed = "; ".join(f"{family}: {', '.join(members)}" for family, members in families.items())
    if method == "none":
        notes.append(
            "p-values are not adjusted for multiple comparisons (multiple_comparisons: none). Families: "
            f"{listed}."
        )
    else:
        notes.append(
            f"Adjusted p on each analysis's first row uses {ADJUSTMENT_NAMES[method]} within its family "
            f"({listed})."
        )
    if contrast_methods:
        parts = []
        if "Games–Howell" in contrast_methods:
            parts.append("Games–Howell after Welch's ANOVA, with simultaneous intervals")
        if "Tukey–Kramer" in contrast_methods:
            parts.append("Tukey–Kramer after one-way ANOVA, with simultaneous intervals")
        if "within" in contrast_methods:
            parts.append(
                f"{ADJUSTMENT_NAMES[method]} over the contrasts of one analysis, with unadjusted intervals"
            )
        notes.append(
            "Rows under an overall test are its contrasts, adjusted within the analysis: "
            + "; ".join(parts)
            + "."
        )
    return notes


def _key_numbers(found: AnalysisResult, prefix: str = "") -> dict:
    row = key_row(found)
    if row is None:
        return {}

    def plain(value):
        return None if value is None or pd.isna(value) else float(value)

    numbers = {
        f"{prefix}n": found.n,
        f"{prefix}estimate": plain(row["estimate"]),
        f"{prefix}ci_low": plain(row["ci_low"]),
        f"{prefix}ci_high": plain(row["ci_high"]),
        f"{prefix}statistic": plain(row["statistic"]),
        f"{prefix}p": plain(row["p"]),
        f"{prefix}p_adjusted": plain(row["p_adjusted"]),
    }
    if found.effect:
        numbers[f"{prefix}effect"] = plain(found.effect.get("value"))
    return numbers


def _depends(found_list: list[AnalysisResult]) -> tuple[str, ...]:
    return (*[f"analyses.{found.spec.id}" for found in found_list], *DATA_SECTIONS, "decisions")


def _add_results_table(result, chosen: list[AnalysisResult], planned: bool) -> None:
    if not chosen:
        return
    study = result.study
    table = results_table(study, result.data, chosen, planned)
    fitted = [found for found in chosen if found.supported]
    used = set()
    for found in fitted:
        if found.fitted is not None:
            used |= set(found.fitted.index)
    rows_used = max([len(used), *(found.n for found in fitted)])
    numbers = {}
    for found in fitted:
        numbers.update(_key_numbers(found, f"{found.spec.id}."))
    references = tuple(dict.fromkeys(reference for found in fitted for reference in found.references))
    methods = "; ".join(dict.fromkeys(f"{found.test} ({found.method})" for found in fitted))
    adjustment = study.decisions.multiple_comparisons
    result.add_table(
        table,
        section="main" if planned else "supplement",
        folder="tables" if planned else "supplement",
        depends=_depends(chosen),
        calculation=(
            f"Each analysis in study.yml on its complete cases: {methods}. Key p-values adjusted within each "
            f"family by {ADJUSTMENT_NAMES[adjustment]}"
            + (" (statsmodels.stats.multitest.multipletests)." if adjustment != "none" else ".")
            if fitted
            else "No analysis in this table could be fitted; its notes give the reasons."
        ),
        code="ada.research.analyses.results_table",
        rows=result.rows(used=rows_used),
        numbers=numbers,
        references=references,
    )


def model_text(study: Study, found: AnalysisResult) -> str:
    """The supplement's record of one model: what was fitted, on which rows, and what it found."""
    spec = found.spec
    percent = f"{study.decisions.confidence * 100:g}%"
    lines = [
        f"Analysis: {spec.id} ({'planned' if spec.planned else 'exploratory'}; family {spec.family})",
        f"Question: {spec.question or '(none given)'}",
        f"Outcome: {study.label(spec.outcome)}",
        f"{spec.kind.capitalize()}: {study.label(spec.predictor)}"
        + (f", by {study.label(spec.by)}" if spec.by else ""),
        f"Covariates: {', '.join(spec.covariates) or 'none'}",
        f"Design: {found.design}",
        "",
    ]
    if not found.supported:
        lines += [f"Not fitted: {found.reason}", ""]
        return "\n".join(lines)
    lines += [
        f"Test: {found.test}",
        f"Library call: {found.method}",
    ]
    if found.formula:
        lines.append(f"Formula: {found.formula}")
    lines.append(f"Rows used: {found.n:,}; rows left out for a missing value: {found.n_missing:,}")
    if found.subjects:
        lines.append(f"Subjects: {found.subjects:,}")
    if found.clusters:
        lines.append(f"Clusters: {found.clusters:,}")
    if found.groups:
        lines.append(
            "Rows per group: "
            + ", ".join(f"{level} (n = {count:,})" for level, count in found.groups.items())
        )
    lines += ["", f"Estimates ({percent} confidence intervals):", _estimates_text(found.estimates, percent)]
    if found.effect:
        lines += ["", f"Effect size: {_effect_text(found.effect)} ({found.effect['name']})"]
    lines += ["", "Diagnostics:"] + [f"- {item}" for item in found.diagnostics]
    lines += ["", "Warnings:"] + ([f"- {item}" for item in found.warnings] or ["- none"])
    if found.summary:
        lines += ["", "Output:", found.summary.rstrip()]
    lines += ["", "References:"] + [f"- {reference}" for reference in found.references]
    return "\n".join(lines) + "\n"


def _add_model_record(result, found: AnalysisResult) -> None:
    study = result.study
    folder = result.folder("supplement") / "models"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{found.spec.id}.txt"
    path.write_text(model_text(study, found), encoding="utf-8")
    result.assets.append(
        Asset(
            id=f"model-{found.spec.id}",
            kind="record",
            title=f"Model output: {found.spec.question or found.spec.id}",
            files=[path],
            depends=_depends([found]),
            calculation=(
                f"{found.test}: {found.method}." if found.supported else f"Not fitted: {found.reason}"
            ),
            references=found.references,
            code="ada.research.analyses.analyse",
            rows=result.rows(used=found.n),
            numbers=_key_numbers(found),
            section="supplement",
        )
    )
