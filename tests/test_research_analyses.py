import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from research_helpers import assay, assay_study, study_folder, weights
from scipy import special, stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway, confint_noncentrality
from statsmodels.stats.proportion import confint_proportions_2indep

from ada.research import build, load
from ada.research.analyses import (
    COLUMNS,
    adjust_families,
    analyse,
    games_howell,
    hedges_g,
    hodges_lehmann_location,
    hodges_lehmann_shift,
    key_row,
    omega_squared,
)
from ada.research.processing import read
from ada.research.results import MEAN_COLUMNS
from ada.research.study import StudyError


def study_dict(design: dict, variables: dict, analyses: list, data: str = "data.csv", **extra) -> dict:
    study = {
        "study": {"title": "Test study"},
        "data": {"path": data, "observation": "one row"},
        "design": design,
        "variables": variables,
        "analyses": analyses,
    }
    study.update(extra)
    return study


def analysis(ident: str, outcome: str, planned: bool = True, **keys) -> dict:
    return {"id": ident, "outcome": outcome, "planned": planned, **keys}


def run_analyses(frame: pd.DataFrame, study: dict):
    """Fit every analysis of ``study`` on ``frame`` the way a build does, and return the study and results."""
    with study_folder(frame, study, "data.csv") as path:
        loaded = load(path)
        data = read(loaded).data
        results = {spec.id: analyse(loaded, data, spec) for spec in loaded.analyses}
        adjust_families(loaded, results)
        return loaded, data, results


def two_groups(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Group": ["Control"] * 12 + ["Treated"] * 9,
            "Score": np.round(np.r_[rng.normal(10, 1.0, 12), rng.normal(12, 2.5, 9)], 3),
            "Age": np.round(rng.uniform(20, 60, 21), 1),
        }
    )


GROUP_VARIABLES = {
    "Group": {"role": "group", "type": "categorical", "levels": ["Control", "Treated"]},
    "Score": {"role": "outcome", "type": "numeric", "unit": "points"},
    "Age": {"role": "covariate", "type": "numeric", "unit": "years"},
}


def three_groups(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for level, (mean, sd, n) in {"A": (5, 1, 10), "B": (6, 2, 12), "C": (8, 3, 9)}.items():
        rows += [{"Arm": level, "Y": round(value, 3)} for value in rng.normal(mean, sd, n)]
    return pd.DataFrame(rows)


THREE_VARIABLES = {
    "Arm": {"role": "group", "type": "categorical", "levels": ["A", "B", "C"]},
    "Y": {"role": "outcome", "type": "numeric"},
}


def paired_scores(seed: int = 4) -> pd.DataFrame:
    """Twelve participants before and after; P12 lacks an after score and P01 has two before rows."""
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(1, 13):
        base = rng.normal(50, 8)
        rows.append({"Participant": f"P{person:02d}", "Session": "Before", "Score": round(base, 2)})
        if person != 12:
            after = base + 3 + rng.normal(0, 2)
            rows.append({"Participant": f"P{person:02d}", "Session": "After", "Score": round(after, 2)})
    rows.append({"Participant": "P01", "Session": "Before", "Score": round(rows[0]["Score"] + 1.0, 2)})
    return pd.DataFrame(rows)


PAIRED_VARIABLES = {
    "Participant": {"role": "identifier", "type": "text"},
    "Session": {"role": "group", "type": "categorical", "levels": ["Before", "After"]},
    "Score": {"role": "outcome", "type": "numeric"},
}


def clustered_cells(seed: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for animal in range(12):
        genotype = "WT" if animal < 6 else "KO"
        level = rng.normal(0, 1.5)
        for _ in range(8):
            rows.append(
                {
                    "Animal": f"A{animal:02d}",
                    "Genotype": genotype,
                    "Current": round(20 + (3 if genotype == "KO" else 0) + level + rng.normal(0, 1), 3),
                }
            )
    return pd.DataFrame(rows)


CLUSTER_VARIABLES = {
    "Animal": {"role": "identifier", "type": "text"},
    "Genotype": {"role": "group", "type": "categorical", "levels": ["WT", "KO"]},
    "Current": {"role": "outcome", "type": "numeric", "unit": "pA"},
}


def binary_outcomes(events=(4, 11), totals=(30, 30)) -> pd.DataFrame:
    rows = []
    for level, event, total in zip(("Placebo", "Drug"), events, totals, strict=True):
        rows += [{"Arm": level, "Response": "Yes" if index < event else "No"} for index in range(total)]
    frame = pd.DataFrame(rows)
    frame["Dose"] = np.linspace(0, 10, len(frame))
    return frame


BINARY_VARIABLES = {
    "Arm": {"role": "group", "type": "categorical", "levels": ["Placebo", "Drug"]},
    "Response": {"role": "outcome", "type": "binary", "levels": ["No", "Yes"]},
    "Dose": {"role": "covariate", "type": "numeric", "unit": "mg"},
}

WEIGHT_VARIABLES = {
    "Mouse": {"role": "identifier", "type": "text"},
    "Diet": {"role": "group", "type": "categorical", "levels": ["Chow", "High fat"]},
    "Day": {"role": "time", "type": "numeric"},
    "Weight (g)": {"role": "outcome", "type": "numeric", "unit": "g"},
}


class IndependentComparisonTests(unittest.TestCase):
    def test_welch_t_matches_scipy(self):
        frame = two_groups()
        _, data, results = run_analyses(
            frame,
            study_dict({"type": "independent"}, GROUP_VARIABLES, [analysis("a", "Score", compare="Group")]),
        )
        found = results["a"]
        control = frame.loc[frame.Group == "Control", "Score"].to_numpy()
        treated = frame.loc[frame.Group == "Treated", "Score"].to_numpy()
        expected = stats.ttest_ind(treated, control, equal_var=False)
        interval = expected.confidence_interval(0.95)
        row = key_row(found)
        self.assertEqual(found.test, "Welch's t-test")
        self.assertAlmostEqual(row["estimate"], treated.mean() - control.mean(), places=10)
        self.assertAlmostEqual(row["statistic"], expected.statistic, places=10)
        self.assertAlmostEqual(row["df"], expected.df, places=10)
        self.assertAlmostEqual(row["p"], expected.pvalue, places=12)
        self.assertAlmostEqual(row["ci_low"], interval.low, places=10)
        self.assertAlmostEqual(row["ci_high"], interval.high, places=10)
        self.assertEqual(list(found.estimates.columns), COLUMNS)
        self.assertEqual(found.n, 21)
        self.assertEqual(found.groups, {"Control": 12, "Treated": 9})
        self.assertEqual(list(found.group_means.columns), MEAN_COLUMNS)
        means = found.group_means.set_index("group")
        self.assertAlmostEqual(means.loc["Treated", "mean"], treated.mean())
        half = stats.t.ppf(0.975, 8) * treated.std(ddof=1) / 3
        self.assertAlmostEqual(means.loc["Treated", "ci_high"], treated.mean() + half)
        self.assertEqual(list(found.fitted.index), list(data.index))
        self.assertAlmostEqual(found.residuals.sum(), 0, places=8)
        self.assertTrue(any("Shapiro–Wilk" in item for item in found.diagnostics))
        self.assertTrue(any("Levene" in item for item in found.diagnostics))
        self.assertTrue(any("Welch, B. L. (1947)" in reference for reference in found.references))

    def test_hedges_g_matches_the_hand_formula(self):
        frame = two_groups()
        control = frame.loc[frame.Group == "Control", "Score"].to_numpy()
        treated = frame.loc[frame.Group == "Treated", "Score"].to_numpy()
        n1, n2 = len(control), len(treated)
        pooled = math.sqrt(((n1 - 1) * control.var(ddof=1) + (n2 - 1) * treated.var(ddof=1)) / (n1 + n2 - 2))
        d = (treated.mean() - control.mean()) / pooled
        m = n1 + n2 - 2
        correction = special.gamma(m / 2) / (math.sqrt(m / 2) * special.gamma((m - 1) / 2))
        effect = hedges_g(control, treated, 0.95)
        self.assertAlmostEqual(effect["value"], correction * d, places=10)
        self.assertAlmostEqual(correction, 1 - 3 / (4 * m - 1), places=3)
        # The interval ends are the noncentral t parameters that put the observed t at 2.5% and 97.5%.
        scale = math.sqrt(1 / n1 + 1 / n2)
        t = d / scale
        low = effect["ci_low"] / correction / scale
        high = effect["ci_high"] / correction / scale
        self.assertAlmostEqual(stats.nct.cdf(t, m, low), 0.975, places=6)
        self.assertAlmostEqual(stats.nct.cdf(t, m, high), 0.025, places=6)
        # Close to Hedges and Olkin's large-sample interval.
        se = math.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
        self.assertAlmostEqual(effect["ci_low"], correction * (d - 1.96 * se), delta=0.15)

    def test_the_effect_interval_survives_the_noncentral_t_tails(self):
        # SciPy's nct.cdf is NaN for t = -3 and a noncentrality of 20 or more on 15 df.
        from ada.research.analyses import noncentral_t_interval  # noqa: PLC0415

        low, high = noncentral_t_interval(-3.0, 15, 0.95)
        self.assertAlmostEqual(stats.nct.cdf(-3.0, 15, low), 0.975, places=6)
        self.assertAlmostEqual(stats.nct.cdf(-3.0, 15, high), 0.025, places=6)
        reference = np.array([90.1, 91.4, 89.7, 92.2, 90.8, 91.1, 90.4, 89.9, 91.7, 90.6])
        other = np.array([86.2, 85.9, 87.1, 86.6, 85.4, 86.8, 87.3])
        effect = hedges_g(reference, other, 0.95)
        self.assertTrue(effect["ci_low"] < effect["value"] < effect["ci_high"] < 0)

    def test_welch_anova_and_games_howell(self):
        frame = three_groups()
        _, _, results = run_analyses(
            frame, study_dict({"type": "independent"}, THREE_VARIABLES, [analysis("a", "Y", compare="Arm")])
        )
        found = results["a"]
        samples = [frame.loc[frame.Arm == level, "Y"].to_numpy() for level in "ABC"]
        expected = anova_oneway(samples, use_var="unequal")
        row = key_row(found)
        self.assertEqual(row["term"], "Arm")
        self.assertAlmostEqual(row["statistic"], expected.statistic, places=10)
        self.assertAlmostEqual(row["df"], expected.df_num)
        self.assertAlmostEqual(row["df2"], expected.df_denom, places=10)
        self.assertAlmostEqual(row["p"], expected.pvalue, places=12)
        contrasts = found.estimates[~found.estimates["key"]]
        self.assertEqual(list(contrasts["label"]), ["B − A", "C − A", "C − B"])
        # Games–Howell by hand for C − A.
        a, c = samples[0], samples[2]
        va, vc = a.var(ddof=1) / len(a), c.var(ddof=1) / len(c)
        df = (va + vc) ** 2 / (va**2 / (len(a) - 1) + vc**2 / (len(c) - 1))
        t = (c.mean() - a.mean()) / math.sqrt(va + vc)
        chosen = contrasts.set_index("label").loc["C − A"]
        self.assertAlmostEqual(chosen["statistic"], t, places=10)
        self.assertAlmostEqual(chosen["df"], df, places=10)
        self.assertAlmostEqual(chosen["p_adjusted"], stats.studentized_range.sf(abs(t) * math.sqrt(2), 3, df))
        margin = stats.studentized_range.ppf(0.95, 3, df) / math.sqrt(2) * math.sqrt(va + vc)
        self.assertAlmostEqual(chosen["ci_low"], c.mean() - a.mean() - margin, places=6)
        self.assertEqual(found.effect["name"], "omega squared")

    def test_games_howell_for_two_groups_is_welch(self):
        frame = two_groups()
        control = frame.loc[frame.Group == "Control", "Score"].to_numpy()
        treated = frame.loc[frame.Group == "Treated", "Score"].to_numpy()
        row = games_howell(["Control", "Treated"], [control, treated], 0.95)[0]
        welch = stats.ttest_ind(treated, control, equal_var=False)
        interval = welch.confidence_interval(0.95)
        self.assertAlmostEqual(row["p_adjusted"], welch.pvalue, places=5)
        self.assertAlmostEqual(row["ci_low"], interval.low, places=4)
        self.assertAlmostEqual(row["ci_high"], interval.high, places=4)

    def test_omega_squared_matches_the_formula_and_noncentral_f(self):
        f, df1, df2, n = 6.2, 2.0, 27.0, 30
        effect = omega_squared(f, df1, df2, 0.95)
        self.assertAlmostEqual(effect["value"], df1 * (f - 1) / (df1 * f + df2 + 1))
        # Equals Hays' estimate from sums of squares for a classic one-way ANOVA.
        frame = three_groups()
        samples = [frame.loc[frame.Arm == level, "Y"].to_numpy() for level in "ABC"]
        classic = stats.f_oneway(*samples)
        grand = np.concatenate(samples).mean()
        between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
        within = sum(((s - s.mean()) ** 2).sum() for s in samples)
        k, total = 3, sum(len(s) for s in samples)
        mse = within / (total - k)
        hays = (between - (k - 1) * mse) / (between + within + mse)
        self.assertAlmostEqual(omega_squared(classic.statistic, k - 1, total - k, 0.95)["value"], hays)
        noncentrality = confint_noncentrality(f, (df1, df2), alpha=0.05)
        upper = noncentrality[1] / df1
        self.assertAlmostEqual(effect["ci_high"], df1 * (upper - 1) / (df1 * upper + df2 + 1))
        self.assertLess(effect["ci_low"], effect["value"])
        del n

    def test_covariates_give_an_ols_adjusted_difference(self):
        frame = two_groups()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "independent"},
                GROUP_VARIABLES,
                [analysis("a", "Score", compare="Group", covariates=["Age"])],
            ),
        )
        found = results["a"]
        frame["Group"] = pd.Categorical(frame["Group"], categories=["Control", "Treated"])
        direct = smf.ols("Score ~ C(Group) + Age", data=frame).fit()
        term = "C(Group)[T.Treated]"
        row = key_row(found)
        self.assertEqual(found.test, "Linear regression (OLS) adjusted for Age")
        self.assertAlmostEqual(row["estimate"], direct.params[term], places=10)
        self.assertAlmostEqual(row["ci_low"], direct.conf_int().loc[term, 0], places=10)
        self.assertAlmostEqual(row["p"], direct.pvalues[term], places=12)
        self.assertEqual(row["df"], direct.df_resid)
        self.assertTrue(any("Breusch–Pagan" in item for item in found.diagnostics))
        self.assertTrue(any("Variance inflation" in item for item in found.diagnostics))
        self.assertIn("Formula as fitted: Score ~ C(Group) + Age", found.summary)
        self.assertNotIn("Time:  ", found.summary.split("Formula")[0])

    def test_student_t_and_mann_whitney_can_be_chosen(self):
        frame = two_groups()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "independent"},
                GROUP_VARIABLES,
                [
                    analysis("student", "Score", compare="Group", test="student-t"),
                    analysis("ranks", "Score", compare="Group", test="mann-whitney"),
                ],
            ),
        )
        control = frame.loc[frame.Group == "Control", "Score"].to_numpy()
        treated = frame.loc[frame.Group == "Treated", "Score"].to_numpy()
        self.assertAlmostEqual(key_row(results["student"])["p"], stats.ttest_ind(treated, control).pvalue)
        ranks = key_row(results["ranks"])
        expected = stats.mannwhitneyu(treated, control, alternative="two-sided")
        self.assertAlmostEqual(ranks["statistic"], expected.statistic)
        self.assertAlmostEqual(ranks["p"], expected.pvalue)
        self.assertAlmostEqual(ranks["estimate"], np.median(np.subtract.outer(treated, control)))

    def test_hodges_lehmann_intervals_invert_the_rank_tests(self):
        rng = np.random.default_rng(9)
        x, y = rng.normal(0, 1, 15), rng.normal(1, 1, 13)
        estimate, low, high = hodges_lehmann_shift(x, y, 0.95)

        def p_at(delta):
            return stats.mannwhitneyu(y - delta, x, alternative="two-sided").pvalue

        step = 1e-7
        self.assertGreaterEqual(p_at(low + step), 0.05)
        self.assertLess(p_at(low - step), 0.05)
        self.assertGreaterEqual(p_at(high - step), 0.05)
        self.assertLess(p_at(high + step), 0.05)
        self.assertLess(low, estimate)
        self.assertLess(estimate, high)
        differences = rng.normal(0.8, 1, 14)
        estimate, low, high = hodges_lehmann_location(differences, 0.95)
        self.assertGreaterEqual(stats.wilcoxon(differences - (low + step)).pvalue, 0.05)
        self.assertLess(stats.wilcoxon(differences - (low - step)).pvalue, 0.05)
        self.assertLess(stats.wilcoxon(differences - (high + step)).pvalue, 0.05)

    def test_relate_gives_pearson_spearman_and_the_slope(self):
        frame = two_groups()
        _, _, results = run_analyses(
            frame,
            study_dict({"type": "independent"}, GROUP_VARIABLES, [analysis("a", "Score", relate="Age")]),
        )
        found = results["a"]
        rows = found.estimates.set_index("term")
        pearson = stats.pearsonr(frame.Age, frame.Score)
        interval = pearson.confidence_interval(0.95)
        regression = stats.linregress(frame.Age, frame.Score)
        self.assertAlmostEqual(rows.loc["pearson", "estimate"], pearson.statistic)
        self.assertAlmostEqual(found.effect["ci_low"], interval.low)
        self.assertAlmostEqual(
            rows.loc["spearman", "estimate"], stats.spearmanr(frame.Age, frame.Score).statistic
        )
        self.assertTrue(rows.loc["Age", "key"])
        self.assertAlmostEqual(rows.loc["Age", "estimate"], regression.slope)
        self.assertAlmostEqual(rows.loc["Age", "p"], regression.pvalue)
        self.assertAlmostEqual(rows.loc["Intercept", "estimate"], regression.intercept)
        self.assertEqual(rows.loc["Intercept", "measure"], "intercept")
        self.assertIsNone(found.group_means)


class PairedTests(unittest.TestCase):
    def test_paired_t_matches_scipy_and_counts_what_it_leaves_out(self):
        frame = paired_scores()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "paired", "subject": "Participant"},
                PAIRED_VARIABLES,
                [analysis("a", "Score", compare="Session")],
            ),
        )
        found = results["a"]
        cells = frame.groupby(["Participant", "Session"])["Score"].mean().unstack().dropna()
        expected = stats.ttest_rel(cells["After"], cells["Before"])
        interval = expected.confidence_interval(0.95)
        row = key_row(found)
        self.assertEqual(found.test, "Paired t-test")
        self.assertAlmostEqual(row["statistic"], expected.statistic, places=10)
        self.assertAlmostEqual(row["p"], expected.pvalue, places=12)
        self.assertAlmostEqual(row["ci_low"], interval.low, places=10)
        self.assertEqual(found.subjects, 11)
        self.assertEqual(found.n, len(frame) - 1)
        self.assertTrue(any("P12" in item and "left out (1 row)" in item for item in found.diagnostics))
        self.assertTrue(any("averaged" in item for item in found.diagnostics))
        differences = (cells["After"] - cells["Before"]).to_numpy()
        self.assertAlmostEqual(found.effect["value"], differences.mean() / differences.std(ddof=1))
        self.assertEqual(found.effect["name"], "Cohen's dz")
        # The additive subject + condition model leaves ±(d − mean d) / 2 for a subject with one row each.
        rows = frame[frame.Participant == "P05"]
        d = differences[list(cells.index).index("P05")]
        after = rows.index[rows.Session == "After"][0]
        self.assertAlmostEqual(found.residuals[after], (d - differences.mean()) / 2)
        self.assertNotIn(frame.index[(frame.Participant == "P12")][0], found.residuals.index)

    def test_three_conditions_are_refused_in_a_paired_design(self):
        frame = paired_scores()
        frame.loc[frame.index[:4], "Session"] = "Later"
        variables = dict(PAIRED_VARIABLES)
        variables["Session"] = {
            "role": "group",
            "type": "categorical",
            "levels": ["Before", "After", "Later"],
        }
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "paired", "subject": "Participant"},
                variables,
                [analysis("a", "Score", compare="Session")],
            ),
        )
        self.assertFalse(results["a"].supported)
        self.assertIn("exactly two conditions", results["a"].reason)


class MixedModelTests(unittest.TestCase):
    def test_clustered_comparison_matches_statsmodels(self):
        frame = clustered_cells()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "clustered", "cluster": "Animal"},
                CLUSTER_VARIABLES,
                [analysis("a", "Current", compare="Genotype")],
            ),
        )
        found = results["a"]
        frame["Genotype"] = pd.Categorical(frame["Genotype"], categories=["WT", "KO"])
        direct = smf.mixedlm("Current ~ C(Genotype)", frame, groups="Animal").fit(reml=True)
        term = "C(Genotype)[T.KO]"
        row = key_row(found)
        self.assertAlmostEqual(row["estimate"], direct.params[term], places=6)
        self.assertAlmostEqual(row["ci_low"], direct.conf_int().loc[term, 0], places=6)
        self.assertAlmostEqual(row["p"], direct.pvalues[term], places=8)
        self.assertEqual(found.clusters, 12)
        icc = direct.cov_re.iloc[0, 0] / (direct.cov_re.iloc[0, 0] + direct.scale)
        self.assertTrue(any(f"ICC) {icc:.2g}" in item for item in found.diagnostics))
        means = found.group_means.set_index("group")
        self.assertEqual(means.loc["KO", "n"], 6)
        self.assertEqual(found.formula, "Current ~ Genotype + (1 | Animal)")
        self.assertEqual(len(found.fitted), len(frame))

    def test_change_over_time_reports_the_interaction(self):
        frame = weights()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "repeated", "subject": "Mouse", "time": "Day"},
                WEIGHT_VARIABLES,
                [
                    analysis("categorical", "Weight (g)", change="Day", by="Diet"),
                    analysis("linear", "Weight (g)", change="Day", by="Diet", time_as="linear"),
                    analysis("overall", "Weight (g)", change="Day"),
                ],
            ),
        )
        data = frame.copy()
        data["Diet"] = pd.Categorical(data["Diet"], categories=["Chow", "High fat"])
        data["DayC"] = pd.Categorical(data["Day"].astype(str), categories=["0", "7", "14", "21", "28"])
        direct = smf.mixedlm("Q('Weight (g)') ~ C(Diet) * C(DayC)", data, groups="Mouse").fit(reml=True)
        names = [name for name in direct.params.index if ":" in name]
        matrix = np.zeros((len(names), len(direct.params)))
        for row, name in enumerate(names):
            matrix[row, list(direct.params.index).index(name)] = 1
        wald = direct.wald_test(matrix, use_f=False, scalar=True)
        categorical = results["categorical"]
        key = key_row(categorical)
        self.assertEqual(key["statistic_name"], "χ²")
        self.assertEqual(key["df"], 4)
        self.assertAlmostEqual(key["statistic"], float(wald.statistic), places=4)
        self.assertEqual(len(categorical.estimates), 5)
        self.assertEqual(len(categorical.group_means), 10)
        self.assertEqual(sorted(categorical.group_means["time"].unique()), [0, 7, 14, 21, 28])
        linear = results["linear"]
        direct_linear = smf.mixedlm("Q('Weight (g)') ~ C(Diet) * Day", data, groups="Mouse").fit(reml=True)
        self.assertAlmostEqual(
            key_row(linear)["estimate"], direct_linear.params["C(Diet)[T.High fat]:Day"], places=6
        )
        slopes = linear.estimates.set_index("label")
        self.assertAlmostEqual(
            slopes.loc["Slope over Day in High fat", "estimate"],
            direct_linear.params["Day"] + direct_linear.params["C(Diet)[T.High fat]:Day"],
            places=6,
        )
        self.assertTrue(any("no random slope" in item for item in linear.diagnostics))
        self.assertEqual(key_row(results["overall"])["term"], "Day")

    def test_repeated_comparison_notes_the_time_it_leaves_out(self):
        frame = weights()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "repeated", "subject": "Mouse", "time": "Day"},
                WEIGHT_VARIABLES,
                [analysis("a", "Weight (g)", compare="Diet")],
            ),
        )
        self.assertTrue(results["a"].supported)
        self.assertEqual(results["a"].subjects, 12)
        self.assertTrue(any("not in this model" in item for item in results["a"].diagnostics))


class BinaryTests(unittest.TestCase):
    def test_rates_give_newcombe_risk_difference_and_odds_ratio(self):
        frame = binary_outcomes(events=(8, 17), totals=(30, 30))
        _, _, results = run_analyses(
            frame,
            study_dict({"type": "independent"}, BINARY_VARIABLES, [analysis("a", "Response", compare="Arm")]),
        )
        found = results["a"]
        row = key_row(found)
        low, high = confint_proportions_2indep(17, 30, 8, 30, method="newcomb", compare="diff")
        chi2 = stats.chi2_contingency(np.array([[8, 22], [17, 13]]), correction=False)
        self.assertAlmostEqual(row["estimate"], 17 / 30 - 8 / 30)
        self.assertAlmostEqual(row["ci_low"], low)
        self.assertAlmostEqual(row["ci_high"], high)
        self.assertAlmostEqual(row["statistic"], chi2.statistic)
        self.assertAlmostEqual(row["p"], chi2.pvalue)
        self.assertAlmostEqual(found.effect["value"], (17 * 22) / (13 * 8))
        self.assertEqual(found.test, "Comparison of proportions (χ² test)")
        self.assertAlmostEqual(found.group_means.set_index("group").loc["Drug", "mean"], 17 / 30)

    def test_small_expected_counts_switch_to_fisher(self):
        frame = binary_outcomes(events=(1, 6), totals=(10, 10))
        _, _, results = run_analyses(
            frame,
            study_dict({"type": "independent"}, BINARY_VARIABLES, [analysis("a", "Response", compare="Arm")]),
        )
        row = key_row(results["a"])
        self.assertAlmostEqual(row["p"], stats.fisher_exact(np.array([[1, 9], [6, 4]])).pvalue)
        self.assertTrue(math.isnan(row["statistic"]))
        self.assertIn("Fisher", results["a"].test)

    def test_logistic_relate_matches_statsmodels(self):
        rng = np.random.default_rng(3)
        dose = rng.uniform(0, 10, 80)
        response = np.where(rng.uniform(size=80) < 1 / (1 + np.exp(-(dose - 5) / 2)), "Yes", "No")
        frame = pd.DataFrame({"Arm": "Drug", "Response": response, "Dose": np.round(dose, 3)})
        _, _, results = run_analyses(
            frame,
            study_dict({"type": "independent"}, BINARY_VARIABLES, [analysis("a", "Response", relate="Dose")]),
        )
        frame["y"] = (frame.Response == "Yes").astype(int)
        direct = smf.logit("y ~ Dose", frame).fit(disp=0)
        row = key_row(results["a"])
        self.assertAlmostEqual(row["estimate"], math.exp(direct.params["Dose"]), places=6)
        self.assertAlmostEqual(row["ci_high"], math.exp(direct.conf_int().loc["Dose", 1]), places=6)
        self.assertAlmostEqual(row["p"], direct.pvalues["Dose"], places=8)
        self.assertEqual(row["label"], "Odds ratio per mg of Dose")


class OtherTestsTests(unittest.TestCase):
    def test_classic_anova_uses_tukey_kramer(self):
        frame = three_groups()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "independent"}, THREE_VARIABLES, [analysis("a", "Y", compare="Arm", test="anova")]
            ),
        )
        found = results["a"]
        samples = [frame.loc[frame.Arm == level, "Y"].to_numpy() for level in "ABC"]
        self.assertAlmostEqual(key_row(found)["statistic"], stats.f_oneway(*samples).statistic)
        tukey = stats.tukey_hsd(*samples)
        row = found.estimates.set_index("label").loc["C − A"]
        self.assertAlmostEqual(row["estimate"], samples[2].mean() - samples[0].mean())
        self.assertAlmostEqual(row["p_adjusted"], tukey.pvalue[2, 0])
        self.assertAlmostEqual(row["ci_low"], tukey.confidence_interval(0.95).low[2, 0])
        self.assertEqual(row["test"], "Tukey–Kramer")

    def test_kruskal_matches_scipy_with_adjusted_pairs(self):
        frame = three_groups()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "independent"}, THREE_VARIABLES, [analysis("a", "Y", compare="Arm", test="kruskal")]
            ),
        )
        found = results["a"]
        samples = [frame.loc[frame.Arm == level, "Y"].to_numpy() for level in "ABC"]
        expected = stats.kruskal(*samples)
        self.assertAlmostEqual(key_row(found)["statistic"], expected.statistic)
        self.assertAlmostEqual(key_row(found)["p"], expected.pvalue)
        pairs = found.estimates[~found.estimates["key"]]
        np.testing.assert_allclose(pairs["p_adjusted"], multipletests(pairs["p"], method="holm")[1])

    def test_wilcoxon_matches_scipy(self):
        frame = paired_scores()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "paired", "subject": "Participant"},
                PAIRED_VARIABLES,
                [analysis("a", "Score", compare="Session", test="wilcoxon")],
            ),
        )
        cells = frame.groupby(["Participant", "Session"])["Score"].mean().unstack().dropna()
        expected = stats.wilcoxon(cells["After"] - cells["Before"])
        row = key_row(results["a"])
        self.assertAlmostEqual(row["statistic"], expected.statistic)
        self.assertAlmostEqual(row["p"], expected.pvalue)
        self.assertEqual(row["measure"], "median difference")

    def test_paired_with_covariates_is_a_mixed_model(self):
        frame = paired_scores()
        frame["Hours"] = np.resize([1.0, 2.5, 4.0, 3.0, 0.5], len(frame))
        variables = {**PAIRED_VARIABLES, "Hours": {"role": "covariate", "type": "numeric"}}
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "paired", "subject": "Participant"},
                variables,
                [analysis("a", "Score", compare="Session", covariates=["Hours"])],
            ),
        )
        self.assertTrue(results["a"].test.startswith("Linear mixed model"))
        self.assertEqual(results["a"].subjects, 12)

    def test_relate_in_a_repeated_design_is_a_mixed_slope(self):
        frame = weights()
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "repeated", "subject": "Mouse", "time": "Day"},
                WEIGHT_VARIABLES,
                [analysis("a", "Weight (g)", relate="Day")],
            ),
        )
        direct = smf.mixedlm("Q('Weight (g)') ~ Day", frame, groups="Mouse").fit(reml=True)
        self.assertAlmostEqual(key_row(results["a"])["estimate"], direct.params["Day"], places=6)
        rows = results["a"].estimates.set_index("term")
        self.assertEqual(key_row(results["a"])["term"], "Day")
        self.assertAlmostEqual(rows.loc["Intercept", "estimate"], direct.params["Intercept"], places=6)

    def test_categorical_time_points_keep_their_order(self):
        frame = weights()
        frame["Visit"] = frame["Day"].map(
            {0: "Baseline", 7: "Week 1", 14: "Week 2", 21: "Week 3", 28: "Week 4"}
        )
        variables = {
            **WEIGHT_VARIABLES,
            "Visit": {
                "role": "time",
                "type": "categorical",
                "levels": ["Baseline", "Week 1", "Week 2", "Week 3", "Week 4"],
            },
        }
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "repeated", "subject": "Mouse", "time": "Visit"},
                variables,
                [analysis("a", "Weight (g)", change="Visit")],
            ),
        )
        labels = list(results["a"].estimates["label"])
        self.assertEqual(labels[1], "Visit Week 1 − Baseline")
        self.assertEqual(list(results["a"].group_means["time"])[:2], ["Baseline", "Week 1"])

    def test_three_groups_with_a_binary_outcome(self):
        frame = binary_outcomes(events=(5, 12), totals=(25, 25))
        extra = pd.DataFrame({"Arm": "High", "Response": ["Yes"] * 15 + ["No"] * 10, "Dose": 5.0})
        frame = pd.concat([frame, extra], ignore_index=True)
        variables = {
            **BINARY_VARIABLES,
            "Arm": {"role": "group", "type": "categorical", "levels": ["Placebo", "Drug", "High"]},
        }
        _, _, results = run_analyses(
            frame, study_dict({"type": "independent"}, variables, [analysis("a", "Response", compare="Arm")])
        )
        found = results["a"]
        table = np.array([[5, 20], [12, 13], [15, 10]])
        expected = stats.chi2_contingency(table, correction=False)
        self.assertAlmostEqual(key_row(found)["statistic"], expected.statistic)
        self.assertEqual(key_row(found)["df"], 2)
        differences = found.estimates[found.estimates["measure"] == "risk difference"]
        np.testing.assert_allclose(
            differences["p_adjusted"], multipletests(differences["p"], method="holm")[1]
        )
        self.assertEqual(found.effect["name"], "Cramér's V")

    def test_small_counts_in_a_larger_table_use_a_seeded_monte_carlo(self):
        frame = binary_outcomes(events=(1, 4), totals=(8, 8))
        extra = pd.DataFrame({"Arm": "High", "Response": ["Yes"] * 6 + ["No"] * 2, "Dose": 5.0})
        frame = pd.concat([frame, extra], ignore_index=True)
        variables = {
            **BINARY_VARIABLES,
            "Arm": {"role": "group", "type": "categorical", "levels": ["Placebo", "Drug", "High"]},
        }
        study = study_dict({"type": "independent"}, variables, [analysis("a", "Response", compare="Arm")])
        first = key_row(run_analyses(frame, study)[2]["a"])["p"]
        second = key_row(run_analyses(frame, study)[2]["a"])["p"]
        self.assertEqual(first, second)
        self.assertGreater(first, 0)

    def test_logistic_comparison_with_a_covariate(self):
        frame = binary_outcomes(events=(9, 18), totals=(40, 40))
        frame["Dose"] = np.resize([1.0, 3.0, 2.0, 5.0, 4.0], len(frame))
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "independent"},
                BINARY_VARIABLES,
                [analysis("a", "Response", compare="Arm", covariates=["Dose"])],
            ),
        )
        frame["y"] = (frame.Response == "Yes").astype(int)
        frame["Arm"] = pd.Categorical(frame.Arm, categories=["Placebo", "Drug"])
        direct = smf.logit("y ~ C(Arm) + Dose", frame).fit(disp=0)
        term = "C(Arm)[T.Drug]"
        row = key_row(results["a"])
        self.assertAlmostEqual(row["estimate"], math.exp(direct.params[term]), places=6)
        self.assertAlmostEqual(row["p"], direct.pvalues[term], places=8)

    def test_perfect_separation_is_refused(self):
        frame = pd.DataFrame(
            {"Arm": "Drug", "Response": ["No"] * 10 + ["Yes"] * 10, "Dose": np.arange(20, dtype=float)}
        )
        _, _, results = run_analyses(
            frame,
            study_dict({"type": "independent"}, BINARY_VARIABLES, [analysis("a", "Response", relate="Dose")]),
        )
        self.assertFalse(results["a"].supported)
        self.assertIn("separates the outcome", results["a"].reason)
        self.assertIn("custom:", results["a"].reason)

    def test_nested_rows_in_a_clustered_design_are_refused(self):
        frame = clustered_cells()
        frame["Cell"] = np.resize(["c1", "c2"], len(frame))
        variables = {**CLUSTER_VARIABLES, "Cell": {"role": "identifier", "type": "text"}}
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "clustered", "cluster": "Animal", "subject": "Cell"},
                variables,
                [analysis("a", "Current", compare="Genotype")],
            ),
        )
        self.assertFalse(results["a"].supported)
        self.assertIn("two levels", results["a"].reason)

    def test_the_event_is_the_second_level_or_the_one_named_so(self):
        frame = binary_outcomes(events=(8, 17), totals=(30, 30))
        frame["Response"] = frame["Response"].map({"Yes": "1", "No": "0"})
        variables = {**BINARY_VARIABLES, "Response": {"role": "outcome", "type": "binary"}}
        _, _, results = run_analyses(
            frame, study_dict({"type": "independent"}, variables, [analysis("a", "Response", compare="Arm")])
        )
        self.assertAlmostEqual(key_row(results["a"])["estimate"], 17 / 30 - 8 / 30)
        self.assertTrue(any("1 is counted as the event" in item for item in results["a"].diagnostics))


class RefusalTests(unittest.TestCase):
    def refused(self, frame, study) -> str:
        _, _, results = run_analyses(frame, study)
        found = next(iter(results.values()))
        self.assertFalse(found.supported)
        self.assertTrue(found.estimates.empty)
        return found.reason

    def test_categorical_outcome_with_three_levels(self):
        frame = three_groups()
        frame["Grade"] = np.resize(["low", "mid", "high"], len(frame))
        variables = {**THREE_VARIABLES, "Grade": {"role": "outcome", "type": "categorical"}}
        reason = self.refused(
            frame, study_dict({"type": "independent"}, variables, [analysis("a", "Grade", compare="Arm")])
        )
        self.assertIn("3 categories", reason)
        self.assertIn("custom", reason)

    def test_group_with_one_level_after_exclusions(self):
        frame = two_groups()
        study = study_dict(
            {"type": "independent"},
            GROUP_VARIABLES,
            [analysis("a", "Score", compare="Group")],
            processing=[
                {"exclude": {"where": {"column": "Group", "equals": "Treated"}}, "reason": "Batch lost."}
            ],
        )
        self.assertIn("fewer than 2 levels", self.refused(frame, study))

    def test_a_group_with_one_row(self):
        frame = two_groups().iloc[:13]
        reason = self.refused(
            frame,
            study_dict({"type": "independent"}, GROUP_VARIABLES, [analysis("a", "Score", compare="Group")]),
        )
        self.assertIn("at least 2 rows", reason)
        self.assertIn("Treated (1)", reason)

    def test_compare_on_a_numeric_predictor(self):
        reason = self.refused(
            two_groups(),
            study_dict({"type": "independent"}, GROUP_VARIABLES, [analysis("a", "Score", compare="Age")]),
        )
        self.assertIn("relate:", reason)

    def test_relate_on_a_categorical_predictor(self):
        reason = self.refused(
            two_groups(),
            study_dict({"type": "independent"}, GROUP_VARIABLES, [analysis("a", "Score", relate="Group")]),
        )
        self.assertIn("compare:", reason)

    def test_change_without_a_subject(self):
        frame = clustered_cells()
        frame["Day"] = np.resize([0, 1, 2, 3], len(frame))
        variables = {**CLUSTER_VARIABLES, "Day": {"role": "time", "type": "numeric"}}
        reason = self.refused(
            frame,
            study_dict(
                {"type": "clustered", "cluster": "Animal"},
                variables,
                [analysis("a", "Current", change="Day")],
            ),
        )
        self.assertIn("design.subject", reason)
        reason = self.refused(
            frame, study_dict({"type": "independent"}, variables, [analysis("a", "Current", change="Day")])
        )
        self.assertIn("independent", reason)

    def test_binary_outcome_in_a_repeated_design(self):
        frame = weights()
        frame["Heavy"] = np.where(frame["Weight (g)"] > 22, "yes", "no")
        variables = {
            **WEIGHT_VARIABLES,
            "Heavy": {"role": "outcome", "type": "binary", "levels": ["no", "yes"]},
        }
        reason = self.refused(
            frame,
            study_dict(
                {"type": "repeated", "subject": "Mouse"}, variables, [analysis("a", "Heavy", compare="Diet")]
            ),
        )
        self.assertIn("binary", reason)
        self.assertIn("custom:", reason)

    def test_a_test_the_design_does_not_allow(self):
        reason = self.refused(
            two_groups(),
            study_dict(
                {"type": "independent"},
                GROUP_VARIABLES,
                [analysis("a", "Score", compare="Group", test="paired-t")],
            ),
        )
        self.assertIn("test: paired-t", reason)
        self.assertIn("welch-t, student-t, mann-whitney, ols", reason)

    def test_by_outside_a_change_analysis(self):
        reason = self.refused(
            two_groups(),
            study_dict(
                {"type": "independent"}, GROUP_VARIABLES, [analysis("a", "Score", relate="Age", by="Group")]
            ),
        )
        self.assertIn("change analyses only", reason)


class MissingAndMultiplicityTests(unittest.TestCase):
    def test_missing_rows_are_counted_per_analysis(self):
        frame = two_groups()
        frame.loc[[0, 15], "Score"] = np.nan
        frame.loc[3, "Age"] = np.nan
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "independent"},
                GROUP_VARIABLES,
                [
                    analysis("plain", "Score", compare="Group"),
                    analysis("adjusted", "Score", compare="Group", covariates=["Age"]),
                ],
            ),
        )
        self.assertEqual((results["plain"].n, results["plain"].n_missing), (19, 2))
        self.assertEqual((results["adjusted"].n, results["adjusted"].n_missing), (18, 3))
        self.assertTrue(any("Score 2, Age 1" in item for item in results["adjusted"].diagnostics))

    def test_holm_adjusts_within_each_family(self):
        frame = two_groups()
        frame["Other"] = frame["Score"] * 0.5 + np.random.default_rng(1).normal(0, 1, len(frame))
        variables = {**GROUP_VARIABLES, "Other": {"role": "outcome", "type": "numeric"}}
        _, _, results = run_analyses(
            frame,
            study_dict(
                {"type": "independent"},
                variables,
                [
                    analysis("one", "Score", compare="Group", family="main"),
                    analysis("two", "Other", compare="Group", family="main"),
                    analysis("three", "Score", relate="Age", family="main"),
                    analysis("alone", "Other", relate="Age"),
                ],
            ),
        )
        p_values = [key_row(results[ident])["p"] for ident in ("one", "two", "three")]
        expected = multipletests(p_values, method="holm")[1]
        for ident, value in zip(("one", "two", "three"), expected, strict=True):
            self.assertAlmostEqual(key_row(results[ident])["p_adjusted"], value)
        self.assertAlmostEqual(key_row(results["alone"])["p_adjusted"], key_row(results["alone"])["p"])
        self.assertIn("family main (3 analyses)", key_row(results["one"])["adjustment"])

    def test_other_methods_and_none(self):
        frame = two_groups()
        frame["Other"] = frame["Score"] + np.random.default_rng(2).normal(0, 2, len(frame))
        variables = {**GROUP_VARIABLES, "Other": {"role": "outcome", "type": "numeric"}}
        analyses = [
            analysis("one", "Score", compare="Group", family="main"),
            analysis("two", "Other", compare="Group", family="main"),
        ]
        for method in ("bonferroni", "fdr_bh", "none"):
            study = study_dict(
                {"type": "independent"}, variables, analyses, decisions={"multiple_comparisons": method}
            )
            _, _, results = run_analyses(frame, study)
            p_values = [key_row(results[ident])["p"] for ident in ("one", "two")]
            adjusted = [key_row(results[ident])["p_adjusted"] for ident in ("one", "two")]
            if method == "none":
                self.assertTrue(all(math.isnan(value) for value in adjusted))
            else:
                np.testing.assert_allclose(adjusted, multipletests(p_values, method=method)[1])

    def test_the_confidence_level_is_the_studys(self):
        frame = two_groups()
        study = study_dict(
            {"type": "independent"},
            GROUP_VARIABLES,
            [analysis("a", "Score", compare="Group")],
            decisions={"confidence": 0.9},
        )
        _, _, results = run_analyses(frame, study)
        control = frame.loc[frame.Group == "Control", "Score"].to_numpy()
        treated = frame.loc[frame.Group == "Treated", "Score"].to_numpy()
        interval = stats.ttest_ind(treated, control, equal_var=False).confidence_interval(0.9)
        self.assertAlmostEqual(key_row(results["a"])["ci_low"], interval.low)


class StudyKeyTests(unittest.TestCase):
    def test_time_as_is_read_and_checked(self):
        study = study_dict(
            {"type": "repeated", "subject": "Mouse"},
            WEIGHT_VARIABLES,
            [analysis("a", "Weight (g)", change="Day", time_as="linear")],
        )
        with study_folder(weights(), study, "data.csv") as path:
            self.assertEqual(load(path).analyses[0].time_as, "linear")
        for value, entry in (("curved", {"change": "Day"}), ("linear", {"compare": "Diet"})):
            study["analyses"] = [analysis("a", "Weight (g)", time_as=value, **entry)]
            with study_folder(weights(), study, "data.csv") as path, self.assertRaises(StudyError) as raised:
                load(path)
            self.assertIn("time_as", str(raised.exception))


class BuildOutputTests(unittest.TestCase):
    def assay_with_analyses(self) -> dict:
        study = assay_study()
        study["processing"] = [
            {"exclude": {"where": {"column": "Viability (%)", "below": 0}}, "reason": "Instrument error."}
        ]
        study["analyses"] = [
            {
                "id": "primary",
                "outcome": "Viability (%)",
                "compare": "Treatment",
                "planned": True,
                "family": "p",
            },
            {
                "id": "adjusted",
                "outcome": "Viability (%)",
                "compare": "Treatment",
                "covariates": ["Passage"],
                "planned": True,
                "family": "p",
            },
            {"id": "passage", "outcome": "Viability (%)", "relate": "Passage", "planned": False},
            {"id": "wrong", "outcome": "Viability (%)", "compare": "Passage", "planned": False},
        ]
        return study

    def test_results_tables_split_planned_from_exploratory(self):
        with study_folder(assay(), self.assay_with_analyses()) as path:
            built = build(path)
            out = built.out
            planned = pd.read_csv(out / "tables" / "results.csv")
            exploratory = pd.read_csv(out / "supplement" / "exploratory-analyses.csv")
            self.assertEqual(set(planned["analysis"]), {"primary", "adjusted"})
            self.assertEqual(set(exploratory["analysis"]), {"passage"})
            self.assertTrue(planned["planned"].all())
            self.assertFalse(exploratory["planned"].any())
            primary = built.results["primary"]
            row = planned[(planned["analysis"] == "primary") & planned["key"]].iloc[0]
            self.assertAlmostEqual(row["statistic"], key_row(primary)["statistic"], places=10)
            self.assertAlmostEqual(row["p_adjusted"], key_row(primary)["p_adjusted"], places=12)
            self.assertAlmostEqual(row["effect"], primary.effect["value"], places=10)
            contrast = planned[planned["label"] == "High − Control"].iloc[0]
            expected = primary.estimates.set_index("label").loc["High − Control"]
            self.assertAlmostEqual(contrast["estimate"], expected["estimate"], places=10)
            tables = {table.id: table for table in built.tables}
            self.assertEqual(tables["results"].number, "Table 2")
            self.assertEqual(tables["exploratory-analyses"].number, "Table S1")
            display = tables["results"].display
            self.assertIn("Planned", set(display["Plan"]))
            self.assertIn("95% CI", display.columns)
            notes = " ".join(tables["exploratory-analyses"].notes)
            self.assertIn("Not fitted: wrong.", notes)
            self.assertIn("Only the analyses listed in study.yml were run.", notes)
            self.assertNotIn("wrong", set(exploratory["analysis"]))
            self.assertNotIn("intercept", set(exploratory["measure"]))
            self.assertIn("Intercept", set(built.results["passage"].estimates["term"]))
            self.assertIn("Holm's method", " ".join(tables["results"].notes))
            model = (out / "supplement" / "models" / "primary.txt").read_text()
            self.assertIn("Welch's ANOVA", model)
            self.assertIn("Rows used: 34; rows left out for a missing value: 1", model)
            self.assertIn("Not fitted:", (out / "supplement" / "models" / "wrong.txt").read_text())
            record = json.loads((out / "provenance.json").read_text())
            asset = record["assets"]["model-primary"]
            self.assertEqual(asset["section"], "supplement")
            self.assertIn("analyses.primary", asset["depends_on"])
            self.assertAlmostEqual(asset["numbers"]["p"], key_row(primary)["p"], delta=1e-12)
            self.assertIn("primary.estimate", record["assets"]["results"]["numbers"])
            self.assertTrue(any("wrong was not fitted" in warning for warning in built.warnings))

    def test_a_planned_analysis_that_cannot_be_fitted_keeps_its_table(self):
        study = assay_study()
        study["analyses"] = [
            {"id": "wrong", "outcome": "Viability (%)", "compare": "Passage", "planned": True, "family": "p"}
        ]
        with study_folder(assay(), study) as path:
            built = build(path)
            table = pd.read_csv(built.out / "tables" / "results.csv")
            notes = next(item for item in built.tables if item.id == "results").notes
        self.assertTrue(table.empty)
        self.assertIn("estimate", table.columns)
        self.assertTrue(any(note.startswith("Not fitted: wrong.") and "relate:" in note for note in notes))

    def test_a_rebuild_gives_identical_results_files(self):
        with study_folder(assay(), self.assay_with_analyses()) as path:
            first = build(path, out=path.parent / "one").out
            second = build(path, out=path.parent / "two").out
            for name in (
                "tables/results.csv",
                "tables/results.tex",
                "supplement/exploratory-analyses.csv",
                "supplement/models/primary.txt",
                "supplement/models/adjusted.txt",
            ):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def test_a_changed_decision_shows_old_and_new_numbers(self):
        with study_folder(assay(), self.assay_with_analyses()) as path:
            build(path)
            study = yaml.safe_load(path.read_text())
            study["decisions"] = {"multiple_comparisons": "bonferroni"}
            path.write_text(yaml.safe_dump(study, sort_keys=False, allow_unicode=True))
            built = build(path)
            changed = {change.asset: change for change in built.changes}
            self.assertIn("model-primary", changed)
            reasons = " ".join(changed["model-primary"].reasons)
            self.assertIn("adjusted", reasons)


class SyntheticStudyTests(unittest.TestCase):
    def test_an_init_study_marks_every_analysis_exploratory(self):
        from ada.research import init  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            assay().to_csv(root / "assay.csv", index=False)
            path = init(root / "assay.csv", study_path=root / "study.yml", interactive=False)
            built = build(path)
            self.assertFalse((built.out / "tables" / "results.csv").exists())
            self.assertTrue((built.out / "supplement" / "exploratory-analyses.csv").exists())
            self.assertTrue(all(found.supported for found in built.results.values()))


if __name__ == "__main__":
    unittest.main()
