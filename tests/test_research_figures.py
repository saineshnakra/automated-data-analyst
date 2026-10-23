import json
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from research_helpers import assay, assay_study, study_folder, weights
from scipy import stats

from ada.research import build, figures, interview, load, plots
from ada.research.processing import read
from ada.research.results import AnalysisResult

FORMATS = ["pdf", "svg", "png", "tiff", "eps"]


def paired_scores(people: int = 8, seed: int = 11) -> pd.DataFrame:
    """Participants scored before and after training; the last one has no score after."""
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(1, people + 1):
        base = rng.normal(50, 8)
        rows.append({"Participant": f"P{person}", "Session": "Before", "Score": round(base, 1)})
        after = round(base + 4 + rng.normal(0, 2), 1)
        rows.append({"Participant": f"P{person}", "Session": "After", "Score": after})
    return pd.DataFrame(rows).iloc[:-1]


def paired_study(**changes) -> dict:
    study = {
        "study": {"title": "Training"},
        "data": {"path": "scores.csv", "observation": "one session of one participant"},
        "design": {"type": "paired", "subject": "Participant"},
        "variables": {
            "Participant": {"role": "identifier", "type": "text"},
            "Session": {"role": "group", "type": "categorical", "levels": ["Before", "After"]},
            "Score": {"role": "outcome", "type": "numeric", "unit": "points"},
        },
        "analyses": [{"id": "training", "outcome": "Score", "compare": "Session", "planned": True}],
        "output": {"figures": ["svg"]},
    }
    study.update(changes)
    return study


def weights_study(**changes) -> dict:
    study = {
        "study": {"title": "Diet"},
        "data": {"path": "weights.csv", "observation": "one mouse on one day"},
        "design": {"type": "repeated", "subject": "Mouse", "time": "Day"},
        "variables": {
            "Mouse": {"role": "identifier", "type": "text"},
            "Diet": {"role": "group", "type": "categorical", "levels": ["Chow", "High fat"]},
            "Day": {"role": "time", "type": "numeric"},
            "Weight (g)": {"role": "outcome", "type": "numeric", "unit": "g"},
        },
        "analyses": [{"id": "gain", "outcome": "Weight (g)", "change": "Day", "by": "Diet", "planned": True}],
        "output": {"figures": ["svg"]},
    }
    study.update(changes)
    return study


def assay_with(**changes) -> dict:
    return assay_study(output={"figures": ["svg"]}, **changes)


def readings(rows: int = 40, seed: int = 7) -> pd.DataFrame:
    """A signal in thousandths falling with a dose in thousands: a negative slope on very different scales."""
    rng = np.random.default_rng(seed)
    dose = rng.uniform(1000, 5000, rows)
    return pd.DataFrame(
        {
            "Sample": [f"S{index:02d}" for index in range(rows)],
            "Dose": np.round(dose, 1),
            "Signal": np.round(0.02 - 2.5e-6 * dose + rng.normal(0, 1e-3, rows), 6),
        }
    )


def readings_study(**changes) -> dict:
    study = {
        "study": {"title": "Signal"},
        "data": {"path": "readings.csv", "observation": "one sample"},
        "design": {"type": "independent"},
        "variables": {
            "Sample": {"role": "identifier", "type": "text"},
            "Dose": {"role": "covariate", "type": "numeric", "unit": "ng/mL"},
            "Signal": {"role": "outcome", "type": "numeric"},
        },
        "analyses": [{"id": "dose", "outcome": "Signal", "relate": "Dose", "planned": True}],
        "output": {"figures": ["svg"]},
    }
    study.update(changes)
    return study


def least_squares(xs, ys, confidence: float = 0.95) -> dict:
    """Intercept and slope with t intervals, by NumPy least squares rather than statsmodels."""
    design = np.column_stack([np.ones(len(xs)), xs])
    coefficients, *_ = np.linalg.lstsq(design, ys, rcond=None)
    residuals = ys - design @ coefficients
    freedom = len(xs) - 2
    errors = np.sqrt(np.diag(residuals @ residuals / freedom * np.linalg.inv(design.T @ design)))
    half = stats.t.ppf(0.5 + confidence / 2, freedom) * errors
    return {
        name: (coefficients[index], coefficients[index] - half[index], coefficients[index] + half[index])
        for index, name in enumerate(("intercept", "slope"))
    }


def confounded(rows: int = 60, seed: int = 13) -> pd.DataFrame:
    """Older patients get higher doses and score higher; with age held, the score falls with dose."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(40, 80, rows)
    batch = np.where(np.arange(rows) % 3 == 0, "B", "A")
    dose = 0.5 * age + rng.normal(0, 3, rows)
    score = 10 - 1.0 * dose + 2.0 * age + 3.0 * (batch == "B") + rng.normal(0, 2, rows)
    return pd.DataFrame(
        {
            "Patient": [f"P{index:02d}" for index in range(rows)],
            "Age": np.round(age, 1),
            "Batch": batch,
            "Dose": np.round(dose, 2),
            "Score": np.round(score, 2),
        }
    )


def visits(people: int = 16, seed: int = 17) -> pd.DataFrame:
    """Patients seen five times each; the score falls with dose within a patient and rises with age."""
    rng = np.random.default_rng(seed)
    rows = []
    for person in range(people):
        age, sex, level = rng.uniform(40, 80), "F" if person % 2 else "M", rng.normal(0, 4)
        usual = 0.5 * age + rng.normal(0, 2)
        for _ in range(5):
            dose = usual + rng.normal(0, 2)
            score = 20 - 0.8 * dose + 0.6 * age + (2.0 if sex == "F" else 0.0) + level + rng.normal(0, 1)
            rows.append(
                {
                    "Patient": f"P{person:02d}",
                    "Age": round(age, 1),
                    "Sex": sex,
                    "Dose": round(dose, 2),
                    "Score": round(score, 2),
                }
            )
    return pd.DataFrame(rows)


def patients_study(design: dict, covariates=(), **changes) -> dict:
    study = {
        "study": {"title": "Dose"},
        "data": {"path": "patients.csv", "observation": "one visit of one patient"},
        "design": design,
        "variables": {
            "Patient": {"role": "identifier", "type": "text"},
            "Age": {"role": "covariate", "type": "numeric", "unit": "years"},
            "Dose": {"role": "covariate", "type": "numeric", "unit": "mg"},
            "Score": {"role": "outcome", "type": "numeric"},
        },
        "analyses": [
            {
                "id": "dose",
                "outcome": "Score",
                "relate": "Dose",
                "covariates": list(covariates),
                "planned": True,
            }
        ],
        "output": {"figures": ["svg"]},
    }
    study.update(changes)
    return study


def reported_slope(built, ident: str) -> pd.Series:
    """The slope row of an analysis as the results table on disk reports it."""
    table = pd.read_csv(built.out / "tables" / "results.csv")
    return table[(table["analysis"] == ident) & (table["measure"] == "slope")].iloc[0]


def fixed_effect_band(fit, rows: np.ndarray, names: list[str], confidence: float = 0.95) -> np.ndarray:
    """A mixed model's prediction and Wald interval at each row of a design matrix, from its fixed effects."""
    mean = rows @ fit.fe_params[names].to_numpy()
    covariance = fit.cov_params().loc[names, names].to_numpy()
    half = stats.norm.ppf(0.5 + confidence / 2) * np.sqrt(np.einsum("ij,jk,ik->i", rows, covariance, rows))
    return np.column_stack([mean, mean - half, mean + half])


def assert_ordered_intervals(test, table: pd.DataFrame) -> None:
    """Every row with an interval has low <= value <= high, whether the value is a fit or an estimate."""
    rows = table.dropna(subset=["ci_low", "ci_high"])
    test.assertGreater(len(rows), 0)
    values = rows["fit"].fillna(rows["estimate"])
    test.assertFalse(values.isna().any())
    test.assertTrue((rows["ci_low"] <= values).all(), rows[rows["ci_low"] > values])
    test.assertTrue((values <= rows["ci_high"]).all(), rows[values > rows["ci_high"]])


def source(built, ident: str) -> pd.DataFrame:
    return pd.read_csv(built.out / "figures" / f"{ident}.source.csv")


def asset(built, ident: str):
    return next(item for item in built.assets if item.id == ident)


def t_interval(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    return stats.t.interval(confidence, len(values) - 1, loc=values.mean(), scale=stats.sem(values))


def with_results(fill):
    """Run the build with ``fill(result)`` standing in for the statistics step."""
    return mock.patch("ada.research.analyses.run", fill)


def processed(path):
    study = load(path)
    return SimpleNamespace(
        study=study, processed=None, data=read(study).data, results={}, warnings=[], style=None
    )


class GroupsTests(unittest.TestCase):
    def test_every_format_is_written_with_source_data_and_caption(self):
        figure = {"id": "viability", "type": "groups", "analysis": "primary"}
        study = assay_study(figures=[figure], output={"figures": FORMATS})
        with study_folder(assay(), study) as path:
            built = build(path)
            folder = built.out / "figures"
            for kind in [*FORMATS, "source.csv"]:
                self.assertTrue((folder / f"viability.{kind}").is_file(), kind)
            data = built.data
            table = source(built, "viability")
            entry = asset(built, "viability")
            record = json.loads((built.out / "provenance.json").read_text())["assets"]["viability"]
        points = table[table["element"] == "point"]
        expected = data.dropna(subset=["Viability (%)"])
        self.assertEqual(len(points), len(expected))
        self.assertEqual(sorted(points["source_row"]), sorted(expected["source_row"]))
        self.assertIn("Well", table.columns)
        means = table[table["element"] == "mean"].set_index("Treatment")
        self.assertEqual(list(means.index), ["Control", "Low", "High"])
        for level, values in expected.groupby("Treatment", observed=True)["Viability (%)"]:
            low, high = t_interval(values)
            self.assertEqual(means.loc[level, "n"], len(values))
            self.assertAlmostEqual(means.loc[level, "mean"], values.mean(), places=9)
            self.assertAlmostEqual(means.loc[level, "ci_low"], low, places=9)
            self.assertAlmostEqual(means.loc[level, "ci_high"], high, places=9)
        self.assertEqual(entry.number, "Figure 1")
        self.assertIn("n = 11 (Control), 12 (Low) and 12 (High)", entry.caption)
        self.assertIn("95% confidence interval", entry.caption)
        self.assertIn("1 row with no value for Viability (%) is left out", entry.caption)
        self.assertIn("was planned before the data were seen", entry.caption)
        self.assertTrue(entry.caption.startswith("Figure 1. Viability by Treatment. Each point is one well"))
        self.assertIn("figures.viability", record["depends_on"])
        self.assertIn("analyses.primary", record["depends_on"])
        self.assertIn("from analysis primary", record["calculation"])
        self.assertEqual(record["rows"]["used"], len(expected))

    def test_the_analysis_means_are_drawn_when_the_statistics_step_gives_them(self):
        means = pd.DataFrame(
            {
                "group": ["Control", "Low", "High"],
                "time": [None] * 3,
                "n": [11, 12, 12],
                "mean": [90.0, 85.0, 78.0],
                "sd": [3.0] * 3,
                "se": [1.0] * 3,
                "ci_low": [88.0, 83.0, 76.0],
                "ci_high": [92.0, 87.0, 80.0],
            }
        )

        def fill(result):
            spec = result.study.analyses[0]
            result.results["primary"] = AnalysisResult(
                spec=spec, design="independent", test="Welch's ANOVA", group_means=means
            )

        with study_folder(assay(), assay_with()) as path, with_results(fill):
            built = build(path)
            table = source(built, "primary-groups")
            caption = asset(built, "primary-groups").caption
        drawn = table[table["element"] == "mean"].set_index("Treatment")
        self.assertEqual(list(drawn["mean"]), [90.0, 85.0, 78.0])
        self.assertEqual(list(drawn["ci_high"]), [92.0, 87.0, 80.0])
        self.assertIn("as estimated in analysis primary", caption)
        self.assertIn("Analysis primary (Welch's ANOVA) was planned", caption)

    def test_colours_follow_the_level_not_its_rank(self):
        frame = assay()
        frame = frame[frame["Treatment"] != "Low"]
        with study_folder(frame, assay_with()) as path:
            result = processed(path)
            figure = mock.MagicMock()
            axes = mock.MagicMock()
            figure.add_subplot.return_value = axes
            plots.draw_groups(figure, result, plots.FigureSpec(id="g", type="groups", analysis="primary"))
        edges = [call.kwargs["edgecolors"][0] for call in axes.scatter.call_args_list]
        self.assertEqual(edges, [figures.PALETTE[0], figures.PALETTE[2]])

    def test_withheld_columns_stay_out_of_the_source_data(self):
        study = assay_with(sharing={"withhold": ["Well"]})
        with study_folder(assay(), study) as path:
            table = source(build(path), "primary-groups")
        self.assertNotIn("Well", table.columns)
        self.assertIn("source_row", table.columns)

    def test_the_confidence_level_comes_from_the_decisions(self):
        study = assay_with(decisions={"confidence": 0.9})
        with study_folder(assay(), study) as path:
            built = build(path)
            table = source(built, "primary-groups")
            caption = asset(built, "primary-groups").caption
            values = built.data.loc[built.data["Treatment"] == "High", "Viability (%)"].dropna()
        self.assertIn("90% confidence interval", caption)
        high = table[(table["element"] == "mean") & (table["Treatment"] == "High")].iloc[0]
        self.assertAlmostEqual(high["ci_low"], t_interval(values, 0.9)[0], places=9)


class PairedTests(unittest.TestCase):
    def test_subjects_without_both_conditions_are_left_out_and_counted(self):
        frame = paired_scores()
        with study_folder(frame, paired_study(), data_name="scores.csv") as path:
            built = build(path)
            table = source(built, "training-paired")
            entry = asset(built, "training-paired")
        wide = frame.pivot(index="Participant", columns="Session", values="Score").dropna()
        self.assertEqual(entry.section, "main")
        self.assertIn("n = 7 subjects with both Before and After", entry.caption)
        self.assertIn("1 subject without a value in every condition is left out (P8)", entry.caption)
        self.assertEqual((table["element"] == "point").sum(), 14)
        differences = table[table["element"] == "difference"].set_index("Participant")["difference"]
        expected = (wide["After"] - wide["Before"]).sort_index()
        np.testing.assert_allclose(differences.sort_index().to_numpy(), expected.to_numpy())
        summary = table[table["element"] == "mean difference"].iloc[0]
        interval = stats.ttest_rel(wide["After"], wide["Before"]).confidence_interval(0.95)
        self.assertAlmostEqual(summary["ci_low"], interval.low, places=9)
        self.assertAlmostEqual(summary["ci_high"], interval.high, places=9)
        before = table[(table["element"] == "mean") & (table["Session"] == "Before")].iloc[0]
        self.assertAlmostEqual(before["mean"], wide["Before"].mean(), places=9)
        self.assertEqual(before["n"], 7)


class LongitudinalTests(unittest.TestCase):
    def test_lines_join_each_subjects_own_points_and_means_match(self):
        frame = weights()
        frame.loc[7, "Weight (g)"] = np.nan
        with study_folder(frame, weights_study(), data_name="weights.csv") as path:
            result = processed(path)
            import matplotlib.pyplot as plt

            figure = plt.figure()
            drawn = plots.draw_longitudinal(
                figure, result, plots.FigureSpec(id="w", type="longitudinal", analysis="gain")
            )
            segments = figure.axes[0].collections[0].get_segments()
            plt.close(figure)
        data = result.data.dropna(subset=["Weight (g)"])
        self.assertEqual(len(segments), data["Mouse"].nunique())
        for segment, (_, rows) in zip(segments, data.groupby("Mouse", sort=True), strict=True):
            rows = rows.sort_values("Day")
            np.testing.assert_allclose(segment[:, 0], rows["Day"].to_numpy(dtype=float))
            np.testing.assert_allclose(segment[:, 1], rows["Weight (g)"].to_numpy())
        means = drawn.source[drawn.source["element"] == "mean"]
        for (diet, day), values in data.groupby(["Diet", "Day"], observed=True)["Weight (g)"]:
            row = means[(means["Diet"] == diet) & (means["Day"].astype(float) == float(day))].iloc[0]
            self.assertAlmostEqual(row["mean"], values.mean(), places=9)
            self.assertAlmostEqual(row["ci_high"], t_interval(values)[1], places=9)
        text = " ".join(drawn.sentences)
        self.assertIn("Thin lines join the measurements of each subject (Mouse); 12 subjects", text)
        self.assertIn("1 row with no value for Weight (g) is left out", text)

    def test_a_built_longitudinal_figure_is_the_default_for_change(self):
        with study_folder(weights(), weights_study(), data_name="weights.csv") as path:
            built = build(path)
            entry = asset(built, "gain-longitudinal")
        self.assertEqual(entry.number, "Figure 1")
        self.assertIn("95% confidence interval at each day for each diet", entry.caption)


class ScatterTests(unittest.TestCase):
    def test_the_band_is_the_ols_interval_for_the_mean(self):
        analyses = [{"id": "passage", "outcome": "Viability (%)", "relate": "Passage", "planned": False}]
        with study_folder(assay(), assay_with(analyses=analyses)) as path:
            built = build(path)
            table = source(built, "passage-scatter")
            entry = asset(built, "passage-scatter")
            data = built.data.dropna(subset=["Viability (%)", "Passage"])
        self.assertEqual(entry.section, "supplement")
        self.assertIn("is exploratory", entry.caption)
        outcome, predictor = data["Viability (%)"].to_numpy(float), data["Passage"].to_numpy(float)
        fit = sm.OLS(outcome, sm.add_constant(predictor)).fit()
        line = table[table["element"] == "fit"]
        grid = line["Passage"].to_numpy()
        expected = fit.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
        np.testing.assert_allclose(line["fit"], expected["mean"], rtol=1e-9)
        np.testing.assert_allclose(line["ci_low"], expected["mean_ci_lower"], rtol=1e-9)
        slope = table[table["element"] == "slope"].iloc[0]
        regression = stats.linregress(data["Passage"], data["Viability (%)"])
        self.assertAlmostEqual(slope["estimate"], regression.slope, places=9)
        self.assertEqual((table["element"] == "point").sum(), len(data))

    def test_a_clustered_design_without_a_fitted_model_draws_no_band(self):
        frame = assay()
        frame["Plate"] = [f"Plate {index % 4}" for index in range(len(frame))]
        study = assay_with(
            design={"type": "clustered", "cluster": "Plate"},
            analyses=[{"id": "passage", "outcome": "Viability (%)", "relate": "Passage", "planned": True}],
        )
        study["variables"]["Plate"] = {"role": "identifier", "type": "categorical"}
        with study_folder(frame, study) as path, with_results(lambda result: None):
            built = build(path)
            caption = asset(built, "passage-scatter").caption
            table = source(built, "passage-scatter")
        self.assertIn(
            "come from 4 clusters (Plate), so the points are not independent and least squares gives no "
            "valid confidence band",
            caption,
        )
        self.assertNotIn("ci_low", table.columns)

    def test_a_clustered_design_draws_the_mixed_model_line(self):
        frame = assay()
        # Neighbouring wells share a plate, so the plates differ and the mixed model converges.
        frame["Plate"] = [f"Plate {index // 9}" for index in range(len(frame))]
        study = assay_with(
            design={"type": "clustered", "cluster": "Plate"},
            analyses=[{"id": "passage", "outcome": "Viability (%)", "relate": "Passage", "planned": True}],
        )
        study["variables"]["Plate"] = {"role": "identifier", "type": "categorical"}
        with study_folder(frame, study) as path:
            caption = asset(build(path), "passage-scatter").caption
        self.assertIn("fixed-effect fit of analysis passage", caption)

    def test_the_coefficient_rows_match_an_independent_fit(self):
        listed = [
            {"id": "linked", "type": "scatter", "analysis": "dose"},
            {"id": "plain", "type": "scatter", "x": "Dose", "y": "Signal"},
        ]
        frame = readings()
        with study_folder(frame, readings_study(figures=listed), "readings.csv") as path:
            built = build(path)
            tables = {ident: source(built, ident) for ident in ("linked", "plain")}
        xs, ys = frame["Dose"].to_numpy(float), frame["Signal"].to_numpy(float)
        expected = least_squares(xs, ys)
        fit = sm.OLS(ys, sm.add_constant(xs)).fit()
        self.assertLess(expected["slope"][0], 0)
        for ident, table in tables.items():
            for position, name in enumerate(("intercept", "slope")):
                with self.subTest(figure=ident, row=name):
                    row = table[table["element"] == name].iloc[0]
                    got = [row["estimate"], row["ci_low"], row["ci_high"]]
                    np.testing.assert_allclose(got, expected[name], rtol=1e-8)
                    np.testing.assert_allclose(
                        got, [fit.params[position], *fit.conf_int(alpha=0.05)[position]], rtol=1e-8
                    )
                    self.assertEqual(row["n"], len(frame))
            assert_ordered_intervals(self, table)

    def test_an_adjusted_regression_draws_its_prediction_at_the_reference_values(self):
        frame = confounded()
        study = patients_study({"type": "independent"}, covariates=["Age", "Batch"])
        study["data"]["observation"] = "one patient"
        study["variables"]["Batch"] = {"role": "covariate", "type": "categorical", "levels": ["A", "B"]}
        with study_folder(frame, study, "patients.csv") as path:
            built = build(path)
            table = source(built, "dose-scatter")
            caption = asset(built, "dose-scatter").caption
            reported = reported_slope(built, "dose")
        fit = smf.ols("Score ~ Dose + Age + C(Batch)", data=frame).fit()
        line = table[table["element"] == "fit"]
        intercept = table[table["element"] == "intercept"].iloc[0]
        slope = table[table["element"] == "slope"].iloc[0]
        at = pd.DataFrame({"Dose": [*line["Dose"], 0.0], "Age": frame["Age"].mean(), "Batch": "A"})
        expected = fit.get_prediction(at).summary_frame(alpha=0.05)
        drawn = np.vstack(
            [line[["fit", "ci_low", "ci_high"]], intercept[["estimate", "ci_low", "ci_high"]].to_numpy(float)]
        )
        np.testing.assert_allclose(drawn, expected[["mean", "mean_ci_lower", "mean_ci_upper"]], rtol=1e-8)
        np.testing.assert_allclose(line["Age"], frame["Age"].mean(), rtol=1e-10)
        self.assertEqual(set(line["Batch"]), {"A"})
        # The slope is the analysis's, which has the opposite sign to least squares on dose alone.
        np.testing.assert_allclose(
            slope[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            reported[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            rtol=1e-10,
        )
        self.assertLess(slope["estimate"], 0)
        self.assertGreater(stats.linregress(frame["Dose"], frame["Score"]).slope, 0)
        assert_ordered_intervals(self, table)
        self.assertIn(
            "least-squares fit of analysis dose (Linear regression (OLS) adjusted for Age, Batch) and its "
            "95% confidence interval for the mean, holding Age at its mean in the rows analysed "
            f"({frame['Age'].mean():.4g} years) and Batch at its reference level (A).",
            caption,
        )
        self.assertIn("The points are the observed values, not adjusted for Age, Batch", caption)
        self.assertNotIn("the line does not", caption)

    def test_a_mixed_model_with_covariates_draws_its_fixed_effect_prediction(self):
        frame = visits()
        study = patients_study({"type": "repeated", "subject": "Patient"}, covariates=["Age", "Sex"])
        study["variables"]["Sex"] = {"role": "covariate", "type": "categorical", "levels": ["M", "F"]}
        with study_folder(frame, study, "patients.csv") as path:
            built = build(path)
            table = source(built, "dose-scatter")
            caption = asset(built, "dose-scatter").caption
            reported = reported_slope(built, "dose")
        data = frame.assign(Sex=pd.Categorical(frame["Sex"], categories=["M", "F"]))
        fit = smf.mixedlm("Score ~ Dose + Age + C(Sex)", data=data, groups="Patient").fit(reml=True)
        names = ["Intercept", "C(Sex)[T.F]", "Dose", "Age"]
        line = table[table["element"] == "fit"]
        doses = line["Dose"].to_numpy()
        rows = np.column_stack(
            [np.ones(len(doses)), np.zeros(len(doses)), doses, np.full(len(doses), frame["Age"].mean())]
        )
        expected = fixed_effect_band(fit, rows, names)
        np.testing.assert_allclose(line[["fit", "ci_low", "ci_high"]], expected, rtol=1e-6)
        at = pd.DataFrame({"Dose": doses, "Age": frame["Age"].mean(), "Sex": "M"})
        np.testing.assert_allclose(line["fit"], fit.predict(at), rtol=1e-6)
        slope = table[table["element"] == "slope"].iloc[0]
        np.testing.assert_allclose(
            slope[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            reported[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            [slope["estimate"], *fit.conf_int(alpha=0.05).loc["Dose"]],
            slope[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            rtol=1e-6,
        )
        self.assertLess(slope["estimate"], 0)
        assert_ordered_intervals(self, table)
        self.assertIn("fixed-effect fit of analysis dose (Linear mixed model", caption)
        self.assertIn("holding Age at its mean in the rows analysed", caption)
        self.assertIn("and Sex at its reference level (M).", caption)
        self.assertIn("measured more than once, the line and band come from the mixed model", caption)
        self.assertNotIn("narrower than it should be", caption)
        self.assertNotIn("the line does not", caption)

    def test_a_mixed_model_without_covariates_draws_its_line_and_band(self):
        frame = visits()
        study = patients_study({"type": "repeated", "subject": "Patient"})
        with study_folder(frame, study, "patients.csv") as path:
            built = build(path)
            table = source(built, "dose-scatter")
            caption = asset(built, "dose-scatter").caption
            reported = reported_slope(built, "dose")
        fit = smf.mixedlm("Score ~ Dose", data=frame, groups="Patient").fit(reml=True)
        line = table[table["element"] == "fit"]
        doses = line["Dose"].to_numpy()
        rows = np.column_stack([np.ones(len(doses)), doses])
        expected = fixed_effect_band(fit, rows, ["Intercept", "Dose"])
        np.testing.assert_allclose(line[["fit", "ci_low", "ci_high"]], expected, rtol=1e-6)
        limits = fit.conf_int(alpha=0.05)
        for term, element in (("Intercept", "intercept"), ("Dose", "slope")):
            row = table[table["element"] == element].iloc[0]
            np.testing.assert_allclose(
                row[["estimate", "ci_low", "ci_high"]].to_numpy(float),
                [fit.fe_params[term], *limits.loc[term]],
                rtol=1e-6,
            )
        slope = table[table["element"] == "slope"].iloc[0]
        np.testing.assert_allclose(
            slope[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            reported[["estimate", "ci_low", "ci_high"]].to_numpy(float),
            rtol=1e-10,
        )
        assert_ordered_intervals(self, table)
        self.assertIn("fixed-effect fit of analysis dose (Linear mixed model", caption)
        self.assertNotIn("holding", caption)

    def test_a_repeated_design_without_a_linked_analysis_draws_no_band(self):
        frame = visits()
        listed = [{"id": "plain", "type": "scatter", "x": "Dose", "y": "Score"}]
        study = patients_study({"type": "repeated", "subject": "Patient"}, figures=listed)
        with study_folder(frame, study, "patients.csv") as path:
            built = build(path)
            table = source(built, "plain")
            caption = asset(built, "plain").caption
            result = processed(path)
            figure = mock.MagicMock()
            axes = mock.MagicMock()
            figure.add_subplot.return_value = axes
            plots.draw_scatter(figure, result, plots.FigureSpec(id="p", type="scatter", x="Dose", y="Score"))
        axes.fill_between.assert_not_called()
        axes.plot.assert_called_once()
        self.assertNotIn("ci_low", table.columns)
        slope = table[table["element"] == "slope"].iloc[0]
        pooled = stats.linregress(frame["Dose"], frame["Score"])
        self.assertAlmostEqual(slope["estimate"], pooled.slope, places=9)
        self.assertIn("drawn only to describe them: the 80 points come from 16 subjects (Patient)", caption)
        self.assertIn("least squares gives no valid confidence band", caption)
        self.assertIn("No analysis is linked to this figure", caption)

    def test_a_refused_analysis_is_not_said_to_account_for_the_repeated_measurements(self):
        frame = visits()
        frame = frame[frame["Patient"] == "P00"]
        analyses = [{"id": "dose", "outcome": "Score", "relate": "Dose", "test": "pearson", "planned": True}]
        study = patients_study({"type": "repeated", "subject": "Patient"}, analyses=analyses)
        with study_folder(frame, study, "patients.csv") as path:
            built = build(path)
            table = source(built, "dose-scatter")
            caption = asset(built, "dose-scatter").caption
        self.assertIn("ADA did not fit it", caption)
        self.assertNotIn("accounts for", caption)
        self.assertIn("the 5 points come from 1 subject (Patient) measured more than once", caption)
        self.assertNotIn("ci_low", table.columns)


class OtherTypesTests(unittest.TestCase):
    def test_missing_values_get_a_supplementary_figure(self):
        with study_folder(assay(), assay_with()) as path:
            built = build(path)
            entry = asset(built, "missing-values")
            table = source(built, "missing-values")
        self.assertEqual(entry.section, "supplement")
        counts = table[table["element"] == "variable"].set_index("variable")
        self.assertEqual(counts.loc["Viability (%)", "missing"], 1)
        self.assertEqual(counts.loc["Treatment", "missing"], 0)
        self.assertEqual(list(counts.index)[0], "Viability (%)")
        cells = table[table["element"] == "missing cell"]
        self.assertEqual(list(cells["source_row"]), [4])
        self.assertIn("Missing-value codes (-99) count as missing", entry.caption)

    def test_diagnostics_follow_an_analysis_with_residuals(self):
        def fill(result):
            data = result.data.dropna(subset=["Viability (%)"])
            dummies = pd.get_dummies(data["Treatment"], drop_first=True, dtype=float)
            model = sm.OLS(data["Viability (%)"], sm.add_constant(dummies)).fit()
            result.results["primary"] = AnalysisResult(
                spec=result.study.analyses[0],
                design="independent",
                test="One-way ANOVA",
                fitted=model.fittedvalues,
                residuals=model.resid,
            )

        with study_folder(assay(), assay_with()) as path, with_results(fill):
            built = build(path)
            entry = asset(built, "primary-diagnostics")
            table = source(built, "primary-diagnostics")
            data = built.data.dropna(subset=["Viability (%)"])
        self.assertEqual(entry.section, "supplement")
        self.assertIn("residuals of analysis primary (One-way ANOVA)", entry.caption)
        self.assertEqual(sorted(table["source_row"]), sorted(data["source_row"]))
        (quantiles, ordered), _ = stats.probplot(table["residual"].to_numpy(), dist="norm")
        np.testing.assert_allclose(np.sort(table["normal_quantile"]), quantiles)
        np.testing.assert_allclose(np.sort(table["residual"]), ordered, rtol=1e-9)

    def test_a_panel_combines_its_figures_with_lettered_panels(self):
        listed = [
            {"id": "viability", "type": "groups", "analysis": "primary"},
            {"id": "passage", "type": "scatter", "x": "Passage", "y": "Viability (%)"},
            {"id": "overview", "type": "panel", "panels": ["viability", "passage"], "title": "Overview"},
        ]
        study = assay_study(figures=listed, output={"figures": ["svg", "png"]})
        with study_folder(assay(), study) as path:
            built = build(path)
            ids = [item.id for item in built.assets if item.kind == "figure"]
            table = source(built, "overview")
            entry = asset(built, "overview")
            record = json.loads((built.out / "provenance.json").read_text())["assets"]["overview"]
            from PIL import Image

            with Image.open(built.out / "figures" / "overview.png") as image:
                width = image.size[0]
            svg = (built.out / "figures" / "overview.svg").read_text()
        self.assertEqual(ids, ["overview"])
        self.assertEqual(sorted(set(table["panel"])), ["A", "B"])
        self.assertEqual(sorted(set(table["figure"])), ["passage", "viability"])
        self.assertIn("(A) Viability by Treatment.", entry.caption)
        self.assertIn("(B) Viability against passage.", entry.caption)
        self.assertTrue(entry.caption.startswith("Figure 1. Overview. (A)"))
        for part in ("figures.overview", "figures.viability", "figures.passage", "analyses.primary"):
            self.assertIn(part, record["depends_on"])
        self.assertAlmostEqual(width, 170 / 25.4 * 300, delta=1)
        self.assertIn(">A<", svg)
        self.assertIn(">B<", svg)

    def test_a_figure_that_cannot_be_drawn_is_reported_not_fatal(self):
        listed = [{"id": "pairs", "type": "paired", "analysis": "primary"}]
        with study_folder(assay(), assay_with(figures=listed)) as path:
            built = build(path)
        self.assertFalse(any(item.kind == "figure" for item in built.assets))
        notes = [note for note in built.warnings if "pairs was not drawn" in note]
        self.assertTrue(notes and "design.subject" in notes[0])


class DefaultTests(unittest.TestCase):
    def test_defaults_are_chosen_by_design_and_planning(self):
        cases = [
            (assay(), assay_with(), "assay.csv", {"primary-groups": "main", "missing-values": "supplement"}),
            (paired_scores(), paired_study(), "scores.csv", {"training-paired": "main"}),
            (weights(), weights_study(), "weights.csv", {"gain-longitudinal": "main"}),
        ]
        clustered = assay()
        clustered["Plate"] = [f"Plate {index % 3}" for index in range(len(clustered))]
        study = assay_with(
            design={"type": "clustered", "cluster": "Plate"},
            analyses=[
                {"id": "primary", "outcome": "Viability (%)", "compare": "Treatment", "planned": False},
                {"id": "passage", "outcome": "Viability (%)", "relate": "Passage", "planned": True},
            ],
        )
        study["variables"]["Plate"] = {"role": "identifier", "type": "categorical"}
        sections = {"primary-groups": "supplement", "passage-scatter": "main", "missing-values": "supplement"}
        cases.append((clustered, study, "assay.csv", sections))
        between = weights_study(
            analyses=[{"id": "diet", "outcome": "Weight (g)", "compare": "Diet", "planned": False}]
        )
        cases.append((weights(), between, "weights.csv", {"diet-groups": "supplement"}))
        for frame, study, name, expected in cases:
            with self.subTest(study=study["study"]["title"], design=study["design"]["type"]):
                with study_folder(frame, study, data_name=name) as path:
                    plans = plots.default_figures(processed(path))
                self.assertEqual({plan.spec.id: plan.section for plan in plans}, expected)

    def test_listed_figures_replace_the_defaults(self):
        listed = [{"id": "viability", "type": "groups", "x": "Treatment", "y": "Viability (%)"}]
        with study_folder(assay(), assay_with(figures=listed)) as path:
            built = build(path)
        self.assertEqual([item.id for item in built.assets if item.kind == "figure"], ["viability"])
        self.assertIn("No analysis is linked to this figure", asset(built, "viability").caption)


class OutputTests(unittest.TestCase):
    def test_a_rebuild_gives_identical_files(self):
        study = assay_study(output={"figures": FORMATS})
        with study_folder(assay(), study) as path:
            built = build(path)
            folder = built.out / "figures"
            first = {item.name: item.read_bytes() for item in sorted(folder.iterdir())}
            build(path)
            second = {item.name: item.read_bytes() for item in sorted(folder.iterdir())}
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            self.assertEqual(first[name], second[name], name)
        self.assertIn("primary-groups.svg", first)
        self.assertIn("primary-groups.pdf", first)

    def test_tiff_is_rgb_lzw_at_the_journal_resolution(self):
        from PIL import Image

        study = assay_study(output={"journal": "plos", "figures": ["tiff", "eps"]})
        transparency = self.assertNoLogs("matplotlib.backends.backend_ps", level="WARNING")
        with study_folder(assay(), study) as path, transparency:
            built = build(path)
            with Image.open(built.out / "figures" / "primary-groups.tiff") as image:
                mode, info, size = image.mode, dict(image.info), image.size
            eps = (built.out / "figures" / "primary-groups.eps").read_text(errors="replace")
        self.assertEqual(mode, "RGB")
        self.assertEqual(info["compression"], "tiff_lzw")
        self.assertEqual(tuple(round(value) for value in info["dpi"]), (300, 300))
        self.assertEqual(size[0], round(132 / 25.4 * 300))
        self.assertIn("%%CreationDate: Thu Jan 01 00:00:00 1970", eps)

    def test_the_journal_presets_resolve_with_their_sizes(self):
        expected = {
            "plos": (132.0, 190.5, 222.3, 300),
            "nature": (89.0, 183.0, 170.0, 450),
            "cell": (85.0, 174.0, 225.0, 500),
            "science": (90.0, 184.0, 199.0, 600),
            "elife": (100.0, 200.0, 200.0, 300),
        }
        self.assertEqual(set(interview.JOURNALS), set(figures.JOURNALS))
        for name, (single, double, height, dpi) in expected.items():
            with self.subTest(journal=name):
                with study_folder(assay(), assay_study(output={"journal": name})) as path:
                    style = figures.resolve_style(load(path))
                self.assertFalse([note for note in style.notes if "Unknown journal" in note])
                self.assertEqual(style.width_mm, single)
                self.assertEqual(style.dpi, dpi)
                width, high = figures.figure_size(style, "double", 0.5)
                self.assertAlmostEqual(width * 25.4, double)
                self.assertAlmostEqual(high * 25.4, double * 0.5)
                self.assertAlmostEqual(figures.figure_size(style, "single", 5)[1] * 25.4, height)
                self.assertTrue(style.journal.guide.startswith("https://"))
                self.assertGreaterEqual(figures.small_font_size(style), style.journal.min_font_size)
        with study_folder(assay(), assay_study(output={"journal": "nature"})) as path:
            style = figures.resolve_style(load(path))
        self.assertEqual(figures.panel_label(style, 1), ("b", 8.0))
        self.assertEqual(figures.small_font_size(style), 6.0)

    def test_spread_is_deterministic_and_stays_inside_its_strip(self):
        values = np.random.default_rng(1).normal(size=40)
        first, second = plots.spread(values, 0.3), plots.spread(values, 0.3)
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(np.abs(first).max(), 0.3)
        self.assertGreater(np.ptp(first), 0.2)
        np.testing.assert_array_equal(plots.spread([1.0], 0.3), [0.0])
        tied = plots.spread([2.0, 2.0, 2.0, 2.0], 0.3)
        self.assertEqual(len(set(tied)), 4)


if __name__ == "__main__":
    unittest.main()
