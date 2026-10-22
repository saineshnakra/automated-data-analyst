import unittest
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep

from ada.compare import (
    MAX_SCAN_COLUMNS,
    SCAN_ROWS,
    benjamini_hochberg,
    cap_flag,
    centre_word,
    combined_groupings,
    compare_groups,
    compare_rates,
    comparison_calculation,
    comparison_table,
    construction,
    date_part,
    describe_comparison,
    describe_rates,
    entity_column,
    find_derivation,
    find_product,
    find_standardised,
    fit_joint_model,
    grouping_rules,
    holm_adjust,
    is_angle,
    lead_sentence,
    most_influential,
    nested_within,
    ordered_levels,
    outcome_event,
    pairwise_correlations,
    rate_calculation,
    same_stem,
    scan_differences,
    scan_relationships,
    summary_families,
    variance_explained,
    within_group_correlation,
    without_extremes,
)
from ada.schema import ColumnRoles


def two_groups(shift: float, size: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Arm": ["Treated"] * size + ["Control"] * size,
        "Score": np.r_[rng.normal(10 + shift, 2, size), rng.normal(10, 2, size)],
    })


class TwoGroupTests(unittest.TestCase):
    def test_a_clear_difference_reports_the_interval_and_effect_size(self):
        comparison = compare_groups(two_groups(3.0), "Score", "Arm")

        assert comparison is not None
        self.assertEqual(comparison.verdict, "clear")
        self.assertEqual(comparison.higher.label, "Treated")
        self.assertGreater(comparison.low, 0)
        self.assertAlmostEqual(comparison.hedges_g, 1.5, delta=0.5)
        statement = describe_comparison(comparison)
        self.assertIn("Score is higher in Treated than in Control: mean", statement)
        self.assertIn("95% CI", statement)
        self.assertIn("Hedges' g", statement)
        self.assertIn("Mann-Whitney agrees", statement)

    def test_the_tests_match_scipy(self):
        frame = two_groups(0.8, size=25, seed=3)
        comparison = compare_groups(frame, "Score", "Arm")
        treated = frame.loc[frame["Arm"] == "Treated", "Score"]
        control = frame.loc[frame["Arm"] == "Control", "Score"]

        assert comparison is not None
        welch = stats.ttest_ind(treated, control, equal_var=False)
        ranks = stats.mannwhitneyu(treated, control, use_continuity=False, method="asymptotic")
        self.assertAlmostEqual(comparison.p_value, welch.pvalue, places=10)
        self.assertAlmostEqual(comparison.check_p_value, ranks.pvalue, places=10)
        self.assertAlmostEqual(comparison.superiority, ranks.statistic / (25 * 25), places=10)

    def test_a_gap_within_the_noise_is_not_a_ranking(self):
        comparison = compare_groups(two_groups(0.1), "Score", "Arm")

        assert comparison is not None
        self.assertEqual(comparison.verdict, "none")
        statement = describe_comparison(comparison)
        self.assertIn("shows no clear difference between", statement)
        self.assertIn("includes zero", statement)
        self.assertNotIn("highest", statement)
        self.assertNotIn("is higher in", statement)

    def test_a_trivial_gap_in_a_huge_sample_is_called_negligible(self):
        comparison = compare_groups(two_groups(0.15, size=40_000), "Score", "Arm")

        assert comparison is not None
        self.assertLess(comparison.p_value, 0.05)
        self.assertEqual(comparison.verdict, "negligible")
        self.assertIn("only by a negligible amount", describe_comparison(comparison))

    def test_a_skewed_reading_is_compared_by_medians_and_ranks(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({
            "Result": ["positive"] * 200 + ["negative"] * 200,
            "Marker": np.r_[rng.lognormal(0, 1.5, 200) * 4, rng.lognormal(0, 1.5, 200)],
        })

        comparison = compare_groups(frame, "Marker", "Result")

        assert comparison is not None
        self.assertTrue(comparison.uses_ranks)
        self.assertEqual(comparison.test, "Mann-Whitney")
        self.assertEqual(comparison.difference, comparison.higher.median - comparison.lower.median)
        statement = describe_comparison(comparison)
        self.assertIn("median", statement)
        self.assertIn("of pairs", statement)
        self.assertIn("strongly skewed", statement)
        self.assertTrue(comparison_calculation(comparison).startswith("median Marker per Result"))

    def test_a_capped_reading_says_where_the_cap_is(self):
        rng = np.random.default_rng(2)
        times = np.r_[rng.uniform(0, 3_600, 90), [3_600.0] * 10]
        frame = pd.DataFrame({"Phase": ["A", "B"] * 50, "Time": times})

        comparison = compare_groups(frame, "Time", "Phase")

        assert comparison is not None
        self.assertTrue(comparison.uses_ranks)
        self.assertIn("values sit at its maximum of 3,600", comparison.shape)

    def test_a_few_values_cannot_make_a_reading_skewed(self):
        frame = pd.DataFrame({"Site": ["N"] * 6 + ["S"] * 6, "Temp": [14.0] * 5 + [12.0] + [12.0] * 6})

        comparison = compare_groups(frame, "Temp", "Site")

        assert comparison is not None
        self.assertFalse(comparison.uses_ranks)

    def test_numeric_codes_are_named_with_their_column(self):
        frame = two_groups(3.0)
        frame["Arm"] = frame["Arm"].map({"Treated": 2, "Control": 1})

        comparison = compare_groups(frame, "Score", "Arm")

        assert comparison is not None
        self.assertEqual(comparison.higher.label, "Arm 2")

    def test_a_group_with_one_value_is_left_out_and_named(self):
        frame = pd.concat([two_groups(3.0), pd.DataFrame({"Arm": ["Pilot"], "Score": [50.0]})])

        comparison = compare_groups(frame, "Score", "Arm")

        assert comparison is not None
        self.assertEqual(len(comparison.groups), 2)
        self.assertEqual(comparison.dropped, ("Pilot",))
        self.assertIn("Left out with fewer than 2 values: Pilot", describe_comparison(comparison))

    def test_nothing_to_compare_returns_none(self):
        frame = pd.DataFrame({"Arm": ["A", "B", "C"], "Score": [1.0, 2.0, 3.0]})

        self.assertIsNone(compare_groups(frame, "Score", "Arm"))
        self.assertIsNone(compare_groups(two_groups(1.0), "Score", "Score"))


class ManyGroupTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.frame = pd.DataFrame({
            "Depth": np.repeat(["0-10", "10-30", "30-60", "60-90"], 12),
            "Ca": np.repeat([12.0, 9.0, 6.0, 5.0], 12) + rng.normal(0, 1, 48),
        })

    def test_welch_anova_names_the_pair_that_differs_most(self):
        comparison = compare_groups(self.frame, "Ca", "Depth")

        assert comparison is not None
        self.assertEqual(comparison.test, "Welch ANOVA")
        self.assertEqual((comparison.higher.label, comparison.lower.label), ("0-10", "60-90"))
        self.assertGreater(comparison.explained, 0.8)
        statement = describe_comparison(comparison)
        self.assertIn("Ca differs across Depth: 0-10 has the highest mean", statement)
        self.assertIn("η² =", statement)

    def test_eta_squared_matches_a_one_way_anova(self):
        comparison = compare_groups(self.frame, "Ca", "Depth")
        values = self.frame["Ca"]
        means = values.groupby(self.frame["Depth"]).transform("mean")

        assert comparison is not None
        expected = ((means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum()
        self.assertAlmostEqual(comparison.explained, expected, places=10)

    def test_the_table_lists_each_group_highest_first(self):
        comparison = compare_groups(self.frame, "Ca", "Depth")

        assert comparison is not None
        table = comparison_table(comparison)
        self.assertEqual(list(table.columns), ["Depth", "Mean Ca", "n", "Median Ca", "SD"])
        self.assertEqual(table.iloc[0]["Depth"], "0-10")
        self.assertEqual(int(table["n"].sum()), 48)

    def test_tukeys_hsd_names_the_pairs_it_tells_apart(self):
        comparison = compare_groups(self.frame, "Ca", "Depth")

        assert comparison is not None
        samples = [part["Ca"].to_numpy() for _, part in self.frame.groupby("Depth")]
        expected = stats.tukey_hsd(*samples).pvalue
        self.assertEqual(comparison.pairs_tried, 6)
        self.assertEqual(len(comparison.pairs_apart), int((expected[np.triu_indices(4, 1)] < 0.05).sum()))
        self.assertIn("Tukey's HSD tells", describe_comparison(comparison))
        self.assertIn("scipy.stats.studentized_range", comparison_calculation(comparison))

    def test_tukeys_hsd_tells_apart_the_same_pairs_as_scipy(self):
        rng = np.random.default_rng(31)
        sizes = [12, 30, 18, 25, 40, 9, 22, 15]
        frame = pd.DataFrame({
            "Dose": np.repeat([f"D{index}" for index in range(len(sizes))], sizes),
            "Response": np.concatenate([rng.normal(step * 0.35, 1, size) for step, size in enumerate(sizes)]),
        })

        comparison = compare_groups(frame, "Response", "Dose")

        assert comparison is not None
        labels = {group.value: group.label for group in comparison.groups}
        order = sorted(labels)
        samples = [frame.loc[frame["Dose"] == value, "Response"].to_numpy() for value in order]
        expected = stats.tukey_hsd(*samples).pvalue
        apart = {
            frozenset((labels[order[first]], labels[order[second]]))
            for first in range(len(order))
            for second in range(first + 1, len(order))
            if expected[first, second] < 0.05
        }
        self.assertEqual({frozenset(pair[:2]) for pair in comparison.pairs_apart}, apart)

    def test_no_clear_difference_gets_no_pairs(self):
        rng = np.random.default_rng(45)
        frame = pd.DataFrame({"Team": np.repeat(list("ABCD"), 30), "Hours": rng.normal(14, 3, 120)})

        comparison = compare_groups(frame, "Hours", "Team")

        assert comparison is not None
        self.assertEqual(comparison.pairs_tried, 0)
        self.assertNotIn("Tukey", describe_comparison(comparison))

    def test_no_clear_difference_across_many_groups_gives_the_range(self):
        rng = np.random.default_rng(5)
        frame = pd.DataFrame({"Team": np.repeat(list("ABCD"), 30), "Hours": rng.normal(14, 3, 120)})

        comparison = compare_groups(frame, "Hours", "Team")

        assert comparison is not None
        self.assertEqual(comparison.verdict, "none")
        self.assertIn("shows no clear difference across the 4 Team groups", describe_comparison(comparison))


class EntityTests(unittest.TestCase):
    def test_repeated_rows_count_once_per_subject(self):
        rng = np.random.default_rng(6)
        subjects = np.repeat(np.arange(20), 5)
        frame = pd.DataFrame({
            "Bird": subjects.astype(str),
            "Phenotype": np.where(subjects < 10, "bold", "shy"),
            "Latency": rng.normal(100, 10, 100) + np.where(subjects < 10, 30, 0),
        })

        comparison = compare_groups(frame, "Latency", "Phenotype", entity="Bird")

        assert comparison is not None
        self.assertEqual([group.n for group in comparison.groups], [10, 10])
        self.assertFalse(comparison.shared_entities)
        self.assertIn("Each Bird counts once per group", describe_comparison(comparison))
        self.assertIn("one value per Bird", comparison_calculation(comparison))

    def test_a_skewed_reading_takes_each_subjects_median_beside_a_plain_one(self):
        rng = np.random.default_rng(44)
        subjects = np.repeat(np.arange(20), 5)
        frame = pd.DataFrame({
            "Bird": subjects.astype(str),
            "Phenotype": np.where(subjects < 10, "bold", "shy"),
            "Latency": rng.lognormal(3, 1.5, 100),
            "Mass": rng.normal(30, 2, 100),
        })

        scan = scan_differences(frame, ["Latency", "Mass"], ["Phenotype"], entity="Bird")

        assert scan is not None
        latency = compare_groups(frame, "Latency", "Phenotype", entity="Bird")
        medians = frame.groupby(["Bird", "Phenotype"])["Latency"].median()
        bold = medians.xs("bold", level="Phenotype")
        self.assertEqual(latency.entity_summary, "median")
        self.assertAlmostEqual(next(g for g in latency.groups if g.label == "bold").mean, bold.mean())

    def test_a_subject_in_both_groups_is_flagged_as_not_independent(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame({
            "Subject": np.repeat(np.arange(15), 4).astype(str),
            "Object": ["familiar", "novel"] * 30,
            "Latency": rng.normal(50, 5, 60),
        })

        comparison = compare_groups(frame, "Latency", "Object", entity="Subject")

        assert comparison is not None
        self.assertTrue(comparison.shared_entities)
        self.assertIn("not independent", describe_comparison(comparison))

    def test_the_entity_column_is_one_that_repeats(self):
        frame = pd.DataFrame({"Patient ID": ["a", "a", "b", "c"], "Order ID": [1, 2, 3, 4], "x": [1.0] * 4})
        roles = ColumnRoles(date=None, measure="x", dimension=None, identifier="Order ID",
                            numeric=("x",), dimensions=())

        self.assertEqual(entity_column(frame, roles), "Patient ID")


class DifferenceScanTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        size = 80
        self.frame = pd.DataFrame({
            "Result": ["positive", "negative"] * size,
            "Sex": rng.choice(["F", "M"], 2 * size),
            "Marker": rng.normal(0, 1, 2 * size) + np.tile([2.0, 0.0], size),
            "Noise": rng.normal(0, 1, 2 * size),
            "Sugar": rng.normal(100, 20, 2 * size),
        })

    def test_the_largest_clear_difference_comes_first(self):
        scan = scan_differences(self.frame, ["Marker", "Noise", "Sugar"], ["Result", "Sex"])

        assert scan is not None
        self.assertEqual(scan.tested, 6)
        self.assertGreaterEqual(scan.clear, 1)
        self.assertEqual((scan.named[0].measure, scan.named[0].grouping), ("Marker", "Result"))
        self.assertEqual(scan.named[0].tests, 6)
        self.assertIn("Holm-adjusted for 6 tests", comparison_calculation(scan.named[0]))

    def test_pairs_reported_elsewhere_are_skipped(self):
        scan = scan_differences(self.frame, ["Marker", "Noise"], ["Result"], skip=(("Marker", "Result"),))

        assert scan is not None
        self.assertEqual(scan.tested, 1)

    def test_a_null_scan_still_names_its_largest_effect(self):
        scan = scan_differences(self.frame, ["Noise", "Sugar"], ["Sex"])

        assert scan is not None
        self.assertEqual(scan.clear, 0)
        self.assertEqual(scan.named, ())
        self.assertIsNotNone(scan.largest)

    def test_a_flag_marking_the_cap_is_not_compared_with_its_reading(self):
        rng = np.random.default_rng(9)
        time = np.r_[rng.uniform(0, 3_000, 40), [3_600.0] * 8]
        frame = pd.DataFrame({"Time": time, "Status": np.where(time == 3_600, 0, 1)})

        scan = scan_differences(frame, ["Time"], ["Status"])

        self.assertIsNone(scan)

    def test_a_code_with_one_value_per_group_is_not_a_finding(self):
        frame = self.frame.assign(Code=self.frame["Result"].map({"positive": 1, "negative": 0}))

        scan = scan_differences(frame, ["Code", "Noise"], ["Result"])

        assert scan is not None
        self.assertEqual(scan.recoded, (("Code", "Result"),))
        self.assertEqual(scan.tested, 1)

    def test_holm_adjustment(self):
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03, 0.5]))
        np.testing.assert_allclose(adjusted, [0.04, 0.09, 0.09, 0.5])

    def test_holm_matches_statsmodels(self):
        p = np.random.default_rng(43).random(40) ** 3
        np.testing.assert_allclose(holm_adjust(p), multipletests(p, method="holm")[1])


class GroupingStructureTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        contour = np.tile(np.repeat(["Top", "Slope", "Depression"], 4), 4)
        depth = np.repeat(["0-10", "10-30", "30-60", "60-90"], 12)
        self.frame = pd.DataFrame({
            "Contour": contour,
            "Depth": depth,
            "Cell": [f"{c[0]}{d[:2]}" for c, d in zip(contour, depth, strict=True)],
            "pH": pd.Series(depth).map({"0-10": 5.0, "10-30": 5.5, "30-60": 6.5, "60-90": 7.0}).to_numpy()
            + rng.normal(0, 0.2, 48),
        })

    def test_each_grouping_gets_its_share_of_variance(self):
        shares = dict(variance_explained(self.frame, "pH", ["Contour", "Depth"]))

        self.assertGreater(shares["Depth"], 0.8)
        self.assertLess(shares["Contour"], 0.1)

    def test_a_cell_code_is_recognised_as_a_combination(self):
        combined = combined_groupings(self.frame, ["Contour", "Depth", "Cell"])
        self.assertEqual(combined, {"Cell": ("Contour", "Depth")})


class CorrelationMatrixTests(unittest.TestCase):
    def test_pairwise_complete_correlations_match_pandas(self):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
        frame["b"] += frame["a"]
        frame.iloc[::7, 1] = np.nan
        frame.iloc[::5, 2] = np.nan

        correlation, counts = pairwise_correlations(frame)

        np.testing.assert_allclose(correlation, frame.corr().to_numpy(), atol=1e-10)
        self.assertEqual(counts[1, 2], int((frame["b"].notna() & frame["c"].notna()).sum()))

    def test_very_large_readings_keep_their_precision(self):
        rng = np.random.default_rng(12)
        x = 7e12 + rng.normal(0, 1e9, 500)
        frame = pd.DataFrame({"x": x, "y": x * 2 + rng.normal(0, 1e9, 500)})
        frame.iloc[0, 1] = np.nan

        correlation, _ = pairwise_correlations(frame)

        self.assertAlmostEqual(correlation[0, 1], frame.corr().iloc[0, 1], places=8)


class InfluenceTests(unittest.TestCase):
    def test_one_impossible_record_is_found(self):
        rng = np.random.default_rng(13)
        x, y = rng.normal(45, 4, 300), rng.normal(30, 3, 300)
        x[17], y[17] = 3_715.0, 3_117.0

        position, without = most_influential(x, y)

        self.assertEqual(position, 17)
        self.assertAlmostEqual(without, np.corrcoef(np.delete(x, 17), np.delete(y, 17))[0, 1], places=10)
        self.assertGreater(np.corrcoef(x, y)[0, 1], 0.9)

    def test_extreme_records_that_hide_a_relationship_are_set_aside(self):
        rng = np.random.default_rng(14)
        x = rng.normal(90, 5, 400)
        y = x / 3 + rng.normal(0, 0.5, 400)
        x[:3] = [-79.0, 900.0, 5_000.0]

        result = without_extremes(x, y)

        assert result is not None
        self.assertEqual(result[1], 3)
        self.assertGreater(result[0], 0.9)

    def test_nothing_extreme_leaves_nothing_to_report(self):
        rng = np.random.default_rng(15)
        self.assertIsNone(without_extremes(rng.normal(size=100), rng.normal(size=100)))


class WithinGroupTests(unittest.TestCase):
    def test_a_pooled_correlation_can_reverse_inside_the_groups(self):
        rng = np.random.default_rng(16)
        parts = []
        for offset in range(5):
            concentration = rng.uniform(0, 10, 40)
            parts.append(pd.DataFrame({
                "Dye": f"dye {offset}",
                "Concentration": concentration + offset * 10,
                "R": 100 + offset * 40 - 3 * concentration + rng.normal(0, 2, 40),
            }))
        frame = pd.concat(parts, ignore_index=True)

        within = within_group_correlation(frame, "Concentration", "R", "Dye")

        assert within is not None
        self.assertGreater(frame["Concentration"].corr(frame["R"]), 0.5)
        self.assertLess(within.within, -0.8)
        self.assertEqual(within.groups, 5)


class DerivationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        names = [f"Correct{index}" for index in range(1, 5)]
        items = pd.DataFrame(rng.uniform(0, 20, (40, 4)).round(2), columns=names)
        self.frame = items.assign(
            CorrectTotal=items.sum(axis=1),
            Mean=items.mean(axis=1),
            Gap=items["Correct1"] - items["Correct2"],
            Unrelated=items["Correct1"] + rng.normal(0, 3, 40),
        )
        self.items = list(items.columns)

    def test_a_total_is_the_sum_of_its_items(self):
        found = find_derivation(self.frame, "CorrectTotal", self.items)

        assert found is not None
        self.assertEqual(found.kind, "sum")
        self.assertEqual(set(found.parts), set(self.items))

    def test_a_mean_and_a_difference_are_named_as_such(self):
        self.assertEqual(find_derivation(self.frame, "Mean", self.items).kind, "mean")
        gap = find_derivation(self.frame, "Gap", self.items)
        assert gap is not None
        self.assertEqual(gap.kind, "difference")
        self.assertEqual(gap.describe(), "Gap = Correct1 − Correct2")

    def test_a_related_column_is_not_derived(self):
        self.assertIsNone(find_derivation(self.frame, "Unrelated", self.items))

    def test_a_few_distinct_rows_repeated_many_times_are_not_matched(self):
        rng = np.random.default_rng(16)
        visits = pd.DataFrame(rng.normal(size=(4, 4)), columns=["na", "fe", "mn", "temp"])
        frame = pd.concat([visits] * 15, ignore_index=True)

        self.assertIsNone(find_derivation(frame, "na", ["fe", "mn", "temp"]))

    def test_a_column_rounded_after_it_was_computed_is_still_derived(self):
        rng = np.random.default_rng(18)
        high = np.round(rng.uniform(40, 90, 300))
        low = high - np.round(rng.uniform(5, 25, 300))
        frame = pd.DataFrame({
            "Max": (high - 32) * 5 / 9,
            "Min": (low - 32) * 5 / 9,
            "Mean": (np.round((high + low) / 2) - 32) * 5 / 9,
        })

        found = find_derivation(frame, "Mean", ["Max", "Min"])

        assert found is not None
        self.assertEqual(found.kind, "mean")

    def test_a_censoring_flag_marks_the_cap(self):
        time = pd.Series([10.0, 200.0, 3_600.0, 50.0, 3_600.0, 900.0], name="Time")
        status = pd.Series([1, 1, 0, 1, 0, 1], name="Status")

        flag = cap_flag(status, time)

        assert flag is not None
        self.assertEqual((flag.value, flag.edge, flag.rows), (3_600.0, "maximum", 2))
        self.assertIsNone(cap_flag(pd.Series([1, 0, 1, 0, 1, 1], name="Other"), time))


class RelationshipScanTests(unittest.TestCase):
    def test_a_total_is_not_paired_with_its_own_items(self):
        rng = np.random.default_rng(19)
        items = pd.DataFrame(rng.uniform(0, 20, (60, 4)), columns=["A1", "A2", "A3", "A4"])
        frame = items.assign(Total=items.sum(axis=1), Outcome=items.sum(axis=1) * 0.5 + rng.normal(0, 4, 60))

        scan = scan_relationships(frame, list(frame.columns), "Outcome")

        assert scan is not None
        self.assertEqual([item.column for item in scan.derived], ["Total"])
        for pair in scan.others:
            self.assertFalse({pair.first, pair.second} & {"Total"} and {pair.first, pair.second} & set(items))

    def test_the_measure_is_ranked_first_by_its_correlates(self):
        rng = np.random.default_rng(20)
        frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
        frame["strength"] = frame["a"] * 2 + rng.normal(size=300)
        frame["c"] = frame["b"] + rng.normal(0, 0.3, size=300)

        scan = scan_relationships(frame, list(frame.columns), "strength")

        assert scan is not None
        self.assertEqual(scan.correlates[0].second, "a")
        self.assertEqual({scan.others[0].first, scan.others[0].second}, {"b", "c"})
        self.assertEqual((scan.measure_tests, scan.pair_tests), (3, 6))

    def test_one_record_cannot_make_the_strongest_pair(self):
        rng = np.random.default_rng(21)
        frame = pd.DataFrame(rng.normal(size=(300, 3)), columns=["hct", "mch", "plt"])
        frame.loc[5, ["hct", "mch"]] = [400.0, 350.0]
        frame["mcv"] = frame["plt"] * 0.6 + rng.normal(0, 0.8, 300)

        scan = scan_relationships(frame, list(frame.columns))

        assert scan is not None
        self.assertEqual({scan.others[0].first, scan.others[0].second}, {"plt", "mcv"})

    def test_same_stem_names_that_move_together_are_one_quantity(self):
        rng = np.random.default_rng(22)
        base = rng.normal(size=200)
        frame = pd.DataFrame({
            "risk_perception": base,
            "risk_perception_extended": base + rng.normal(0, 0.2, 200),
            "other": rng.normal(size=200),
        })

        scan = scan_relationships(frame, list(frame.columns))

        assert scan is not None
        self.assertEqual(
            [pair[:2] for pair in scan.shared_stems], [("risk_perception", "risk_perception_extended")]
        )

    def test_a_shared_unit_is_not_a_shared_stem(self):
        rng = np.random.default_rng(23)
        base = rng.normal(size=200)
        frame = pd.DataFrame({
            "Acidity cmol(+)/kg": base,
            "Aluminium cmol(+)/kg": base + rng.normal(0, 0.2, 200),
        })

        scan = scan_relationships(frame, list(frame.columns))

        assert scan is not None
        self.assertEqual(scan.shared_stems, ())
        self.assertEqual(len(scan.others), 1)

    def test_a_wide_table_is_scanned_on_its_most_variable_columns(self):
        rng = np.random.default_rng(24)
        width = MAX_SCAN_COLUMNS + 50
        frame = pd.DataFrame(rng.normal(size=(50, width)) * np.arange(1, width + 1))
        frame.columns = [f"c{index}" for index in range(frame.shape[1])]

        scan = scan_relationships(frame, list(frame.columns), "c0")

        assert scan is not None
        self.assertEqual(len(scan.columns), MAX_SCAN_COLUMNS)
        self.assertIn("c0", scan.columns)
        self.assertEqual(scan.available, MAX_SCAN_COLUMNS + 50)

    def test_too_few_complete_pairs_gives_no_scan(self):
        frame = pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0) ** 2})

        self.assertIsNone(scan_relationships(frame, ["x", "y"]))

    def test_one_huge_record_does_not_make_two_columns_one_quantity(self):
        rng = np.random.default_rng(25)
        first = rng.gamma(1.0, 10.0, 400)
        second = first * rng.uniform(0.2, 1.8, 400)
        first[0], second[0] = 500_000.0, 450_000.0
        frame = pd.DataFrame({"SampleA": first, "SampleB": second})

        scan = scan_relationships(frame, list(frame.columns), "SampleA")

        assert scan is not None
        self.assertEqual(scan.near_duplicates, ())
        self.assertEqual(scan.correlates[0].second, "SampleB")

    def test_a_copy_in_other_units_stays_one_quantity(self):
        rng = np.random.default_rng(26)
        celsius = rng.normal(20, 5, 300)
        frame = pd.DataFrame({"TempC": celsius, "TempF": celsius * 1.8 + 32, "Wind": rng.normal(size=300)})

        scan = scan_relationships(frame, list(frame.columns))

        assert scan is not None
        self.assertEqual([pair[:2] for pair in scan.near_duplicates], [("TempC", "TempF")])

    def test_shares_that_add_up_to_a_total_are_not_paired(self):
        rng = np.random.default_rng(27)
        shares = rng.dirichlet([4, 3, 2, 1], 500) * 100
        frame = pd.DataFrame(shares.round(1), columns=["Bact", "Firm", "Prot", "Other"])
        frame["Other"] = (100 - frame[["Bact", "Firm", "Prot"]].sum(axis=1)).round(1)
        broken = frame.index[:30]
        frame.loc[broken, "Other"] = 0.0
        frame.loc[broken, "Bact"] += 50.0
        frame["Weight"] = rng.normal(400, 50, 500)

        scan = scan_relationships(frame, list(frame.columns), "Weight")

        assert scan is not None
        block = next(item for item in scan.derived if item.kind == "complement")
        self.assertEqual(set(block.block), {"Bact", "Firm", "Prot", "Other"})
        self.assertEqual(block.breaks, 30)
        self.assertIn("On 30 rows, Bact, Firm, Prot and Other add up to", block.exceptions()[0])
        for pair in scan.others:
            self.assertFalse({pair.first, pair.second} <= set(block.block))

    def test_a_score_is_not_paired_with_items_read_as_codes(self):
        rng = np.random.default_rng(28)
        items = pd.DataFrame(rng.integers(0, 11, (200, 3)), columns=["Item1", "Item2", "Item3"])
        frame = items.assign(Score=items.mean(axis=1), Other=rng.normal(size=200))

        built = construction(frame, ["Score", "Other"], "Score", items=list(items.columns))

        assert built is not None
        self.assertEqual(built.derivation("Score").describe(), "Score = mean of Item1, Item2 and Item3")
        self.assertTrue(built.built_from("Score", "Item2"))
        self.assertEqual(built.relation("Score", "Item1"), "built")

    def test_a_compass_direction_is_left_out(self):
        rng = np.random.default_rng(29)
        frame = pd.DataFrame({
            "Wind Direction": rng.uniform(0, 360, 200).round(),
            "Temperature": rng.normal(20, 5, 200),
            "Humidity": rng.normal(60, 10, 200),
        })

        scan = scan_relationships(frame, list(frame.columns))

        assert scan is not None
        self.assertEqual(scan.angles, ("Wind Direction",))
        self.assertNotIn("Wind Direction", scan.columns)
        self.assertFalse(is_angle("Temperature", frame["Temperature"]))

    def test_two_coordinates_are_not_a_relationship(self):
        rng = np.random.default_rng(30)
        latitude = rng.uniform(21, 23, 200)
        frame = pd.DataFrame({
            "latitude": latitude,
            "longitude": -89 + latitude * 0.5 + rng.normal(0, 0.1, 200),
            "count": rng.poisson(5, 200).astype(float),
        })

        scan = scan_relationships(frame, list(frame.columns))

        assert scan is not None
        self.assertEqual(scan.coordinates, (("latitude", "longitude"),))
        self.assertNotIn({"latitude", "longitude"}, [{pair.first, pair.second} for pair in scan.others])

    def test_a_long_table_is_screened_on_a_sample(self):
        rng = np.random.default_rng(31)
        frame = pd.DataFrame({"x": rng.normal(size=SCAN_ROWS + 500)})
        frame["y"] = frame["x"] + rng.normal(size=len(frame))

        scan = scan_relationships(frame, ["x", "y"], "y")

        assert scan is not None
        self.assertTrue(scan.sampled)
        self.assertEqual(scan.correlates[0].pairs, SCAN_ROWS)


class TransformTests(unittest.TestCase):
    def test_a_log_of_a_reading_is_that_reading_on_another_scale(self):
        rng = np.random.default_rng(48)
        conductivity = rng.lognormal(5, 1, 400).round(1)
        frame = pd.DataFrame({
            "COND": conductivity,
            "lcond": np.log10(conductivity),
            "pH": rng.normal(7, 0.5, 400),
        })

        built = construction(frame, list(frame.columns), "COND")

        assert built is not None
        self.assertEqual(built.derivation("lcond").describe(), "lcond = log10(COND)")
        self.assertEqual(built.relation("COND", "lcond"), "copy")
        scan = scan_relationships(frame, list(frame.columns), "COND", built=built)
        self.assertNotIn("lcond", [pair.second for pair in scan.correlates])
        self.assertFalse(any("lcond" in (pair.first, pair.second) for pair in scan.others))

    def test_a_ratio_of_powers_is_found_and_named(self):
        rng = np.random.default_rng(49)
        weight = rng.normal(80, 15, 500).round(1)
        height = rng.normal(170, 10, 500).round(1)
        frame = pd.DataFrame({"weight_kg": weight, "height_cm": height, "age": rng.integers(20, 80, 500)})
        frame["bmi"] = (weight / (height / 100) ** 2).round(2)
        frame.loc[:9, "bmi"] = 40.0

        found = find_product(frame, "bmi", ["weight_kg", "height_cm", "age"])

        assert found is not None
        self.assertEqual(found.describe(), "bmi = weight_kg ÷ height_cm² × 10,000")
        self.assertIsNone(find_product(frame, "age", ["weight_kg", "height_cm"]))


class NeighbourRowTests(unittest.TestCase):
    def test_a_rate_is_the_row_to_row_change_of_its_reading(self):
        rng = np.random.default_rng(50)
        curves = []
        for _ in range(12):
            time = np.arange(0.01, 1.0, 0.01)
            angle = np.sin(2 * np.pi * time + rng.uniform(0, 1)) * rng.uniform(0.5, 1.5)
            velocity = np.r_[0.0, np.diff(angle) / 0.01]
            acceleration = np.r_[0.0, np.diff(velocity) / 0.01]
            curves.append(pd.DataFrame({"angle": angle, "velocity": velocity, "acceleration": acceleration}))
        frame = pd.concat(curves, ignore_index=True)

        built = construction(frame, list(frame.columns), "angle")

        assert built is not None
        rate = built.derivation("velocity").describe()
        self.assertEqual(rate, "velocity = 100 × the row-to-row change in angle")
        self.assertEqual(built.derivation("acceleration").parts, ("velocity",))
        self.assertEqual(built.relation("angle", "acceleration"), "built")

    def test_a_season_using_the_year_before_and_an_annual_mean_of_seasons_are_built(self):
        rng = np.random.default_rng(51)
        months = pd.DataFrame(
            rng.normal(27, 1, (80, 12)).round(2),
            columns=["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"],
        )
        months += np.linspace(0, 2, 80)[:, None]
        frame = months.assign(
            DJF=((months["DEC"].shift(1) + months["JAN"] + months["FEB"]) / 3).round(2),
            MAM=months[["MAR", "APR", "MAY"]].mean(axis=1).round(2),
            JJA=months[["JUN", "JUL", "AUG"]].mean(axis=1).round(2),
            SON=months[["SEP", "OCT", "NOV"]].mean(axis=1).round(2),
        )
        frame["ANN"] = frame[["DJF", "MAM", "JJA", "SON"]].mean(axis=1).round(2)

        built = construction(frame, list(frame.columns), "MAM")

        assert built is not None
        self.assertEqual(built.derivation("DJF").describe(), "DJF = mean of JAN, FEB and DEC (row before)")
        self.assertEqual(built.derivation("ANN").describe(), "ANN = mean of DJF, MAM, JJA and SON")

    def test_a_column_filled_in_from_its_neighbours_is_the_one_named_as_derived(self):
        rng = np.random.default_rng(52)
        frame = pd.DataFrame(rng.normal(0, 0.4, (300, 4)).round(2), columns=["40", "50", "70", "80"])
        frame.insert(2, "60", ((frame["50"] + frame["70"]) / 2).round(3))

        built = construction(frame, list(frame.columns))

        assert built is not None
        self.assertEqual([item.describe() for item in built.derived], ["60 = mean of 50 and 70"])


class SummaryFamilyTests(unittest.TestCase):
    def test_the_statistics_of_one_reading_are_one_family(self):
        rng = np.random.default_rng(45)
        low = rng.uniform(5, 50, 300)
        frame = pd.DataFrame({
            "pollutant_min": low,
            "pollutant_avg": low + rng.uniform(5, 30, 300),
            "max_speed": rng.uniform(0, 100, 300),
            "min_price": rng.uniform(0, 100, 300),
        })
        frame["pollutant_max"] = frame["pollutant_avg"] + rng.uniform(5, 60, 300)

        families = summary_families(frame, list(frame.columns))

        self.assertEqual(families, (("pollutant_min", "pollutant_avg", "pollutant_max"),))

    def test_a_summary_of_two_windows_of_one_test_is_one_family_but_numbered_items_are_not(self):
        rng = np.random.default_rng(46)
        frame = pd.DataFrame(
            rng.uniform(0, 10, (100, 6)),
            columns=["OBS1_bilirubin_max", "OBS2_bilirubin_min", "Q1_mean", "Q2_mean", "Q3_mean", "age"],
        )

        families = summary_families(frame, list(frame.columns))

        self.assertEqual(families, (("OBS1_bilirubin_max", "OBS2_bilirubin_min"),))

    def test_two_statistics_of_one_reading_are_not_paired(self):
        rng = np.random.default_rng(47)
        low = rng.uniform(5, 50, 300)
        frame = pd.DataFrame({"temp_min": low, "temp_max": low + rng.uniform(5, 40, 300)})
        frame["rain"] = frame["temp_max"] * 0.5 + rng.normal(0, 5, 300)

        scan = scan_relationships(frame, list(frame.columns), "temp_max")

        assert scan is not None
        self.assertEqual(scan.set_aside, (("temp_min", "summary"),))
        self.assertEqual(scan.summaries, (("temp_min", "temp_max"),))
        self.assertEqual([pair.second for pair in scan.correlates], ["rain"])


class SummaryDerivationTests(unittest.TestCase):
    def test_a_row_maximum_is_found_and_its_blank_zeros_counted(self):
        rng = np.random.default_rng(40)
        parts = pd.DataFrame(
            rng.integers(10, 200, (300, 4)).astype(float), columns=["pm10", "so2", "co", "o3"]
        )
        frame = parts.assign(index=parts.max(axis=1))
        frame.loc[:2, ["pm10", "so2", "co", "o3"]] = np.nan
        frame.loc[:2, "index"] = 0.0

        built = construction(frame, list(frame.columns))

        assert built is not None
        found = built.derivation("index")
        self.assertEqual(found.kind, "maximum")
        self.assertEqual(set(found.parts), {"pm10", "so2", "co", "o3"})
        self.assertEqual(found.blank_rows, 3)
        self.assertTrue(built.masked(frame).loc[:2, "index"].isna().all())

    def test_a_mean_of_whole_number_parts_truncated_after_averaging_is_still_a_mean(self):
        rng = np.random.default_rng(41)
        exact = rng.uniform(50, 200, (400, 6))
        frame = pd.DataFrame(np.floor(exact), columns=[f"mq{index}" for index in range(6)])
        frame["GasIndex"] = exact.mean(axis=1)

        found = find_derivation(frame, "GasIndex", [f"mq{index}" for index in range(6)])

        assert found is not None
        self.assertEqual(found.kind, "mean")

    def test_a_free_combination_must_hold_on_nearly_every_row(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
        frame["c"] = 0.37 * frame["a"] + 1.43 * frame["b"]
        off = frame.index[:25]
        frame.loc[off, "c"] += rng.normal(0, 0.05, 25)

        self.assertIsNone(find_derivation(frame, "c", ["a", "b"]))

    def test_a_score_standardised_within_each_group_names_its_source(self):
        rng = np.random.default_rng(43)
        drugs = np.repeat([f"drug{index}" for index in range(12)], 60)
        source = rng.normal(0, 1, len(drugs)) + np.repeat(rng.normal(0, 3, 12), 60)
        frame = pd.DataFrame({"Drug": drugs, "Arm": rng.choice(["A", "B", "C"], len(drugs)), "IC50": source})
        grouped = frame.groupby("Drug")["IC50"]
        frame["Z"] = (frame["IC50"] - grouped.transform("mean")) / grouped.transform("std")

        found = find_standardised(frame, "Z", ["IC50"], ["Arm", "Drug"])

        assert found is not None
        self.assertEqual((found.parts, found.within), (("IC50",), "Drug"))
        self.assertEqual(found.describe(), "Z is IC50 standardised within each Drug (mean 0, SD 1)")
        self.assertIsNone(find_standardised(frame, "IC50", [], ["Drug"]))


class GroupingRuleTests(unittest.TestCase):
    def test_bands_of_a_reading_are_a_rule(self):
        rng = np.random.default_rng(50)
        acres = rng.integers(1, 101, 600)
        frame = pd.DataFrame({
            "Acres": acres,
            "Size": np.select([acres <= 10, acres <= 50], ["Small", "Medium"], "Large"),
            "Weight": rng.normal(400, 50, 600),
        })

        rules = grouping_rules(frame, ["Acres", "Weight"], ["Size"])

        self.assertEqual(len(rules), 1)
        self.assertEqual(
            rules[0].describe(),
            "Size is set by Acres: Small up to 10, Medium 11 to 50 and Large from 51, on 100% of rows",
        )

    def test_cut_offs_that_move_with_another_grouping_are_a_rule(self):
        rng = np.random.default_rng(51)
        sex = rng.integers(0, 2, 800)
        hemoglobin = rng.normal(13, 2, 800).round(1)
        frame = pd.DataFrame({
            "Gender": sex,
            "Hemoglobin": hemoglobin,
            "Result": np.where(hemoglobin < np.where(sex == 0, 12.0, 13.5), 1, 0),
            "MCV": rng.normal(85, 5, 800),
        })

        rules = grouping_rules(frame, ["Hemoglobin", "MCV"], ["Result", "Gender"])

        self.assertEqual([rule.kind for rule in rules], ["cut-offs"])
        self.assertEqual(
            rules[0].describe(),
            "Result is set by Hemoglobin: Result 1 below 12.0 for Gender 0 and below 13.5 for Gender 1, "
            "on 100% of rows",
        )

    def test_a_label_naming_the_largest_column_is_a_rule(self):
        rng = np.random.default_rng(52)
        frame = pd.DataFrame(rng.uniform(0, 100, (300, 3)), columns=["pm10", "o3", "so2"])
        frame["critical"] = frame.idxmax(axis=1).str.upper()

        rules = grouping_rules(frame, ["pm10", "o3", "so2"], ["critical"])

        self.assertEqual(rules[0].kind, "largest")
        self.assertEqual(rules[0].readings, ("pm10", "o3", "so2"))

    def test_groups_that_differ_a_lot_but_overlap_are_not_a_rule(self):
        rng = np.random.default_rng(53)
        frame = pd.DataFrame({
            "Class": np.repeat([0, 1], 300),
            "Marker": np.r_[rng.normal(16, 4, 300), rng.normal(51, 12, 300)],
        })

        self.assertEqual(grouping_rules(frame, ["Marker"], ["Class"]), ())

    def test_phases_of_a_run_must_separate_a_reading_on_every_row(self):
        rng = np.random.default_rng(54)
        reading = np.r_[rng.uniform(0, 100, 500), rng.uniform(100, 5_000, 500)]
        reading[[600, 601]] = [50.0, 60.0]
        frame = pd.DataFrame({"Phase": np.repeat([0, 1], 500), "TVOC": reading.round()})

        self.assertEqual(grouping_rules(frame, ["TVOC"], ["Phase"]), ())
        shuffled = frame.sample(frac=1, random_state=0).reset_index(drop=True)
        self.assertEqual(len(grouping_rules(shuffled, ["TVOC"], ["Phase"])), 1)


class GroupingLayoutTests(unittest.TestCase):
    def test_a_month_column_is_a_part_of_the_date(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=400, freq="D"))

        self.assertEqual(date_part(dates.dt.month, dates), "month")
        self.assertEqual(date_part(dates.dt.year % 100, dates), "year")
        self.assertIsNone(date_part(pd.Series(np.arange(400) % 7 * 3), dates))

    def test_nesting_is_told_from_crossing(self):
        frame = pd.DataFrame({
            "Dye": np.repeat(["blue", "green", "red", "yellow"], 60),
            "Concentration": np.r_[np.tile([3, 6, 9], 40), np.tile([15, 25], 60)],
            "Condition": np.tile(["Open", "Closed"], 120),
        })

        self.assertTrue(nested_within(frame, "Concentration", "Dye"))
        self.assertFalse(nested_within(frame, "Dye", "Condition"))
        self.assertFalse(nested_within(frame, "Condition", "Dye"))

    def test_a_reading_fixed_within_each_group_is_an_attribute(self):
        rng = np.random.default_rng(55)
        station = np.repeat(["North", "South", "East"], 100)
        frame = pd.DataFrame({
            "Station": station,
            "Battery": np.repeat([11.9, 13.4, 15.1], 100) + rng.normal(0, 0.05, 300),
            "Rain": rng.gamma(1, 1, 300),
            "latitude": np.repeat([41.9, 41.8, 41.7], 100),
        })

        scan = scan_differences(frame, ["Battery", "Rain", "latitude"], ["Station"])

        assert scan is not None
        self.assertEqual(scan.attributes, (("Battery", "Station"),))
        self.assertEqual(scan.placed, ("latitude",))
        self.assertEqual(scan.tested, 1)

    def test_within_group_r_is_given_for_each_group(self):
        rng = np.random.default_rng(56)
        frame = pd.DataFrame({"Block": np.repeat(["1", "2"], 200), "x": rng.normal(size=400)})
        frame["y"] = frame["x"] + rng.normal(0, 0.3, 400) + np.repeat([0.0, 5.0], 200)

        within = within_group_correlation(frame, "x", "y", "Block")

        assert within is not None
        self.assertEqual([level for level, _, _ in within.per_group], ["1", "2"])
        self.assertEqual([rows for _, _, rows in within.per_group], [200, 200])


class SmallGroupTests(unittest.TestCase):
    def test_a_group_of_two_switches_the_test_to_pooled_variances(self):
        rng = np.random.default_rng(30)
        frame = pd.DataFrame({
            "Shoulder": np.r_[np.repeat(["14", "16", "18", "20"], 7), ["22", "22"]],
            "Charpy": np.r_[rng.normal(6, 1.5, 28), [9.25, 8.70]],
        })

        comparison = compare_groups(frame, "Charpy", "Shoulder")

        assert comparison is not None
        self.assertTrue(comparison.pooled)
        self.assertEqual(comparison.test, "one-way ANOVA")
        groups = [part["Charpy"] for _, part in frame.groupby("Shoulder")]
        self.assertAlmostEqual(comparison.p_value, stats.f_oneway(*groups).pvalue, places=10)
        self.assertIn("variances pooled", comparison_calculation(comparison))

    def test_a_tiny_group_is_never_named_as_the_extreme(self):
        rng = np.random.default_rng(31)
        frame = pd.DataFrame({
            "Cancer": np.r_[["PAAD"] * 27, ["LUAD"] * 30, ["CLL"] * 2],
            "Z": np.r_[rng.normal(0.7, 0.4, 27), rng.normal(0.0, 0.4, 30), [-1.1, -1.08]],
        })

        comparison = compare_groups(frame, "Z", "Cancer")

        assert comparison is not None
        self.assertEqual((comparison.higher.label, comparison.lower.label), ("PAAD", "LUAD"))
        self.assertEqual(comparison.groups[-1].label, "CLL")
        self.assertEqual(comparison.verdict, "clear")
        self.assertIn("Not ranked, with fewer than 5 values: CLL", describe_comparison(comparison))

    def test_a_difference_resting_on_two_values_is_not_clear(self):
        frame = pd.DataFrame({"Nudibranchs": [1] * 2 + [0] * 108, "Crabs": [9.0, 11.0] + [0.0, 1.0] * 54})

        comparison = compare_groups(frame, "Crabs", "Nudibranchs")

        assert comparison is not None
        self.assertLess(comparison.p_value, 0.05)
        self.assertEqual(comparison.verdict, "thin")
        statement = describe_comparison(comparison)
        self.assertIn("too few to call the difference clear", statement)
        self.assertNotIn("is higher in", statement)

    def test_a_check_that_disagrees_makes_the_difference_not_clear(self):
        comparison = compare_groups(two_groups(3.0), "Score", "Arm")
        assert comparison is not None

        disputed = replace(comparison, check_p_value=0.31)

        self.assertEqual(disputed.verdict, "disputed")
        statement = describe_comparison(disputed)
        self.assertIn("shows no clear difference", statement)
        self.assertIn("Mann-Whitney gives p = 0.31", statement)
        self.assertIn("disagree", statement)


class EffectSizeTests(unittest.TestCase):
    def test_a_small_effect_in_unequal_groups_is_not_negligible(self):
        frame = pd.DataFrame({
            "MSI": ["stable"] * 900 + ["unstable"] * 100,
            "Z": np.r_[np.linspace(-1.7, 1.7, 900) + 0.29, np.linspace(-1.7, 1.7, 100)],
        })

        comparison = compare_groups(frame, "Z", "MSI")

        assert comparison is not None
        self.assertLess(comparison.explained, 0.01)
        self.assertGreater(abs(comparison.hedges_g), 0.2)
        self.assertEqual(comparison.verdict, "clear")
        self.assertIn("a small difference", describe_comparison(comparison))

    def test_the_large_sample_caveat_needs_a_large_sample(self):
        values = np.tile(np.linspace(-1.7, 1.7, 500), 2) + np.repeat([0.15, 0.0], 500)
        frame = pd.DataFrame({"Arm": np.repeat(["A", "B"], 500), "Score": values})

        comparison = compare_groups(frame, "Score", "Arm")

        assert comparison is not None
        self.assertEqual(comparison.verdict, "negligible")
        self.assertNotIn("With this many records", describe_comparison(comparison))


class ZeroHeavyTests(unittest.TestCase):
    def test_a_mostly_zero_reading_is_compared_on_means_and_the_share_above_zero(self):
        rng = np.random.default_rng(33)
        rain = np.where(rng.random(600) < 0.9, 0.0, rng.exponential(2.0, 600))
        rain[:300] *= 2
        frame = pd.DataFrame({"Station": np.repeat(["Oak", "Foster"], 300), "Rain": rain})

        comparison = compare_groups(frame, "Rain", "Station")

        assert comparison is not None
        self.assertTrue(comparison.zero_heavy)
        self.assertIsNotNone(comparison.presence_p)
        self.assertEqual(comparison.centre(comparison.higher), comparison.higher.mean)
        statement = describe_comparison(comparison)
        self.assertIn("share of values that are not 0", statement)
        self.assertNotIn("median 0 vs 0", statement)
        self.assertIn("Share not 0", comparison_table(comparison).columns)


class OrderedTrendTests(unittest.TestCase):
    def test_a_numeric_setting_gets_a_trend_test(self):
        rng = np.random.default_rng(34)
        speed = np.repeat([20, 30, 40, 50, 60], 6)
        elongation = 20 - 0.1 * speed + rng.normal(0, 1.5, 30)
        frame = pd.DataFrame({"Welding speed": speed, "Elongation": elongation})

        comparison = compare_groups(frame, "Elongation", "Welding speed")

        assert comparison is not None and comparison.trend is not None
        rho, p = stats.spearmanr(frame["Welding speed"], frame["Elongation"])
        self.assertAlmostEqual(comparison.trend.rho, rho, places=10)
        self.assertAlmostEqual(comparison.trend.p_value, p, places=6)
        self.assertEqual(comparison.trend.verdict, "clear")
        self.assertIn("Elongation falls as Welding speed increases", describe_comparison(comparison))

    def test_labels_named_as_groups_get_no_trend(self):
        frame = pd.DataFrame({"Status": np.repeat([0, 1, 2], 10), "TVOC": np.arange(30.0)})

        comparison = compare_groups(frame, "TVOC", "Status")

        assert comparison is not None
        self.assertIsNone(comparison.trend)
        self.assertFalse(ordered_levels(frame["Status"], "Status"))

    def test_a_scan_finds_a_trend_the_unordered_test_misses(self):
        rng = np.random.default_rng(35)
        dose = np.repeat([0, 1, 2, 3, 4], 8)
        frame = pd.DataFrame({
            "Dose": dose,
            "Response": 0.5 * dose + rng.normal(0, 1, 40),
            **{f"Noise{index}": rng.normal(0, 1, 40) for index in range(8)},
        })

        scan = scan_differences(frame, ["Response", *[f"Noise{index}" for index in range(8)]], ["Dose"])

        assert scan is not None
        self.assertEqual(scan.trends, 9)
        self.assertEqual(scan.tested, 18)
        found = [item for item in scan.named if item.measure == "Response"]
        self.assertTrue(found)
        self.assertEqual(found[0].trend.verdict, "clear")
        self.assertEqual(found[0].verdict, "none")
        self.assertIn("Response rises as Dose increases", lead_sentence(found[0]))


class EntitySummaryTests(unittest.TestCase):
    def test_a_per_entity_figure_says_what_it_is(self):
        rng = np.random.default_rng(36)
        birds = np.repeat(np.arange(16), 6)
        frame = pd.DataFrame({
            "Bird": birds.astype(str),
            "Phenotype": np.where(birds < 8, "bold", "shy"),
            "Time": rng.normal(100, 10, 96) + np.where(birds < 8, 30, 0),
        })

        comparison = compare_groups(frame, "Time", "Phenotype", entity="Bird")

        assert comparison is not None
        self.assertEqual(centre_word(comparison), "mean of per-Bird means")
        self.assertIn("mean of per-Bird means", describe_comparison(comparison))

    def test_variance_shares_count_each_entity_once(self):
        frame = pd.DataFrame({
            "Cell": ["a"] * 10 + ["b"] * 10 + ["c", "d"],
            "Type": ["x"] * 10 + ["y"] * 10 + ["x", "y"],
            "Z": np.r_[np.linspace(0, 9, 10), np.linspace(1, 10, 10), [4.0, 6.0]],
        })

        per_row = dict(variance_explained(frame, "Z", ["Type"]))["Type"]
        per_cell = dict(variance_explained(frame, "Z", ["Type"], entity="Cell"))["Type"]

        self.assertNotAlmostEqual(per_row, per_cell)
        means = pd.Series([4.5, 4.0, 5.5, 6.0])
        types = pd.Series(["x", "x", "y", "y"])
        expected = ((means.groupby(types).transform("mean") - means.mean()) ** 2).sum() / (
            (means - means.mean()) ** 2
        ).sum()
        self.assertAlmostEqual(per_cell, expected, places=10)

    def test_a_decimal_reading_is_never_the_unit(self):
        rng = np.random.default_rng(37)
        frame = pd.DataFrame({"NO": rng.normal(20, 5, 50).round(2), "O3": rng.normal(40, 5, 50)})
        frame.loc[1, "NO"] = frame.loc[0, "NO"]
        roles = ColumnRoles(date=None, measure="O3", dimension=None, identifier=None,
                            numeric=("NO", "O3"), dimensions=())

        self.assertIsNone(entity_column(frame, roles))


class ScreenTests(unittest.TestCase):
    def test_a_wide_screen_tests_every_reading_with_a_false_discovery_rate(self):
        rng = np.random.default_rng(38)
        genes = pd.DataFrame(rng.normal(0, 1, (120, 400)), columns=[f"Gene_{index}" for index in range(400)])
        genes.iloc[:60, :10] += 1.5
        genes["Diagnosis"] = ["AML"] * 60 + ["ALL"] * 60

        scan = scan_differences(genes, [f"Gene_{index}" for index in range(400)], ["Diagnosis"], limit=20)

        assert scan is not None
        self.assertEqual(scan.correction, "Benjamini-Hochberg")
        self.assertEqual(len(scan.readings), 400)
        self.assertGreaterEqual(scan.clear, 10)
        self.assertIn("false-discovery-rate adjusted", comparison_calculation(scan.named[0]))
        self.assertIn("fdr_bh", comparison_calculation(scan.named[0]))

    def test_benjamini_hochberg_matches_the_textbook_values(self):
        adjusted = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
        np.testing.assert_allclose(adjusted, [0.04, 0.05333333, 0.05333333, 0.5])

    def test_a_finding_on_few_rows_is_listed_after_a_well_covered_one(self):
        rng = np.random.default_rng(39)
        size = 400
        frame = pd.DataFrame({
            "Arm": ["A", "B"] * (size // 2),
            "Common": rng.normal(0, 1, size) + np.tile([0.8, 0.0], size // 2),
            "Rare": rng.normal(0, 1, size) + np.tile([3.0, 0.0], size // 2),
        })
        frame.loc[40:, "Rare"] = np.nan

        scan = scan_differences(frame, ["Common", "Rare"], ["Arm"])

        assert scan is not None
        self.assertEqual([item.measure for item in scan.named], ["Common", "Rare"])
        self.assertLess(scan.named[1].coverage, 0.2)


class StemTests(unittest.TestCase):
    def test_a_column_and_its_copy_share_a_stem(self):
        self.assertTrue(same_stem("diabetes", "diabetes.1"))
        self.assertTrue(same_stem("MaxTemp", "Temp"))
        self.assertFalse(same_stem("Calcium (cmol(+)/kg)", "Sodium (cmol(+)/kg)"))


class RateTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Gender": [0] * 681 + [1] * 740,
            "Result": [1] * 208 + [0] * 473 + [1] * 412 + [0] * 328,
        })

    def test_rates_match_scipy_and_statsmodels(self):
        comparison = compare_rates(self.frame, "Result", "Gender")

        assert comparison is not None
        self.assertEqual(comparison.higher.label, "Gender 1")
        self.assertAlmostEqual(comparison.higher.rate, 412 / 740)
        table = np.array([[412, 328], [208, 473]])
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        self.assertAlmostEqual(comparison.p_value, p, places=12)
        self.assertAlmostEqual(comparison.cramers_v, np.sqrt(chi2 / table.sum()), places=12)
        low, high = confint_proportions_2indep(412, 740, 208, 681, method="newcomb", compare="diff")
        self.assertAlmostEqual(comparison.low, low)
        self.assertAlmostEqual(comparison.high, high)
        self.assertEqual(comparison.verdict, "clear")
        statement = describe_rates(comparison)
        self.assertIn("55.7% in Gender 1 (n = 740) against 30.5% in Gender 0 (n = 681)", statement)
        self.assertIn("percentage points (95% CI", statement)
        self.assertIn("scipy.stats.chi2_contingency", rate_calculation(comparison))

    def test_small_counts_use_fishers_exact_test(self):
        frame = pd.DataFrame({"Arm": ["A"] * 8 + ["B"] * 8, "Event": [1] * 6 + [0] * 2 + [1] * 1 + [0] * 7})

        comparison = compare_rates(frame, "Event", "Arm")

        assert comparison is not None
        self.assertEqual(comparison.test, "Fisher's exact test")
        self.assertAlmostEqual(comparison.p_value, stats.fisher_exact([[6, 2], [1, 7]])[1], places=12)

    def test_a_yes_no_grouping_is_named_with_its_column(self):
        frame = pd.DataFrame({"Smoker": ["Y", "N"] * 40, "Asthma": [1, 0, 0, 0] * 20})

        comparison = compare_rates(frame, "Asthma", "Smoker")

        assert comparison is not None
        self.assertEqual({group.label for group in comparison.groups}, {"Smoker = Y", "Smoker = N"})

    def test_yes_no_text_counts_yes(self):
        self.assertEqual(outcome_event(pd.Series(["Yes", "No", "No"])), "Yes")
        self.assertEqual(outcome_event(pd.Series([True, False])), True)
        self.assertIsNone(outcome_event(pd.Series([1, 2, 1])))
        self.assertIsNone(outcome_event(pd.Series(["North", "South"])))


class JointModelTests(unittest.TestCase):
    def test_coefficients_and_fit_match_least_squares(self):
        rng = np.random.default_rng(40)
        frame = pd.DataFrame(rng.uniform(0, 10, (300, 3)), columns=["T", "H", "SM"])
        noise = rng.normal(0, 1, 300)
        frame["Growth"] = 0.7 + 1.2 * frame["T"] + 0.8 * frame["H"] + 0.5 * frame["SM"] + noise

        model = fit_joint_model(frame, "Growth", ["H", "T", "SM"])

        assert model is not None
        design = np.column_stack([np.ones(300), frame[["H", "T", "SM"]].to_numpy()])
        coefficients, *_ = np.linalg.lstsq(design, frame["Growth"].to_numpy(), rcond=None)
        np.testing.assert_allclose([term.coefficient for term in model.terms], coefficients[1:], rtol=1e-8)
        residual = frame["Growth"].to_numpy() - design @ coefficients
        total = ((frame["Growth"] - frame["Growth"].mean()) ** 2).sum()
        self.assertAlmostEqual(model.r2, 1 - (residual**2).sum() / total, places=10)
        self.assertLess(model.adjusted_r2, model.r2)
        self.assertGreater(model.r2, model.best_single.pairwise**2 + 0.1)
        self.assertEqual(model.collinear, ())
        self.assertTrue(all(term.low < term.coefficient < term.high for term in model.terms))

    def test_a_confounded_reading_can_reverse_sign(self):
        rng = np.random.default_rng(41)
        family = rng.normal(0, 1, 400)
        cobalt = family + rng.normal(0, 0.3, 400)
        nickel = family + rng.normal(0, 0.3, 400)
        strength = 3 * nickel - 0.8 * cobalt + rng.normal(0, 0.5, 400)
        frame = pd.DataFrame({"co": cobalt, "ni": nickel, "strength": strength})

        model = fit_joint_model(frame, "strength", ["ni", "co"])

        assert model is not None
        self.assertEqual([term.name for term in model.reversed], ["co"])
        self.assertEqual(set(model.collinear), {"ni", "co"})

    def test_too_few_rows_per_predictor_fit_nothing(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(25, 4)), columns=["y", "a", "b", "c"])

        self.assertIsNone(fit_joint_model(frame, "y", ["a", "b", "c"]))
        self.assertIsNotNone(fit_joint_model(frame, "y", ["a", "b"]))

    def test_repeated_rows_count_once_towards_the_rows_needed(self):
        rng = np.random.default_rng(43)
        visits = pd.DataFrame(rng.normal(size=(6, 4)), columns=["y", "a", "b", "c"])
        frame = pd.concat([visits] * 20, ignore_index=True)

        self.assertIsNone(fit_joint_model(frame, "y", ["a", "b", "c"]))

    def test_an_identity_is_an_exact_fit(self):
        rng = np.random.default_rng(44)
        frame = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "noise"])
        frame["total"] = frame["a"] + frame["b"]

        model = fit_joint_model(frame, "total", ["a", "b", "noise"])

        assert model is not None
        self.assertTrue(model.exact)
        self.assertEqual([term.name for term in model.clear], ["a", "b"])
        noisy = frame.assign(total=frame["total"] + frame["noise"])
        self.assertFalse(fit_joint_model(noisy, "total", ["a", "b"]).exact)


if __name__ == "__main__":
    unittest.main()
