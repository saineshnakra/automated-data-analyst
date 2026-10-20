import unittest

import numpy as np
import pandas as pd
from scipy import stats

from ada.compare import (
    MAX_SCAN_COLUMNS,
    cap_flag,
    combined_groupings,
    compare_groups,
    comparison_calculation,
    comparison_table,
    describe_comparison,
    entity_column,
    find_derivation,
    holm_adjust,
    most_influential,
    pairwise_correlations,
    scan_differences,
    scan_relationships,
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


if __name__ == "__main__":
    unittest.main()
