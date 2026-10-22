import time
import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from ada.quality import (
    blank_codes,
    flagged_index,
    name_bounds,
    prefers_log_scale,
    quality_frame,
    scan,
    show_value,
    statistics_frame,
    statistics_mask,
)
from ada.schema import ColumnRoles


def roles_for(frame, measure=None, dimension=None, factors=()):
    numeric = tuple(column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column]))
    return ColumnRoles(
        date=None,
        measure=measure,
        dimension=dimension,
        identifier=None,
        numeric=numeric,
        dimensions=(dimension,) if dimension else (),
        factors=tuple(factors),
    )


def checks(issues, check):
    return [issue for issue in issues if issue.check == check]


class NameBoundTests(unittest.TestCase):
    def test_names_imply_ranges(self):
        self.assertEqual(name_bounds("Viability %")[:2], (0.0, 100.0))
        self.assertEqual(name_bounds("NEUTp")[:2], (0.0, 100.0))
        self.assertEqual(name_bounds("Soil_pH")[:2], (0.0, 14.0))
        self.assertEqual(name_bounds("NDVI")[:2], (-1.0, 1.0))
        self.assertEqual(name_bounds("Expected_Yield")[:2], (0.0, float("inf")))
        self.assertEqual(name_bounds("Phosphorus mg/kg")[:2], (0.0, float("inf")))

    def test_signed_and_business_names_have_no_bound(self):
        self.assertIsNone(name_bounds("Revenue Growth %"))
        self.assertIsNone(name_bounds("Air Temperature"))
        self.assertIsNone(name_bounds("Revenue"))
        self.assertIsNone(name_bounds("Units"))

    def test_a_quantile_label_is_not_a_percentage(self):
        self.assertIsNone(name_bounds("25%"))
        self.assertIsNone(name_bounds("p95%"))


class ImpossibleValueTests(unittest.TestCase):
    def test_percentages_over_100_and_negative_yields(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "Coverage %": np.r_[rng.uniform(0, 100, 98), [180.0, 250.0]],
            "Yield": np.r_[rng.uniform(100, 500, 99), [-40.0]],
        })

        issues = checks(scan(frame), "impossible")

        found = {issue.columns[0]: issue for issue in issues}
        self.assertEqual(found["Coverage %"].rows, 2)
        self.assertIn("a percentage runs from 0 to 100", found["Coverage %"].rule)
        self.assertEqual(found["Yield"].rows, 1)
        self.assertIn("-40", found["Yield"].detail)

    def test_a_column_mostly_outside_its_name_is_on_another_scale(self):
        frame = pd.DataFrame({"Coverage %": np.linspace(50, 900, 100)})

        self.assertFalse(checks(scan(frame), "impossible"))

    def test_a_sign_slip_is_flagged_but_a_tail_through_zero_is_not(self):
        rng = np.random.default_rng(1)
        haemoglobin = np.r_[rng.normal(13, 1.5, 199), [-12.0]]
        cold = np.r_[rng.normal(60, 15, 195), [-1.0, -3.0, -8.0, -15.0, -20.0]]
        frame = pd.DataFrame({"HGB": haemoglobin, "MIN": cold})

        issues = checks(scan(frame), "impossible")

        self.assertEqual([issue.columns for issue in issues], [("HGB",)])

    def test_refunds_in_a_sales_column_are_not_errors(self):
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({"Revenue": np.r_[rng.uniform(20, 200, 199), [-80.0]]})

        self.assertFalse(scan(frame))


class SentinelTests(unittest.TestCase):
    def test_code_values_far_from_the_rest_are_flagged(self):
        rng = np.random.default_rng(3)
        speed = np.r_[rng.gamma(2, 1.5, 996), [999.9] * 4]
        radiation = np.r_[rng.uniform(0, 900, 998), [-100000.0] * 2]
        frame = pd.DataFrame({"Wind Speed": speed, "Solar Radiation": radiation})

        issues = checks(scan(frame), "sentinel")

        found = {issue.columns[0]: issue for issue in issues}
        self.assertEqual(found["Wind Speed"].rows, 4)
        self.assertEqual(found["Wind Speed"].examples, ("999.9",))
        self.assertEqual(found["Solar Radiation"].rows, 2)

    def test_a_code_inside_a_populated_range_is_a_reading(self):
        rng = np.random.default_rng(4)
        counts = np.r_[rng.integers(0, 300, 1000), [99, 99, 999]].astype(float)
        counts[:100] = rng.uniform(500, 2000, 100)

        issues = checks(scan(pd.DataFrame({"Reads": counts})), "sentinel")

        self.assertFalse(issues)

    def test_a_converted_blank_is_found_as_a_spike_in_the_tail(self):
        rng = np.random.default_rng(5)
        fahrenheit = np.round(np.r_[rng.normal(80, 8, 4800), rng.normal(10, 15, 200)])
        fahrenheit[:120] = 0  # blanks written as 0 °F before conversion
        celsius = (fahrenheit - 32) * 5 / 9

        issues = checks(scan(pd.DataFrame({"MaxTemp": celsius})), "sentinel")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rows, int((fahrenheit == 0).sum()))
        self.assertIn("0 °F", issues[0].detail)

    def test_the_same_code_in_several_columns_is_one_issue(self):
        rng = np.random.default_rng(6)
        frame = pd.DataFrame({
            "Max Temp": np.r_[rng.normal(30, 3, 500), [-9999.0] * 3],
            "Min Temp": np.r_[rng.normal(20, 3, 501), [-9999.0] * 2],
        })

        issues = checks(scan(frame), "sentinel")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].columns, ("Max Temp", "Min Temp"))
        self.assertEqual(issues[0].rows, 5)


class CapAndFillTests(unittest.TestCase):
    def test_a_value_piled_at_the_maximum_is_a_cap(self):
        rng = np.random.default_rng(7)
        times = np.r_[rng.uniform(0, 3000, 340), [3600.0] * 13]

        issues = checks(scan(pd.DataFrame({"Latency": times})), "cap")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].group, ("maximum", 3600.0))
        self.assertEqual(issues[0].rows, 13)

    def test_small_counts_thinning_out_from_their_minimum_are_not_a_floor(self):
        counts = np.repeat([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0], [11, 9, 9, 7, 6, 5, 4])

        self.assertFalse(checks(scan(pd.DataFrame({"Accounts": counts})), "cap"))

    def test_zero_is_a_natural_floor(self):
        rng = np.random.default_rng(8)
        rain = np.where(rng.random(1000) < 0.8, 0.0, rng.gamma(1, 3, 1000))

        self.assertFalse(checks(scan(pd.DataFrame({"Rain": rain})), "cap"))

    def test_a_repeated_non_round_value_on_the_same_rows_is_one_fill_issue(self):
        rng = np.random.default_rng(9)
        filled = rng.random(400) < 0.6
        frame = pd.DataFrame({
            "HCT": np.where(filled, 46.1526, rng.normal(40, 4, 400).round(1)),
            "PCT": np.where(filled, 0.26028, rng.normal(0.25, 0.05, 400).round(2)),
        })

        issues = checks(scan(frame), "imputed")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].columns, ("HCT", "PCT"))
        self.assertEqual(issues[0].rows, int(filled.sum()))

    def test_a_common_round_value_is_not_a_fill(self):
        rng = np.random.default_rng(10)
        rate = np.where(rng.random(500) < 0.3, 60.0, rng.normal(75, 10, 500).round())

        self.assertFalse(checks(scan(pd.DataFrame({"Heart rate": rate})), "imputed"))


class ExtremeValueTests(unittest.TestCase):
    def test_values_cut_off_far_from_the_bulk_are_extreme(self):
        rng = np.random.default_rng(11)
        values = np.r_[rng.normal(40, 4, 997).round(1), [316.0, 741.0, 3715.0]]

        issues = checks(scan(pd.DataFrame({"HCT": values})), "extreme")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rows, 3)
        self.assertIn("3,715", issues[0].detail)

    def test_a_long_right_tail_is_judged_on_a_log_scale(self):
        rng = np.random.default_rng(12)
        values = rng.lognormal(0, 1.2, 5000)

        self.assertFalse(checks(scan(pd.DataFrame({"Troponin": values})), "extreme"))

    def test_extreme_values_on_the_same_rows_are_one_issue(self):
        rng = np.random.default_rng(24)
        frame = pd.DataFrame({f"{400 + band}": rng.normal(1.0, 0.05, 300) for band in range(5)})
        frame.iloc[[7, 8], :] = 5e6  # two spectra recorded on another scale

        issues = checks(scan(frame), "extreme")

        self.assertEqual(len(issues), 1)
        self.assertEqual(len(issues[0].columns), 5)
        self.assertEqual(issues[0].rows, 2)

    def test_a_continuous_tail_is_not_extreme(self):
        rng = np.random.default_rng(13)
        tropics = rng.normal(30, 2, 9000)
        arctic = rng.normal(-15, 10, 300)
        values = np.round(np.r_[tropics, arctic] * 1.8) / 1.8

        self.assertFalse(checks(scan(pd.DataFrame({"MaxTemp": values})), "extreme"))


class SurveyCodeTests(unittest.TestCase):
    def test_refusal_codes_above_the_answers_are_flagged(self):
        insurance = np.r_[[1] * 300, [2] * 100, [7, 7, 9, 9]]
        likert = np.r_[np.repeat([1, 2, 3, 4, 5, 6, 7], 30)]
        frame = pd.DataFrame({"insurance": insurance}).assign(
            score_scale=np.resize(likert, len(insurance))
        )

        issues = checks(scan(frame), "survey_code")

        self.assertEqual([issue.columns for issue in issues], [("insurance",)])
        self.assertEqual(issues[0].rows, 4)


class PairTests(unittest.TestCase):
    def test_a_column_copied_into_another_within_one_group(self):
        rng = np.random.default_rng(14)
        city = np.repeat(["A", "B", "C", "D"], 300)
        co = rng.gamma(2, 0.5, 1200)
        no = rng.gamma(2, 10, 1200)
        no[:240] = co[:240]  # most of city A
        frame = pd.DataFrame({"City": city, "CO": co, "NO": no})

        issues = checks(scan(frame, roles_for(frame, "CO", "City")), "equal_columns")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].group, "A")

    def test_a_wide_table_pairs_only_the_measure(self):
        rng = np.random.default_rng(25)
        frame = pd.DataFrame({f"channel_{i}": rng.normal(10, 2, 60) for i in range(120)})
        frame["channel_5"] = frame["channel_4"]
        frame["channel_9"] = frame["channel_0"]

        issues = checks(scan(frame, roles_for(frame, "channel_0")), "equal_columns")

        self.assertEqual([issue.columns for issue in issues], [("channel_0", "channel_9")])

    def test_max_below_min_is_flagged(self):
        frame = pd.DataFrame({
            "MaxTemp": [30.5, 31.25, 25.75, 10.5, 28.0, 29.5, 33.25, 27.75] * 5,
            "MinTemp": [20.5, 21.25, 15.75, 18.5, 18.0, 19.5, 23.25, 17.75] * 5,
        })

        issues = checks(scan(frame), "min_max")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].rows, 5)

    def test_a_group_off_by_a_unit_factor_is_out_of_scale(self):
        rng = np.random.default_rng(15)
        base = rng.normal(2.0, 0.2, 200)
        base[:50] *= 1000  # one city recorded in µg rather than mg
        frame = pd.DataFrame({"City": np.repeat(["A", "B", "C", "D"], 50), "CO": base})

        issues = checks(scan(frame, roles_for(frame, "CO", "City")), "group_level")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].group, "A")
        self.assertIn("1,000", issues[0].detail)

    def test_a_group_far_off_but_not_by_a_unit_factor_is_left_to_the_comparison(self):
        rng = np.random.default_rng(26)
        base = rng.normal(2.0, 0.1, 200)
        base[:50] *= 45
        frame = pd.DataFrame({"Status": np.repeat(["fire", "a", "b", "c"], 50), "TVOC": base})

        self.assertFalse(checks(scan(frame, roles_for(frame, "TVOC", "Status")), "group_level"))

    def test_a_phase_filling_one_stretch_of_time_is_not_a_unit_problem(self):
        rng = np.random.default_rng(27)
        base = rng.normal(2.0, 0.1, 200)
        base[150:] *= 100
        frame = pd.DataFrame({
            "Time": pd.date_range("2024-01-01", periods=200, freq="s"),
            "Status": np.repeat(["before", "warming", "cooling", "fire"], 50),
            "TVOC": base,
        })
        roles = replace(roles_for(frame, "TVOC", "Status"), date="Time")

        self.assertFalse(checks(scan(frame, roles), "group_level"))

    def test_two_groups_far_apart_are_left_to_the_comparison(self):
        rng = np.random.default_rng(16)
        frame = pd.DataFrame({
            "Result": np.repeat(["positive", "negative"], 100),
            "Troponin": np.r_[rng.gamma(2, 0.05, 100), rng.gamma(2, 0.003, 100)],
        })

        self.assertFalse(checks(scan(frame, roles_for(frame, "Troponin", "Result")), "group_level"))


class MissingDataTests(unittest.TestCase):
    def test_columns_missing_on_the_same_rows_form_a_block(self):
        rng = np.random.default_rng(17)
        fasting = rng.random(500) < 0.5
        frame = pd.DataFrame({
            "age": rng.uniform(20, 80, 500),
            "glucose": np.where(fasting, rng.normal(100, 10, 500), np.nan),
            "insulin": np.where(fasting, rng.normal(10, 2, 500), np.nan),
        })

        blocks = checks(scan(frame), "missing_block")

        self.assertEqual([issue.columns for issue in blocks], [("glucose", "insulin")])

    def test_missingness_that_follows_another_column_is_described(self):
        rng = np.random.default_rng(18)
        ph = rng.uniform(4, 8, 1000)
        acidity = np.where(ph < 5.5, rng.gamma(2, 1, 1000), np.nan)
        frame = pd.DataFrame({"pH": ph, "Acidity": acidity})

        found = checks(scan(frame), "missing_dependency")

        self.assertEqual(len(found), 1)
        self.assertIn("missing mostly where pH is high", found[0].detail)

    def test_a_group_that_never_records_a_column(self):
        rng = np.random.default_rng(19)
        station = np.repeat(["North", "South", "East"], 200)
        wet_bulb = np.where(station == "East", np.nan, rng.normal(10, 3, 600))
        frame = pd.DataFrame({"Station": station, "Wet Bulb": wet_bulb, "Air": rng.normal(12, 3, 600)})

        found = checks(scan(frame, roles_for(frame, "Air", "Station")), "missing_by_group")

        self.assertEqual(len(found), 1)
        self.assertIn("East never records Wet Bulb", found[0].detail)


class HelperTests(unittest.TestCase):
    def test_values_are_shown_exactly(self):
        self.assertEqual(show_value(3715.0), "3,715")
        self.assertEqual(show_value(46.1526), "46.1526")
        self.assertEqual(show_value(-100000.0), "-100,000")

    def test_the_frame_lists_every_issue_with_its_rule(self):
        frame = pd.DataFrame({"Yield": [1.5, 2.25, 3.75, -4.5] + [5.5, 6.25, 7.75, 8.5, 9.25] * 4})

        table = quality_frame(scan(frame))

        self.assertEqual(
            list(table.columns), ["Check", "Columns", "Rows", "Share %", "Detail", "Rule", "Examples"]
        )
        self.assertEqual(table.iloc[0]["Check"], "Impossible values")

    def test_flagged_index_returns_the_flagged_rows(self):
        rng = np.random.default_rng(20)
        values = np.r_[rng.normal(40, 4, 99), [4000.0]]
        frame = pd.DataFrame({"HCT": values})

        self.assertEqual(list(flagged_index(scan(frame), "HCT")), [99])

    def test_identifiers_and_codes_are_not_scanned_as_readings(self):
        frame = pd.DataFrame({
            "seqn": range(1000, 1100),
            "Status": [0, 1, 2, 1] * 25,
            "Value": np.linspace(1, 100, 100),
        })

        self.assertFalse(scan(frame, roles_for(frame, "Value", factors=("Status",))))

    def test_infinite_values_are_ignored(self):
        frame = pd.DataFrame({"Revenue": [np.inf, 1.0, 2.0, 3.0, -np.inf] * 5})

        self.assertEqual(scan(frame), ())


class PhysicalLimitTests(unittest.TestCase):
    def blood_panel(self, rows=400, seed=40):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "Na": rng.normal(140, 4, rows).round(1),
            "K": rng.normal(4.3, 0.4, rows).round(2),
            "RBC": rng.normal(4.8, 0.5, rows).round(2),
            "ALB": rng.normal(4.1, 0.4, rows).round(1),
        })

    def test_clinical_analytes_outside_survivable_ranges_are_impossible(self):
        frame = self.blood_panel()
        frame.loc[:59, "K"] = np.linspace(10.5, 19.6, 60)
        frame.loc[:9, "Na"] = 278.8
        frame.loc[100:139, "RBC"] = 33.9

        issues = {issue.columns[0]: issue for issue in checks(scan(frame), "impossible")}

        self.assertEqual(issues["K"].rows, 60)
        self.assertIn("potassium level in mmol/L", issues["K"].rule)
        self.assertIn("in a living patient", issues["K"].rule)
        self.assertEqual(issues["Na"].rows, 10)
        self.assertEqual(issues["RBC"].rows, 40)

    def test_a_short_analyte_name_needs_a_second_analyte(self):
        rng = np.random.default_rng(41)
        potassium = np.r_[rng.normal(4.5, 0.5, 199), [15.0]]
        frame = pd.DataFrame({"K": potassium, "Depth": rng.uniform(0, 30, 200)})

        self.assertFalse(checks(scan(frame), "impossible"))

    def test_the_unit_follows_the_median(self):
        frame = self.blood_panel()
        rng = np.random.default_rng(42)
        frame["Haemoglobin"] = rng.normal(135, 12, len(frame)).round()  # g/L, not g/dL
        frame.loc[0, "Haemoglobin"] = 240.0
        frame.loc[1, "Haemoglobin"] = 310.0

        issues = {issue.columns[0]: issue for issue in checks(scan(frame), "impossible")}

        self.assertEqual(issues["Haemoglobin"].rows, 1)
        self.assertIn("g/L", issues["Haemoglobin"].rule)

    def test_a_unit_the_table_does_not_list_leaves_the_column_alone(self):
        frame = self.blood_panel()
        frame["Sodium (mg)"] = np.r_[[140.0] * 10, np.linspace(100, 900, len(frame) - 10)]

        self.assertNotIn("Sodium (mg)", [issue.columns[0] for issue in checks(scan(frame), "impossible")])

    def test_wind_speed_in_metres_a_second_is_bounded(self):
        rng = np.random.default_rng(43)
        frame = pd.DataFrame({"WS(m/s)": np.r_[rng.uniform(0.1, 1.0, 140), [136.4, 279.6]]})

        issues = checks(scan(frame), "impossible")

        self.assertEqual(issues[0].rows, 2)
        self.assertIn("stays below 115", issues[0].rule)

    def test_temperatures_are_never_judged_on_a_log_scale(self):
        rng = np.random.default_rng(44)
        values = np.sort(np.r_[rng.gamma(9, 0.8, 146) + 22, [80.6]])

        self.assertFalse(prefers_log_scale(values, "TEMP(degreeC)"))
        frame = pd.DataFrame({"TEMP(degreeC)": values})
        self.assertEqual(checks(scan(frame), "extreme")[0].examples, ("80.6",))


class FalseAlarmTests(unittest.TestCase):
    def test_a_repeated_digit_that_reads_as_a_typo_is_an_entry_error(self):
        rng = np.random.default_rng(45)
        frame = pd.DataFrame({"Heart rate": np.r_[rng.normal(75, 12, 997).round(), [1111.0] * 3]})

        issues = checks(scan(frame), "sentinel")

        self.assertIn("likely entry error", issues[0].summary)
        self.assertIn("111", issues[0].detail)

    def test_a_shared_floor_is_not_a_copy(self):
        rng = np.random.default_rng(46)
        v = np.where(rng.random(300) < 0.65, 0.01, rng.uniform(0.02, 2.0, 300)).round(2)
        nb = np.where(rng.random(300) < 0.7, 0.01, rng.uniform(0.02, 1.0, 300)).round(2)
        nb[v == 0.01] = 0.01
        frame = pd.DataFrame({"v": v, "nb": nb})

        self.assertFalse(checks(scan(frame), "equal_columns"))

    def test_identical_columns_are_a_copy_even_when_most_rows_sit_at_their_floor(self):
        rng = np.random.default_rng(66)
        codes = rng.choice(np.r_[[0.0], 10.0 ** np.arange(1, 10), 10.0 ** np.arange(1, 11) + 1], 600)
        codes[rng.random(600) < 0.9] = 1.0
        frame = pd.DataFrame({"PoorWeather": codes, "Flags": codes.copy(), "Rain": rng.gamma(2, 3, 600)})

        issues = checks(scan(frame), "equal_columns")

        self.assertEqual([issue.columns for issue in issues], [("PoorWeather", "Flags")])

    def test_a_row_maximum_is_not_a_copy(self):
        rng = np.random.default_rng(47)
        parts = pd.DataFrame(rng.gamma(4, 15, (400, 4)).round(), columns=["pm10", "so2", "co", "o3"])
        parts["o3"] *= 2
        frame = parts.assign(max=parts.max(axis=1))

        self.assertFalse(checks(scan(frame), "equal_columns"))

    def test_spreads_and_two_groups_are_not_max_below_min(self):
        frame = pd.DataFrame({
            "high_mp_mean": [101.05, 140.0, 150.0, 160.0] * 6,
            "low_mp_mean": [104.56, 100.0, 110.0, 120.0] * 6,
            "high_mp_sd": [0.55, 1.0, 2.0, 3.0] * 6,
            "low_mp_sd": [0.68, 2.0, 3.0, 4.0] * 6,
            "high_mp_n": [10, 12, 14, 16] * 6,
            "low_mp_n": [11, 13, 15, 17] * 6,
        })

        self.assertFalse(checks(scan(frame), "min_max"))

    def test_a_sample_size_is_not_capped(self):
        frame = pd.DataFrame({"sample_size": [2000] * 4 + list(range(30, 58))})

        self.assertFalse(checks(scan(frame), "cap"))

    def test_a_value_shared_by_one_category_is_not_a_floor(self):
        rng = np.random.default_rng(48)
        group = np.repeat(["Herbivore", "Planktivore", "Carnivore", "Piscivore"], 200)
        level = np.r_[
            [2.0] * 200, rng.uniform(2.5, 3.2, 200), rng.uniform(3.3, 4.0, 200), rng.uniform(4.0, 4.5, 200)
        ]
        frame = pd.DataFrame({"trophic_group": group, "trophic_level": level.round(2)})

        self.assertFalse(checks(scan(frame), "cap"))

    def test_a_common_value_at_the_end_of_a_range_is_not_a_floor(self):
        rng = np.random.default_rng(49)
        recipe = np.r_[[594.0] * 30, [755.8] * 30, [670.0] * 23, rng.uniform(600, 900, 1000).round(1)]
        self.assertFalse(checks(scan(pd.DataFrame({"Fine Aggregate (kg/m3)": recipe})), "cap"))

    def test_a_simple_fraction_is_not_a_fill_value(self):
        rng = np.random.default_rng(50)
        means = np.r_[[1 / 60] * 120, rng.lognormal(0, 1.5, 880)]
        self.assertFalse(checks(scan(pd.DataFrame({"Abundance": means})), "imputed"))


class ClusterAndSwapTests(unittest.TestCase):
    def test_a_detached_cluster_above_two_percent_is_reported(self):
        rng = np.random.default_rng(51)
        cluster = rng.uniform(6975, 83148, 140).round()
        values = rng.permutation(np.r_[rng.normal(83, 10, 4400).round(), cluster])
        frame = pd.DataFrame({"mq135": values})

        issues = checks(scan(frame), "cluster")

        self.assertEqual(issues[0].rows, 140)
        self.assertIn("second cluster", issues[0].detail)
        self.assertIn("mean", issues[0].detail)

    def test_a_plausible_tail_value_needs_checking_rather_than_an_error_verdict(self):
        rng = np.random.default_rng(52)
        values = np.r_[rng.normal(1.0, 0.1, 300).round(3), [1.5, 2.1, 2.2]]

        issue = checks(scan(pd.DataFrame({"v": values})), "extreme")[0]

        self.assertIn("Check that they are real", issue.detail)
        self.assertNotIn("entry or unit errors", issue.detail)

    def test_two_columns_swapped_on_a_block_of_rows(self):
        rng = np.random.default_rng(53)
        days = pd.date_range("2023-01-01", periods=147, freq="D")
        speed = rng.uniform(0.1, 0.99, 147).round(2)
        direction = rng.uniform(103, 283, 147).round(1)
        speed[135:], direction[135:] = direction[135:].copy(), speed[135:].copy()
        frame = pd.DataFrame({
            "Date": days,
            "Wind speed": speed,
            "Wind heading": direction,
            "Temp": 20.0 + rng.normal(0, 3, 147),
        })
        roles = replace(roles_for(frame), date="Date")

        issues = scan(frame, roles)

        swapped = checks(issues, "swapped")
        self.assertEqual(swapped[0].rows, 12)
        self.assertIn("from 16 May 2023 to 27 May 2023", swapped[0].detail)
        # The swap explains the far values, so they are not listed again.
        self.assertFalse(checks(issues, "cluster") + checks(issues, "extreme"))

    def test_empty_numeric_columns_are_skipped_by_the_value_checks(self):
        frame = pd.DataFrame({
            "Food": [f"item {index}" for index in range(60)],
            "Particles": ["5.66 ± 1.69 items/10 g"] * 60,
            "Mean concentration": np.full(60, np.nan),
            "Sample size": np.full(60, np.nan),
        })

        issues = scan(frame)

        self.assertFalse([issue for issue in issues if not issue.check.startswith("missing")])


class TableCheckTests(unittest.TestCase):
    def test_copied_rows_are_counted_with_the_distinct_rows_left(self):
        rng = np.random.default_rng(54)
        base = pd.DataFrame({
            "Hemoglobin": rng.normal(13, 2, 534).round(1), "MCV": rng.normal(85, 9, 534).round(1),
        })
        frame = pd.concat([base, base.iloc[:300], base.iloc[:200]], ignore_index=True)

        issue = checks(scan(frame), "duplicates")[0]

        self.assertGreaterEqual(issue.rows, 500)
        self.assertIn(f"{len(frame) - issue.rows:,} distinct rows", issue.detail)

    def test_repeats_in_a_table_of_codes_and_counts_are_not_listed(self):
        rng = np.random.default_rng(55)
        frame = pd.DataFrame({"Region": rng.choice(["N", "S"], 400), "Units": rng.integers(1, 40, 400)})

        self.assertFalse(checks(scan(frame), "duplicates"))

    def test_flat_unrelated_columns_look_simulated(self):
        rng = np.random.default_rng(56)
        frame = pd.DataFrame({f"Gene_{i}": rng.uniform(0, 5, 1000) for i in range(30)})

        issues = checks(scan(frame), "generated")

        self.assertEqual(len(issues), 1)
        self.assertIn("may be simulated", issues[0].detail)

    def test_related_readings_do_not_look_simulated(self):
        rng = np.random.default_rng(57)
        shared = rng.uniform(0, 5, 1000)
        frame = pd.DataFrame({f"Gene_{i}": shared * 0.3 + rng.uniform(0, 5, 1000) for i in range(10)})

        self.assertFalse(checks(scan(frame), "generated"))

    def test_runs_of_zeros_and_a_stuck_value_are_reported(self):
        rng = np.random.default_rng(58)
        hours = pd.date_range("2020-01-01", periods=3000, freq="h")
        benzene = rng.lognormal(0, 0.6, 3000).round(2)
        benzene[1000:1400] = 0.0
        speed = rng.gamma(3, 1.2, 3000).round(1)
        speed[2000:2600] = 3.3
        rain = np.where(rng.random(3000) < 0.9, 0.0, rng.gamma(1, 2, 3000)).round(1)
        frame = pd.DataFrame({"Time": hours, "Benzene": benzene, "Wind Speed": speed, "Rain": rain,
                              "Total Rain": np.cumsum(rain)})
        roles = replace(roles_for(frame), date="Time")

        stuck = {issue.columns[0]: issue for issue in checks(scan(frame, roles), "flatline")}

        self.assertEqual(stuck["Benzene"].rows, 400)
        self.assertIn("exactly 0", stuck["Benzene"].detail)
        self.assertEqual(stuck["Wind Speed"].rows, 600)
        self.assertNotIn("Rain", stuck)
        self.assertNotIn("Total Rain", stuck)

    def test_a_matrix_row_off_scale_is_named_as_a_row(self):
        rng = np.random.default_rng(59)
        spectra = pd.DataFrame(rng.lognormal(0, 0.3, (200, 150)), columns=[str(400 + i) for i in range(150)])
        spectra.iloc[:, 40] *= 60  # a bright emission line in every spectrum
        spectra.iloc[7] *= 1e5  # one spectrum on another scale
        frame = pd.concat([pd.DataFrame({"file": [f"s{i}.txt" for i in range(200)]}), spectra], axis=1)
        roles = replace(roles_for(frame), identifier="file", table_shape="matrix")

        issues = scan(frame, roles)

        self.assertEqual([issue.check for issue in issues if issue.check in ("extreme", "cluster")], [])
        rows = checks(issues, "matrix_rows")
        self.assertEqual(rows[0].examples, ("s7.txt",))


class StatisticsFrameTests(unittest.TestCase):
    def test_codes_impossible_values_and_fills_are_blanked(self):
        rng = np.random.default_rng(60)
        filled = rng.random(400) < 0.3
        frame = pd.DataFrame({
            "Wind Speed": np.r_[rng.gamma(2, 1.5, 396), [999.9] * 4],
            "Coverage %": np.r_[rng.uniform(0, 100, 399), [250.0]],
            "HCT": np.where(filled, 46.1526, rng.normal(40, 4, 400).round(1)),
        })
        issues = scan(frame)

        mask = statistics_mask(frame, issues)
        cleaned, note = statistics_frame(frame, issues)

        self.assertEqual(int(mask["Wind Speed"].sum()), 4)
        self.assertEqual(int(cleaned["Coverage %"].isna().sum()), 1)
        self.assertEqual(int(cleaned["HCT"].isna().sum()), int(filled.sum()))
        self.assertIn(f"{4 + 1 + int(filled.sum()):,} cells", note)
        self.assertEqual(int(frame.isna().sum().sum()), 0)

    def test_nothing_flagged_returns_the_frame_itself(self):
        frame = pd.DataFrame({"Value": np.linspace(1, 100, 100)})

        cleaned, note = statistics_frame(frame, scan(frame))

        self.assertIs(cleaned, frame)
        self.assertEqual(note, "")


class RankingTests(unittest.TestCase):
    def test_issues_covering_more_rows_come_first(self):
        rng = np.random.default_rng(61)
        filled = rng.random(1000) < 0.6
        frame = pd.DataFrame({
            "MCV": np.r_[rng.normal(86, 6, 999), [-79.3]],
            "HCT": np.where(filled, 46.1526, rng.normal(40, 4, 1000).round(1)),
        })

        issues = scan(frame, roles_for(frame, "MCV"))

        self.assertEqual(issues[0].check, "imputed")
        self.assertEqual(issues[1].columns, ("MCV",))

    def test_small_shares_read_under_a_tenth_of_a_percent(self):
        rng = np.random.default_rng(62)
        frame = pd.DataFrame({"City": np.repeat(["A", "B", "C"], 4000), "CO": rng.gamma(2, 0.5, 12000)})
        frame["NO"] = rng.gamma(2, 10, 12000)
        frame.loc[:3500, "NO"] = frame.loc[:3500, "CO"]
        frame.loc[5000, "NO"] = frame.loc[5000, "CO"]

        issue = checks(scan(frame, roles_for(frame, "CO", "City")), "equal_columns")[0]

        self.assertIn("under 0.1% elsewhere", issue.detail)


class BlankCodeTests(unittest.TestCase):
    def test_a_clear_code_becomes_missing_with_a_note_and_an_issue(self):
        rng = np.random.default_rng(21)
        frame = pd.DataFrame({"Wind Speed": np.r_[rng.gamma(2, 1.5, 996), [999.9] * 4]})

        masked, issues, notes = blank_codes(frame)

        self.assertEqual(int(masked["Wind Speed"].isna().sum()), 4)
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].startswith("4 cells in Wind Speed held 999.9"), notes)
        self.assertEqual(int(frame["Wind Speed"].isna().sum()), 0)
        self.assertEqual([issue.check for issue in issues], ["sentinel"])
        self.assertEqual(issues[0].rows, 4)

    def test_a_code_in_every_column_gets_a_note_per_column_and_one_issue(self):
        rng = np.random.default_rng(24)
        pressure = np.r_[rng.normal(955, 8, 380), [-999.0] * 20]
        temperature = np.r_[rng.normal(9, 7, 390), [-999.0] * 10]
        frame = pd.DataFrame({"Pressure": pressure, "Temp": temperature})

        masked, issues, notes = blank_codes(frame)

        self.assertEqual(masked.isna().sum().to_dict(), {"Pressure": 20, "Temp": 10})
        self.assertEqual(len(notes), 2)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].columns, ("Pressure", "Temp"))
        self.assertIn("set those cells to missing", issues[0].detail)

    def test_a_value_that_only_looks_like_a_code_stays(self):
        rng = np.random.default_rng(22)
        frame = pd.DataFrame({"Price": np.r_[rng.uniform(10, 3000, 999), [9999.0]]})

        masked, issues, notes = blank_codes(frame)

        self.assertEqual((issues, notes), ((), ()))
        self.assertIs(masked, frame)

    def test_a_negative_value_shared_on_the_same_rows_is_a_code(self):
        rng = np.random.default_rng(25)
        frame = pd.DataFrame({name: rng.gamma(3, 30, 200) for name in ("mq2", "mq7", "pm10", "pm2_5")})
        frame.iloc[[5, 60, 61, 150]] = -1.0

        masked, issues, notes = blank_codes(frame)

        self.assertEqual(masked.isna().sum().tolist(), [4, 4, 4, 4])
        self.assertEqual(len(notes), 4)
        self.assertEqual(issues[0].rows, 4)
        self.assertIn("on 4 rows at once", issues[0].summary)

    def test_a_negative_value_in_one_column_or_on_different_rows_stays(self):
        rng = np.random.default_rng(26)
        frame = pd.DataFrame({name: rng.gamma(3, 30, 200) for name in ("a", "b", "c")})
        frame.loc[3, "a"] = frame.loc[4, "b"] = frame.loc[5, "c"] = -1.0

        masked, issues, _ = blank_codes(frame)

        self.assertIs(masked, frame)
        self.assertEqual(issues, ())

    def test_a_scale_of_minus_one_to_one_is_not_a_code(self):
        rng = np.random.default_rng(27)
        frame = pd.DataFrame({f"item_{i}": rng.choice([-1, 0, 1], 100) for i in range(4)})
        frame.iloc[:10] = -1

        masked, issues, _ = blank_codes(frame)

        self.assertIs(masked, frame)

    def test_keys_are_left_alone(self):
        frame = pd.DataFrame({"Station ID": [-999] * 5 + list(range(100, 195)), "Value": np.arange(100.0)})

        masked, _, _ = blank_codes(frame)

        self.assertIs(masked, frame)

    def test_scan_lists_the_codes_cleaning_set_aside(self):
        rng = np.random.default_rng(28)
        frame = pd.DataFrame({"Wind Speed": np.r_[rng.gamma(2, 1.5, 996), [999.9] * 4]})
        masked, issues, _ = blank_codes(frame)

        found = scan(masked, blanked=issues)

        self.assertIn("sentinel", [issue.check for issue in found])
        self.assertNotIn("sentinel", [issue.check for issue in scan(masked)])


class HeaderAndPanelTests(unittest.TestCase):
    def test_a_range_written_in_the_header_bounds_the_values(self):
        rng = np.random.default_rng(70)
        sound = np.r_[rng.uniform(40, 100, 500).round(), [0.0] * 6, [180.0, 245.0]]

        issues = checks(scan(pd.DataFrame({"Sound Level (dBA)\n(40 - 100)": sound})), "impossible")

        self.assertEqual(issues[0].rows, 8)
        self.assertIn("40 to 100", issues[0].rule)

    def test_a_stretch_may_pass_100_percent(self):
        self.assertEqual(name_bounds("Elongation (%)")[:2], (0.0, float("inf")))
        self.assertEqual(name_bounds("Viability %")[:2], (0.0, 100.0))

    def test_water_readings_are_bounded_in_a_water_table(self):
        rng = np.random.default_rng(71)
        frame = pd.DataFrame({
            "WaterTemp (C)": np.r_[rng.normal(20, 6, 498).round(1), [59.0, 74.0]],
            "pH": np.r_[rng.normal(7.2, 0.6, 499).round(1), [0.3]],
            "Salinity (ppt)": rng.gamma(2, 1, 500).round(2),
        })

        found = {issue.columns[0]: issue for issue in checks(scan(frame), "impossible")}

        self.assertEqual(found["WaterTemp (C)"].rows, 2)
        self.assertEqual(found["pH"].rows, 1)
        self.assertIn("natural surface water", found["pH"].rule)

    def test_a_ph_alone_is_not_read_as_natural_water(self):
        rng = np.random.default_rng(72)
        frame = pd.DataFrame({"pH": np.r_[rng.normal(7, 0.5, 199), [1.2]], "Yield": rng.gamma(3, 2, 200)})

        self.assertFalse(checks(scan(frame), "impossible"))

    def test_many_values_beyond_critical_limits_are_implausible(self):
        rng = np.random.default_rng(73)
        sodium = rng.normal(140, 4, 500).round(1)
        sodium[:100] = rng.uniform(165, 195, 100).round(1)
        frame = pd.DataFrame({"Na": sodium, "K": rng.normal(4.3, 0.4, 500).round(2)})

        issues = checks(scan(frame), "implausible")

        self.assertEqual(issues[0].columns, ("Na",))
        self.assertEqual(issues[0].rows, 100)
        self.assertFalse(checks(scan(frame.iloc[100:]), "implausible"))

    def test_a_lone_seven_among_broken_codes_is_a_category(self):
        histology = np.r_[[1] * 200, [2] * 40, [4] * 20, [5] * 10, [7]]
        likert = np.r_[np.repeat([1, 2, 3, 4, 5], 54), [7]]
        frame = pd.DataFrame({"histtype": histology, "answer": likert})

        issues = checks(scan(frame, roles_for(frame, factors=("histtype", "answer"))), "survey_code")

        self.assertEqual([issue.columns for issue in issues], [("answer",)])

    def test_a_rare_zero_below_codes_from_one_is_named(self):
        thal = np.r_[[0] * 2, [1] * 18, [2] * 166, [3] * 117]
        counts = np.r_[[0] * 150, [1] * 90, [2] * 50, [3] * 13]
        frame = pd.DataFrame({"thal": thal, "children": counts})

        issues = checks(scan(frame), "rare_code")

        self.assertEqual([issue.columns for issue in issues], [("thal",)])
        self.assertEqual(issues[0].rows, 2)


class LimitShapeTests(unittest.TestCase):
    def test_a_converted_blank_at_the_minimum_stands_apart(self):
        rng = np.random.default_rng(74)
        fahrenheit = rng.normal(68, 15, 2300).round()
        fahrenheit = fahrenheit[fahrenheit > 12]
        celsius = np.r_[(fahrenheit - 32) * 5 / 9, [(0 - 32) * 5 / 9] * 70]

        issues = checks(scan(pd.DataFrame({"AirTemp (C)": celsius})), "sentinel")

        self.assertEqual(issues[0].rows, 70)
        self.assertIn("0 °F", issues[0].detail)

    def test_a_round_maximum_hit_a_few_times_is_a_ceiling(self):
        rng = np.random.default_rng(75)
        index = rng.lognormal(4, 0.8, 2000).round()
        index = np.r_[index[index < 480], [500.0] * 5]

        issues = checks(scan(pd.DataFrame({"AQI": index})), "cap")

        self.assertEqual(issues[0].group, ("maximum", 500.0))

    def test_a_column_at_its_reporting_limit_is_almost_constant(self):
        phosphate = np.r_[[0.001] * 381, [0.002] * 20, [0.003] * 2, [0.005]]

        issues = checks(scan(pd.DataFrame({"phosphate": phosphate})), "cap")

        self.assertEqual(issues[0].rows, 381)
        self.assertIn("almost no information", issues[0].detail)

    def test_values_bunched_under_a_round_maximum_are_squeezed(self):
        rng = np.random.default_rng(76)
        potassium = np.r_[rng.uniform(25, 79.4, 830), rng.uniform(79.5, 79.99, 40), [80.0] * 5].round(3)

        issues = checks(scan(pd.DataFrame({"Potassium": potassium})), "cap")

        self.assertEqual(issues[0].group, ("near maximum", 80.0))
        self.assertGreaterEqual(issues[0].rows, 45)

    def test_columns_clipped_at_both_ends_are_one_issue(self):
        rng = np.random.default_rng(77)

        def winsorised(values):
            return np.clip(values, *np.percentile(values, [0.5, 99.5]))

        frame = pd.DataFrame({
            "weight": winsorised(rng.normal(80, 20, 5000).round(1)),
            "height": winsorised(rng.normal(170, 10, 5000).round(1)),
            "glucose": winsorised(rng.lognormal(4.9, 0.3, 5000).round()),
        })

        issues = scan(frame)

        clipped = checks(issues, "clipped")
        self.assertEqual(len(clipped), 1)
        self.assertEqual(len(clipped[0].columns), 3)
        self.assertFalse(checks(issues, "cap"))

    def test_a_limit_shared_by_many_columns_is_one_clip(self):
        rng = np.random.default_rng(78)
        frame = pd.DataFrame({f"gene_{i}": np.clip(rng.standard_t(3, 300) * 0.3, -2, 2) for i in range(40)})
        for i in range(40):
            frame.iloc[i, i] = 2.0

        issues = scan(frame)

        caps = checks(issues, "cap")
        self.assertEqual(len(caps), 1)
        self.assertEqual(caps[0].group[0], "shared")
        self.assertFalse(checks(issues, "extreme"))

    def test_a_round_pile_inside_the_range_within_one_group_is_a_cap(self):
        rng = np.random.default_rng(79)
        test = np.r_[["C"] * 300, ["T"] * 300]
        compression = np.r_[rng.uniform(1, 83, 240), [50.0] * 60]
        elongation = np.r_[compression, rng.uniform(1, 105, 297), [50.0] * 3].round(1)
        frame = pd.DataFrame({"Type of test": test, "Elongation (%)": elongation})

        issues = checks(scan(frame), "cap")

        self.assertEqual(issues[0].group[:2], ("within", 50.0))
        self.assertIn("C", issues[0].detail)

    def test_a_detached_handful_far_out_on_a_log_scale_is_extreme(self):
        rng = np.random.default_rng(80)
        tail = rng.lognormal(4.3, 0.6, 600)
        bulk = rng.lognormal(4.05, 0.08, 4000)
        sensor = np.r_[bulk, tail[tail < 600], rng.uniform(1800, 10300, 16)].round()

        issues = checks(scan(pd.DataFrame({"mq8": sensor})), "extreme")

        self.assertEqual(issues[0].rows, 16)

    def test_extreme_values_in_many_columns_are_listed_once(self):
        rng = np.random.default_rng(81)
        frame = pd.DataFrame({f"probe_{i}": rng.normal(0, 0.1, 300) for i in range(15)})
        for i in range(15):
            frame.iloc[i, i] = 50.0 + i

        issues = checks(scan(frame), "extreme")

        self.assertEqual(len(issues), 1)
        self.assertEqual(len(issues[0].columns), 15)


class SpeedTests(unittest.TestCase):
    def test_a_large_table_is_scanned_quickly(self):
        rng = np.random.default_rng(23)
        rows = 250_000
        data = {"Site": rng.choice([f"S{i}" for i in range(20)], rows)}
        for i in range(10):
            data[f"normal_{i}"] = rng.normal(50, 10, rows).round(2)
            data[f"skewed_{i}"] = rng.lognormal(0, 1, rows)
        for i in range(5):
            data[f"count_{i}"] = rng.poisson(3, rows).astype(float)
            data[f"gappy_{i}"] = np.where(rng.random(rows) < 0.3, np.nan, rng.normal(100, 5, rows))
        frame = pd.DataFrame(data)
        roles = roles_for(frame, "normal_0", "Site")

        started = time.perf_counter()
        scan(frame, roles)
        elapsed = time.perf_counter() - started

        # Well under a second on a laptop; the margin absorbs slow CI machines.
        self.assertLess(elapsed, 3.0)


if __name__ == "__main__":
    unittest.main()
