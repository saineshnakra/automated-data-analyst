import time
import unittest

import numpy as np
import pandas as pd

from ada.quality import (
    flagged_index,
    mask_sentinels,
    name_bounds,
    quality_frame,
    scan,
    show_value,
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

    def test_a_group_far_above_every_other_is_out_of_scale(self):
        rng = np.random.default_rng(15)
        frame = pd.DataFrame({
            "City": np.repeat(["A", "B", "C", "D"], 50),
            "CO": np.r_[rng.gamma(2, 5, 50), rng.gamma(2, 0.5, 150)],
        })

        issues = checks(scan(frame, roles_for(frame, "CO", "City")), "group_level")

        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].group, "A")

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


class MaskSentinelTests(unittest.TestCase):
    def test_a_clear_code_in_the_measure_becomes_missing_with_a_note(self):
        rng = np.random.default_rng(21)
        frame = pd.DataFrame({"Wind Speed": np.r_[rng.gamma(2, 1.5, 996), [999.9] * 4]})

        masked, note = mask_sentinels(frame, "Wind Speed")

        self.assertEqual(int(masked["Wind Speed"].isna().sum()), 4)
        self.assertIn("999.9", note)
        self.assertEqual(int(frame["Wind Speed"].isna().sum()), 0)

    def test_a_value_that_only_looks_like_a_code_stays(self):
        rng = np.random.default_rng(22)
        frame = pd.DataFrame({"Price": np.r_[rng.uniform(10, 3000, 999), [9999.0]]})

        masked, note = mask_sentinels(frame, "Price")

        self.assertIsNone(note)
        self.assertIs(masked, frame)


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
