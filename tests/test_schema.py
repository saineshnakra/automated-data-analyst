from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ada import analyze
from ada.schema import (
    IDENTIFIER_NAME_WORDS,
    MEASURE_KEYWORDS,
    add_row_summary,
    detect_roles,
    is_angle_column,
    is_design_name,
    is_entity_name,
    is_identifier_name,
    is_month_name,
    is_quality_flag,
    is_subject_name,
    is_time_part,
    is_year_name,
    looks_like_identifier,
    looks_like_subject,
    row_noun,
    row_summary_columns,
    running_total_columns,
)


class CountTableTests(unittest.TestCase):
    def test_a_table_of_counts_is_not_a_long_table_of_readings(self):
        rows = []
        for age in ("0-9", "10-19", "20-49", "50-64", "65+"):
            rows += [{"age": age, "result": "Died", "Freq": 10},
                     {"age": age, "result": "Injured", "Freq": 990}]
        analysis = analyze(pd.DataFrame(rows))
        self.assertEqual(len(analysis.frame), 10)
        self.assertEqual(set(analysis.frame["result"]), {"Died", "Injured"})


class IdentifierNameTests(unittest.TestCase):
    def test_the_last_word_decides(self):
        for name in ("Order No", "Customer ID", "Product Code", "Postal Code", "Store Number"):
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))

    def test_a_name_that_is_only_the_word_counts(self):
        for word in ("ID", "Code", "SKU", "UUID"):
            with self.subTest(word=word):
                self.assertTrue(is_identifier_name(word))

    def test_a_measure_that_merely_contains_the_word_does_not(self):
        # Only the last word counts, so "Identified Revenue" is a measure.
        for name in ("Identified Revenue", "Code Coverage", "Number of Orders", "Keyboard Sales"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))

    def test_an_ordinary_measure_is_not_a_key(self):
        for name in ("Revenue", "Units Sold", "Conversion Rate", "Cost"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))

    def test_case_and_separators_do_not_matter(self):
        for name in ("order_no", "ORDER NO", "Order-No", "  order   no  "):
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))

    def test_an_empty_name_is_not_a_key(self):
        for name in ("", "   ", "_"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))

    def test_the_word_list_is_reachable_for_callers_that_need_it(self):
        self.assertIn("id", IDENTIFIER_NAME_WORDS)
        self.assertIn("sku", IDENTIFIER_NAME_WORDS)
        self.assertNotIn("revenue", IDENTIFIER_NAME_WORDS)


class RoleDeterminismTests(unittest.TestCase):
    def test_the_event_date_wins_whatever_order_the_columns_arrive_in(self):
        months = pd.date_range("2024-01-01", periods=10, freq="MS")
        frame = pd.DataFrame(
            {"Order Date": months, "Ship Date": months + pd.Timedelta(days=3), "Revenue": range(10)}
        )

        self.assertEqual(detect_roles(frame).date, "Order Date")
        self.assertEqual(detect_roles(frame[["Ship Date", "Order Date", "Revenue"]]).date, "Order Date")

    def test_the_month_described_wins_over_the_month_of_release(self):
        months = pd.date_range("2023-01-01", periods=12, freq="MS")
        frame = pd.DataFrame({"year_month": months, "month_of_release": months + pd.DateOffset(months=3),
                              "estimate": range(12)})
        self.assertEqual(detect_roles(frame).date, "year_month")


class DesignColumnTests(unittest.TestCase):
    def test_a_replicate_number_is_not_picked_as_the_measure(self):
        frame = pd.DataFrame({
            "Treatment": ["Control", "Drug"] * 6,
            "Replicate": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "Yield": [4.1, 3.2, 4.0, 3.1, 4.3, 3.3, 3.9, 3.0, 4.2, 3.4, 4.0, 3.2],
        })
        self.assertEqual(detect_roles(frame).measure, "Yield")
        self.assertTrue(is_design_name("Replicate"))
        self.assertTrue(is_design_name("Plate"))
        self.assertFalse(is_design_name("Run Date"))


class PlateLayoutTests(unittest.TestCase):
    def setUp(self):
        rows = [letter for letter in "ABCDEFGH" for _ in range(12)]
        self.frame = pd.DataFrame({
            "Row": rows,
            "Column": list(range(1, 13)) * 8,
            "Sample": ["Drug X", "Drug Y", "Control", "Blank"] * 24,
            "Abs 450nm": [round(0.1 + 0.01 * index, 3) for index in range(96)],
        })

    def test_a_plate_row_is_not_the_grouping_when_a_sample_column_exists(self):
        self.assertEqual(detect_roles(self.frame).dimension, "Sample")

    def test_a_plate_row_is_still_used_when_nothing_else_qualifies(self):
        self.assertEqual(detect_roles(self.frame.drop(columns=["Sample"])).dimension, "Row")

    def test_layout_names_are_design_names(self):
        for name in ("Row", "Column", "col", "Well Position", "Plate Row"):
            with self.subTest(name=name):
                self.assertTrue(is_design_name(name))

    def test_a_position_on_its_own_is_not_a_layout(self):
        self.assertFalse(is_design_name("Position"))
        self.assertFalse(is_design_name("Player Position"))


class ResearchMeasureTests(unittest.TestCase):
    def test_an_outcome_beats_the_temperature_it_was_measured_at(self):
        # Readings below the detection limit leave the outcome less complete.
        frame = pd.DataFrame({
            "Site": ["River A", "River B"] * 6,
            "Concentration (mg/L)": [5.0, 0.7, None, 8.2, 4.1, None] * 2,
            "Temp (°C)": [11.7, 12.1, 12.5, 12.0, 11.9, 12.3] * 2,
        })
        self.assertEqual(detect_roles(frame).measure, "Concentration (mg/L)")

    def test_a_count_beats_air_temperature_even_without_fractions(self):
        frame = pd.DataFrame({
            "count": [0, 3, 7, 2, 5, 1] * 2,
            "air_temp_c": [10.7, 11.2, 15.0, 10.8, 13.1, 12.4] * 2,
            "humidity": [55.5, 60.1, 48.2, 52.0, 58.3, 61.7] * 2,
        })
        self.assertEqual(detect_roles(frame).measure, "count")

    def test_a_covariate_is_still_the_measure_when_it_is_all_there_is(self):
        frame = pd.DataFrame({"Station": ["N", "S"] * 6, "Wind Speed": [3.5, 4.1, 2.2] * 4})
        self.assertEqual(detect_roles(frame).measure, "Wind Speed")

    def test_coordinates_are_never_the_measure(self):
        frame = pd.DataFrame({
            "Latitude": [51.501, 51.502, 51.503] * 4,
            "lon": [-0.141, -0.142, -0.143] * 4,
            "Depth": [3, 5, 8] * 4,
        })
        self.assertEqual(detect_roles(frame).measure, "Depth")

    def test_research_outcome_words_score_below_money(self):
        for word in ("count", "abundance", "concentration", "yield"):
            with self.subTest(word=word):
                self.assertGreaterEqual(MEASURE_KEYWORDS[word], 5)
                self.assertLess(MEASURE_KEYWORDS[word], MEASURE_KEYWORDS["price"])


class EntityNameTests(unittest.TestCase):
    def test_the_unit_a_row_belongs_to_is_an_entity(self):
        for name in ("subject", "Participant", "Patient ID", "mouse", "Donor", "specimen_no"):
            with self.subTest(name=name):
                self.assertTrue(is_entity_name(name))

    def test_a_quantity_measured_on_the_unit_is_not(self):
        for name in ("Animal Weight", "Patient Age", "score", "Heart Rate"):
            with self.subTest(name=name):
                self.assertFalse(is_entity_name(name))

    def test_a_bare_no_is_nitric_oxide_not_an_entity(self):
        self.assertFalse(is_entity_name("NO"))
        for name in ("No.", "Specimen No", "Station No #"):
            with self.subTest(name=name):
                self.assertTrue(is_entity_name(name))


class SubjectColumnTests(unittest.TestCase):
    def test_names_of_subjects_measured_more_than_once(self):
        for name in ("subject#", "Subject", "Patient ID", "Mouse #", "specimen_no", "Station", "ID"):
            with self.subTest(name=name):
                self.assertTrue(is_subject_name(name))

    def test_time_points_counts_groups_and_contact_details_are_not_subjects(self):
        for name in ("Visit Number", "Run No", "Group ID", "Day #", "Number", "Code", "ZIP", "Phone", "NO"):
            with self.subTest(name=name):
                self.assertFalse(is_subject_name(name))

    def test_repeated_number_codes_name_subjects(self):
        codes = pd.Series(np.repeat(np.arange(1, 43), 5))
        self.assertTrue(looks_like_subject("subject#", codes))
        # A few people seen once still leave a subject column.
        self.assertTrue(looks_like_subject("subject#", pd.Series([*np.repeat([1, 2, 3, 4], 3), 5, 6])))

    def test_a_key_on_every_row_or_a_single_value_is_not_a_subject(self):
        self.assertFalse(looks_like_subject("Patient ID", pd.Series(range(1001, 1061))))
        self.assertFalse(looks_like_subject("Patient ID", pd.Series([*range(1001, 1059), 1001, 1002])))
        self.assertFalse(looks_like_subject("station", pd.Series(["Aotizhongxin"] * 50)))
        self.assertFalse(looks_like_subject("subject", pd.Series(np.repeat([1.5, 2.25, 3.75], 4))))

    def test_counts_ratings_and_years_repeat_but_are_not_subjects(self):
        rng = np.random.default_rng(3)
        for name, values in (
            ("children", rng.integers(0, 5, 60)),
            ("rating", rng.integers(1, 6, 60)),
            ("year", rng.integers(2015, 2020, 60)),
            ("Visit Number", np.tile([1, 2, 3], 20)),
            ("Number", rng.integers(1, 6, 60)),
        ):
            with self.subTest(name=name):
                self.assertFalse(looks_like_subject(name, pd.Series(values)))


class CalendarPartTests(unittest.TestCase):
    def test_a_month_name_needs_month_values(self):
        self.assertTrue(is_month_name("MO", pd.Series([1, 6, 12])))
        self.assertTrue(is_month_name("Month", pd.Series(["Jan", "February", "Mar"])))
        self.assertFalse(is_month_name("Age_months", pd.Series([12, 30, 66])))
        self.assertTrue(is_month_name("Age_months"))

    def test_a_year_name_needs_year_values(self):
        self.assertTrue(is_year_name("survey_yr", pd.Series([1999, 2004])))
        self.assertTrue(is_year_name("YR", pd.Series([40, 45])))
        self.assertFalse(is_year_name("Age_years", pd.Series([3, 41, 80])))
        self.assertFalse(is_year_name("Years", pd.Series([1, 12, 30])))

    def test_an_age_in_months_is_a_reading_not_a_part_of_the_date(self):
        frame = pd.DataFrame({"Age_months": [12 + index % 55 for index in range(220)]})

        self.assertFalse(is_time_part("Age_months", frame["Age_months"]))
        self.assertFalse(is_time_part("Duration Hours", pd.Series([3, 48, 200])))
        self.assertTrue(is_time_part("DA", pd.Series([1, 15, 31])))


class IdentifierNamePunctuationTests(unittest.TestCase):
    def test_key_names_with_punctuation_or_symbols(self):
        for name in ("Phone No.", "Account #", "Zipcode", "Postcode", "Telephone", "subject#"):
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))
        for name in ("Width", "Humidity", "Paid Amount"):
            with self.subTest(name=name):
                self.assertFalse(is_identifier_name(name))


def readings(rows: int, seed: int = 3, centre: float = 50.0) -> np.ndarray:
    return np.round(np.random.default_rng(seed).normal(centre, 10, rows), 2)


class FactorTests(unittest.TestCase):
    def setUp(self):
        rows = 120
        self.frame = pd.DataFrame({
            "gender": [0, 1] * (rows // 2),
            "race": [1, 2, 3, 4, 6, 7] * (rows // 6),
            "Status": [0, 1, 2] * (rows // 3),
            "bmi": readings(rows),
            "weight_kg": readings(rows, seed=4, centre=80),
        })

    def test_a_reading_in_whole_numbers_stays_a_quantity(self):
        frame = self.frame.assign(Humidity=[60 + n % 11 for n in range(len(self.frame))])

        self.assertNotIn("Humidity", detect_roles(frame).factors)

    def test_flags_codes_and_labels_are_factors_and_groupings(self):
        roles = detect_roles(self.frame)

        self.assertEqual(roles.factors, ("gender", "race", "Status"))
        self.assertEqual(roles.flags, ("gender",))
        for column in roles.factors:
            self.assertIn(column, roles.dimensions)

    def test_a_factor_is_never_the_measure_and_comes_after_the_quantities(self):
        roles = detect_roles(self.frame)

        self.assertIn(roles.measure, ("bmi", "weight_kg"))
        self.assertEqual(roles.numeric[:2], ("bmi", "weight_kg"))

    def test_a_constant_column_is_listed_after_the_quantities(self):
        frame = self.frame.assign(attention_check=6)[["attention_check", "bmi", "gender"]]

        self.assertEqual(detect_roles(frame).numeric, ("bmi", "attention_check", "gender"))

    def test_a_label_name_can_be_the_grouping(self):
        self.assertEqual(detect_roles(self.frame).dimension, "Status")

    def test_the_notes_name_the_categories_and_give_flag_shares(self):
        notes = detect_roles(self.frame).notes
        note = next(note for note in notes if note.startswith("Read as categories"))

        self.assertIn("gender, race, Status", note)
        self.assertIn("gender 50.0%", note)

    def test_counts_with_few_values_stay_quantities(self):
        frame = pd.DataFrame({
            "cylinders": [4, 6, 8] * 40,
            "Number of Children": [0, 1, 2, 3] * 30,
            "household_size": [1, 2, 3, 4, 5, 6] * 20,
            "mpg": readings(120),
        })

        self.assertEqual(detect_roles(frame).factors, ())

    def test_a_count_named_flag_is_still_a_flag(self):
        frame = pd.DataFrame({"Reopened Count": [0, 1, 0, 0] * 10, "Hours": readings(40)})

        self.assertEqual(detect_roles(frame).flags, ("Reopened Count",))

    def test_a_station_number_with_many_levels_is_a_label(self):
        frame = pd.DataFrame({"Station": list(range(10001, 10041)) * 5, "Rain": readings(200)})

        self.assertEqual(detect_roles(frame).factors, ("Station",))

    def test_a_set_concentration_crossed_with_a_dye_is_a_design_level(self):
        frame = pd.DataFrame({
            "Dye": [dye for dye in ("Red", "Blue", "Green") for _ in range(40)],
            "Concentration": [0.5, 1.0, 2.5, 5.0] * 30,
            "R": readings(120),
            "B": readings(120, seed=5),
        })

        roles = detect_roles(frame)

        self.assertIn("Concentration", roles.factors)
        self.assertIn(roles.measure, ("R", "B"))

    def test_numbered_rating_items_stay_quantities(self):
        rng = np.random.default_rng(21)
        frame = pd.DataFrame({
            "condition": ["A", "B", "C", "D"] * 30,
            "fsi_high": [0, 1] * 60,
            "RDB1": rng.integers(0, 11, 120),
            "RDB2": rng.integers(0, 11, 120),
            "Realism_1": rng.integers(2, 8, 120),
            "Realism_2": rng.integers(2, 8, 120),
            "risky_decision": readings(120, centre=5),
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.factors, ("fsi_high",))
        for column in ("RDB1", "RDB2", "Realism_1", "Realism_2"):
            with self.subTest(column=column):
                self.assertNotIn(column, roles.dimensions)

    def test_columns_sharing_a_rating_scale_are_items_but_a_lone_code_is_not(self):
        rng = np.random.default_rng(22)
        frame = pd.DataFrame({
            "success_likelihood": rng.integers(1, 8, 120),
            "direct_riskiness": rng.integers(1, 8, 120),
            "perceived_harm": rng.integers(1, 8, 120),
            "marital": rng.integers(1, 7, 120),
            "bmi": readings(120),
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.factors, ("marital",))

    def test_counts_beside_a_count_that_passes_one_are_not_flags(self):
        def counts(ones: int) -> list[float]:
            return [1.0] * ones + [0.0] * (110 - ones) + [np.nan] * 10

        frame = pd.DataFrame({
            "Location_m": [step * 5 for step in range(10)] * 12,
            "Nudibranchs": counts(2),
            "Crabs": [75.0, 36.0, 3.0, 2.0] + [1.0] * 8 + [0.0] * 98 + [np.nan] * 10,
            "Octopuses": counts(3),
            "Side": ["Left", "Right"] * 60,
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.flags, ())
        self.assertNotIn("Nudibranchs", roles.dimensions)
        self.assertEqual(roles.measure, "Crabs")

    def test_two_yes_no_columns_side_by_side_stay_flags(self):
        frame = pd.DataFrame({
            "Antibiotics": [0, 1, 0] * 40,
            "Probiotics": [1, 0] * 60,
            "Shannon_Index": readings(120, centre=3),
        })

        self.assertEqual(detect_roles(frame).flags, ("Antibiotics", "Probiotics"))

    def test_a_swept_setting_is_a_design_factor_not_a_measure(self):
        rng = np.random.default_rng(23)
        frequency = np.tile(np.arange(100, 600, 10), 12).astype(float)
        frame = pd.DataFrame({
            "Frequency (GHz)": frequency,
            "Sample": np.repeat(list("ABCDEFGHIJKL"), 50),
            "HG (mV)": np.round(frequency / 3 + rng.normal(0, 5, frequency.size), 2),
            "LG (mV)": np.round(frequency / 9 + rng.normal(0, 5, frequency.size), 2),
        })

        roles = detect_roles(frame)

        self.assertIn("Frequency (GHz)", roles.factors)
        self.assertNotIn("Frequency (GHz)", (roles.measure, *roles.measure_alternatives))

    def test_a_reading_rounded_to_whole_numbers_is_not_a_sweep(self):
        frame = pd.DataFrame({"Humidity": [40 + index % 41 for index in range(410)], "Rain": readings(410)})

        self.assertEqual(detect_roles(frame).factors, ())

    def test_a_short_scale_is_the_quantity_when_nothing_else_is_measured(self):
        frame = pd.DataFrame({"Class": ["A", "B"] * 30, "Mark": [3, 5, 7, 9, 10, 6] * 10})

        roles = detect_roles(frame)

        self.assertEqual(roles.factors, ())
        self.assertEqual(roles.measure, "Mark")


class IdentifierContentTests(unittest.TestCase):
    def test_unique_whole_numbers_in_uneven_steps_down_the_first_column_are_a_key(self):
        frame = pd.DataFrame({
            "respondent": [62161 + 2 * index + index % 2 for index in range(40)],
            "bmi": readings(40),
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.identifier, "respondent")
        self.assertNotIn("respondent", roles.numeric)
        self.assertTrue(any("different whole number on every row" in note for note in roles.notes))

    def test_a_row_key_name_is_enough_with_unique_whole_numbers(self):
        frame = pd.DataFrame({"HG": readings(30), "index_original": range(500, 530)})

        self.assertEqual(detect_roles(frame).identifier, "index_original")

    def test_an_evenly_stepped_sweep_is_not_a_key(self):
        roles = detect_roles(pd.DataFrame({"Frequency": range(100, 500, 10), "Signal": readings(40)}))

        self.assertIsNone(roles.identifier)
        self.assertIn("Frequency", roles.numeric)

    def test_a_unique_text_column_is_the_row_label(self):
        frame = pd.DataFrame({
            "Taxonomy": [f"g__Genus{index};s__species {index}" for index in range(30)],
            "Abundance": readings(30),
        })

        self.assertEqual(detect_roles(frame).identifier, "Taxonomy")

    def test_sentences_are_not_a_row_label(self):
        frame = pd.DataFrame({
            "Comment": [f"sample {index} looked cloudy after the second wash" for index in range(30)],
            "Turbidity": readings(30),
        })

        self.assertIsNone(detect_roles(frame).identifier)

    def test_a_bare_no_is_not_a_key_name_but_a_marked_one_is(self):
        self.assertFalse(is_identifier_name("NO"))
        self.assertFalse(is_identifier_name("No"))
        for name in ("No.", "No #", "Order No", "determinand.notation"):
            with self.subTest(name=name):
                self.assertTrue(is_identifier_name(name))

    def test_decimals_under_a_key_name_are_a_measurement(self):
        decimals = pd.Series([2.31e5 + index / 7 for index in range(20)])

        self.assertFalse(looks_like_identifier("Reynolds Number", decimals))
        self.assertTrue(looks_like_identifier("Order Number", pd.Series(range(1000, 1020))))

    def test_a_number_of_things_counts_them_however_many_counts_differ(self):
        counts = pd.Series([5352, 163446, 70596, 37750, 3602, 30000])

        for name in ("Number of residents", "Count of visits", "Total number of births"):
            with self.subTest(name=name):
                self.assertFalse(looks_like_identifier(name, counts))
        frame = pd.DataFrame({"Kind": ["All", "Home"] * 3, "Number of residents": counts})
        self.assertIsNone(detect_roles(frame).identifier)


class IdentifierByContentTests(unittest.TestCase):
    def test_a_barcode_unique_on_every_row_is_an_identifier_not_a_reading(self):
        frame = pd.DataFrame({
            "ID": [str(index) for index in range(60)],
            "barcode": np.arange(4256, 4316),
            "esr1": readings(60, centre=0),
        })

        roles = detect_roles(frame)

        self.assertNotIn("barcode", roles.numeric)

    def test_codes_under_a_date_like_header_label_the_rows(self):
        frame = pd.DataFrame({
            "time": [f"YAL{index:03d}C" for index in range(40)],
            "expression": readings(40, centre=0),
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.identifier, "time")
        self.assertTrue(any("despite its name" in note for note in roles.notes))

    def test_unique_times_written_as_text_are_not_a_row_label(self):
        frame = pd.DataFrame({"time": [f"10:{index:02d}" for index in range(40)], "Level": readings(40)})

        self.assertIsNone(detect_roles(frame).identifier)


class MeasureTieTests(unittest.TestCase):
    def test_a_tie_goes_to_the_column_with_more_information(self):
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"OBJECT": rng.integers(0, 60, 300), "TIME": rng.integers(0, 3600, 300)})
        frame.loc[:200, "OBJECT"] = 0

        self.assertEqual(detect_roles(frame).measure, "TIME")

    def test_a_mostly_zero_column_loses_to_temperatures(self):
        frame = pd.DataFrame({
            "MaxTemp": readings(200, centre=25),
            "MinTemp": readings(200, seed=6, centre=12),
            "SNF": [0.0] * 196 + [0.1, 0.4, 1.2, 0.3],
        })

        self.assertIn(detect_roles(frame).measure, ("MaxTemp", "MinTemp"))

    def test_the_one_column_in_a_different_unit_is_the_outcome(self):
        frame = pd.DataFrame({
            "Cement (kg/m3)": readings(60, centre=300),
            "Blast-furnace Slag (kg/m3)": readings(60, seed=7, centre=80),
            "Fly Ash (kg/m3)": readings(60, seed=8, centre=60),
            "Coarse Aggregate (kg/m3)": readings(60, seed=9, centre=950),
            "Concrete Hardness (MPa)": readings(60, seed=10, centre=35),
        })

        self.assertEqual(detect_roles(frame).measure, "Concrete Hardness (MPa)")

    def test_a_manipulation_check_gives_way_to_the_outcome(self):
        frame = pd.DataFrame({
            "realism": readings(50, centre=5),
            "risky_decision": readings(50, seed=11, centre=5),
            "risk_perception": readings(50, seed=12, centre=4),
        })

        self.assertEqual(detect_roles(frame).measure, "risky_decision")

    def test_a_close_call_records_the_runners_up(self):
        frame = pd.DataFrame({
            name: readings(80, seed=seed, centre=centre)
            for name, seed, centre in (
                ("HGB", 13, 13), ("HCT", 14, 40), ("MCV", 15, 90), ("PLT", 16, 250), ("WBC", 17, 7),
            )
        })

        roles = detect_roles(frame)

        self.assertEqual(len(roles.measure_alternatives), 3)
        self.assertNotIn(roles.measure, roles.measure_alternatives)

    def test_a_clear_pick_has_no_runners_up(self):
        frame = pd.DataFrame({"Revenue": readings(40, centre=500), "Visits": readings(40, seed=2)})

        roles = detect_roles(frame)

        self.assertEqual(roles.measure, "Revenue")
        self.assertEqual(roles.measure_alternatives, ())


class MeasureChoiceTests(unittest.TestCase):
    def test_a_date_stored_as_a_number_is_never_the_measure(self):
        days = pd.date_range("1954-06-01", periods=60, freq="D")
        frame = pd.DataFrame({
            "MESS_DATUM": days.strftime("%Y%m%d").astype(int),
            "TMK": readings(60, centre=10),
            "RSKF": [0, 6, 7] * 20,
        })

        self.assertEqual(detect_roles(frame).measure, "TMK")

    def test_a_constant_column_is_never_the_measure(self):
        frame = pd.DataFrame({
            "Year": pd.date_range("2005-01-01", periods=30, freq="YS"),
            "Farmland Type": ["dry", "paddy", "dry"] * 10,
            "Multiple Cropping Index (%)": [100] * 30,
        })

        roles = detect_roles(frame)

        self.assertIsNone(roles.measure)
        self.assertEqual(roles.measure_alternatives, ())

    def test_a_statistic_of_another_column_gives_way_to_a_reading(self):
        frame = pd.DataFrame({
            "p_value": np.linspace(0.001, 0.9, 40),
            "beta_continuous": readings(40, seed=5, centre=0),
            "overall_mean": readings(40, centre=100),
        })

        self.assertEqual(detect_roles(frame).measure, "overall_mean")

    def test_a_column_recorded_on_few_rows_gives_way(self):
        sparse = [np.nan] * 36 + [1.5, 2.5, 3.5, 4.5]
        frame = pd.DataFrame({"Concentration": sparse, "Turbidity": readings(40)})

        self.assertEqual(detect_roles(frame).measure, "Turbidity")

    def test_a_summary_statistics_table_is_measured_by_its_mean(self):
        mean = readings(50, centre=7)
        frame = pd.DataFrame({
            "probe": [f"{index}_at" for index in range(50)],
            "count": 46,
            "mean": mean,
            "std": np.abs(readings(50, seed=8, centre=1)),
            "min": mean - 2,
            "25%": mean - 1,
            "50%": mean,
            "75%": mean + 1,
            "max": mean + 2,
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.measure, "mean")
        note = next(note for note in roles.notes if note.startswith("Each row summarises"))
        self.assertIn("count is 46 on every row", note)

    def test_an_outcome_score_beats_a_body_weight(self):
        frame = pd.DataFrame({
            "Weight_kg": readings(80, centre=400),
            "Farm_Size_Acres": readings(80, seed=6, centre=50),
            "Dysbiosis_Risk": np.round(np.random.default_rng(9).random(80), 2),
        })

        self.assertEqual(detect_roles(frame).measure, "Dysbiosis_Risk")

    def test_the_reading_that_best_separates_an_outcome_label_wins_a_tie(self):
        rng = np.random.default_rng(10)
        result = np.array(["positive", "negative"] * 60)
        positive = result == "positive"
        frame = pd.DataFrame({
            "Result": result,
            "CK-MB": np.round(rng.lognormal(1 + 0.2 * positive, 1), 2),
            "Troponin": np.round(rng.lognormal(-4 + 2 * positive, 1), 3),
            "Blood sugar": readings(120, centre=140),
        })

        self.assertEqual(detect_roles(frame).measure, "Troponin")


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.label = np.array([0, 1] * 60)
        self.separating = np.round(rng.normal(100 + 25 * self.label, 10), 1)
        self.noise = np.round(rng.normal(25, 4, 120), 2)

    def test_a_column_named_as_the_label_is_the_grouping_and_the_measure_tells_it_apart(self):
        frame = pd.DataFrame({
            "body_mass_index": self.noise,
            "glucose": self.separating,
            "sex": [0, 0, 1] * 40,
            "target": self.label,
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.dimension, "target")
        self.assertEqual(roles.measure, "glucose")
        self.assertTrue(any("target names the label" in note for note in roles.notes))

    def test_the_column_a_file_is_named_for_is_the_outcome(self):
        frame = pd.DataFrame({"Rainfall": self.noise, "Slope": self.separating, "Landslide": self.label})

        named = detect_roles(frame, "landslide_dataset.csv")

        self.assertEqual(named.dimension, "Landslide")
        self.assertEqual(named.measure, "Slope")
        self.assertTrue(any("what the file is named for" in note for note in named.notes))
        self.assertFalse(any("what the file is named for" in note for note in detect_roles(frame).notes))

    def test_a_flag_filed_first_after_the_identifier_is_the_outcome(self):
        frame = pd.DataFrame({
            "id": [f"P{index}" for index in range(120)],
            "fasting_ability": self.label,
            "gender": ["F", "M", "M"] * 40,
            "body_mass_index": self.noise,
            "glucose": self.separating,
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.dimension, "fasting_ability")
        self.assertEqual(roles.measure, "glucose")

    def test_a_reading_the_outcome_was_cut_from_is_not_the_measure(self):
        frame = pd.DataFrame({
            "Quake": np.where(self.label == 1, 4 + self.noise / 10, 3.9 - self.noise / 10),
            "Slope": self.separating,
            "Rainfall": self.noise,
            "Landslide": self.label,
        })

        roles = detect_roles(frame, "landslide.csv")

        self.assertEqual(roles.measure, "Slope")
        self.assertTrue(any("Landslide is set by Quake" in note for note in roles.notes))

    def test_follow_up_time_is_not_compared_across_its_own_event(self):
        frame = pd.DataFrame({
            "eventdeath": self.label,
            "survival": np.round(10 - 5 * self.label + self.noise / 10, 2),
            "esr1": self.separating,
            "diam": self.noise,
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.dimension, "eventdeath")
        self.assertNotEqual(roles.measure, "survival")
        self.assertTrue(any("survival is follow-up time" in note for note in roles.notes))

    def test_a_sex_code_filed_first_is_not_an_outcome(self):
        frame = pd.DataFrame({"sex": self.label, "weight": self.noise, "glucose": self.separating})

        self.assertNotIn("names the label", " ".join(detect_roles(frame).notes))
        self.assertFalse(any("reads as the label" in note for note in detect_roles(frame).notes))


class QuantityNotCodeTests(unittest.TestCase):
    def test_a_reading_in_a_unit_stays_a_reading_however_few_its_values(self):
        frame = pd.DataFrame({
            "O content (wppm)": np.repeat([100, 250, 400, 700, 900, 1200, 1500, 2000, 2500, 3000], 6),
            "YS (MPa)": readings(60, centre=900),
        })

        self.assertNotIn("O content (wppm)", detect_roles(frame).factors)

    def test_a_reading_rounded_to_a_coarse_grid_is_a_quantity(self):
        rng = np.random.default_rng(16)
        frame = pd.DataFrame({
            "bp": rng.choice([50, 60, 70, 80, 90, 100, 110, 120, 140, 180], 300, p=[
                0.02, 0.12, 0.3, 0.3, 0.12, 0.08, 0.02, 0.02, 0.01, 0.01,
            ]),
            "al": rng.integers(0, 6, 300),
            "hemo": readings(300, centre=12),
        })

        roles = detect_roles(frame)

        self.assertNotIn("bp", roles.factors)
        self.assertIn("al", roles.factors)

    def test_a_reading_filled_on_a_few_rows_is_not_a_category(self):
        values = [np.nan] * 170 + [193.0, 199.0, 212.0, 307.0, 523.0] * 6
        frame = pd.DataFrame({"al": values, "COND": readings(200, centre=300)})

        self.assertNotIn("al", detect_roles(frame).factors)


class ExperimentUnitTests(unittest.TestCase):
    def test_a_numbered_run_of_rows_under_fixed_settings_is_a_unit_not_a_reading(self):
        rng = np.random.default_rng(4)
        cases = np.repeat(np.arange(1, 9), 10)
        frame = pd.DataFrame({
            "case": cases,
            "run": np.tile(np.arange(1, 11), 8),
            "VB": np.round(np.tile(np.linspace(0, 1, 10), 8) + rng.normal(0, 0.02, 80), 3),
            "feed": np.where(cases % 2 == 0, 0.5, 0.25),
            "material": np.where(cases <= 4, 1, 2),
            "vibration": np.round(rng.normal(0.3, 0.05, 80), 4),
            "current": np.round(rng.normal(0.6, 0.2, 80), 4),
        })

        roles = detect_roles(frame)

        self.assertIn("case", roles.factors)
        self.assertIn("feed", roles.factors)
        self.assertEqual(roles.measure, "VB")
        self.assertTrue(any("case numbers blocks of consecutive rows" in note for note in roles.notes))

    def test_a_reading_sorted_down_the_file_is_still_a_reading(self):
        frame = pd.DataFrame({
            "Revenue": np.repeat([100.0, 250.0, 400.0, 900.0, 1200.0, 1500.0], 5),
            "Region": np.repeat(["N", "S", "E", "W", "C", "X"], 5),
        })

        self.assertEqual(detect_roles(frame).measure, "Revenue")


class MeasureFamilyTests(unittest.TestCase):
    def test_total_among_lab_readings_is_not_a_money_word(self):
        rng = np.random.default_rng(8)
        frame = pd.DataFrame({
            "secchi": np.round(rng.normal(6, 2, 60), 1),
            "calcium": np.round(rng.normal(20, 3, 60), 2),
            "nitrates_total": np.round(rng.normal(0.2, 0.05, 60), 3),
            "phosphorus_total": np.round(rng.normal(4, 1, 60), 2),
        })

        self.assertNotIn(detect_roles(frame).measure, ("nitrates_total", "phosphorus_total"))
        self.assertEqual(detect_roles(frame, "georgian_bay_secchi.csv").measure, "secchi")

    def test_total_beside_money_still_scores(self):
        frame = pd.DataFrame({
            "Unit Price": readings(40, centre=20),
            "Order Total": readings(40, seed=5, centre=90),
        })

        self.assertEqual(detect_roles(frame).measure, "Order Total")

    def test_the_lone_column_outside_a_family_of_descriptors_is_the_measure(self):
        rng = np.random.default_rng(12)
        columns = {"Hardness": np.round(rng.uniform(1, 10, 80), 1)}
        for stem in ("allelectrons", "density", "val_e", "ionenergy", "zaratio"):
            columns[f"{stem}_Average"] = np.round(rng.normal(10, 2, 80), 4)
        columns["density_Total"] = np.round(rng.normal(40, 9, 80), 4)

        self.assertEqual(detect_roles(pd.DataFrame(columns)).measure, "Hardness")

    def test_the_mean_of_the_other_columns_is_the_measure(self):
        rng = np.random.default_rng(13)
        frame = pd.DataFrame({
            name: np.round(rng.normal(27, 0.7, 60), 2) for name in ("D-J-F", "M-A-M", "J-J-A", "S-O-N")
        })
        frame["metANN"] = frame.mean(axis=1).round(2)

        self.assertEqual(detect_roles(frame).measure, "metANN")

    def test_the_average_of_a_reading_beats_its_extremes(self):
        rng = np.random.default_rng(14)
        average = rng.uniform(5, 100, 90)
        frame = pd.DataFrame({
            "pollutant_min": np.round(average * 0.5, 0),
            "pollutant_max": np.round(average * 1.8, 0),
            "pollutant_avg": np.round(average, 0),
        })

        self.assertEqual(detect_roles(frame).measure, "pollutant_avg")

    def test_a_reading_in_a_unit_of_its_own_beats_unlabelled_link_figures(self):
        rng = np.random.default_rng(15)
        frame = pd.DataFrame({
            "RSSI": rng.integers(-90, -50, 200),
            "SNR": np.round(rng.normal(8, 2, 200), 2),
            "Sound Level (dBA) (40 - 100)": rng.integers(40, 90, 200),
        })

        self.assertEqual(detect_roles(frame).measure, "Sound Level (dBA) (40 - 100)")


class GroupingRuleTests(unittest.TestCase):
    def test_many_districts_with_several_samples_each_are_a_grouping(self):
        frame = pd.DataFrame({
            "District": [f"District {index % 101}" for index in range(606)],
            "Potassium value": readings(606, centre=60),
        })

        self.assertEqual(detect_roles(frame).dimension, "District")

    def test_names_that_each_appear_once_or_twice_are_not(self):
        frame = pd.DataFrame({
            "Name": [f"Person {index // 2}" for index in range(400)],
            "Score": readings(400),
        })

        self.assertNotIn("Name", detect_roles(frame).dimensions)

    def test_a_grouping_that_shares_no_rows_with_the_measure_is_not_picked(self):
        frame = pd.DataFrame({
            "food_category": ["Fish", "Salt", "Water"] * 10 + [None] * 10,
            "biomarker": [None] * 30 + ["CRP", "aPTT"] * 5,
            "overall_mean": [np.nan] * 30 + list(readings(10)),
        })

        self.assertEqual(detect_roles(frame).dimension, "biomarker")

    def test_the_more_complete_of_two_twin_groupings_is_preferred(self):
        types = ["LUAD", "BRCA", "SCLC", "COAD"] * 30
        frame = pd.DataFrame({
            "TCGA_DESC": types,
            "Cancer Type": [None if index % 5 == 0 else kind for index, kind in enumerate(types)],
            "LN_IC50": readings(120, centre=2),
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.dimension, "TCGA_DESC")
        self.assertTrue(any(note.startswith("TCGA_DESC holds the same groups") for note in roles.notes))

    def test_text_labels_beat_a_numeric_code_for_the_same_groups(self):
        cells = ["T0", "T1", "S0", "S1", "D0", "D1"]
        frame = pd.DataFrame({
            "Group": [index % 6 + 1 for index in range(48)],
            "Gp": [cells[index % 6] for index in range(48)],
            "Ca": readings(48, centre=9),
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.dimension, "Gp")
        self.assertTrue(any("ADA groups by Gp" in note for note in roles.notes))

    def test_two_columns_that_split_the_rows_alike_are_named(self):
        frame = pd.DataFrame({
            "side": ["Leeward", "Windward", "Bajos"] * 20,
            "protection_status": ["NTZ", "MPA", "OA"] * 20,
            "region": ["Alacranes", "Alacranes", "Bajos"] * 20,
            "quantity": [1, 4, 2, 8, 3] * 12,
        })

        notes = detect_roles(frame).notes

        self.assertTrue(any("side and protection_status" in note for note in notes))

    def test_a_swept_setting_that_explains_the_reading_is_the_grouping(self):
        rng = np.random.default_rng(24)
        frequency = np.tile(np.arange(100, 600, 10), 48).astype(float)
        sample = np.repeat(list("ABCDEFGHIJKL"), 200)
        frame = pd.DataFrame({
            "Frequency (GHz)": frequency,
            "Sample_original": np.repeat([f"{letter}{n}" for letter in "ABCDEFGHIJKL" for n in range(4)], 50),
            "Sample": sample,
            "HG (mV)": np.round(np.sin(frequency / 40) * 50 + rng.normal(0, 10, frequency.size), 2),
        })

        self.assertEqual(detect_roles(frame).dimension, "Frequency (GHz)")

    def test_when_no_grouping_explains_the_measure_the_handier_one_stays(self):
        frame = pd.DataFrame({
            "Location_m": [step * 5 for step in range(10)] * 12,
            "Side": ["Left"] * 60 + ["Right"] * 60,
            "Crabs": [0, 0, 1, 0, 3, 0, 0, 2, 0, 0, 0, 1] * 10,
        })

        self.assertEqual(detect_roles(frame).dimension, "Location_m")

    def test_a_date_that_never_changes_is_not_the_date(self):
        frame = pd.DataFrame({
            "last_updated": pd.Timestamp("2025-02-01"),
            "Revenue": readings(30, centre=100),
            "Region": ["N", "S", "E"] * 10,
        })

        self.assertIsNone(detect_roles(frame).date)


class RunningTotalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        interval = np.where(rng.random(240) < 0.15, rng.gamma(2.0, 1.5, 240).round(2), 0.0)
        total = np.cumsum(interval)
        total[120:] -= total[119]  # the gauge is emptied half-way
        self.frame = pd.DataFrame({
            "Measurement Timestamp": pd.date_range("2024-05-01", periods=240, freq="h"),
            "Interval Rain": interval,
            "Total Rain": total.round(2),
            "Air Temperature": readings(240, centre=18),
        })

    def test_a_running_total_is_never_the_measure_and_its_interval_column_is(self):
        roles = detect_roles(self.frame)

        self.assertEqual(roles.measure, "Interval Rain")
        note = next(note for note in roles.notes if note.startswith("Total Rain"))
        self.assertIn("Interval Rain", note)

    def test_rain_that_falls_back_to_zero_each_hour_is_not_a_running_total(self):
        roles = detect_roles(self.frame.drop(columns=["Total Rain"]))

        self.assertFalse(any("keeps climbing" in note for note in roles.notes))

    def test_a_counter_that_steps_up_by_one_is_never_the_measure_even_without_a_reset(self):
        rng = np.random.default_rng(5)
        packets = 48_991 + np.arange(300) + np.cumsum(rng.random(300) < 0.05)
        packets[150] = packets[149] - 7
        frame = pd.DataFrame({
            "Time": pd.date_range("2024-05-01", periods=300, freq="5min"),
            "Packet Count": packets,
            "Sound Level (dBA)": rng.integers(40, 90, 300),
        })

        roles = detect_roles(frame)

        self.assertEqual(roles.measure, "Sound Level (dBA)")
        self.assertIn("Packet Count", roles.non_readings)
        self.assertTrue(any(note.startswith("Packet Count keeps climbing") for note in roles.notes))

    def test_days_counted_within_each_subject_are_a_time_axis_not_a_counter(self):
        rng = np.random.default_rng(6)
        frame = pd.DataFrame({
            "Subject": np.repeat(np.arange(308, 326), 10),
            "Days": np.tile(np.arange(10), 18),
            "Reaction": 250 + 10 * np.tile(np.arange(10), 18) + rng.normal(0, 25, 180),
        })

        roles = detect_roles(frame)

        self.assertNotIn("Days", running_total_columns(roles))
        self.assertFalse(any(note.startswith("Days keeps climbing") for note in roles.notes))


class LongTableTests(unittest.TestCase):
    def setUp(self):
        parameters = {"Ammonia": ("mg/l", 0.2), "pH": ("phunits", 7.5), "Conductivity": ("us/cm", 450.0)}
        self.frame = pd.DataFrame([
            {"site": f"Site {site}", "determinand.label": parameter,
             "result": level * (1 + 0.05 * repeat), "determinand.unit.label": unit}
            for site in range(12)
            for parameter, (unit, level) in parameters.items()
            for repeat in range(3 if parameter == "Ammonia" else 2)
        ])

    def test_the_parameter_is_the_grouping_and_the_commonest_one_the_focus(self):
        roles = detect_roles(self.frame)

        self.assertEqual(roles.table_shape, "long")
        self.assertEqual(roles.dimension, "determinand.label")
        self.assertEqual(roles.focus, "Ammonia")
        note = next(note for note in roles.notes if note.startswith("Each row holds"))
        self.assertIn("3 different units", note)

    def test_the_units_of_another_column_do_not_split_the_table(self):
        # Storm damage written "1.50K" and "2.00M" gets a column of its units; the measure is the duration.
        frame = pd.DataFrame({
            "event": [f"E{index}" for index in range(40)],
            "duration_hours": [float(index % 7 + 1) for index in range(40)],
            "damage": ["1.50K", "2.00K", "0.50M", "3.00K", None] * 8,
        })
        analysis = analyze(frame)
        self.assertIn("damage unit", analysis.frame.columns)
        self.assertEqual(analysis.roles.table_shape, "flat")
        self.assertEqual(len(analysis.frame), 40)

    def test_values_on_very_different_scales_make_a_long_table_without_units(self):
        roles = detect_roles(self.frame.drop(columns=["determinand.unit.label"]))

        self.assertEqual(roles.focus, "Ammonia")

    def test_a_parameter_never_named_beside_a_value_is_no_long_table(self):
        # Parameters and values on different rows, as when a file stacks two tables.
        frame = self.frame.copy()
        stacked = frame.index % 2 == 0
        frame.loc[stacked, "result"] = None
        frame.loc[~stacked, ["determinand.label", "determinand.unit.label"]] = None

        roles = detect_roles(frame)

        self.assertEqual(roles.table_shape, "flat")
        self.assertIsNone(roles.focus)

    def test_an_ab_test_group_is_not_a_parameter(self):
        frame = pd.DataFrame({"Test Group": ["A", "B"] * 30, "Revenue": readings(60, centre=100)})

        self.assertEqual(detect_roles(frame).table_shape, "flat")

    def test_a_parameter_column_with_an_unexpected_name_is_found_by_how_its_rows_behave(self):
        levels = {"OZONE": 40.0, "PM10": 100.0, "NH3": 6.0, "SO2": 12.0}
        rng = np.random.default_rng(17)
        frame = pd.DataFrame([
            {"state": f"State {station % 5}", "station": f"Station {station}", "pollutant_id": pollutant,
             "pollutant_avg": round(level * rng.uniform(0.7, 1.3), 1)}
            for station in range(30)
            for pollutant, level in levels.items()
        ])

        roles = detect_roles(frame)

        self.assertEqual(roles.table_shape, "long")
        self.assertEqual(roles.dimension, "pollutant_id")

    def test_revenue_by_store_and_category_is_not_split_by_category(self):
        levels = {"Furniture": 4000.0, "Books": 40.0, "Toys": 300.0}
        frame = pd.DataFrame([
            {"Store": f"Store {store}", "Category": category, "Revenue": level * (1 + store / 50)}
            for store in range(30)
            for category, level in levels.items()
        ])

        self.assertEqual(detect_roles(frame).table_shape, "flat")


class MatrixTests(unittest.TestCase):
    def setUp(self):
        samples = [f"A{index}" for index in range(1, 16)] + [f"B{index}" for index in range(1, 16)]
        self.frame = pd.DataFrame(np.random.default_rng(8).poisson(5, (40, 30)), columns=samples)
        self.frame.insert(0, "Taxonomy", [f"taxon {index}" for index in range(40)])

    def test_a_matrix_has_no_single_measure_and_says_what_it_is(self):
        roles = detect_roles(self.frame)

        self.assertEqual(roles.table_shape, "matrix")
        self.assertIsNone(roles.measure)
        self.assertEqual(roles.identifier, "Taxonomy")
        note = next(note for note in roles.notes if note.startswith("This table looks like a matrix"))
        self.assertIn("one row per Taxonomy", note)
        self.assertIn("A* (15), B* (15)", note)

    def test_the_row_mean_becomes_the_measure(self):
        frame, roles = add_row_summary(self.frame, detect_roles(self.frame))

        self.assertEqual(roles.measure, "Row mean (30 columns)")
        self.assertAlmostEqual(frame.loc[0, roles.measure], self.frame.iloc[0, 1:].mean())

    def test_counts_spread_across_named_sample_columns_are_a_matrix(self):
        samples = [
            "CL3", "CC1", "SV1", "M31Fcsw", "AQC1cm", "AQC4cm", "NP2", "TRRsed1", "TS28", "Even1", "Even2",
        ]
        counts = np.random.default_rng(31).poisson(30, (60, len(samples)))
        counts[:, 4] *= 20
        frame = pd.DataFrame(counts, columns=samples)
        frame.insert(0, "OTU", [f"OTU{549000 + index}" for index in range(60)])

        detected = detect_roles(frame)
        _, roles = add_row_summary(frame, detected)

        self.assertEqual(detected.table_shape, "matrix")
        self.assertEqual(roles.measure, "Row mean (11 columns)")
        note = next(note for note in roles.notes if note.startswith("This table looks like a matrix"))
        self.assertIn("columns of counts", note)
        self.assertIn("common scale", note)
        self.assertEqual(row_summary_columns(roles, frame), samples)

    def test_codes_beside_structured_sample_names_are_left_out_of_the_matrix(self):
        rng = np.random.default_rng(32)
        samples = [f"{index}A_{100 + index}_1_{2000 + index}" for index in range(20)]
        frame = pd.DataFrame(rng.poisson(5000, (50, 20)), columns=samples)
        frame.insert(0, "AQ", rng.choice([10220, 20110, 20210], 50))
        frame.insert(0, "Feature_ID", [f"F{index}" for index in range(50)])

        _, roles = add_row_summary(frame, detect_roles(frame))

        self.assertEqual(roles.measure, "Row mean (20 columns)")
        self.assertEqual(row_summary_columns(roles, frame), samples)

    def test_a_few_count_columns_beside_readings_are_not_a_matrix(self):
        frame = pd.DataFrame({
            "Station": [f"S{index}" for index in range(40)],
            "Crabs": np.arange(40) % 4,
            "Seastars": np.arange(40) % 3,
            "Water Temp": readings(40),
        })

        self.assertEqual(detect_roles(frame).table_shape, "flat")

    def test_centred_rows_are_followed_by_their_spread_not_their_mean(self):
        rng = np.random.default_rng(9)
        values = rng.normal(0, 0.4, (60, 23))
        minutes = [str(10 * (index + 4)) for index in range(23)]
        frame = pd.DataFrame(values - values.mean(axis=1, keepdims=True), columns=minutes)
        frame.insert(0, "time", [f"Y{chr(65 + index % 26)}L{index:03d}C" for index in range(60)])

        detected = detect_roles(frame)
        summarised, roles = add_row_summary(frame, detected)

        self.assertEqual(detected.identifier, "time")
        self.assertEqual(roles.measure, "Row SD (23 columns)")
        self.assertAlmostEqual(summarised.loc[0, roles.measure], frame.iloc[0, 1:].std())
        note = next(note for note in roles.notes if note.startswith("This table looks like a matrix"))
        self.assertIn("centred near zero", note)
        self.assertNotIn("one row per row", note)
        self.assertEqual(len(row_summary_columns(roles, summarised)), 23)

    def test_a_matrix_without_row_labels_does_not_say_one_row_per_row(self):
        genes = [f"Gene_{index}" for index in range(30)]
        frame = pd.DataFrame(np.random.default_rng(10).poisson(5, (40, 30)), columns=genes)

        notes = detect_roles(frame).notes
        note = next(note for note in notes if note.startswith("This table looks like a matrix"))

        self.assertNotIn("one row per row", note)

    def test_a_wide_table_of_named_measures_is_not_a_matrix(self):
        names = "abcdefghijklmnopqrstuvwxy"
        frame = pd.DataFrame({f"marker_{name}": readings(30, seed=seed) for seed, name in enumerate(names)})

        self.assertEqual(detect_roles(frame).table_shape, "flat")


class NonReadingTests(unittest.TestCase):
    def test_compass_directions_are_set_aside(self):
        frame = pd.DataFrame({
            "Wind Direction": [10, 350, 180, 90, 270] * 8,
            "Aspect Ratio": readings(40, centre=2),
            "Air Temperature": readings(40, seed=4, centre=15),
        })

        roles = detect_roles(frame)

        self.assertTrue(is_angle_column("Wind Direction", frame["Wind Direction"]))
        self.assertFalse(is_angle_column("Aspect Ratio", frame["Aspect Ratio"]))
        self.assertEqual(roles.non_readings, ("Wind Direction",))
        self.assertNotEqual(roles.measure, "Wind Direction")
        self.assertEqual(roles.numeric[-1], "Wind Direction")

    def test_quality_flags_are_never_groupings_or_readings(self):
        frame = pd.DataFrame({
            "TMK": readings(60, centre=10),
            "TMK_flag": [1, 1, 3] * 20,
            "QN_4": [1, 10, 9] * 20,
            "RSKF": [0, 6, 7, 8] * 15,
        })

        roles = detect_roles(frame)

        self.assertEqual(set(roles.non_readings), {"TMK_flag", "QN_4"})
        self.assertEqual(roles.factors, ("RSKF",))
        self.assertEqual(roles.dimension, "RSKF")
        self.assertTrue(any("flag the quality of other readings" in note for note in roles.notes))

    def test_a_flag_that_names_an_outcome_is_still_a_grouping(self):
        frame = pd.DataFrame({"Fraud_Flag": [0, 1, 0, 0] * 10, "Amount": readings(40, centre=80)})

        self.assertFalse(is_quality_flag("Fraud_Flag", frame.columns))
        self.assertEqual(detect_roles(frame).flags, ("Fraud_Flag",))

    def test_map_coordinates_and_numbered_rows_are_set_aside(self):
        frame = pd.DataFrame({
            "result": readings(40, centre=1),
            "MESS_DATUM": pd.date_range("2020-01-01", periods=40).strftime("%Y%m%d").astype(int),
            "sample.samplingPoint.easting": np.arange(40) * 113 % 997 + 300_000,
            "x_coord": readings(40, seed=5, centre=500),
        })

        roles = detect_roles(frame)

        self.assertEqual(
            set(roles.non_readings), {"MESS_DATUM", "sample.samplingPoint.easting", "x_coord"}
        )
        self.assertEqual(roles.measure, "result")

    def test_an_id_before_the_name_marks_a_code(self):
        frame = pd.DataFrame({"id_reef": [index % 15 + 1 for index in range(150)], "quantity": readings(150)})

        self.assertEqual(detect_roles(frame).factors, ("id_reef",))


class StackedTableTests(unittest.TestCase):
    def setUp(self):
        food = pd.DataFrame({
            "source_table": "food",
            "food_item": [f"item {index}" for index in range(40)],
            "food_category": ["Fish", "Salt", "Water", "Honey"] * 10,
            "particles_per_g": readings(40, centre=5),
            "reference": [f"study {index % 7}" for index in range(40)],
        })
        blood = pd.DataFrame({
            "source_table": "blood",
            "biomarker": ["CRP", "aPTT", "Fibrinogen"] * 3,
            "overall_mean": readings(9, centre=100),
            "p_value_group": np.linspace(0.01, 0.5, 9),
            "unit": ["mg/dl", "s", "mg/dl"] * 3,
        })
        self.frame = pd.concat([food, blood], ignore_index=True)

    def test_the_largest_of_stacked_tables_is_the_focus(self):
        roles = detect_roles(self.frame)

        self.assertEqual(roles.table_shape, "stacked")
        self.assertEqual(roles.dimension, "source_table")
        self.assertEqual(roles.focus, "food")
        self.assertEqual(roles.measure, "particles_per_g")
        self.assertEqual(roles.dimensions[0], "food_category")
        note = next(note for note in roles.notes if note.startswith("This file stacks 2 tables"))
        self.assertIn("blood (9)", note)

    def test_a_column_filled_only_on_some_rows_is_no_stack(self):
        frame = pd.DataFrame({
            "Status": ["Completed"] * 30 + ["Returned"] * 10,
            "Revenue": readings(40, centre=100),
            "Units": [1, 2, 3, 4] * 10,
            "Region": ["N", "S"] * 20,
            "Refund Amount": [np.nan] * 30 + list(readings(10, centre=40)),
            "Return Reason": [None] * 30 + ["Damaged", "Late"] * 5,
        })

        self.assertEqual(detect_roles(frame).table_shape, "flat")


class GroupingChoiceTests(unittest.TestCase):
    def test_an_outcome_label_beats_a_sex_code(self):
        frame = pd.DataFrame({
            "Gender": [0, 1] * 30,
            "Result": ["positive", "negative", "negative"] * 20,
            "Troponin": readings(60, centre=1),
        })

        self.assertEqual(detect_roles(frame).dimension, "Result")

    def test_a_lopsided_grouping_gives_way_to_a_balanced_one(self):
        frame = pd.DataFrame({
            "Detector": ["OFF"] * 57 + ["ON"] * 3,
            "Phase": ["Baseline", "Fire", "Smoulder"] * 20,
            "MQ139": readings(60),
        })

        self.assertEqual(detect_roles(frame).dimension, "Phase")

    def test_one_subject_each_is_a_grouping_of_last_resort(self):
        frame = pd.DataFrame({
            "ID": [str(bird) for bird in range(20) for _ in range(3)],
            "PHENOTYPE": [bird % 3 for bird in range(20) for _ in range(3)],
            "TIME": readings(60),
        })

        self.assertEqual(detect_roles(frame).dimension, "PHENOTYPE")

    def test_a_month_column_beside_a_date_is_not_a_grouping(self):
        days = pd.date_range("2024-01-01", periods=120, freq="D")
        frame = pd.DataFrame({"Date": days, "MO": days.month, "MaxTemp": readings(120)})

        roles = detect_roles(frame)

        self.assertNotIn("MO", roles.factors)
        self.assertNotIn("MO", roles.dimensions)
        self.assertIsNone(roles.dimension)
        self.assertEqual(roles.measure, "MaxTemp")


class RowNounTests(unittest.TestCase):
    def test_the_last_plural_in_a_file_name_names_the_rows(self):
        for name, noun in (
            ("penguins.csv", "penguins"), ("olist_orders_dataset.csv", "orders"), ("People.csv", "people"),
            ("/data/flights.csv", "flights"), ("bus_routes.tsv", "routes"),
            ("data.csv", None), ("ChickWeight.csv", None), ("analysis.csv", None), (None, None),
        ):
            with self.subTest(name=name):
                self.assertEqual(row_noun(name), noun)

    def test_a_noun_that_is_part_of_a_column_name_is_not_the_rows(self):
        frame = pd.DataFrame({"subject": [1, 2, 3], "total_UPDRS": [20.0, 25.0, 30.0]})
        self.assertIsNone(detect_roles(frame, "parkinsons_updrs.csv").row_noun)
        self.assertEqual(detect_roles(frame, "visits.csv").row_noun, "visits")


class NumberSignTests(unittest.TestCase):
    def test_a_number_sign_after_a_unit_word_names_the_unit(self):
        for name in ("subject#", "Patient #", "station#"):
            with self.subTest(name=name):
                self.assertTrue(is_entity_name(name))
        self.assertFalse(is_entity_name("Jitter(%)"))



class ClockTimeTests(unittest.TestCase):
    def test_times_on_one_day_give_way_to_dates(self):
        frame = pd.DataFrame({
            "fastestLapTime": pd.to_datetime(["2024-01-01 00:01:27", "2024-01-01 00:01:29",
                                              "2024-01-01 00:01:26", "2024-01-01 00:01:30"]),
            "date": pd.to_datetime(["2019-03-17", "2019-03-31", "2019-04-14", "2020-07-05"]),
            "points": [25.0, 18.0, 25.0, 10.0],
        })
        self.assertEqual(detect_roles(frame).date, "date")
        self.assertEqual(detect_roles(frame.drop(columns="date")).date, "fastestLapTime")


if __name__ == "__main__":
    unittest.main()
