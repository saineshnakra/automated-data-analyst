from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ada.schema import (
    IDENTIFIER_NAME_WORDS,
    MEASURE_KEYWORDS,
    add_row_summary,
    detect_roles,
    is_design_name,
    is_entity_name,
    is_identifier_name,
    is_month_name,
    is_time_part,
    is_year_name,
    looks_like_identifier,
)


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
        for name in ("Phone No.", "Account #", "Zipcode", "Postcode", "Telephone"):
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

    def test_a_wide_table_of_named_measures_is_not_a_matrix(self):
        names = "abcdefghijklmnopqrstuvwxy"
        frame = pd.DataFrame({f"marker_{name}": readings(30, seed=seed) for seed, name in enumerate(names)})

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


if __name__ == "__main__":
    unittest.main()
