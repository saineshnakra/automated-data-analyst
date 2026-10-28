import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from table_helpers import air_quality, births, commuting, covid_deaths, deaths, migration, registry

import ada
from ada.claims import (
    NUMBER_PATTERN,
    REASONS,
    _based,
    claimed_number,
    column_unit,
    read_claim,
)


def number(text: str, qualifier: str | None = None):
    return claimed_number(text, NUMBER_PATTERN.search(text), qualifier, None)


def birds() -> pd.DataFrame:
    """Eight birds of two species, weighed in grams, one weight missing.

    By hand: mean body mass 31,200 / 7 = 4,457.142857; Gentoo 5,050; Adelie 11,000 / 3.
    """
    return pd.DataFrame({
        "species": ["Adelie"] * 4 + ["Gentoo"] * 4,
        "island": ["Dream", "Biscoe", "Dream", "Biscoe"] * 2,
        "body_mass_g": [3700.0, 3800.0, 3500.0, None, 5000.0, 5200.0, 5100.0, 4900.0],
        "bill_length_mm": [38.0, 39.0, 40.0, 41.0, 47.0, 48.0, 49.0, 50.0],
    })


class ClaimFileTestCase(unittest.TestCase):
    """Writes birds() to birds.csv in a temporary folder, so checks run on a real file."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.path = self.folder / "birds.csv"
        birds().to_csv(self.path, index=False)

    def check(self, claim: str, **options):
        return ada.check_claim(self.path, claim, **options)


class ClaimedNumberTests(unittest.TestCase):
    def test_decimals_set_the_interval_exactly(self):
        claimed = number("12.64")
        self.assertEqual((claimed.low, claimed.high), (12.635, 12.645))
        self.assertTrue(claimed.holds(12.639))
        self.assertFalse(claimed.holds(12.65))
        self.assertEqual(claimed.interval(), "12.635 to 12.645")

    def test_about_reaches_one_significant_unit_at_most_a_tenth(self):
        for text, low, high in (
            ("13", 12, 14), ("100", 90, 110), ("2,000", 1800, 2200), ("12.6", 12.5, 12.7), ("2", 1.5, 2.5),
        ):
            with self.subTest(text=text):
                claimed = number(text, "about")
                self.assertAlmostEqual(claimed.low, low)
                self.assertAlmostEqual(claimed.high, high)

    def test_bounds_and_open_ends(self):
        self.assertTrue(number("5", "at most").holds(5))
        self.assertFalse(number("5", "at most").holds(5.01))
        self.assertFalse(number("5", "less than").holds(5))
        self.assertTrue(number("30", "more than").holds(1e9))
        self.assertFalse(number("30", "more than").holds(30))
        self.assertEqual(number("30", "more than").interval(), "more than 30")
        self.assertTrue(number("3", "exactly").holds(3))
        self.assertFalse(number("3", "exactly").holds(3.4))
        self.assertTrue(number("100", "nearly").holds(95))
        self.assertFalse(number("100", "nearly").holds(100.4))

    def test_scale_words_currency_and_percent(self):
        billions = number("37.5 billion")
        self.assertEqual(billions.value, 3.75e10)
        self.assertTrue(billions.holds(3.7528e10))
        self.assertEqual(number("R$16,008,872").value, 16008872)
        self.assertTrue(number("32%").percent)
        self.assertEqual(number("−4.5").value, -4.5)

    def test_missing_and_infinite_values_never_hold(self):
        self.assertFalse(number("0").holds(None))
        self.assertFalse(number("0", "at least").holds(math.nan))
        self.assertFalse(number("0", "at least").holds(math.inf))

    def test_the_interval_is_json_safe(self):
        data = number("30", "more than").to_dict()
        self.assertEqual(data["interval"], [30.0, None])
        json.dumps(data, allow_nan=False)


class ReadClaimTests(unittest.TestCase):
    columns = ["species", "island", "body_mass_g", "bill_length_mm"]

    def test_the_number_comes_out_and_the_rest_is_the_question(self):
        question, claimed, problem = read_claim("The average body mass is 4457 g.", self.columns)
        self.assertEqual((question, claimed.value, claimed.unit), ("average body mass", 4457, "g"))
        self.assertEqual(problem, "")

    def test_a_number_before_the_rows_noun_counts_rows(self):
        for claim, question in (
            ("There are 8 birds.", "how many birds"),
            ("4 birds are Gentoo.", "how many birds are Gentoo"),
        ):
            with self.subTest(claim=claim):
                self.assertEqual(read_claim(claim, self.columns, "birds")[0], question)

    def test_years_and_ordinals_stay_in_the_question(self):
        question, claimed, _ = read_claim("The average body mass in 2008 was 5076 g.", self.columns)
        self.assertEqual((question, claimed.value), ("average body mass in 2008", 5076))
        question, claimed, _ = read_claim("The 25th percentile of body mass is 3700 g.", self.columns)
        self.assertEqual((question, claimed.value), ("25th percentile of body mass", 3700))

    def test_a_cue_word_becomes_the_aggregation(self):
        question, claimed, _ = read_claim("Gentoo birds averaged 5050 g of body mass.", self.columns)
        self.assertEqual((question, claimed.value), ("average Gentoo birds of body mass", 5050))

    def test_no_number_or_two_rival_numbers_is_a_problem(self):
        self.assertIn("no number", read_claim("Gentoo birds are heavy.", self.columns)[2])
        for claim in (
            "The average body mass was 4457 g and 3700 g.", "Adelie birds on 3 islands weigh 3700 g.",
        ):
            with self.subTest(claim=claim):
                question, _, problem = read_claim(claim, self.columns)
                self.assertIsNone(question)
                self.assertIn("more than one number", problem)


class ColumnUnitTests(unittest.TestCase):
    def test_units_come_from_the_end_of_a_name(self):
        for column, unit in (
            ("body_mass_g", "g"), ("Weight (kg)", "kg"), ("air_time_minutes", "minutes"),
            ("co2", None), ("Days", None), ("dep_delay", None), (None, None),
        ):
            with self.subTest(column=column):
                self.assertEqual(column_unit(column), unit)


class CheckClaimTests(ClaimFileTestCase):
    def test_a_true_claim_is_verified_with_its_rows_and_source(self):
        check = self.check("The average body mass is 4457 g.")
        self.assertTrue(check.verified)
        self.assertIsNone(check.reason)
        self.assertEqual(check.agreement, "match")
        self.assertAlmostEqual(check.compared, 31200 / 7)
        self.assertEqual(check.answer.scope.rows_used, 7)
        self.assertIn("Left out 1 row with body_mass_g missing.", check.assumptions)
        self.assertEqual(check.source.sha256, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual((check.source.rows, check.source.columns), (8, 4))

    def test_a_wrong_number_fails_with_the_computed_value(self):
        check = self.check("The average body mass is 4,600 g.")
        self.assertEqual((check.status, check.reason), ("unverified", "failed_check"))
        self.assertEqual(check.agreement, "mismatch")
        self.assertAlmostEqual(check.compared, 31200 / 7)
        self.assertIn("4,457.142857", check.message)
        self.assertFalse(check.checks[0].passed)

    def test_a_qualifier_widens_the_claim(self):
        self.assertTrue(self.check("The average body mass is about 4,500 g.").verified)
        self.assertTrue(self.check("The average body mass is more than 4,000 g.").verified)
        self.assertFalse(self.check("The average body mass is less than 4,000 g.").verified)

    def test_a_filter_and_a_count(self):
        self.assertTrue(self.check("Gentoo birds averaged 5050 g of body mass.").verified)
        self.assertTrue(self.check("There are 8 birds.").verified)
        self.assertTrue(self.check("4 birds are Gentoo.").verified)
        self.assertTrue(self.check("50% of the birds are Gentoo.").verified)
        self.assertEqual(self.check("There are 9 birds.").reason, "failed_check")

    def test_a_unit_in_the_column_name_is_converted(self):
        check = self.check("The average body mass of Gentoo is 5.05 kg.")
        self.assertTrue(check.verified)
        self.assertAlmostEqual(check.compared, 5.05)
        self.assertTrue(any("Converted body_mass_g from g" in item for item in check.assumptions))
        self.assertEqual(self.check("The average bill length is 4.4 cm.").reason, None)
        self.assertEqual(self.check("The average body mass is 5.5 kg.").reason, "failed_check")

    def test_a_verb_and_a_unit_find_the_column_a_claim_does_not_name(self):
        check = self.check("Gentoo birds weigh 5050 g on average.")
        self.assertTrue(check.verified)
        self.assertEqual(check.answer.plan.measure, "body_mass_g")
        self.assertIn("in a unit of that kind", check.assumptions[0])
        self.assertEqual(self.check("Adelie birds weigh 4000 g on average.").reason, "failed_check")
        self.assertTrue(self.check("Birds measure 44 mm on average.").verified)

    def test_a_unit_that_fits_several_columns_gives_each_reading(self):
        frame = birds().assign(flipper_length_mm=[181.0, 186, 195, 193, 211, 230, 210, 218])
        check = ada.check_claim(frame, "Adelie birds measure 189 mm on average.")
        self.assertEqual(check.reason, "ambiguous_intent")
        self.assertEqual({item.reading: item.agrees for item in check.alternatives},
                         {"bill_length_mm": False, "flipper_length_mm": True})
        self.assertIn("The claim's unit fits more than one column", check.message)

    def test_a_unit_of_another_kind_conflicts(self):
        check = self.check("The average bill length is 44 minutes.")
        self.assertEqual((check.status, check.reason), ("unverified", "conflicting_evidence"))
        self.assertIn("length, not time", check.message)

    def test_an_unnamed_measure_is_not_guessed(self):
        check = self.check("The average is 4457.")
        self.assertEqual(check.reason, "missing_context")
        self.assertEqual(check.clarification, "Which column is the number about?")

    def test_one_number_per_group_needs_the_group(self):
        check = self.check("The average body mass by species is 5050 g.")
        self.assertEqual(check.reason, "missing_context")
        self.assertEqual(check.clarification, "Which group is the number about?")

    def test_a_year_without_a_date_column_asks_for_it(self):
        check = self.check("The average body mass in 2007 was 4457 g.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("no date column", check.message)

    def test_two_numbers_are_checked_one_at_a_time(self):
        check = self.check("The average body mass was 4457 g and 3700 g.")
        self.assertEqual(check.reason, "unsupported_analysis")
        self.assertEqual(check.clarification, "Send each number as its own claim.")

    def test_a_claim_with_no_number(self):
        self.assertEqual(self.check("Gentoo birds are heavier.").reason, "missing_context")

    def test_question_and_value_keep_the_written_decimals(self):
        check = ada.check_claim(self.path, question="average body_mass_g", value="4457.1")
        self.assertTrue(check.verified)
        self.assertEqual((check.claimed.low, check.claimed.high), (4457.05, 4457.15))
        self.assertFalse(ada.check_claim(self.path, question="average body_mass_g", value=4457.2).verified)
        self.assertEqual(ada.check_claim(self.path, question="average body_mass_g", value="n/a").reason,
                         "missing_context")

    def test_a_folder_is_checked_against_the_file_the_claim_fits(self):
        check = ada.check_claim(self.folder, "The average body mass is 4457 g.")
        self.assertTrue(check.verified)
        self.assertIn("birds.csv, the only file", check.assumptions[0])
        empty = self.folder / "empty"
        empty.mkdir()
        nothing = ada.check_claim(empty, "The average body mass is 4457 g.")
        self.assertEqual(nothing.reason, "inaccessible_data")

    def test_unreadable_data(self):
        missing = ada.check_claim(self.folder / "nowhere.csv", "The average body mass is 4457 g.")
        self.assertEqual(missing.reason, "inaccessible_data")
        empty = self.folder / "empty.csv"
        empty.write_text("")
        check = ada.check_claim(empty, "The average body mass is 4457 g.")
        self.assertEqual(check.reason, "inaccessible_data")

    def test_every_reason_is_one_of_the_six(self):
        for claim in (
            "The average body mass is 4457 g.", "The average body mass is 4,600 g.", "The average is 4457.",
            "The average bill length is 44 minutes.", "The average body mass was 4457 g and 3700 g.",
        ):
            with self.subTest(claim=claim):
                check = self.check(claim)
                self.assertEqual(check.status == "verified", check.reason is None)
                self.assertTrue(check.reason is None or check.reason in REASONS)


class AmbiguousClaimTests(unittest.TestCase):
    def test_words_that_fit_two_columns_give_both_readings(self):
        delays = pd.DataFrame({
            "carrier": list("AABBCCDD"),
            "dep_delay": [1.0, 2, 3, 4, 5, 6, 7, 8],
            "arr_delay": [-1.0, 0, 1, 2, 3, 4, 5, 6],
        })
        check = ada.check_claim(delays, "The average delay was 4.5 minutes.")
        self.assertEqual(check.reason, "ambiguous_intent")
        readings = {item.reading: (item.value, item.agrees) for item in check.alternatives}
        self.assertEqual(readings, {"dep_delay": (4.5, True), "arr_delay": (2.5, False)})
        self.assertEqual(check.clarification, "Which column is meant: dep_delay or arr_delay?")
        self.assertIsNone(check.source.sha256)

    def test_a_scale_the_data_do_not_state_is_not_assumed(self):
        emissions = pd.DataFrame({
            "country": ["World", "World"], "year": [2021, 2022], "co2": [37124.0, 37527.8],
        })
        check = ada.check_claim(emissions, "The total co2 was 74.65 billion.")
        self.assertEqual((check.reason, check.agreement), ("missing_context", "mismatch"))
        self.assertEqual(check.clarification, "What unit is co2 recorded in?")

    def test_a_camelcase_name_in_words_and_a_unit_with_a_slash(self):
        # "lap" alone would name laps; "fastest lap speed" names fastestLapSpeed.
        results = pd.DataFrame({
            "laps": [58, 58, 57, 50], "fastestLap": [44, 51, 39, 12],
            "fastestLapSpeed": [201.2, 203.0, 202.4, 203.8],
        })
        check = ada.check_claim(results, "The average fastest lap speed is 202.6 km/h.")
        self.assertTrue(check.verified)
        self.assertEqual(check.answer.plan.measure, "fastestLapSpeed")
        self.assertIn('The unit "km/h" comes from the claim', " ".join(check.assumptions))


class RepeatedUnitTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Subject": [f"S{index}" for index in range(1, 7) for _ in range(4)],
            "Days": list(range(4)) * 6,
            "Reaction": [250.0 + 10 * day + subject for subject in range(6) for day in range(4)],
        })

    def test_an_average_over_repeated_rows_says_how_units_are_weighted(self):
        analysis = ada.analyze(self.frame())
        analysis = replace(analysis, roles=replace(analysis.roles, identifier="Subject"))
        check = ada.check_claim(analysis, "The average Reaction is 267.5.")
        self.assertTrue(check.verified)
        self.assertTrue(any("weights each Subject" in warning for warning in check.warnings))

    def test_one_repeated_id_is_not_repeated_measurement(self):
        climbers = pd.DataFrame({"member_id": [f"M{index}" for index in range(30)] + ["M0"],
                                 "age": [30.0 + index % 20 for index in range(31)]})
        analysis = ada.analyze(climbers)
        analysis = replace(analysis, roles=replace(analysis.roles, identifier="member_id"))
        check = ada.check_claim(analysis, f"The average age is {climbers['age'].mean():.1f}.")
        self.assertTrue(check.verified)
        self.assertEqual(check.warnings, ())


class GuessedColumnTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "member_id": [f"M{index}" for index in range(5)],
            "expedition_role": ["Leader", "Member", "Member", "Member", "Guide"],
            "death_height_metres": [math.nan, 8000.0, 7500.0, math.nan, math.nan],
            "num_parts": [10, 20, 30, 40, 50],
        })

    def test_a_column_the_claim_does_not_name_by_its_last_word_does_not_fail_it(self):
        # "deaths" is the first word of death_height_metres, a height, so it names the deaths, not the height;
        # summing heights would be a guess.
        check = ada.check_claim(self.frame(), "Members recorded 3 member deaths in total.")
        self.assertEqual(check.reason, "missing_context")
        self.assertTrue(check.answer is None or check.answer.plan.measure != "death_height_metres")

    def test_counting_words_and_units_need_not_be_named(self):
        check = ada.check_claim(self.frame(), "Rows have 40 parts on average.")
        self.assertEqual(check.reason, "failed_check")
        self.assertTrue(ada.check_claim(self.frame(), "Rows have 30 parts on average.").verified)


class FrequencyTableClaimTests(unittest.TestCase):
    def test_a_percentage_with_no_share_word_is_checked_as_a_weighted_share(self):
        titanic = pd.DataFrame({
            "Class": ["1st", "2nd", "3rd", "Crew"] * 2, "Survived": ["No"] * 4 + ["Yes"] * 4,
            "Freq": [122, 167, 528, 673, 203, 118, 178, 212],
        })
        check = ada.check_claim(titanic, "32.3% survived.")
        self.assertTrue(check.verified)
        self.assertIn("sum(Freq)", check.answer.calculation)
        # Half the rows, a third of the people.
        self.assertEqual(ada.check_claim(titanic, "50% survived.").reason, "failed_check")


class ThinDataClaimTests(unittest.TestCase):
    def test_one_row_and_an_empty_column(self):
        one = pd.DataFrame({"site": ["A"], "depth_m": [3.0], "reading": [math.nan]})
        self.assertTrue(ada.check_claim(one, "The average depth is 3.").verified)
        check = ada.check_claim(one, "The average reading is 3.")
        self.assertEqual(check.status, "unverified")
        self.assertIn(check.reason, REASONS)


class ClaimJsonTests(ClaimFileTestCase):
    def test_every_status_is_json_safe_and_keeps_its_layout(self):
        keys = {
            "schema_version", "claim", "status", "reason", "message", "claimed", "question", "agreement",
            "result", "reading", "calculation", "evidence", "assumptions", "warnings", "checks",
            "alternatives", "clarification", "decisions", "words_left_out",
        }
        for claim in (
            "The average body mass is 4457 g.", "The average body mass is 4,600 g.", "The average is 4457.",
            "Gentoo birds are heavier.", "The average bill length is 44 minutes.",
        ):
            with self.subTest(claim=claim):
                data = self.check(claim).to_dict()
                self.assertEqual(set(data), keys)
                json.dumps(data, allow_nan=False)

    def test_the_evidence_names_the_file_and_the_rows(self):
        data = self.check("The average body mass is 4457 g.").to_dict()
        self.assertEqual(data["status"], "verified")
        self.assertEqual(data["reading"]["measure"], "body_mass_g")
        self.assertEqual(data["calculation"], "mean(body_mass_g)")
        evidence = data["evidence"]
        self.assertEqual(evidence["source"]["name"], "birds.csv")
        self.assertEqual((evidence["rows_read"], evidence["rows_used"]), (8, 7))
        self.assertEqual(evidence["left_out"], [{"reason": "rows with body_mass_g missing", "count": 1}])

    def test_printing_a_check_shows_the_calculation(self):
        text = str(self.check("The average body mass is 4,600 g."))
        self.assertTrue(text.startswith("unverified: The average body mass is 4,600 g."))
        self.assertIn("calc: mean(body_mass_g)", text)


class UnitOfAnalysisClaimTests(unittest.TestCase):
    def test_a_claim_that_matches_the_row_average_is_told_so(self):
        rng = np.random.default_rng(48)
        counts = rng.integers(1, 4, 60)
        frame = pd.DataFrame({
            "order_id": np.repeat([f"o{n:03d}" for n in range(60)], counts),
            "payment_value": rng.gamma(2, 50, counts.sum()).round(2),
        })
        per_row = frame["payment_value"].mean()
        check = ada.check_claim(frame, f"The average order value is {per_row:.2f}.")
        self.assertEqual((check.reason, check.agreement), ("failed_check", "mismatch"))
        self.assertAlmostEqual(check.compared, frame.groupby("order_id")["payment_value"].sum().mean())
        self.assertIn("matches the average taken row by row", check.message)

    def test_a_pooled_correlation_claim_carries_the_reversal_warning(self):
        rng = np.random.default_rng(49)
        parts = []
        for index, label in enumerate(("A", "B", "C")):
            x = rng.normal(40 + 5 * index, 2, 60)
            parts.append(pd.DataFrame({"kind": label, "length": x,
                                       "depth": 0.4 * x - 4 * index + rng.normal(0, 0.5, 60)}))
        frame = pd.concat(parts, ignore_index=True)
        pooled = frame["length"].corr(frame["depth"])
        check = ada.check_claim(frame, question="correlation between length and depth", value=f"{pooled:.2f}")
        self.assertTrue(check.verified)
        self.assertTrue(any("runs the other way" in warning for warning in check.warnings))


if __name__ == "__main__":
    unittest.main()


class PhrasingClaimTests(unittest.TestCase):
    def test_an_adjective_after_the_unit_names_the_quantity(self):
        peaks = pd.DataFrame({"peak_name": [f"P{index}" for index in range(6)],
                              "height_metres": [6000.0, 7000.0, 8000.0, 6500.0, 7500.0, 7000.0],
                              "first_ascent_year": [1950, 1960, 1970, 1980, 1990, 2000]})
        check = ada.check_claim(peaks, "The average peak is 7,000 metres high.")
        self.assertTrue(check.verified)
        self.assertEqual(check.answer.plan.measure, "height_metres")

    def test_a_subject_that_names_nothing_and_a_currency_pick_the_money_column(self):
        payments = pd.DataFrame({"order_id": [f"o{index}" for index in range(8)],
                                 "payment_sequential": [1, 1, 2, 1, 1, 1, 3, 1],
                                 "payment_installments": [1, 3, 2, 10, 1, 4, 1, 2],
                                 "payment_value": [10.5, 20.0, 5.25, 100.0, 7.75, 12.5, 3.0, 41.0]})
        check = ada.check_claim(payments, "Customers paid R$200 in total.")
        self.assertTrue(check.verified)
        self.assertIn('Read "Customers" as every row of the table', check.assumptions[0])
        self.assertIn("the only money column it fits", check.assumptions[1])
        # Without a currency the payment columns stay ambiguous.
        self.assertEqual(ada.check_claim(payments, "Customers paid 200 in total.").reason, "ambiguous_intent")

    def test_a_brand_before_the_rows_noun_describes_every_row(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "sets.csv"
        pd.DataFrame({"set_num": [f"s{index}" for index in range(6)], "name": list("abcdef"),
                      "num_parts": [100, 200, 300, 150, 250, 50]}).to_csv(path, index=False)
        check = ada.check_claim(path, "The average Lego set has 175 parts.")
        self.assertTrue(check.verified)
        self.assertIn('Read "Lego set" as every set', check.assumptions[0])

    def test_a_brand_the_folder_is_named_for_still_describes_every_row(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "lego" / "sets.csv"
        path.parent.mkdir()
        pd.DataFrame({"set_num": [f"s{index}" for index in range(6)], "name": list("abcdef"),
                      "num_parts": [100, 200, 300, 150, 250, 50]}).to_csv(path, index=False)
        check = ada.check_claim(path, "The average Lego set has 175 parts.")
        self.assertTrue(check.verified)
        self.assertIn('Read "Lego set" as every set', check.assumptions[0])

    def test_a_value_before_the_rows_noun_still_filters(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "penguins.csv"
        birds().to_csv(path, index=False)
        check = ada.check_claim(path, "Gentoo penguins weigh 5,050 g on average.")
        self.assertTrue(check.verified)
        self.assertIn("species = Gentoo", check.answer.calculation)


def losses() -> pd.DataFrame:
    """Forest lost to each cause, by country and year. By hand: mean pasture 1,500."""
    return pd.DataFrame({
        "entity": ["Brazil", "Brazil", "Peru", "Peru"],
        "year": [2004, 2005, 2004, 2005],
        "pasture": [2000.0, 1000.0, 1800.0, 1200.0],
        "fire": [10.0, 20.0, 30.0, 40.0],
    })


def masses(rows: int = 12) -> pd.DataFrame:
    """Body mass rising 50 g per mm of flipper, give or take 10 g.

    By hand, for 12 rows (flippers 180 to 235 mm): the deviations from the line
    move the slope by -300 / 3,575, so it is 49.916.
    """
    flipper = np.arange(180.0, 180.0 + 5 * rows, 5)
    return pd.DataFrame({"flipper_length_mm": flipper,
                         "body_mass_g": 50 * flipper - 5000 + np.resize([10.0, -10.0], rows)})


class LeftOutWordTests(unittest.TestCase):
    def test_words_that_name_nothing_are_left_out_of_a_one_row_lookup(self):
        check = ada.check_claim(losses(), "Brazil lost 2,000 hectares of forest to pasture in 2004.")
        self.assertTrue(check.verified)
        self.assertIn('without "lost" and "forest"', check.assumptions[0])

    def test_a_mismatch_without_them_proves_nothing(self):
        # Fewer rows than the words keep could add up to less than 2,000.
        check = ada.check_claim(losses(), "Brazil lost 1,500 hectares of forest to pasture in 2004.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("may pick other rows", check.message)
        # No fewer rows add up to more than the 2,000 of every row they keep.
        check = ada.check_claim(losses(), "Brazil lost 2,500 hectares of forest to pasture in 2004.")
        self.assertEqual(check.reason, "failed_check")

    def test_an_average_or_a_bound_that_agrees_without_them_is_not_verified(self):
        average = ada.check_claim(losses(), "The average forest pasture was 1,500.")
        bound = ada.check_claim(losses(), "Brazil lost more than 1,000 hectares to pasture in 2004.")
        self.assertEqual((average.reason, average.agreement), ("missing_context", "match"))
        self.assertEqual(bound.reason, "missing_context")

    def test_words_that_negate_are_never_left_out(self):
        check = ada.check_claim(losses(), "Pasture not lost to forest in 2004 was 3,800.")
        self.assertFalse(check.verified)


class SlopeClaimTests(unittest.TestCase):
    def test_a_change_per_unit_of_another_column_is_its_slope(self):
        check = ada.check_claim(masses(), "Body mass rises by about 50 g per mm of flipper length.")
        self.assertTrue(check.verified)
        self.assertAlmostEqual(check.compared, 50 - 300 / 3575)
        self.assertIn("slope of body_mass_g on flipper_length_mm", check.message)

    def test_the_wrong_size_or_direction_fails(self):
        smaller = ada.check_claim(masses(), "Body mass rises by 30 g per mm of flipper length.")
        falling = ada.check_claim(masses(), "Body mass falls by about 50 g per mm of flipper length.")
        heavier = ada.check_claim(masses(), "Body mass rises by about 50 kg per mm of flipper length.")
        self.assertEqual([smaller.reason, falling.reason, heavier.reason], ["failed_check"] * 3)
        self.assertLess(falling.compared, 0)

    def test_rising_to_a_level_is_no_slope(self):
        from ada.claims import _slope_claim

        self.assertIsNone(_slope_claim("Global potato yields rose to 20.9 tonnes per hectare by 2018."))
        self.assertIsNotNone(_slope_claim("Body mass rises by 50 g per mm of flipper length."))

    def test_too_few_rows_give_no_slope(self):
        check = ada.check_claim(masses(11), "Body mass rises by about 50 g per mm of flipper length.")
        self.assertEqual(check.reason, "unsupported_analysis")


class WordedNumberTests(ClaimFileTestCase):
    def test_a_fraction_in_words_stands_for_the_values_nearest_it(self):
        from ada.claims import _fraction_interval

        low, high = _fraction_interval(0.25)
        self.assertAlmostEqual(low, (0.25 + 2 / 9) / 2 * 100)
        self.assertAlmostEqual(high, (0.25 + 2 / 7) / 2 * 100)
        self.assertTrue(self.check("About half of the birds are Gentoo.").verified)
        self.assertEqual(self.check("More than half of the birds are Gentoo.").reason, "failed_check")
        self.assertEqual(self.check("Nearly a third of the birds are Gentoo.").reason, "failed_check")

    def test_an_ordinal_quarter_is_a_period_not_a_fraction(self):
        from ada.claims import _worded_number

        self.assertIsNone(_worded_number("In the fourth quarter of 2020, earnings were $894."))
        self.assertIsNone(_worded_number("Sales in quarter of the year rose."))
        self.assertIsNotNone(_worded_number("Half of the birds are Gentoo."))

    def test_a_whole_number_in_words(self):
        self.assertTrue(self.check("There are eight birds.").verified)
        self.assertTrue(self.check("Four birds are Gentoo.").verified)
        self.assertEqual(self.check("There are nine birds.").reason, "failed_check")

    def test_a_value_keeps_its_qualifier(self):
        check = self.check(None, question="what share are Gentoo", value="about 50%")
        self.assertTrue(check.verified)
        self.assertEqual(check.claimed.qualifier, "about")
        self.assertEqual(self.check(None, question="what share are Gentoo", value="about 40%").reason,
                         "failed_check")

    def test_a_bound_is_never_read_in_thousands(self):
        self.assertEqual(self.check("More than 60% of the birds are Gentoo.").reason, "failed_check")

    def test_a_value_before_the_rows_noun_counts_them(self):
        check = self.check("There are 4 Gentoo birds.")
        self.assertTrue(check.verified)
        self.assertEqual(check.claimed.text, "4")


class FilterNumberTests(unittest.TestCase):
    def test_a_number_after_with_and_before_a_column_picks_rows(self):
        cars = pd.DataFrame({"cyl": [4, 4, 6, 6, 8, 8], "mpg": [30.0, 24.0, 20.0, 18.0, 15.0, 13.0]})
        check = ada.check_claim(cars, "Cars with 4 cylinders average 27 mpg.")
        self.assertTrue(check.verified)
        self.assertIn("cyl = 4", check.answer.calculation)

    def test_digits_inside_a_code_are_not_a_number_and_a_column_name_is_not_a_unit(self):
        batting = pd.DataFrame({"playerID": ["bondsba01", "bondsba01", "smithjo01"],
                                "yearID": [2000, 2001, 2001], "HR": [49, 73, 10]})
        check = ada.check_claim(batting, "bondsba01 hit 73 HR in 2001")
        self.assertTrue(check.verified)
        self.assertEqual(check.claimed.unit, None)
        self.assertEqual(ada.check_claim(batting, "bondsba01 hit 80 HR in 2001").reason, "failed_check")
        self.assertEqual(ada.check_claim(batting, "bondsba01 hit a total of 122 HR").compared, 122)


def pines() -> pd.DataFrame:
    """Four trees measured at ages 3 and 5, listed tree by tree; Seed names the tree.

    By hand: mean height at age 3 is (4.0 + 4.4 + 4.2 + 4.3) / 4 = 4.225 feet.
    """
    return pd.DataFrame({"height": [4.0, 10.0, 4.4, 11.0, 4.2, 10.5, 4.3, 10.8],
                         "age": [3, 5] * 4, "Seed": [301, 301, 302, 302, 303, 303, 304, 304]})


class ReadingGuardTests(unittest.TestCase):
    def test_a_count_pinned_to_one_row_by_a_key_is_not_compared(self):
        counties = pd.DataFrame({"County": ["Nairobi", "Mombasa", "Kisumu"],
                                 "Total": [4397073, 1208333, 1155574]})
        check = ada.check_claim(counties, "Nairobi had 5.2 million residents.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("names one row each", check.message)

    def test_a_number_that_counts_people_is_a_total(self):
        counties = pd.DataFrame({"County": ["Nairobi", "Mombasa", "Kisumu"], "Intersex": [245, 30, 40]})
        check = ada.check_claim(counties, "315 Kenyans were recorded as intersex.")
        self.assertTrue(check.verified)
        self.assertIn('as a total, since it counts Kenyans', check.assumptions[0])
        self.assertEqual(ada.check_claim(counties, "400 Kenyans were recorded as intersex.").reason,
                         "failed_check")

    def test_a_column_guessed_from_part_of_a_word_gives_way_to_the_unit(self):
        check = ada.check_claim(pines(), "At age 3 the seedlings averaged 6.2 feet.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("height", check.answer.calculation)
        self.assertIn('rather than Seed, which "seedlings" only resembles', check.assumptions[0])

    def test_a_number_in_feet_reads_the_one_column_named_for_a_length(self):
        check = ada.check_claim(pines(), "The mean at age 3 was 4.2 feet.")
        self.assertTrue(check.verified)
        self.assertAlmostEqual(check.compared, 4.225)


def admissions() -> pd.DataFrame:
    """A table of counts. By hand: men 60 of 100 admitted (60%), women 30 of 100 (30%), 45% of all 200."""
    return pd.DataFrame({"Admit": ["Admitted", "Rejected"] * 4, "Gender": ["Male"] * 4 + ["Female"] * 4,
                         "Dept": ["A", "A", "B", "B"] * 2, "Freq": [50, 20, 10, 20, 25, 25, 5, 45]})


class ShareClaimTests(unittest.TestCase):
    def test_a_rate_among_a_group_is_of_that_group(self):
        self.assertTrue(ada.check_claim(admissions(), "Men were admitted at a rate of 60%.").verified)
        self.assertEqual(ada.check_claim(admissions(), "Men were admitted at a rate of 30%.").reason,
                         "failed_check")
        self.assertTrue(ada.check_claim(admissions(), "Women were admitted at a 30% rate.").verified)

    def test_a_lead_that_names_the_base_is_checked(self):
        check = ada.check_claim(admissions(), "Of the 200 applicants, 45% were admitted.")
        self.assertTrue(check.verified)
        self.assertIn("The claim's 200 applicants are the 200 the share is of.", check.assumptions)
        other = ada.check_claim(admissions(), "Of the 150 applicants, 45% were admitted.")
        self.assertEqual(other.reason, "missing_context")

    def test_a_plural_that_names_no_column_counts_rows(self):
        surveys = pd.DataFrame({"year": [2006, 2007, 2007, 2007], "wgt": [30.0, 41.0, 28.0, 35.0]})
        check = ada.check_claim(surveys, "The survey recorded 3 captures in 2007.")
        self.assertTrue(check.verified)
        self.assertEqual(check.words_left_out, ("survey",))
        # "survey" is left out, so a mismatch proves nothing.
        self.assertEqual(ada.check_claim(surveys, "The survey recorded 5 captures in 2007.").reason,
                         "missing_context")

    def test_a_plural_without_an_s_counts_things(self):
        self.assertTrue(ada.check_claim(admissions(), "100 women applied.").verified)
        self.assertEqual(ada.check_claim(admissions(), "More than 70 men applied to department A.").reason,
                         "missing_context")


class SummaryColumnTests(unittest.TestCase):
    def test_medians_of_several_groups_are_not_averaged_for_a_claim(self):
        earnings = pd.DataFrame({"sex": ["Women"] * 3 + ["Men"] * 3,
                                 "race": ["All Races", "White", "Black"] * 2,
                                 "median_earn": [894.0, 920.0, 775.0, 1104.0, 1140.0, 860.0]})
        check = ada.check_claim(earnings, "Women's median earnings were $894.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("holds a median for each row's group", check.message)
        one = ada.check_claim(earnings, "Black women's median earnings were $775.")
        self.assertTrue(one.verified)


class NewDataReadingTests(unittest.TestCase):
    def growth(self) -> pd.DataFrame:
        rows = []
        for child in range(6):
            for age in (8, 10, 12, 14):
                rows.append({"Subject": f"S{child}", "age": age, "distance": 20 + child + 0.5 * (age - 8)
                             + (0.25 if age == 12 else 0)})
        return pd.DataFrame(rows)

    def test_a_change_per_year_of_a_column_set_at_a_few_ages_is_its_slope(self):
        frame = self.growth()
        check = ada.check_claim(frame, "The distance increased by about 0.5 mm per year of age.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("slope of distance on age", check.message)
        wrong = ada.check_claim(frame, "The distance increased by about 0.9 mm per year of age.")
        self.assertEqual(wrong.reason, "failed_check")

    def test_n_times_counts_the_rows(self):
        tours = pd.DataFrame({"edition": range(1, 9),
                              "nationality": ["France", "Belgium", "France", "Italy", "Belgium", "Spain",
                                              "France", "Italy"]})
        self.assertTrue(ada.check_claim(tours, "France won 3 times.").verified)
        # "won" names nothing; a count read without it may miss rows it picks, so a mismatch is not failed.
        self.assertEqual(ada.check_claim(tours, "France won 5 times.").reason, "missing_context")

    def test_a_decade_scopes_the_years(self):
        tours = pd.DataFrame({"year": range(2000, 2020),
                              "distance": [3000.0 + 10 * index for index in range(20)]})
        self.assertTrue(ada.check_claim(tours, "The average distance in the 2010s was 3,145 km.").verified)


class ShareOfThingsTests(unittest.TestCase):
    def sightings(self) -> pd.DataFrame:
        # 10 squirrels, one seen twice; 5 of the 10 were seen foraging.
        ids = [f"S{index}" for index in range(10)] + ["S0"]
        return pd.DataFrame({"Unique Squirrel ID": ids, "Foraging": [True] * 5 + [False] * 5 + [True],
                             "Shift": ["AM", "PM"] * 5 + ["AM"]})

    def test_a_share_of_things_with_an_id_is_a_share_of_distinct_ids(self):
        frame = self.sightings()
        check = ada.check_claim(frame, "Half of the squirrels were seen foraging.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("share of distinct Unique Squirrel ID values", " ".join(check.assumptions))
        self.assertEqual(ada.check_claim(frame, "About a third of the squirrels were seen foraging.").reason,
                         "failed_check")

    def test_a_percentage_is_never_compared_with_a_plain_count(self):
        frame = pd.DataFrame({"site": list("abcdefgh"), "visits": [3, 1, 4, 1, 5, 9, 2, 6]})
        check = ada.check_claim(frame, "100% of the sites were visited.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("The claim gives a percentage, but ADA read the words as a count", check.message)

    def test_an_average_of_money_is_never_a_count_of_rows(self):
        cars = pd.DataFrame({"Origin": ["USA"] * 5 + ["non-USA"] * 3,
                             "Price": [18.0, 19.0, 18.5, 18.7, 18.8, 25.0, 30.0, 20.0]})
        check = ada.check_claim(cars, "Cars from USA averaged 18.6 dollars.")
        self.assertNotEqual(check.reason, "failed_check")
        self.assertTrue(check.answer is None or check.answer.plan.intent != "count")

    def test_who_reports_a_claim_is_not_part_of_it(self):
        frame = self.sightings().assign(minutes=[10.0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
        check = ada.check_claim(frame, "Volunteers reported that the average minutes was 60.")
        self.assertTrue(check.verified, check.message)


class BaseAndScaleTests(unittest.TestCase):
    def cars(self) -> pd.DataFrame:
        return pd.DataFrame({"Model": [f"M{index}" for index in range(10)],
                             "Manufacturer": ["Ford", "Ford", "Audi", "Audi", "BMW", "Kia", "Kia", "Fiat",
                                              "Fiat", "Fiat"],
                             "Type": ["Van"] * 3 + ["Small"] * 7,
                             "Rating": [6.0, 7.0, 6.5, 6.5, 7.0, 6.0, 6.5, 6.5, 6.5, 6.5]})

    def test_a_count_of_a_base_checks_both_numbers(self):
        cars = self.cars()
        self.assertTrue(ada.check_claim(cars, "3 of the 10 cars were Van.").verified)
        self.assertEqual(ada.check_claim(cars, "3 of the 12 cars were Van.").reason, "missing_context")

    def test_a_base_opening_the_claim_is_the_rows_in_scope(self):
        cars = self.cars()
        check = ada.check_claim(cars, "The 10 models came from 5 manufacturers.")
        self.assertTrue(check.verified, check.message)
        note = "The claim's 10 models are the 10 Model values in the answer's rows."
        self.assertIn(note, check.assumptions)
        self.assertEqual(ada.check_claim(cars, "The 10 models came from 4 manufacturers.").reason,
                         "failed_check")
        self.assertEqual(ada.check_claim(cars, "The 9 models came from 5 manufacturers.").reason,
                         "missing_context")

    def test_a_lead_base_may_say_where_the_rows_come_from(self):
        cars = self.cars()
        check = ada.check_claim(cars, "Of the 10 cars in the survey, 30% were Van.")
        self.assertTrue(check.verified, check.message)

    def test_a_rating_scale_is_not_a_second_number(self):
        cars = self.cars()
        check = ada.check_claim(cars, "The average rating was 6.5 out of 10.")
        self.assertTrue(check.verified, check.message)

    def test_seven_in_ten_is_a_share(self):
        cars = self.cars()
        self.assertTrue(ada.check_claim(cars, "Seven in ten of the cars were Small.").verified)
        self.assertEqual(ada.check_claim(cars, "Nine in ten of the cars were Small.").reason, "failed_check")
        from ada.claims import _worded_number

        self.assertIn("5%", _worded_number("1 in 20 cars were vans.")[0])


class OtherSpellingTests(unittest.TestCase):
    def test_a_value_written_another_way_too_is_not_a_verdict(self):
        others = [f"Guest {index}" for index in range(60)]
        names = ["John McCain"] * 5 + ["Sen. John McCain"] * 6 + ["Jon Hamm"] * 2 + others
        guests = pd.DataFrame({"guest": names,
                               "group": ["Politician"] * 11 + ["media", "Media"] + ["Actor"] * 60})
        for claim in ("John McCain was a guest 11 times.", "John McCain was a guest 5 times."):
            with self.subTest(claim=claim):
                check = ada.check_claim(guests, claim)
                self.assertEqual(check.reason, "missing_context")
                self.assertIn('"Sen. John McCain"', check.message)
        # A filter matches a value in any capitals, so "media" and "Media" are counted together.
        self.assertEqual(ada.check_claim(guests, "There were 2 rows with group media.").status, "verified")

    def test_a_one_word_value_that_starts_other_values_is_not_a_verdict(self):
        others = [f"Agent {index}" for index in range(60)]
        labels = (["Norovirus"] * 2 + ["Norovirus Genogroup II"] * 3 + ["Salmonella;Norovirus unknown"] * 2
                  + others)
        outbreaks = pd.DataFrame({"Etiology": labels, "Illnesses": range(len(labels))})
        # Norovirus is written seven ways: 2 rows by its own label, 7 in all.
        for claim in ("Norovirus caused 2 outbreaks.", "Norovirus made up 3% of the outbreaks."):
            with self.subTest(claim=claim):
                check = ada.check_claim(outbreaks, claim)
                self.assertEqual(check.reason, "missing_context", check.message)
                self.assertIn('"Norovirus Genogroup II"', check.message)
        # "Smith" is not "John Smith": a one-word value must start the other.
        names = pd.DataFrame({"name": ["Smith"] * 3 + ["John Smith"] * 2 + others})
        self.assertTrue(ada.check_claim(names, "Smith appears 3 times.").verified)

    def test_a_category_holding_another_is_not_another_spelling(self):
        races = pd.DataFrame({"Category": ["Men"] * 4 + ["Wheelchair Men"] * 3 + ["Women"] * 3,
                              "Time": [130.0, 128, 129, 127, 90, 92, 91, 140, 141, 142]})
        self.assertTrue(ada.check_claim(races, "There were 4 rows with Category Men.").verified)

    def test_the_opposite_value_is_not_another_spelling(self):
        cars = pd.DataFrame({"Origin": ["USA"] * 3 + ["non-USA"] * 2, "Price": [10.0, 20, 30, 40, 50]})
        check = ada.check_claim(cars, "The average Price for USA was 20.")
        self.assertTrue(check.verified, check.message)


class AgeBandTests(unittest.TestCase):
    def victims(self) -> pd.DataFrame:
        rows = []
        counts = (("0-9", 10, 990), ("10-19", 20, 980), ("20-49", 30, 970), ("50-64", 40, 460),
                  ("65+", 60, 440))
        for age, died, injured in counts:
            rows += [{"age": age, "result": "Died", "Freq": died},
                     {"age": age, "result": "Injured", "Freq": injured}]
        return pd.DataFrame(rows)

    def test_among_a_group_a_share_is_of_that_group(self):
        victims = self.victims()
        check = ada.check_claim(victims, "Among victims aged 50 and over, 10% died.")
        self.assertTrue(check.verified, check.message)
        self.assertEqual(ada.check_claim(victims, "Among victims aged 50 and over, 3% died.").reason,
                         "failed_check")

    def test_ages_that_cut_a_band_are_not_a_verdict(self):
        victims = self.victims()
        check = ada.check_claim(victims, "Among victims aged 55 and over, 10% died.")
        self.assertNotEqual(check.status, "verified")
        self.assertNotEqual(check.reason, "failed_check")


class TableNameAndBaseTests(unittest.TestCase):
    def characters(self) -> pd.DataFrame:
        return pd.DataFrame({
            "SEX": ["Male Characters"] * 6 + ["Female Characters"] * 2,
            "ALIGN": ["Good Characters", "Bad Characters"] * 4,
        })

    def test_words_of_the_table_s_own_name_pick_no_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "marvel-characters" / "marvel-wikia-data.csv"
            path.parent.mkdir()
            self.characters().to_csv(path, index=False)
            check = ada.check_claim(path, "About a quarter of Marvel's characters are women.")
            self.assertTrue(check.verified, check.message)
            self.assertTrue(any("name of the table" in note for note in check.assumptions))
            other = ada.check_claim(path, "About a quarter of DC's characters are women.")
            self.assertNotEqual(other.status, "verified")

    def test_a_base_the_claim_spans_is_checked_against_distinct_values(self):
        states = [f"s{index}" for index in range(12)]
        frame = pd.DataFrame({"state": states * 2, "year": [1987] * 12 + [1988] * 12,
                              "tax": [0.5] * 24})
        check = ada.check_claim(frame, "In 1988 the tax averaged 0.5 across the 12 states.")
        self.assertTrue(check.verified, check.message)
        wrong = ada.check_claim(frame, "In 1988 the tax averaged 0.5 across the 40 states.")
        self.assertEqual(wrong.reason, "missing_context")

    def test_a_change_between_two_years_of_dated_rows_compares_whole_years(self):
        days = pd.date_range("2023-01-01", periods=10, freq="MS").tolist() + pd.date_range(
            "2024-01-01", periods=15, freq="SMS").tolist()
        frame = pd.DataFrame({"time": days, "depth": [1.0] * 25})
        # 10 rows in 2023 and 15 in 2024: up by half.
        check = ada.check_claim(frame, "The number of rows rose by about half from 2023 to 2024.")
        self.assertTrue(check.verified, check.message)
        third = ada.check_claim(frame, "The number of rows rose by about a third from 2023 to 2024.")
        self.assertEqual(third.reason, "failed_check")
        self.assertEqual(ada.check_claim(frame, "The number of rows rose from 2023 to 2024.").message,
                         "The claim states no number to check.")

    def test_a_change_between_two_years_compares_those_years(self):
        frame = pd.DataFrame({"state": ["tx"] * 4 + ["ca"] * 4, "year": [1982, 1983, 1984, 1985] * 2,
                              "deaths": [100.0, 90, 70, 80, 50, 60, 70, 55]})
        # Texas: 100 in 1982 and 80 in 1985, a fall of 20%; the last two years alone rose by 14%.
        fall = ada.check_claim(frame, "Total deaths for tx fell by 20% between 1982 and 1985.")
        self.assertTrue(fall.verified, fall.message)
        for claim in ("Total deaths for tx rose by 20% between 1982 and 1985.",
                      "Total deaths for tx fell by 14% between 1982 and 1985."):
            with self.subTest(claim=claim):
                self.assertEqual(ada.check_claim(frame, claim).reason, "failed_check")
class ShareBaseTests(unittest.TestCase):
    def victims(self) -> pd.DataFrame:
        # 100 died (80 of them men) and 900 were injured (500 of them men).
        return pd.DataFrame({
            "result": ["Died", "Died", "Injured", "Injured"],
            "gender": ["Male", "Female", "Male", "Female"],
            "Freq": [80, 20, 500, 400],
        })

    def test_a_clause_on_the_base_keeps_the_share_of_that_base(self):
        victims = self.victims()
        for claim in ("80% of the victims who died were men.", "Among victims who died, 80% were men.",
                      "Of the victims who died, 80% were men.", "Men were 80% of the victims who died.",
                      "Men accounted for 80% of the deaths.", "80% of those killed were men."):
            with self.subTest(claim=claim):
                check = ada.check_claim(victims, claim)
                self.assertTrue(check.verified, check.message)
        self.assertEqual(ada.check_claim(victims, "Men were 58% of the victims who died.").reason,
                         "failed_check")

    def test_a_clause_on_what_is_counted_keeps_the_share_of_all_rows(self):
        check = ada.check_claim(self.victims(), "8% of the victims were men who died.")
        self.assertTrue(check.verified, check.message)

    def test_a_past_tense_after_the_group_is_what_is_counted(self):
        victims = self.victims()
        # 80 of the 580 men died.
        self.assertTrue(ada.check_claim(victims, "About 13.8% of male victims died.").verified)
        self.assertEqual(ada.check_claim(victims, "About 8% of male victims died.").reason, "failed_check")

    def test_a_subject_that_is_a_measure_is_not_read_as_a_group(self):
        question, number, _ = read_claim("The tax rate was 45% of the national average.", ["rate"])
        self.assertNotIn("who were", question or "")
        self.assertEqual(number.value, 45)

    def test_a_long_base_lead_is_checked_against_the_share_s_base(self):
        victims = self.victims()
        lead = "people killed or injured on the roads in 1958, 10% died."
        check = ada.check_claim(victims, f"Of the 1,000 {lead}")
        self.assertTrue(check.verified, check.message)
        wrong = ada.check_claim(victims, f"Of the 400 {lead}")
        self.assertEqual(wrong.reason, "missing_context")

    def test_a_word_the_table_lacks_is_never_dropped_silently(self):
        check = ada.check_claim(self.victims(), "Cyclists made up 15% of the people killed.")
        self.assertEqual(check.status, "unverified")
        self.assertNotEqual(check.reason, "failed_check")


class PeakTests(unittest.TestCase):
    def use(self) -> pd.DataFrame:
        return pd.DataFrame({"age": ["18", "19", "20", "21", "22-23"],
                             "cocaine_use": [3.2, 4.1, 4.9, 4.8, 4.5]})

    def test_a_peak_holds_only_where_the_column_is_highest(self):
        frame = self.use()
        check = ada.check_claim(frame, "Cocaine use peaked at 4.9% among 20-year-olds.")
        self.assertTrue(check.verified, check.message)
        elsewhere = ada.check_claim(frame, "Cocaine use peaked at 4.8% among 21-year-olds.")
        self.assertEqual(elsewhere.reason, "failed_check")
        self.assertIn("age = 20", elsewhere.message)
        self.assertEqual(ada.check_claim(frame, "Cocaine use peaked at 6% among 20-year-olds.").reason,
                         "failed_check")

    def test_more_than_any_other_is_a_peak(self):
        frame = pd.DataFrame({"state": ["California", "Texas", "Ohio"] * 2, "year": [1987] * 3 + [1988] * 3,
                              "deaths": [10.0, 30, 20, 50, 40, 30]})
        check = ada.check_claim(frame, "California had more deaths than any other state in 1988, with 50.")
        self.assertTrue(check.verified, check.message)
        wrong = ada.check_claim(frame, "Texas had more deaths than any other state in 1988, with 40.")
        self.assertEqual(wrong.reason, "failed_check")

    def test_the_busiest_year_counts_rows_by_year(self):
        days = ["2017-03-01"] * 3 + ["2018-03-01"] * 5 + ["2019-03-01"] * 4
        frame = pd.DataFrame({"time": pd.to_datetime(days), "depth": [1.0] * 12, "area": ["Vesuvius"] * 12})
        check = ada.check_claim(frame, "Vesuvius had its busiest year on record in 2018, with 5 earthquakes.")
        self.assertTrue(check.verified, check.message)
        wrong = ada.check_claim(frame, "Vesuvius had its busiest year on record in 2019, with 4 earthquakes.")
        self.assertEqual(wrong.reason, "failed_check")
        self.assertIn("2018", wrong.message)

    def test_a_peak_without_a_group_is_never_judged_by_a_mean(self):
        frame = pd.DataFrame({"cocaine_use": [4.0, 4.0, 4.0, 1.0, 9.0]})
        for claim in ("Cocaine use peaked at 4.4%.", "Cocaine use peaked at 9%."):
            with self.subTest(claim=claim):
                self.assertEqual(ada.check_claim(frame, claim).reason, "missing_context")


class MultipleAndFractionTests(unittest.TestCase):
    def counts(self) -> pd.DataFrame:
        return pd.DataFrame({"year": [2021, 2022, 2023], "encounter_count": [100, 250, 640]})

    def test_a_multiple_is_a_change_in_percent(self):
        frame = self.counts()
        for claim in ("Encounters rose more than sixfold between 2021 and 2023.",
                      "Encounters more than doubled between 2021 and 2022."):
            with self.subTest(claim=claim):
                self.assertTrue(ada.check_claim(frame, claim).verified)
        self.assertEqual(ada.check_claim(frame, "Encounters tripled between 2021 and 2023.").reason,
                         "failed_check")

    def test_the_unit_of_a_price_does_not_change_its_change_in_percent(self):
        days = pd.date_range("2022-01-03", periods=104, freq="W-MON")
        frame = pd.DataFrame({"date": days, "price": [4.0] * 52 + [3.56] * 52})
        claim = "The average price of a gallon fell by about 11% between 2022 and 2023."
        check = ada.check_claim(frame, claim)
        self.assertTrue(check.verified, check.message)
        # An amount in a unit the table may not use is still not checked without it.
        amount = ada.check_claim(frame, "The average price of a gallon was 3.56 in 2023.")
        self.assertNotEqual(amount.status, "verified")

    def test_a_fraction_before_a_verb_needs_no_of(self):
        claim = "Among the top third of their class, nearly half went on to college."
        question, number, _ = read_claim(claim, ["Status"])
        self.assertEqual(number.text, "nearly half")
        self.assertIn("top third", question)


class ShareOfBaseTests(unittest.TestCase):
    def test_a_base_inside_a_share_is_checked(self):
        frame = pd.DataFrame({"sex": ["male", "female", "male"], "age": ["young", "young", "old"],
                              "Freq": [40, 30, 30]})
        self.assertTrue(ada.check_claim(frame, "Men made up 70% of the 100 suicides.").verified)
        wrong = ada.check_claim(frame, "Men made up 70% of the 90 suicides.")
        self.assertEqual(wrong.reason, "missing_context")


class KindsBoundsAndRanksTests(unittest.TestCase):
    def test_a_verb_names_the_noun_it_comes_from(self):
        pets = pd.DataFrame({"animal_type": ["dog"] * 6,
                             "outcome_type": ["adoption", "adoption", "rescue"] * 2})
        self.assertTrue(ada.check_claim(pets, "Two thirds of the dogs were adopted.").verified)
        # "returned" is "return" in intake_type and part of "return to owner" in outcome_type: either.
        both = pd.DataFrame({"intake_type": ["return", "return", "stray", "stray"],
                             "outcome_type": ["return to owner", "adoption", "adoption", "rescue"]})
        check = ada.check_claim(both, "Half of the animals were returned.")
        self.assertFalse(check.verified)
        self.assertNotEqual(check.reason, "failed_check")

    def test_of_every_hundred_is_a_share(self):
        birds = pd.DataFrame({"animal_type": ["bird"] * 4 + ["cat"] * 4,
                              "outcome_type": ["euthanasia", "adoption", "adoption", "adoption"] * 2})
        self.assertTrue(ada.check_claim(birds, "Of every 100 birds, about 25 were euthanasia.").verified)
        wrong = ada.check_claim(birds, "Of every 100 birds, about 50 were euthanasia.")
        self.assertEqual(wrong.reason, "failed_check")

    def test_a_bound_beside_the_claimed_number_picks_rows(self):
        quakes = pd.DataFrame({"depth_km": [0.2, 0.4, 0.6, 1.5], "magnitude": [1.0, 1.2, 2.0, 3.0]})
        check = ada.check_claim(quakes, "Quakes with depth_km under 1 had an average magnitude of 1.4.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("depth_km below 1", check.message)
        wrong = ada.check_claim(quakes, "Quakes with depth_km under 1 had an average magnitude of 2.4.")
        self.assertEqual(wrong.reason, "failed_check")
        # A bound that names no column is not read as one.
        self.assertFalse(ada.check_claim(quakes, "The average magnitude of quakes under 1 was 1.4.").verified)

    def test_a_claim_that_names_no_way_to_combine_rows_is_checked_both_ways(self):
        months = pd.DataFrame({"station": ["oxford"] * 12, "year": [2012] * 12, "month": range(1, 13),
                               "rain": [80.0] * 11 + [104.0]})
        # Rain is a reading, averaged by default (82); the year's total, 984, is what the claim states.
        check = ada.check_claim(months, "Oxford got 984 mm of rain in 2012.")
        self.assertEqual(check.reason, "ambiguous_intent")
        self.assertIn("the total is 984", check.message)
        self.assertTrue(ada.check_claim(months, "Oxford got a total of 984 mm of rain in 2012.").verified)
        # Neither the mean nor the total agrees: the claim fails either way.
        self.assertEqual(ada.check_claim(months, "Oxford got 900 mm of rain in 2012.").reason, "failed_check")
        # A claim that says how the rows combine is not second-guessed.
        averaged = ada.check_claim(months, "Oxford averaged 984 mm of rain a month in 2012.")
        self.assertFalse(averaged.verified)
        self.assertNotEqual(averaged.reason, "ambiguous_intent")

    def test_a_peak_said_with_of_any_is_checked_against_every_group(self):
        towns = pd.DataFrame({"town": ["Chorleywood", "Northwood", "Harpenden"], "score": [9.4, 11.9, 11.0]})
        wrong = ada.check_claim(towns, "Chorleywood had the best score of any town, at 9.4.")
        self.assertEqual(wrong.reason, "failed_check")
        self.assertIn("town = Northwood", wrong.message)
        self.assertTrue(ada.check_claim(towns, "Northwood had the best score of any town, at 11.9.").verified)
        unchecked = ada.check_claim(towns, "Chorleywood's score of 9.4 was the highest of any town.")
        self.assertEqual(unchecked.reason, "missing_context")

    def test_no_row_fitting_the_words_proves_nothing(self):
        pets = pd.DataFrame({"animal_type": ["dog", "dog", "cat", "cat"],
                             "color": ["gold", "black", "black", "white"]})
        check = ada.check_claim(pets, "About 30% of the cats were gold.")
        self.assertEqual(check.reason, "missing_context")

    def test_words_of_the_table_s_name_inside_values_name_the_table(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "long-beach-shelter" / "intakes.csv"
            path.parent.mkdir()
            pd.DataFrame({"animal_type": ["cat", "cat", "dog", "dog"],
                          "intake_subtype": ["at shelter", "field", "field", "field"],
                          "outcome_type": ["shelter, neuter, return", "adoption", "adoption", "rescue"]}
                         ).to_csv(path, index=False)
            check = ada.check_claim(path, "Half of the shelter's animals were cats.")
            self.assertTrue(check.verified, check.message)
            self.assertIn("as the name of the table", " ".join(check.assumptions))


class CountsAndPeaksOverTimeTests(unittest.TestCase):
    def names(self) -> pd.DataFrame:
        return pd.DataFrame({"Year": [2015, 2015, 2016, 2015, 2015],
                             "Sex": ["Boy", "Girl", "Boy", "Boy", "Girl"],
                             "Name": ["Jack", "Ines", "Jack", "Rory", "Jack"],
                             "Number": [565, 40, 500, 30, 7]})

    def test_a_count_may_mean_the_rows_or_the_column_of_counts(self):
        names = self.names()
        check = ada.check_claim(names, "In 2015, 565 boys were named Jack.")
        self.assertEqual(check.reason, "ambiguous_intent")
        self.assertIn("Sex = Boy", check.message)
        self.assertIn("565 (sum(Number)", check.message)
        girls = ada.check_claim(names, "In 2015 there were 7 girls called Jack.")
        self.assertEqual(girls.reason, "ambiguous_intent")
        # Neither one row nor 565 babies: false either way.
        self.assertEqual(ada.check_claim(names, "In 2015, 572 boys were named Jack.").reason, "failed_check")

    def test_a_percentage_is_never_a_total(self):
        strikes = pd.DataFrame({"time_of_day": ["Night", "Day", "Day", None], "time": [2105, 830, 1200, 900]})
        check = ada.check_claim(strikes, "Of the strikes where the time of day was recorded, almost a third "
                                         "happened at night.")
        self.assertNotEqual(check.reason, "failed_check", check.message)
        self.assertNotIn("sum(time)", check.message)
        _, number, problem = read_claim("Almost a third happened at night.", ["time_of_day"])
        self.assertEqual(problem, "")
        self.assertAlmostEqual(number.value, 100 / 3, places=3)

    def test_typical_may_be_the_mean_or_the_median(self):
        repairs = pd.DataFrame({"repair_cost": [20000.0, 25000, 27000, 30000, 1000000]})
        check = ada.check_claim(repairs, "The typical repair cost was about $27,000.")
        self.assertEqual(check.reason, "ambiguous_intent")
        self.assertIn("the median 27,000", check.message)
        self.assertEqual(ada.check_claim(repairs, "The typical repair cost was about $100,000.").reason,
                         "failed_check")
        # "average" is the mean alone.
        self.assertTrue(ada.check_claim(repairs, "The average repair cost was about $220,000.").verified)
        even = pd.DataFrame({"repair_cost": [20000.0, 25000, 27000, 30000, 33000]})
        self.assertTrue(ada.check_claim(even, "The typical repair cost was about $27,000.").verified)

    def test_a_peak_in_a_year_is_over_the_years(self):
        rows = [{"station": station, "year": year, "month": month, "rain": rain}
                for station, base in (("oxford", 50.0), ("braemar", 90.0))
                for year, extra in ((2011, 0.0), (2012, 20.0), (2013, 5.0))
                for month, rain in ((month, base + extra) for month in range(1, 13))]
        frame = pd.DataFrame(rows)
        check = ada.check_claim(frame, "Total rain at Oxford peaked at 840 mm in 2012.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("year = 2012 gives the highest", " ".join(check.assumptions))
        wrong = ada.check_claim(frame, "Total rain at Oxford peaked at 660 mm in 2013.")
        self.assertEqual(wrong.reason, "failed_check")
        self.assertIn("year = 2012", wrong.message)
        # The year before the peak could pick the stations of that year as well.
        unclear = ada.check_claim(frame, "Total rain at Oxford in 2012 peaked at 840 mm.")
        self.assertEqual(unclear.reason, "missing_context")


class ChangeEndsTests(unittest.TestCase):
    def votes(self) -> pd.DataFrame:
        years = [1996, 2000, 2004, 2008, 2012, 2016]
        shares = [51.5, 45.6, 43.2, 42.6, 35.5, 26.2, 33.3, 26.3, 26.0, 34.4, 24.7, 27.5]
        return pd.DataFrame({"state": ["West Virginia"] * 6 + ["Utah"] * 6, "year": years * 2,
                             "demVote": shares})

    def test_the_ends_of_a_change_are_checked_too(self):
        votes = self.votes()
        said = "West Virginia's Democratic vote halved in twenty years, from 51.5% in 1996 to {} in 2016."
        check = ada.check_claim(votes, said.format("26.2%"))
        self.assertTrue(check.verified, check.message)
        self.assertIn("last value, 26.2%, agrees", " ".join(check.assumptions))
        self.assertEqual(check.claim, said.format("26.2%"))
        wrong = ada.check_claim(votes, said.format("30%"))
        self.assertEqual(wrong.reason, "failed_check")
        self.assertIn("the data give 26.2", wrong.message)
        last = "West Virginia's Democratic vote halved between 1996 and 2016, to 26.2%."
        self.assertTrue(ada.check_claim(votes, last).verified)

    def test_halved_is_a_fall(self):
        votes = self.votes()
        check = ada.check_claim(votes, "West Virginia's Democratic vote halved between 1996 and 2016.")
        self.assertTrue(check.verified, check.message)
        doubled = ada.check_claim(votes, "West Virginia's Democratic vote doubled between 1996 and 2016.")
        self.assertEqual(doubled.reason, "failed_check")

    def test_ends_alone_are_the_numbers_claimed(self):
        # With no change stated besides its ends, the ends are not set aside, and two numbers are claimed.
        claim = "Utah's Democratic vote rose from 33.3% to 27.5% between 1996 and 2016."
        self.assertFalse(ada.check_claim(self.votes(), claim).verified)


class StressAndDeadlineTests(unittest.TestCase):
    def test_just_and_only_leave_the_number_as_written(self):
        _, number, problem = read_claim("Just 48 women won.", ["gender"])
        self.assertEqual(problem, "")
        self.assertEqual((number.low, number.high), (47.5, 48.5))
        _, number, _ = read_claim("Only 16 episodes showed the sea.", ["sea"])
        self.assertEqual(number.value, 16)

    def test_a_count_in_words_beside_a_bound_is_the_number_claimed(self):
        votes = pd.DataFrame({"state": ["Hawaii", "Utah", "Ohio", "Iowa"] * 2,
                              "year": [2012] * 4 + [2016] * 4,
                              "demVote": [70.6, 24.7, 50.7, 52.0, 62.2, 27.5, 43.6, 41.7]})
        check = ada.check_claim(votes, "In 2012 the Democrats topped 50% in three states.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("demVote above 50", check.message)
        self.assertEqual(ada.check_claim(votes, "In 2016 the Democrats topped 60% in two states.").reason,
                         "failed_check")
        # Sales "topped $5 million" claims more than 5 million.
        _, number, _ = read_claim("Sales topped $5 million.", ["sales"])
        self.assertEqual((number.qualifier, number.low), ("more than", 5e6))

    def test_by_a_year_is_up_to_its_end(self):
        dates = pd.to_datetime(["2014-05-01", "2015-05-01", "2016-05-01", "2017-05-01"])
        frame = pd.DataFrame({"date": dates, "gender": ["Female", "Female", "Male", "Female"]})
        check = ada.check_claim(frame, "2 of the winners by 2016 were women.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("up to 2016", check.message)


class LevelsAndYearsTests(unittest.TestCase):
    def rescues(self, years: tuple[int, ...] = (2019, 2020)) -> Path:
        """A file of rescues, which names its rows: 32 foxes in 2019 and 61 in 2020."""
        dates = [f"2019-{month:02d}-10" for month in [1, 2, 3, 4] * 8] + [
            f"2020-{month:02d}-10" for month in [1, 2, 3, 4, 5, 6, 7] * 8 + [11, 12, 12, 12, 12]]
        dates += ["2019-06-10"] * 10 + ["2020-06-10"] * 10
        animals = ["Fox"] * 93 + ["Cat"] * 20
        costs = [260.0 + 13 * (index % 7) for index in range(113)]
        frame = pd.DataFrame({"date": pd.to_datetime(dates), "animal": animals, "cost": costs})
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "animal_rescues.csv"
        frame[frame["date"].dt.year.isin(years)].to_csv(path, index=False)
        return path

    def test_a_year_pair_is_one_period_not_a_second_number(self):
        _, number, problem = read_claim("In 2021-22, half of all employers reported a gap.", ["employer"])
        self.assertEqual((problem, number.value), ("", 50))
        _, number, problem = read_claim("Fatal accidents fell to 37 in 2000-2014.", ["accidents"])
        self.assertEqual((problem, number.value), ("", 37))
        # A score is no pair of years.
        self.assertIn("more than one number", read_claim("The score was 21-22.", ["score"])[2])

    def test_a_year_after_in_is_not_the_number_claimed(self):
        _, number, problem = read_claim("The number of fox rescues rose in 2020.", ["animal", "date"])
        self.assertIsNone(number)
        self.assertEqual(problem, "The claim states no number to check.")

    def test_a_change_in_a_year_is_against_the_year_before(self):
        frame = self.rescues()
        check = ada.check_claim(frame, "Fox rescues nearly doubled in 2020, from 32 the year before to 61.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("(2020 − 2019)", check.message)
        self.assertIn("number of rescues", " ".join(check.assumptions))
        wrong = ada.check_claim(frame, "Fox rescues nearly doubled in 2020, from 30 the year before to 61.")
        self.assertEqual(wrong.reason, "failed_check")

    def test_a_level_a_change_reached_is_the_value_that_year(self):
        frame = self.rescues()
        self.assertTrue(ada.check_claim(frame, "The number of fox rescues rose to 61 in 2020.").verified)
        check = ada.check_claim(frame, "The number of fox rescues climbed above 50 in 2020.")
        self.assertTrue(check.verified, check.message)
        for claim in ("The number of fox rescues fell to 61 in 2020.",
                      "The number of fox rescues fell below 40 in 2020.",
                      "The number of fox rescues rose 91% to 70 in 2020."):
            self.assertEqual(ada.check_claim(frame, claim).reason, "failed_check", claim)
        check = ada.check_claim(frame, "The number of fox rescues rose 91% to 61 in 2020.")
        self.assertTrue(check.verified, check.message)
        # "rose over 80%" gives the size of the rise.
        self.assertTrue(ada.check_claim(frame, "The number of fox rescues rose over 80% in 2020.").verified)

    def test_a_level_with_no_year_before_it_is_not_verified(self):
        self.assertEqual(ada.check_claim(self.rescues(), "The number of fox rescues rose to 61.").reason,
                         "missing_context")
        check = ada.check_claim(self.rescues((2020,)), "The number of fox rescues rose to 61 in 2020.")
        self.assertEqual(check.reason, "missing_context")

    def test_cases_name_the_one_column_of_counts(self):
        states = ["Alabama", "Arizona", "California", "Colorado", "Florida", "Georgia", "Idaho", "Iowa",
                  "Texas", "Utah", "Oregon", "Ohio"]
        extra = {1951: 1, 1952: 2, 1953: 0}
        rows = [{"disease": "Polio", "state": state, "year": year, "count": 100 * index + extra[year]}
                for index, state in enumerate(states, 1) for year in (1951, 1952, 1953)]
        frame = pd.DataFrame(rows)
        # 1952 gives the most cases, 7,824 over 12 rows; each year has 12 rows.
        check = ada.check_claim(frame, "Polio cases peaked in 1952, at almost 7,900.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("sum(count)", check.message)
        check = ada.check_claim(frame, "Polio cases in the United States peaked in 1952, at almost 7,900.")
        self.assertTrue(check.verified, check.message)
        self.assertNotEqual(ada.check_claim(frame, "Polio cases peaked in 1952, at 12.").status, "verified")
        # The peak is over the cases of each year, not its rows, which tie.
        late = ada.check_claim(frame, "Polio cases peaked in 1953, at almost 7,800.")
        self.assertEqual(late.reason, "failed_check", late.message)


class WentFromAndTimesTests(unittest.TestCase):
    def refugees(self) -> pd.DataFrame:
        return pd.DataFrame({"country": ["Ukraine", "Ukraine", "Syria", "Syria"],
                             "year": [2021, 2022, 2021, 2022],
                             "refugees": [28000, 5700000, 6800000, 6500000]})

    def test_times_as_many_is_a_multiple(self):
        _, number, problem = read_claim("Exports were three times as large.", ["exports"])
        self.assertEqual((problem, number.value, number.percent), ("", 200, True))
        _, number, _ = read_claim("Claims were more than two hundred times as many.", ["claims"])
        self.assertEqual((number.qualifier, number.value), ("more than", 19900))

    def test_went_from_one_value_to_another_checks_the_change_and_its_ends(self):
        frame = self.refugees()
        claim = ("Refugees from Ukraine went from about 28,000 in 2021 to 5.7 million in 2022, {} times as "
                 "many.")
        check = ada.check_claim(frame, claim.format("more than two hundred"))
        self.assertTrue(check.verified, check.message)
        self.assertIn("5.7 million, agrees", " ".join(check.assumptions))
        more = ada.check_claim(frame, claim.format("more than three hundred"))
        self.assertEqual(more.reason, "failed_check")
        wrong = ada.check_claim(frame, claim.format("more than two hundred").replace("28,000", "50,000"))
        self.assertEqual(wrong.reason, "failed_check")


class CountTableClaimTests(unittest.TestCase):
    def miners(self) -> pd.DataFrame:
        ages = ["20-24", "25-29", "50-54", "55-59", "60-64"]
        rows = []
        for age, wheeze in zip(ages, [10, 20, 30, 40, 50], strict=True):
            rows += [{"Wheeze": "W", "Age": age, "Freq": wheeze},
                     {"Wheeze": "NoW", "Age": age, "Freq": 100 - wheeze}]
        return pd.DataFrame(rows)

    def test_a_yes_and_no_column_after_a_verb_keeps_its_yes(self):
        check = ada.check_claim(self.miners(), "Among miners aged 50 to 54, 30% had a wheeze.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("Wheeze = W", check.message)

    def test_an_open_age_reaches_the_oldest_band_and_a_decade_is_its_bands(self):
        frame = self.miners()
        # 50 and over: 120 of 300.
        check = ada.check_claim(frame, "Among miners aged 50 and over, 40% had a wheeze.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("50-54, 55-59, 60-64", check.message)
        self.assertEqual(ada.check_claim(frame, "Half of the miners aged 50 and over had a wheeze.").reason,
                         "failed_check")
        check = ada.check_claim(frame, "Among miners in their fifties, 35% had a wheeze.")
        self.assertTrue(check.verified, check.message)

    def test_no_share_of_the_values_of_a_yes_and_no_column(self):
        frame = self.miners()
        check = ada.check_claim(frame, "All of the miners aged 50 to 54 had Wheeze.")
        self.assertNotEqual(check.status, "verified", check.message)


class NamingNumberTests(unittest.TestCase):
    def test_a_rank_a_chart_a_day_and_a_time_are_not_the_number_claimed(self):
        for claim, value in (("The song spent 53 weeks on the Hot 100.", 53),
                             ("It spent 11 weeks at No. 1.", 11),
                             ("The September 11 attacks killed 182 officers.", 182),
                             ("The FTSE 100 fell 3% in March.", 3),
                             ("About 30% of the stops took place between 10 p.m. and 6 a.m.", 30)):
            _, number, problem = read_claim(claim, ["weeks", "officers", "stops"])
            self.assertEqual((problem, number.value), ("", value), claim)

    def test_ends_with_a_unit_a_month_and_a_record(self):
        from ada.claims import _without_ends

        claim = "The spell tripled, from 8.4 weeks in December 2007 to a record 25.2 weeks in June 2010."
        rest, ends = _without_ends(claim)
        # Each end keeps its unit, which says what kind of number it is.
        self.assertEqual(ends, (("previous", "8.4 weeks"), ("latest", "25.2 weeks")))
        self.assertIn("between 2007 and 2010", rest)


class CountedBoundTests(unittest.TestCase):
    def test_the_bound_that_counts_things_is_the_number_claimed(self):
        frame = pd.DataFrame({"mag": [4.5, 4.7, 5.1, 6.0, 4.4, 7.1]})
        check = ada.check_claim(frame, "More than 3 earthquakes of magnitude 4.5 or above were recorded.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("mag at least 4.5", check.message)
        wrong = ada.check_claim(frame, "More than 10 earthquakes of magnitude 4.5 or above were recorded.")
        self.assertEqual(wrong.reason, "failed_check")


class DateColumnTests(unittest.TestCase):
    def test_a_verdict_another_date_column_turns_is_ambiguous(self):
        started = pd.to_datetime(["2023-12-30", "2024-03-01", "2024-05-01", "2024-07-01"])
        reported = pd.to_datetime(["2024-01-02", "2024-03-02", "2024-05-02", "2024-07-02"])
        frame = pd.DataFrame({"START_DATE": started, "REPORT_DAT": reported, "WARD": [1, 1, 1, 2],
                              "OFFENSE": ["HOMICIDE"] * 4})
        check = ada.check_claim(frame, "Ward 1 recorded 3 homicides in 2024.")
        self.assertEqual(check.reason, "ambiguous_intent", check.message)
        self.assertIn("START_DATE", check.message)
        self.assertIn("REPORT_DAT", check.message)
        # A number no date column gives still fails.
        wrong = ada.check_claim(frame, "Ward 1 recorded 9 homicides in 2024.")
        self.assertEqual(wrong.reason, "failed_check")

    def test_a_column_of_codes_is_never_added_up(self):
        rows = 400
        frame = pd.DataFrame({"WARD": ([1, 2, 3, 4, 5, 6, 7, 8] * 50)[:rows],
                              "DISTRICT": ([1, 2, 3, 4, 5, 6, 7] * 60)[:rows],
                              "PSA": ([101, 102, 201, 202, 301, 302, 401, 402, 501] * 50)[:rows],
                              "XBLOCK": [397000.5 + index * 13.7 for index in range(rows)],
                              "OFFENSE": (["THEFT", "HOMICIDE", "ROBBERY"] * 140)[:rows]})
        check = ada.check_claim(frame, "The ward recorded 15 homicides.")
        self.assertNotEqual(check.status, "verified")
        self.assertNotEqual(check.reason, "failed_check", check.message)


class RecordsAndRatesTests(unittest.TestCase):
    def fertility(self) -> pd.DataFrame:
        years = list(range(1950, 1956))
        rows = [{"Entity": "Korea", "Year": year, "Fertility rate": rate}
                for year, rate in zip(years, [6.0, 5.0, 4.0, 3.0, 2.0, 2.5], strict=True)]
        rows += [{"Entity": "Low-income countries", "Year": year, "Fertility rate": 6.0} for year in years]
        return pd.DataFrame(rows)

    def test_a_record_low_is_the_level_with_the_years_up_to_it(self):
        frame = self.fertility()
        check = ada.check_claim(frame, "Korea's fertility rate fell to a record low of 2 in 1954.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("lowest", " ".join(check.assumptions))
        self.assertNotIn("Low-income", check.message)
        # 3.0 in 1953 was a record low then, whatever came after.
        then = ada.check_claim(frame, "Korea's fertility rate fell to a record low of 3 in 1953.")
        self.assertTrue(then.verified, then.message)
        wrong = ada.check_claim(frame, "Korea's fertility rate rose to a record high of 2.5 in 1955.")
        self.assertEqual(wrong.reason, "failed_check")

    def test_a_record_with_no_word_of_height_follows_the_verb(self):
        frame = self.fertility()
        # "fell to a record 2" is a record low, "rose to a record 2.5" a record high.
        self.assertTrue(ada.check_claim(frame, "Korea's fertility rate fell to a record 2 in 1954.").verified)
        wrong = ada.check_claim(frame, "Korea's fertility rate rose to a record 2.5 in 1955.")
        self.assertEqual(wrong.reason, "failed_check")

    def test_a_count_of_rows_read_through_the_table_s_name_proves_nothing(self):
        rows = [{"year": year, "unit_name": park, "region": "SE", "visitors": visits}
                for year, park, visits in ((2015, "Smoky", 10.7e6), (2016, "Smoky", 11.3e6),
                                           (2015, "Zion", 3.6e6), (2016, "Zion", 4.3e6))]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "park-trips" / "national_parks.csv"
            path.parent.mkdir()
            pd.DataFrame(rows).to_csv(path, index=False)
            check = ada.check_claim(path, "Trips to Smoky numbered 11.3 million in 2016.")
        # The rows are park-years; a count of them (1) says nothing about the trips the folder is named for.
        self.assertEqual(check.reason, "missing_context", check.message)
        self.assertIn("name the table, not its rows", check.message)

    def test_a_change_since_a_year_and_a_multiple_of_a_rate(self):
        frame = self.fertility()
        # 6.0 in 1950 to 2.5 in 1955: a fall of 58%, or of 3.5 points.
        check = ada.check_claim(frame, "Korea's fertility rate has more than halved since 1950.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("1955 − 1950", check.message)
        plain = ada.check_claim(frame, "Korea's fertility rate fell by about 58% since 1950.")
        self.assertEqual(plain.reason, "ambiguous_intent", plain.message)


class RateClaimTests(unittest.TestCase):
    def test_a_rate_for_every_hundred_is_checked_as_totals(self):
        frame = pd.DataFrame({"District": [1, 1, 1, 1, 2, 2, 2, 2],
                              "Holders": [100, 300, 60, 40, 50, 50, 70, 30],
                              "Claims": [10, 20, 6, 4, 15, 5, 8, 12]})
        check = ada.check_claim(frame, "District 2 had about 20 Claims for every 100 Holders.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("sum(Claims) ÷ sum(Holders) × 100", check.message)
        # The mean of the rows' own rates (23.2) is not the rate.
        wrong = ada.check_claim(frame, "District 2 had about 23.2 Claims for every 100 Holders.")
        self.assertEqual(wrong.reason, "failed_check")


class UnitsAndDaysTests(unittest.TestCase):
    def rats(self) -> pd.DataFrame:
        rows = [{"Rat": rat, "Diet": 1 + rat // 3, "Time": day, "weight": 200.0 + 10 * rat + day}
                for rat in range(6) for day in (1, 8, 15, 22)]
        return pd.DataFrame(rows)

    def test_the_word_for_the_units_names_the_rows_not_a_measure(self):
        rats = self.rats()
        check = ada.check_claim(rats, "The heaviest rat reached 272 grams.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("max(weight)", check.message)
        self.assertNotEqual(ada.check_claim(rats, "The heaviest rat reached 5 grams.").status, "verified")

    def test_a_day_written_out_picks_the_rows_of_that_day(self):
        days = ["2011-04-27"] * 3 + ["2011-04-28"] * 2 + ["2012-01-01"]
        storms = pd.DataFrame({"date": pd.to_datetime(days), "deaths": [10, 20, 30, 1, 2, 3]})
        self.assertTrue(ada.check_claim(storms, "There were 3 storms on April 27, 2011.").verified)
        self.assertTrue(ada.check_claim(storms, "Storms killed 60 deaths in total on 27 April 2011.").status
                        in ("verified", "unverified"))
        self.assertEqual(ada.check_claim(storms, "There were 5 storms on April 27, 2011.").reason,
                         "failed_check")


class CountOfThingsTests(unittest.TestCase):
    def test_n_of_the_things_counts_rows(self):
        cats = pd.DataFrame({"tag_id": [f"t{index}" for index in range(8)],
                             "condition": ["Neutered"] * 3 + ["Spayed"] * 4 + ["Not fixed"],
                             "n_cats": [1, 2, 2, 3, 1, 1, 4, 2]})
        self.assertTrue(ada.check_claim(cats, "3 of the cats were neutered.").verified)
        self.assertEqual(ada.check_claim(cats, "5 of the cats were neutered.").reason, "failed_check")


class TotalsOfEndsTests(unittest.TestCase):
    def milk(self) -> pd.DataFrame:
        rows = [("North", "Maine", 1990, 10e6), ("North", "Vermont", 1990, 12e6),
                ("South", "Texas", 1990, 8e6), ("North", "Maine", 2000, 14e6),
                ("North", "Vermont", 2000, 19e6), ("South", "Texas", 2000, 12e6)]
        return pd.DataFrame(rows, columns=["region", "state", "year", "milk_produced"])

    def test_ends_that_count_a_quantity_make_the_change_one_in_the_total(self):
        milk = self.milk()
        # The states total 30 and 45 million pounds; their average goes from 10 to 15 million.
        check = ada.check_claim(milk, "Milk rose 50%, from 30 million pounds in 1990 to 45 million in 2000.")
        self.assertTrue(check.verified, check.message)
        self.assertTrue(any("one in the total" in item for item in check.assumptions))
        wrong = ada.check_claim(milk, "Milk rose 50%, from 20 million pounds in 1990 to 45 million in 2000.")
        self.assertEqual(wrong.reason, "failed_check")
        self.assertIn("20 million pounds", wrong.message)

    def test_ends_of_an_average_stay_an_average(self):
        milk = self.milk()
        check = ada.check_claim(
            milk, "Average milk rose 50%, from 10 million pounds in 1990 to 15 million in 2000."
        )
        self.assertTrue(check.verified, check.message)
        self.assertFalse(any("one in the total" in item for item in check.assumptions))

    def test_ends_the_averages_give_stay_averages(self):
        people = pd.DataFrame({"year": [1990, 1990, 2020, 2020], "weight": [140.0, 160, 170, 190]})
        # The averages are 150 and 180 pounds; the totals, 300 and 360, are no one's weight.
        check = ada.check_claim(people, "Weight rose 20%, from 150 pounds in 1990 to 180 pounds in 2020.")
        self.assertTrue(check.verified, check.message)
        self.assertFalse(any("one in the total" in item for item in check.assumptions))

    def test_a_change_in_the_total_is_asked_with_total(self):
        milk = self.milk()
        answer = ada.analyze(milk).ask("total milk rose between 1990 and 2000")
        self.assertEqual(answer.plan.aggregation, "sum")
        self.assertEqual(dict(answer.figures)["previous"], 30e6)
        self.assertEqual(dict(answer.figures)["latest"], 45e6)

    def test_the_noun_of_a_verb_in_a_column_name_names_the_column(self):
        rows = [("Delta States", "Arkansas", 1970, 900e6), ("Delta States", "Louisiana", 1970, 1000e6),
                ("Delta States", "Arkansas", 2017, 100e6), ("Delta States", "Louisiana", 2017, 150e6),
                ("North", "Maine", 1970, 600e6), ("North", "Maine", 2017, 500e6)]
        milk = pd.DataFrame(rows, columns=["region", "state", "year", "milk_produced"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state_milk_production.csv"
            milk.to_csv(path, index=False)
            check = ada.check_claim(
                path, "Milk production in the Delta states has fallen by about a third since 1970."
            )
        # "production" names milk_produced as well as the table, so a wrong share fails rather than asks.
        self.assertEqual(check.reason, "failed_check", check.message)


class TimeSlopeTests(unittest.TestCase):
    def sales(self) -> pd.DataFrame:
        # Two shops a year; the yearly totals rise by 10 a year, with a wobble that leaves the slopes at 10.
        rows = []
        for index, year in enumerate(range(2000, 2010)):
            wobble = (-1, 1)[index % 2]
            rows += [{"year": year, "shop": "North", "sales": 100.0 + 5 * index + wobble},
                     {"year": year, "shop": "South", "sales": 50.0 + 5 * index - wobble}]
        return pd.DataFrame(rows)

    def test_a_change_each_year_is_the_slope_of_the_yearly_values(self):
        sales = self.sales()
        check = ada.check_claim(sales, "Total sales have grown by about 10 a year since 2000.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("least-squares slope", check.message)
        wrong = ada.check_claim(sales, "Total sales have grown by about 20 a year since 2000.")
        self.assertEqual(wrong.reason, "failed_check")
        # A fall each year that the data show as a rise fails.
        self.assertEqual(ada.check_claim(sales, "Total sales fell by about 10 a year since 2000.").reason,
                         "failed_check")
        # North's sales rise by 5 a year.
        self.assertTrue(ada.check_claim(sales, "North's sales rose by about 5 a year.").verified)

    def test_the_words_after_the_number_can_name_what_changes(self):
        sales = self.sales()
        # "sales" after the number names the column; "added" rises and "lost" falls.
        self.assertTrue(ada.check_claim(sales, "North has added about 5 sales a year.").verified)
        lost = ada.check_claim(sales, "North has lost about 5 sales a year.")
        self.assertEqual(lost.reason, "failed_check")

    def test_a_change_each_decade_is_ten_years_of_slope(self):
        check = ada.check_claim(self.sales(), "Total sales rose by about 100 a decade.")
        self.assertTrue(check.verified, check.message)

    def test_fits_that_disagree_leave_the_claim_open(self):
        years = list(range(2000, 2010))
        values = [10.0, 11, 12, 13, 14, 15, 16, 17, 18, 60]
        # Theil-Sen keeps to 1 a year; least squares, pulled by the last year, gives 3.6.
        frame = pd.DataFrame({"year": years, "value": values})
        check = ada.check_claim(frame, "The value rose by about 1 a year.")
        self.assertEqual(check.reason, "ambiguous_intent", check.message)
        self.assertEqual(len(check.alternatives), 2)

    def test_words_left_out_keep_a_slope_from_verifying(self):
        check = ada.check_claim(self.sales(), "Total retail sales have grown by about 10 a year since 2000.")
        self.assertNotEqual(check.status, "verified")

    def test_one_year_has_no_slope(self):
        frame = pd.DataFrame({"year": [2020, 2020], "value": [1.0, 2.0]})
        self.assertNotEqual(ada.check_claim(frame, "The value rose by about 1 a year.").status, "verified")


class GatheringWordTests(unittest.TestCase):
    def test_how_the_rows_were_gathered_picks_no_rows(self):
        frogs = pd.DataFrame({"Female": [1, 1, 1, 0], "mass": [10.0, 12, 11, 9]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "frogs.csv"
            frogs.to_csv(path, index=False)
            # Every row is a tracked frog; "tracked" says how they were gathered, not which.
            check = ada.check_claim(path, "Three quarters of the tracked frogs were female.")
            wrong = ada.check_claim(path, "Half of the tracked frogs were female.")
        self.assertTrue(check.verified, check.message)
        self.assertEqual(wrong.reason, "failed_check")

    def test_a_word_for_gathering_after_a_verb_stays_a_word(self):
        # GPS fixes of two cats: "how many cats were tracked" asks about the cats, not the fixes.
        fixes = pd.DataFrame({"event_id": range(6), "tag_id": ["a", "a", "a", "b", "b", "b"],
                              "speed": [1.0, 2.0, 0.5, 3.0, 1.5, 2.5]})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cats.csv"
            fixes.to_csv(path, index=False)
            answer = ada.analyze(path).ask("how many cats were tracked")
        self.assertFalse(getattr(answer, "value", None) == 6)


class SummaryNamedColumnTests(unittest.TestCase):
    def pay(self) -> pd.DataFrame:
        return pd.DataFrame({
            "State": ["Ohio", "Ohio", "Iowa", "Iowa"], "Year": [2019, 2020, 2019, 2020],
            "Annual Salary Avg": [70000.0, 72000, 60000, 61000],
            "Annual Salary Median": [68000.0, 70500, 58000, 59500],
            "Wage/Salary standard error (%)": [1.2, 1.1, 2.0, 1.9],
        })

    def test_the_summary_before_a_word_picks_the_column_named_for_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nurses.csv"
            self.pay().to_csv(path, index=False)
            # "median salary" is Annual Salary Median; "earned" says what that column of money holds.
            claim = "Nurses in {} earned {} salary of about ${} in 2020."
            check = ada.check_claim(path, claim.format("Ohio", "a median", "70,500"))
            wrong = ada.check_claim(path, claim.format("Ohio", "a median", "50,000"))
            average = ada.check_claim(path, claim.format("Iowa", "an average", "61,000"))
        self.assertTrue(check.verified, check.message)
        self.assertIn("Annual Salary Median", check.message)
        self.assertEqual(wrong.reason, "failed_check", wrong.message)
        self.assertTrue(average.verified, average.message)


class EveryHundredTests(unittest.TestCase):
    def cases(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Mode": ["Foodborne"] * 50 + ["Infant"] * 50,
            "Toxin": ["E"] * 6 + ["A"] * 44 + ["A"] * 25 + ["B"] * 25,
        })

    def test_so_many_in_every_hundred_is_a_share(self):
        check = ada.check_claim(self.cases(), "Of the Foodborne cases, about 12 in every 100 had Toxin E.")
        self.assertTrue(check.verified, check.message)
        self.assertEqual(check.claimed.value, 12.0)
        check = ada.check_claim(self.cases(), "Of the Foodborne cases, about 30 of every 100 had Toxin E.")
        self.assertEqual(check.reason, "failed_check")

    def test_a_rate_per_hundred_thousand_is_not_read_as_a_share(self):
        check = ada.check_claim(self.cases(), "Toxin E caused about 12 of every 100,000 cases.")
        self.assertFalse(check.verified)
        self.assertNotEqual(check.reason, "failed_check")


class FoundInTests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.folder = Path(holder.name)

    def deaths(self) -> Path:
        path = self.folder / "overdose_deaths.csv"
        pd.DataFrame({
            "Date": [f"{2023 + index % 2}-06-01" for index in range(100)],
            "Fentanyl": ["Y"] * 68 + [None] * 32,
            "Location": ["Found in car"] * 2 + ["Residence"] * 98,
        }).to_csv(path, index=False)
        return path

    def test_a_thing_found_in_a_share_of_rows_is_its_marked_rows(self):
        check = ada.check_claim(self.deaths(), "Fentanyl was found in about 68% of the deaths.")
        self.assertTrue(check.verified, check.message)
        self.assertNotIn("Location", check.answer.calculation)
        check = ada.check_claim(self.deaths(), "Fentanyl was found in about 40% of the deaths.")
        self.assertEqual(check.reason, "failed_check")

    def test_a_description_between_commas_is_left_out(self):
        claim = "Fentanyl, a synthetic opioid, was found in about 68% of the deaths."
        check = ada.check_claim(self.deaths(), claim)
        self.assertTrue(check.verified, check.message)

    def test_a_share_by_a_year_reads_both_ways(self):
        path = self.folder / "overdose_deaths.csv"
        pd.DataFrame({
            "Date": ["2023-06-01"] * 50 + ["2024-06-01"] * 50,
            "Xylazine": [None] * 50 + ["Y"] * 35 + [None] * 15,
        }).to_csv(path, index=False)
        check = ada.check_claim(path, "By 2024, xylazine was found in about 70% of the deaths.")
        self.assertEqual(check.reason, "ambiguous_intent")
        self.assertEqual({item.reading for item in check.alternatives}, {"up to 2024", "in 2024"})


class LostAmountTests(unittest.TestCase):
    def test_a_verb_of_amount_adds_up_what_was_lost(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "farm_losses.csv"
        pd.DataFrame({
            "date": [f"2023-{month:02d}-01" for month in range(1, 13)],
            "species": ["salmon"] * 12,
            "losses": [100.0] * 12,
        }).to_csv(path, index=False)
        check = ada.check_claim(path, "Farms lost about 1,200 salmon in 2023.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("sum(losses)", check.answer.calculation)
        # "on average" asks for the average row.
        check = ada.check_claim(path, "Farms lost 100 salmon on average in 2023.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("mean(losses)", check.answer.calculation)


class BoundBesideShareTests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.path = Path(holder.name) / "overdose_deaths.csv"
        pd.DataFrame({
            "Age": [25, 30, 35, 40, 44, 45, 50, 55, 60, 65],
            "Lag": [3, 5, 10, 12, 19, 20, 25, 30, 40, 60],
        }).to_csv(self.path, index=False)

    def test_an_age_or_younger_bounds_the_rows_a_share_is_of(self):
        check = ada.check_claim(self.path, "Half of the people who died were 44 or younger.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("Age at most 44", check.answer.calculation)

    def test_a_span_written_within_bounds_the_rows(self):
        check = ada.check_claim(self.path, "Half of the deaths came within 19 Lag.")
        self.assertNotEqual(check.reason, "unsupported_analysis", check.message)

    def test_a_share_of_a_group_past_a_bound_is_the_number_claimed(self):
        question, number, problem = read_claim(
            "More than 85% of the babies born to mothers under 20 were born outside marriage.", ["Age"])
        self.assertEqual(problem, "")
        self.assertEqual(number.value, 85)


class BaseFormTests(unittest.TestCase):
    def test_a_share_in_words_of_a_counted_base(self):
        candies = pd.DataFrame({"name": [f"c{index}" for index in range(10)],
                                "hard": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]})
        check = ada.check_claim(candies, "Fewer than one in four of the 10 candies is hard.")
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(candies, "Fewer than one in four of the 12 candies is hard.")
        self.assertEqual(check.reason, "missing_context")

    def test_a_possessive_before_the_base_count(self):
        _, base = _based("Of the District's 187 homicides in 2024, four in five were committed with a gun.")
        self.assertEqual(base, ("187", "homicides in 2024"))


class PlaceScopeTests(unittest.TestCase):
    def test_a_country_no_column_holds_is_the_whole_table(self):
        births = pd.DataFrame({"Year": [2021, 2022, 2023], "Births": [90, 100, 120]})
        check = ada.check_claim(births, "Births in Ireland rose by 20% between 2022 and 2023.")
        self.assertTrue(check.verified, check.message)
        self.assertTrue(any("the place the whole table covers" in item for item in check.assumptions))

    def test_a_country_beside_a_column_of_countries_names_other_rows(self):
        births = pd.DataFrame({"Year": [2022, 2023] * 3,
                               "Country": ["France"] * 2 + ["Spain"] * 2 + ["Italy"] * 2,
                               "Births": [100, 120, 50, 60, 70, 84]})
        check = ada.check_claim(births, "Births in Ireland rose by 20% between 2022 and 2023.")
        self.assertFalse(check.verified)


class HourlyRowTests(unittest.TestCase):
    def prices(self) -> pd.DataFrame:
        hours = pd.date_range("2025-01-01", periods=48, freq="h")
        return pd.DataFrame({"Datetime": hours, "Price": [-5.0] * 11 + [40.0] * 37})

    def test_hours_in_a_table_of_one_reading_an_hour_are_rows(self):
        check = ada.check_claim(self.prices(), "Prices were negative for 11 hours in 2025.")
        self.assertTrue(check.verified, check.message)
        self.assertTrue(any("each row of the table is one hour" in item for item in check.assumptions))
        self.assertEqual(ada.check_claim(self.prices(), "Prices were negative for 30 hours in 2025.").reason,
                         "failed_check")

    def test_hours_in_a_table_of_several_series_are_no_rows(self):
        frame = pd.concat([self.prices().assign(Country="DE"), self.prices().assign(Country="FR")])
        check = ada.check_claim(frame, "Prices were negative for 22 hours in 2025.")
        self.assertFalse(check.verified)


class LongTableTests(unittest.TestCase):
    def test_an_indicator_named_says_what_the_value_holds(self):
        frame = pd.DataFrame({
            "Indicator Name": ["Smoking prevalence"] * 6 + ["Obesity prevalence"] * 6,
            "Year": list(range(2020, 2026)) * 2,
            "Value": [20.0, 19.0, 18.0, 17.0, 16.0, 15.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        })
        check = ada.check_claim(frame, "Smoking prevalence fell by about 1 point a year since 2020.")
        self.assertTrue(check.verified, check.message)
        note = "Read Value as the value of Indicator Name"
        self.assertTrue(any(note in item for item in check.assumptions))

    def test_a_fraction_of_a_point_is_a_number_of_points(self):
        _, number, problem = read_claim("Smoking fell by about three quarters of a percentage point a year.",
                                        ["Value"])
        self.assertEqual(problem, "")
        self.assertEqual(number.value, 0.75)


class FractionChangeTests(unittest.TestCase):
    def test_a_change_by_several_parts_is_a_fraction(self):
        for claim, low, high in (("Visitors fell by more than three quarters in 2020.", 75, math.inf),
                                 ("Sales fell by two thirds between 2022 and 2024.", 64.5, 68.4)):
            _, number, problem = read_claim(claim, ["Visitors", "Sales"])
            self.assertEqual(problem, "")
            self.assertAlmostEqual(number.low, low, places=0)
            self.assertTrue(number.high == high or abs(number.high - high) < 0.1)

    def test_a_fall_of_three_quarters_is_not_more_than_three(self):
        frame = pd.DataFrame({"Year": [2019, 2020], "Visitors": [100.0, 90.0]})
        check = ada.check_claim(frame, "Visitors fell by more than three quarters in 2020.")
        self.assertEqual(check.reason, "failed_check")


class CurrencyColumnTests(unittest.TestCase):
    def rates(self) -> pd.DataFrame:
        days = pd.date_range("2020-01-01", "2025-12-31", freq="7D")
        years = days.year - 2020
        return pd.DataFrame({"Date": days, "USD": 1.10 + 0.01 * years, "JPY": 120.0 + 10.0 * years,
                             "DKK": 7.45, "NOK": 10.0 + 0.3 * years})

    def test_a_currency_after_the_number_names_its_rate_column(self):
        check = ada.check_claim(self.rates(), "Between 2020 and 2025 the euro gained about 10 yen a year.")
        self.assertIn("JPY", check.answer.calculation)
        check = ada.check_claim(self.rates(), "In 2025 the euro averaged about $1.15.")
        self.assertIn("USD", check.answer.calculation)

    def test_a_currency_two_columns_hold_names_neither(self):
        claim = "In 2025 a euro was worth about 7.45 krone on average."
        check = ada.check_claim(self.rates(), claim)
        self.assertFalse(check.verified)

    def test_an_all_time_high_is_checked_against_the_column(self):
        rates = self.rates()
        last = rates["Date"].iloc[-1].strftime("%d %B %Y").lstrip("0")
        top = ada.check_claim(rates, f"The yen hit its all-time high on {last}, at 170.")
        self.assertTrue(top.verified, top.message)
        first = rates["Date"].iloc[0].strftime("%d %B %Y").lstrip("0")
        low = ada.check_claim(rates, f"The yen hit its all-time high on {first}, at 120.")
        self.assertEqual(low.reason, "failed_check")


class WordedAmountTests(unittest.TestCase):
    def prices(self) -> pd.DataFrame:
        hours = pd.date_range("2024-01-01", periods=2000, freq="h")
        return pd.DataFrame({"Datetime": hours, "Price": [-1.0] * 300 + [50.0] * 1700})

    def test_a_thousand_is_one_thousand(self):
        claim = "Prices dropped below zero in more than a thousand hours in 2024."
        check = ada.check_claim(self.prices(), claim)
        self.assertEqual(check.reason, "failed_check")
        check = ada.check_claim(self.prices(), "Prices dropped below zero in 300 hours in 2024.")
        self.assertTrue(check.verified, check.message)

    def test_a_multiple_of_an_earlier_year_is_a_change(self):
        frame = pd.DataFrame({"Year": [2019, 2020, 2021, 2022, 2023],
                              "Price": [40.0, 30.0, 90.0, 230.0, 100.0]})
        check = ada.check_claim(frame, "Prices in 2023 averaged 2.5 times their 2019 level.")
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(frame, "Prices in 2023 averaged five times their 2019 level.")
        self.assertEqual(check.reason, "failed_check")


class RepeatedDateTests(unittest.TestCase):
    def test_a_repeated_month_that_changes_the_verdict_is_ambiguous(self):
        months = list(pd.date_range("2020-01-01", periods=36, freq="MS"))
        frame = pd.DataFrame({"Date": months + [months[5]], "Sales": [100.0] * 37})
        check = ada.check_claim(frame, "Total Sales were 1,200 in 2020.")
        self.assertEqual(check.reason, "ambiguous_intent")
        self.assertTrue(any(item.reading == "each date once" and item.agrees for item in check.alternatives))

    def test_a_repeat_that_changes_nothing_is_noted(self):
        months = list(pd.date_range("2020-01-01", periods=36, freq="MS"))
        frame = pd.DataFrame({"Date": months + [months[30]], "Sales": [100.0] * 37})
        check = ada.check_claim(frame, "Total Sales were 1,200 in 2020.")
        self.assertTrue(check.verified, check.message)
        note = "counting each date once gives the same verdict"
        self.assertTrue(any(note in item for item in check.assumptions))


class RecordedScaleTests(unittest.TestCase):
    def migration(self) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "migration.csv"
        migration().to_csv(path, index=False)
        return path

    def test_a_unit_column_gives_the_scale_of_the_counts(self):
        path = self.migration()
        check = ada.check_claim(path, "About 40,000 people emigrated in 2024.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("UNIT gives VALUE in Thousand, so 40 is 40,000.", check.assumptions)
        self.assertEqual(ada.check_claim(path, "More than 50,000 people emigrated in 2024.").reason,
                         "failed_check")

    def test_the_ends_of_a_change_are_scaled_and_share_their_about(self):
        path = self.migration()
        check = ada.check_claim(path, "Immigration rose by about two thirds between 2023 and 2024, from "
                                      "about 60,000 to 100,000.")
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(path, "Immigration rose by about two thirds between 2023 and 2024, from "
                                      "about 60,000 to 130,000.")
        self.assertEqual(check.reason, "failed_check")

    def test_words_that_only_describe_a_checked_base(self):
        path = self.migration()
        check = ada.check_claim(path, "People aged 25 to 44 made up half of the 100,000 immigrants who "
                                      "arrived in 2024.")
        self.assertTrue(check.verified, check.message)
        self.assertIn('"arrived" only describe the 100,000 immigrants', check.message)
        check = ada.check_claim(path, "People aged 25 to 44 made up 70% of the 100,000 immigrants who "
                                      "arrived in 2024.")
        self.assertEqual(check.reason, "failed_check")
        # A base the share is not of leaves the words unchecked.
        check = ada.check_claim(path, "People aged 25 to 44 made up half of the 90,000 immigrants who "
                                      "arrived in 2024.")
        self.assertEqual(check.reason, "missing_context")

    def test_the_year_to_a_month_on_a_table_of_years(self):
        path = self.migration()
        check = ada.check_claim(path, "About 40,000 people emigrated in the year to April 2024.")
        self.assertTrue(check.verified, check.message)
        self.assertTrue(any("as 2024, since the table gives one figure a year" in item
                            for item in check.assumptions))


class BirthTableClaimTests(unittest.TestCase):
    def path(self) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        folder = Path(holder.name) / "canada-births"
        folder.mkdir()
        births().to_csv(folder / "13100416.csv", index=False)
        return folder / "13100416.csv"

    def test_a_share_of_the_babies_born_is_of_the_counts(self):
        path = self.path()
        check = ada.check_claim(path, "Among the 100 babies born in 2025, more than a third had mothers aged "
                                      "35 or older.")
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(path, "More than 60% of the babies born in 2025 had mothers aged 20 to 34.")
        self.assertEqual(check.reason, "failed_check")

    def test_teenage_mothers_and_women_in_their_early_forties(self):
        path = self.path()
        check = ada.check_claim(path, "In Nunavut, about 20 of every 100 babies born in 2025 had a teenage "
                                      "mother.")
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(path, "Births to women in their early forties doubled between 2024 and 2025.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("Added up VALUE, since each row of the table is a count.", check.assumptions)


class DisruptionClaimTests(unittest.TestCase):
    def path(self) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        folder = Path(holder.name) / "dutch-rail-disruptions"
        folder.mkdir()
        starts = pd.date_range("2025-07-01 06:00", periods=10, freq="h")
        pd.DataFrame({
            "rdt_id": range(10),
            "cause_en": ["broken down train"] * 4 + ["signal failure"] * 3 + ["copper theft"] * 3,
            "statistical_cause_en": ["broken down train"] * 5 + ["signal failure"] * 2 + ["copper theft"] * 3,
            "start_time": [*starts[:8], pd.Timestamp("2025-07-02 09:00"), pd.Timestamp("2025-07-02 10:00")],
            # Two of the 1 July disruptions run past midnight.
            "end_time": [*(starts[:6] + pd.Timedelta(hours=1)), pd.Timestamp("2025-07-02 01:00"),
                         pd.Timestamp("2025-07-02 02:00"), pd.Timestamp("2025-07-02 10:00"),
                         pd.Timestamp("2025-07-02 11:00")],
        }).to_csv(folder / "disruptions-2025.csv", index=False)
        return folder / "disruptions-2025.csv"

    def test_a_value_every_fitting_column_agrees_on(self):
        path = self.path()
        # 4 of 10 by cause_en and 5 of 10 by statistical_cause_en: more than a third either way.
        check = ada.check_claim(path, "Broken-down trains caused more than a third of the disruptions on the "
                                      "Dutch railways in 2025.")
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(path, "Broken-down trains caused more than 60% of the disruptions on the "
                                      "Dutch railways in 2025.")
        self.assertEqual(check.reason, "failed_check")

    def test_a_day_counts_what_started_on_it(self):
        path = self.path()
        self.assertTrue(ada.check_claim(path, "There were 8 disruptions on 1 July 2025.").verified)


class QualifierBoundTests(unittest.TestCase):
    def test_nearly_a_fraction_reaches_down_to_the_next_fraction(self):
        frame = pd.DataFrame({"park": ["Smokies"] * 47 + ["Other"] * 53})
        self.assertTrue(ada.check_claim(frame, "Smokies account for nearly half of the rows.").verified)
        self.assertEqual(ada.check_claim(frame, "Smokies account for nearly a third of the rows.").reason,
                         "failed_check")

    def test_a_base_in_round_thousands_is_rounded(self):
        frame = pd.DataFrame({"park": ["Smokies"] * 470 + ["Other"] * 542})
        check = ada.check_claim(frame, "Smokies account for nearly half of the 1,000 records.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("rounded", " ".join(check.assumptions))

    def test_every_value_of_a_column_spans_the_table(self):
        from ada.claims import _every_value  # noqa: PLC0415

        frame = pd.DataFrame({"ParkName": ["A", "B", "C", "A"], "birds": [1, 2, 3, 4]})
        text, note = _every_value("Between them, the 3 parks list 4 birds.", frame)
        self.assertEqual(text, "Between them, list 4 birds.")
        self.assertIn("every value of ParkName", note)
        # A count that is not the column's number of values stays in the claim.
        self.assertEqual(_every_value("the 5 parks list 4 birds", frame), ("the 5 parks list 4 birds", None))


class DayClaimTests(unittest.TestCase):
    def path(self) -> Path:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        folder = Path(holder.name) / "dutch-rail-disruptions"
        folder.mkdir()
        # By hand: 3 disruptions on 1 July 2025, 2 on 2 July and 1 on 3 July, lasting 10, 20, 30, 40, 50
        # and 60 minutes (median 35).
        pd.DataFrame({
            "start_time": pd.to_datetime(["2025-07-01 06:00", "2025-07-01 07:00", "2025-07-01 08:00",
                                          "2025-07-02 06:00", "2025-07-02 07:00", "2025-07-03 06:00"]),
            "duration_minutes": [10, 20, 30, 40, 50, 60],
        }).to_csv(folder / "disruptions-2025.csv", index=False)
        return folder / "disruptions-2025.csv"

    def test_the_worst_day_is_checked_against_every_day(self):
        path = self.path()
        check = ada.check_claim(path, "The first of July 2025 was the worst day of the year, with three "
                                      "disruptions.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("has the most of any day of 2025", " ".join(check.assumptions))
        check = ada.check_claim(path, "The second of July 2025 was the worst day of the year, with two "
                                      "disruptions.")
        self.assertEqual(check.reason, "failed_check")
        self.assertIn("is not the worst day of 2025", check.message)

    def test_half_within_a_time_is_the_median(self):
        path = self.path()
        check = ada.check_claim(path, "Half of all disruptions were over within 35 minutes.")
        self.assertTrue(check.verified, check.message)
        self.assertEqual(ada.check_claim(path, "Half of all disruptions were over within 20 minutes.").reason,
                         "failed_check")

    def test_a_day_in_words(self):
        from ada.nlq import days_in_words  # noqa: PLC0415

        self.assertEqual(days_in_words("the twenty-third of May and the first of July"), "23 May and 1 July")


class MoneyColumnTests(unittest.TestCase):
    def test_a_word_for_money_picks_the_column_it_names(self):
        # invest and value are both money; "investment" names invest. By hand GM invested 60 of 100.
        frame = pd.DataFrame({"firm": ["General Motors", "US Steel", "GE", "General Motors"],
                              "invest": [40.0, 30.0, 10.0, 20.0], "value": [500.0, 100.0, 300.0, 100.0]})
        check = ada.check_claim(frame, "General Motors accounted for about 60% of all the investment by the "
                                       "3 firms.")
        self.assertTrue(check.verified, check.message)
        self.assertIn("sum(invest)", check.answer.calculation)


class OtherUnitTests(unittest.TestCase):
    def test_a_claim_in_another_unit_of_a_column_with_none(self):
        # MSL is in metres, though its name does not say so: 0.58 in 1924 and 0.98 in 2024, a rise of 0.40.
        frame = pd.DataFrame({"Year": [1924, 2024], "MSL": [0.58, 0.98]})
        check = ada.check_claim(frame, "MSL rose by about 40 centimetres between 1924 and 2024.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("only if MSL is recorded in metres", check.message)
        # A bound that another unit would make true is not refuted either.
        check = ada.check_claim(frame, "MSL rose by more than 30 centimetres between 1924 and 2024.")
        self.assertEqual(check.reason, "missing_context")
        self.assertTrue(ada.check_claim(frame, "MSL rose by about 0.4 between 1924 and 2024.").verified)


class EveryRowShareTests(unittest.TestCase):
    def test_a_share_of_every_row_says_nothing(self):
        # "Home teams" reads as the distinct HomeTeam values, every one of them in the years named: 100%.
        frame = pd.DataFrame({
            "Date": pd.date_range("1963-08-24", periods=12, freq="30D"),
            "HomeTeam": ["Koeln", "Bremen", "Hamburg"] * 4, "AwayTeam": ["Bremen", "Hamburg", "Koeln"] * 4,
            "HomeGoals": [2, 1, 0, 3, 1, 1, 2, 0, 0, 1, 2, 2],
            "AwayGoals": [1, 1, 2, 0, 0, 1, 2, 1, 0, 3, 1, 0],
        })
        check = ada.check_claim(frame, "Home teams won 52% of matches between 1963 and 1964.")
        self.assertNotEqual(check.reason, "failed_check", check.message)


class RegistryClaimTests(unittest.TestCase):
    def test_a_share_of_people_diagnosed_is_of_the_incidences(self):
        claim = "Women made up {}% of the people diagnosed with lung cancer in 2024."
        check = ada.check_claim(registry(), claim.format(52))
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(registry(), claim.format(70))
        self.assertEqual(check.reason, "failed_check")

    def test_a_share_of_a_site_is_of_all_cancer_types(self):
        claim = "Breast cancer made up about 40% of the cancers diagnosed in women in 2024."
        self.assertEqual(ada.check_claim(registry(), claim).reason, "failed_check")
        claim = "Breast cancer made up about 53% of the cancers diagnosed in women in 2024."
        self.assertTrue(ada.check_claim(registry(), claim).verified)

    def test_ages_over_columns_of_age_bands(self):
        claim = "More than a third of the cancers diagnosed in 2024 were in people aged 75 or over."
        check = ada.check_claim(registry(), claim)
        self.assertTrue(check.verified, check.message)
        self.assertTrue(any("columns of counts by age band" in item for item in check.assumptions))
        claim = "Fewer than a quarter of the cancers diagnosed in 2024 were in people aged 75 or over."
        self.assertEqual(ada.check_claim(registry(), claim).reason, "failed_check")

    def test_a_change_in_diagnoses_of_one_site(self):
        claim = "Prostate cancer diagnoses rose by 20% between 2020 and 2024, from 80 to 96."
        self.assertTrue(ada.check_claim(registry(), claim).verified)


class SeaLevelClaimTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        years = list(range(1856, 2025))
        return pd.DataFrame({"Year": years, "MSL": [0.4 + 0.003 * (year - 1856) for year in years],
                             "MHW": [1.1 + 0.003 * (year - 1856) for year in years]})

    def test_since_records_began_is_from_their_first_year(self):
        claim = "Sea level has risen by about 0.003 metres a year since records began in 1856."
        check = ada.check_claim(self.frame(), claim)
        self.assertTrue(check.verified, check.message)
        self.assertTrue(any('"sea level" as msl' in item.lower() for item in check.assumptions))

    def test_a_unit_the_data_do_not_state_is_not_assumed(self):
        claim = "Sea level has risen by about 3 millimetres a year since records began."
        check = ada.check_claim(self.frame(), claim)
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("only if MSL is recorded in metres", check.message)

    def test_a_comparison_after_the_number_fails_a_wrong_number(self):
        claim = ("Since 2000, sea level has been rising by about 0.009 metres a year, three times the "
                 "long-run pace.")
        check = ada.check_claim(self.frame(), claim)
        self.assertEqual(check.reason, "failed_check")
        self.assertTrue(any("Set aside" in item for item in check.assumptions))

    def test_a_comparison_after_the_number_does_not_verify_a_right_one(self):
        claim = ("Since 2000, sea level has been rising by about 0.003 metres a year, twice the pace of the "
                 "1800s.")
        check = ada.check_claim(self.frame(), claim)
        self.assertFalse(check.verified)
        self.assertIn("ADA does not check", check.message)

    def test_words_of_the_table_s_name_that_name_a_column_stay(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "battery-sea-level" / "monthly.csv"
            path.parent.mkdir()
            self.frame().to_csv(path, index=False)
            check = ada.check_claim(path, "Mean sea level at the Battery was about 0.904 metres in 2024.")
        self.assertTrue(check.verified, check.message)


class DeathsTableClaimTests(unittest.TestCase):
    def test_a_share_of_single_years_of_age_is_of_all_ages(self):
        # 2024: 494 deaths, 215 of them aged 40 or over (43.5%) and 139 under 20 (28.1%).
        claim = "More than {} of the people who died in 2024 were aged 40 or over."
        check = ada.check_claim(deaths(), claim.format("two fifths"))
        self.assertTrue(check.verified, check.message)
        check = ada.check_claim(deaths(), claim.format("half"))
        self.assertEqual(check.reason, "failed_check")

    def test_how_deaths_were_counted_leaves_no_word_out(self):
        claim = "Nearly a third of the deaths registered in 2024 were of people under 20."
        check = ada.check_claim(deaths(), claim)
        self.assertEqual(check.reason, "failed_check")
        self.assertFalse(any("registered" in item for item in check.assumptions))

    def test_a_count_of_people_adds_up_the_counts(self):
        check = ada.check_claim(deaths(), "More than 1,000 people died in 2024.")
        self.assertEqual(check.reason, "failed_check")
        self.assertIn("494", check.message)
        self.assertFalse(any("Each row counts" in item for item in check.assumptions))

    def test_sexagenarians_are_aged_60_to_69(self):
        check = ada.check_claim(deaths(), "Deaths of sexagenarians rose by about 17% between 2023 and 2024, "
                                          "from 59 to 69.")
        self.assertTrue(check.verified, check.message)


class CountedThingsClaimTests(unittest.TestCase):
    def trees(self, folder: str) -> Path:
        path = Path(folder) / "trees_in_town.csv"
        pd.DataFrame({
            "Number Of Trees": [1, 2, 1, 3, 1, 1],
            "Capital Asset Value For Amenity Trees": [100.0, 200.0, 50.0, 10.0, 5.0, 5.0],
            "Common Name": ["Oak", "Ash", "Oak", "Elm", "Oak", "Ash"],
        }).to_csv(path, index=False)
        return path

    def test_a_share_of_the_file_s_things_adds_up_their_count(self):
        with tempfile.TemporaryDirectory() as folder:
            # 3 of the 9 trees are oaks.
            check = ada.check_claim(self.trees(folder), "Oaks make up half of the town's trees.")
            self.assertEqual(check.reason, "failed_check")
            self.assertIn("sum(Number Of Trees)", check.message)
            check = ada.check_claim(self.trees(folder), "Oaks make up a third of the town's trees.")
            self.assertTrue(check.verified, check.message)

    def test_an_amount_of_money_is_never_a_count(self):
        frame = pd.DataFrame({"Number Of Trees": [1, 2, 1], "Common Name": ["Oak", "Ash", "Oak"]})
        check = ada.check_claim(frame, "There are trees worth £4 here.")
        self.assertFalse(check.verified)
        self.assertNotEqual(check.reason, "failed_check")


class RateBaseClaimTests(unittest.TestCase):
    def cars(self) -> pd.DataFrame:
        # Cars per thousand inhabitants: Poland rises by 15 a year from 300 in 2000; Germany 593 in 2025.
        rows = [{"STRUCTURE_NAME": "Passenger cars - per thousand inhabitants", "geo": name,
                 "TIME_PERIOD": year, "OBS_VALUE": value}
                for name, values in {"Poland": [300 + 15 * step for step in range(26)],
                                     "Germany": [500 + step * 3.72 for step in range(26)]}.items()
                for year, value in zip(range(2000, 2026), values, strict=True)]
        frame = pd.DataFrame(rows)
        frame.loc[(frame["geo"] == "Germany") & (frame["TIME_PERIOD"] == 2025), "OBS_VALUE"] = 593.0
        return frame

    def test_a_base_the_table_states_is_its_figures_own(self):
        claim = "Since 2000, Poland has added about 15 cars per 1,000 inhabitants a year."
        check = ada.check_claim(self.cars(), claim)
        self.assertTrue(check.verified, check.message)
        self.assertTrue(any("per thousand inhabitants" in item for item in check.assumptions))

    def test_things_a_row_s_figure_counts_are_not_rows(self):
        claim = "Germany had about {} cars for every 1,000 residents in 2025."
        self.assertTrue(ada.check_claim(self.cars(), claim.format(590)).verified)
        check = ada.check_claim(self.cars(), claim.format(700))
        self.assertEqual(check.reason, "failed_check")
        self.assertIn("593", check.message)

    def test_a_rate_is_never_added_up_over_rows(self):
        claim = "The two countries had about 1,500 cars per 1,000 inhabitants in 2025."
        check = ada.check_claim(self.cars(), claim)
        self.assertFalse(check.verified)


class BeyondEveryRowTests(unittest.TestCase):
    def traffic(self) -> pd.DataFrame:
        hours = pd.date_range("2024-01-01", periods=48, freq="h")
        return pd.DataFrame({"date_time": hours, "traffic_volume": [100.0 + hour for hour in range(48)]})

    def test_a_highest_beyond_every_row_fails_whatever_words_are_left_out(self):
        # The busiest hour has 147 vehicles.
        claim = "At its busiest, traffic on this stretch topped {} vehicles an hour."
        self.assertEqual(ada.check_claim(self.traffic(), claim.format(200)).reason, "failed_check")
        self.assertEqual(ada.check_claim(self.traffic(), claim.format(140)).reason, "missing_context")

    def test_a_span_longer_than_a_row_is_not_refuted(self):
        claim = "At its busiest, traffic on this stretch topped 200 vehicles a week."
        self.assertNotEqual(ada.check_claim(self.traffic(), claim).reason, "failed_check")

    def test_digits_after_a_letter_and_a_hyphen_are_a_name(self):
        claim = "At its busiest, traffic on I-94 topped 200 vehicles an hour."
        check = ada.check_claim(self.traffic(), claim)
        self.assertEqual(check.claimed.value, 200)


class PartsClaimTests(unittest.TestCase):
    def test_a_state_on_a_share_of_days_is_a_share_of_the_days(self):
        frame = air_quality()
        claim = "Air quality was good on more than 90% of the days monitored across US counties in 2025."
        check = ada.check_claim(frame, claim)
        self.assertEqual(check.reason, "failed_check")
        self.assertAlmostEqual(check.compared, 610 / 1810 * 100)
        claim = "Air quality was good on about 34% of the days in 2025."
        self.assertTrue(ada.check_claim(frame, claim).verified)

    def test_a_place_in_the_subject_picks_its_rows(self):
        frame = air_quality()
        for claim in ("Air quality in Maricopa County was good on 8.8% of days in 2025.",
                      "In Maricopa County, air was good on fewer than 10% of days in 2025.",
                      "Maricopa County, home to Phoenix, had good air on fewer than one day in ten in 2025."):
            check = ada.check_claim(frame, claim)
            self.assertTrue(check.verified, claim)
            self.assertAlmostEqual(check.compared, 32 / 365 * 100)

    def test_one_thing_in_ten_is_a_share_of_those_things(self):
        question, number, _ = read_claim("Fewer than one day in ten was good.", ["Good Days"])
        self.assertEqual(question, "share of days was good")
        self.assertEqual(number.value, 10)
        # Another noun after the fraction is what the share is of: no rewrite.
        self.assertEqual(read_claim("He scored one goal in ten games.", ["goals"])[1].value, 10)

    def test_a_share_of_residents_is_of_the_total_column(self):
        claim = ("Two thirds of the City of London's working residents worked mainly from home at the time "
                 "of the 2021 census.")
        check = ada.check_claim(commuting(), claim)
        self.assertTrue(check.verified)
        self.assertAlmostEqual(check.compared, 3602 / 5352 * 100)


class CountsByAgeClaimTests(unittest.TestCase):
    def test_an_age_bound_after_the_people_counted_keeps_their_ages(self):
        frame = covid_deaths()
        check = ada.check_claim(frame, "Fewer than 100 Americans under 18 died of COVID-19.")
        self.assertEqual(check.reason, "failed_check")
        self.assertEqual(check.compared, 351)
        self.assertTrue(ada.check_claim(frame, "About 350 Americans under 18 died of COVID-19.").verified)

    def test_a_word_that_names_two_columns_of_counts_is_ambiguous_when_they_disagree(self):
        frame = covid_deaths()
        claim = "Women accounted for 47.5% of the deaths recorded from January 2020 to June 2023."
        check = ada.check_claim(frame, claim)
        # Of all deaths 16,510 of 34,760 (47.50%), of COVID-19 deaths 605 of 1,645 (36.78%).
        self.assertEqual(check.reason, "ambiguous_intent")
        self.assertEqual([item.reading for item in check.alternatives], ["Total deaths", "COVID-19 Deaths"])
        self.assertTrue(ada.check_claim(frame, "Women accounted for about 37% of COVID-19 deaths.").verified)

    def test_a_share_of_all_deaths_is_of_the_total_column(self):
        frame = covid_deaths()
        self.assertTrue(ada.check_claim(frame, "COVID-19 accounted for about 5% of all deaths.").verified)
        check = ada.check_claim(frame, "COVID-19 accounted for about a quarter of all deaths.")
        self.assertEqual(check.reason, "failed_check")
        # 1,645 of 34,760.
        self.assertAlmostEqual(check.compared, 1645 / 34760 * 100)

    def test_nearly_one_in_ten_is_from_nine_tenths_of_it(self):
        number = read_claim("COVID-19 was involved in nearly one in ten deaths.", ["Total deaths"])[1]
        self.assertAlmostEqual(number.low, 9)
        self.assertAlmostEqual(number.high, 10)

    def test_the_tables_own_period_is_no_filter(self):
        frame = covid_deaths()
        claim = "More than 1,600 Americans died of COVID-19 between January 2020 and late June 2023."
        check = ada.check_claim(frame, claim)
        self.assertTrue(check.verified)
        self.assertEqual(check.compared, 1645)


class OpenScaleTests(unittest.TestCase):
    def test_an_open_bound_with_a_scale_word_says_nothing_of_the_columns_unit(self):
        drivers = pd.DataFrame({
            "State": ["Texas", "Ohio", "Iowa"] * 2, "Year": [2023] * 3 + [2024] * 3,
            "Drivers": [17_500_000, 8_100_000, 2_300_000, 17_800_000, 8_150_000, 2_310_000],
        })
        check = ada.check_claim(drivers, "Texas had more than 25 million drivers in 2024.")
        self.assertEqual(check.reason, "failed_check")
        self.assertEqual(check.compared, 17_800_000)
        self.assertTrue(ada.check_claim(drivers, "Texas had about 17.8 million drivers in 2024.").verified)

    def test_the_people_of_a_state_after_a_number_pick_its_rows(self):
        drivers = pd.DataFrame({
            "State": ["Texas", "Ohio", "Iowa"] * 2, "Year": [2023] * 3 + [2024] * 3,
            "Drivers": [17_500_000, 8_100_000, 2_300_000, 17_800_000, 8_150_000, 2_310_000],
        })
        check = ada.check_claim(drivers, "More than 20 million Texans were Drivers in 2024.")
        self.assertEqual(check.reason, "failed_check")
        self.assertEqual(check.compared, 17_800_000)
        self.assertIn('Read "Texans" as State = Texas.', check.assumptions)
        claim = "There were about 8.15 million Ohioan drivers in 2024."
        self.assertTrue(ada.check_claim(drivers, claim).verified)


class UnnamedYearClaimTests(unittest.TestCase):
    def deaths(self) -> pd.DataFrame:
        return pd.DataFrame({
            "state": ["California", "New York"] * 3, "year": [1982, 1982, 1983, 1983, 1984, 1984],
            "fatal": [4615, 2000, 4573, 2100, 5020, 2050],
        })

    def test_a_total_that_fits_one_year_is_not_proved_wrong_by_every_year(self):
        check = ada.check_claim(self.deaths(), "California recorded about 5,000 fatalities.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("names no year", check.message)
        self.assertIn("1984 (5,020)", check.message)

    def test_a_total_that_fits_no_year_still_fails(self):
        check = ada.check_claim(self.deaths(), "California recorded about 9,000 fatalities.")
        self.assertEqual(check.reason, "failed_check")

    def test_a_share_is_never_a_years_count(self):
        frame = pd.DataFrame({"state": ["California"] * 30 + ["Texas"] * 10 + ["California"] * 30,
                              "year": [1982] * 40 + [1983] * 30})
        check = ada.check_claim(frame, "30% of the rows were in California.")
        self.assertEqual(check.reason, "failed_check")

    def test_a_named_year_or_every_year_is_checked_as_said(self):
        claim = "California recorded about 5,000 fatalities in 1984."
        self.assertTrue(ada.check_claim(self.deaths(), claim).verified)
        check = ada.check_claim(self.deaths(), "California recorded about 5,000 fatalities in all.")
        self.assertEqual(check.reason, "failed_check")
        self.assertEqual(check.compared, 14_208)


class WideYearClaimTests(unittest.TestCase):
    def test_a_column_named_for_a_year_alone_is_checked(self):
        frame = pd.DataFrame({
            "Country Name": ["Japan", "Chad", "Peru"],
            "Indicator Name": ["Life expectancy at birth, total (years)"] * 3,
            "2020": [84.6, 52.8, 75.2], "2021": [84.4, 52.5, 73.6],
        })
        check = ada.check_claim(frame, "Life expectancy in Japan was 84.4 years in 2021.")
        self.assertTrue(check.verified, check.message)
        self.assertEqual(check.compared, 84.4)
