import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import ada
from ada.claims import (
    NUMBER_PATTERN,
    REASONS,
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
        # "deaths" is the first word of death_height_metres, a height; summing heights is a guess.
        check = ada.check_claim(self.frame(), "Members recorded 3 member deaths in total.")
        self.assertEqual(check.answer.plan.measure, "death_height_metres")
        self.assertEqual((check.reason, check.agreement), ("missing_context", "mismatch"))
        self.assertEqual(check.clarification,
                         "Is the number about death_height_metres, or about another column?")

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
        check = ada.check_claim(losses(), "Brazil lost 2,500 hectares of forest to pasture in 2004.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("may pick other rows", check.message)

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
        self.assertIn("The claim's 10 models are the 10 rows in scope.", check.assumptions)
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
