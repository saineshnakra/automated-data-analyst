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

    def test_a_folder_asks_for_the_file(self):
        check = ada.check_claim(self.folder, "The average body mass is 4457 g.")
        self.assertEqual(check.reason, "missing_context")
        self.assertIn("birds.csv", check.clarification)

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
            "alternatives", "clarification",
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
