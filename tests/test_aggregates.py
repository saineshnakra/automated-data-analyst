import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import ada
from ada.aggregates import aggregate_rows
from ada.cli import main
from ada.decisions import Decision, UnknownDecision, checked_choices, parse_choices


def emissions() -> pd.DataFrame:
    """Three countries over two years, with the World and Africa rows beside them.

    By hand, 2022: the countries give 10 + 20 + 30 = 60; World is 63, holding 3
    that no country does; Africa is 30 (Kenya 10 + Nigeria 20). Every 2022 row
    together gives 153, and its average is 30.6; the countries' average is 20.
    """
    return pd.DataFrame({
        "country": ["Kenya", "Nigeria", "Chile", "Africa", "World"] * 2,
        "year": [2021] * 5 + [2022] * 5,
        "co2": [9.0, 18.0, 27.0, 27.0, 57.0, 10.0, 20.0, 30.0, 30.0, 63.0],
    })


def places(labels: list[str], column: str = "country") -> pd.DataFrame:
    return pd.DataFrame({column: labels, "value": [float(index + 1) for index in range(len(labels))]})


class AggregateRowsTests(unittest.TestCase):
    def test_world_and_regions_among_countries_are_found(self):
        found = aggregate_rows(emissions())
        self.assertEqual((found.column, found.labels, found.totals),
                         ("country", ("Africa", "World"), ("World",)))

    def test_regions_and_groups_are_found_as_tables_publish_them(self):
        groups = [
            "Asia, Central", "Micronesia (region)", "High-income countries", "Asia (excl. China and India)",
            "Europe & Central Asia", "European Union (27)", "Least developed countries: UN classification",
        ]
        countries = [f"Country {index}" for index in range(10)]
        found = aggregate_rows(places(countries + groups, "entity"))
        self.assertEqual(found.labels, tuple(groups))
        self.assertEqual(found.totals, ())

    def test_a_lone_group_code_among_county_names_is_a_county(self):
        counties = ["LEE", "IDA", "SAN DIEGO", "COOK", "KING", "LOS ANGELES"]
        self.assertIsNone(aggregate_rows(places(counties, "county")))
        countries = [f"Country {index}" for index in range(10)]
        found = aggregate_rows(places(countries + ["IDA", "World"], "entity"))
        self.assertEqual(found.labels, ("IDA", "World"))

    def test_countries_named_after_regions_are_not_aggregates(self):
        labels = ["South Africa", "United States", "Central African Republic",
                  "Federated States of Micronesia", "Dominican Republic", "World"]
        self.assertEqual(aggregate_rows(places(labels)).labels, ("World",))

    def test_a_total_row_outside_a_place_column_is_found_when_it_adds_up(self):
        sheet = pd.DataFrame({"product": ["Tea", "Coffee", "Cocoa", "Total"], "sales": [1.0, 2.0, 3.0, 6.0]})
        found = aggregate_rows(sheet)
        self.assertEqual((found.column, found.labels, found.totals), ("product", ("Total",), ("Total",)))

    def test_a_total_label_that_does_not_add_up_is_a_value_like_any_other(self):
        sheet = pd.DataFrame({"theme": ["City", "World", "Space", "Castle"], "sets": [10.0, 5.0, 7.0, 8.0]})
        self.assertIsNone(aggregate_rows(sheet))

    def test_a_column_of_regions_alone_has_none(self):
        continents = ["Africa", "Americas", "Asia", "Europe", "Oceania"]
        self.assertIsNone(aggregate_rows(places(continents, "continent")))

    def test_codes_are_not_read_as_group_names(self):
        self.assertIsNone(aggregate_rows(places(["USA", "IDA", "KEN", "CHL", "NGA"], "country_code")))

    def test_thin_data_leaves_nothing_out(self):
        self.assertIsNone(aggregate_rows(places(["Kenya", "World"])))
        # Leaving World and Africa out would leave one country, nothing to add up.
        self.assertIsNone(aggregate_rows(places(["Kenya", "Africa", "World"])))
        self.assertIsNone(aggregate_rows(pd.DataFrame({"country": pd.Series([], dtype=object)})))


class AggregateAnswerTests(unittest.TestCase):
    def setUp(self):
        self.analysis = ada.analyze(emissions())

    def test_a_total_is_read_from_the_world_row(self):
        answer = self.analysis.ask("total co2 in 2022")
        self.assertAlmostEqual(answer.value, 63.0)
        self.assertIn("Read the total from the World row; the other country values add up to 60",
                      answer.answer)
        self.assertEqual([(item.key, item.chosen, item.by_caller) for item in answer.decisions],
                         [("aggregate_rows", "leave_out", False), ("total_from", "total_row", False)])

    def test_the_sum_of_the_parts_when_the_caller_chooses_it(self):
        answer = self.analysis.ask("total co2 in 2022", decisions={"total_from": "sum_of_parts"})
        self.assertAlmostEqual(answer.value, 60.0)
        self.assertTrue(answer.decisions[1].by_caller)
        self.assertIn(("rows that add up other rows (World and Africa)", 2), answer.scope.left_out)

    def test_keeping_them_counts_every_row(self):
        answer = self.analysis.ask("total co2 in 2022", decisions={"aggregate_rows": "keep"})
        self.assertAlmostEqual(answer.value, 153.0)
        self.assertEqual([(item.key, item.chosen) for item in answer.decisions], [("aggregate_rows", "keep")])

    def test_an_average_leaves_them_out_and_says_so(self):
        answer = self.analysis.ask("average co2 in 2022")
        self.assertAlmostEqual(answer.value, 20.0)
        self.assertIn("Left out 2 rows that add up other rows (World and Africa).", answer.answer)
        record = answer.to_dict()
        self.assertEqual(record["decisions"][0]["decided_by"], "default")
        self.assertEqual(record["reading"]["left_out_values"],
                         [{"column": "country", "values": ["World", "Africa"]}])

    def test_a_ranking_leaves_them_out(self):
        answer = self.analysis.ask("top 2 country by co2 in 2022")
        self.assertEqual({group for group, _ in answer.values}, {"Kenya", "Nigeria", "Chile"})

    def test_a_question_that_names_an_aggregate_is_about_it(self):
        answer = self.analysis.ask("co2 of World in 2022")
        self.assertAlmostEqual(answer.value, 63.0)
        self.assertEqual(answer.decisions, ())

    def test_an_unknown_choice_is_refused(self):
        with self.assertRaises(UnknownDecision):
            self.analysis.ask("total co2", decisions={"aggregate_row": "keep"})
        with self.assertRaises(UnknownDecision):
            self.analysis.ask("total co2", decisions={"aggregate_rows": "drop"})

    def test_a_table_without_aggregates_takes_no_decision(self):
        analysis = ada.analyze(pd.DataFrame({"site": ["A", "B", "C", "D"], "co2": [1.0, 2.0, 3.0, 4.0]}))
        plain = analysis.ask("total co2")
        chosen = analysis.ask("total co2", decisions={"aggregate_rows": "keep"})
        self.assertEqual((plain.decisions, plain.value), ((), 10.0))
        self.assertEqual((chosen.decisions, chosen.value), ((), 10.0))


class YearColumnTests(unittest.TestCase):
    def setUp(self):
        # Years before 1800 are not turned into dates, so the year stays a column of numbers.
        self.analysis = ada.analyze(pd.DataFrame({
            "site": ["A", "B", "C"] * 3, "year": [1750] * 3 + [1751] * 3 + [1752] * 3,
            "co2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }))

    def test_a_year_scopes_the_answer_without_a_date_column(self):
        answer = self.analysis.ask("total co2 in 1751")
        self.assertIsNone(self.analysis.roles.date)
        self.assertAlmostEqual(answer.value, 15.0)
        self.assertEqual(answer.scope.filters, ("year = 1751",))

    def test_a_month_still_needs_a_date(self):
        answer = self.analysis.ask("total co2 in March 1751")
        self.assertIsNone(answer.value)
        self.assertIn("no date column", answer.answer)


class AggregateClaimTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "emissions.csv"
        emissions().to_csv(self.path, index=False)

    def test_a_claim_that_counts_the_world_row_is_asked_about(self):
        check = ada.check_claim(self.path, "The average co2 in 2022 was 30.6.")
        self.assertEqual((check.status, check.reason), ("unverified", "ambiguous_intent"))
        self.assertEqual([(item.reading, item.agrees) for item in check.alternatives],
                         [("aggregate_rows=leave_out", False), ("aggregate_rows=keep", True)])
        self.assertIn("aggregate_rows", check.clarification)
        chosen = ada.check_claim(self.path, "The average co2 in 2022 was 30.6.",
                                 decisions={"aggregate_rows": "keep"})
        self.assertTrue(chosen.verified)

    def test_a_claim_about_the_countries_is_verified_with_the_decision_listed(self):
        check = ada.check_claim(self.path, "The average co2 in 2022 was 20.")
        self.assertTrue(check.verified)
        self.assertEqual(check.to_dict()["decisions"][0]["key"], "aggregate_rows")
        self.assertIn("decides: Leave out the 2 rows", str(check))

    def test_a_total_is_checked_against_the_world_row(self):
        self.assertTrue(ada.check_claim(self.path, "Total co2 in 2022 was 63.").verified)
        added = ada.check_claim(self.path, "Total co2 in 2022 was 60.")
        self.assertEqual(added.reason, "ambiguous_intent")
        self.assertEqual(added.alternatives[1].reading, "total_from=sum_of_parts")
        self.assertEqual(ada.check_claim(self.path, "Total co2 in 2022 was 70.").reason, "failed_check")

    def test_the_command_line_takes_a_choice(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["ask", str(self.path), "total co2 in 2022", "--decide", "total_from=sum_of_parts",
                         "--json"])
        self.assertEqual(code, 0)
        self.assertAlmostEqual(json.loads(out.getvalue())["result"]["value"], 60.0)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(main(["ask", str(self.path), "total co2", "--decide", "keep"]), 2)
        self.assertIn("key=option", err.getvalue())


class DecisionTests(unittest.TestCase):
    def test_choices_are_read_as_key_and_option(self):
        self.assertEqual(parse_choices(["aggregate_rows=keep", " total_from = sum_of_parts "]),
                         {"aggregate_rows": "keep", "total_from": "sum_of_parts"})
        self.assertEqual(parse_choices([]), {})

    def test_a_wrong_choice_says_what_is_allowed(self):
        with self.assertRaisesRegex(UnknownDecision, "aggregate_rows, total_from"):
            checked_choices({"rows": "keep"})
        with self.assertRaisesRegex(UnknownDecision, "leave_out, keep"):
            checked_choices({"aggregate_rows": "all"})
        with self.assertRaisesRegex(UnknownDecision, "key=option"):
            parse_choices(["aggregate_rows"])

    def test_a_decision_says_who_chose(self):
        options = ("leave_out", "keep")
        decision = Decision("aggregate_rows", "Leave out World?", options, "leave_out", "keep", "")
        self.assertTrue(decision.by_caller)
        self.assertEqual(decision.to_dict()["decided_by"], "caller")
        self.assertEqual(decision.describe(), "Leave out World? keep (chosen by the caller; also: leave_out)")


if __name__ == "__main__":
    unittest.main()
