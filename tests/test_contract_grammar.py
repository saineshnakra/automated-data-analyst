"""The chat grammar accounts for every part of a question, or refuses it.

A question that names two metrics, an arithmetic expression, a single day, a
quarter, a year the file does not cover, or a word that names nothing in the
file must never be answered as the nearest easier question. These are the
reviewer's fixtures; every expected number was checked by hand.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from ai_insights import AIQueryPlan, plan_query_with_ai
from nlq import (
    QueryPlan,
    ValueFilter,
    answer_question,
    parse_question,
    plan_accounts_for_numbers,
    question_is_representable,
    unsupported_phrasing,
)
from schema import detect_roles


def grammar_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-15", "2024-04-30"]),
            "Revenue": [100, 200, 300, 400],
            "Profit": [10, 20, 30, 40],
            "Country": ["UK", "US", "UK", "US"],
        }
    )


class CountingClient:
    """A planner client that records whether it was asked anything."""

    def __init__(self, parsed):
        self.calls = 0
        self.parsed = parsed
        self.responses = self

    def parse(self, **kwargs: object) -> object:
        self.calls += 1
        return SimpleNamespace(output_parsed=self.parsed)


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.frame = grammar_fixture()
        self.roles = detect_roles(self.frame)

    def ask(self, question: str):
        return answer_question(question, self.frame, self.roles)

    def test_a_single_day_is_a_scope_of_its_own(self):
        # D-04: "Revenue 2024-01-31" answered with all of 2024.
        for question in ("Revenue 2024-01-31", "Revenue on 2024/01/31"):
            with self.subTest(question=question):
                result = self.ask(question)
                self.assertEqual(result.plan.day, 31)
                self.assertEqual(result.plan.month, 1)
                self.assertIn("$200.00", result.answer)
                self.assertIn("calculated from 1 rows", result.answer)

    def test_a_month_and_day_resolve_the_day(self):
        for question in ("Revenue in January 15", "Revenue on the 15th of January", "Revenue on 15 January"):
            with self.subTest(question=question):
                result = self.ask(question)
                self.assertIn("$100.00", result.answer)
                self.assertIn("January 15", result.answer)

    def test_a_quarter_is_a_scope(self):
        for question in (
            "Revenue in q1",
            "Revenue in Q1 2024",
            "Revenue in the first quarter",
            "revenue in quarter 1",
        ):
            with self.subTest(question=question):
                result = self.ask(question)
                self.assertEqual(result.plan.intent, "aggregate")
                self.assertEqual(result.plan.quarter, 1)
                self.assertIn("$600.00", result.answer)
                self.assertIn("Q1", result.answer)

    def test_a_year_the_file_does_not_cover_is_not_dropped(self):
        result = self.ask("Revenue in 2100")
        self.assertEqual(result.plan.year, 2100)
        self.assertIn("No rows match", result.answer)
        self.assertNotIn("$", result.answer)

    def test_a_year_alone_still_scopes(self):
        self.assertIn("$1.0K", self.ask("revenue in 2024").answer)
        self.assertIn("$300.00", self.ask("revenue in january").answer)


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.frame = grammar_fixture()
        self.roles = detect_roles(self.frame)

    def test_two_metrics_or_arithmetic_between_them_are_refused(self):
        for question in (
            "total Revenue and Profit",
            "total Revenue / Profit",
            "total Revenue - Profit",
            "Revenue plus Profit",
            "Revenue divided by Profit",
            "Revenue vs Profit",
            "Revenue compared to Profit",
        ):
            with self.subTest(question=question):
                self.assertIsNone(parse_question(question, self.frame, self.roles))
                self.assertFalse(question_is_representable(question, self.frame, self.roles))

    def test_top_n_without_a_group_is_refused(self):
        for question in ("top 2 Revenue", "bottom 2 Revenue"):
            with self.subTest(question=question):
                self.assertIsNone(parse_question(question, self.frame, self.roles))

    def test_a_word_that_names_nothing_in_the_file_is_refused(self):
        # "per customer" and "france" name nothing here; the old parser
        # answered the all-time total for both.
        for question in ("revenue per customer", "revenue for france", "revenue in the north"):
            with self.subTest(question=question):
                self.assertIsNone(parse_question(question, self.frame, self.roles))

    def test_a_number_nobody_uses_is_refused(self):
        for question in ("revenue 200", "top revenue 3", "revenue in h1"):
            with self.subTest(question=question):
                self.assertIsNone(parse_question(question, self.frame, self.roles))

    def test_a_count_over_time_is_a_count_trend(self):
        result = answer_question("count rows over time", self.frame, self.roles)
        self.assertEqual(result.plan.intent, "trend")
        self.assertEqual(result.plan.aggregation, "count")
        self.assertIn("Records per month", result.answer)
        self.assertIn("row count", result.calculation)
        self.assertEqual(result.table["Value"].tolist(), [2, 1, 0, 1])

    def test_growth_and_trend_without_a_date_explain_instead_of_totalling(self):
        dateless = self.frame.drop(columns=["Date"])
        roles = detect_roles(dateless)
        for question in ("Revenue growth", "Revenue over time", "monthly revenue"):
            with self.subTest(question=question):
                result = answer_question(question, dateless, roles)
                self.assertIsNotNone(result)
                self.assertIn("no date column", result.answer)
                self.assertNotIn("$", result.answer)

    def test_average_growth_is_refused_rather_than_measured_on_sums(self):
        # Monthly row revenues [100], [100], [100, 100]: the sums double,
        # the average stays 100 throughout.
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-05", "2024-02-05", "2024-03-05", "2024-03-20"]),
                "Revenue": [100, 100, 100, 100],
            }
        )
        roles = detect_roles(frame)
        self.assertIsNone(parse_question("average Revenue growth", frame, roles))
        self.assertIn("up 100.0%", answer_question("Revenue growth", frame, roles).answer)

    def test_an_empty_previous_period_is_named_not_treated_as_zero(self):
        result = answer_question("Revenue growth", self.frame, self.roles)
        self.assertIn("Mar 2024 has no rows", result.answer)
        self.assertNotIn("%", result.answer)


class ValueFilterTests(unittest.TestCase):
    def test_a_short_value_is_matched_when_a_preposition_grounds_it(self):
        frame = grammar_fixture()
        roles = detect_roles(frame)
        result = answer_question("total Revenue for UK", frame, roles)
        self.assertEqual(result.plan.filters, (ValueFilter(column="Country", values=("UK",)),))
        self.assertIn("$400.00", result.answer)
        self.assertIn("calculated from 2 rows", result.answer)

    def test_the_longest_value_wins_over_a_value_inside_it(self):
        frame = pd.DataFrame(
            {
                "City": ["New York", "York", "York", "New York"],
                "Revenue": [100, 200, 200, 100],
            }
        )
        roles = detect_roles(frame)
        new_york = answer_question("total Revenue for New York", frame, roles)
        york = answer_question("total Revenue for York", frame, roles)
        self.assertEqual(new_york.plan.filters, (ValueFilter(column="City", values=("New York",)),))
        self.assertIn("$200.00", new_york.answer)
        self.assertEqual(york.plan.filters, (ValueFilter(column="City", values=("York",)),))
        self.assertIn("$400.00", york.answer)

    def test_a_grouped_answer_keeps_a_filter_on_its_own_dimension(self):
        frame = pd.DataFrame({"Region": ["West", "East", "West", "East"], "Revenue": [100, 300, 100, 300]})
        roles = detect_roles(frame)
        result = answer_question("Revenue by Region for West", frame, roles)
        self.assertEqual(result.plan.dimension, "Region")
        self.assertEqual(result.plan.filters, (ValueFilter(column="Region", values=("West",)),))
        self.assertEqual(result.table["Region"].tolist(), ["West"])
        self.assertIn("West is the leading Region", result.answer)
        self.assertNotIn("East", result.answer)

    def test_a_value_that_lives_in_two_columns_is_not_guessed(self):
        frame = pd.DataFrame(
            {
                "Region": ["North", "South", "North", "South"],
                "Territory": ["South", "South", "North", "North"],
                "Revenue": [1, 2, 3, 4],
            }
        )
        roles = detect_roles(frame)
        self.assertIsNone(parse_question("revenue for North", frame, roles))


class DistinctCountTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "Customer ID": ["001", "001", "002", "003", "003"],
                "Region": ["West", "West", "West", "East", "East"],
                "Revenue": [1, 1, 3, 4, 5],
            }
        )
        self.roles = detect_roles(self.frame)

    def test_entities_are_counted_distinctly_with_and_without_a_group(self):
        # D-07: three customers overall, two in West and one in East -- not
        # the three and two rows the grouped path used to count.
        alone = answer_question("how many customers", self.frame, self.roles)
        self.assertIn("3 distinct Customer ID", alone.answer)

        for question in (
            "how many customers by region",
            "customers by region",
            "number of customers per region",
        ):
            with self.subTest(question=question):
                grouped = answer_question(question, self.frame, self.roles)
                self.assertEqual(grouped.plan.count_column, "Customer ID")
                counts = dict(
                    zip(grouped.table["Region"], grouped.table["Distinct Customer ID"], strict=True)
                )
                self.assertEqual(counts, {"West": 2, "East": 1})
                self.assertIn("count distinct Customer ID", grouped.calculation)

    def test_rows_stay_rows(self):
        rows = answer_question("how many rows by region", self.frame, self.roles)
        self.assertIsNone(rows.plan.count_column)
        self.assertEqual(
            dict(zip(rows.table["Region"], rows.table["Rows"], strict=True)), {"West": 3, "East": 2}
        )

    def test_the_ai_plan_carries_the_counted_column_through_approval_and_execution(self):
        from ai_insights import _to_query_plan, describe_query_plan
        from nlq import execute_plan

        parsed = AIQueryPlan(answerable=True, intent="count", aggregation="count", count_column="Customer ID")
        plan = _to_query_plan(parsed, self.frame, self.roles)
        self.assertEqual(plan.count_column, "Customer ID")
        self.assertIn("Customer ID", describe_query_plan(plan))
        self.assertIn("3 distinct", execute_plan(plan, self.frame, self.roles).answer)

    def test_top_n_of_an_entity_ranks_one_row_per_entity(self):
        result = answer_question("top 2 customers by revenue", self.frame, self.roles)
        self.assertEqual(result.plan.dimension, "Customer ID")
        self.assertEqual(result.table["Customer ID"].tolist(), ["003", "002"])


class PlannerGateTests(unittest.TestCase):
    """The optional planner is only asked what a plan can hold."""

    def setUp(self):
        self.frame = grammar_fixture()
        self.roles = detect_roles(self.frame)

    def test_unrepresentable_questions_never_reach_the_model(self):
        parsed = AIQueryPlan(answerable=True, intent="aggregate", aggregation="sum", measure="Revenue")
        for question in (
            "total Revenue and Profit",
            "Revenue / Profit",
            "revenue in January and February",
            "revenue > 200",
        ):
            with self.subTest(question=question):
                client = CountingClient(parsed)
                plan = plan_query_with_ai(
                    question, self.frame, self.roles, api_key="k", safety_identifier="s", client=client
                )
                self.assertIsNone(plan)
                self.assertEqual(client.calls, 0)

    def test_a_plan_that_drops_a_number_from_the_question_is_refused(self):
        # The model read "Revenue 2024-01-31" as January 2024.
        parsed = AIQueryPlan(
            answerable=True, intent="aggregate", aggregation="sum", measure="Revenue", year=2024, month=1
        )
        client = CountingClient(parsed)
        plan = plan_query_with_ai(
            "Revenue 2024-01-31", self.frame, self.roles, api_key="k", safety_identifier="s", client=client
        )
        self.assertEqual(client.calls, 1)
        self.assertIsNone(plan)

        whole = AIQueryPlan(
            answerable=True,
            intent="aggregate",
            aggregation="sum",
            measure="Revenue",
            year=2024,
            month=1,
            day=31,
        )
        plan = plan_query_with_ai(
            "Revenue 2024-01-31",
            self.frame,
            self.roles,
            api_key="k",
            safety_identifier="s",
            client=CountingClient(whole),
        )
        self.assertEqual((plan.year, plan.month, plan.day), (2024, 1, 31))

    def test_number_accounting(self):
        plan = QueryPlan(intent="aggregate", measure="Revenue", year=2024, month=1)
        self.assertTrue(plan_accounts_for_numbers("Revenue in January 2024", plan))
        self.assertFalse(plan_accounts_for_numbers("Revenue in January 15 2024", plan))
        self.assertFalse(plan_accounts_for_numbers("Revenue in q1", plan))
        self.assertTrue(plan_accounts_for_numbers("top 3 countries", QueryPlan(intent="rank", top_n=3)))

    def test_the_planner_sees_the_same_denylist_as_the_rules(self):
        for question in ("Revenue - Profit", "Revenue × Profit", "revenue vs profit"):
            with self.subTest(question=question):
                self.assertTrue(unsupported_phrasing(question))


if __name__ == "__main__":
    unittest.main()
