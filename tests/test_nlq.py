import unittest

import numpy as np
import pandas as pd

from demo_data import make_demo_data
from nlq import QueryPlan, answer_question, execute_plan, parse_question, suggested_questions
from pipeline import prepare_analysis
from schema import ColumnRoles, detect_roles


class NLQParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = prepare_analysis(make_demo_data(rows=900), row_limit=900)
        cls.dataframe = prepared.dataframe
        cls.roles = prepared.detected_roles

    def ask(self, question: str):
        result = answer_question(question, self.dataframe, self.roles)
        self.assertIsNotNone(result, f"engine could not answer: {question}")
        return result

    def test_total_aggregate(self):
        result = self.ask("What is the total revenue?")
        self.assertEqual(result.plan.intent, "aggregate")
        self.assertEqual(result.plan.aggregation, "sum")
        self.assertIn("Total Revenue", result.answer)
        self.assertIn("sum(Revenue)", result.calculation)

    def test_average_breakdown_by_dimension(self):
        result = self.ask("average profit by region")
        self.assertEqual(result.plan.intent, "breakdown")
        self.assertEqual(result.plan.aggregation, "mean")
        self.assertEqual(result.plan.dimension, "Region")
        self.assertEqual(result.chart, "bar")
        self.assertIn("Average Profit", result.table.columns)

    def test_top_n_rank(self):
        result = self.ask("top 3 products by revenue")
        self.assertEqual(result.plan.intent, "rank")
        self.assertEqual(result.plan.top_n, 3)
        self.assertEqual(len(result.table), 3)
        self.assertIn("Share %", result.table.columns)
        first, second = result.table.iloc[0], result.table.iloc[1]
        self.assertGreaterEqual(float(first["Total Revenue"]), float(second["Total Revenue"]))

    def test_bottom_rank_is_ascending(self):
        result = self.ask("bottom 2 regions by profit")
        self.assertTrue(result.plan.ascending)
        self.assertEqual(len(result.table), 2)

    def test_trend_with_grain(self):
        result = self.ask("monthly revenue trend")
        self.assertEqual(result.plan.intent, "trend")
        self.assertEqual(result.plan.grain, "M")
        self.assertEqual(result.chart, "line")
        self.assertGreater(len(result.table), 12)

    def test_growth_ranking_by_segment(self):
        result = self.ask("which product grew fastest?")
        self.assertEqual(result.plan.intent, "growth")
        self.assertEqual(result.plan.dimension, "Product")
        self.assertIn("Change %", result.table.columns)
        self.assertIn("fastest", result.answer)

    def test_count_distinct_entities(self):
        result = self.ask("how many orders are there?")
        self.assertEqual(result.plan.intent, "count")
        self.assertEqual(result.plan.count_column, "Order ID")
        self.assertIn("distinct", result.answer)

    def test_plain_row_count(self):
        result = self.ask("how many rows do we have?")
        self.assertIsNone(result.plan.count_column)
        self.assertIn("rows match", result.answer)

    def test_year_filter_reduces_scope(self):
        full = self.ask("total revenue")
        scoped = self.ask("total revenue in 2025")
        self.assertEqual(scoped.plan.year, 2025)
        self.assertIn("2025", scoped.answer)
        self.assertNotEqual(full.answer, scoped.answer)

    def test_month_and_year_filter(self):
        result = self.ask("total revenue in March 2025")
        self.assertEqual(result.plan.month, 3)
        self.assertEqual(result.plan.year, 2025)
        self.assertIn("March 2025", result.answer)

    def test_segment_value_filter(self):
        result = self.ask("total revenue in the West")
        self.assertEqual(len(result.plan.filters), 1)
        self.assertEqual(result.plan.filters[0].column, "Region")
        self.assertEqual(result.plan.filters[0].values, ("West",))

    def test_unreadable_question_returns_none(self):
        self.assertIsNone(answer_question("tell me a joke", self.dataframe, self.roles))

    def test_questions_about_unknown_columns_return_none(self):
        self.assertIsNone(
            answer_question("what is the craziest product we sell", self.dataframe, self.roles)
        )
        self.assertIsNone(answer_question("total sales by region", self.dataframe, self.roles))

    def test_known_bare_dimension_questions_remain_answerable(self):
        self.assertIsNotNone(answer_question("which product", self.dataframe, self.roles))
        self.assertIsNotNone(answer_question("revenue by region", self.dataframe, self.roles))

    def test_supported_grammar_and_contractions_remain_answerable(self):
        for question in (
            "what's total revenue",
            "average revenue per region",
            "revenue timeline",
            "revenue history",
            "revenue trajectory",
            "total revenue in the West",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(answer_question(question, self.dataframe, self.roles))

    def test_suggestions_are_all_answerable(self):
        for question in suggested_questions(self.dataframe, self.roles):
            self.assertIsNotNone(
                answer_question(question, self.dataframe, self.roles),
                f"suggested question failed: {question}",
            )

    def test_execute_plan_rejects_unknown_columns(self):
        plan = QueryPlan(intent="aggregate", measure="Nonexistent")
        with self.assertRaises(ValueError):
            execute_plan(plan, self.dataframe, self.roles)

    def test_parse_survives_dataset_without_dates(self):
        frame = self.dataframe.drop(columns=[self.roles.date]).copy()
        roles = detect_roles(frame)
        result = answer_question("top 2 products by revenue", frame, roles)
        self.assertIsNotNone(result)
        plan = parse_question("monthly revenue trend", frame, roles)
        self.assertNotEqual(plan.intent if plan else None, "trend")
    def test_asking_for_the_bottom_of_a_ranking_ranks_ascending(self):
        """"Least" and "fewest" ask for the bottom, not the top."""
        leader = self.ask("top product by revenue").answer

        for question in ("what is the least sold product", "which product sells the fewest"):
            with self.subTest(question=question):
                result = self.ask(question)
                self.assertEqual(result.plan.intent, "rank")
                self.assertTrue(result.plan.ascending)
                self.assertIn("lowest", result.answer)
                self.assertNotEqual(result.answer, leader)

    def test_every_ascending_word_is_also_a_superlative(self):
        """A word in only one list would rank from the wrong end."""
        from nlq import ASCENDING_WORDS, SUPERLATIVE_WORDS

        self.assertTrue(set(ASCENDING_WORDS).issubset(SUPERLATIVE_WORDS))



class TimelineDisclosureTests(unittest.TestCase):
    def test_a_chat_trend_answer_discloses_an_excluded_partial_period(self):
        rng = np.random.default_rng(4)
        dates = pd.date_range("2024-01-01", "2025-07-12", freq="D")
        frame = pd.DataFrame({"Date": dates, "Revenue": 100 + rng.normal(0, 4, len(dates))})

        answer = answer_question("revenue over time", frame, detect_roles(frame))

        self.assertIn("still in progress", answer.calculation)
        self.assertIn("Jul 2025", answer.calculation)

    def test_a_complete_timeline_adds_no_disclosure(self):
        rng = np.random.default_rng(4)
        dates = pd.date_range("2024-01-01", "2025-06-30", freq="D")
        frame = pd.DataFrame({"Date": dates, "Revenue": 100 + rng.normal(0, 4, len(dates))})

        answer = answer_question("revenue over time", frame, detect_roles(frame))

        self.assertNotIn("still in progress", answer.calculation)
        self.assertNotIn("counted as zero", answer.calculation)



class TimeScopeTests(unittest.TestCase):
    """A question about a year must not be answered with the all-time number."""

    def setUp(self):
        self.dated = pd.DataFrame(
            {
                "Order Date": pd.date_range("2025-01-01", periods=365, freq="D"),
                "Revenue": [100.0] * 365,
            }
        )
        self.dated_roles = detect_roles(self.dated)
        self.undated = pd.DataFrame(
            {"Period Label": ["FY2024 Q1", "FY2024 Q2"], "Revenue": [100.0, 200.0]}
        )
        self.undated_roles = detect_roles(self.undated)

    def test_a_year_scope_without_a_date_column_is_refused_not_ignored(self):
        plan = QueryPlan(intent="aggregate", aggregation="sum", measure="Revenue", year=2024)

        answer = execute_plan(plan, self.undated, self.undated_roles)

        self.assertIn("no date column", answer.answer)
        # The all-time total must not be presented as the 2024 total.
        self.assertNotIn("300", answer.answer)

    def test_may_is_named_in_the_answer_like_every_other_month(self):
        for month, label in ((3, "March"), (5, "May"), (7, "July"), (12, "December")):
            with self.subTest(month=label):
                plan = QueryPlan(
                    intent="aggregate", aggregation="sum", measure="Revenue", month=month
                )

                answer = execute_plan(plan, self.dated, self.dated_roles)

                self.assertIn(f"Order Date in {label}", answer.answer)
                self.assertIn(f"Order Date in {label}", answer.calculation)

    def test_a_month_and_year_together_name_both(self):
        plan = QueryPlan(
            intent="aggregate", aggregation="sum", measure="Revenue", month=5, year=2025
        )

        answer = execute_plan(plan, self.dated, self.dated_roles)

        self.assertIn("Order Date in May 2025", answer.answer)



class ShareOfTotalTests(unittest.TestCase):
    """A share is only a share when the parts add up to the whole."""

    def _roles(self, frame):
        return ColumnRoles(
            date=None, measure="Revenue", dimension="Region",
            identifier=None, numeric=("Revenue",), dimensions=("Region",),
        )

    def test_mixed_signs_produce_no_share_column(self):
        frame = pd.DataFrame(
            {"Region": ["A", "B", "C"], "Revenue": [100.0, -50.0, -30.0]}
        )
        plan = QueryPlan(
            intent="breakdown", aggregation="sum", measure="Revenue", dimension="Region"
        )

        answer = execute_plan(plan, frame, self._roles(frame))

        self.assertNotIn("Share %", answer.table.columns)
        self.assertNotIn("%", answer.answer)

    def test_an_all_negative_measure_keeps_its_share(self):
        frame = pd.DataFrame({"Region": ["A", "B"], "Revenue": [-75.0, -25.0]})
        plan = QueryPlan(
            intent="breakdown", aggregation="sum", measure="Revenue", dimension="Region"
        )

        answer = execute_plan(plan, frame, self._roles(frame))

        # Uniform signs, so the shares are real and sum to 100.
        self.assertIn("Share %", answer.table.columns)
        self.assertAlmostEqual(float(answer.table["Share %"].sum()), 100.0)
        self.assertEqual(
            dict(zip(answer.table["Region"], answer.table["Share %"], strict=True)),
            {"A": 75.0, "B": 25.0},
        )

    def test_unlabelled_rows_are_a_group_not_a_deletion(self):
        frame = pd.DataFrame(
            {"Region": ["South", None, "North", None], "Revenue": [200.0, 400.0, 150.0, 300.0]}
        )
        plan = QueryPlan(
            intent="breakdown", aggregation="sum", measure="Revenue", dimension="Region"
        )

        answer = execute_plan(plan, frame, self._roles(frame))

        # The denominator is the real total, not the total of the labelled rows.
        self.assertAlmostEqual(float(answer.table["Total Revenue"].sum()), 1050.0)
        self.assertIn("(not recorded)", list(answer.table["Region"]))

    def test_a_real_not_recorded_value_is_not_merged_with_blanks(self):
        frame = pd.DataFrame(
            {"Region": [None, "(not recorded)", None, "(not recorded)"], "Revenue": [300.0] * 4}
        )
        plan = QueryPlan(
            intent="breakdown", aggregation="sum", measure="Revenue", dimension="Region"
        )

        answer = execute_plan(plan, frame, self._roles(frame))

        # Two populations, two rows, each with its own 600.
        self.assertEqual(len(answer.table), 2)
        self.assertEqual(set(answer.table["Total Revenue"]), {600.0})

    def test_a_dimension_that_is_entirely_blank_does_not_crash(self):
        frame = pd.DataFrame({"Region": [None, None], "Revenue": [1.0, 2.0]})
        plan = QueryPlan(
            intent="rank", aggregation="sum", measure="Revenue", dimension="Region", top_n=5
        )

        answer = execute_plan(plan, frame, self._roles(frame))

        self.assertTrue(answer.answer)



class AnswerHonestyTests(unittest.TestCase):
    """Every sentence has to match the arithmetic that produced it."""

    def _roles(self, dated=True):
        return ColumnRoles(
            date="Month" if dated else None, measure="Revenue", dimension="Region",
            identifier=None, numeric=("Revenue",), dimensions=("Region",),
        )

    def test_a_row_count_is_described_as_a_row_count(self):
        frame = pd.DataFrame({"Region": ["N"] * 6 + ["S"] * 6, "Revenue": [1.0] * 12})
        plan = QueryPlan(
            intent="breakdown", aggregation="count", measure="Revenue", dimension="Region"
        )

        answer = execute_plan(plan, frame, self._roles(dated=False))

        self.assertIn("row count", answer.calculation)
        self.assertNotIn("count(Revenue)", answer.calculation)
        # A count of rows is not money.
        self.assertNotIn("$", answer.answer)

    def test_a_trend_from_zero_quotes_no_percentage(self):
        frame = pd.DataFrame(
            {"Month": pd.date_range("2024-01-01", periods=6, freq="MS"),
             "Revenue": [0.0, 100.0, 400.0, 800.0, 1200.0, 1600.0],
             "Region": ["N"] * 6}
        )
        plan = QueryPlan(intent="trend", aggregation="sum", measure="Revenue", grain="M")

        answer = execute_plan(plan, frame, self._roles())

        self.assertNotIn("+0.0%", answer.answer)
        self.assertIn("starting period of zero", answer.answer)

    def test_a_scope_with_no_values_is_not_a_total_of_zero(self):
        frame = pd.DataFrame({"Region": ["N", "S"], "Revenue": [float("nan")] * 2})
        plan = QueryPlan(intent="aggregate", aggregation="sum", measure="Revenue")

        answer = execute_plan(plan, frame, self._roles(dated=False))

        self.assertIn("no values", answer.answer)
        self.assertNotIn("$0.00", answer.answer)

    def test_a_segment_growing_from_zero_is_named_not_deleted(self):
        frame = pd.DataFrame(
            {
                "Month": list(pd.date_range("2024-03-01", periods=2, freq="MS")) * 2,
                "Region": ["Alpha", "Alpha", "Beta", "Beta"],
                "Revenue": [300.0, 400.0, 0.0, 5_000.0],
            }
        )
        plan = QueryPlan(
            intent="growth", aggregation="sum", measure="Revenue", dimension="Region", grain="M"
        )

        answer = execute_plan(plan, frame, self._roles())

        self.assertIn("Beta", answer.answer)
        self.assertIn("started from zero", answer.answer)

if __name__ == "__main__":
    unittest.main()