import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import ada
from ada import nlq, ui
from ada.business_insights import describe_correlation, measure_correlation
from ada.demo_data import make_demo_data
from ada.file_io import list_sample_datasets as samples
from ada.formatting import format_number
from ada.nlq import (
    Condition,
    QueryPlan,
    WholeTable,
    answer_question,
    execute_plan,
    parse_question,
    rephrase_hint,
    suggested_questions,
    unsupported_phrasing,
)
from ada.pipeline import prepare_analysis
from ada.quality import QualityIssue
from ada.schema import ColumnRoles, detect_roles

MONTHS = pd.date_range("2024-01-01", periods=12, freq="MS")


def frame_with(measure: str) -> tuple[pd.DataFrame, ColumnRoles]:
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-05", "2025-02-05"] * 2),
        "Team": ["A", "A", "B", "B"],
        measure: [100.0, 200.0, 300.0, 400.0],
    })
    roles = ColumnRoles(date="Date", measure=measure, dimension="Team", identifier=None,
                        numeric=(measure,), dimensions=("Team",))
    return frame, roles


def rate_frame():
    frame = pd.DataFrame({
        "Date": pd.to_datetime(["2025-01-05", "2025-02-05"] * 2),
        "Region": ["East", "East", "West", "West"],
        "Churn Rate": [2.0, 3.0, 4.0, 5.0],
    })
    roles = ColumnRoles(date="Date", measure="Churn Rate", dimension="Region", identifier=None,
                        numeric=("Churn Rate",), dimensions=("Region",))
    return frame, roles


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
        self.assertIn("rose most", result.answer)

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

    def test_ordinary_phrasing_with_function_words_remains_answerable(self):
        for question in (
            "which region grew the most",
            "how much revenue did we make",
            "give me revenue by channel",
            "whats the total revenue",
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
        leader = self.ask("top product by revenue").answer

        for question in ("what is the least sold product", "which product sells the fewest"):
            with self.subTest(question=question):
                result = self.ask(question)
                self.assertEqual(result.plan.intent, "rank")
                self.assertTrue(result.plan.ascending)
                self.assertIn("lowest", result.answer)
                self.assertNotEqual(result.answer, leader)

    def test_every_ascending_word_is_also_a_superlative(self):
        from ada.nlq import ASCENDING_WORDS, SUPERLATIVE_WORDS

        self.assertTrue(set(ASCENDING_WORDS).issubset(SUPERLATIVE_WORDS))


class SuperlativeRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = prepare_analysis(make_demo_data(rows=900), row_limit=900)
        cls.frame, cls.roles = prepared.dataframe, prepared.detected_roles

    def test_most_ranks_the_named_dimension(self):
        plan = parse_question("which region made the most revenue", self.frame, self.roles)
        self.assertIsNotNone(plan, "the question was refused outright")
        self.assertEqual(plan.intent, "rank")
        self.assertEqual(plan.dimension, "Region")
        self.assertFalse(plan.ascending)

    def test_most_and_least_are_symmetric(self):
        most = parse_question("which region made the most revenue", self.frame, self.roles)
        least = parse_question("which region made the least revenue", self.frame, self.roles)
        self.assertEqual(most.intent, least.intent)
        self.assertEqual(most.dimension, least.dimension)
        self.assertNotEqual(most.ascending, least.ascending)

    def test_has_is_an_ordinary_function_word(self):
        answer = answer_question("which region has the highest revenue", self.frame, self.roles)
        self.assertIsNotNone(answer)
        self.assertIn("Region", answer.calculation)

    def test_total_revenue_has_no_dimension(self):
        plan = parse_question("total revenue", self.frame, self.roles)
        self.assertEqual(plan.intent, "aggregate")
        self.assertIsNone(plan.dimension)


class ColumnNameIntentTests(unittest.TestCase):
    def test_a_count_column_is_summed_not_counted(self):
        frame, roles = frame_with("Cell Count")
        answer = answer_question("total cell count", frame, roles)
        self.assertIsNotNone(answer)
        self.assertIn("sum(Cell Count)", answer.calculation, answer.calculation)
        self.assertIn("1,000", answer.answer, answer.answer)

    def test_a_count_column_breaks_down_by_its_own_values(self):
        frame, roles = frame_with("Cell Count")
        answer = answer_question("cell count by team", frame, roles)
        self.assertIn("Cell Count", answer.calculation, answer.calculation)
        self.assertNotIn("row count", answer.calculation, answer.calculation)

    def test_an_explicit_average_beats_the_word_total_in_the_column_name(self):
        frame, roles = frame_with("Total Sales")
        answer = answer_question("average total sales", frame, roles)
        self.assertIn("mean(Total Sales)", answer.calculation, answer.calculation)

    def test_a_growth_column_still_totals_when_asked_for_a_total(self):
        frame, roles = frame_with("Revenue Growth")
        plan = parse_question("total revenue growth", frame, roles)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.intent, "aggregate", f"got {plan.intent}")

    def test_a_change_column_still_totals(self):
        frame, roles = frame_with("Change Score")
        plan = parse_question("total change score", frame, roles)
        self.assertEqual(plan.intent, "aggregate", f"got {plan.intent}")

    def test_the_reader_can_still_ask_for_each_intent(self):
        """Words in the measure's name are ignored for intent; the rest of the question still sets it."""
        frame, roles = frame_with("Cell Count")
        self.assertEqual(parse_question("how many teams", frame, roles).intent, "count")
        self.assertEqual(parse_question("cell count per month", frame, roles).intent, "trend")
        self.assertEqual(
            parse_question("how did cell count change", frame, roles).intent, "growth"
        )
        self.assertEqual(parse_question("cell count by team", frame, roles).intent, "breakdown")

    def test_an_ordinary_measure_is_unaffected(self):
        frame, roles = frame_with("Revenue")
        self.assertIn("sum(Revenue)", answer_question("total revenue", frame, roles).calculation)
        self.assertIn("mean(Revenue)", answer_question("average revenue", frame, roles).calculation)


class RateMeasureTests(unittest.TestCase):
    def test_a_rate_growth_reports_points(self):
        """The monthly mean moves from 3.0 to 4.0: one point, or 33.3% relative."""
        frame, roles = rate_frame()
        answer = answer_question("how did churn rate change", frame, roles)
        self.assertIn("percentage points", answer.answer, answer.answer)
        self.assertIn("1.0", answer.answer, answer.answer)

    def test_a_rate_trend_reports_points(self):
        frame, roles = rate_frame()
        answer = answer_question("churn rate per month", frame, roles)
        self.assertIn("percentage points", answer.answer, answer.answer)

    def test_growth_and_trend_answers_both_use_points(self):
        frame, roles = rate_frame()
        for question in ("how did churn rate change", "churn rate per month"):
            with self.subTest(question=question):
                self.assertIn("percentage points", answer_question(question, frame, roles).answer)

    def test_an_additive_measure_still_reports_a_percentage(self):
        frame, roles = rate_frame()
        frame = frame.rename(columns={"Churn Rate": "Revenue"})
        roles = ColumnRoles(date="Date", measure="Revenue", dimension="Region", identifier=None,
                            numeric=("Revenue",), dimensions=("Region",))
        answer = answer_question("how did revenue change", frame, roles)
        self.assertIn("%", answer.answer)
        self.assertNotIn("percentage points", answer.answer)

    def test_chat_averages_a_rate_unless_told_to_total_it(self):
        frame = pd.DataFrame(
            {"Date": MONTHS[:4], "Conversion Rate": [0.1, 0.2, 0.3, 0.4], "Channel": ["a", "b"] * 2}
        )
        roles = detect_roles(frame)

        averaged = answer_question("conversion rate", frame, roles)
        totalled = answer_question("total conversion rate", frame, roles)

        self.assertEqual(averaged.plan.aggregation, "mean")
        self.assertIn("25.0%", averaged.answer)
        self.assertEqual(totalled.plan.aggregation, "sum")


class RefusedPhrasingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {"Month": MONTHS[:4], "Region": ["West", "East", "West", "East"],
             "Revenue": [100.0, 200.0, 300.0, 400.0]}
        )
        self.roles = detect_roles(self.frame)

    def test_unrepresentable_phrasings_are_refused_before_parsing(self):
        for question in (
            "Revenue in January and February", "Revenue this year", "Revenue for West or Partner",
            "total revenue last quarter", "revenue in 2024 and 2025", "revenue != 200",
        ):
            with self.subTest(question=question):
                self.assertTrue(unsupported_phrasing(question))
                self.assertIsNone(answer_question(question, self.frame, self.roles))

    def test_a_threshold_on_a_named_column_counts_the_rows_it_keeps(self):
        for question, kept in (
            ("Revenue > 200", 2), ("revenue at least 300", 2), ("revenue over 1000", 0),
            ("revenue more than 50", 4), ("how many rows have revenue below 250", 2),
        ):
            with self.subTest(question=question):
                self.assertFalse(unsupported_phrasing(question))
                answer = answer_question(question, self.frame, self.roles)
                self.assertEqual(answer.plan.intent, "count")
                self.assertTrue(answer.answer.startswith(f"{kept} of the 4 rows"), answer.answer)

    def test_a_threshold_that_names_no_column_is_refused(self):
        self.assertIsNone(answer_question("total revenue above 200 units", self.frame, self.roles))
        self.assertIsNone(answer_question("how many are over 50", self.frame, self.roles))

    def test_plans_the_executor_would_misanswer_are_refused_by_the_parser(self):
        # These pass the phrasing check; the parser rejects the plan.
        for question in ("average Revenue monthly", "top 0 Region by Revenue"):
            with self.subTest(question=question):
                self.assertFalse(unsupported_phrasing(question))
                self.assertIsNone(answer_question(question, self.frame, self.roles))

    def test_supported_phrasings_are_not_caught_by_the_refusal(self):
        for question in ("total revenue", "revenue over time", "top 2 region by revenue",
                         "revenue by region", "revenue in march 2024"):
            with self.subTest(question=question):
                self.assertFalse(unsupported_phrasing(question))
                self.assertIsNotNone(answer_question(question, self.frame, self.roles))

    def test_naming_the_column_does_not_steal_its_value_from_the_filter(self):
        answer = answer_question("total Revenue for Region West", self.frame, self.roles)

        self.assertIsNotNone(answer)
        self.assertIn("$400.00", answer.answer)

    def test_an_explicit_minimum_is_computed_not_summed(self):
        answer = answer_question("minimum Revenue by Region", self.frame, self.roles)

        self.assertIsNotNone(answer)
        self.assertEqual(answer.plan.aggregation, "min")

    def test_top_n_applies_to_a_growth_ranking(self):
        answer = answer_question("top 1 Region growth", self.frame, self.roles)

        self.assertIsNotNone(answer)
        self.assertEqual(len(answer.table), 1)


class RelativeMonthTests(unittest.TestCase):
    def setUp(self):
        frame = pd.DataFrame({
            "Order Date": pd.to_datetime(["2023-12-05", "2024-12-05", "2025-12-05"]),
            "Region": ["East"] * 3,
            "Revenue": [100.0, 100.0, 100.0],
        })
        prepared = prepare_analysis(frame, row_limit=900)
        self.frame, self.roles = prepared.dataframe, prepared.detected_roles

    def test_last_december_is_refused_rather_than_summed_across_years(self):
        """The file has three Decembers, one in each year."""
        self.assertIsNone(answer_question("revenue last december", self.frame, self.roles))

    def test_next_december_is_refused(self):
        self.assertIsNone(answer_question("revenue next december", self.frame, self.roles))

    def test_a_plain_month_still_answers(self):
        answer = answer_question("revenue in december", self.frame, self.roles)
        self.assertIsNotNone(answer)


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
        # 300 is the all-time total.
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

    def test_a_year_inside_a_web_address_is_still_the_year(self):
        dates = pd.to_datetime(["2019-03-17", "2019-07-14", "2020-07-05", "2020-07-12", "2020-08-02",
                                "2021-03-28"])
        frame = pd.DataFrame({
            "date": dates,
            "name": ["Australian", "British", "Austrian", "Styrian", "British", "Bahrain"],
            "round": [1, 10, 1, 2, 4, 1],
            "url": [f"https://en.wikipedia.org/wiki/{day.year}_{name}_Grand_Prix"
                    for day, name in zip(dates, ["Australian", "British", "Austrian", "Styrian", "British",
                                                 "Bahrain"], strict=True)],
        })

        answer = answer_question("how many races in 2020", frame, detect_roles(frame))

        self.assertEqual(answer.plan.year, 2020)
        self.assertEqual(answer.value, 3)


class ShareOfTotalTests(unittest.TestCase):
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

        # All values have the same sign, so shares are defined and sum to 100.
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

        # 1050 includes the 700 from rows with no region.
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

        # Blank regions and the literal "(not recorded)" value stay separate, 600 each.
        self.assertEqual(len(answer.table), 2)
        self.assertEqual(set(answer.table["Total Revenue"]), {600.0})

    def test_a_dimension_that_is_entirely_blank_does_not_crash(self):
        frame = pd.DataFrame({"Region": [None, None], "Revenue": [1.0, 2.0]})
        plan = QueryPlan(
            intent="rank", aggregation="sum", measure="Revenue", dimension="Region", top_n=5
        )

        answer = execute_plan(plan, frame, self._roles(frame))

        self.assertTrue(answer.answer)


class AnswerWordingTests(unittest.TestCase):
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
        # A row count is not formatted as currency.
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


class LevelQuestionTests(unittest.TestCase):
    def setUp(self):
        months = pd.date_range("2024-01-01", periods=6, freq="MS")
        rows = []
        for index, month in enumerate(months):
            rows.append((month, "North", 14.0 - (2.0 if index == 5 else 0.0)))
            rows.append((month, "South", 12.0 + (0.5 if index == 5 else 0.0)))
        self.frame = pd.DataFrame(rows, columns=["Survey Date", "Site", "Water Temp (°C)"])
        self.roles = detect_roles(self.frame)

    def ask(self, question):
        return answer_question(question, self.frame, self.roles)

    def test_growth_of_a_level_is_a_difference(self):
        result = self.ask("water temp growth")
        self.assertIn("down 0.75", result.answer)
        self.assertIn("period average", result.calculation)

    def test_segments_of_a_level_rank_by_change_in_average(self):
        result = self.ask("which site grew fastest")
        self.assertIn("South rose most: +0.5 (12 → 12.50)", result.answer)
        self.assertIn("Change", result.table.columns)

    def test_the_suggested_question_averages_a_level(self):
        self.assertIn("Average Water Temp (°C)", suggested_questions(self.frame, self.roles))


class UnfamiliarMeasureTests(unittest.TestCase):
    def test_the_highest_group_of_an_unfamiliar_reading_is_by_its_average(self):
        # Three USA cars against one Japanese car: a total would crown the USA.
        frame = pd.DataFrame({"origin": ["usa", "usa", "usa", "japan"], "mpg": [15.0, 16.0, 17.0, 32.0]})

        result = answer_question("which origin has the highest mpg", frame, detect_roles(frame))

        self.assertEqual(result.plan.aggregation, "mean")
        self.assertTrue(
            result.answer.startswith("japan has the highest average mpg of any origin"), result.answer
        )

    def test_a_daily_price_trend_by_year_averages_the_days(self):
        days = pd.date_range("2020-01-01", "2021-12-31", freq="D")
        frame = pd.DataFrame({"Date": days, "Price": np.where(days.year == 2020, 100.0, 110.0)})

        result = answer_question("yearly Price trend", frame, detect_roles(frame))

        self.assertIn("moved from $100.00 to $110.00", result.answer)


class AggregationOverrideTests(unittest.TestCase):
    def setUp(self):
        months = pd.date_range("2024-01-01", periods=4, freq="MS")
        self.frame = pd.DataFrame(
            {
                "Date": np.repeat(months, 2),
                "Gauge": ["North", "South"] * 4,
                "Rainfall (mm)": [10.0, 20.0, 12.0, 22.0, 14.0, 24.0, 16.0, 26.0],
                "Water Temp": [5.0, 7.0, 5.5, 7.5, 6.0, 8.0, 6.5, 8.5],
            }
        )
        detected = detect_roles(self.frame)
        self.roles = ColumnRoles(
            date="Date", measure="Rainfall (mm)", dimension="Gauge", identifier=None,
            numeric=detected.numeric, dimensions=detected.dimensions, aggregation="sum",
        )

    def ask(self, question):
        return answer_question(question, self.frame, self.roles)

    def test_the_measure_follows_the_override(self):
        breakdown = self.ask("rainfall by gauge")
        self.assertEqual(breakdown.plan.aggregation, "sum")
        self.assertIn("Total Rainfall (mm)", breakdown.table.columns)

        trend = self.ask("monthly rainfall trend")
        self.assertIn("moved from 30 to 42", trend.answer)
        self.assertIn("Total Rainfall (mm)", suggested_questions(self.frame, self.roles))

    def test_a_summed_measure_grows_by_a_percentage(self):
        self.assertIn("up 10.5%", self.ask("how did rainfall change").answer)

    def test_another_measure_keeps_its_own_aggregation(self):
        result = self.ask("water temp by gauge")
        self.assertEqual(result.plan.aggregation, "mean")

        trend = self.ask("monthly water temp trend")
        self.assertIn("moved from 6 to 7.50", trend.answer)


class PluralAndWordMatchTests(unittest.TestCase):
    def test_an_ies_plural_matches_a_y_column(self):
        frame = pd.DataFrame({"Category": list("abcabcabca"), "Revenue": [float(i) for i in range(10)]})
        result = answer_question("top 2 categories by revenue", frame, detect_roles(frame))
        self.assertEqual(result.plan.dimension, "Category")

    def test_a_name_that_shortens_the_words_names_its_column(self):
        frame = pd.DataFrame({"continent": ["Africa", "Asia", "Europe"] * 4,
                              "lifeExp": [52.0, 70.0, 77.0] * 4, "pop": [1e6, 5e6, 2e6] * 4})
        roles = detect_roles(frame)

        answer = answer_question("average life expectancy by continent", frame, roles)
        unrelated = answer_question("average lift exposure by continent", frame, roles)

        self.assertEqual(answer.plan.measure, "lifeExp")
        self.assertIn('Read "life expectancy" as lifeExp', answer.answer)
        self.assertNotEqual(getattr(unrelated, "plan", None) and unrelated.plan.measure, "lifeExp")

    def test_a_column_name_does_not_flip_the_ranking(self):
        frame = pd.DataFrame({"Store": ["A", "A", "B", "B"], "Lowest Price": [1.0, 2.0, 10.0, 20.0]})
        result = answer_question("which store has the highest lowest price", frame, detect_roles(frame))
        self.assertFalse(result.plan.ascending)
        self.assertTrue(result.answer.startswith("B "))


class SuggestionWordingTests(unittest.TestCase):
    def test_a_reading_is_compared_and_related_rather_than_ranked(self):
        frame = pd.DataFrame({
            "Run Date": pd.date_range("2024-01-01", periods=60, freq="W").repeat(2),
            "Treatment": ["Control", "Dosed"] * 60,
            "Viability %": np.linspace(80, 95, 120),
            "Live Cells": np.linspace(1e6, 2e6, 120),
        })

        suggestions = suggested_questions(frame, detect_roles(frame))

        self.assertIn("Does Viability % differ by Treatment?", suggestions)
        self.assertIn("What relates to Viability %?", suggestions)
        self.assertNotIn("Which Treatment grew fastest?", suggestions)

    def test_a_measure_named_average_is_not_prefixed_again(self):
        frame = pd.DataFrame({"Average Order Value": [1.0, 2.0, 3.0, 4.0], "Region": list("abab")})
        suggestions = suggested_questions(frame, detect_roles(frame))
        self.assertIn("Average Order Value", suggestions)
        self.assertNotIn("Average Average Order Value", suggestions)


def measurement_frame() -> tuple[pd.DataFrame, ColumnRoles]:
    """Unit-suffixed measurements whose default measure, bill_depth_mm, is never the one asked for."""
    frame = pd.DataFrame({
        "island": ["Biscoe", "Dream", "Torgersen"] * 4,
        "bill_depth_mm": [18.0, 17.5, 19.0, 18.2] * 3,
        "bill_length_mm": [45.0, 44.0, 39.0, 41.0] * 3,
        "flipper_length_mm": [210.0, 190.0, 195.0, 200.0] * 3,
        "body_mass_g": [5000.0, 3700.0, 3800.0, 4200.0] * 3,
    })
    numeric = ("bill_depth_mm", "bill_length_mm", "flipper_length_mm", "body_mass_g")
    roles = ColumnRoles(date=None, measure="bill_depth_mm", dimension="island", identifier=None,
                        numeric=numeric, dimensions=("island",))
    return frame, roles


class UnitAnnotatedNameTests(unittest.TestCase):
    def test_a_name_without_its_unit_names_the_column(self):
        frame, roles = measurement_frame()
        for question, column in (
            ("average body mass by island", "body_mass_g"),
            ("average flipper length by island", "flipper_length_mm"),
            ("average body mass g by island", "body_mass_g"),
        ):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertEqual(answer.plan.measure, column)
                self.assertIn(f"mean({column})", answer.calculation)

    def test_a_bracketed_unit_is_left_out(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="MS").repeat(2)
        frame = pd.DataFrame({
            "Collection Date": dates,
            "Site": ["River A", "Lake C"] * 12,
            "Analyte": ["Nitrate", "Lead", "Lead", "Nitrate"] * 6,
            "Concentration (mg/L)": np.linspace(1.0, 5.0, 24),
            "Temp (°C)": [12.0, 14.0] * 12,
        })
        roles = ColumnRoles(date="Collection Date", measure="Temp (°C)", dimension="Site", identifier=None,
                            numeric=("Concentration (mg/L)", "Temp (°C)"), dimensions=("Site", "Analyte"))
        for question, intent in (
            ("which site has the highest concentration", "rank"),
            ("average concentration by analyte", "breakdown"),
            ("monthly concentration trend", "trend"),
        ):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertEqual(answer.plan.intent, intent)
                self.assertEqual(answer.plan.measure, "Concentration (mg/L)")

    def test_a_partial_name_one_column_fits_names_that_column(self):
        frame, roles = measurement_frame()

        answer = answer_question("average body by island", frame, roles)

        self.assertEqual(answer.plan.measure, "body_mass_g")
        self.assertIn('Read "body" as body_mass_g.', answer.answer)

    def test_a_word_in_several_names_names_each_of_them(self):
        frame, roles = measurement_frame()

        self.assertIsNone(answer_question("average length by island", frame, roles))
        self.assertEqual(
            rephrase_hint("average length by island", frame, roles),
            "Did you mean bill_length_mm or flipper_length_mm?",
        )

    def test_min_and_max_are_not_read_as_units(self):
        frame = pd.DataFrame({"Station": list("abab"), "temp_max": [20.0, 22.0, 19.0, 25.0],
                              "temp_min": [5.0, 7.0, 4.0, 9.0]})
        roles = detect_roles(frame)

        self.assertIsNone(answer_question("average temp by station", frame, roles))
        self.assertEqual(
            rephrase_hint("average temp by station", frame, roles), "Did you mean temp_max or temp_min?"
        )
        answer = answer_question("average temp max by station", frame, roles)
        self.assertEqual(answer.plan.measure, "temp_max")

    def test_a_core_name_two_columns_share_names_neither(self):
        frame = pd.DataFrame({"Site": list("abab"), "Temp (°C)": [10.0, 12.0, 11.0, 13.0],
                              "Temp (°F)": [50.0, 53.6, 51.8, 55.4]})
        roles = detect_roles(frame)

        self.assertIsNone(answer_question("average temp by site", frame, roles))
        self.assertIn("Temp (°C) or Temp (°F)", rephrase_hint("average temp by site", frame, roles))


def timepoint_frame() -> tuple[pd.DataFrame, ColumnRoles]:
    """Twenty timepoints, three rows each; the signal peaks at timepoint 5."""
    points = np.repeat(np.arange(20), 3)
    frame = pd.DataFrame({
        "Region": np.tile(["North", "South", "East"], 20),
        "timepoint": points,
        "signal": 10.0 - np.abs(points - 5),
        "weight": np.arange(60, dtype=float) + 0.5,
    })
    roles = ColumnRoles(date=None, measure="signal", dimension="Region", identifier=None,
                        numeric=("timepoint", "signal", "weight"), dimensions=("Region",))
    return frame, roles


class NumericGroupingTests(unittest.TestCase):
    def test_a_numeric_column_after_by_is_the_grouping(self):
        frame, roles = timepoint_frame()

        answer = answer_question("mean signal by timepoint", frame, roles)

        self.assertEqual(answer.plan.intent, "breakdown")
        self.assertEqual((answer.plan.measure, answer.plan.dimension), ("signal", "timepoint"))
        self.assertEqual(answer.plan.aggregation, "mean")
        self.assertIn("timepoint 5 has the highest average signal", answer.answer)

    def test_a_numeric_code_among_the_segments_still_groups(self):
        frame, roles = timepoint_frame()
        roles = replace(roles, dimensions=("Region", "timepoint"), factors=("timepoint",))

        for question in ("mean signal by timepoint", "which timepoint has the highest signal"):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertEqual((answer.plan.measure, answer.plan.dimension), ("signal", "timepoint"))

    def test_numbered_groups_keep_their_order_past_the_breakdown_limit(self):
        frame, roles = timepoint_frame()

        answer = answer_question("mean signal by timepoint", frame, roles)

        self.assertEqual(list(answer.table["timepoint"]), list(range(20)))
        self.assertIn("in timepoint order, showing 20", answer.calculation)
        self.assertEqual(answer.chart, "line")

    def test_a_ranking_still_sorts_numbered_groups_by_value(self):
        frame, roles = timepoint_frame()

        which = answer_question("which timepoint has the highest signal", frame, roles)
        top = answer_question("top 3 timepoints by signal", frame, roles)

        self.assertEqual((which.plan.intent, which.plan.dimension), ("rank", "timepoint"))
        self.assertEqual(which.table.iloc[0]["timepoint"], 5)
        self.assertEqual(len(top.table), 3)
        self.assertEqual(top.table.iloc[0]["timepoint"], 5)

    def test_a_continuous_grouping_is_split_at_its_quartiles(self):
        frame, roles = timepoint_frame()

        answer = answer_question("average signal by weight", frame, roles)

        self.assertEqual((answer.plan.dimension, answer.plan.bins), ("weight", 4))
        self.assertEqual(answer.table["weight"].astype(str).tolist()[0], "Q1: 0.5 to 15.25")
        self.assertEqual(answer.table["n"].tolist(), [15, 15, 15, 15])
        self.assertIn("split into 4 groups of about equal size at its quartiles", answer.answer)

    def test_two_groupings_give_each_cell_with_its_count(self):
        frame, roles = timepoint_frame()

        answer = answer_question("mean signal by timepoint by region", frame, roles)

        self.assertEqual(answer.plan.intent, "crosstab")
        self.assertIn("highest for timepoint 5", answer.answer)
        self.assertIn("fewer than 5 values", answer.answer)

    def test_after_a_segment_the_column_after_by_is_still_the_measure(self):
        frame, roles = timepoint_frame()

        answer = answer_question("top 2 region by signal", frame, roles)

        self.assertEqual((answer.plan.dimension, answer.plan.measure), ("Region", "signal"))

    def test_the_default_measure_is_used_when_only_the_grouping_is_named(self):
        frame, roles = timepoint_frame()

        answer = answer_question("average by timepoint", frame, roles)

        self.assertEqual((answer.plan.dimension, answer.plan.measure), ("timepoint", "signal"))

    def test_a_text_column_kept_out_of_the_segments_by_its_name_can_group(self):
        frame = pd.DataFrame({"diet": ["low fat", "no fat"] * 6, "pulse": np.arange(80, 92),
                              "time": ["1 min", "15 min", "30 min"] * 4})
        roles = detect_roles(frame)
        self.assertNotIn("time", roles.dimensions)

        answer = answer_question("mean pulse by time", frame, roles)

        self.assertEqual((answer.plan.intent, answer.plan.dimension), ("breakdown", "time"))

    def test_a_date_column_after_by_asks_for_the_trend(self):
        frame = pd.DataFrame({"Order Date": pd.date_range("2024-01-01", periods=120, freq="D"),
                              "Revenue": np.arange(120, dtype=float)})

        answer = answer_question("revenue by order date", frame, detect_roles(frame))

        self.assertEqual(answer.plan.intent, "trend")

    def test_the_chart_follows_the_table_order(self):
        frame, roles = timepoint_frame()

        figure = ui._chat_answer_figure(answer_question("mean signal by timepoint", frame, roles))

        self.assertEqual(list(figure.data[0].x), list(range(20)))


class NumericCountTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"subject": np.tile(np.arange(1, 21), 3),
                                   "Units": np.tile([1, 2, 3, 4, 5], 12),
                                   "score": np.linspace(1.0, 9.0, 60)})
        self.roles = detect_roles(self.frame)

    def test_a_numbered_entity_column_counts_distinct_values(self):
        answer = answer_question("how many subjects", self.frame, self.roles)

        self.assertEqual(answer.plan.count_column, "subject")
        self.assertIn("20 distinct subject values", answer.answer)

    def test_a_quantity_column_is_not_counted_as_entities(self):
        answer = answer_question("how many units", self.frame, self.roles)

        self.assertIsNone(answer.plan.count_column)


def paired_frame(rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = np.arange(rows, dtype=float)
    y = 2 * x + rng.normal(0, 4, rows)
    return pd.DataFrame({"x": x, "y": y, "group": ["Alpha", "Beta"] * (rows // 2)})


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.frame = paired_frame()
        self.roles = detect_roles(self.frame)

    def test_each_phrasing_asks_for_a_correlation(self):
        for question in ("correlation between x and y", "is x correlated with y",
                         "how does x relate to y", "relationship between x and y"):
            with self.subTest(question=question):
                plan = parse_question(question, self.frame, self.roles)
                self.assertEqual((plan.intent, plan.measure, plan.partner), ("correlation", "x", "y"))

    def test_the_answer_reports_r_its_interval_n_and_the_rank_correlation(self):
        answer = answer_question("correlation between x and y", self.frame, self.roles)
        expected = np.corrcoef(self.frame["x"], self.frame["y"])[0, 1]

        self.assertIn(f"Pearson r = {expected:.2f}", answer.answer)
        self.assertIn("95% CI", answer.answer)
        self.assertIn("n = 40", answer.answer)
        self.assertIn("rank correlation", answer.answer)
        self.assertIn("not evidence of cause", answer.answer)
        self.assertNotIn("strongest", answer.answer)
        self.assertIn("Fisher", answer.calculation)
        self.assertIn("Spearman", answer.calculation)

    def test_the_wording_is_the_brief_wording(self):
        answer = answer_question("correlation between x and y", self.frame, self.roles)

        brief = describe_correlation(measure_correlation(self.frame[["x", "y"]]))
        self.assertTrue(answer.answer.startswith(brief), answer.answer)

    def test_the_answer_carries_the_pairs_for_a_scatter(self):
        answer = answer_question("correlation between x and y", self.frame, self.roles)

        self.assertEqual(answer.chart, "scatter")
        self.assertEqual(list(answer.table.columns), ["x", "y"])
        self.assertEqual(ui._chat_answer_figure(answer).data[0].mode, "markers")

    def test_a_filter_narrows_the_pairs(self):
        answer = answer_question("correlation between x and y for Alpha", self.frame, self.roles)

        self.assertEqual(answer.plan.filters[0].values, ("Alpha",))
        self.assertIn("n = 20", answer.answer)

    def test_too_few_complete_pairs_are_refused_with_the_count(self):
        frame = self.frame.copy()
        frame.loc[11:, "y"] = np.nan

        answer = answer_question("correlation between x and y", frame, detect_roles(frame))

        self.assertIn("Only 11 rows have both x and y", answer.answer)
        self.assertNotIn("Pearson", answer.answer)

    def test_a_text_column_cannot_be_correlated(self):
        question = "correlation between x and group"

        self.assertIsNone(answer_question(question, self.frame, self.roles))
        self.assertIn("group is not numeric", rephrase_hint(question, self.frame, self.roles))
        plan = QueryPlan(intent="correlation", measure="x", partner="group")
        with self.assertRaises(ValueError):
            execute_plan(plan, self.frame, self.roles)

    def test_one_correlation_per_group_is_refused(self):
        self.assertIsNone(answer_question("correlation between x and y by group", self.frame, self.roles))

    def test_a_constant_column_has_no_correlation(self):
        frame = self.frame.assign(y=1.0)

        answer = answer_question("correlation between x and y", frame, detect_roles(frame))

        self.assertIn("y has a single value", answer.answer)

    def test_an_unrelated_pair_says_it_cannot_be_told_from_none(self):
        rng = np.random.default_rng(3)
        frame = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})

        answer = answer_question("correlation between x and y", frame, detect_roles(frame))

        self.assertIn("cannot be told apart from no relationship", answer.answer)


def daily_frame() -> pd.DataFrame:
    """Two years of daily revenue with a spike through July 2024."""
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    revenue = np.where((dates.year == 2024) & (dates.month == 7), 1_000.0, 100.0)
    return pd.DataFrame({"Order Date": dates, "Revenue": revenue})


class PeriodRankTests(unittest.TestCase):
    def setUp(self):
        self.frame = daily_frame()
        self.roles = detect_roles(self.frame)

    def test_which_month_names_the_top_period_and_its_value(self):
        answer = answer_question("which month had the highest revenue", self.frame, self.roles)

        self.assertEqual((answer.plan.intent, answer.plan.grain, answer.plan.dimension), ("rank", "M", None))
        self.assertTrue(answer.answer.startswith("Jul 2024 had the highest total revenue of any month"))
        self.assertIn("$31.0K", answer.answer)
        self.assertIn("sum(Revenue) per month", answer.calculation)
        self.assertEqual(answer.chart, "line")

    def test_the_lowest_period_ranks_ascending(self):
        answer = answer_question("which year had the lowest revenue", self.frame, self.roles)

        self.assertTrue(answer.plan.ascending)
        self.assertTrue(answer.answer.startswith("2025 had the lowest"), answer.answer)

    def test_an_aggregation_the_trend_does_not_use_is_refused(self):
        self.assertIsNone(
            answer_question("which month had the highest average revenue", self.frame, self.roles)
        )

    def test_rows_rank_row_counts(self):
        answer = answer_question("which month had the most rows", self.frame, self.roles)

        self.assertIsNone(answer.plan.measure)
        self.assertIn("row count", answer.answer)

    def test_without_a_date_column_there_are_no_periods_to_rank(self):
        frame = self.frame.drop(columns=["Order Date"])

        plan = parse_question("which month had the highest revenue", frame, detect_roles(frame))

        self.assertFalse(plan is not None and plan.intent == "rank" and plan.grain)


class ConsistentAggregationTests(unittest.TestCase):
    """Every intent combines a measure the same way unless the question names an aggregation."""

    def test_a_level_is_averaged_by_every_intent(self):
        frame = pd.DataFrame({"Survey Date": MONTHS.repeat(2), "Site": ["North", "South"] * 12,
                              "Water Temp (°C)": np.linspace(10.0, 14.0, 24)})
        roles = detect_roles(frame)
        for question in ("which site has the highest water temp", "water temp by site",
                         "top 1 site by water temp", "water temp", "monthly water temp trend",
                         "how did water temp change", "which month had the highest water temp"):
            with self.subTest(question=question):
                self.assertEqual(parse_question(question, frame, roles).aggregation, "mean")

    def test_an_amount_is_summed_by_every_intent(self):
        frame = pd.DataFrame({"Month": MONTHS.repeat(2), "Region": ["North", "South"] * 12,
                              "Revenue": np.linspace(100.0, 400.0, 24)})
        roles = detect_roles(frame)
        for question in ("which region has the highest revenue", "revenue by region",
                         "revenue", "monthly revenue trend", "how did revenue change"):
            with self.subTest(question=question):
                self.assertEqual(parse_question(question, frame, roles).aggregation, "sum")

    def test_a_numeric_grouping_takes_its_aggregation_from_the_measure(self):
        frame = pd.DataFrame({"timepoint": np.repeat(np.arange(4), 3), "Temp (°C)": np.linspace(10, 12, 12)})

        plan = parse_question("temp by timepoint", frame, detect_roles(frame))

        self.assertEqual((plan.measure, plan.aggregation), ("Temp (°C)", "mean"))


class RankingWordingTests(unittest.TestCase):
    def test_groups_tied_at_the_top_are_all_named(self):
        frame = pd.DataFrame({"Site": ["North", "South", "East"] * 4, "pH": [7.0, 7.0, 6.5] * 4})

        answer = answer_question("which site has the highest pH", frame, detect_roles(frame))

        self.assertTrue(answer.answer.startswith("North and South tie for the highest average pH"))

    def test_a_symbol_keeps_its_case_and_a_total_is_not_named_twice(self):
        frame = pd.DataFrame({"Station": ["A", "B"] * 6, "Total Rain": [1.5, 0.5] * 6, "Ca": [13.4, 4.8] * 6})
        roles = detect_roles(frame)

        rain = answer_question("total rain by station", frame, roles)
        calcium = answer_question("average Ca by station", frame, roles)

        self.assertNotIn("total total", rain.answer.lower())
        self.assertIn("average Ca", calcium.answer)

    def test_a_coded_band_is_not_answered_in_dollars(self):
        frame = pd.DataFrame({"income": [1.0, 5.0, 12.0, 15.0] * 5})

        answer = answer_question("average income", frame, detect_roles(frame))

        self.assertNotIn("$", answer.answer)


def dye_frame() -> tuple[pd.DataFrame, ColumnRoles]:
    """Two dyes with long labels; blue falls with concentration only for the yellow dye."""
    concentration = np.tile(np.arange(1, 21, dtype=float), 2)
    blue = np.concatenate([200 - 5 * np.arange(20), 100 + 0 * np.arange(20)]).astype(float)
    blue[::7] += 3
    frame = pd.DataFrame({
        "Dye Label": ["Tartrazine, FD&C Yellow 5 (US), E102 (EU)"] * 20
        + ["Sunset Yellow, FD&C Yellow 6 (US)"] * 20,
        "Condition": ["Open", "Closed"] * 20,
        "Concentration": concentration,
        "B": blue,
    })
    roles = ColumnRoles(date=None, measure="Concentration", dimension="Dye Label", identifier=None,
                        numeric=("Concentration", "B"), dimensions=("Dye Label", "Condition"))
    return frame, roles


class NothingLeftOutTests(unittest.TestCase):
    """Every column, value and grouping a question names is used, or the question is refused."""

    def test_a_value_matches_on_the_first_word_of_a_long_label(self):
        frame, roles = dye_frame()

        answer = answer_question("correlation between Concentration and B for Tartrazine", frame, roles)

        self.assertEqual(answer.plan.filters[0].values, ("Tartrazine, FD&C Yellow 5 (US), E102 (EU)",))
        self.assertIn("n = 20", answer.answer)
        self.assertIn("Based on 20 of 40 rows.", answer.answer)

    def test_a_word_in_several_labels_is_refused_with_them(self):
        frame, roles = dye_frame()
        question = "average B for Yellow"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("matches several Dye Label values", rephrase_hint(question, frame, roles))

    def test_a_value_the_table_lacks_is_refused_and_named(self):
        frame, roles = dye_frame()
        question = "average B for A1_1"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn('"A1_1"', rephrase_hint(question, frame, roles))
        hint = rephrase_hint(question, frame, roles, rows_not_analyzed=500)
        self.assertIn("the row limit left out 500 others", hint)

    def test_a_second_measure_is_not_dropped(self):
        frame, roles = dye_frame()
        question = "average B and Concentration by Condition"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("names B and Concentration", rephrase_hint(question, frame, roles))

    def test_a_change_with_another_column_is_their_correlation(self):
        frame, roles = dye_frame()

        answer = answer_question("does B change with Concentration", frame, roles)

        plan = answer.plan
        self.assertEqual((plan.intent, plan.measure, plan.partner), ("correlation", "B", "Concentration"))

    def test_a_filter_written_with_its_column(self):
        frame, roles = dye_frame()

        answer = answer_question("average B by Dye Label where Condition is Open", frame, roles)

        self.assertEqual((answer.plan.dimension, answer.plan.filters[0].values), ("Dye Label", ("Open",)))

    def test_a_grouping_with_many_groups_shows_the_highest_and_lowest(self):
        frame = pd.DataFrame({"Site": [f"Site {index:02d}" for index in range(30)] * 2,
                              "Reading": np.arange(60, dtype=float)})
        roles = ColumnRoles(date=None, measure="Reading", dimension=None, identifier=None,
                            numeric=("Reading",), dimensions=())

        answer = answer_question("average Reading by Site", frame, roles)

        self.assertEqual((answer.plan.intent, answer.plan.dimension), ("breakdown", "Site"))
        self.assertEqual(len(answer.table), 20)
        self.assertIn("Site has 30 groups; the table shows the 10 highest and the 10 lowest", answer.answer)
        self.assertEqual(answer.table.iloc[-1]["Site"], "Site 00")

    def test_a_grouping_with_one_value_is_refused_with_the_reason(self):
        frame = pd.DataFrame({"Sample": ["O"] * 20, "Signal": np.arange(20, dtype=float)})
        roles = detect_roles(frame)
        question = "average Signal by Sample"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("Sample has a single value (O)", rephrase_hint(question, frame, roles))

    def test_a_text_column_is_not_swapped_for_the_default_measure(self):
        frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=60, freq="D"),
                              "Precip": ["T", "0", "1.2"] * 20, "Snow": np.arange(60) / 2})
        roles = detect_roles(frame)
        question = "average Precip by month"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("Precip holds text", rephrase_hint(question, frame, roles))

    def test_top_n_without_a_grouping_is_refused(self):
        frame = pd.DataFrame({"Reading": np.linspace(0.5, 30.0, 60)})
        question = "top 5 Reading"

        self.assertIsNone(answer_question(question, frame, detect_roles(frame)))
        self.assertIn('"top 5" needs a column to rank', rephrase_hint(question, frame, detect_roles(frame)))

    def test_a_value_of_the_grouping_column_is_refused(self):
        frame, roles = dye_frame()
        question = "average B by Condition for Open"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("which the answer groups by", rephrase_hint(question, frame, roles))

    def test_two_groupings_give_a_cross_table(self):
        frame, roles = dye_frame()

        answer = answer_question("average B by Dye Label by Condition", frame, roles)

        self.assertEqual(answer.plan.intent, "crosstab")
        self.assertEqual((answer.plan.dimension, answer.plan.second_dimension), ("Dye Label", "Condition"))
        self.assertEqual(set(answer.table.columns), {"Dye Label", "Closed", "Open", "n Closed", "n Open"})

    def test_an_average_per_unit_by_a_grouping_is_refused_with_the_reason(self):
        frame, roles = dye_frame()

        question = "average B per Dye Label by Condition"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("two groupings", rephrase_hint(question, frame, roles))


class AggregationWordTests(unittest.TestCase):
    def test_an_explicit_total_is_honoured_in_a_breakdown(self):
        frame = pd.DataFrame({"Station": ["North", "South"] * 5, "Interval Rain": np.arange(10, dtype=float)})
        roles = detect_roles(frame)

        answer = answer_question("total Interval Rain by Station", frame, roles)

        self.assertEqual(answer.plan.aggregation, "sum")
        self.assertIn("Total Interval Rain", answer.table.columns)

    def test_an_aggregation_word_that_starts_a_column_name_is_read_as_the_aggregation(self):
        frame = pd.DataFrame({"Wind Speed": [2.0, 3.0, 999.9], "Maximum Wind Speed": [4.0, 5.0, 6.0]})
        roles = detect_roles(frame)

        answer = answer_question("maximum Wind Speed", frame, roles)
        column = answer_question("average Maximum Wind Speed", frame, roles)

        self.assertEqual((answer.plan.measure, answer.plan.aggregation), ("Wind Speed", "max"))
        self.assertIn("999.90", answer.answer)
        self.assertIn('ask for "average Maximum Wind Speed"', answer.answer)
        self.assertEqual((column.plan.measure, column.plan.aggregation), ("Maximum Wind Speed", "mean"))


class ShortNameTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Depth": ["0-10", "10-30"] * 6,
            "Conduc": np.linspace(1.0, 2.0, 12),
            "Concrete compressive strength (MPa)": np.linspace(20.0, 60.0, 12),
            "1001": np.arange(12, dtype=float),
            "1": np.arange(12, dtype=float) ** 2,
        })
        self.roles = detect_roles(self.frame)

    def test_a_word_one_column_name_holds(self):
        answer = answer_question("what is the maximum strength", self.frame, self.roles)

        self.assertEqual(answer.plan.measure, "Concrete compressive strength (MPa)")
        self.assertIn('Read "strength" as Concrete compressive strength (MPa).', answer.answer)

    def test_an_abbreviated_name_that_starts_the_word(self):
        answer = answer_question("average conductivity by depth", self.frame, self.roles)

        self.assertEqual((answer.plan.measure, answer.plan.dimension), ("Conduc", "Depth"))

    def test_numeric_column_names(self):
        answer = answer_question("correlation between 1001 and 1", self.frame, self.roles)

        self.assertEqual((answer.plan.measure, answer.plan.partner), ("1001", "1"))


class ConditionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "OBJECT": [0, 1, 2, 0] * 6,
            "Status": [0, 1, 2, 2] * 6,
            "TIME": np.arange(24, dtype=float) * 10,
            "Yield": [-5.0, 10.0, 20.0, 30.0] * 6,
            "Age of testing (day)": [7, 28, 28, 90] * 6,
        })
        self.roles = ColumnRoles(date=None, measure="TIME", dimension=None, identifier=None,
                                 numeric=("OBJECT", "Status", "TIME", "Yield", "Age of testing (day)"),
                                 dimensions=("Status",), factors=("Status",))

    def ask(self, question):
        answer = answer_question(question, self.frame, self.roles)
        self.assertIsNotNone(answer, rephrase_hint(question, self.frame, self.roles))
        return answer

    def test_a_value_after_is_or_a_code_column(self):
        for question, column, value in (
            ("average TIME when OBJECT is 0", "OBJECT", 0.0),
            ("average TIME for Status 2", "Status", 2.0),
            ("average TIME at 28 days", "Age of testing (day)", 28.0),
        ):
            with self.subTest(question=question):
                answer = self.ask(question)
                self.assertEqual(answer.plan.conditions, (Condition(column, "=", value),))
                self.assertEqual(answer.plan.measure, "TIME")

    def test_thresholds_and_signs(self):
        for question, operator, value in (
            ("how many rows have Yield below 15", "<", 15.0),
            ("where Yield < 0 how many rows", "<", 0.0),
            ("how many rows have negative Yield", "<", 0.0),
            ("average TIME where Yield >= 20", ">=", 20.0),
        ):
            with self.subTest(question=question):
                self.assertEqual(self.ask(question).plan.conditions, (Condition("Yield", operator, value),))

    def test_a_count_with_a_condition_gives_its_share(self):
        answer = self.ask("how many rows have negative Yield")

        self.assertIn("6 of the 24 rows (25.0%) have Yield below 0", answer.answer)

    def test_a_numeric_grouping_is_named_with_its_column(self):
        answer = self.ask("average TIME by Status")

        self.assertEqual(answer.plan.dimension, "Status")
        self.assertTrue(answer.answer.startswith("Status 2 has the highest average TIME"), answer.answer)


class CountTests(unittest.TestCase):
    def test_how_many_in_each_group(self):
        frame = pd.DataFrame({"ParticipantId": [str(index) for index in range(10)],
                              "Condition": [1, 2] * 5, "Score": np.arange(10, dtype=float)})
        roles = detect_roles(frame)

        each = answer_question("how many in each Condition", frame, roles)
        participants = answer_question("how many participants per Condition", frame, roles)

        self.assertEqual((each.plan.intent, each.plan.aggregation, each.plan.dimension),
                         ("breakdown", "count", "Condition"))
        self.assertEqual(participants.plan.dimension, "Condition")
        self.assertEqual(list(participants.table.iloc[:, 1]), [5, 5])

    def test_entity_nouns_count_rows_or_distinct_ids(self):
        frame = pd.DataFrame({"ID": ["43", "43", "47", "47", "68"],
                              "Result": ["positive", "negative"] * 2 + ["positive"],
                              "TIME": [1.0, 2.0, 3.0, 4.0, 5.0]})
        roles = detect_roles(frame)

        birds = answer_question("how many birds are there", frame, roles)
        unnumbered = frame.drop(columns=["ID"])
        positive = answer_question("how many patients are positive", unnumbered, detect_roles(unnumbered))

        self.assertEqual(birds.plan.count_column, "ID")
        self.assertIn("There are 3 distinct ID values", birds.answer)
        self.assertIn("Counted the birds as distinct ID values.", birds.answer)
        self.assertEqual(positive.plan.filters[0].values, ("positive",))
        self.assertTrue(positive.answer.startswith("3 of the 5 rows (60.0%) have Result = positive"))
        self.assertIn("Each row counts as one of the patients.", positive.answer)

    def test_a_flag_gives_its_share_of_the_rows_that_record_it(self):
        frame = pd.DataFrame({"diabetes": [0.0, 1.0, 0.0, 0.0, np.nan],
                              "age": [30.0, 40.0, 50.0, 60.0, 70.0]})
        roles = detect_roles(frame)

        share = answer_question("what share have diabetes", frame, roles)
        people = answer_question("how many people have diabetes", frame, roles)

        self.assertEqual(share.plan.conditions, (Condition("diabetes", "=", 1.0),))
        self.assertIn("1 of the 4 rows with diabetes recorded (25.0%)", share.answer)
        self.assertIn("Each row counts as one of the people.", people.answer)

    def test_a_quantity_counts_by_its_total(self):
        frame = pd.DataFrame({"Region": ["a", "b"] * 3, "Units": [1, 2, 3, 4, 5, 6]})

        answer = answer_question("how many units", frame, detect_roles(frame))

        self.assertEqual((answer.plan.intent, answer.plan.aggregation), ("aggregate", "sum"))


class RowLabelTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Taxonomy": ["Hungatella hathewayi", "Prevotella copri", "Bacteroides fragilis",
                         "Dorea longicatena"],
            "A101": [6849.0, 120.0, 3000.0, 120.0],
        })
        self.roles = ColumnRoles(date=None, measure="A101", dimension=None, identifier=None,
                                 numeric=("A101",), dimensions=())

    def test_which_label_has_the_highest_value_names_the_row(self):
        answer = answer_question("which Taxonomy has the highest A101", self.frame, self.roles)

        self.assertEqual((answer.plan.intent, answer.plan.dimension), ("rank", "Taxonomy"))
        self.assertTrue(answer.answer.startswith("Hungatella hathewayi has the highest A101"), answer.answer)

    def test_top_n_labels_and_ties(self):
        top = answer_question("top 2 Taxonomy by A101", self.frame, self.roles)
        lowest = answer_question("which Taxonomy has the shortest A101", self.frame, self.roles)

        self.assertEqual(list(top.table["Taxonomy"]), ["Hungatella hathewayi", "Bacteroides fragilis"])
        self.assertIn("Dorea longicatena and Prevotella copri have the lowest A101, at 120", lowest.answer)


class MissingValueTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"Condition": [1, 2] * 5, "EngTime": [1.0, np.nan] * 5,
                                   "Button": [np.nan] * 3 + [1.0] * 7})
        self.roles = detect_roles(self.frame)

    def test_which_columns_have_missing_values(self):
        for question in ("which columns have missing values", "how much is missing"):
            with self.subTest(question=question):
                answer = answer_question(question, self.frame, self.roles)
                self.assertEqual(answer.plan.intent, "missing")
                self.assertEqual(list(answer.table["Column"]), ["EngTime", "Button"])
                self.assertIn("2 of 3 columns have missing values", answer.answer)

    def test_how_many_rows_miss_one_column(self):
        answer = answer_question("how many rows have missing EngTime", self.frame, self.roles)

        self.assertIn("5 of the 10 rows (50.0%) have EngTime missing", answer.answer)


class TimePartTests(unittest.TestCase):
    def test_hour_and_weekday_groupings(self):
        stamps = pd.date_range("2024-01-01", periods=24 * 14, freq="h")
        frame = pd.DataFrame({"Datetime": stamps, "PM2.5": np.where(stamps.hour == 8, 90.0, 30.0)})
        roles = detect_roles(frame)

        hourly = answer_question("average PM2.5 by hour", frame, roles)
        weekly = answer_question("average PM2.5 by weekday", frame, roles)

        self.assertEqual((hourly.plan.time_part, len(hourly.table)), ("hour", 24))
        self.assertTrue(hourly.answer.startswith("Hour 8 has the highest average PM2.5"), hourly.answer)
        self.assertEqual(list(weekly.table["Weekday"])[:2], ["Monday", "Tuesday"])

    def test_an_hour_needs_times_of_day(self):
        frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=30, freq="D"),
                              "Value": np.arange(30, dtype=float)})
        question = "average Value by hour"

        self.assertIsNone(answer_question(question, frame, detect_roles(frame)))
        self.assertIn("without times of day", rephrase_hint(question, frame, detect_roles(frame)))


class ScopeEchoTests(unittest.TestCase):
    def test_each_answer_says_how_many_rows_it_used(self):
        frame, roles = dye_frame()

        breakdown = answer_question("average B by Condition", frame, roles)
        limited = answer_question("average B by Condition", frame, roles, rows_not_analyzed=300)

        self.assertIn("Based on all 40 rows.", breakdown.answer)
        self.assertIn("Based on the 40 sampled rows (of 340).", limited.answer)


class SubDayTrendTests(unittest.TestCase):
    def setUp(self):
        stamps = pd.date_range("2024-05-01 15:00:00", periods=3000, freq="s")
        self.frame = pd.DataFrame({"Time": stamps, "TVOC": np.arange(3000) % 50 + 10.0})
        self.roles = detect_roles(self.frame)

    def test_a_trend_by_the_minute_is_named_in_minutes(self):
        answer = answer_question("TVOC trend by minute", self.frame, self.roles)

        self.assertEqual(answer.plan.grain, "min")
        self.assertIn("TVOC per minute", answer.answer)
        self.assertIn("across 50 minutes", answer.answer)

    def test_hour_of_day_is_not_read_as_an_hourly_trend(self):
        plan = parse_question("average TVOC by hour of day", self.frame, self.roles)

        self.assertTrue(plan is None or plan.grain != "h")


def trial_frame() -> tuple[pd.DataFrame, ColumnRoles]:
    """Patients by result and condition: troponin separates the results, sugar does not."""
    rng = np.random.default_rng(11)
    result = np.where(np.arange(200) % 2 == 0, "positive", "negative")
    frame = pd.DataFrame({
        "Troponin": rng.normal(1.0, 0.3, 200) + np.where(result == "positive", 0.8, 0.0),
        "Sugar": rng.normal(140, 40, 200),
        "Age": rng.normal(55, 10, 200),
        "Condition": np.tile([1, 2], 100),
        "Result": result,
    })
    frame["Score"] = frame["Age"] * 0.5 + rng.normal(0, 2, 200)
    roles = ColumnRoles(
        date=None, measure="Sugar", dimension="Result", identifier=None,
        numeric=("Troponin", "Sugar", "Age", "Condition", "Score"), dimensions=("Result",),
    )
    return frame, roles


class GroupComparisonQuestionTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.roles = trial_frame()

    def ask(self, question):
        return answer_question(question, self.frame, self.roles)

    def test_each_phrasing_asks_for_a_comparison(self):
        for question in (
            "does Troponin differ by Result", "is Troponin different between results",
            "compare Troponin across Result", "does Troponin vary by result",
        ):
            with self.subTest(question=question):
                plan = parse_question(question, self.frame, self.roles)
                self.assertEqual(
                    (plan.intent, plan.measure, plan.dimension), ("compare", "Troponin", "Result")
                )

    def test_the_answer_carries_the_test_effect_size_and_a_table(self):
        answer = self.ask("does Troponin differ by Result")

        self.assertTrue(answer.answer.startswith("Troponin is higher in positive than in negative"))
        self.assertIn("95% CI", answer.answer)
        self.assertIn("Hedges' g", answer.calculation)
        self.assertEqual(answer.chart, "bar")
        self.assertEqual(list(answer.table.columns[:3]), ["Result", "Mean Troponin", "n"])

    def test_a_directed_question_gets_its_yes_or_no_first(self):
        higher = self.ask("is Troponin higher in positive patients than negative")
        lower = self.ask("is Troponin lower in positive than negative")

        self.assertEqual(higher.plan.filters[0].values, ("positive", "negative"))
        self.assertTrue(higher.answer.startswith("Yes. Troponin is higher in positive"))
        self.assertTrue(lower.answer.startswith("No, the other way round."))

    def test_no_clear_difference_is_said_plainly(self):
        answer = self.ask("does Sugar differ between positive and negative")

        self.assertIn("Sugar shows no clear difference between", answer.answer)
        directed = self.ask("is Sugar higher in positive than negative")
        self.assertTrue(directed.answer.startswith("Not clearly."))

    def test_numbered_groups_can_be_named(self):
        plan = parse_question("is Age higher in condition 2 than condition 1", self.frame, self.roles)

        self.assertEqual((plan.intent, plan.dimension), ("compare", "Condition"))
        self.assertEqual(plan.filters[0].values, ("2", "1"))
        self.assertEqual(plan.aggregation, "max")

    def test_comparing_a_summed_measure_is_left_to_the_breakdown(self):
        frame, roles = frame_with("Revenue")

        plan = parse_question("compare Revenue across Team", frame, roles)

        self.assertEqual(plan.intent, "breakdown")

    def test_a_correlation_of_two_named_columns_is_unchanged(self):
        plan = parse_question("is Age related to Score", self.frame, self.roles)

        self.assertEqual((plan.intent, plan.measure, plan.partner), ("correlation", "Age", "Score"))


class CorrelatesQuestionTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.roles = trial_frame()

    def test_each_phrasing_ranks_the_columns_against_one(self):
        for question in ("what correlates with Score", "what is most related to Score",
                         "which variables are most correlated with Score"):
            with self.subTest(question=question):
                plan = parse_question(question, self.frame, self.roles)
                self.assertEqual((plan.intent, plan.measure), ("correlates", "Score"))

    def test_the_answer_names_the_strongest_and_ranks_the_rest(self):
        answer = answer_question("what correlates with Score", self.frame, self.roles)

        self.assertTrue(answer.answer.startswith("Age relates most to Score of the 3 numeric columns tested"))
        self.assertIn("Pearson r =", answer.answer)
        self.assertEqual(answer.chart, "bar")
        self.assertEqual(answer.table.iloc[0]["Column"], "Age")
        self.assertEqual(
            list(answer.table.columns),
            ["Column", "Pearson r", "95% CI low", "95% CI high", "Spearman r", "n"],
        )
        self.assertNotIn("Condition", set(answer.table["Column"]))

    def test_nothing_related_is_a_plain_answer(self):
        answer = answer_question("what correlates with Sugar", self.frame, self.roles)

        self.assertTrue(answer.answer.startswith("Nothing relates to Sugar"))

    def test_the_columns_left_out_of_the_ranking_are_named(self):
        rng = np.random.default_rng(12)
        base = rng.normal(size=200)
        frame = pd.DataFrame({
            "Even1": base,
            "Even2": base + rng.normal(0, 0.05, 200),
            "Depth": base * 0.5 + rng.normal(size=200),
        })

        answer = answer_question("what correlates with Even1", frame, detect_roles(frame))

        self.assertIn("Left out of the ranking for Even1: Even2 (moving with it almost", answer.answer)
        self.assertEqual(answer.answer.count("Left out"), 1)
        self.assertIn("Depth is the only other numeric column tested against Even1", answer.answer)
        self.assertNotIn("picked from several columns", answer.answer)

    def test_a_measure_with_only_copies_is_not_called_unpaired(self):
        rng = np.random.default_rng(13)
        base = rng.normal(size=200)
        frame = pd.DataFrame({"Even1": base, "Even2": base + rng.normal(0, 0.05, 200)})

        answer = answer_question("what correlates with Even1", frame, detect_roles(frame))

        self.assertIn("nothing is left to rank", answer.answer)
        self.assertIn("Left out of the ranking for Even1: Even2", answer.answer)


def survey_frame() -> tuple[pd.DataFrame, ColumnRoles]:
    """A table without a date whose column names hold time and aggregation words."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "gender": np.tile([0, 1], 30),
        "average_weekly_activity": rng.normal(100, 10, 60).round(1),
        "Age of testing (day)": np.repeat([7, 28, 90], 20),
        "Strength": rng.normal(30, 5, 60).round(2),
    })
    return frame, detect_roles(frame)


def summary_frame() -> tuple[pd.DataFrame, ColumnRoles]:
    """An exported summary table whose columns are called mean, std, max and min."""
    rng = np.random.default_rng(4)
    mean = rng.normal(7, 1, 80)
    frame = pd.DataFrame({
        "Unnamed: 0": [f"probe_{index}" for index in range(80)],
        "mean": mean,
        "std": np.abs(rng.normal(1, 0.3, 80)),
        "max": mean + 2 + rng.normal(0, 0.1, 80),
        "min": mean - 2 + rng.normal(0, 0.1, 80),
    })
    return frame, detect_roles(frame)


class ColumnNameWordTests(unittest.TestCase):
    def test_time_words_inside_a_name_ask_for_no_timeline(self):
        frame, roles = survey_frame()
        for question in (
            "average average_weekly_activity by gender",
            "Top 5 gender by average_weekly_activity",
            "average Strength by Age of testing (day)",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(answer_question(question, frame, roles))

    def test_a_column_called_max_is_a_column_when_typed_as_its_name(self):
        dates = pd.date_range("2024-01-01", periods=120, freq="D")
        frame = pd.DataFrame({"Date": dates, "MAX": np.arange(120.0) % 30, "Precip": np.arange(120.0) % 7})
        roles = detect_roles(frame)

        for question in ("MAX trend", "Monthly MAX trend", "correlation between MAX and Precip"):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertIsNotNone(answer)
                self.assertEqual(answer.plan.measure, "MAX")

    def test_names_that_are_aggregation_words_work_as_names(self):
        frame, roles = summary_frame()

        correlation = answer_question("correlation between mean and std", frame, roles)
        condition = answer_question("how many rows have std above 2", frame, roles)

        self.assertEqual((correlation.plan.measure, correlation.plan.partner), ("mean", "std"))
        self.assertEqual(condition.plan.conditions[0].column, "std")
        self.assertEqual(answer_question("correlation between max and min", frame, roles).plan.partner, "min")

    def test_every_suggested_question_is_answered(self):
        frames = [survey_frame(), summary_frame()]
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        weather = pd.DataFrame({"Date": dates, "Station": ["a", "b", "c"] * 30, "MAX": np.arange(90.0) % 11})
        frames.append((weather, detect_roles(weather)))
        for path in samples().values():
            prepared = prepare_analysis(pd.read_csv(path), row_limit=50_000)
            frames.append((prepared.dataframe, prepared.detected_roles))
        for frame, roles in frames:
            for question in suggested_questions(frame, roles):
                with self.subTest(question=question):
                    answer = answer_question(question, frame, roles)
                    self.assertIsNotNone(answer, rephrase_hint(question, frame, roles))


class NamesBeforeKeywordsTests(unittest.TestCase):
    def test_a_research_word_inside_a_column_name_asks_for_no_relationship(self):
        frame = pd.DataFrame({
            "Pin Dia (mm)": [6, 7, 8] * 10,
            "Charpy Impact Strength (kg-m)": [5.0, 6.0, 7.5] * 9 + [5.5, 6.5, 7.0],
        })
        roles = detect_roles(frame)

        answer = answer_question("average Charpy Impact Strength (kg-m) by Pin Dia (mm)", frame, roles)

        self.assertEqual(answer.plan.intent, "breakdown")
        self.assertEqual(answer.plan.dimension, "Pin Dia (mm)")

    def test_a_column_called_max_after_a_superlative_or_in_is_the_column(self):
        dates = pd.date_range("2022-01-01", periods=730, freq="D")
        rng = np.random.default_rng(1)
        frame = pd.DataFrame({"Date": dates, "max": rng.gamma(4, 20, 730), "o3": rng.gamma(3, 10, 730)})
        roles = detect_roles(frame)

        month = answer_question("month of year with the highest max", frame, roles)
        outliers = answer_question("outliers in max", frame, roles)

        self.assertEqual((month.plan.measure, month.plan.time_part), ("max", "month"))
        self.assertEqual(outliers.plan.measure, "max")

    def test_columns_named_like_months_are_columns(self):
        frame = pd.DataFrame({
            "YEAR": pd.to_datetime([f"{year}-01-01" for year in range(1950, 1990)]),
            "SEP": np.linspace(20, 24, 40) + np.tile([0.3, -0.2], 20),
            "OCT": np.linspace(21, 25, 40) + np.tile([-0.1, 0.4], 20),
        })
        roles = detect_roles(frame)

        self.assertFalse(unsupported_phrasing("correlation between SEP and OCT", frame.columns))
        answer = answer_question("correlation between SEP and OCT", frame, roles)
        self.assertEqual((answer.plan.measure, answer.plan.partner), ("SEP", "OCT"))

    def test_a_column_named_by_a_number_takes_a_threshold(self):
        frame = pd.DataFrame({"time": [f"gene{n}" for n in range(30)], "120": np.linspace(-2, 2, 30),
                              "130": np.linspace(-1, 1, 30)})
        roles = detect_roles(frame)

        answer = answer_question("how many rows have 120 above 1", frame, roles)

        self.assertEqual(answer.plan.conditions, (Condition("120", ">", 1.0),))
        self.assertTrue(answer.answer.startswith("8 of the 30 rows"), answer.answer)

    def test_a_value_typed_in_full_wins_over_a_column_name_inside_it(self):
        frame = pd.DataFrame({
            "source_table": ["human_organs_tissues"] * 12 + ["human_blood_by_type"] * 12,
            "organ": ["liver", "lung", "skin"] * 8,
            "sample_size": np.arange(24.0),
        })
        roles = detect_roles(frame)

        answer = answer_question("average sample_size for source_table human_organs_tissues", frame, roles)

        self.assertEqual(answer.plan.filters[0].values, ("human_organs_tissues",))
        self.assertIn("is 5.50", answer.answer)


class NamedGroupTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.frame = pd.DataFrame({
            "condition": np.repeat([1, 2, 3], 40),
            "Probiotics": np.tile([0, 1], 60),
            "Weight_kg": rng.normal(400, 20, 120),
        })
        self.frame["angle"] = self.frame["condition"] * 0.1 + rng.normal(0, 0.01, 120)
        self.roles = detect_roles(self.frame)

    def ask(self, question):
        return answer_question(question, self.frame, self.roles)

    def test_between_two_levels_after_a_grouping_picks_those_groups(self):
        answer = answer_question("compare Weight_kg between Probiotics 0 and 1", self.frame, self.roles)

        self.assertEqual(answer.plan.dimension, "Probiotics")
        self.assertEqual(answer.plan.conditions, ())
        self.assertNotIn("No rows match", answer.answer)

    def test_levels_named_with_their_column_are_the_groups_not_a_filter(self):
        compared = self.ask("compare angle between condition 1 and condition 2")
        directed = self.ask("is angle higher in condition 3 than condition 1")

        self.assertEqual(compared.plan.filters[0].values, ("1", "2"))
        self.assertEqual(compared.plan.conditions, ())
        self.assertTrue(directed.answer.startswith("Yes."), directed.answer)

    def test_a_range_on_the_measure_is_still_a_range(self):
        answer = answer_question("how many rows have Weight_kg between 390 and 410", self.frame, self.roles)

        self.assertEqual(
            answer.plan.conditions, (Condition("Weight_kg", ">=", 390.0), Condition("Weight_kg", "<=", 410.0))
        )


class RowNounTests(unittest.TestCase):
    def test_the_files_noun_for_its_rows_names_the_rows_not_a_value(self):
        # members.csv: "members" are all its rows; "Member" is a role held by a few of them.
        frame = pd.DataFrame({
            "member_id": [f"M{index}" for index in range(40)],
            "expedition_role": ["Member"] * 4 + ["Climber"] * 36,
            "success": [True] * 10 + [False] * 30,
        })
        roles = replace(detect_roles(frame), row_noun="members")

        everyone = answer_question("what share of members had success", frame, roles)
        role = answer_question("what share of Member rows had success", frame, roles)

        self.assertNotIn("expedition_role", everyone.calculation)
        self.assertIn("10 of the 40 rows", everyone.answer)
        self.assertIn("expedition_role = Member", role.calculation)


class FrequencyTableTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        # Titanic's layout: one row per combination of class and outcome, with the people in it.
        return pd.DataFrame({
            "rownames": range(1, 9),
            "Class": ["1st", "2nd", "3rd", "Crew"] * 2,
            "Survived": ["No"] * 4 + ["Yes"] * 4,
            "Freq": [122, 167, 528, 673, 203, 118, 178, 212],
        })

    def test_shares_and_counts_weight_each_row_by_its_count(self):
        frame = self.frame()
        roles = detect_roles(frame)

        share = answer_question("what share survived", frame, roles)
        total = answer_question("how many people", frame, roles)

        self.assertEqual(nlq.frequency_column(frame), "Freq")
        self.assertIn("711 of the 2,201 cases (32.3%) have Survived = Yes", share.answer)
        self.assertEqual(total.value, 2201)
        self.assertNotIn("one of the people", total.answer)

    def test_shares_within_each_group_weight_each_row_by_its_count(self):
        frame = self.frame()

        shares = answer_question("share of Survived = Yes by Class", frame, detect_roles(frame))

        self.assertIn("1st 62.5% (203 of 325)", shares.answer)
        self.assertIn("Crew 24.0% (212 of 885)", shares.answer)
        self.assertIn("Each row counts Freq cases", shares.answer)
        self.assertIn("sum(Freq)", shares.calculation)
        self.assertIn("chi-squared", shares.calculation)
        self.assertNotIn("50.0%", shares.answer)

    def test_counts_by_group_sum_the_counts(self):
        frame = self.frame()

        counts = answer_question("how many people were in each Class", frame, detect_roles(frame))

        self.assertIn("Crew has the most cases of any Class, at 885", counts.answer)
        self.assertEqual(counts.calculation.split(" by ")[0], "sum(Freq)")
        self.assertEqual(dict(counts.values)["1st"], 325)

    def test_shares_of_a_small_table_of_counts_use_the_exact_test(self):
        frame = pd.DataFrame({
            "Group": ["A", "A", "B", "B"],
            "Outcome": ["No", "Yes", "No", "Yes"],
            "Freq": [6, 1, 2, 5],
        })

        shares = answer_question("share of Outcome = Yes by Group", frame, detect_roles(frame))

        self.assertIn("B 71.4% (5 of 7)", shares.answer)
        self.assertIn("A 14.3% (1 of 7)", shares.answer)
        self.assertIn("Fisher", shares.calculation)

    def test_a_count_column_with_repeated_combinations_is_not_a_table_of_counts(self):
        frame = pd.concat([self.frame(), self.frame()], ignore_index=True)
        frame["rownames"] = range(1, 17)
        self.assertIsNone(nlq.frequency_column(frame))
        self.assertIsNone(nlq.frequency_column(self.frame().iloc[:1]))

    def test_a_yes_no_column_without_counts_gives_the_share_of_rows(self):
        frame = self.frame().drop(columns="Freq")

        share = answer_question("what share survived", frame, detect_roles(frame))

        self.assertIn("4 of the 8 rows (50.0%) have Survived = Yes", share.answer)


class YesNoOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "grade": [1] * 40 + [2] * 40 + [3] * 40,
            "chemo": [0, 1] * 60,
            "eventdeath": [1] * 4 + [0] * 36 + [1] * 12 + [0] * 28 + [1] * 22 + [0] * 18,
            "age": np.linspace(30, 70, 120),
        })
        self.roles = detect_roles(self.frame)

    def ask(self, question):
        return answer_question(question, self.frame, self.roles)

    def test_share_rate_and_average_of_a_flag_by_a_group_are_rates_with_a_rate_test(self):
        for question in (
            "share of eventdeath by grade", "rate of eventdeath by grade", "eventdeath rate by grade",
            "average eventdeath by grade", "percentage of eventdeath by grade",
        ):
            with self.subTest(question=question):
                answer = self.ask(question)
                self.assertEqual(answer.plan.counted, (Condition("eventdeath", "=", 1.0),))
                self.assertIn("grade 3 55.0% (22 of 40)", answer.answer)
                self.assertIn("chi-squared test", answer.answer)

    def test_does_a_flag_differ_by_a_group_compares_rates(self):
        answer = self.ask("does eventdeath differ by chemo")

        self.assertEqual(answer.plan.intent, "compare")
        self.assertIn("percentage points", answer.answer)
        self.assertNotIn("Welch", answer.answer)

    def test_the_group_with_the_highest_share_is_named(self):
        answer = self.ask("which grade has the highest share of eventdeath 1")

        self.assertTrue(answer.answer.startswith("grade 3 has the highest share of rows with eventdeath = 1"))

    def test_a_small_two_by_two_table_uses_fishers_exact_test(self):
        frame = pd.DataFrame({"arm": ["a"] * 8 + ["b"] * 8, "cured": [1] * 7 + [0] + [1] * 2 + [0] * 6})

        answer = answer_question("share of cured by arm", frame, detect_roles(frame))

        self.assertIn("Fisher's exact test", answer.answer)

    def test_a_rate_with_nothing_to_count_is_refused_with_the_reason(self):
        question = "rate of age by grade"
        self.assertIsNone(self.ask(question))
        self.assertIn("A rate is the share of rows", rephrase_hint(question, self.frame, self.roles))

    def test_a_cue_word_narrows_the_rows_instead_of_naming_what_is_counted(self):
        answer = self.ask("share of eventdeath 1 by grade for chemo 1")

        self.assertEqual(answer.plan.counted, (Condition("eventdeath", "=", 1.0),))
        self.assertEqual(answer.plan.conditions, (Condition("chemo", "=", 1.0),))


class MorePhrasingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        dates = pd.date_range("1990-01-01", periods=30 * 12, freq="MS")
        self.frame = pd.DataFrame({
            "Date": dates,
            "DO": 8 - (dates.year >= 2000) * 2 + np.cos(dates.month / 12 * 2 * np.pi)
            + rng.normal(0, 0.3, len(dates)),
            "Temp": 15 + 10 * np.sin(dates.month / 12 * 2 * np.pi) + rng.normal(0, 1, len(dates)),
        })
        self.roles = detect_roles(self.frame)

    def ask(self, question):
        return answer_question(question, self.frame, self.roles)

    def test_ranges_of_years_are_conditions_on_the_date(self):
        for question, operator, year in (
            ("average DO since 2000", ">=", 2000.0), ("average DO after 1999", ">", 1999.0),
            ("average DO before 1995", "<", 1995.0), ("average DO until 1995", "<=", 1995.0),
            ("average DO where Date > 1999", ">", 1999.0),
        ):
            with self.subTest(question=question):
                answer = self.ask(question)
                self.assertEqual(answer.plan.conditions, (Condition("Date", operator, year, part="year"),))
        between = self.ask("average DO between 1990 and 1994")
        self.assertIn("Date between 1990 and 1994", between.answer)
        self.assertFalse(unsupported_phrasing("average DO between 1990 and 1994", self.frame.columns))

    def test_before_and_after_a_year_compares_the_two_spans(self):
        answer = self.ask("average DO before and after 2000")

        self.assertEqual((answer.plan.intent, answer.plan.split), ("compare", 2000.0))
        self.assertTrue(answer.answer.startswith("DO is higher in the years before 2000 than in 2000 and"))

    def test_a_grouping_beside_a_split_at_a_year_is_refused(self):
        frame = self.frame.assign(site=np.tile(["North", "South", "East"], len(self.frame) // 3))
        roles = detect_roles(frame)
        question = "does DO differ by site before and after 2000"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("two groupings, site and Date", rephrase_hint(question, frame, roles))

    def test_by_month_over_several_years_is_the_month_of_the_year(self):
        answer = self.ask("average DO by month")

        self.assertEqual(answer.plan.time_part, "month")
        self.assertIn('Read "by month" as the month of the year', answer.answer)
        self.assertEqual(self.ask("monthly DO trend").plan.intent, "trend")

    def test_by_season_groups_three_month_spans(self):
        answer = self.ask("average Temp by season")

        self.assertEqual(answer.plan.time_part, "season")
        self.assertEqual(sorted(answer.table["Season"]), ["Dec–Feb", "Jun–Aug", "Mar–May", "Sep–Nov"])

    def test_range_is_the_spread_from_lowest_to_highest(self):
        answer = self.ask("range of Temp")

        expected = self.frame["Temp"].max() - self.frame["Temp"].min()
        self.assertEqual(answer.plan.aggregation, "range")
        self.assertIn(format_number(expected, "Temp"), answer.answer)

    def test_a_comparison_after_accounting_for_a_reading_holds_it_fixed(self):
        rng = np.random.default_rng(5)
        time = rng.uniform(0, 50, 200)
        feed = np.tile([0.25, 0.5], 100)
        wear = 0.01 * time + (feed == 0.5) * 0.1 + rng.normal(0, 0.02, 200)
        frame = pd.DataFrame({"time": time, "feed": feed, "VB": wear})
        question = "does VB differ by feed after accounting for time"

        answer = answer_question(question, frame, detect_roles(frame))

        self.assertEqual(answer.plan.adjusted_for, ("time",))
        self.assertTrue(answer.answer.startswith("With time held fixed, VB is 0.0998 higher in feed 0.5"))
        self.assertIn("Without adjusting", answer.answer)

    def test_a_yes_no_outcome_is_not_compared_unadjusted_when_a_reading_should_be_held_fixed(self):
        rng = np.random.default_rng(6)
        frame = pd.DataFrame({
            "dose": rng.uniform(0, 10, 200), "site": np.tile(["North", "South"], 100),
            "relapse": rng.integers(0, 2, 200),
        })
        roles = detect_roles(frame)
        question = "does relapse differ by site after accounting for dose"

        self.assertIsNone(answer_question(question, frame, roles))
        hint = rephrase_hint(question, frame, roles)
        self.assertIn("relapse holds only 0 and 1", hint)
        self.assertIn("holds dose fixed only for a numeric measure", hint)


class FilterPhrasingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "admixture": [0.0, 0.0, 2.5, 3.0, 0.0, 4.0] * 5,
            "VMA": [0.0, 1.0, 0.0, 0.0, 2.0, 0.0] * 5,
            "strength": [40.0, 42.0, 55.0, 58.0, 41.0, 60.0] * 5,
            "sampleID": [f"RB{n % 3}.Ivc.{n}" if n % 2 else f"RB{n % 3}.Ctl.{n}" for n in range(30)],
            "fasting_ability": [1, 0, 1, 1, 0, 1] * 5,
        })
        self.roles = detect_roles(self.frame)

    def ask(self, question):
        return answer_question(question, self.frame, self.roles)

    def test_a_noun_for_the_rows_may_come_before_a_condition(self):
        answer = self.ask("average strength for mixes with admixture above 0")

        self.assertEqual(answer.plan.conditions, (Condition("admixture", ">", 0.0),))

    def test_a_value_after_a_column_with_or_without_of_is_equality(self):
        for question in ("how many rows have VMA of 0", "how many rows have VMA 0"):
            with self.subTest(question=question):
                self.assertEqual(self.ask(question).plan.conditions, (Condition("VMA", "=", 0.0),))

    def test_part_of_a_text_filters_the_rows_that_contain_it(self):
        answer = self.ask("average strength for sampleID containing Ivc")

        self.assertEqual(answer.plan.conditions, (Condition("sampleID", "contains", text="Ivc"),))
        self.assertIn('sampleID containing "Ivc"', answer.answer)

    def test_a_word_names_the_column_named_by_a_longer_form_of_it(self):
        answer = self.ask("what percentage of patients can fast")

        self.assertIn('Read "fast" as fasting_ability', answer.answer)
        self.assertTrue(answer.answer.startswith("20 of the 30 rows"), answer.answer)


class NameFormTests(unittest.TestCase):
    def test_a_prefix_most_names_share_and_latex_are_optional(self):
        rng = np.random.default_rng(2)
        frame = pd.DataFrame({
            "PROPERTY: YS (MPa)": rng.normal(1000, 200, 60),
            "PROPERTY: Test temperature ($^\\circ$C)": rng.choice([25.0, 600.0, 800.0], 60),
            "PROPERTY: Type of test": rng.choice(["C", "T"], 60),
            "PROPERTY: BCC/FCC/other": rng.choice(["BCC", "FCC", "other"], 60),
        })
        roles = detect_roles(frame)

        grouped = answer_question("average YS by Type of test", frame, roles)
        related = answer_question("correlation between YS (MPa) and Test temperature", frame, roles)
        slashed = answer_question("average YS (MPa) by BCC/FCC/other", frame, roles)

        self.assertEqual(grouped.plan.measure, "PROPERTY: YS (MPa)")
        self.assertEqual(grouped.plan.dimension, "PROPERTY: Type of test")
        self.assertEqual(related.plan.partner, "PROPERTY: Test temperature ($^\\circ$C)")
        self.assertEqual(slashed.plan.dimension, "PROPERTY: BCC/FCC/other")

    def test_column_words_are_never_read_as_the_row_label(self):
        frame = pd.DataFrame({"Unnamed: 0": range(40), "VB": np.linspace(0, 1, 40), "time": np.arange(40.0),
                              "load": np.tile([1.0, 3.0, 2.0, 5.0], 10)})
        roles = detect_roles(frame)

        answer = answer_question("which columns relate most to VB", frame, roles)

        self.assertEqual((answer.plan.intent, answer.plan.measure), ("correlates", "VB"))


class GroupRankingTests(unittest.TestCase):
    def test_single_row_groups_rank_the_rows_and_placeholders_are_never_named(self):
        frame = pd.DataFrame({
            "Name": ["-"] * 30 + [f"compound {n}" for n in range(10)],
            "Intensity": [900.0] * 30 + [float(100 * n) for n in range(10)],
        })
        roles = detect_roles(frame)

        top = answer_question("top 5 Name by Intensity", frame, roles)

        self.assertEqual(top.table["Name"].tolist(), [f"compound {n}" for n in (9, 8, 7, 6, 5)])
        self.assertTrue(top.answer.startswith("compound 9 has the highest Intensity"), top.answer)
        self.assertIn('30 rows labelled "-" are left out', top.answer)

    def test_a_breakdown_names_no_placeholder_group_as_the_highest(self):
        frame = pd.DataFrame({
            "method": [None] * 10 + ["cast"] * 10 + ["forged"] * 10,
            "HV": [700.0] * 10 + [400.0] * 10 + [500.0] * 10,
        })
        answer = answer_question("average HV by method", frame, detect_roles(frame))

        self.assertTrue(answer.answer.startswith("forged has the highest average HV"), answer.answer)

    def test_per_sample_in_a_matrix_totals_each_sample_column(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.poisson(20, size=(200, 12)), columns=[f"S{n}" for n in range(12)])
        frame.insert(0, "OTU", [f"otu{n}" for n in range(200)])
        prepared = prepare_analysis(frame, row_limit=10_000)

        answer = answer_question("total reads per sample", prepared.dataframe, prepared.detected_roles)

        self.assertEqual(answer.plan.intent, "columns")
        self.assertNotIn("Row mean", " ".join(answer.table["Column"]))
        self.assertIn('Read "reads" as the values in the 12 columns', answer.answer)


class BriefAgreementTests(unittest.TestCase):
    def test_tests_run_on_the_distinct_rows_when_most_rows_are_copies(self):
        rng = np.random.default_rng(6)
        base = pd.DataFrame({"Gender": np.tile([0, 1], 50), "MCV": np.linspace(70.05, 99.95, 100),
                             "Result": rng.integers(0, 2, 100)})
        frame = pd.concat([base, base.iloc[:40], base.iloc[:40]], ignore_index=True)
        roles = detect_roles(frame)

        answer = answer_question("share of Result 1 by Gender", frame, roles)
        count = answer_question("how many rows have Result 1", frame, roles)

        self.assertIn("this uses the 100 distinct rows, leaving out 80 exact repeats", answer.answer)
        self.assertEqual(int(answer.table["Rows"].sum()), 100)
        share = frame["Result"].mean() * 100
        self.assertTrue(count.answer.endswith(f"of the 180 rows ({share:.1f}%) have Result = 1."))

    def test_correlations_leave_out_the_cells_the_data_checks_set_aside(self):
        frame = pd.DataFrame({"PM10": np.linspace(50, 100, 40), "WS": np.linspace(1, 2, 40)})
        frame.loc[30:, "WS"] = 250.0
        given = QualityIssue(
            "swapped", ("WS",), 10, 25.0, "WS holds wind directions.", "WS swapped on 10 rows",
            "a test rule", cells={"WS": pd.Index(range(30, 40))},
        )

        question = "correlation between PM10 and WS"
        answer = answer_question(question, frame, detect_roles(frame), checks=(given,))

        self.assertIn("n = 30", answer.answer)
        self.assertIn("10 values the data checks flag are left out (WS swapped on 10 rows)", answer.answer)
        self.assertEqual(answer.scope.rows_used, 30)
        self.assertEqual(answer.scope.left_out, (("values the data checks flag", 10),))
        self.assertEqual(dict(answer.figures)["n"], 30)

    def test_a_value_named_after_by_is_refused_rather_than_read_as_a_filter(self):
        rng = np.random.default_rng(10)
        frame = pd.DataFrame({
            "Soil_Type": rng.choice(["Gravel", "Silt", "Sand"], 90), "Activity": rng.normal(3, 1, 90),
        })
        roles = detect_roles(frame)

        grouped = "average Activity by Soil_Type_Gravel"
        for question in (grouped, "which Soil_Type_Gravel has the highest Activity"):
            with self.subTest(question=question):
                self.assertIsNone(answer_question(question, frame, roles))
                hint = rephrase_hint(question, frame, roles)
                self.assertIn("Gravel is a value of Soil_Type, so the answer would keep only those", hint)
        self.assertIsNotNone(answer_question("average Activity for Soil_Type Gravel", frame, roles))

    def test_a_count_of_columns_in_a_name_is_not_a_unit_to_set_a_condition_on(self):
        values = np.random.default_rng(9).normal(50, 10, 30)
        frame = pd.DataFrame({"Name": [f"m{n}" for n in range(30)], "Row mean (3 columns)": values})
        roles = detect_roles(frame)

        self.assertIsNone(answer_question("which Name has the highest Row mean (4 columns)", frame, roles))
        self.assertIsNotNone(answer_question("which Name has the highest Row mean (3 columns)", frame, roles))

    def test_one_cell_left_out_is_said_in_the_singular(self):
        frame = pd.DataFrame({"hemo": np.linspace(10, 17, 40), "pcv": np.linspace(30, 52, 40)})
        frame.loc[0, "pcv"] = 9.0
        given = QualityIssue(
            "impossible", ("pcv",), 1, 2.5, "pcv below 10.", "pcv: 1 value below 10 (9)",
            "a test rule", cells={"pcv": pd.Index([0])},
        )

        question = "correlation between hemo and pcv"
        answer = answer_question(question, frame, detect_roles(frame), checks=(given,))

        self.assertIn("1 value the data checks flag is left out (pcv: 1 value below 10 (9))", answer.answer)

    def test_a_count_on_flagged_values_and_a_code_set_to_missing_are_named(self):
        frame = pd.DataFrame({"Na": [140.0] * 30 + [250.0] * 5, "pm2_5": [10.0] * 35})
        flagged = QualityIssue(
            "impossible", ("Na",), 5, 14.0, "Sodium above 200.", "Na: 5 values above 200", "a test rule",
            cells={"Na": pd.Index(range(30, 35))},
        )
        blanked = QualityIssue(
            "sentinel", ("pm2_5",), 4, 10.0, "-1 set to missing.", "pm2_5: -1 on 4 rows", "a test rule",
            group=-1.0,
        )
        roles = detect_roles(frame)

        checks = (flagged, blanked)
        high = answer_question("how many rows have Na above 150", frame, roles, checks=checks)
        negative = answer_question("how many rows have pm2_5 below 0", frame, roles, checks=checks)

        self.assertIn("Caution: 5 of these rows have Na values the data checks flag", high.answer)
        self.assertIn("-1 in pm2_5 was read as a missing-value code", negative.answer)

    def test_an_answer_says_it_rests_on_the_rows_with_the_measure(self):
        frame = pd.DataFrame({"Side": ["L", "R"] * 60, "Crabs": [1.0, 2.0] * 55 + [np.nan] * 10})

        answer = answer_question("total Crabs by Side", frame, detect_roles(frame))

        self.assertIn("Based on the 110 rows with Crabs recorded, of 120 rows.", answer.answer)

    def test_an_unfiltered_aggregate_says_its_rows_once(self):
        frame = pd.DataFrame({"Side": ["L", "R"] * 60, "Crabs": [1.0, 2.0] * 55 + [np.nan] * 10})

        answer = answer_question("average Crabs", frame, detect_roles(frame))

        self.assertIn("calculated from 110 rows", answer.answer)
        self.assertNotIn("Based on", answer.answer)


class CorrelatesWordingTests(unittest.TestCase):
    def test_nothing_to_correlate_gives_the_real_reason(self):
        frame = pd.DataFrame(
            {"Crabs": np.arange(40.0) % 7, "Nudibranchs": [0, 1] * 20, "Cucumbers": [0.0] * 40}
        )

        answer = answer_question("what correlates with Crabs", frame, detect_roles(frame))

        self.assertIn("Nudibranchs holds only 0 and 1", answer.answer)
        self.assertIn("Cucumbers never varies", answer.answer)
        self.assertNotIn("shares 12 or more rows", answer.answer)

    def test_one_outlier_in_ten_sits_outside(self):
        frame = pd.DataFrame(
            {"DDO": [40.0, 42.0, 45.0, 47.0, 50.0, 41.0, 44.0, 46.0, 43.0, 48.0, 49.0, 45.5, 95.0]}
        )

        answer = answer_question("outliers in DDO", frame, detect_roles(frame))

        self.assertIn("1 of the 13 DDO values sits outside", answer.answer)


class WideScanAndGroupingRuleTests(unittest.TestCase):
    def test_a_screen_of_many_columns_names_its_false_discovery_adjustment(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(rng.normal(2.5, 0.5, size=(120, 400)), columns=[f"Gene_{n}" for n in range(400)])
        frame["Diagnosis"] = np.repeat(["AML", "ALL", "Healthy"], 40)
        roles = detect_roles(frame)

        answer = answer_question("which genes differ between AML and ALL", frame, roles)

        self.assertEqual(answer.plan.filters[0].values, ("AML", "ALL"))
        self.assertIn("between AML and ALL", answer.answer)
        self.assertIn("false-discovery-rate adjusted", answer.answer)
        self.assertNotIn("Holm", answer.answer + answer.calculation)

    def test_a_grouping_cut_from_the_measure_gets_its_rule_not_a_test(self):
        values = np.linspace(5, 300, 200)
        cuts = [values <= 50, values <= 100, values <= 200]
        band = np.select(cuts, ["good", "moderate", "unhealthy"], "severe")
        frame = pd.DataFrame({"max": values, "categori": band, "station": np.tile(["a", "b"], 100)})
        roles = detect_roles(frame)

        breakdown = answer_question("average max by categori", frame, roles)
        compared = answer_question("does max differ by categori", frame, roles)

        self.assertIn("categori is set by max", breakdown.answer)
        self.assertNotIn("Welch", breakdown.answer)
        self.assertIn("no test is run", compared.answer)


class StackedTableTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        food = pd.DataFrame({"source_table": "food", "food_category": rng.choice(["fish", "salt"], 40),
                             "particles": rng.poisson(30, 40).astype(float)})
        organs = pd.DataFrame({"source_table": "organs", "organ_system": rng.choice(["skin", "liver"], 20),
                               "sample_size": rng.integers(10, 200, 20).astype(float)})
        blood = pd.DataFrame({"source_table": "blood", "p_value": rng.uniform(0, 1, 8)})
        self.frame = pd.concat([food, organs, blood], ignore_index=True)
        roles = detect_roles(self.frame)
        self.whole = WholeTable(self.frame, roles, "source_table", "food")
        self.focused = self.frame[self.frame["source_table"] == "food"].reset_index(drop=True)
        self.roles = replace(roles, dimension="food_category")

    def ask(self, question):
        return answer_question(question, self.focused, self.roles, whole=self.whole)

    def test_columns_recorded_in_one_other_part_are_answered_there(self):
        answer = self.ask("average sample_size by organ_system")

        self.assertEqual(answer.plan.dimension, "organ_system")
        self.assertIn("only recorded in the rows with source_table = organs (20 rows)", answer.answer)

    def test_a_column_in_another_part_gives_the_real_reason(self):
        question = "correlation between particles and p_value"

        self.assertIsNone(self.ask(question))
        hint = rephrase_hint(question, self.focused, self.roles, whole=self.whole)
        self.assertIn("p_value has no values in the source_table = food part the brief follows", hint)
        self.assertNotIn("not numeric", hint)

    def test_rows_per_part_are_counted_and_ranked(self):
        answer = self.ask("how many rows per source_table")

        self.assertTrue(answer.answer.startswith("food has the most rows of any source_table, at 40"))
        self.assertNotIn("own scale", answer.answer)

    def test_a_long_table_inside_one_part_answers_from_the_column_marking_the_parts(self):
        # The brief follows a unit beside the food readings; the other tables have no unit at all.
        units = np.where(self.frame["source_table"] == "food", "items", None)
        frame = self.frame.assign(**{"particles unit": units})
        frame.loc[:9, "particles unit"] = "mg"
        roles = detect_roles(frame)
        whole = WholeTable(frame, roles, "particles unit", "items")
        focused = frame[frame["particles unit"] == "items"].reset_index(drop=True)
        roles = replace(roles, dimension="food_category")

        answer = answer_question("average sample_size by organ_system", focused, roles, whole=whole)

        self.assertIn("only recorded in the rows with source_table = organs (20 rows)", answer.answer)
        self.assertNotIn("None", answer.answer)

    def test_a_unit_column_named_for_one_reading_leaves_the_others_whole(self):
        rng = np.random.default_rng(11)
        organs = pd.DataFrame({
            "organ_system": rng.choice(["skin", "liver"], 20),
            "sample_size": rng.integers(10, 200, 20).astype(float),
            "abundance": rng.uniform(1, 90, 20), "abundance unit": np.tile(["particles", "particles/g"], 10),
        })
        others = self.frame[self.frame["source_table"] != "organs"]
        frame = pd.concat([others, organs.assign(source_table="organs")], ignore_index=True)
        roles = detect_roles(frame)
        whole = WholeTable(frame, roles, "source_table", "food")
        focused = frame[frame["source_table"] == "food"].reset_index(drop=True)
        roles = replace(roles, dimension="food_category")

        answer = answer_question("average sample_size by organ_system", focused, roles, whole=whole)

        self.assertIn("Based on all 20 rows", answer.answer)
        self.assertNotIn("abundance unit", answer.answer)


class PerUnitTests(unittest.TestCase):
    def test_a_measure_per_unit_by_a_grouping_compares_the_units(self):
        rows = []
        for reef, status in (("North", "NTZ"), ("South", "OA"), ("East", "OA"), ("West", "NTZ")):
            for transect in range(1, 7):
                for fish in range(5):
                    rows.append({"reef": reef, "protection_status": status, "transect": transect,
                                 "quantity": float(fish + (5 if status == "NTZ" else 1))})
        frame = pd.DataFrame(rows)

        question = "average quantity per transect by protection_status"
        answer = answer_question(question, frame, detect_roles(frame))

        self.assertEqual(answer.plan.per, ("transect",))
        self.assertIn("reef and transect together, 24 in all", answer.answer)
        self.assertEqual(answer.table["n"].tolist(), [12, 12])


class RefusalHintTests(unittest.TestCase):
    def test_only_the_unknown_words_are_quoted_as_typed(self):
        frame, roles = summary_frame()

        hint = rephrase_hint("compare mean with Zeta values", frame, roles)

        self.assertIn('"Zeta"', hint)
        self.assertNotIn("compare", hint)

    def test_a_word_naming_many_columns_lists_five(self):
        frame = pd.DataFrame({f"Gene_{index}": np.arange(30.0) + index for index in range(40)})
        frame["Label"] = ["x", "y", "z"] * 10

        hint = rephrase_hint("average gene by Label", frame, detect_roles(frame))

        self.assertIn("or 35 others", hint)

    def test_a_word_the_plan_cannot_use_is_refused(self):
        frame, roles = dye_frame()

        self.assertIsNone(answer_question("average B within", frame, roles))

    def test_a_grouping_by_rows_no_column_names_is_refused(self):
        frame, roles = dye_frame()

        question = "average B per patient"

        self.assertIsNone(answer_question(question, frame, roles))
        self.assertIn("needs a column that names each one", rephrase_hint(question, frame, roles))


class ResearchQuestionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        dose = rng.uniform(0, 10, 200)
        treated = np.tile([0, 1], 100)
        self.frame = pd.DataFrame({
            "Dose": dose,
            "Treated": treated,
            "Growth": 3 * dose + 8 * treated + rng.normal(0, 2, 200),
            "Weight": rng.normal(50, 5, 200),
        })
        self.roles = detect_roles(self.frame)

    def plan(self, question: str) -> QueryPlan:
        plan = parse_question(question, self.frame, self.roles)
        self.assertIsNotNone(plan, question)
        return plan

    def test_what_predicts_ranks_the_columns(self):
        questions = ("what predicts Growth", "what drives Growth", "which factors affect Growth the most")
        for question in questions:
            with self.subTest(question=question):
                plan = self.plan(question)
                self.assertEqual((plan.intent, plan.measure), ("correlates", "Growth"))

    def test_the_effect_of_a_quantity_is_a_correlation(self):
        plan = self.plan("what is the effect of Dose on Growth")

        self.assertEqual((plan.intent, plan.measure, plan.partner), ("correlation", "Growth", "Dose"))

    def test_the_effect_of_a_flag_is_a_comparison(self):
        plan = self.plan("does Treated affect Growth")

        self.assertEqual((plan.intent, plan.measure, plan.dimension), ("compare", "Growth", "Treated"))

    def test_higher_with_a_flag_compares_the_rows_with_it_to_those_without(self):
        answer = answer_question("is Growth higher with Treated", self.frame, self.roles)

        self.assertTrue(answer.answer.startswith("Yes."))
        self.assertEqual(answer.plan.filters[0].values, ("1", "0"))

    def test_a_trend_along_another_column_is_their_correlation(self):
        plan = self.plan("is there a trend in Growth with Dose")

        self.assertEqual((plan.intent, plan.partner), ("correlation", "Dose"))

    def test_is_the_difference_significant_compares_the_groups(self):
        plan = self.plan("is the difference in Growth between Treated groups significant")

        self.assertEqual((plan.intent, plan.measure, plan.dimension), ("compare", "Growth", "Treated"))

    def test_high_and_low_split_a_column_at_its_median(self):
        answer = answer_question("does Growth differ between high and low Dose", self.frame, self.roles)

        self.assertEqual((answer.plan.dimension, answer.plan.bins), ("Dose", 2))
        self.assertIn("split at the median", answer.answer)


class CodeCorrelationTests(unittest.TestCase):
    def test_a_correlation_with_a_code_compares_its_groups(self):
        frame, roles = trial_frame()
        roles = replace(roles, factors=("Condition",))

        answer = answer_question("correlation between Score and Condition", frame, roles)

        self.assertEqual((answer.plan.intent, answer.plan.dimension), ("compare", "Condition"))
        self.assertIn("compared across its groups rather than correlated", answer.answer)

    def test_two_codes_are_cross_tabulated(self):
        frame = pd.DataFrame({"Object": np.tile([0, 1, 2], 40), "Phenotype": np.repeat([1, 2], 60)})
        roles = replace(detect_roles(frame), factors=("Object", "Phenotype"))

        answer = answer_question("correlation between Object and Phenotype", frame, roles)

        self.assertEqual(answer.plan.intent, "crosstab")
        self.assertIn("chi-square", answer.answer)

    def test_a_code_with_a_unit_still_correlates(self):
        frame = pd.DataFrame({"Location_m": np.repeat([0, 5, 10, 15], 10), "Crabs": np.arange(40.0) % 9})
        roles = replace(detect_roles(frame), factors=("Location_m",))

        plan = parse_question("correlation between Crabs and Location_m", frame, roles)

        self.assertEqual(plan.intent, "correlation")


class SummaryQuestionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "Site": ["a", "b", "c", "d"] * 10,
            "Value": [float(value) for value in range(1, 40)] + [400.0],
            "Kind": ["x"] * 30 + ["y"] * 10,
        })
        self.roles = detect_roles(self.frame)

    def ask(self, question: str):
        answer = answer_question(question, self.frame, self.roles)
        self.assertIsNotNone(answer, question)
        return answer

    def test_spreads_and_percentiles(self):
        values = self.frame["Value"]
        self.assertIn(f"{values.std():.2f}", self.ask("standard deviation of Value").answer)
        self.assertEqual(self.ask("variance of Value").plan.aggregation, "var")
        self.assertEqual(self.ask("IQR of Value").plan.aggregation, "iqr")
        percentile = self.ask("90th percentile of Value")
        self.assertEqual((percentile.plan.aggregation, percentile.plan.quantile), ("quantile", 0.9))
        self.assertTrue(percentile.answer.startswith("90th percentile of Value"))

    def test_a_distribution_gives_the_summary_statistics(self):
        answer = self.ask("distribution of Value")

        self.assertEqual(answer.plan.intent, "describe")
        statistics = ["n", "Mean", "SD", "Median", "Q1", "Q3", "Min", "Max", "Skew"]
        self.assertEqual(answer.table["Statistic"].tolist(), statistics)
        self.assertIn("skewed to the right", answer.answer)
        self.assertEqual(self.ask("describe Value by Kind").table["Kind"].tolist(), ["y", "x"])

    def test_most_common_names_the_largest_group(self):
        answer = self.ask("which Kind is most common")

        self.assertEqual((answer.plan.intent, answer.plan.aggregation), ("rank", "count"))
        self.assertTrue(answer.answer.startswith("x has the most rows"))

    def test_outliers_list_the_values_outside_the_fences(self):
        answer = self.ask("outliers in Value")

        self.assertEqual(answer.plan.intent, "outliers")
        self.assertIn("400", answer.answer)
        self.assertEqual(answer.table["Value"].tolist(), [400.0])
        self.assertIn("data checks", self.ask("are there any unusual values").answer)

    def test_duplicates_and_distinct_counts(self):
        frame = pd.concat([self.frame, self.frame.head(3)], ignore_index=True)
        roles = detect_roles(frame)

        self.assertIn("3 of the 43 rows", answer_question("how many duplicate rows", frame, roles).answer)
        distinct = answer_question("how many distinct Value", frame, roles)
        self.assertEqual((distinct.plan.count_column, distinct.answer[:14]), ("Value", "There are 40 d"))
        per_group = answer_question("how many unique Kind per Site", frame, roles)
        self.assertEqual((per_group.plan.dimension, per_group.plan.count_column), ("Site", "Kind"))

    def test_equality_with_a_number_and_zeros_are_conditions(self):
        frame = pd.DataFrame({"Reads": [0, 0, 3, 5, 0, 8] * 5, "Site": ["a", "b"] * 15})
        roles = detect_roles(frame)

        for question in ("how many Reads values are 0", "how many zeros in Reads"):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertTrue(answer.answer.startswith("15 of the 30 rows"))


class CountingNounTests(unittest.TestCase):
    def test_a_noun_counts_the_column_named_for_it(self):
        frame = pd.DataFrame({
            "DRUG_ID": [str(1000 + index % 7) for index in range(42)],
            "DRUG_NAME": [f"drug {index % 6}" for index in range(42)],
            "Plot Code": [f"P{index % 5}" for index in range(42)],
            "Plot Name": [f"plot {index % 4}" for index in range(42)],
            "Plot Type": ["field", "forest"] * 21,
            "Current Crop": ["maize", "rice", "soy"] * 14,
            "Crop Rotation System": ["a", "b"] * 21,
            "Yield": np.arange(42.0),
        })
        roles = detect_roles(frame)

        self.assertEqual(answer_question("how many drugs", frame, roles).plan.count_column, "DRUG_NAME")
        plots = answer_question("how many plots", frame, roles)
        self.assertEqual(plots.plan.count_column, "Plot Code")
        self.assertIn("Plot Name has 4 distinct values", plots.answer)
        self.assertEqual(answer_question("how many crops", frame, roles).plan.count_column, "Current Crop")
        self.assertIsNone(answer_question("how many samples", frame, roles).plan.count_column)


class ColumnSetTests(unittest.TestCase):
    def test_which_column_ranks_the_numeric_readings(self):
        frame = pd.DataFrame({"A": [1.0, 2, 3], "B": [10.0, 20, 30], "C": [5.0, 6, 7]})

        answer = answer_question("which column has the highest average", frame, detect_roles(frame))

        self.assertEqual(answer.plan.intent, "columns")
        self.assertTrue(answer.answer.startswith("B has the highest average of the 3 numeric columns, at 20"))

    def test_a_word_naming_many_columns_ranks_them(self):
        rng = np.random.default_rng(6)
        frame = pd.DataFrame({f"Gene_{index}": rng.normal(index, 1, 60) for index in range(1, 6)})
        frame["Diagnosis"] = ["AML", "ALL", "Healthy"] * 20
        frame["Gene_3"] += np.where(frame["Diagnosis"] == "AML", 3, 0)
        roles = detect_roles(frame)

        highest = answer_question("which gene has the highest average", frame, roles)
        differing = answer_question("which genes differ between AML and ALL", frame, roles)

        self.assertTrue(highest.answer.startswith("Gene_5 has the highest average"))
        self.assertIn("Gene_3", differing.answer)
        self.assertIn("Holm-adjusted", differing.answer)


class DatePartTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2018-06-01", "2022-12-31", freq="D")
        warming = (dates.year - 2018) * 0.5
        season = -10 * np.cos((dates.dayofyear - 15) / 365.25 * 2 * np.pi)
        self.frame = pd.DataFrame({"Date": dates, "Temp": 10 + season + warming})
        self.roles = detect_roles(self.frame)

    def test_month_and_quarter_of_the_year_pool_the_years(self):
        month = answer_question("which month of the year has the highest Temp", self.frame, self.roles)
        quarter = answer_question("average Temp by quarter of the year", self.frame, self.roles)

        self.assertEqual(month.plan.time_part, "month")
        self.assertTrue(month.answer.startswith("July has the highest average"))
        self.assertIn("of any month of the year", month.answer)
        self.assertEqual(quarter.table.iloc[0][quarter.table.columns[0]], "Q1")

    def test_warmest_year_ranks_the_years_and_says_how_it_read_the_word(self):
        answer = answer_question("which year was the warmest", self.frame, self.roles)

        self.assertTrue(answer.answer.startswith("2022 had the highest"))
        self.assertIn('Read "warmest" as the highest Temp', answer.answer)

    def test_a_trend_gives_its_slope_peak_and_leaves_out_a_partial_first_year(self):
        answer = answer_question("average Temp by year", self.frame, self.roles)

        self.assertIn("from 2019 to 2022", answer.answer)
        self.assertIn("The Theil-Sen trend is +0.5 per year", answer.answer)
        self.assertIn("It peaked at", answer.answer)
        self.assertIn("2018 is left out because the data starts part-way through it", answer.calculation)


class TrendSlopeTests(unittest.TestCase):
    def test_a_slope_over_sparse_months_is_per_month(self):
        months = ["2003-06-15", "2003-09-15", "2003-11-15", "2004-04-15", "2004-05-15",
                  "2004-08-15", "2004-10-15", "2005-04-15", "2005-07-15", "2005-10-15"]
        dates = pd.to_datetime(months)
        elapsed = (dates.year - 2003) * 12 + dates.month - 6
        frame = pd.DataFrame({"Date": dates, "Depth": 1.0 + 0.1 * elapsed})
        roles = replace(detect_roles(frame), date="Date", measure="Depth")

        answer = answer_question("Depth trend by month", frame, roles)

        self.assertIn("The Theil-Sen trend is +0.1 per month", answer.answer)

    def test_a_slope_whose_interval_spans_zero_is_no_clear_trend(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=5, freq="MS"),
            "AQI": [60.0, 70.0, 58.0, 65.0, 68.0],
        })
        roles = replace(detect_roles(frame), date="Date", measure="AQI")

        answer = answer_question("AQI trend", frame, roles)

        self.assertIn("95% CI", answer.answer)
        self.assertIn("no clear trend", answer.answer)

    def test_a_reading_that_starts_late_counts_only_its_own_years(self):
        days = pd.date_range("2010-01-01", "2024-12-31", freq="7D")
        late = np.where(days.year >= 2021, 80.0 + days.year - 2021, np.nan)
        frame = pd.DataFrame({"Date": days, "o3": 50.0, "pm25": late})
        roles = replace(detect_roles(frame), date="Date", measure="o3")

        answer = answer_question("pm25 per year", frame, roles)

        self.assertIn("across 4 years", answer.answer)
        self.assertIn("pm25 is recorded only from 2021", answer.answer)

    def test_a_seasonal_series_compares_whole_years(self):
        months = pd.date_range("2018-01-01", periods=60, freq="MS")
        frame = pd.DataFrame({"Date": months, "TMK": 10 + 8 * np.cos(2 * np.pi * (months.month - 7) / 12)})
        roles = replace(detect_roles(frame), date="Date", measure="TMK")

        answer = answer_question("TMK trend by quarter", frame, roles)

        self.assertIn("TMK, averaged over each complete year, moved from 10.00 to 10.00", answer.answer)
        self.assertIn("across 5 complete years", answer.answer)

    def test_a_series_that_never_moves_is_stated_once(self):
        frame = pd.DataFrame({"Year": [2005 + index for index in range(11)], "Index": [100.0] * 11})
        frame["Year"] = pd.to_datetime(frame["Year"].astype(str))
        roles = replace(detect_roles(frame), date="Year", measure="Index")

        answer = answer_question("Index trend by year", frame, roles)

        self.assertIn("in every year", answer.answer)
        self.assertNotIn("peaked", answer.answer)
        self.assertNotIn("Theil-Sen", answer.answer)


class GroupedAnswerTests(unittest.TestCase):
    def test_a_share_within_groups_is_the_rate_in_each_group(self):
        frame = pd.DataFrame({"Gender": [0] * 10 + [1] * 10, "Result": [1] * 3 + [0] * 7 + [1] * 6 + [0] * 4})
        roles = detect_roles(frame)

        answer = answer_question("share of Result 1 by Gender", frame, roles)

        self.assertEqual(answer.plan.aggregation, "share")
        self.assertEqual(answer.table["Share %"].tolist(), [60.0, 30.0])
        self.assertIn("Gender 1 60.0% (6 of 10)", answer.answer)

    def test_a_ranking_shows_n_leaves_out_small_groups_and_tests_the_difference(self):
        frame = pd.DataFrame({
            "Breed": ["a"] * 20 + ["b"] * 20 + ["c"] * 2,
            "Index": list(np.linspace(2.70, 2.80, 20)) + list(np.linspace(2.69, 2.79, 20)) + [9.0, 9.0],
        })
        roles = detect_roles(frame)

        answer = answer_question("which Breed has the highest Index", frame, roles)

        self.assertNotIn("c has", answer.answer)
        self.assertIn("(n = 20)", answer.answer)
        self.assertIn("left out of the ranking", answer.answer)
        self.assertIn("do not clearly differ", answer.answer)
        self.assertIn("n", answer.table.columns)

    def test_a_mean_carried_by_a_few_values_says_so(self):
        frame = pd.DataFrame({
            "Level": ["low"] * 20 + ["high"] * 20, "Gas": [1.0] * 20 + [1.0] * 17 + [500.0] * 3,
        })

        answer = answer_question("average Gas by Level", frame, detect_roles(frame))

        self.assertIn("more than twice its median", answer.answer)

    def test_counts_by_two_groupings_give_a_cross_table_with_a_test(self):
        frame = pd.DataFrame({"Crop": ["maize", "rice"] * 30, "Region": ["north"] * 30 + ["south"] * 30})

        answer = answer_question("count by Crop and Region", frame, detect_roles(frame))

        self.assertEqual(answer.plan.intent, "crosstab")
        self.assertEqual(answer.table.columns.tolist(), ["Crop", "north", "south"])
        self.assertIn("chi-square", answer.answer)

    def test_a_segment_by_year_is_a_cross_table(self):
        frame = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=60, freq="MS"),
            "Site": ["a", "b", "c"] * 20,
            "pH": np.linspace(6.5, 7.5, 60),
        })

        answer = answer_question("mean pH by Site and year", frame, detect_roles(frame))

        self.assertEqual((answer.plan.intent, answer.plan.time_part), ("crosstab", "year"))
        self.assertEqual(answer.plan.second_dimension, "Site")

    def test_a_date_of_whole_years_beside_a_segment_is_a_cross_table(self):
        frame = pd.DataFrame({
            "Year": pd.to_datetime([f"{2005 + index // 3}-01-01" for index in range(30)]),
            "Crop": ["maize", "rice", "soy"] * 10,
        })

        answer = answer_question("count of Crop by Year", frame, detect_roles(frame))

        self.assertEqual((answer.plan.intent, answer.plan.time_part), ("crosstab", "year"))
        self.assertEqual(answer.table.columns.tolist(), ["Year", "maize", "rice", "soy"])

    def test_a_filter_cue_reads_the_filter(self):
        frame, roles = trial_frame()

        plan = parse_question("correlation between Age and Score within Condition 1", frame, roles)

        self.assertEqual(plan.conditions[0].column, "Condition")

    def test_a_numbered_row_label_names_the_row(self):
        frame = pd.DataFrame({
            "Unnamed: 0": range(100, 160), "Reads": [float(index % 17) + index / 100 for index in range(60)],
        })
        roles = detect_roles(frame)

        answer = answer_question("which OTU has the highest Reads", frame, roles)

        self.assertIsNone(answer.plan.bins)
        self.assertTrue(answer.answer.startswith("Unnamed: 0 150 has the highest Reads"))

    def test_an_unnamed_row_label_is_read_by_what_it_labels(self):
        frame, roles = summary_frame()

        answer = answer_question("which probes have the highest mean", frame, roles)

        self.assertEqual(answer.plan.dimension, "Unnamed: 0")
        self.assertIn('Read "probes" as Unnamed: 0, the row label.', answer.answer)


class ScopeAndCautionTests(unittest.TestCase):
    def test_a_long_table_answers_another_parameter_from_every_row(self):
        frame = pd.DataFrame({
            "Determinand": ["Ammonia"] * 30 + ["pH"] * 20,
            "Result": [0.5] * 30 + [7.5] * 20,
            "Site": ["a", "b"] * 25,
        })
        roles = detect_roles(frame)
        focused = frame[frame["Determinand"] == "Ammonia"].reset_index(drop=True)
        whole = WholeTable(frame, roles, "Determinand", "Ammonia")

        other = answer_question("average Result for pH", focused, roles, whole=whole)
        own = answer_question("average Result by Site", focused, roles, whole=whole)

        self.assertIn("is 7.50", other.answer)
        self.assertIn("This covers the 30 rows with Determinand = Ammonia", own.answer)

    def test_a_running_total_is_not_summed_or_averaged(self):
        frame = pd.DataFrame({"Total Rain": np.arange(40.0), "Interval Rain": np.ones(40)})
        roles = replace(
            detect_roles(frame),
            notes=("Total Rain keeps climbing, dropping back only when it resets, like a meter reading.",),
        )

        self.assertIsNone(answer_question("average Total Rain", frame, roles))
        self.assertIn("meter reading", rephrase_hint("average Total Rain", frame, roles))
        self.assertIsNotNone(answer_question("maximum Total Rain", frame, roles))

    def test_an_answer_using_flagged_values_says_so(self):
        frame = pd.DataFrame({"Wind Speed": [3.0, 4.0, 5.0, 2.5, 3.5] * 8 + [999.9]})
        roles = detect_roles(frame)

        answer = answer_question("maximum Wind Speed", frame, roles)

        self.assertIn("Caution: the data checks flag this value", answer.answer)

    def test_the_data_checks_given_are_used_instead_of_a_new_scan(self):
        frame = pd.DataFrame({"Wind Speed": [3.0, 4.0, 5.0, 2.5, 3.5] * 8})
        roles = detect_roles(frame)
        given = QualityIssue(
            "impossible", ("Wind Speed",), 1, 2.5, "A gust the sensor can't record.", "Wind Speed: 1 value",
            "a test rule", cells={"Wind Speed": pd.Index([2])},
        )

        answer = answer_question("maximum Wind Speed", frame, roles, checks=(given,))

        self.assertIn("Caution: the data checks flag this value (Wind Speed: 1 value)", answer.answer)

    def test_a_wide_table_reads_its_column_names_once(self):
        frame = pd.DataFrame(
            np.arange(40 * 3000, dtype=float).reshape(40, 3000) % 7,
            columns=[f"Probe_{number}" for number in range(3000)],
        ).assign(Group=["a", "b"] * 20)
        roles = detect_roles(frame)

        first = answer_question("average Probe_17 by Group", frame, roles)
        second = answer_question("correlation between Probe_17 and Probe_2999", frame, roles)

        self.assertEqual((first.plan.measure, first.plan.dimension), ("Probe_17", "Group"))
        self.assertEqual((second.plan.measure, second.plan.partner), ("Probe_17", "Probe_2999"))
        self.assertIs(nlq._name_index(frame.columns), nlq._name_index(list(frame.columns)))


def nitrate_frame() -> pd.DataFrame:
    """Two sites, five readings each, one depth missing. By hand: mean nitrate 25.65, North 14.3, South 37."""
    return pd.DataFrame({
        "Site": ["North"] * 5 + ["South"] * 5,
        "Depth": [1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 4.0, 6.0, 8.0, np.nan],
        "Nitrate": [10.0, 12.0, 15.0, 15.5, 19.0, 30.0, 33.0, 37.0, 41.0, 44.0],
    })


class AnswerRecordTests(unittest.TestCase):
    """Each answer carries its numbers at full precision and the rows they rest on."""

    def ask(self, question: str, frame: pd.DataFrame | None = None):
        frame = nitrate_frame() if frame is None else frame
        return answer_question(question, frame, detect_roles(frame))

    def test_an_aggregate_has_its_value_and_rows(self):
        answer = self.ask("average nitrate")
        self.assertAlmostEqual(answer.value, 25.65)
        self.assertEqual(dict(answer.figures)["n"], 10)
        self.assertEqual((answer.scope.rows_read, answer.scope.rows_used), (10, 10))
        self.assertAlmostEqual(self.ask("total nitrate").value, 256.5)

    def test_a_breakdown_has_one_value_per_group(self):
        answer = self.ask("average nitrate by site")
        self.assertIsNone(answer.value)
        self.assertEqual(dict(answer.values), {"South": 37.0, "North": 14.3})

    def test_a_condition_count_has_its_denominator(self):
        answer = self.ask("how many rows have nitrate above 20")
        self.assertEqual(answer.value, 5)
        self.assertEqual(dict(answer.figures), {"share_percent": 50.0, "denominator": 10.0})
        self.assertEqual((answer.scope.rows_in_scope, answer.scope.rows_used), (5, 5))

    def test_a_comparison_has_group_centres_and_the_test(self):
        answer = self.ask("compare nitrate between sites")
        self.assertEqual(dict(answer.values), {"South": 37.0, "North": 14.3})
        figures = dict(answer.figures)
        # Welch's t-test by hand with scipy.stats.ttest_ind(..., equal_var=False): p = 0.000169050805.
        self.assertAlmostEqual(figures["p_value"], 0.000169050805, places=10)
        self.assertAlmostEqual(figures["difference"], 22.7)

    def test_rows_left_out_are_counted_by_reason(self):
        frame = pd.concat([nitrate_frame()] * 3, ignore_index=True)
        frame["Depth"] = frame["Depth"] + np.arange(len(frame)) * 0.01
        answer = self.ask("correlation between depth and nitrate", frame)
        self.assertAlmostEqual(answer.value, frame[["Depth", "Nitrate"]].dropna().corr().iloc[0, 1])
        self.assertEqual(dict(answer.figures)["n"], 27)
        self.assertEqual(answer.scope.rows_used, 27)
        self.assertEqual(answer.scope.left_out, (("rows with Depth or Nitrate missing", 3),))

    def test_thin_data_gives_no_number(self):
        answer = self.ask("correlation between depth and nitrate")
        self.assertIsNone(answer.value)
        self.assertIn("needs at least 12", answer.answer)

    def test_exact_repeats_are_left_out_of_a_test_and_counted(self):
        frame = pd.concat([nitrate_frame()] * 3, ignore_index=True)
        frame["Depth"] = frame["Depth"] + np.arange(len(frame)) * 0.01
        frame = pd.concat([frame, frame.iloc[:4]], ignore_index=True)
        answer = self.ask("compare depth between sites", frame)
        # The four copied rows, and two of the three copies of the row with no depth.
        self.assertEqual(int(frame.duplicated().sum()), 6)
        self.assertEqual(answer.scope.left_out[0], ("exact repeats of other rows", 6))
        self.assertEqual(answer.scope.rows_read, 34)

    def test_the_record_is_json_safe(self):
        import json

        keys = {"schema_version", "status", "question", "answer", "result", "reading", "calculation",
                "evidence", "assumptions", "warnings", "decisions"}
        for question in ("average nitrate", "average nitrate by site", "compare nitrate between sites",
                         "describe nitrate", "correlation between depth and nitrate", "missing depth"):
            with self.subTest(question=question):
                data = self.ask(question).to_dict()
                self.assertEqual(set(data), keys)
                self.assertEqual(data["question"], question)
                json.dumps(data, allow_nan=False)
        data = self.ask("average nitrate by site").to_dict()
        self.assertEqual(data["reading"]["intent"], "breakdown")
        self.assertEqual((data["reading"]["measure"], data["reading"]["group_by"]), ("Nitrate", "Site"))
        self.assertEqual(data["result"]["values"][0], {"group": "South", "value": 37.0})


def visits_frame() -> pd.DataFrame:
    """Twelve numbered subjects, twenty visits each; sex is fixed within a subject."""
    rng = np.random.default_rng(44)
    subjects = np.repeat(np.arange(1, 13), 20)
    sex = np.repeat(np.tile([0, 1], 6), 20)
    level = np.repeat(rng.normal(50, 10, 12), 20)
    return pd.DataFrame({"subject#": subjects, "sex": sex, "score": level + rng.normal(0, 1, 240)})


class UnitOfAnalysisTests(unittest.TestCase):
    def test_a_grouping_fixed_within_numbered_subjects_compares_subjects(self):
        frame = visits_frame()
        answer = answer_question("compare score between sex", frame, detect_roles(frame))
        self.assertIn("Each subject# counts once per group", answer.answer)
        self.assertIn("n = 6 and 6", answer.answer)
        # By hand: Welch's t-test on the twelve subject means.
        means = frame.groupby(["subject#", "sex"])["score"].mean().reset_index()
        from scipy import stats
        expected = stats.ttest_ind(means.loc[means.sex == 0, "score"], means.loc[means.sex == 1, "score"],
                                   equal_var=False).pvalue
        self.assertAlmostEqual(dict(answer.figures)["p_value"], expected, places=10)

    def test_rows_with_no_group_are_not_counted_as_used_by_a_comparison(self):
        frame = nitrate_frame()
        frame.loc[[0, 6], "Site"] = None
        answer = answer_question("compare nitrate between sites", frame, detect_roles(frame))
        self.assertEqual(answer.scope.rows_used, 8)
        self.assertEqual(answer.scope.left_out, (("rows with Nitrate or Site missing", 2),))


class InitialsTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "playerID": [f"p{n}" for n in range(8)], "yearID": [2001] * 4 + [2002] * 4,
            "HR": [73, 49, 10, 0, 46, 34, 12, 3], "Hr": [1, 2, 3, 4, 5, 6, 7, 8],
        })

    def test_words_whose_initials_spell_a_capitals_name_read_as_it(self):
        frame = self.frame()
        answer = answer_question("total home runs", frame, detect_roles(frame))
        self.assertEqual(answer.calculation, "sum(HR)")
        self.assertEqual(answer.value, 227)
        self.assertIn('Read "home runs" as HR.', answer.answer)

    def test_initials_never_stand_for_a_name_not_in_capitals(self):
        frame = self.frame().drop(columns="HR")
        self.assertIsNone(answer_question("total home runs", frame, detect_roles(frame)))


def days_frame() -> pd.DataFrame:
    """Eighteen subjects measured on ten days each, each rising by a slope of their own."""
    rng = np.random.default_rng(45)
    day = np.tile(np.arange(10), 18)
    return pd.DataFrame({
        "Subject": np.repeat(np.arange(308, 326), 10), "Days": day,
        "Reaction": np.repeat(rng.normal(250, 30, 18), 10) + np.repeat(rng.normal(10, 6, 18), 10) * day
        + rng.normal(0, 20, 180),
    })


def orders_frame() -> pd.DataFrame:
    """Sixty orders paid in one or more instalments, so an order's value is the sum of its rows."""
    rng = np.random.default_rng(46)
    counts = rng.integers(1, 4, 60)
    return pd.DataFrame({
        "order_id": np.repeat([f"o{n:03d}" for n in range(60)], counts),
        "payment_type": rng.choice(["card", "voucher"], counts.sum()),
        "payment_value": rng.gamma(2, 50, counts.sum()).round(2),
    })


class RepeatedUnitRelationshipTests(unittest.TestCase):
    def test_a_correlation_over_repeated_subjects_counts_each_subject_once(self):
        frame = days_frame()
        answer = answer_question("does Reaction increase with Days", frame, detect_roles(frame))
        slopes = frame.groupby("Subject").apply(
            lambda rows: np.polyfit(rows["Days"], rows["Reaction"], 1)[0], include_groups=False)
        figures = dict(answer.figures)
        self.assertAlmostEqual(figures["slope"], slopes.mean())
        self.assertEqual(figures["units"], 18)
        self.assertIn("each Subject counts once", answer.answer)
        self.assertIn("per Day", answer.answer)
        self.assertNotIn("n = 180", answer.answer)

    def test_few_subjects_on_thin_data(self):
        frame = days_frame()
        few = frame[frame["Subject"] < 311]
        answer = answer_question("does Reaction increase with Days", few, detect_roles(few))
        self.assertNotIn("counts once", answer.answer)
        self.assertIsNotNone(answer.value)
        from ada.compare import RepeatedCorrelation

        few_units = RepeatedCorrelation("Baseline", "Reaction", "Subject", 8, 80, False, 0.5, 0.2, -0.3, 0.9)
        plan = QueryPlan(intent="correlation", measure="Reaction", partner="Baseline")
        answer = nlq._execute_repeated_correlation(plan, few_units, "", [])
        self.assertIsNone(answer.value)
        self.assertIn("the evidence is one value per Subject, and only 8 Subjects", answer.answer)

    def test_a_pooled_correlation_that_reverses_within_groups_says_so(self):
        rng = np.random.default_rng(47)
        parts = []
        for index, label in enumerate(("Adelie", "Gentoo", "Chinstrap")):
            x = rng.normal(40 + 5 * index, 2, 60)
            parts.append(pd.DataFrame({"species": label, "bill_length": x,
                                       "bill_depth": 0.4 * x - 4 * index + rng.normal(0, 0.5, 60)}))
        frame = pd.concat(parts, ignore_index=True)
        answer = answer_question("correlation between bill_length and bill_depth", frame, detect_roles(frame))
        self.assertLess(answer.value, 0)
        self.assertIn("Within each species the relationship runs the other way", answer.answer)
        self.assertTrue(any("runs the other way" in caution for caution in answer.scope.cautions))


class UnitAverageTests(unittest.TestCase):
    def test_an_average_per_key_combines_each_key_first(self):
        frame = orders_frame()
        expected = frame.groupby("order_id")["payment_value"].sum().mean()
        for question in ("average payment_value per order_id", "average payment value per order",
                         "average order value"):
            with self.subTest(question=question):
                answer = answer_question(question, frame, detect_roles(frame))
                self.assertAlmostEqual(answer.value, expected)
                self.assertEqual(dict(answer.figures)["units"], 60)
                self.assertAlmostEqual(dict(answer.figures)["per_row"], frame["payment_value"].mean())
        self.assertIn("then mean over the 60 order_id values", answer.calculation)

    def test_a_short_grouping_after_per_is_still_a_breakdown(self):
        frame = orders_frame()
        answer = answer_question("average payment_value per payment_type", frame, detect_roles(frame))
        self.assertEqual(answer.plan.intent, "breakdown")

    def test_a_key_with_few_values_is_listed(self):
        frame = orders_frame()
        frame = frame[frame["order_id"].isin(frame["order_id"].unique()[:10])]
        answer = answer_question("average payment_value per order_id", frame, detect_roles(frame))
        self.assertNotEqual(answer.plan.intent, "aggregate")


if __name__ == "__main__":
    unittest.main()


class EventAndFillerTests(unittest.TestCase):
    def setUp(self):
        frame = pd.DataFrame({"set_num": [f"s{index}" for index in range(12)],
                              "year": [2018, 2019, 2019, 2020, 2020, 2020, 2021, 2021, 2018, 2019, 2020,
                                       2021],
                              "num_parts": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]})
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        path = Path(holder.name) / "sets.csv"
        frame.to_csv(path, index=False)
        self.analysis = ada.analyze(path)

    def test_a_verb_that_dates_an_event_is_grammar(self):
        answer = self.analysis.ask("how many sets were released in 2020")
        self.assertEqual(answer.value, 4)

    def test_from_a_year_after_the_rows_noun_is_that_year(self):
        answer = self.analysis.ask("average num_parts of sets from 2019")
        self.assertAlmostEqual(answer.value, 50.0)
        onwards = self.analysis.ask("average num_parts from 2019 onwards")
        self.assertAlmostEqual(onwards.value, float(np.mean([20, 30, 40, 50, 60, 70, 80, 100, 110, 120])))
        ranged = self.analysis.ask("average num_parts from 2020")
        self.assertAlmostEqual(ranged.value, float(np.mean([40, 50, 60, 70, 80, 110, 120])))

    def test_where_the_rows_are_names_nothing(self):
        answer = self.analysis.ask("how many sets in the dataset")
        self.assertEqual(answer.value, 12)


class CountedFilterTests(unittest.TestCase):
    def test_a_passive_verb_after_what_is_counted_is_grammar(self):
        frame = pd.DataFrame({"athlete": list("abcdef"),
                              "medal": ["Gold", "Silver", "Gold", None, "Bronze", "Gold"]})
        answer = answer_question("how many Gold medals were won", frame, detect_roles(frame))
        self.assertEqual(answer.value, 3)
        self.assertIn("medal = Gold", answer.calculation)

    def test_a_column_of_years_named_as_a_date_scopes_a_year(self):
        frame = pd.DataFrame({"ref": range(6), "review_date": [2019, 2019, 2020, 2021, 2019, 2020],
                              "rating": [3.0, 3.5, 2.75, 4.0, 3.25, 3.0]})
        roles = detect_roles(frame)
        counted = answer_question("how many reviews in 2019", frame, roles)
        average = answer_question("average rating in 2019", frame, roles)
        self.assertEqual(counted.value, 3)
        self.assertNotIn("distinct", counted.calculation)
        self.assertAlmostEqual(average.value, 3.25)

    def test_a_column_of_dates_that_are_not_years_is_not_a_year_column(self):
        frame = pd.DataFrame({"review_date": [20190101, 20190102, 20200101], "rating": [3.0, 3.5, 4.0]})
        self.assertIsNone(nlq._year_column(frame, detect_roles(frame)))

    def test_a_share_of_the_value_a_column_is_filtered_to_counts_rows(self):
        frame = pd.DataFrame({"Hair": ["Black", "Black", "Blond", "Blond"],
                              "Eye": ["Brown", "Blue", "Brown", "Blue"], "Freq": [30, 10, 5, 55]})
        answer = answer_question("what share have blue eyes", frame, detect_roles(frame))
        self.assertIn("65 of the 100 cases (65.0%) have Eye = Blue", answer.answer)

    def test_a_code_in_a_column_too_large_to_index_is_found(self):
        rows = nlq.MAX_FILTER_CANDIDATES + 1
        frame = pd.DataFrame({"playerID": [f"pl{index:05d}x" for index in range(rows)],
                              "HR": np.arange(rows) * 7919 % 61})
        answer = answer_question("total HR for pl00123x", frame, detect_roles(frame))
        self.assertEqual(answer.value, 123 * 7919 % 61)
        self.assertIn("playerID = pl00123x", answer.calculation)
        self.assertIsNone(answer_question("total HR for zz99999q", frame, detect_roles(frame)))


class RepeatedUnitCountTests(unittest.TestCase):
    def test_rows_are_not_counted_as_units_that_repeat(self):
        # Six trees measured at six ages, listed tree by tree, as Loblolly lists 14.
        ages = [3, 5, 10, 15, 20, 25]
        frame = pd.DataFrame({"height": [age * 2.5 + tree / 10 for tree in range(6) for age in ages],
                              "age": ages * 6, "Seed": [301 + tree for tree in range(6) for _ in ages]})
        roles = detect_roles(frame)
        self.assertIsNone(answer_question("how many trees were measured", frame, roles))
        self.assertEqual(answer_question("how many rows", frame, roles).value, 36)


class TimePointTests(unittest.TestCase):
    def setUp(self):
        # Two chicks on each of two diets, weighed on days 0 and 21.
        self.frame = pd.DataFrame({"weight": [42.0, 41.0, 300.0, 240.0, 40.0, 43.0, 260.0, 280.0],
                                   "Time": [0, 0, 21, 21] * 2, "Chick": [1, 2, 1, 2, 3, 4, 3, 4],
                                   "Diet": [1] * 4 + [3] * 4})

    def test_day_and_a_number_pick_the_time_point(self):
        answer = answer_question("average weight on day 21, diet 3", self.frame, detect_roles(self.frame))
        self.assertAlmostEqual(answer.value, 270.0)
        self.assertIn("Time = 21", answer.calculation)
        self.assertIn('Read "on day 21" as Time = 21.', answer.scope.readings)

    def test_a_day_the_column_does_not_hold_is_not_read(self):
        answer = answer_question("average weight on day 22", self.frame, detect_roles(self.frame))
        self.assertTrue(answer is None or "Time" not in answer.calculation)


class ShareBaseTests(unittest.TestCase):
    def setUp(self):
        # Adelie: 4 female of 6; Gentoo: 1 female of 4. By hand: 40% of all, 66.7% of Adelie.
        self.frame = pd.DataFrame({"species": ["Adelie"] * 6 + ["Gentoo"] * 4,
                                   "sex": ["female"] * 4 + ["male"] * 2 + ["female"] + ["male"] * 3,
                                   "body_mass_g": [3500.0, 3600, 3400, 3700, 3900, 4000, 5000, 5200, 5400,
                                                   5600]})
        self.roles = detect_roles(self.frame)

    def test_a_share_of_a_group_is_of_that_group(self):
        answer = answer_question("what percentage of Adelie are female", self.frame, self.roles)
        self.assertIn("4 of the 6 rows with species = Adelie (66.7%) have sex = female", answer.answer)
        self.assertAlmostEqual(dict(answer.figures)["share_percent"], 400 / 6)

    def test_a_condition_after_the_verb_is_counted(self):
        answer = answer_question("what share of Gentoo have body_mass_g above 5100", self.frame, self.roles)
        self.assertIn("3 of the 4 rows with species = Gentoo", answer.answer)

    def test_values_side_by_side_are_both_counted(self):
        answer = answer_question("share of female Adelie", self.frame, self.roles)
        self.assertIn("4 of the 10 rows", answer.answer)

    def test_a_yes_no_outcome_is_counted_among_the_rest(self):
        frame = pd.DataFrame({"Sex": ["Male", "Male", "Female", "Female"],
                              "Survived": ["No", "Yes", "No", "Yes"], "Freq": [1364, 367, 126, 344]})
        answer = answer_question("what share of women survived", frame, detect_roles(frame))
        self.assertIn("344 of the 470 cases with Sex = Female (73.2%) have Survived = Yes", answer.answer)
        everyone = answer_question("what share survived", frame, detect_roles(frame))
        self.assertIn("711 of the 2,201 cases", everyone.answer)


class SexWordTests(unittest.TestCase):
    def test_men_and_women_name_a_column_of_sexes(self):
        frame = pd.DataFrame({"Gender": ["Male", "Female", "Female", "Male"], "score": [1.0, 2.0, 3.0, 4.0]})
        answer = answer_question("average score for women", frame, detect_roles(frame))
        self.assertAlmostEqual(answer.value, 2.5)
        self.assertIn('Read "women" as Gender = Female.', answer.scope.readings)

    def test_single_letter_codes_count_too(self):
        frame = pd.DataFrame({"sex": ["M", "F", "F", "M", "F"], "score": [1.0, 2.0, 3.0, 4.0, 5.0]})
        answer = answer_question("average score for men", frame, detect_roles(frame))
        self.assertAlmostEqual(answer.value, 2.5)

    def test_a_column_that_is_not_only_sexes_is_not_read(self):
        frame = pd.DataFrame({"team": ["Male", "Female", "Mixed", "Youth"], "score": [1.0, 2.0, 3.0, 4.0]})
        self.assertIsNone(answer_question("average score for women", frame, detect_roles(frame)))


class AbbreviatedNameTests(unittest.TestCase):
    def test_a_word_reads_the_column_that_abbreviates_it(self):
        frame = pd.DataFrame({"Dept": ["A", "A", "B", "B"], "Freq": [10, 20, 30, 40],
                              "Admit": ["Yes", "No", "Yes", "No"]})
        answer = answer_question("how many cases in department A", frame, detect_roles(frame))
        self.assertEqual(answer.value, 30)
        self.assertIn('Read "department" as Dept.', answer.scope.readings)

    def test_an_abbreviation_with_two_vowels_or_too_close_a_word_is_not_read(self):
        from ada.nlq import _skeleton_columns

        self.assertEqual(_skeleton_columns("department", {"Dept": ("dept",)}), ["Dept"])
        self.assertEqual(_skeleton_columns("weight", {"wgt": ("wgt",)}), ["wgt"])
        self.assertEqual(_skeleton_columns("route", {"rate": ("rate",)}), [])
        self.assertEqual(_skeleton_columns("depot", {"dept": ("dept",)}), [])


class OrdinalColumnTests(unittest.TestCase):
    def test_an_ordinal_before_a_columns_name_picks_its_value(self):
        frame = pd.DataFrame({"year": [2020] * 4 + [2021] * 4, "quarter": [1, 2, 3, 4] * 2,
                              "employed": [10.0, 11, 12, 13, 14, 15, 16, 17]})
        answer = answer_question("total employed in the fourth quarter of 2020", frame, detect_roles(frame))
        self.assertEqual(answer.value, 13)
        self.assertIn('Read "the fourth quarter" as quarter = 4.', answer.scope.readings)

    def test_an_ordinal_the_column_does_not_hold_is_not_read(self):
        frame = pd.DataFrame({"quarter": [1, 2, 3] * 2, "employed": [10.0, 11, 12, 13, 14, 15]})
        answer = answer_question("total employed in the fourth quarter", frame, detect_roles(frame))
        self.assertTrue(answer is None or "quarter = 4" not in answer.calculation)


class ValueBesideColumnTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        levels = ["Low", "Medium", "High"]
        return pd.DataFrame({"Sat": levels * 3, "Infl": sorted(levels * 3),
                             "Cont": ["Low", "High", "High"] * 3, "Freq": [10, 20, 30, 5, 5, 40, 20, 10, 10]})

    def test_the_word_after_a_value_names_which_column_holds_it(self):
        frame = self.frame()
        for question in ("share of residents with high satisfaction", "share of residents highly satisfied"):
            with self.subTest(question=question):
                answer = answer_question(question, frame, detect_roles(frame))
                self.assertIn("Sat = High", answer.calculation)
                self.assertIn("80 of the 150 cases (53.3%)", answer.answer)

    def test_a_filler_may_stand_between_the_value_and_the_column(self):
        frame = self.frame()
        answer = answer_question("how many residents had a high level of contact", frame, detect_roles(frame))
        self.assertIn("Cont = High", answer.calculation)
        self.assertIn('Read "high level of contact" as Cont = High.', answer.scope.readings)

    def test_a_value_of_several_columns_with_no_column_beside_it_is_refused(self):
        frame = self.frame()
        roles = detect_roles(frame)
        self.assertIsNone(answer_question("share of residents with high", frame, roles))
        self.assertIn('"high" is a value of', rephrase_hint("share of residents with high", frame, roles))

    def test_an_adverb_with_no_column_after_it_is_not_a_value(self):
        frame = self.frame()
        self.assertIsNone(answer_question("share of residents highly", frame, detect_roles(frame)))


class SeenFlagTests(unittest.TestCase):
    def test_a_verb_of_seeing_before_a_yes_no_column_keeps_its_yes_rows(self):
        frame = pd.DataFrame({"id": range(8), "Foraging": [True, False] * 4,
                              "Eating": [True] * 2 + [False] * 6})
        answer = answer_question("how many rows were seen foraging", frame, detect_roles(frame))
        self.assertEqual(answer.value, 4)
        self.assertIn("Foraging = 1", answer.calculation)


class DistinctShareTests(unittest.TestCase):
    def test_a_distinct_count_with_a_filter_gives_its_share_of_all_distinct_values(self):
        frame = pd.DataFrame({"tree": ["a", "a", "b", "c", "d"], "sick": ["yes", "yes", "no", "yes", "no"],
                              "height": [1.0, 2, 3, 4, 5]})
        answer = answer_question("how many distinct tree with sick yes", frame, detect_roles(frame))
        self.assertEqual(answer.value, 2)
        self.assertEqual(dict(answer.figures)["share_percent"], 50)
        self.assertIn("50.0% of the 4 in all", answer.answer)


class FlagVerbAndFreeTextTests(unittest.TestCase):
    def test_a_verb_of_showing_keeps_a_flag_s_yes_rows(self):
        frame = pd.DataFrame({"TREES": [1, 1, 0, 1], "LAKES": [0, 1, 0, 0]})
        roles = detect_roles(frame)
        answer = answer_question("share of episodes that featured a lake", frame, roles)
        self.assertAlmostEqual(dict(answer.figures)["share_percent"], 25.0)

    def test_a_word_inside_free_text_picks_no_rows(self):
        titles = [f"A long study of {word} in the north of the country" for word in
                  ("rivers", "lakes", "soil", "birds", "TCD data", "fish")]
        frame = pd.DataFrame({"body": ["Trinity College Dublin (TCD)", "UCD", "UCD", "UCC", "UCC", "UCD"],
                              "title": titles, "amount": [10.0, 20, 30, 40, 50, 60]})
        roles = detect_roles(frame)
        answer = answer_question("total amount for TCD", frame, roles)
        self.assertEqual(answer.calculation, "sum(amount) · filtered to body = Trinity College Dublin (TCD)")
        self.assertIsNone(answer_question("total amount for soil", frame, roles))

    def test_a_share_of_all_the_money_weighs_by_the_money_column(self):
        frame = pd.DataFrame({"body": ["TCD", "UCD", "UCD", "TCD"], "commitment": [10.0, 30, 50, -10]})
        answer = answer_question("share of all the money for body TCD", frame, detect_roles(frame))
        self.assertEqual(dict(answer.figures)["share_percent"], 0.0)
        self.assertIn("sum(commitment)", answer.calculation)


class KindsAndBoundsTests(unittest.TestCase):
    def pets(self, names: int = 60) -> pd.DataFrame:
        half = names // 2
        return pd.DataFrame({"animal_name": ["Dog"] + [f"Pet{index}" for index in range(names - 1)],
                             "animal_type": ["dog"] * half + ["cat"] * (names - half)})

    def test_a_word_in_lower_case_is_a_kind_not_a_name(self):
        pets = self.pets()
        answer = answer_question("how many dogs", pets, detect_roles(pets))
        self.assertEqual(answer.value, 30)
        self.assertIn("animal_type = dog", answer.calculation)
        # With fewer than 50 names, animal_name is not a column of names, and "dogs" could mean either.
        thin = self.pets(names=20)
        self.assertIsNone(answer_question("how many dogs", thin, detect_roles(thin)))

    def test_a_noun_for_rows_that_is_a_kind_picks_those_rows(self):
        birds = pd.DataFrame({"animal_type": ["bird"] * 4 + ["cat"] * 4,
                              "outcome_type": ["euthanasia", "adoption", "adoption", "adoption"] * 2})
        answer = answer_question("share of the birds had outcome_type euthanasia", birds, detect_roles(birds))
        self.assertEqual(dict(answer.figures)["share_percent"], 25.0)
        self.assertIn("animal_type = bird", answer.calculation)

    def test_a_determiner_is_no_value(self):
        frame = pd.DataFrame({"citizenship": ["Other", "Mexico", "Mexico", "Other"],
                              "encounters": [5.0, 10.0, 20.0, 1.0]})
        roles = detect_roles(frame)
        self.assertEqual(answer_question("total encounters for Other", frame, roles).value, 6)
        answer = answer_question("total encounters for Mexico more than any other", frame, roles)
        self.assertTrue(answer is None or "Other" not in answer.calculation)

    def test_a_word_of_two_letters_has_no_plural(self):
        frame = pd.DataFrame({"Year": [2015, 2015, 2016], "Name": ["Jack", "Ines", "Jack"],
                              "babies": [565.0, 40.0, 500.0]})
        answer = answer_question("total babies named Jack in 2015", frame, detect_roles(frame))
        self.assertEqual(answer.value, 565)
        self.assertNotIn("Ines", answer.calculation)

    def test_a_bound_reads_its_unit_and_its_adjective(self):
        quakes = pd.DataFrame({"depth_km": [0.2, 0.4, 0.6, 1.5], "magnitude": [1.0, 1.2, 2.0, 3.0]})
        roles = detect_roles(quakes)
        questions = ("share of rows with depth less than 1 km", "share of the rows were less than 1 km deep")
        for question in questions:
            answer = answer_question(question, quakes, roles)
            self.assertEqual(dict(answer.figures)["share_percent"], 75.0, question)
            self.assertIn("depth_km below 1", answer.calculation)
        deeper = answer_question("share of rows deeper than 500 metres", quakes, roles)
        self.assertIn("depth_km above 0.5", deeper.calculation)
        # A column whose name gives no unit cannot say what 1 km is in it.
        plain = quakes.rename(columns={"depth_km": "depth"})
        question = "share of rows with depth less than 1 km"
        self.assertIsNone(answer_question(question, plain, detect_roles(plain)))


class CountColumnsAndNamesTests(unittest.TestCase):
    def test_a_column_s_own_name_wins_over_a_plural(self):
        frame = pd.DataFrame({"LAKE": [1, 0, 1, 0], "LAKES": [0, 0, 0, 0], "TREE": [1, 1, 1, 0]})
        roles = detect_roles(frame)
        answer = answer_question("share of episodes that featured a lake", frame, roles)
        self.assertIn("LAKE = 1", answer.calculation)
        self.assertNotIn("LAKES", answer.calculation)
        self.assertIn("LAKES = 1", answer_question("share of episodes with lakes", frame, roles).calculation)

    def test_a_verb_names_the_column_that_counts_it(self):
        dates = pd.to_datetime(["2014-01-01", "2014-02-01", "2014-01-01"])
        frame = pd.DataFrame({"city": ["Austin", "Austin", "Dallas"], "date": dates,
                              "sales": [1500, 1700, 4000], "median": [2e5, 2.1e5, 1.9e5]})
        answer = answer_question("how many homes were sold in Austin in 2014", frame, detect_roles(frame))
        self.assertEqual(answer.value, 3200)
        self.assertIn("sum of sales", " ".join(answer.plan.notes))
        # With no column of sales, the rows are not homes sold.
        rows = frame.drop(columns="sales")
        answer = answer_question("how many homes were sold in Austin in 2014", rows, detect_roles(rows))
        self.assertTrue(answer is None or "sales" not in answer.calculation)

    def test_a_word_in_lower_case_is_not_part_of_a_name(self):
        frame = pd.DataFrame({"country": ["New Zealand", "Kenya", "Papua New Guinea"], "cases": [10, 20, 30]})
        roles = detect_roles(frame)
        self.assertEqual(answer_question("total cases in New Zealand", frame, roles).value, 10)
        answer = answer_question("total new cases in Kenya", frame, roles)
        self.assertTrue(answer is None or "New" not in answer.calculation)


class GroupsAndModifiersTests(unittest.TestCase):
    def rescues(self) -> pd.DataFrame:
        boroughs = ["Enfield"] * 5 + ["Croydon"] * 4 + ["City of London"] * 2 + ["Barnet"] * 3
        costs = [100.0] * 5 + [900.0] * 4 + [50.0] * 2 + [10.0] * 3
        return pd.DataFrame({"borough": boroughs, "cost": costs})

    def test_the_most_with_no_measure_counts_each_group_s_rows(self):
        frame = self.rescues()
        answer = answer_question("which borough has the most cases", frame, detect_roles(frame))
        # Croydon has the most cost; Enfield the most rows.
        self.assertIn("Enfield", answer.answer)
        self.assertIsNone(answer.plan.measure)

    def test_part_of_a_value_before_its_column_s_name_is_no_filter(self):
        frame = self.rescues()
        answer = answer_question("which London boroughs have the most cases", frame, detect_roles(frame))
        self.assertTrue(answer is None or "City of London" not in answer.calculation)
        # A value written in full still picks its rows.
        whole = answer_question("how many rows for City of London borough", frame, detect_roles(frame))
        self.assertEqual(whole.value, 2)

    def test_part_of_a_value_before_its_column_s_name_in_the_singular_is_that_value(self):
        regions = ["Southwest Land Border"] * 3 + ["Northern Land Border"] * 2 + ["Other"]
        frame = pd.DataFrame({"land_border_region": regions, "encounters": [5.0, 6.0, 7.0, 1.0, 2.0, 3.0]})
        answer = answer_question("total encounters at the southwest border", frame, detect_roles(frame))
        self.assertEqual(answer.value, 18)


class OneYearChangeAndCountryTests(unittest.TestCase):
    def rescues(self) -> pd.DataFrame:
        dates = [f"2019-{month:02d}-10" for month in (1, 3, 6, 11)] + [
            f"2020-{month:02d}-10" for month in (1, 2, 3, 4, 5, 6, 11, 12)]
        dates += ["2019-05-10", "2019-12-10", "2020-05-10", "2020-12-10"]
        animals = ["Fox"] * 12 + ["Cat"] * 4
        costs = [100.0] * 4 + [150.0] * 8 + [500.0, 500.0, 600.0, 500.0]
        return pd.DataFrame({"date": pd.to_datetime(dates), "animal": animals, "cost": costs})

    def test_a_change_in_a_year_is_against_the_year_before(self):
        frame = self.rescues()
        answer = answer_question("how did the number of Fox change in 2020", frame, detect_roles(frame))
        figures = dict(answer.figures)
        self.assertEqual((figures["previous"], figures["latest"]), (4.0, 8.0))
        self.assertIn("Read a change in 2020 as 2020 against 2019.", answer.answer)

    def test_groups_ranked_by_a_change_between_two_years_compare_whole_years(self):
        frame = self.rescues()
        answer = answer_question("which animal rose most from 2019 to 2020", frame, detect_roles(frame))
        # Fox: 400 to 1,200; Cat: 1,000 to 1,100. The last two months alone would rank them otherwise.
        self.assertIn("Fox", answer.answer.split(":")[0])
        self.assertIn("+200.0%", answer.answer)
        self.assertIn("from 2019 to 2020", answer.answer)
        self.assertIn("2020 vs 2019", answer.calculation)

    def test_the_united_states_is_all_the_rows_of_a_table_of_its_states(self):
        names = ["Alabama", "Arizona", "California", "Colorado", "Florida", "Georgia", "Idaho", "Iowa",
                 "New Mexico", "Texas", "Utah", "Oregon"]
        frame = pd.DataFrame({"state": names, "cases": [float(index) for index in range(1, 13)]})
        answer = answer_question("total cases in the United States", frame, detect_roles(frame))
        self.assertEqual(answer.value, 78)
        self.assertIn("as all the rows", answer.answer)
        # With a row for the country, that row is what the words pick.
        national = pd.concat([frame, pd.DataFrame({"state": ["United States"], "cases": [78.0]})])
        answer = answer_question("total cases in the United States", national, detect_roles(national))
        self.assertNotIn("as all the rows", answer.answer)


class TextPeriodsAndAgesTests(unittest.TestCase):
    def papers(self) -> pd.DataFrame:
        dates = pd.to_datetime(["2019-11-01", "2019-12-01", "2021-02-01", "2021-06-01", "2021-07-01",
                                "2021-12-01"])
        titles = ["COVID and jobs", "Taxes", "covid-19 schools", "Trade", "COVID again", "Banks"]
        return pd.DataFrame({"date": dates, "title": titles, "pages": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})

    def members(self) -> pd.DataFrame:
        return pd.DataFrame({"chamber": ["house"] * 4 + ["senate"] * 3, "state": ["TX"] * 7,
                             "age": [39.5, 40.0, 52.0, 70.0, 64.9, 65.0, 80.0]})

    def test_a_word_before_a_text_column_is_part_of_it(self):
        frame = self.papers()
        answer = answer_question("how many papers had COVID in the title", frame, detect_roles(frame))
        self.assertEqual(answer.value, 3)
        self.assertIn('title containing "COVID"', answer.calculation)

    def test_part_of_a_year_is_its_days(self):
        frame = self.papers()
        roles = detect_roles(frame)
        answer = answer_question("how many papers in the first half of 2021", frame, roles)
        self.assertEqual(answer.value, 2)
        self.assertIn("between 2021-01-01 and 2021-06-30", answer.calculation)
        last = answer_question("how many papers in the last quarter of 2019", frame, roles)
        self.assertEqual(last.value, 2)
        # A quarter relative to today is still refused.
        self.assertIsNone(answer_question("how many papers last quarter", frame, roles))

    def test_an_age_bound_after_a_verb_is_on_the_age_column(self):
        frame = self.members()
        roles = detect_roles(frame)
        answer = answer_question("how many rows were under 40", frame, roles)
        self.assertEqual(answer.value, 1)
        self.assertIn("age below 40", answer.calculation)
        self.assertEqual(answer_question("how many rows were 65 or older", frame, roles).value, 3)
        # "or" between two values is still refused.
        self.assertIsNone(answer_question("how many rows for house or senate", frame, roles))

    def test_a_plural_formed_from_a_value_is_not_every_row(self):
        frame = self.members()
        reading = nlq._read_question("how many senators were 65 or older", frame, detect_roles(frame))
        self.assertIn("chamber = senate", reading.problem or "")
        self.assertIsNone(answer_question("how many senators were 65 or older", frame, detect_roles(frame)))


class CountTableCrossTests(unittest.TestCase):
    def miners(self) -> pd.DataFrame:
        ages = ["20-24"] * 2 + ["40-44"] * 2 + ["60-64"] * 2
        return pd.DataFrame({"Breathlessness": ["B", "NoB"] * 3, "Age": ages,
                             "Freq": [16, 1936, 200, 1800, 600, 900]})

    def test_a_cross_table_of_a_table_of_counts_adds_up_the_counts(self):
        frame = self.miners()
        answer = answer_question("how does breathlessness change with age", frame, detect_roles(frame))
        self.assertIn("sum(Freq)", answer.calculation)
        # Each band's rows tie at one each; the counts they hold do not.
        self.assertIn("The two are associated", answer.answer)
        shares = dict(zip(answer.table["Age"], answer.table["share B"], strict=True))
        self.assertEqual(shares, {"20-24": round(16 / 1952, 4), "40-44": 0.1, "60-64": 0.4})
        self.assertIn("highest for 60-64, at 40.0%", answer.answer)

    def test_yes_and_no_levels(self):
        from ada.nlq import _yes_level

        self.assertEqual(_yes_level(["NoB", "B"]), "B")
        self.assertEqual(_yes_level(["Yes", "No"]), "Yes")
        self.assertEqual(_yes_level(["non smoker", "smoker"]), "smoker")
        self.assertIsNone(_yes_level(["Male", "Female"]))
        self.assertIsNone(_yes_level(["B", "NoB", "Other"]))


class FullNamesAndBoundsTests(unittest.TestCase):
    def quakes(self) -> pd.DataFrame:
        return pd.DataFrame({"mag": [4.6, 5.0, 6.1, 7.2, 7.5], "magType": ["mb", "mb", "mww", "mww", "mww"],
                             "magError": [0.1, 0.1, 0.05, 0.03, 0.02],
                             "depth": [10.0, 35.0, 12.0, 20.0, 10.0]})

    def test_a_word_names_the_one_column_it_spells_in_full(self):
        frame = self.quakes()
        answer = answer_question("average magnitude", frame, detect_roles(frame))
        self.assertIn("mean(mag)", answer.calculation)
        plants = pd.DataFrame({"capacity_mw": [10.0, 20.0, 30.0], "year_of_capacity_data": [2017, 2018, 2019],
                               "fuel": ["Coal", "Gas", "Hydro"]})
        answer = answer_question("total capacity", plants, detect_roles(plants))
        self.assertEqual(answer.value, 60)

    def test_a_number_or_more_after_a_column_bounds_it(self):
        frame = self.quakes()
        roles = detect_roles(frame)
        self.assertEqual(answer_question("how many rows reached magnitude 7 or more", frame, roles).value, 2)
        self.assertEqual(answer_question("how many rows with depth 12 or less", frame, roles).value, 3)

    def test_a_time_period_column_of_years_is_the_year(self):
        frame = pd.DataFrame({"TIME_PERIOD: Time": [2004, 2004, 2024, 2024], "geo": ["SE", "DE", "SE", "DE"],
                              "OBS_VALUE": [38.7, 6.2, 66.4, 22.0]})
        answer = answer_question("average OBS_VALUE in 2024", frame, detect_roles(frame))
        self.assertAlmostEqual(answer.value, 44.2)


class StateAndNounTests(unittest.TestCase):
    def test_the_most_of_a_value_counts_rows_by_period(self):
        days = ["2020-01-01", "2020-01-01", "2020-01-02"]
        frame = pd.DataFrame({"time": pd.to_datetime(days), "type": ["quake"] * 3, "depth": [1.0, 2.0, 9.0]})
        answer = answer_question("which day had the most quakes", frame, detect_roles(frame))
        self.assertIn("row count", answer.answer)
        self.assertIn("1 Jan 2020", answer.answer)

    def test_a_state_s_name_picks_its_postal_code(self):
        codes = ["al", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "id", "tx", "wv", "va"]
        frame = pd.DataFrame({"state": codes, "deaths": [float(index) for index in range(len(codes))]})
        roles = detect_roles(frame)
        self.assertEqual(answer_question("deaths in Texas", frame, roles).value, 10)
        self.assertEqual(answer_question("deaths in West Virginia", frame, roles).value, 11)

    def test_a_postal_code_or_a_press_abbreviation_picks_a_state_name(self):
        names = ["Alabama", "Arizona", "California", "Colorado", "Florida", "Georgia", "Idaho", "Iowa",
                 "New Mexico", "Texas", "Utah", "Oregon"]
        frame = pd.DataFrame({"state": names, "colonies": [float(index) for index in range(len(names))]})
        roles = detect_roles(frame)
        self.assertEqual(answer_question("colonies in NM", frame, roles).value, 8)
        self.assertEqual(answer_question("colonies in Calif.", frame, roles).value, 2)
        # "OR" in capitals is a word, and "calif" without its full stop is no abbreviation.
        self.assertIsNone(answer_question("colonies in OR", frame, roles))
        self.assertIsNone(answer_question("colonies in calif", frame, roles))

    def test_few_codes_are_not_read_as_states(self):
        frame = pd.DataFrame({"state": ["tx", "ca"], "deaths": [1.0, 2.0]})
        reading = nlq._read_question("deaths in Texas", frame, detect_roles(frame))
        self.assertFalse(reading.filters)

    def test_a_plural_the_values_of_several_columns_share_names_the_rows(self):
        frame = pd.DataFrame({
            "SEX": ["Male Characters", "Female Characters", "Agender Characters", "Male Characters"],
            "ALIGN": ["Good Characters", "Bad Characters", "Bad Characters", "Neutral Characters"],
        })
        roles = detect_roles(frame)
        self.assertEqual(answer_question("how many characters are women", frame, roles).value, 1)
        self.assertEqual(answer_question("how many characters are men", frame, roles).value, 2)
class OutcomeWordTests(unittest.TestCase):
    def test_words_for_dying_pick_the_value_that_spells_it(self):
        frame = pd.DataFrame({"result": ["Died", "Injured", "Died"], "mode": ["Car", "Car", "Bus"],
                              "Freq": [5, 50, 2]})
        roles = detect_roles(frame)
        for question in ("how many deaths", "how many were killed", "how many dead"):
            with self.subTest(question=question):
                self.assertEqual(answer_question(question, frame, roles).value, 7)

    def test_a_yes_no_column_named_for_surviving_reads_both_ways(self):
        frame = pd.DataFrame({"Sex": ["Male", "Female", "Male", "Female"],
                              "Survived": ["Yes", "Yes", "No", "No"], "Freq": [10, 30, 90, 10]})
        roles = detect_roles(frame)
        self.assertEqual(answer_question("how many survivors", frame, roles).value, 40)
        self.assertEqual(answer_question("how many passengers died", frame, roles).value, 100)

    def test_a_death_word_names_the_one_column_called_for_deaths(self):
        frame = pd.DataFrame({"year": [2000, 2000, 2001], "fatal": [5.0, 7, 9], "nfatal": [1.0, 2, 3]})
        roles = detect_roles(frame)
        self.assertEqual(answer_question("total deaths in 2000", frame, roles).value, 12)
        two = pd.DataFrame({"year": [2000, 2001], "fatal": [5.0, 7], "deaths": [1.0, 2]})
        reading = nlq._read_question("total killed in 2000", two, detect_roles(two))
        self.assertNotIn("fatal", reading.spans)

    def test_two_columns_that_spell_an_outcome_leave_the_word_unread(self):
        frame = pd.DataFrame({"driver": ["Died", "Lived"], "passenger": ["Lived", "Died"], "n": [1, 2]})
        reading = nlq._read_question("how many deaths", frame, detect_roles(frame))
        self.assertFalse(reading.filters)


class ShareSplitTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"result": ["Died", "Died", "Injured", "Injured"],
                             "gender": ["Male", "Female", "Male", "Female"], "Freq": [80, 20, 500, 400]})

    def test_the_sentence_s_own_verb_ends_the_base(self):
        frame = self.frame()
        roles = detect_roles(frame)
        for question, share in (("share of victims who died were men", 80.0),
                                ("share of victims were men who died", 8.0),
                                ("share of male victims died", 80 / 580 * 100),
                                ("share of victims who were men and had died", 8.0)):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertAlmostEqual(dict(answer.figures)["share_percent"], share)

    def test_a_second_plural_in_a_table_of_counts_is_not_the_rows(self):
        frame = self.frame()
        hint = rephrase_hint("how many people were dentists", frame, detect_roles(frame))
        self.assertIn("dentists", hint)


class RateTests(unittest.TestCase):
    def policies(self) -> pd.DataFrame:
        return pd.DataFrame({"District": [1, 1, 1, 1, 2, 2, 2, 2],
                             "Holders": [100, 300, 60, 40, 50, 50, 70, 30],
                             "Claims": [10, 20, 6, 4, 15, 5, 8, 12]})

    def test_a_rate_is_one_total_over_another(self):
        frame = self.policies()
        roles = detect_roles(frame)
        answer = answer_question("Claims per Holders", frame, roles)
        self.assertAlmostEqual(answer.value, 80 / 700)
        self.assertEqual(answer.calculation, "sum(Claims) ÷ sum(Holders)")
        scaled = answer_question("Claims per 1,000 Holders for District 2", frame, roles)
        self.assertAlmostEqual(scaled.value, 40 / 200 * 1000)
        self.assertEqual(dict(scaled.figures)["denominator"], 200)

    def test_a_plural_that_names_a_column_of_counts_adds_it_up(self):
        frame = pd.DataFrame({"year": [2022, 2022, 2023, 2023],
                              "citizenship": ["MEXICO", "CUBA", "MEXICO", "CUBA"],
                              "encounter_count": [30, 10, 50, 10]})
        roles = detect_roles(frame)
        self.assertEqual(answer_question("how many encounters in 2023", frame, roles).value, 60)
        share = answer_question("share of encounters with citizenship MEXICO", frame, roles)
        self.assertAlmostEqual(dict(share.figures)["share_percent"], 80 / 100 * 100)
        self.assertEqual(answer_question("how many rows in 2023", frame, roles).value, 2)
        grouped = answer_question("number of encounters by citizenship", frame, roles)
        self.assertIn("sum(encounter_count)", grouped.calculation)

    def test_a_column_named_for_one_thing_numbers_it_and_is_not_added_up(self):
        rats = pd.DataFrame({"Rat": [1, 1, 2, 2, 3, 3], "Diet": [1, 1, 1, 1, 2, 2],
                             "weight": [1.0, 2, 3, 4, 5, 6]})
        answer = answer_question("how many rats were on each diet", rats, detect_roles(rats))
        self.assertIn("count distinct Rat", answer.calculation)

    def test_two_columns_of_counts_for_one_noun_are_not_guessed_between(self):
        frame = pd.DataFrame({"year": [2022, 2023], "encounter_count": [3, 5], "encounters": [1, 2]})
        self.assertIsNone(answer_question("how many encounters in 2023", frame, detect_roles(frame)))

    def test_a_code_or_an_average_after_per_is_no_rate(self):
        frame = self.policies()
        roles = detect_roles(frame)
        for question in ("Claims per District", "average Claims per District"):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertNotEqual(getattr(answer, "calculation", ""), "sum(Claims) ÷ sum(District)")

    def test_a_rate_of_negative_numbers_is_not_read(self):
        frame = pd.DataFrame({"profit": [-5.0, 10.0, 3.0], "staff": [2.0, 3.0, 4.0]})
        answer = answer_question("profit per staff", frame, detect_roles(frame))
        self.assertFalse(answer is not None and answer.plan.aggregation == "ratio")


class AgeTests(unittest.TestCase):
    def bands(self) -> pd.DataFrame:
        return pd.DataFrame({"age": ["12", "13", "20", "22-23", "30-34", "35-49", "50-64", "65+"],
                             "use": [1.0, 2, 30, 40, 35, 25, 20, 10]})

    def test_ages_written_out_pick_the_bands_that_make_them_up(self):
        frame = self.bands()
        roles = detect_roles(frame)
        for question, values in (("average use among 30- to 34-year-olds", "age = 30-34"),
                                 ("average use among people 65 and older", "age = 65+"),
                                 ("average use among 20-year-olds", "age = 20"),
                                 ("average use for people aged 35 and over", "age in (35-49, 50-64, 65+)")):
            with self.subTest(question=question):
                self.assertIn(values, answer_question(question, frame, roles).calculation)

    def test_ages_that_cut_a_band_in_two_are_refused(self):
        frame = self.bands()
        roles = detect_roles(frame)
        self.assertIsNone(answer_question("average use for people aged 40 and over", frame, roles))
        hint = rephrase_hint("average use for people aged 40 and over", frame, roles)
        self.assertIn("cuts the age band 35-49", hint)

    def test_ages_on_a_numeric_column_are_conditions(self):
        frame = pd.DataFrame({"age": [10, 20, 30, 40, 50, 60], "income": [0.0, 10, 20, 30, 40, 50]})
        answer = answer_question("average income for people aged 30 and over", frame, detect_roles(frame))
        self.assertEqual(answer.value, 35)

    def test_numbers_without_an_age_word_are_not_ages(self):
        frame = self.bands()
        reading = nlq._read_question("average use rose from 12 to 20", frame, detect_roles(frame))
        self.assertFalse(reading.filters)


class DayTests(unittest.TestCase):
    def test_a_day_written_out_or_in_iso_form_is_a_condition_on_the_date(self):
        times = ["2011-04-27 10:00", "2011-04-27 18:00", "2011-04-28 09:00"]
        frame = pd.DataFrame({"date": pd.to_datetime(times), "deaths": [10.0, 20, 30]})
        roles = detect_roles(frame)
        for question in ("total deaths on April 27, 2011", "total deaths on 27 April 2011",
                         "total deaths on 2011-04-27"):
            with self.subTest(question=question):
                answer = answer_question(question, frame, roles)
                self.assertEqual(answer.value, 30)
                self.assertIn("date = 2011-04-27", answer.calculation)

    def test_no_such_day_is_not_read(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2011-04-27", "2011-04-28"]), "deaths": [1.0, 2]})
        self.assertIsNone(answer_question("total deaths on 2011-02-30", frame, detect_roles(frame)))

    def test_a_day_needs_a_date_column(self):
        frame = pd.DataFrame({"site": ["a", "b"], "deaths": [1.0, 2]})
        roles = detect_roles(frame)
        self.assertIsNone(answer_question("total deaths on April 27, 2011", frame, roles))
        self.assertIn("no date column", rephrase_hint("total deaths on April 27, 2011", frame, roles))


class UnitColumnTests(unittest.TestCase):
    def rats(self) -> pd.DataFrame:
        rows = [{"Rat": rat, "Diet": 1 + rat // 3, "Time": day, "weight": 200.0 + 10 * rat + day}
                for rat in range(6) for day in (1, 8, 15, 22)]
        return pd.DataFrame(rows)

    def test_the_column_numbering_the_units_is_not_summarised(self):
        rats = self.rats()
        roles = detect_roles(rats)
        self.assertIsNone(answer_question("max rat", rats, roles))
        self.assertIn("numbers the 6 units", rephrase_hint("max rat", rats, roles))

    def test_the_word_for_the_units_with_a_measure_names_the_rows(self):
        rats = self.rats()
        roles = detect_roles(rats)
        self.assertEqual(answer_question("max rat weight", rats, roles).calculation, "max(weight)")
        self.assertIn("by Rat", answer_question("average weight by Rat", rats, roles).calculation)
        self.assertEqual(answer_question("how many rats", rats, roles).value, 6)

    def test_a_plural_for_a_label_of_each_row_names_the_rows(self):
        cars = pd.DataFrame({"Model": ["a", "b", "c", "d"], "Maker": ["X", "X", "Y", "Z"]})
        answer = answer_question("how many distinct makers the models came from", cars, detect_roles(cars))
        self.assertEqual(answer.value, 3)


class DecadeTests(unittest.TestCase):
    def test_a_decade_is_its_ten_years(self):
        frame = pd.DataFrame({"year": range(1995, 2025), "distance": [float(value) for value in range(30)]})
        answer = answer_question("average distance in the 2010s", frame, detect_roles(frame))
        self.assertEqual(answer.value, np.mean(range(15, 25)))
        self.assertIn('Read "2010s" as 2010 to 2019.', answer.scope.readings)

    def test_a_decade_needs_a_year_column(self):
        frame = pd.DataFrame({"site": list("abcdef"), "distance": [1.0, 2, 3, 4, 5, 6]})
        roles = detect_roles(frame)
        self.assertIsNone(answer_question("average distance in the 2010s", frame, roles))
        self.assertIn("no date or year column", rephrase_hint("average distance in the 2010s", frame, roles))


class DemonymTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"winner": list("abcdefgh"),
                             "nationality": ["France", "France", "Belgium", "Italy", "Belgium", "Spain",
                                             "France", "Italy"],
                             "birth_country": ["France", "Belgium", "Belgium", "Italy", "France", "Spain",
                                               "France", "Italy"],
                             "distance": [1.0, 2, 3, 4, 5, 6, 7, 8]})

    def test_an_adjective_of_nationality_picks_the_nationality_column(self):
        frame = self.frame()
        answer = answer_question("how many rows Belgian", frame, detect_roles(frame))
        self.assertEqual(answer.value, 2)
        self.assertIn('Read "Belgian" as nationality = Belgium.', answer.scope.readings)

    def test_two_columns_of_countries_without_a_nationality_column_are_not_read(self):
        frame = self.frame().rename(columns={"nationality": "home_country"})
        self.assertIsNone(answer_question("how many rows Belgian", frame, detect_roles(frame)))


class LeadValueTests(unittest.TestCase):
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"age": [8, 10, 12, 14] * 3, "Sex": ["Female"] * 8 + ["Male"] * 4,
                             "distance": [20.0, 21, 22, 23, 21, 22, 23, 25, 22, 24, 26, 28]})

    def test_at_a_number_before_a_comma_picks_the_one_column_of_few_values_that_holds_it(self):
        answer = answer_question("average At 14, girls distance", self.frame(), detect_roles(self.frame()))
        self.assertEqual(answer.value, 24)
        self.assertIn("age = 14", answer.calculation)

    def test_a_number_two_columns_hold_is_not_read(self):
        frame = self.frame().assign(visit=[14, 1, 2, 3] * 3)
        answer = answer_question("average At 14, girls distance", frame, detect_roles(frame))
        self.assertTrue(answer is None or "= 14" not in answer.calculation)

    def test_a_column_of_many_values_is_not_read(self):
        frame = pd.DataFrame({"age": range(1, 41), "distance": [float(value) for value in range(40)]})
        answer = answer_question("average At 14, distance", frame, detect_roles(frame))
        self.assertTrue(answer is None or "age = 14" not in answer.calculation)
