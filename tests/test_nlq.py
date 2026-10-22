import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from ada import ui
from ada.business_insights import describe_correlation, measure_correlation
from ada.demo_data import make_demo_data
from ada.file_io import list_sample_datasets as samples
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

    def test_a_column_name_does_not_flip_the_ranking(self):
        frame = pd.DataFrame({"Store": ["A", "A", "B", "B"], "Lowest Price": [1.0, 2.0, 10.0, 20.0]})
        result = answer_question("which store has the highest lowest price", frame, detect_roles(frame))
        self.assertFalse(result.plan.ascending)
        self.assertTrue(result.answer.startswith("B "))


class SuggestionWordingTests(unittest.TestCase):
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

        self.assertIn("Left out of the ranking for Even1: Even2 (the same measurement", answer.answer)
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


if __name__ == "__main__":
    unittest.main()
