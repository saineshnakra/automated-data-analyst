import unittest

import numpy as np
import pandas as pd

from ada.demo_data import make_demo_data
from ada.nlq import (
    QueryPlan,
    answer_question,
    execute_plan,
    parse_question,
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
        self.assertIn("1.0K", answer.answer, answer.answer)

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
            "Revenue > 200", "Revenue in January and February", "Revenue this year",
            "Revenue for West or Partner", "total revenue last quarter", "revenue in 2024 and 2025",
            "revenue at least 300", "revenue over 1000", "revenue more than 50",
        ):
            with self.subTest(question=question):
                self.assertTrue(unsupported_phrasing(question))
                self.assertIsNone(answer_question(question, self.frame, self.roles))

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


if __name__ == "__main__":
    unittest.main()