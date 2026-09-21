"""One metric contract, honoured on every surface.

A rate is an average. It is never summed, it moves in percentage points, no
segment holds a share of it, and nothing that draws or ranks it may quietly
compute something else. These tests use the reviewer's fixtures, chosen so
that a sum and a mean give different numbers.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from aggregation import build_trend, driver_frame, segment_period_change
from ai_insights import AIQueryPlan, _to_query_plan, describe_query_plan
from autovis import ChartSpec
from business_insights import analyze_business, build_business_report
from formatting import format_number
from metrics import resolve_metric
from nlq import answer_question, execute_plan
from schema import detect_roles
from ui import _explore_frame


def rate_fixture() -> pd.DataFrame:
    """December and January: one Alpha at 10% and three Beta at 20%.

    February: three Alpha at 10% and one Beta at 40%. The overall mean is
    17.5% in every month, while the segment means move.
    """
    rows = []
    for month in ("2024-12-31", "2025-01-31"):
        rows.append((month, "Alpha", 0.10))
        rows.extend((month, "Beta", 0.20) for _ in range(3))
    rows.extend(("2025-02-28", "Alpha", 0.10) for _ in range(3))
    rows.append(("2025-02-28", "Beta", 0.40))
    frame = pd.DataFrame(rows, columns=["Date", "Region", "Conversion Rate"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


class RateContractTests(unittest.TestCase):
    def setUp(self):
        self.frame = rate_fixture()
        self.roles = detect_roles(self.frame)
        self.assertEqual(self.roles.measure, "Conversion Rate")
        self.assertEqual(self.roles.dimension, "Region")

    def test_the_contract_says_what_a_rate_is(self):
        metric = resolve_metric("Conversion Rate")
        self.assertEqual(metric.aggregation, "mean")
        self.assertEqual(metric.unit, "rate")
        self.assertFalse(metric.additive)
        self.assertEqual(metric.combines_as, "average")

        amount = resolve_metric("Revenue")
        self.assertEqual(amount.aggregation, "sum")
        self.assertTrue(amount.additive)
        self.assertEqual(resolve_metric(None).aggregation, "count")

    def test_the_overall_rate_is_flat_and_the_brief_says_so(self):
        series = build_trend(self.frame, self.roles)
        np.testing.assert_allclose(series.frame["Value"].to_numpy(), [0.175, 0.175, 0.175])

        brief = analyze_business(self.frame, self.roles)
        trend = next(item for item in brief.evidence if item.kind == "trend")
        self.assertEqual(trend.tone, "neutral")
        self.assertIn("was unchanged", trend.statement)
        self.assertIn("17.5%", trend.statement)
        self.assertNotIn("+0.0%", brief.headline)

    def test_segment_mean_changes_are_never_added_up_as_a_net(self):
        # Alpha 10 -> 10 and Beta 20 -> 40 sum to twenty points of "movement"
        # in a metric that did not move. Nothing may present that sum.
        brief = analyze_business(self.frame, self.roles)
        report = build_business_report(self.frame, brief, source_name="rates.csv")
        driver = next(item for item in brief.evidence if item.kind == "driver")

        self.assertIn("percentage points", driver.value)
        self.assertIn("from 20.0% to 40.0%", driver.statement)
        self.assertNotIn("increased by 20.0%", driver.statement)
        self.assertNotIn("net movement", report)
        self.assertNotIn("of all movement", report)
        self.assertIn("do not add up", driver.statement)

    def test_no_segment_holds_a_share_of_a_rate(self):
        brief = analyze_business(self.frame, self.roles)
        leader = next(item for item in brief.evidence if item.kind == "leader")
        kinds = [item.kind for item in brief.evidence]

        self.assertNotIn("concentration", kinds)
        self.assertNotIn("contributing", leader.statement)
        self.assertNotIn("69.6%", leader.statement)
        self.assertIn("highest average", leader.statement)
        self.assertNotIn("coming from the leading", brief.headline)

    def test_the_movement_chart_has_no_net_bar_for_a_rate(self):
        # driver_frame feeds the movement chart; for a rate it lists segment
        # point changes and never an "Other segments" remainder to sum.
        drivers = driver_frame(self.frame, self.roles)
        self.assertNotIn("Other segments", drivers["Segment"].tolist())
        by_segment = dict(zip(drivers["Segment"], drivers["Change"], strict=True))
        self.assertAlmostEqual(by_segment["Beta"], 0.20)
        self.assertAlmostEqual(by_segment["Alpha"], 0.0)

    def test_a_segment_absent_from_a_period_has_no_change(self):
        # Gamma appears only in February: it has no January average to have
        # moved from, so it is not a driver "rising from 0.0%".
        frame = pd.concat(
            [
                self.frame,
                pd.DataFrame(
                    {"Date": [pd.Timestamp("2025-02-28")], "Region": ["Gamma"], "Conversion Rate": [0.5]}
                ),
            ],
            ignore_index=True,
        )
        grouped, _, _ = segment_period_change(frame, detect_roles(frame))
        self.assertNotIn("Gamma", grouped.index.tolist())
        report = build_business_report(frame, analyze_business(frame), source_name="rates.csv")
        self.assertNotIn("from 0.0%", report)

    def test_explore_groups_a_rate_by_its_mean(self):
        # D-02: Alpha's records are all 10%; Explore used to show 0.5.
        spec = ChartSpec(form="bar", rationale="", x="Region", y="Conversion Rate")
        frame = _explore_frame(self.frame, spec)
        by_region = dict(zip(frame["Region"], frame["Conversion Rate"], strict=True))
        self.assertAlmostEqual(by_region["Alpha"], 0.10)
        self.assertAlmostEqual(by_region["Beta"], 1.6 / 7)

    def test_ranked_growth_of_a_rate_compares_period_means_in_points(self):
        # D-02: sums said Alpha grew 200% (10% -> 30%) while every Alpha record
        # stayed at 10%, and ranked Beta at -33% while its mean doubled.
        result = answer_question("top 2 Region conversion rate growth", self.frame, self.roles)
        self.assertIsNotNone(result)
        self.assertIn("Beta moved fastest", result.answer)
        self.assertIn("+20.0 percentage points", result.answer)
        self.assertIn("(20.0% → 40.0%)", result.answer)
        self.assertIn("Change (pp)", result.table.columns)
        by_region = dict(zip(result.table["Region"], result.table["Change (pp)"], strict=True))
        self.assertAlmostEqual(by_region["Alpha"], 0.0)
        self.assertNotIn("200", result.answer)

    def test_overall_growth_of_a_rate_is_in_points(self):
        result = answer_question("conversion rate growth", self.frame, self.roles)
        self.assertIn("0.0 percentage points", result.answer)
        self.assertIn("(17.5% → 17.5%)", result.answer)
        self.assertIn("average", result.calculation)

    def test_ai_time_plans_are_validated_against_the_executed_aggregation(self):
        # D-03: a sum over a rate trend was approved as "total ... per month"
        # and executed as means. Now it is refused, and the honest plan runs.
        summed = AIQueryPlan(
            answerable=True, intent="trend", aggregation="sum", measure="Conversion Rate", grain="M"
        )
        self.assertIsNone(_to_query_plan(summed, self.frame, self.roles))

        averaged = AIQueryPlan(
            answerable=True, intent="trend", aggregation="mean", measure="Conversion Rate", grain="M"
        )
        plan = _to_query_plan(averaged, self.frame, self.roles)
        self.assertIsNotNone(plan)
        sentence = describe_query_plan(plan)
        self.assertIn("average", sentence.lower())
        self.assertNotIn("total", sentence.lower())
        answer = execute_plan(plan, self.frame, self.roles)
        self.assertIn("mean(Conversion Rate)", answer.calculation)
        self.assertIn("17.5%", answer.answer)

    def test_every_intent_agrees_with_the_contract_for_amounts_and_rates(self):
        amounts = self.frame.assign(Revenue=[100, 200, 200, 200, 100, 200, 200, 200, 100, 100, 100, 400])
        roles = detect_roles(amounts)
        for measure, wanted in (("Conversion Rate", "mean"), ("Revenue", "sum")):
            for intent, kwargs in (
                ("aggregate", {}),
                ("breakdown", {"dimension": "Region"}),
                ("rank", {"dimension": "Region", "top_n": 1}),
                ("trend", {"grain": "M"}),
                ("growth", {}),
            ):
                other = "sum" if wanted == "mean" else "mean"
                with self.subTest(measure=measure, intent=intent):
                    honest = AIQueryPlan(
                        answerable=True, intent=intent, aggregation=wanted, measure=measure, **kwargs
                    )
                    plan = _to_query_plan(honest, amounts, roles)
                    self.assertIsNotNone(plan)
                    self.assertEqual(plan.aggregation, wanted)
                    if intent in ("trend", "growth"):
                        dishonest = AIQueryPlan(
                            answerable=True, intent=intent, aggregation=other, measure=measure, **kwargs
                        )
                        self.assertIsNone(_to_query_plan(dishonest, amounts, roles))

    def test_an_explicit_grouped_sum_is_honoured_or_refused_like_the_ungrouped_one(self):
        # D-06: "total conversion rate" summed; adding "by Region" silently
        # switched to a mean. The two must agree.
        alone = answer_question("total conversion rate", self.frame, self.roles)
        grouped = answer_question("total conversion rate by Region", self.frame, self.roles)
        plain = answer_question("conversion rate by Region", self.frame, self.roles)

        self.assertEqual(alone.plan.aggregation, "sum")
        self.assertEqual(grouped.plan.aggregation, "sum")
        self.assertEqual(plain.plan.aggregation, "mean")
        self.assertIn("average conversion rate", plain.answer)
        self.assertIn("22.9%", plain.answer)


class RateScaleTests(unittest.TestCase):
    def test_a_scalar_is_formatted_against_its_column(self):
        # D-08: [0.5, 2.0] is in percentage points. The minimum is 0.5%, not
        # the 50.0% that formatting the scalar alone produced.
        frame = pd.DataFrame({"Conversion Rate": [0.5, 2.0], "Channel": ["a", "b"]})
        roles = detect_roles(frame)
        answer = answer_question("minimum conversion rate", frame, roles)
        self.assertIn("0.5%", answer.answer)
        self.assertNotIn("50.0%", answer.answer)

        brief = analyze_business(frame, roles)
        values = [item.value for item in brief.kpis if "Conversion Rate" in item.label]
        self.assertIn("1.2%", values)
        self.assertEqual(
            format_number(0.5, "Conversion Rate", column_values=frame["Conversion Rate"]), "0.5%"
        )

    def test_a_rate_change_is_reported_in_points_on_the_dashboard(self):
        frame = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]),
                "Conversion Rate": [0.10, 0.10, 0.10, 0.20],
                "Channel": ["a", "a", "a", "a"],
            }
        )
        brief = analyze_business(frame, detect_roles(frame))
        trend = next(item for item in brief.evidence if item.kind == "trend")
        self.assertIn("10.0 percentage points", trend.statement)
        self.assertEqual(trend.value, "+10.0 pp")
        self.assertNotIn("100.0%", trend.statement)
        self.assertIn("percentage points", trend.calculation)


if __name__ == "__main__":
    unittest.main()
