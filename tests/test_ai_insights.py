import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from ai_insights import (
    MODEL_PRESETS,
    AIAction,
    AINarrative,
    AIQueryFilter,
    AIQueryPlan,
    _to_query_plan,
    build_ai_payload,
    build_planner_payload,
    describe_query_plan,
    execute_approved_ai_plan,
    generate_ai_narrative,
    narrative_to_markdown,
    plan_query_with_ai,
)
from business_insights import analyze_business
from demo_data import make_demo_data
from nlq import QueryPlan, execute_plan
from schema import detect_roles


class FakeResponses:
    def __init__(self, parsed):
        self.parsed = parsed
        self.arguments = None

    def parse(self, **kwargs: object) -> object:
        self.arguments = kwargs
        return SimpleNamespace(output_parsed=self.parsed)


class FakeClient:
    def __init__(self, parsed):
        self.responses = FakeResponses(parsed)


class AIInsightTests(unittest.TestCase):
    def setUp(self):
        dataframe = make_demo_data(rows=240)
        self.brief = analyze_business(dataframe, detect_roles(dataframe))
        self.narrative = AINarrative(
            executive_summary="Growth is concentrated in a small set of operating signals.",
            strategic_read="Validate whether the latest movement persists before changing the plan.",
            actions=[
                AIAction(
                    title="Validate the leading segment",
                    recommendation="Compare the leader with the rest of the portfolio next period.",
                    evidence="The deterministic segment calculation identifies a clear leader.",
                    confidence="medium",
                )
            ],
            watchouts=["The file establishes correlation, not causation."],
        )

    def test_payload_contains_computed_evidence_not_raw_rows(self):
        payload = json.loads(build_ai_payload(self.brief, context="Weekly revenue review"))

        self.assertEqual(payload["business_context"], "Weekly revenue review")
        self.assertTrue(payload["computed_evidence"])
        self.assertTrue(payload["deterministic_recommendations"])
        self.assertNotIn("rows", payload)
        self.assertNotIn("records", payload)

    def test_generation_uses_typed_responses_contract(self):
        client = FakeClient(self.narrative)
        config = MODEL_PRESETS["Fast · Luna"]

        result = generate_ai_narrative(
            self.brief,
            api_key="test-key",
            config=config,
            context="Weekly revenue review",
            safety_identifier="anonymous-session",
            client=client,
        )

        self.assertEqual(result, self.narrative)
        arguments = client.responses.arguments
        self.assertIsNotNone(arguments)
        assert arguments is not None
        self.assertEqual(arguments["model"], "gpt-5.6-luna")
        self.assertEqual(arguments["reasoning"], {"effort": "low"})
        self.assertIs(arguments["text_format"], AINarrative)
        self.assertEqual(arguments["safety_identifier"], "anonymous-session")
        self.assertFalse(arguments["store"])

    def test_api_key_is_required_only_for_optional_narrative(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            generate_ai_narrative(
                self.brief,
                api_key=" ",
                config=MODEL_PRESETS["Fast · Luna"],
                safety_identifier="anonymous-session",
            )

    def test_markdown_preserves_ai_provenance(self):
        report = narrative_to_markdown(self.narrative, model="gpt-5.6-luna")

        self.assertIn("Optional AI strategic read", report)
        self.assertIn("Validate the leading segment", report)
        self.assertIn("gpt-5.6-luna", report)
        self.assertIn("raw rows were not sent", report)


class PlanValidationTests(unittest.TestCase):
    """A plan the user approves must be a plan ADA will run exactly as described."""

    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "Order Date": pd.date_range("2024-01-01", periods=40, freq="D"),
                "Region": ["West", "East"] * 20,
                "Ticket": [f"T-{index:04d}" for index in range(40)],
                "Revenue": [100.0 + index for index in range(40)],
            }
        )
        self.roles = detect_roles(self.frame)
        self.undated = self.frame.drop(columns=["Order Date"])
        self.undated_roles = detect_roles(self.undated)

    def _plan(self, **fields):
        return _to_query_plan(AIQueryPlan(answerable=True, **fields), self.frame, self.roles)

    def test_a_text_column_is_refused_as_a_measure(self):
        for aggregation in ("sum", "mean", "median", "min", "max"):
            with self.subTest(aggregation=aggregation):
                self.assertIsNone(
                    self._plan(intent="aggregate", aggregation=aggregation, measure="Region")
                )

    def test_a_date_column_is_refused_as_a_measure(self):
        self.assertIsNone(self._plan(intent="aggregate", aggregation="sum", measure="Order Date"))

    def test_counting_a_text_column_is_still_allowed(self):
        self.assertIsNotNone(self._plan(intent="count", aggregation="count", measure="Region"))

    def test_a_column_cannot_be_grouped_by_itself(self):
        self.assertIsNone(
            self._plan(intent="rank", aggregation="sum", measure="Revenue", dimension="Revenue")
        )

    def test_a_date_is_refused_as_a_grouping_dimension(self):
        self.assertIsNone(
            self._plan(intent="breakdown", aggregation="sum", measure="Revenue", dimension="Order Date")
        )

    def test_a_row_identifier_is_refused_as_a_grouping_dimension(self):
        self.assertIsNone(
            self._plan(intent="breakdown", aggregation="sum", measure="Revenue", dimension="Ticket")
        )

    def test_a_time_filter_without_a_date_column_is_refused(self):
        parsed = AIQueryPlan(answerable=True, intent="aggregate", aggregation="sum",
                             measure="Revenue", year=2024)

        self.assertIsNone(_to_query_plan(parsed, self.undated, self.undated_roles))

    def test_an_average_over_time_is_refused_rather_than_answered_as_a_sum(self):
        self.assertIsNone(
            self._plan(intent="trend", aggregation="mean", measure="Revenue")
        )

    def test_modifiers_the_intent_ignores_are_stripped(self):
        plan = self._plan(
            intent="aggregate", aggregation="sum", measure="Revenue",
            dimension="Region", top_n=3, grain="M",
        )

        assert plan is not None
        self.assertIsNone(plan.dimension)
        self.assertIsNone(plan.top_n)
        self.assertIsNone(plan.grain)
        self.assertNotIn("Region", describe_query_plan(plan))

    def test_growth_keeps_the_dimension_and_grain_its_executor_uses(self):
        plan = self._plan(intent="growth", aggregation="sum", measure="Revenue",
                          dimension="Region", grain="M")

        assert plan is not None
        self.assertEqual(plan.dimension, "Region")
        self.assertEqual(plan.grain, "M")

    def test_the_breakdown_sentence_names_the_aggregation_that_runs(self):
        """A mean described as "total" is the exact lie the gate exists to stop."""
        for aggregation, word in (("sum", "total"), ("mean", "average"), ("median", "median"),
                                  ("min", "minimum"), ("max", "maximum")):
            with self.subTest(aggregation=aggregation):
                plan = self._plan(intent="breakdown", aggregation=aggregation,
                                  measure="Revenue", dimension="Region")
                assert plan is not None
                self.assertIn(f"{word} Revenue", describe_query_plan(plan))

    def test_a_small_pre_aggregated_table_can_be_broken_down(self):
        """Two regions in a two-row summary are unique by construction, not an id."""
        summary = pd.DataFrame({"Region": ["West", "East"], "Revenue": [100.0, 200.0]})

        plan = _to_query_plan(
            AIQueryPlan(answerable=True, intent="breakdown", aggregation="sum",
                        measure="Revenue", dimension="Region"),
            summary, detect_roles(summary),
        )

        self.assertIsNotNone(plan)

    def test_every_approved_plan_executes_without_raising(self):
        """Nothing that survives validation may blow up in the executor."""
        for intent in ("aggregate", "count", "rank", "breakdown", "trend", "growth"):
            for measure in (None, "Revenue", "Region", "Order Date", "Ticket"):
                for dimension in (None, "Region", "Revenue", "Order Date", "Ticket"):
                    plan = self._plan(
                        intent=intent, aggregation="sum", measure=measure, dimension=dimension
                    )
                    if plan is None:
                        continue
                    with self.subTest(intent=intent, measure=measure, dimension=dimension):
                        answer = execute_plan(plan, self.frame, self.roles)
                        self.assertTrue(answer.answer)


class PayloadContentTests(unittest.TestCase):
    """The privacy claim is only as good as a test that could falsify it.

    Asserting on key names cannot catch a leak, because a leak arrives inside
    a value. These put distinctive markers in the data and search the payload
    text for them.
    """

    def _frame(self):
        return pd.DataFrame(
            {
                "Order Date": pd.date_range("2024-01-01", periods=24, freq="MS"),
                "Customer": ["Northwind Ltd", "Barclays plc"] * 12,
                "Notes": [f"SECRET-NOTE-{index}" for index in range(24)],
                "Account": [f"ACCT-{index:04d}" for index in range(24)],
                "Revenue": [100 + index * 10 for index in range(24)],
            }
        )

    def test_no_value_from_a_non_segment_column_reaches_the_narrative_payload(self):
        frame = self._frame()
        brief = analyze_business(frame, detect_roles(frame))

        payload = build_ai_payload(brief, context="review")

        self.assertNotIn("SECRET-NOTE", payload)
        self.assertNotIn("ACCT-", payload)

    def test_no_value_from_a_non_segment_column_reaches_the_planner_payload(self):
        frame = self._frame()

        payload = build_planner_payload("what is total revenue", frame, detect_roles(frame))

        self.assertNotIn("SECRET-NOTE", payload)
        self.assertNotIn("ACCT-", payload)

    def test_the_segment_names_that_do_travel_are_documented_as_travelling(self):
        """Evidence sentences name the segment they describe, and that is stated.

        This is not a leak to be fixed silently -- it is the documented
        boundary. The test exists so the boundary cannot move without someone
        noticing, in either direction.
        """
        frame = self._frame()
        brief = analyze_business(frame, detect_roles(frame))

        payload = build_ai_payload(brief, context="review")

        self.assertTrue(
            "Northwind Ltd" in payload or "Barclays plc" in payload,
            "an evidence sentence names the segment it describes; if that stopped "
            "being true the privacy documentation should be widened, not narrowed",
        )


class AIQueryPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataframe = make_demo_data(rows=400)
        cls.roles = detect_roles(cls.dataframe)

    def plan(self, **overrides):
        defaults = {
            "answerable": True,
            "intent": "rank",
            "aggregation": "sum",
            "measure": "Revenue",
            "dimension": "Product",
            "top_n": 2,
        }
        defaults.update(overrides)
        return AIQueryPlan(**defaults)

    def test_payload_sends_schema_and_question_but_no_cell_values(self):
        payload = build_planner_payload("top products in the west", self.dataframe, self.roles)
        parsed = json.loads(payload)

        self.assertEqual(parsed["question"], "top products in the west")
        columns = {entry["column"] for entry in parsed["columns"]}
        self.assertIn("Revenue", columns)
        self.assertIn("Product", columns)
        for cell_value in ("Core", "Enterprise", "Northeast", "ORD-100000"):
            self.assertNotIn(cell_value, payload)

    def test_planned_query_executes_locally_with_ai_provenance(self):
        client = FakeClient(self.plan(filters=[AIQueryFilter(column="Region", value="west")]))

        plan = plan_query_with_ai(
            "top 2 products by revenue in the west",
            self.dataframe,
            self.roles,
            api_key="test-key",
            safety_identifier="anonymous-session",
            client=client,
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.source, "ai")
        self.assertEqual(plan.filters[0].values, ("West",))
        result = execute_plan(plan, self.dataframe, self.roles)
        self.assertIn("Product", result.answer)
        self.assertEqual(len(result.table), 2)

        arguments = client.responses.arguments
        self.assertFalse(arguments["store"])
        self.assertEqual(arguments["safety_identifier"], "anonymous-session")
        self.assertIs(arguments["text_format"], AIQueryPlan)

    def test_unanswerable_or_invalid_plans_are_refused(self):
        for parsed in (
            self.plan(answerable=False),
            self.plan(measure="Imaginary Column"),
            self.plan(filters=[AIQueryFilter(column="Region", value="Atlantis")]),
        ):
            client = FakeClient(parsed)
            plan = plan_query_with_ai(
                "question",
                self.dataframe,
                self.roles,
                api_key="test-key",
                safety_identifier="anonymous-session",
                client=client,
            )
            self.assertIsNone(plan)

    def test_time_intents_require_a_date_role(self):
        dateless = self.dataframe.drop(columns=["Order Date"])
        roles = detect_roles(dateless)
        client = FakeClient(self.plan(intent="trend", dimension=None))

        plan = plan_query_with_ai(
            "monthly revenue",
            dateless,
            roles,
            api_key="test-key",
            safety_identifier="anonymous-session",
            client=client,
        )

        self.assertIsNone(plan)

    def test_api_key_is_required(self):
        with self.assertRaisesRegex(ValueError, "API key"):
            plan_query_with_ai(
                "total revenue",
                self.dataframe,
                self.roles,
                api_key=" ",
                safety_identifier="anonymous-session",
            )

    def test_plan_description_is_human_readable(self):
        plan = self.plan(filters=[AIQueryFilter(column="Region", value="west")])
        plan = plan_query_with_ai(
            "top 2 products by revenue in the west",
            self.dataframe,
            self.roles,
            api_key="test-key",
            safety_identifier="anonymous-session",
            client=FakeClient(plan),
        )

        self.assertIsNotNone(plan)
        assert plan is not None
        description = describe_query_plan(plan)
        # The sentence has to describe the ranking that actually runs, not the
        # aggregate the plan's fields would suggest on their own.
        self.assertIn("Rank Product", description)
        self.assertIn("total Revenue", description)
        self.assertIn("2 highest", description)
        self.assertIn("Region = West", description)

    def test_rejected_plan_is_not_executed(self):
        plan = QueryPlan(intent="aggregate", aggregation="sum", measure="Revenue", source="ai")

        with patch("ai_insights.execute_plan") as execute:
            self.assertIsNone(
                execute_approved_ai_plan(
                    "total revenue",
                    plan,
                    self.dataframe,
                    self.roles,
                    approved=False,
                )
            )
            execute.assert_not_called()

            execute.return_value = execute_plan(plan, self.dataframe, self.roles)
            result = execute_approved_ai_plan(
                "total revenue",
                plan,
                self.dataframe,
                self.roles,
                approved=True,
            )
            self.assertIsNotNone(result)
            execute.assert_called_once_with(plan, self.dataframe, self.roles)


if __name__ == "__main__":
    unittest.main()
