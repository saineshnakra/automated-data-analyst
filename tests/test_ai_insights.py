import json
import unittest
from types import SimpleNamespace

from ai_insights import (
    MODEL_PRESETS,
    AIAction,
    AINarrative,
    AIQueryFilter,
    AIQueryPlan,
    build_ai_payload,
    build_planner_payload,
    generate_ai_narrative,
    narrative_to_markdown,
    plan_query_with_ai,
)
from business_insights import analyze_business, detect_roles
from demo_data import make_demo_data
from nlq import execute_plan


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


if __name__ == "__main__":
    unittest.main()
